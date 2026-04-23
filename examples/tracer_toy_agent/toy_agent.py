"""
ToyAgent --- a minimal, framework-agnostic agent interface for Tracer.

Why this exists
---------------
Tracer audits *Python scripts*: it parses a source string, executes step-by-step,
and asks an LLM judge to verdict each function call's observed I/O.

Students building toy agents for Week 5 want to:
  1. write each "step" of their agent as a plain Python function,
  2. chain steps together,
  3. run Tracer on the whole thing and see a localized blame report.

ToyAgent is the thinnest possible glue between (1)-(2) and Tracer.

The one convention you must follow
----------------------------------
    Each step takes the previous step's return value and returns something
    the next step will consume. The FIRST step receives the "initial state"
    produced by `agent.setup`.

Why: Tracer wraps each function so its `__globals__` is fixed at function
definition time. That means your step functions CANNOT close over tables or
constants declared at module level --- those names are invisible inside the
wrapped call. So we thread everything the steps need via their arguments.
The common pattern is a `state` dict:

    agent.setup = '''
        import pandas as pd
        state = {
            'customers': pd.DataFrame({...}),
            'orders':    pd.DataFrame({...}),
        }
    '''

    @agent.add_step
    def build_totals(state):
        '''Compute per-customer totals. Adds state["totals"].'''
        state['totals'] = state['orders'].groupby('customer_id')['total_amount'].sum().reset_index()
        return state

Backbone-agnostic
-----------------
Step functions can call anything: OpenAI, Anthropic, a local model, or
nothing. Tracer only sees each step's inputs and outputs. Give every step
a clear docstring --- Tracer passes it to the judge as context.
"""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Step:
    """One step of a toy agent. Wraps a user-written function."""
    name: str
    fn: Callable[[Any], Any]

    @property
    def source(self) -> str:
        """
        The function's source code, as Tracer will see it.

        `inspect.getsource` returns any decorator lines too (e.g.
        `@agent.add_step`), which would fail when Tracer executes the
        generated script in a fresh namespace. We strip the leading
        decorator lines so only the `def ...` remains.
        """
        src = textwrap.dedent(inspect.getsource(self.fn))
        lines = src.split("\n")
        i = 0
        while i < len(lines) and lines[i].lstrip().startswith("@"):
            i += 1
        return "\n".join(lines[i:])


@dataclass
class ToyAgent:
    """
    A tiny framework-agnostic agent: a goal + setup code + ordered steps.

    Fields
    ------
    goal : str
        One-sentence description of what the agent must do. Passed to
        Tracer's LLM judge as the script goal.
    imports : list[str]
        Import statements prepended to the generated script. E.g.
        ["import pandas as pd"].
    setup : str
        Python code that creates a variable named ``state`` ---
        the initial input fed to the first step. Dedented automatically.
        Example::

            agent.setup = '''
                state = {"question": "top 3 by spending",
                         "orders":   pd.DataFrame(...)}
            '''
    steps : list[Step]
        Populated by `add_step()`.
    """
    goal: str = ""
    imports: list = field(default_factory=list)
    setup: str = ""
    steps: list = field(default_factory=list)

    # ----- building the agent -----

    def add_step(self, fn: Callable) -> Callable:
        """
        Register a function as the next step of the agent.

        Usable as a decorator:

            @agent.add_step
            def my_step(state):
                ...
                return state

        The function takes the previous step's return value and returns
        whatever the next step should consume. By convention `state` is a
        dict the steps mutate, but any threaded value works.
        """
        self.steps.append(Step(name=fn.__name__, fn=fn))
        return fn

    # ----- running without Tracer (sanity check) -----

    def initial_state(self) -> Any:
        """
        Execute ``imports`` + ``setup`` in a fresh namespace and return
        whatever the setup code assigned to ``state``.
        """
        ns: dict = {}
        src = "\n".join(self.imports) + "\n" + textwrap.dedent(self.setup)
        exec(compile(src, "<ToyAgent.initial_state>", "exec"), ns)
        if "state" not in ns:
            raise ValueError(
                "`agent.setup` must define a variable named `state`. "
                "Example: `state = {\"question\": ..., \"orders\": ...}`"
            )
        return ns["state"]

    def run_local(self, initial: Any = None) -> Any:
        """
        Run the steps in-process without Tracer.

        If ``initial`` is not given, `initial_state()` is called to build
        it from `setup`. Useful to confirm the agent finishes at all
        before auditing it.
        """
        state = self.initial_state() if initial is None else initial
        for step in self.steps:
            state = step.fn(state)
        return state

    # ----- emitting a script for Tracer -----

    def to_script(self) -> str:
        """
        Stitch imports + setup + step sources + a runner into one Python
        source string. This is what Tracer's `parse_source` will consume.

        The generated script looks like::

            <imports>

            # --- setup initial state ---
            <setup body, including a line `state = ...`>

            <step 1 source>
            <step 2 source>
            ...

            # --- agent main ---
            state = step_1(state)
            state = step_2(state)
            ...
            print('final:', state)
        """
        if not self.steps:
            raise ValueError("ToyAgent has no steps; register some with @agent.add_step.")
        if "state" not in self.setup:
            raise ValueError(
                "`agent.setup` must assign to a variable named `state`. "
                "See the ToyAgent docstring for an example."
            )

        parts: list[str] = []
        parts.append("# Auto-generated by ToyAgent for Tracer.")
        parts.append(f"# Goal: {self.goal}")
        parts.append("")

        if self.imports:
            parts.extend(self.imports)
            parts.append("")

        parts.append("# --- setup: build the initial `state` ---")
        parts.append(textwrap.dedent(self.setup).strip())
        parts.append("")

        for step in self.steps:
            parts.append(step.source.rstrip())
            parts.append("")

        parts.append("# --- agent main ---")
        for step in self.steps:
            parts.append(f"state = {step.name}(state)")
        parts.append("print('final:', state)")
        parts.append("")
        return "\n".join(parts)

    # ----- convenience: run under Tracer in one call -----

    def audit_with_tracer(
        self,
        api_key: str,
        *,
        model: str | None = None,
        continue_on_error: bool = True,
        use_colors: bool = False,
    ):
        """
        One-shot: build the script, run Tracer, print the report, and
        return the raw `ExecutionResult` so you can inspect `.errors`.

        Tracer's `parser`, `executor`, `judge`, `reporter` modules must be
        importable (add the Tracer directory to `sys.path`).
        """
        from parser import parse_source        # noqa: E402 -- late import
        from executor import TracingExecutor   # noqa: E402
        from judge import LLMJudge             # noqa: E402
        from reporter import Reporter          # noqa: E402

        source = self.to_script()
        parsed = parse_source(source)
        judge_kwargs = {"api_key": api_key, "script_goal": self.goal}
        if model:
            judge_kwargs["model"] = model
        judge = LLMJudge(**judge_kwargs)
        executor = TracingExecutor(parsed, judge=judge, continue_on_error=continue_on_error)
        result = executor.execute()
        Reporter(use_colors=use_colors).report_result(result)
        return result
