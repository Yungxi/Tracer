# Tracer interface contract

This folder ships one implementation of the contract (`toy_agent.py`), but
the contract itself is **framework-agnostic**. Any agent — LangChain,
LangGraph, AutoGen, hand-rolled — can be audited by Tracer as long as a
thin wrapper emits a script that honors these six rules.

## The six rules

### 1. The unit of localization is a top-level Python function.

Tracer wraps every `def` at the **top level** of the parsed source. Methods
on classes and nested `def`s inside other functions are **not** wrapped.
Whatever you want the judge to verdict must appear as a standalone
top-level function.

### 2. State threads through arguments, not module globals.

The wrapper fixes each function's `__globals__` at definition time, so a
step cannot read variables declared at module level in the calling
notebook. Everything a step needs must arrive via its arguments and leave
via its return value.

### 3. The agent must serialize to a single Python source string.

Tracer consumes a string, not a live Python object. Framework objects
(DB connections, API clients, LLM handles, DataFrames) must be
*reconstructed* inside a setup block that runs in the script's fresh
namespace — not smuggled in from the caller.

### 4. Each call must be isolation-judgeable.

The judge only sees `(function_name, source, args, kwargs, return_value,
docstring, script_goal)` per call. Step boundaries should be narrow
enough that the return value visibly encodes the step's effect — a step
that mutates shared state and returns `None` is invisible to the judge.

### 5. Docstrings become judge context.

Every step should say in prose what it reads and what it produces. Tracer
passes the docstring into the judge prompt verbatim. This is free
accuracy.

### 6. The main block decides granularity and control flow.

Linear chains, `while` loops (for ReAct), `if`/`elif` dispatch (for
LangGraph edges) — any top-level construct Tracer can execute
statement-by-statement. Function calls inside those constructs still go
through the wrapper. What's **not** allowed: defining the functions at
runtime inside another function.

---

## How `toy_agent.py` implements the contract

ToyAgent is the thinnest wrapper we could write — ~120 lines — that
mechanically enforces all six rules for a linear-pipeline agent.

| Rule | How ToyAgent enforces it |
|---|---|
| **1. Top-level functions** | `Step.source` at `toy_agent.py:63-78` strips the `@agent.add_step` decorator before emitting, so each registered function lands at the top level of the generated script. `to_script()` emits them with no indentation at `toy_agent.py:206-208`. |
| **2. Threaded state** | `add_step()` at `toy_agent.py:113-129` registers functions with the signature `(state) -> state`; `to_script()` emits the chain `state = step_1(state); state = step_2(state); ...` at `toy_agent.py:210-213`. The module docstring at `toy_agent.py:16-40` explains why closures over module-level vars fail, and recommends a `state` dict. |
| **3. Serializable source** | `to_script()` at `toy_agent.py:163-215` returns a complete Python source string. `agent.setup` (emitted at `toy_agent.py:202-203`) reconstructs whatever the steps need — DataFrames, LLM clients, configs — inside the script's namespace. `initial_state()` at `toy_agent.py:133-146` validates that the setup block actually assigns a variable named `state`. |
| **4. Isolation-judgeable calls** | The `(state) -> state` convention means each call's return value is visible between steps. Tracer's judge sees a distinct `(args, return)` pair for every step call. |
| **5. Docstrings** | `inspect.getsource` at `toy_agent.py:73` pulls the function source *including* its docstring, which flows into Tracer's judge prompt via `judge.py:207-210`. |
| **6. Main block** | `to_script()` at `toy_agent.py:210-213` emits a linear chain. This is the simplest valid main — fine for pipelines, not enough for ReAct or conditional graphs. Adapters for those frameworks replace just this section. |

---

## Writing an adapter for another framework

To audit a LangChain / LangGraph / AutoGen agent, write a function that
produces the **same shape** of source string that `to_script()`
produces — not by subclassing `ToyAgent`, but by emitting
contract-compliant code directly:

1. **Flatten** your agent's control flow into a handful of named top-level
   functions. One function per LangGraph node, per AutoGen role-turn, per
   ReAct tick, etc.
2. **Thread state** explicitly: each emitted function takes and returns
   whatever bundle holds the conversation / graph state.
3. **Reconstruct runtime objects** in the setup block — `llm =
   ChatOpenAI(...)`, `tools = [...]`, authentication — so the script runs
   standalone.
4. **Pick your main.** Static chain, `while not state['done']: state =
   react_tick(state)` for ReAct, `if state['route'] == 'x': ...` for a
   conditional graph edge.
5. **Write a one-line docstring on every emitted step** saying what it
   reads and writes. The judge uses it.

A first-pass adapter is typically as short as ToyAgent itself.

---

## What the contract does NOT give you

Be honest with students about the limits:

- **No cross-step correlation.** Each call is judged in isolation. A buggy
  turn-1 that poisons turn-3's input will not be linked by the judge on
  its own. `TestPropertyJudge` (in `judge_test_property.py`) sharpens the
  per-call signal with executable assertions, but does not recover
  provenance across calls.
- **No visibility inside a step.** If a step makes five internal LLM
  calls, the judge sees one verdict for the whole step. Split the step to
  get finer blame.
- **Framework-native tracing is independent.** LangSmith, AutoGen logs,
  OpenTelemetry spans — these still work; they run alongside Tracer's
  verdict stream, not inside it.
