# Tracer + ToyAgent: audit your own toy agent in ~20 lines

A tiny, framework-agnostic scaffold for writing a toy agent and auditing it
with [Tracer](../../README.md). Originally developed for Week 5 projects
in Stats C163/C263 --- you can use it no matter which LLM backbone you
adopt (OpenAI, Anthropic, a local model, or a mock).

## What's in here

| File | What it is |
|---|---|
| `toy_agent.py` | The interface: `ToyAgent` + `@agent.add_step`. ~120 lines. |
| `example_table_qa.py` | A runnable toy table-QA agent with one silent bug. |
| `tracer_toy_agent.ipynb` | Notebook walkthrough: define steps → sanity-run → audit with Tracer → fix → re-audit. |
| [`INTERFACE.md`](INTERFACE.md) | The framework-agnostic contract Tracer expects. `toy_agent.py` is one implementation; LangChain / LangGraph / AutoGen agents can adapt via a thin wrapper. |

## The mental model

Tracer audits *Python scripts*: it parses source, executes step-by-step,
and asks an LLM judge to verdict each function call's observed I/O.

A **ToyAgent** is just:

1. a **goal** string (what the agent should do),
2. a **setup** block that creates an initial `state`,
3. an ordered list of **steps**, each a plain Python function that takes
   the previous step's return value and returns the next one.

You register steps with a decorator; `ToyAgent.to_script()` stitches
everything into a single source string that Tracer parses and runs.

```python
from toy_agent import ToyAgent

agent = ToyAgent(goal="Find the top 3 customers by spending.")

agent.imports = ["import pandas as pd"]
agent.setup = """
    state = {
        'customers': pd.DataFrame({...}),
        'orders':    pd.DataFrame({...}),
    }
"""

@agent.add_step
def build_totals(state):
    """Sum total_amount per customer; adds state['totals']."""
    state['totals'] = state['orders'].groupby('customer_id')['total_amount'].sum().reset_index()
    return state

@agent.add_step
def pick_top_n(state):
    """Top 3 customers, highest spender first; adds state['top']."""
    state['top'] = state['totals'].sort_values('total_amount', ascending=True).head(3)  # bug
    return state

result = agent.audit_with_tracer(api_key="sk-...")
for err in result.errors:
    print(err.lineno, err.error_type, err.error_message)
```

**Why a `state` dict instead of closures?** Tracer wraps each step at
definition time; once wrapped, the function can't see variables declared
at module level (its `__globals__` is fixed to Tracer's internal
namespace). Threading a `state` object through the arguments avoids the
issue.

## Backbone-agnostic: how to plug in any LLM

Your step functions can call *anything*. Tracer only sees each step's
inputs and outputs --- it does not care which LLM (if any) you used inside.

```python
# Option A: OpenAI
from openai import OpenAI
client = OpenAI()

@agent.add_step
def draft_plan(state):
    """Ask the planner LLM for a short plan; adds state['plan']."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Plan: {state['question']}"}],
    )
    state['plan'] = resp.choices[0].message.content
    return state
```

```python
# Option B: Anthropic
from anthropic import Anthropic
client = Anthropic()

@agent.add_step
def draft_plan(state):
    """Ask the planner LLM for a short plan; adds state['plan']."""
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": f"Plan: {state['question']}"}],
    )
    state['plan'] = msg.content[0].text
    return state
```

```python
# Option C: mock (no LLM at all, for offline dev)
@agent.add_step
def draft_plan(state):
    """Return a stubbed plan so you can wire the rest of the agent first."""
    state['plan'] = "1) filter rows  2) sort  3) take top 3"
    return state
```

**Give every step a docstring.** Tracer's judge uses it as context when
deciding whether the step's observed output matches the goal.

## Install

```bash
# from inside a Tracer checkout:
cd examples/tracer_toy_agent
pip install -r requirements.txt

# make Tracer importable -- it lives two dirs up (the Tracer repo root):
export PYTHONPATH=$(pwd)/../..:$PYTHONPATH
export OPENAI_API_KEY=sk-...
```

If you are using this folder standalone (e.g. in Colab), the notebook's
setup cell will auto-clone Tracer into a local `_tracer/` directory on
first run, so no manual clone is needed there.

## Run the worked example

```bash
python example_table_qa.py
```

You will see (roughly):

```
=== Local run (no Tracer) ===
agent output: ['Alice', 'Carol', 'David']       # wrong! these are the lowest spenders

=== Generated script (what Tracer will see) ===
import pandas as pd

# --- setup: build the initial `state` ---
state = {'customers': pd.DataFrame(...), 'orders': pd.DataFrame(...)}

def pick_top_n(state):
    """Pick the top 3 customers by total spending..."""
    state['top'] = state['totals'].sort_values('total_amount', ascending=True).head(3)
    return state
...

=== Tracer audit ===
[...execution trace...]
  LLM Judge: INCORRECT -- returns LOWEST spenders due to ascending=True

=== Findings ===
  line XX: LogicError -- ...
```

Flip `ascending=True` → `False`, rerun, watch the verdict flip to
`CORRECT`. That single before/after is the backbone of your Week 5
slide.

## Notebook walkthrough

Open `tracer_toy_agent.ipynb` in Jupyter / VS Code. It walks through:

1. Importing Tracer and ToyAgent.
2. Defining a goal and registering steps.
3. Sanity-running without Tracer.
4. Inspecting the generated script.
5. Auditing with Tracer's LLM judge.
6. Reading the step-level findings.
7. Fixing one line and re-auditing (before/after).
8. Swapping in *your* agent.

## Design choices (if you're curious)

- **Steps are plain functions**, not subclasses. No framework lock-in.
- **Source is recovered via `inspect.getsource`**. Step functions must
  be defined at module/notebook-top-level, not inside closures.
- **Decorator lines are stripped** from the emitted source, so the
  generated script is clean Python Tracer can run in a fresh namespace.
- **State threads through arguments.** Tracer wraps each step at
  definition time, fixing its `__globals__`. Rather than fight that, we
  pass everything the steps need via the threaded `state` argument.
- **`audit_with_tracer()` is a convenience**, not a required path. Power
  users can call `parse_source`, `TracingExecutor`, `LLMJudge`, and
  `Reporter` directly.
