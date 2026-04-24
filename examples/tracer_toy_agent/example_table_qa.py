"""
A runnable example: toy table-QA agent built on ToyAgent, audited by Tracer.

Scenario
--------
The agent answers "Find the top 3 customers by spending." over two tiny
in-memory tables. It has one intentional silent bug (the sort direction).

Run it
------
    # from the Tracer repo root:
    cd examples/tracer_toy_agent
    export OPENAI_API_KEY=sk-...
    export PYTHONPATH=../..:$PYTHONPATH
    python example_table_qa.py
"""

from toy_agent import ToyAgent


# --- 1. Define the agent --------------------------------------------------

agent = ToyAgent(
    goal="Find the top 3 customers by total spending and return their names, "
         "sorted from HIGHEST to LOWEST spender.",
)

agent.imports = [
    "import pandas as pd",
]

# `setup` must create a variable named `state` --- whatever the first step
# will receive as its input. Here we bundle the two tables into a dict.
agent.setup = """
    state = {
        'customers': pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'name':        ['Alice', 'Bob', 'Carol', 'David', 'Eve'],
        }),
        'orders': pd.DataFrame({
            'order_id':     [101, 102, 103, 104, 105, 106],
            'customer_id':  [  1,   2,   1,   3,   2,   4],
            'total_amount': [150.0, 200.0, 300.0, 175.5, 250.0, 400.0],
        }),
    }
"""


@agent.add_step
def build_totals(state):
    """Aggregate total spending per customer.

    Adds state['totals']: a DataFrame with ['customer_id', 'total_amount'].
    """
    state['totals'] = (
        state['orders']
        .groupby('customer_id')['total_amount']
        .sum()
        .reset_index()
    )
    return state


@agent.add_step
def pick_top_n(state):
    """Pick the top 3 customers by total spending.

    Reads state['totals']; adds state['top'] sorted from HIGHEST to LOWEST.
    """
    # BUG: ascending=True returns the three LOWEST spenders.
    state['top'] = state['totals'].sort_values('total_amount', ascending=True).head(3)
    return state


@agent.add_step
def format_result(state):
    """Join customer names and return the top-3 names as a list (final output)."""
    merged = state['customers'].merge(state['top'], on='customer_id')
    return merged['name'].tolist()


# --- 2. Run locally first (no Tracer) -------------------------------------

if __name__ == "__main__":
    import os

    print("=== Local run (no Tracer) ===")
    print("agent output:", agent.run_local())
    print()

    print("=== Generated script (what Tracer will see) ===")
    print(agent.to_script())
    print()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set. Skipping Tracer audit.")
        print("Set it and rerun to see the step-level blame report.")
        raise SystemExit(0)

    print("=== Tracer audit ===")
    result = agent.audit_with_tracer(api_key=api_key)
    print()
    print("=== Findings ===")
    for err in result.errors:
        print(f"  line {err.lineno}: {err.error_type} -- {err.error_message}")
