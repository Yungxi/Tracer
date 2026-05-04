#!/usr/bin/env python3
"""
Tracer Demo - Shows Tracer finding bugs in data science code.

Uses a BIRD/SPIDER-style database question as an example.
Run with: python test_scripts/demo_tracer.py (from project root)

Options:
    --verbose    Show all judgments (including CORRECT verdicts)
    --detailed   Judge every statement (not just function calls)
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# BIRD/SPIDER STYLE QUESTION
# =============================================================================
QUESTION = """
Find the top 3 customers who spent the most money and calculate
each customer's average order value.
"""

# =============================================================================
# BUGGY CODE (contains 3 intentional bugs)
# =============================================================================
BUGGY_CODE = '''
import pandas as pd

# Database tables
customers_data = {
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Carol', 'David', 'Eve'],
}

orders_data = {
    'order_id': [101, 102, 103, 104, 105, 106],
    'customer_id': [1, 2, 1, 3, 2, 4],
    'total_amount': [150.0, 200.0, 300.0, 175.5, 250.0, 400.0]
}

customers = pd.DataFrame(customers_data)
orders = pd.DataFrame(orders_data)


def get_top_customers(orders_df, customers_df, n=3):
    """Get top N customers by total spending."""
    totals = orders_df.groupby('customer_id')['total_amount'].sum().reset_index()
    totals.columns = ['customer_id', 'total']
    result = customers_df.merge(totals, on='customer_id')
    # BUG: ascending=True returns LOWEST spenders
    top_n = result.sort_values('total', ascending=True).head(n)
    return top_n


def calculate_average(orders_df, customer_ids):
    """Calculate average order value for customers."""
    filtered = orders_df[orders_df['customer_id'].isin(customer_ids)]
    # BUG: Wrong calculation - divides by total count
    avg_values = filtered.groupby('customer_id')['total_amount'].sum() / len(filtered)
    return avg_values.to_dict()


def format_output(top_df, averages):
    """Format results for display."""
    results = []
    for _, row in top_df.iterrows():
        cid = row['customer_id']
        # BUG: Wrong column name
        results.append({
            'name': row['customer_name'],
            'total': row['total'],
            'avg': averages.get(cid, 0)
        })
    return results


# Main execution
top_3 = get_top_customers(orders, customers, n=3)
print(f"Top 3: {top_3['name'].tolist()}")

customer_ids = top_3['customer_id'].tolist()
averages = calculate_average(orders, customer_ids)
print(f"Averages: {averages}")

final = format_output(top_3, averages)
print(f"Results: {final}")
'''


def load_api_key():
    """Load API key from config.json or environment."""
    import json
    # Config is in parent directory (project root)
    config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")

    if os.path.exists(config_file):
        with open(config_file) as f:
            config = json.load(f)
            key = config.get('openai_api_key', '')
            if key:
                return key

    return os.environ.get('OPENAI_API_KEY', '')


def main():
    """Run Tracer on buggy code."""
    parser = argparse.ArgumentParser(description="Tracer Demo - Find bugs in data science code")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show all judgments including CORRECT verdicts")
    parser.add_argument("--detailed", "-d", action="store_true",
                        help="Judge every statement (not just function calls)")
    args = parser.parse_args()

    from parser import parse_source
    from judge import LLMJudge
    from executor import TracingExecutor
    from reporter import Reporter

    api_key = load_api_key()
    if not api_key:
        print("WARNING: No API key found.")
        print("Set in config.json or OPENAI_API_KEY env var")
        print("Running without LLM judge (basic execution only)")

    # Print question
    print("=" * 60)
    print("QUESTION:", QUESTION.strip())
    print("=" * 60)

    # Parse code
    print("\n[1] Parsing code...")
    parsed = parse_source(BUGGY_CODE)

    # Show structure
    reporter = Reporter(use_colors=True, verbose=args.verbose)
    reporter.report_parsed_code(parsed)

    # Create judge with goal
    goal = "Find top 3 customers by spending and calculate average order values"
    if api_key:
        judge = LLMJudge(api_key=api_key, model="gpt-4o-mini", script_goal=goal, verbose=args.verbose)
        mode_flags = []
        if args.verbose:
            mode_flags.append("verbose")
        if args.detailed:
            mode_flags.append("detailed")
        mode_status = f" ({', '.join(mode_flags)})" if mode_flags else ""
        print(f"\n[2] LLM Judge goal: {goal}{mode_status}")
    else:
        judge = None
        print("\n[2] LLM Judge disabled (no API key)")

    # Execute with tracing - continue on error to find ALL bugs
    print("\n[3] Executing with Tracer (continue_on_error=True)...")
    reporter.report_execution_start()

    executor = TracingExecutor(parsed, judge=judge, continue_on_error=True, detailed=args.detailed)
    result = executor.execute()

    # Report steps
    for i, step in enumerate(result.steps, 1):
        reporter.report_step(step, i)

    # Report result
    reporter.report_result(result)

    # Show Tracer's findings
    print("\n" + "=" * 60)
    print("TRACER'S FINDINGS:")
    print("=" * 60)

    if result.errors:
        print(f"\nTracer found {len(result.errors)} error(s):\n")
        for i, err in enumerate(result.errors, 1):
            print(f"  [{i}] Line {err.lineno}: {err.error_type}")
            print(f"      Code: {err.code}")
            print(f"      Message: {err.error_message}")
            print()
    else:
        print("\nNo errors detected.")

    # Show ground truth for comparison
    print("=" * 60)
    print("GROUND TRUTH (Known Bugs):")
    print("=" * 60)
    print("""
1. get_top_customers(): ascending=True returns LOWEST spenders

2. calculate_average(): Divides by total count, not per-customer

3. format_output(): Uses 'customer_name' but column is 'name'
""")


if __name__ == "__main__":
    main()
