#!/usr/bin/env python3
"""
DSDBench Demo - Shows Tracer analyzing real DSDBench benchmark instances.

This loads actual data science debugging examples from DSDBench and
runs Tracer to find bugs in the code.

Run with: python test_scripts/demo_dsdbench.py [--instance ID] [--limit N]
"""

import argparse
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# CONFIGURATION
# =============================================================================

def load_api_key():
    """Load API key from config.json or environment."""
    # Config is in parent directory (project root)
    config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")

    if os.path.exists(config_file):
        with open(config_file) as f:
            config = json.load(f)
            key = config.get('openai_api_key', '')
            if key:
                return key

    return os.environ.get('OPENAI_API_KEY', '')


def load_dsdbench_instance(data_dir: str, instance_id: int):
    """Load a specific DSDBench instance."""
    filepath = os.path.join(data_dir, "bench_final_annotation_single_error.jsonl")

    if not os.path.exists(filepath):
        print(f"ERROR: DSDBench data not found at {filepath}")
        print("Make sure dsd_bench_data/ directory exists with benchmark files.")
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if data['id'] == instance_id:
                return data

    return None


def list_instances(data_dir: str, limit: int = 10):
    """List available DSDBench instances."""
    filepath = os.path.join(data_dir, "bench_final_annotation_single_error.jsonl")

    if not os.path.exists(filepath):
        print(f"ERROR: DSDBench data not found at {filepath}")
        return []

    instances = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            instances.append({
                'id': data['id'],
                'question': data['question'][:80] + '...' if len(data['question']) > 80 else data['question']
            })
            if len(instances) >= limit:
                break

    return instances


def run_tracer_on_instance(instance: dict, api_key: str):
    """Run Tracer on a DSDBench instance. Returns (success, tracer_findings, ground_truth)."""
    from parser import parse_source
    from judge import LLMJudge
    from executor import TracingExecutor
    from reporter import Reporter

    question = instance['question']
    error_version = instance['error_versions'][0]  # Use first error variant
    buggy_code = error_version['modified_code']
    ground_truth_cause = error_version.get('cause_error_line', 'N/A')
    ground_truth_effect = error_version.get('effect_error_line', 'N/A')
    execution_output = error_version.get('execution_output', 'N/A')

    # Store ground truth for comparison
    ground_truth = {
        'cause_line': ground_truth_cause,
        'effect_line': ground_truth_effect,
        'execution_output': execution_output
    }

    # Store Tracer's findings
    tracer_findings = {
        'errors': [],  # List of all errors found
        'status': None
    }

    # Print instance info
    print("=" * 70)
    print(f"DSDBENCH INSTANCE #{instance['id']}")
    print("=" * 70)
    print(f"\nQUESTION:\n{question[:500]}{'...' if len(question) > 500 else ''}")

    # Parse code
    print("\n" + "=" * 70)
    print("TRACER ANALYSIS")
    print("=" * 70)

    print("\n[1] Parsing code...")
    try:
        parsed = parse_source(buggy_code)
    except Exception as e:
        print(f"    Parse error: {e}")
        print("\n    Buggy code preview:")
        lines = buggy_code.split('\n')[:20]
        for i, line in enumerate(lines, 1):
            print(f"    {i:3d} | {line}")
        tracer_findings['status'] = 'parse_error'
        tracer_findings['error_message'] = str(e)
        return False, tracer_findings, ground_truth

    # Show structure
    reporter = Reporter(use_colors=True)
    reporter.report_parsed_code(parsed)

    # Create judge
    goal = f"Complete the task: {question[:200]}"
    if api_key:
        judge = LLMJudge(api_key=api_key, model="gpt-4o-mini", script_goal=goal)
        print(f"\n[2] LLM Judge enabled")
    else:
        judge = None
        print(f"\n[2] LLM Judge disabled (no API key)")

    # Execute with tracing - continue on error to find ALL bugs
    print("\n[3] Executing with Tracer (continue_on_error=True)...")
    reporter.report_execution_start()

    try:
        executor = TracingExecutor(parsed, judge=judge, continue_on_error=True)
        result = executor.execute()

        # Report steps
        for i, step in enumerate(result.steps, 1):
            reporter.report_step(step, i)

        # Report result
        reporter.report_result(result)

        # Extract Tracer's findings from result
        tracer_findings['status'] = result.stop_reason.value if hasattr(result, 'stop_reason') else 'unknown'

        # Get all errors found
        if hasattr(result, 'errors') and result.errors:
            for err in result.errors:
                tracer_findings['errors'].append({
                    'line': err.lineno,
                    'code': err.code.strip() if err.code else '',
                    'type': err.error_type,
                    'message': err.error_message
                })

        return True, tracer_findings, ground_truth

    except Exception as e:
        print(f"\n    Execution error: {e}")
        tracer_findings['status'] = 'exception'
        tracer_findings['errors'].append({
            'line': 0,
            'code': '',
            'type': 'Exception',
            'message': str(e)
        })
        return False, tracer_findings, ground_truth


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='DSDBench Demo with Tracer')
    parser.add_argument('--instance', '-i', type=int, default=1,
                        help='DSDBench instance ID to analyze (default: 1)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available instances')
    parser.add_argument('--limit', type=int, default=10,
                        help='Number of instances to list (default: 10)')
    parser.add_argument('--data-dir', default='dsd_bench_data',
                        help='Directory containing DSDBench data files')
    args = parser.parse_args()

    # Get data directory
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir)

    # List mode
    if args.list:
        print("=" * 70)
        print("DSDBENCH INSTANCES")
        print("=" * 70)
        instances = list_instances(data_dir, args.limit)
        if instances:
            for inst in instances:
                print(f"\n[{inst['id']}] {inst['question']}")
            print(f"\nShowing {len(instances)} instances. Use --limit N to see more.")
        return

    # Load API key
    api_key = load_api_key()
    if not api_key:
        print("=" * 70)
        print("WARNING: No API key found!")
        print("Set your key in config.json or OPENAI_API_KEY env var")
        print("Running without LLM judge (basic execution only)")
        print("=" * 70)

    # Load instance
    print(f"Loading DSDBench instance #{args.instance}...")
    instance = load_dsdbench_instance(data_dir, args.instance)

    if not instance:
        print(f"ERROR: Instance #{args.instance} not found.")
        print("Use --list to see available instances.")
        sys.exit(1)

    # Import Tracer components
    try:
        from parser import parse_source
        from judge import LLMJudge
        from executor import TracingExecutor
        from reporter import Reporter
    except ImportError as e:
        print(f"Error importing Tracer components: {e}")
        print("Make sure you're running from the Tracer directory.")
        sys.exit(1)

    # Run Tracer
    success, tracer_findings, ground_truth = run_tracer_on_instance(instance, api_key)

    # Comparison section
    print("\n" + "=" * 70)
    print("COMPARISON: TRACER vs GROUND TRUTH")
    print("=" * 70)

    print("\n--- Ground Truth (DSDBench Annotation) ---")
    print(f"  Cause Line:  {ground_truth['cause_line']}")
    print(f"  Effect Line: {ground_truth['effect_line']}")
    if ground_truth['execution_output'] != 'N/A':
        output_preview = ground_truth['execution_output']
        if len(output_preview) > 150:
            output_preview = output_preview[:150] + '...'
        print(f"  Expected Output: {output_preview}")

    print("\n--- Tracer's Findings ---")
    print(f"  Status: {tracer_findings['status']}")
    errors = tracer_findings['errors']
    if errors:
        print(f"  Errors Found: {len(errors)}")
        for i, err in enumerate(errors, 1):
            code_preview = err['code'][:60] + '...' if len(err['code']) > 60 else err['code']
            print(f"\n    [{i}] Line {err['line']}: {err['type']}")
            print(f"        Code: {code_preview}")
            msg = err['message'][:100] + '...' if len(err['message']) > 100 else err['message']
            print(f"        Message: {msg}")
    else:
        print(f"  Errors Found: 0 (none detected)")

    # Check if Tracer found the same error
    print("\n--- Match Analysis ---")
    gt_cause = ground_truth['cause_line'].strip() if ground_truth['cause_line'] else ''

    if gt_cause and errors:
        matched = False
        for err in errors:
            tracer_code = err['code'].strip()
            if gt_cause == tracer_code:
                print(f"  Result: EXACT MATCH - Tracer found the exact bug line!")
                matched = True
                break
            elif gt_cause in tracer_code or tracer_code in gt_cause:
                print(f"  Result: PARTIAL MATCH - Tracer found related code")
                matched = True
                break
        if not matched:
            print(f"  Result: DIFFERENT - Tracer found errors at different locations")
    elif errors:
        print("  Result: Tracer found errors (ground truth comparison not possible)")
    else:
        print("  Result: Tracer did not detect any errors")

    # Summary
    print("\n" + "=" * 70)
    if success:
        print("Demo complete! Tracer analyzed the DSDBench instance.")
    else:
        print("Demo finished with errors.")
    print("=" * 70)


if __name__ == "__main__":
    main()
