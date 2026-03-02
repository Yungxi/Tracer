#!/usr/bin/env python3
"""
Python Code Tracer with LLM Judge

Traces Python code execution, evaluates function outputs using an LLM,
and stops when errors or incorrect outputs are detected.

Usage:
    python tracer.py <file.py>                    # Trace a Python file
    python tracer.py <file.py> --goal "..."       # Trace with a specific goal
    python tracer.py --script <name>              # Run script from scripts.json
    python tracer.py --list                       # List available scripts
"""

import argparse
import json
import os
import sys

from parser import parse_file, parse_source
from judge import LLMJudge
from executor import TracingExecutor
from reporter import Reporter


# =============================================================================
# CONFIGURATION - Set your API key here
# =============================================================================
OPENAI_API_KEY = ""  # <-- Put your OpenAI API key here
SCRIPTS_FILE = "scripts.json"  # Path to scripts JSON file
# =============================================================================


def load_scripts_json(filepath: str) -> dict:
    """Load scripts from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_scripts(scripts_data: dict):
    """Print available scripts."""
    print("\nAvailable scripts:\n")
    for script in scripts_data.get("scripts", []):
        name = script.get("name", "unnamed")
        goal = script.get("goal", "No goal specified")
        print(f"  {name}")
        print(f"    Goal: {goal}\n")


def get_script_by_name(scripts_data: dict, name: str) -> dict:
    """Get a script by name from the scripts data."""
    for script in scripts_data.get("scripts", []):
        if script.get("name") == name:
            return script
    return None


def main():
    args = parse_args()

    # Handle --list flag
    if args.list:
        scripts_path = os.path.join(os.path.dirname(__file__) or '.', SCRIPTS_FILE)
        if os.path.isfile(scripts_path):
            scripts_data = load_scripts_json(scripts_path)
            list_scripts(scripts_data)
        else:
            print(f"Scripts file not found: {scripts_path}", file=sys.stderr)
        sys.exit(0)

    # Get API key
    api_key = OPENAI_API_KEY or args.api_key or os.environ.get('OPENAI_API_KEY')

    if not api_key:
        print("Error: OpenAI API key required.", file=sys.stderr)
        print("Set OPENAI_API_KEY in tracer.py or provide via --api-key flag.", file=sys.stderr)
        sys.exit(1)

    # Determine source: script from JSON or file
    script_goal = args.goal
    parsed = None
    script_name = None

    if args.script:
        # Load from scripts.json
        scripts_path = os.path.join(os.path.dirname(__file__) or '.', SCRIPTS_FILE)
        if not os.path.isfile(scripts_path):
            print(f"Error: Scripts file not found: {scripts_path}", file=sys.stderr)
            sys.exit(1)

        scripts_data = load_scripts_json(scripts_path)
        script = get_script_by_name(scripts_data, args.script)

        if not script:
            print(f"Error: Script '{args.script}' not found in {SCRIPTS_FILE}", file=sys.stderr)
            print("\nAvailable scripts:")
            for s in scripts_data.get("scripts", []):
                print(f"  - {s.get('name')}")
            sys.exit(1)

        script_name = script.get("name")
        script_goal = script.get("goal")
        code = script.get("code")

        parsed = parse_source(code)

    elif args.file:
        # Load from file
        if not os.path.isfile(args.file):
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)

        parsed = parse_file(args.file)
        script_name = os.path.basename(args.file)

    else:
        print("Error: Provide either a file path or --script <name>", file=sys.stderr)
        sys.exit(1)

    # Create reporter
    reporter = Reporter(use_colors=not args.no_color)

    try:
        # Show script info
        if script_goal:
            print(f"\n{'='*60}")
            print(f"Script: {script_name}")
            print(f"Goal: {script_goal}")
            print(f"{'='*60}\n")

        # Show code structure
        if args.show_structure:
            reporter.report_parsed_code(parsed)

        # Create LLM judge with the script goal
        judge = LLMJudge(api_key=api_key, model=args.model, script_goal=script_goal)

        # Create executor and run
        reporter.report_execution_start()

        executor = TracingExecutor(parsed, judge=judge)
        result = executor.execute()

        # Report each step
        for i, step in enumerate(result.steps, 1):
            reporter.report_step(step, i)

        # Report final result
        reporter.report_result(result)

        # Output JSON if requested
        if args.json:
            print("\n--- JSON Report ---")
            print(reporter.report_json(result))

        # Exit code based on success
        sys.exit(0 if result.success else 1)

    except SyntaxError as e:
        print(f"\nSyntax Error: {e}", file=sys.stderr)
        sys.exit(2)

    except FileNotFoundError as e:
        print(f"\nFile Error: {e}", file=sys.stderr)
        sys.exit(3)

    except Exception as e:
        print(f"\nUnexpected Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(4)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Trace Python code execution with LLM-based output judgment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tracer.py example.py                     # Trace a file
  python tracer.py example.py --goal "Calculate sums"  # With goal
  python tracer.py --script math_operations       # From scripts.json
  python tracer.py --list                         # List available scripts

Scripts are stored in scripts.json with their goals.
The LLM judge uses the goal to evaluate if outputs are correct.
        """
    )

    parser.add_argument(
        'file',
        nargs='?',
        help='Python file to trace'
    )

    parser.add_argument(
        '--script', '-S',
        help='Run a script by name from scripts.json'
    )

    parser.add_argument(
        '--goal', '-g',
        help='Goal/purpose of the script (used by LLM to judge correctness)'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available scripts from scripts.json'
    )

    parser.add_argument(
        '--api-key', '-k',
        help='OpenAI API key (overrides OPENAI_API_KEY in file)'
    )

    parser.add_argument(
        '--model', '-m',
        default='gpt-4o-mini',
        help='OpenAI model to use for judging (default: gpt-4o-mini)'
    )

    parser.add_argument(
        '--show-structure', '-s',
        action='store_true',
        help='Show parsed code structure before execution'
    )

    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output results as JSON'
    )

    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
