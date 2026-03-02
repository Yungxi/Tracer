#!/usr/bin/env python3
"""
SWE-bench Integration - Load and test against SWE-bench Lite dataset.

Usage:
    python swe_bench.py --list                    # List available instances
    python swe_bench.py --instance <id>           # Run on specific instance
    python swe_bench.py --run-all                 # Run on all instances
    python swe_bench.py --run-dev                 # Run on dev split only
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from patcher import LLMPatcher, PatchResult, generate_diff


# =============================================================================
# CONFIGURATION
# =============================================================================
OPENAI_API_KEY = ""  # <-- Set your API key here or use --api-key flag
# =============================================================================


@dataclass
class SWEBenchInstance:
    """A single SWE-bench problem instance."""
    instance_id: str
    repo: str
    problem_statement: str
    hints_text: str
    base_commit: str
    patch: str  # The ground truth patch
    test_patch: str
    fail_to_pass: str
    pass_to_pass: str
    version: str
    created_at: str


@dataclass
class EvaluationResult:
    """Result of evaluating the patcher on one instance."""
    instance_id: str
    repo: str
    success: bool
    generated_patch: Optional[str]
    ground_truth_patch: str
    explanation: str
    error: Optional[str] = None


class SWEBenchLoader:
    """Loads and manages SWE-bench Lite dataset."""

    DATASET_NAME = "princeton-nlp/SWE-bench_Lite"

    def __init__(self):
        if load_dataset is None:
            raise ImportError(
                "datasets package required. Install with: pip install datasets"
            )
        self._dataset = None
        self._instances: Dict[str, SWEBenchInstance] = {}

    def load(self) -> None:
        """Load the dataset from Hugging Face."""
        print("Loading SWE-bench Lite dataset...")
        self._dataset = load_dataset(self.DATASET_NAME)
        self._parse_instances()
        print(f"Loaded {len(self._instances)} instances")

    def _parse_instances(self) -> None:
        """Parse dataset into SWEBenchInstance objects."""
        for split in self._dataset:
            for item in self._dataset[split]:
                instance = SWEBenchInstance(
                    instance_id=item["instance_id"],
                    repo=item["repo"],
                    problem_statement=item["problem_statement"],
                    hints_text=item.get("hints_text", ""),
                    base_commit=item["base_commit"],
                    patch=item["patch"],
                    test_patch=item.get("test_patch", ""),
                    fail_to_pass=item.get("FAIL_TO_PASS", ""),
                    pass_to_pass=item.get("PASS_TO_PASS", ""),
                    version=item.get("version", ""),
                    created_at=item.get("created_at", "")
                )
                self._instances[instance.instance_id] = instance

    def get_instance(self, instance_id: str) -> Optional[SWEBenchInstance]:
        """Get a specific instance by ID."""
        return self._instances.get(instance_id)

    def get_all_instances(self) -> List[SWEBenchInstance]:
        """Get all instances."""
        return list(self._instances.values())

    def get_dev_instances(self) -> List[SWEBenchInstance]:
        """Get dev split instances."""
        dev_ids = set()
        if "dev" in self._dataset:
            for item in self._dataset["dev"]:
                dev_ids.add(item["instance_id"])
        return [inst for inst in self._instances.values() if inst.instance_id in dev_ids]

    def get_test_instances(self) -> List[SWEBenchInstance]:
        """Get test split instances."""
        test_ids = set()
        if "test" in self._dataset:
            for item in self._dataset["test"]:
                test_ids.add(item["instance_id"])
        return [inst for inst in self._instances.values() if inst.instance_id in test_ids]

    def list_instances(self) -> None:
        """Print list of all instances."""
        print("\nSWE-bench Lite Instances:\n")
        print(f"{'Instance ID':<50} {'Repo':<30}")
        print("-" * 80)

        # Group by repo
        by_repo: Dict[str, List[SWEBenchInstance]] = {}
        for inst in self._instances.values():
            if inst.repo not in by_repo:
                by_repo[inst.repo] = []
            by_repo[inst.repo].append(inst)

        for repo in sorted(by_repo.keys()):
            print(f"\n{repo}:")
            for inst in sorted(by_repo[repo], key=lambda x: x.instance_id):
                print(f"  {inst.instance_id}")

        print(f"\nTotal: {len(self._instances)} instances")


class SWEBenchEvaluator:
    """Evaluates the patcher against SWE-bench instances."""

    def __init__(self, patcher: LLMPatcher):
        self.patcher = patcher
        self.results: List[EvaluationResult] = []

    def evaluate_instance(self, instance: SWEBenchInstance) -> EvaluationResult:
        """
        Evaluate the patcher on a single instance.

        Args:
            instance: The SWE-bench instance to evaluate

        Returns:
            EvaluationResult with generated patch and comparison
        """
        print(f"\n{'='*60}")
        print(f"Instance: {instance.instance_id}")
        print(f"Repo: {instance.repo}")
        print(f"{'='*60}")

        # Show problem statement (truncated)
        problem_preview = instance.problem_statement[:500]
        if len(instance.problem_statement) > 500:
            problem_preview += "..."
        print(f"\nProblem:\n{problem_preview}\n")

        # Generate patch
        print("Generating patch...")
        patch_result = self.patcher.patch_from_swe_bench(
            problem_statement=instance.problem_statement,
            repo=instance.repo,
            hints_text=instance.hints_text if instance.hints_text else None
        )

        if not patch_result.success or not patch_result.patches:
            result = EvaluationResult(
                instance_id=instance.instance_id,
                repo=instance.repo,
                success=False,
                generated_patch=None,
                ground_truth_patch=instance.patch,
                explanation="Failed to generate patch",
                error=patch_result.error_message
            )
            self.results.append(result)
            print(f"FAILED: {patch_result.error_message}")
            return result

        generated_patch = patch_result.patches[0]

        # Compare with ground truth (basic comparison)
        result = EvaluationResult(
            instance_id=instance.instance_id,
            repo=instance.repo,
            success=True,
            generated_patch=generated_patch.fixed_code,
            ground_truth_patch=instance.patch,
            explanation=generated_patch.explanation
        )
        self.results.append(result)

        print(f"\nGenerated patch explanation:\n{generated_patch.explanation}")
        print(f"\nConfidence: {generated_patch.confidence:.0%}")

        # Show a preview of the generated fix
        if generated_patch.fixed_code:
            preview = generated_patch.fixed_code[:300]
            if len(generated_patch.fixed_code) > 300:
                preview += "..."
            print(f"\nGenerated fix preview:\n{preview}")

        return result

    def evaluate_all(self, instances: List[SWEBenchInstance]) -> None:
        """Evaluate on multiple instances."""
        print(f"\nEvaluating {len(instances)} instances...\n")

        for i, instance in enumerate(instances, 1):
            print(f"\n[{i}/{len(instances)}]", end="")
            self.evaluate_instance(instance)

        self.print_summary()

    def print_summary(self) -> None:
        """Print evaluation summary."""
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)

        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total instances: {total}")
        print(f"Patches generated: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success rate: {successful/total*100:.1f}%")

        if any(r.error for r in self.results):
            print("\nErrors:")
            for r in self.results:
                if r.error:
                    print(f"  {r.instance_id}: {r.error}")

    def save_results(self, filepath: str) -> None:
        """Save results to JSON file."""
        data = {
            "total": len(self.results),
            "successful": sum(1 for r in self.results if r.success),
            "results": [
                {
                    "instance_id": r.instance_id,
                    "repo": r.repo,
                    "success": r.success,
                    "generated_patch": r.generated_patch,
                    "explanation": r.explanation,
                    "error": r.error
                }
                for r in self.results
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to {filepath}")


def main():
    args = parse_args()

    # Get API key
    api_key = args.api_key or OPENAI_API_KEY or os.environ.get('OPENAI_API_KEY')

    if not api_key and not args.list:
        print("Error: OpenAI API key required.", file=sys.stderr)
        print("Set OPENAI_API_KEY in swe_bench.py, via --api-key, or environment variable.",
              file=sys.stderr)
        sys.exit(1)

    # Load dataset
    try:
        loader = SWEBenchLoader()
        loader.load()
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    # Handle --list
    if args.list:
        loader.list_instances()
        sys.exit(0)

    # Create patcher
    patcher = LLMPatcher(api_key=api_key, model=args.model)
    evaluator = SWEBenchEvaluator(patcher)

    # Run evaluation
    if args.instance:
        instance = loader.get_instance(args.instance)
        if not instance:
            print(f"Error: Instance '{args.instance}' not found", file=sys.stderr)
            sys.exit(1)
        evaluator.evaluate_instance(instance)

    elif args.run_dev:
        instances = loader.get_dev_instances()
        if not instances:
            print("No dev instances found", file=sys.stderr)
            sys.exit(1)
        evaluator.evaluate_all(instances)

    elif args.run_all:
        instances = loader.get_all_instances()
        evaluator.evaluate_all(instances)

    else:
        # Default: run on first 5 dev instances
        instances = loader.get_dev_instances()[:5]
        if instances:
            print("Running on first 5 dev instances (use --run-all for full evaluation)")
            evaluator.evaluate_all(instances)
        else:
            print("No instances available")
            sys.exit(1)

    # Save results if requested
    if args.output:
        evaluator.save_results(args.output)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate patcher against SWE-bench Lite dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python swe_bench.py --list                     # List all instances
  python swe_bench.py --instance django__django-11179  # Run specific instance
  python swe_bench.py --run-dev                  # Run on dev split (23 instances)
  python swe_bench.py --run-all --output results.json  # Full evaluation
        """
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available instances'
    )

    parser.add_argument(
        '--instance', '-i',
        help='Run on a specific instance by ID'
    )

    parser.add_argument(
        '--run-dev',
        action='store_true',
        help='Run on all dev split instances'
    )

    parser.add_argument(
        '--run-all',
        action='store_true',
        help='Run on all instances'
    )

    parser.add_argument(
        '--output', '-o',
        help='Save results to JSON file'
    )

    parser.add_argument(
        '--api-key', '-k',
        help='OpenAI API key'
    )

    parser.add_argument(
        '--model', '-m',
        default='gpt-4o-mini',
        help='OpenAI model to use (default: gpt-4o-mini)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
