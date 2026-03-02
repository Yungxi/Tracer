#!/usr/bin/env python3
"""
SWE-bench Integration - Load, test, and VERIFY patches against SWE-bench Lite dataset.

Usage:
    python swe_bench.py --list                    # List available instances
    python swe_bench.py --instance <id>           # Run on specific instance
    python swe_bench.py --instance <id> --verify  # Run and verify with tests
    python swe_bench.py --run-dev                 # Run on dev split only
    python swe_bench.py --run-dev --verify        # Run and verify all dev
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
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
WORKSPACE_DIR = "/tmp/swe_bench_workspace"  # Where to clone repos
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
class VerificationResult:
    """Result of verifying a patch."""
    tests_passed: bool
    fail_to_pass_results: Dict[str, bool] = field(default_factory=dict)
    pass_to_pass_results: Dict[str, bool] = field(default_factory=dict)
    error: Optional[str] = None
    stdout: str = ""
    stderr: str = ""


@dataclass
class EvaluationResult:
    """Result of evaluating the patcher on one instance."""
    instance_id: str
    repo: str
    patch_generated: bool
    generated_patch: Optional[str]
    ground_truth_patch: str
    explanation: str
    verification: Optional[VerificationResult] = None
    patch_matches_ground_truth: Optional[bool] = None
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


class PatchVerifier:
    """Verifies patches by cloning repos and running tests."""

    # Map repo names to GitHub URLs
    REPO_URLS = {
        "astropy/astropy": "https://github.com/astropy/astropy.git",
        "django/django": "https://github.com/django/django.git",
        "matplotlib/matplotlib": "https://github.com/matplotlib/matplotlib.git",
        "pallets/flask": "https://github.com/pallets/flask.git",
        "psf/requests": "https://github.com/psf/requests.git",
        "pytest-dev/pytest": "https://github.com/pytest-dev/pytest.git",
        "scikit-learn/scikit-learn": "https://github.com/scikit-learn/scikit-learn.git",
        "sphinx-doc/sphinx": "https://github.com/sphinx-doc/sphinx.git",
        "sympy/sympy": "https://github.com/sympy/sympy.git",
    }

    def __init__(self, workspace_dir: str = WORKSPACE_DIR):
        self.workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)

    def verify_patch(
        self,
        instance: SWEBenchInstance,
        generated_patch: str,
        timeout: int = 300
    ) -> VerificationResult:
        """
        Verify a patch by cloning the repo, applying it, and running tests.

        Args:
            instance: The SWE-bench instance
            generated_patch: The patch to verify (as a unified diff)
            timeout: Timeout for test execution in seconds

        Returns:
            VerificationResult with test outcomes
        """
        repo_dir = None
        try:
            # Clone and setup repo
            repo_dir = self._setup_repo(instance)
            if not repo_dir:
                return VerificationResult(
                    tests_passed=False,
                    error="Failed to clone repository"
                )

            # Apply the patch
            patch_applied = self._apply_patch(repo_dir, generated_patch)
            if not patch_applied:
                return VerificationResult(
                    tests_passed=False,
                    error="Failed to apply generated patch"
                )

            # Run the tests
            return self._run_tests(repo_dir, instance, timeout)

        except Exception as e:
            return VerificationResult(
                tests_passed=False,
                error=f"Verification error: {str(e)}"
            )
        finally:
            # Cleanup
            if repo_dir and os.path.exists(repo_dir):
                shutil.rmtree(repo_dir, ignore_errors=True)

    def _setup_repo(self, instance: SWEBenchInstance) -> Optional[str]:
        """Clone repo at the base commit."""
        repo_url = self.REPO_URLS.get(instance.repo)
        if not repo_url:
            print(f"  Unknown repo: {instance.repo}")
            return None

        # Create unique directory for this instance
        repo_dir = os.path.join(self.workspace_dir, instance.instance_id.replace("/", "_"))
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)

        print(f"  Cloning {instance.repo}...")
        try:
            # Shallow clone to save time/space
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, repo_dir],
                check=True,
                capture_output=True,
                timeout=120
            )

            # Fetch the specific commit
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", instance.base_commit],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                timeout=60
            )

            # Checkout the base commit
            subprocess.run(
                ["git", "checkout", instance.base_commit],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                timeout=30
            )

            return repo_dir

        except subprocess.TimeoutExpired:
            print("  Clone timed out")
            return None
        except subprocess.CalledProcessError as e:
            print(f"  Clone failed: {e.stderr.decode() if e.stderr else str(e)}")
            return None

    def _apply_patch(self, repo_dir: str, patch: str) -> bool:
        """Apply a patch to the repo."""
        print("  Applying patch...")
        try:
            # Write patch to temp file
            patch_file = os.path.join(repo_dir, "generated.patch")
            with open(patch_file, 'w') as f:
                f.write(patch)

            # Try to apply with git apply
            result = subprocess.run(
                ["git", "apply", "--check", patch_file],
                cwd=repo_dir,
                capture_output=True
            )

            if result.returncode == 0:
                subprocess.run(
                    ["git", "apply", patch_file],
                    cwd=repo_dir,
                    check=True,
                    capture_output=True
                )
                return True
            else:
                # Try patch command as fallback
                result = subprocess.run(
                    ["patch", "-p1", "-i", patch_file],
                    cwd=repo_dir,
                    capture_output=True
                )
                return result.returncode == 0

        except Exception as e:
            print(f"  Patch application failed: {e}")
            return False

    def _run_tests(
        self,
        repo_dir: str,
        instance: SWEBenchInstance,
        timeout: int
    ) -> VerificationResult:
        """Run the specified tests."""
        print("  Running tests...")

        # Parse test names from FAIL_TO_PASS
        fail_to_pass_tests = []
        if instance.fail_to_pass:
            try:
                fail_to_pass_tests = json.loads(instance.fail_to_pass)
            except json.JSONDecodeError:
                fail_to_pass_tests = [instance.fail_to_pass]

        if not fail_to_pass_tests:
            return VerificationResult(
                tests_passed=False,
                error="No FAIL_TO_PASS tests specified"
            )

        # Run pytest on the tests
        fail_to_pass_results = {}
        all_passed = True

        for test in fail_to_pass_tests[:5]:  # Limit to first 5 tests
            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", test, "-v", "--tb=short"],
                    cwd=repo_dir,
                    capture_output=True,
                    timeout=timeout
                )
                passed = result.returncode == 0
                fail_to_pass_results[test] = passed
                if not passed:
                    all_passed = False
                    print(f"    FAIL: {test}")
                else:
                    print(f"    PASS: {test}")

            except subprocess.TimeoutExpired:
                fail_to_pass_results[test] = False
                all_passed = False
                print(f"    TIMEOUT: {test}")
            except Exception as e:
                fail_to_pass_results[test] = False
                all_passed = False
                print(f"    ERROR: {test} - {e}")

        return VerificationResult(
            tests_passed=all_passed,
            fail_to_pass_results=fail_to_pass_results
        )


class SWEBenchEvaluator:
    """Evaluates the patcher against SWE-bench instances."""

    def __init__(self, patcher: LLMPatcher, verifier: Optional[PatchVerifier] = None):
        self.patcher = patcher
        self.verifier = verifier
        self.results: List[EvaluationResult] = []

    def evaluate_instance(
        self,
        instance: SWEBenchInstance,
        verify: bool = False
    ) -> EvaluationResult:
        """Evaluate the patcher on a single instance."""
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
                patch_generated=False,
                generated_patch=None,
                ground_truth_patch=instance.patch,
                explanation="Failed to generate patch",
                error=patch_result.error_message
            )
            self.results.append(result)
            print(f"FAILED: {patch_result.error_message}")
            return result

        generated_patch = patch_result.patches[0]

        # Compare with ground truth
        patch_similarity = self._compare_patches(
            generated_patch.fixed_code,
            instance.patch
        )

        result = EvaluationResult(
            instance_id=instance.instance_id,
            repo=instance.repo,
            patch_generated=True,
            generated_patch=generated_patch.fixed_code,
            ground_truth_patch=instance.patch,
            explanation=generated_patch.explanation,
            patch_matches_ground_truth=patch_similarity > 0.8
        )

        print(f"\nGenerated patch explanation:\n{generated_patch.explanation}")
        print(f"\nConfidence: {generated_patch.confidence:.0%}")
        print(f"Similarity to ground truth: {patch_similarity:.0%}")

        # Show ground truth patch
        print(f"\n--- Ground Truth Patch ---")
        gt_preview = instance.patch[:500]
        if len(instance.patch) > 500:
            gt_preview += "..."
        print(gt_preview)

        # Show generated patch
        print(f"\n--- Generated Patch ---")
        if generated_patch.fixed_code:
            gen_preview = generated_patch.fixed_code[:500]
            if len(generated_patch.fixed_code) > 500:
                gen_preview += "..."
            print(gen_preview)

        # Verify with actual tests if requested
        if verify and self.verifier and generated_patch.fixed_code:
            print(f"\n--- Verifying Patch ---")
            verification = self.verifier.verify_patch(
                instance,
                generated_patch.fixed_code
            )
            result.verification = verification

            if verification.tests_passed:
                print("VERIFICATION: PASSED - All tests pass!")
            elif verification.error:
                print(f"VERIFICATION: ERROR - {verification.error}")
            else:
                passed = sum(1 for v in verification.fail_to_pass_results.values() if v)
                total = len(verification.fail_to_pass_results)
                print(f"VERIFICATION: FAILED - {passed}/{total} tests passed")

        self.results.append(result)
        return result

    def _compare_patches(self, generated: str, ground_truth: str) -> float:
        """Compare two patches and return similarity score (0-1)."""
        if not generated or not ground_truth:
            return 0.0

        # Simple similarity: ratio of common lines
        gen_lines = set(generated.strip().split('\n'))
        gt_lines = set(ground_truth.strip().split('\n'))

        if not gen_lines or not gt_lines:
            return 0.0

        common = gen_lines & gt_lines
        total = gen_lines | gt_lines

        return len(common) / len(total) if total else 0.0

    def evaluate_all(
        self,
        instances: List[SWEBenchInstance],
        verify: bool = False
    ) -> None:
        """Evaluate on multiple instances."""
        print(f"\nEvaluating {len(instances)} instances...")
        if verify:
            print("(with test verification enabled)")
        print()

        for i, instance in enumerate(instances, 1):
            print(f"\n[{i}/{len(instances)}]", end="")
            self.evaluate_instance(instance, verify=verify)

        self.print_summary()

    def print_summary(self) -> None:
        """Print evaluation summary."""
        total = len(self.results)
        patches_generated = sum(1 for r in self.results if r.patch_generated)
        verified_passed = sum(
            1 for r in self.results
            if r.verification and r.verification.tests_passed
        )
        verified_total = sum(1 for r in self.results if r.verification)

        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total instances: {total}")
        print(f"Patches generated: {patches_generated}/{total} ({patches_generated/total*100:.1f}%)")

        if verified_total > 0:
            print(f"\nVERIFICATION RESULTS:")
            print(f"Tests passed: {verified_passed}/{verified_total} ({verified_passed/verified_total*100:.1f}%)")

            # Show details
            print("\nDetails:")
            for r in self.results:
                if r.verification:
                    status = "PASS" if r.verification.tests_passed else "FAIL"
                    print(f"  [{status}] {r.instance_id}")
                    if r.verification.error:
                        print(f"        Error: {r.verification.error}")

        if any(r.error for r in self.results):
            print("\nGeneration Errors:")
            for r in self.results:
                if r.error:
                    print(f"  {r.instance_id}: {r.error}")

    def save_results(self, filepath: str) -> None:
        """Save results to JSON file."""
        data = {
            "total": len(self.results),
            "patches_generated": sum(1 for r in self.results if r.patch_generated),
            "verified_passed": sum(
                1 for r in self.results
                if r.verification and r.verification.tests_passed
            ),
            "results": [
                {
                    "instance_id": r.instance_id,
                    "repo": r.repo,
                    "patch_generated": r.patch_generated,
                    "generated_patch": r.generated_patch,
                    "ground_truth_patch": r.ground_truth_patch,
                    "explanation": r.explanation,
                    "patch_matches_ground_truth": r.patch_matches_ground_truth,
                    "verification": {
                        "tests_passed": r.verification.tests_passed,
                        "fail_to_pass_results": r.verification.fail_to_pass_results,
                        "error": r.verification.error
                    } if r.verification else None,
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

    # Create patcher and optional verifier
    patcher = LLMPatcher(api_key=api_key, model=args.model)
    verifier = PatchVerifier() if args.verify else None
    evaluator = SWEBenchEvaluator(patcher, verifier)

    # Run evaluation
    if args.instance:
        instance = loader.get_instance(args.instance)
        if not instance:
            print(f"Error: Instance '{args.instance}' not found", file=sys.stderr)
            sys.exit(1)
        evaluator.evaluate_instance(instance, verify=args.verify)

    elif args.run_dev:
        instances = loader.get_dev_instances()
        if not instances:
            print("No dev instances found", file=sys.stderr)
            sys.exit(1)
        evaluator.evaluate_all(instances, verify=args.verify)

    elif args.run_all:
        instances = loader.get_all_instances()
        evaluator.evaluate_all(instances, verify=args.verify)

    else:
        # Default: run on first 3 dev instances
        instances = loader.get_dev_instances()[:3]
        if instances:
            print("Running on first 3 dev instances (use --run-all for full evaluation)")
            evaluator.evaluate_all(instances, verify=args.verify)
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
  python swe_bench.py --list                          # List all instances
  python swe_bench.py --instance django__django-11179 # Run specific instance
  python swe_bench.py --instance django__django-11179 --verify  # Run and verify
  python swe_bench.py --run-dev                       # Run on dev split
  python swe_bench.py --run-dev --verify              # Run and verify all dev
  python swe_bench.py --run-all --output results.json # Full evaluation
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
        '--verify', '-V',
        action='store_true',
        help='Verify patches by cloning repos and running tests'
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
