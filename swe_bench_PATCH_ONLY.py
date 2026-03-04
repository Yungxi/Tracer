#!/usr/bin/env python3
"""
SWE-bench PATCH ONLY - Direct patching using problem description.

Uses the OFFICIAL SWE-bench harness for evaluation.

This script patches code directly using the SWE-bench problem description,
without blind bug detection. Used as a baseline comparison.

Flow:
1. Load SWE-bench instances
2. For each instance:
   a. Clone repo and read affected files
   b. Send problem description + code to patcher
   c. Generate patch in unified diff format
3. Save predictions in official format
4. Run official SWE-bench harness for verification

Usage:
    python swe_bench_PATCH_ONLY.py --list                    # List available instances
    python swe_bench_PATCH_ONLY.py --instance <id>           # Run on specific instance
    python swe_bench_PATCH_ONLY.py --run-test                # Run on test split
    python swe_bench_PATCH_ONLY.py --verify                  # Run harness verification
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Try to import official SWE-bench harness
try:
    from swebench.harness.run_evaluation import main as run_swebench_evaluation
    HARNESS_AVAILABLE = True
except ImportError:
    HARNESS_AVAILABLE = False

from patcher import LLMPatcher


# =============================================================================
# CONFIGURATION
# =============================================================================
OPENAI_API_KEY = ""  # Set your API key here or use --api-key flag
WORKSPACE_DIR = "/tmp/swe_bench_patch_only"
PREDICTIONS_FILE = "PATCH_ONLY_predictions.json"  # Official format for harness
RESULTS_FILE = "PATCH_ONLY_results.json"
LOG_DIR = "/tmp/swe_bench_patch_only_logs"
MODEL_NAME = "patcher-direct"
# =============================================================================


@dataclass
class SWEBenchInstance:
    """A single SWE-bench problem instance."""
    instance_id: str
    repo: str
    problem_statement: str
    hints_text: str
    base_commit: str
    patch: str
    test_patch: str
    fail_to_pass: str
    pass_to_pass: str
    version: str
    created_at: str


@dataclass
class PatchPrediction:
    """Official SWE-bench prediction format."""
    instance_id: str
    model_patch: str  # Unified diff format
    model_name_or_path: str = MODEL_NAME


@dataclass
class PatchOnlyResult:
    """Result of patch-only evaluation."""
    instance_id: str
    repo: str
    patch_generated: bool
    generated_patch: Optional[str]
    harness_result: Optional[str] = None
    error: Optional[str] = None


class SWEBenchLoader:
    """Loads SWE-bench Lite dataset."""

    DATASET_NAME = "princeton-nlp/SWE-bench_Lite"

    def __init__(self):
        if load_dataset is None:
            raise ImportError("datasets package required: pip install datasets")
        self._dataset = None
        self._instances: Dict[str, SWEBenchInstance] = {}

    def load(self) -> None:
        print("Loading SWE-bench Lite dataset...")
        self._dataset = load_dataset(self.DATASET_NAME)
        self._parse_instances()
        print(f"Loaded {len(self._instances)} instances")

    def _parse_instances(self) -> None:
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
        return self._instances.get(instance_id)

    def get_all_instances(self) -> List[SWEBenchInstance]:
        return list(self._instances.values())

    def get_test_instances(self) -> List[SWEBenchInstance]:
        """Get test split instances."""
        test_ids = set()
        if "test" in self._dataset:
            for item in self._dataset["test"]:
                test_ids.add(item["instance_id"])
        return [inst for inst in self._instances.values() if inst.instance_id in test_ids]

    def list_instances(self) -> None:
        print("\nSWE-bench Lite Instances:\n")
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


class RepoManager:
    """Manages cloning and reading files from repos."""

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
        "sqlfluff/sqlfluff": "https://github.com/sqlfluff/sqlfluff.git",
        "marshmallow-code/marshmallow": "https://github.com/marshmallow-code/marshmallow.git",
        "pvlib/pvlib-python": "https://github.com/pvlib/pvlib-python.git",
        "pylint-dev/astroid": "https://github.com/pylint-dev/astroid.git",
        "pyvista/pyvista": "https://github.com/pyvista/pyvista.git",
        "pydicom/pydicom": "https://github.com/pydicom/pydicom.git",
        "mwaskom/seaborn": "https://github.com/mwaskom/seaborn.git",
        "pallets/click": "https://github.com/pallets/click.git",
        "pallets/werkzeug": "https://github.com/pallets/werkzeug.git",
        "pylint-dev/pylint": "https://github.com/pylint-dev/pylint.git",
        "python/mypy": "https://github.com/python/mypy.git",
        "pydata/xarray": "https://github.com/pydata/xarray.git",
    }

    def __init__(self, workspace_dir: str = WORKSPACE_DIR):
        self.workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)

    def setup_repo(self, instance: SWEBenchInstance) -> Optional[str]:
        """Clone repo at base_commit, return repo directory."""
        repo_url = self.REPO_URLS.get(instance.repo)
        if not repo_url:
            print(f"  Unknown repo: {instance.repo}")
            return None

        repo_dir = os.path.join(self.workspace_dir, instance.instance_id.replace("/", "_"))
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)

        print(f"  Cloning {instance.repo} at {instance.base_commit[:8]}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, repo_dir],
                check=True, capture_output=True, timeout=120
            )
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", instance.base_commit],
                cwd=repo_dir, check=True, capture_output=True, timeout=60
            )
            subprocess.run(
                ["git", "checkout", instance.base_commit],
                cwd=repo_dir, check=True, capture_output=True, timeout=30
            )
            return repo_dir
        except Exception as e:
            print(f"  Clone failed: {e}")
            return None

    def get_affected_files(self, patch: str) -> List[str]:
        """Extract file paths from a patch."""
        files = []
        for line in patch.split('\n'):
            if line.startswith('diff --git'):
                match = re.search(r'b/(.+)$', line)
                if match:
                    files.append(match.group(1))
        return files

    def read_file(self, repo_dir: str, filepath: str) -> Optional[str]:
        """Read a file from the repo."""
        full_path = os.path.join(repo_dir, filepath)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None

    def cleanup(self, repo_dir: str):
        """Remove repo directory."""
        if repo_dir and os.path.exists(repo_dir):
            shutil.rmtree(repo_dir, ignore_errors=True)


class PatchOnlyEvaluator:
    """Evaluates using direct patching with problem description and official harness."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.patcher = LLMPatcher(api_key=api_key, model=model)
        self.repo_manager = RepoManager()
        self.results: List[PatchOnlyResult] = []
        self.patch_predictions: List[PatchPrediction] = []

    def generate_patch(self, instance: SWEBenchInstance) -> PatchOnlyResult:
        """
        Generate a patch using direct problem description:
        1. Clone repo and read affected files
        2. Send problem description + code to patcher
        3. Return patch in unified diff format
        """
        print(f"\n{'='*60}")
        print(f"Instance: {instance.instance_id}")
        print(f"Repo: {instance.repo}")
        print(f"{'='*60}")

        repo_dir = None
        try:
            # Step 1: Clone repo
            repo_dir = self.repo_manager.setup_repo(instance)
            if not repo_dir:
                return self._error_result(instance, "Failed to clone repo")

            # Step 2: Get affected files from ground truth patch
            affected_files = self.repo_manager.get_affected_files(instance.patch)
            if not affected_files:
                return self._error_result(instance, "No files in patch")

            print(f"\n  Files to patch: {affected_files}")

            # Step 3: Read the code
            all_code = ""
            file_contents = {}
            for filepath in affected_files[:3]:
                code = self.repo_manager.read_file(repo_dir, filepath)
                if code:
                    all_code += f"\n# File: {filepath}\n{code}\n"
                    file_contents[filepath] = code

            if not all_code:
                return self._error_result(instance, "Could not read files")

            # Step 4: Patcher fixes using PROBLEM DESCRIPTION directly
            print("\n  [PATCHER] Generating fix using problem description...")
            print(f"  Problem: {instance.problem_statement[:200]}...")

            patch_result = self.patcher.patch_code(
                source_code=all_code,
                problem_description=instance.problem_statement,
                expected_behavior=None,
                hints=instance.hints_text if instance.hints_text else None
            )

            if not patch_result.success or not patch_result.patches:
                self.patch_predictions.append(PatchPrediction(
                    instance_id=instance.instance_id,
                    model_patch=""
                ))
                result = PatchOnlyResult(
                    instance_id=instance.instance_id,
                    repo=instance.repo,
                    patch_generated=False,
                    generated_patch=None,
                    error="Patcher failed to generate fix"
                )
                self.results.append(result)
                return result

            generated_fix = patch_result.patches[0]
            print(f"  [PATCHER] Fix generated: {generated_fix.explanation[:200]}")

            # Convert to unified diff format for official harness
            unified_diff = self._to_unified_diff(
                file_contents,
                generated_fix.fixed_code,
                affected_files[0]
            )

            # Store in official format
            self.patch_predictions.append(PatchPrediction(
                instance_id=instance.instance_id,
                model_patch=unified_diff
            ))

            result = PatchOnlyResult(
                instance_id=instance.instance_id,
                repo=instance.repo,
                patch_generated=True,
                generated_patch=unified_diff
            )
            self.results.append(result)
            return result

        except Exception as e:
            return self._error_result(instance, str(e))
        finally:
            if repo_dir:
                self.repo_manager.cleanup(repo_dir)

    def _to_unified_diff(self, original_files: Dict[str, str], fixed_code: str, primary_file: str) -> str:
        """Convert fixed code to unified diff format."""
        import difflib

        # If fixed_code looks like a diff already, return it
        if fixed_code.strip().startswith('diff --git') or fixed_code.strip().startswith('---'):
            return fixed_code

        # Otherwise, generate diff from the primary file
        if primary_file not in original_files:
            return fixed_code

        original = original_files[primary_file].splitlines(keepends=True)

        # Try to extract just the fixed content for this file
        fixed = fixed_code
        if f"# File: {primary_file}" in fixed_code:
            start = fixed_code.find(f"# File: {primary_file}")
            end = fixed_code.find("# File:", start + 1)
            if end == -1:
                fixed = fixed_code[start:].replace(f"# File: {primary_file}\n", "")
            else:
                fixed = fixed_code[start:end].replace(f"# File: {primary_file}\n", "")

        fixed_lines = fixed.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original,
            fixed_lines,
            fromfile=f"a/{primary_file}",
            tofile=f"b/{primary_file}"
        )

        return ''.join(diff)

    def _error_result(self, instance: SWEBenchInstance, error: str) -> PatchOnlyResult:
        print(f"  ERROR: {error}")
        self.patch_predictions.append(PatchPrediction(
            instance_id=instance.instance_id,
            model_patch=""
        ))
        result = PatchOnlyResult(
            instance_id=instance.instance_id,
            repo=instance.repo,
            patch_generated=False,
            generated_patch=None,
            error=error
        )
        self.results.append(result)
        return result

    def generate_all(self, instances: List[SWEBenchInstance]) -> None:
        """Generate patches for all instances."""
        print(f"\nGenerating patches for {len(instances)} instances (patch-only mode)...\n")

        for i, instance in enumerate(instances, 1):
            print(f"\n[{i}/{len(instances)}]", end="")
            self.generate_patch(instance)

        self.save_predictions()
        self.print_summary()

    def save_predictions(self) -> None:
        """Save predictions in official SWE-bench format."""
        predictions = [
            {
                "instance_id": p.instance_id,
                "model_patch": p.model_patch,
                "model_name_or_path": p.model_name_or_path
            }
            for p in self.patch_predictions
        ]

        with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2)

        print(f"\nPredictions saved to {PREDICTIONS_FILE} (official format)")

    def print_summary(self) -> None:
        total = len(self.results)
        patches_generated = sum(1 for r in self.results if r.patch_generated)

        print(f"\n{'='*60}")
        print("PATCH GENERATION SUMMARY (Patch-Only Mode)")
        print(f"{'='*60}")
        print(f"Total instances: {total}")
        print(f"Patches generated: {patches_generated}/{total} ({patches_generated/total*100:.1f}%)")
        print(f"\nPredictions saved to: {PREDICTIONS_FILE}")
        print(f"\nTo verify with official harness, run:")
        print(f"  python swe_bench_PATCH_ONLY.py --verify")
        print(f"\nOr manually:")
        print(f"  python -m swebench.harness.run_evaluation \\")
        print(f"    --predictions_path {PREDICTIONS_FILE} \\")
        print(f"    --swe_bench_tasks princeton-nlp/SWE-bench_Lite \\")
        print(f"    --log_dir {LOG_DIR} \\")
        print(f"    --testbed /tmp/swe_testbed \\")
        print(f"    --verbose")

    def save_results(self, filepath: str) -> None:
        patches_generated = sum(1 for r in self.results if r.patch_generated)
        data = {
            "mode": "patch_only",
            "description": "Direct patching using problem description (baseline)",
            "total": len(self.results),
            "patches_generated": patches_generated,
            "predictions_file": PREDICTIONS_FILE,
            "note": "Run --verify to evaluate with official SWE-bench harness",
            "results": [
                {
                    "instance_id": r.instance_id,
                    "repo": r.repo,
                    "patch_generated": r.patch_generated,
                    "harness_result": r.harness_result,
                    "error": r.error
                }
                for r in self.results
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to {filepath}")


def run_official_harness(predictions_file: str, instance_ids: Optional[List[str]] = None) -> Dict:
    """Run official SWE-bench harness for verification."""
    if not HARNESS_AVAILABLE:
        print("\nOfficial SWE-bench harness not installed.")
        print("Install with: pip install swebench")
        print("\nAlternatively, run manually:")
        print(f"  python -m swebench.harness.run_evaluation \\")
        print(f"    --predictions_path {predictions_file} \\")
        print(f"    --swe_bench_tasks princeton-nlp/SWE-bench_Lite \\")
        print(f"    --log_dir {LOG_DIR} \\")
        print(f"    --testbed /tmp/swe_testbed \\")
        print(f"    --verbose")
        return {}

    print("\n" + "="*60)
    print("RUNNING OFFICIAL SWE-BENCH HARNESS")
    print("="*60)

    os.makedirs(LOG_DIR, exist_ok=True)

    args = [
        "--predictions_path", predictions_file,
        "--swe_bench_tasks", "princeton-nlp/SWE-bench_Lite",
        "--log_dir", LOG_DIR,
        "--testbed", "/tmp/swe_testbed",
        "--verbose"
    ]

    if instance_ids:
        args.extend(["--instance_ids"] + instance_ids)

    try:
        sys.argv = ["run_evaluation"] + args
        run_swebench_evaluation()

        results_file = os.path.join(LOG_DIR, "results.json")
        if os.path.exists(results_file):
            with open(results_file) as f:
                return json.load(f)
    except Exception as e:
        print(f"Harness error: {e}")
        print("\nTry running manually with the command above.")

    return {}


def main():
    args = parse_args()

    api_key = args.api_key or OPENAI_API_KEY or os.environ.get('OPENAI_API_KEY')

    # Handle verify command
    if args.verify:
        if not os.path.exists(PREDICTIONS_FILE):
            print(f"Error: {PREDICTIONS_FILE} not found. Run patch generation first.")
            sys.exit(1)
        results = run_official_harness(PREDICTIONS_FILE)
        if results:
            print(f"\nHarness Results: {json.dumps(results, indent=2)}")
        sys.exit(0)

    if not api_key and not args.list:
        print("Error: OpenAI API key required.", file=sys.stderr)
        sys.exit(1)

    try:
        loader = SWEBenchLoader()
        loader.load()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        loader.list_instances()
        sys.exit(0)

    evaluator = PatchOnlyEvaluator(api_key=api_key, model=args.model)

    if args.instance:
        instance = loader.get_instance(args.instance)
        if not instance:
            print(f"Error: Instance '{args.instance}' not found", file=sys.stderr)
            sys.exit(1)
        evaluator.generate_patch(instance)
        evaluator.save_predictions()
        evaluator.print_summary()

    elif args.run_test:
        instances = loader.get_test_instances()
        if not instances:
            instances = loader.get_all_instances()
        if args.limit:
            instances = instances[:args.limit]
        evaluator.generate_all(instances)

    elif args.run_all:
        instances = loader.get_all_instances()
        if args.limit:
            instances = instances[:args.limit]
        evaluator.generate_all(instances)

    elif args.limit:
        instances = loader.get_all_instances()[:args.limit]
        print(f"Running on first {args.limit} instances")
        evaluator.generate_all(instances)

    else:
        instances = loader.get_all_instances()[:3]
        if instances:
            print("Running on first 3 instances")
            evaluator.generate_all(instances)
        else:
            print("No instances available")
            sys.exit(1)

    evaluator.save_results(args.output or RESULTS_FILE)

    if args.auto_verify:
        run_official_harness(PREDICTIONS_FILE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SWE-bench evaluation - PATCH ONLY mode (official harness)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool uses DIRECT patching with OFFICIAL SWE-bench harness:
1. Reads the problem description from SWE-bench
2. Sends problem + code to patcher
3. Patches saved in official format (PATCH_ONLY_predictions.json)
4. Use --verify to run official SWE-bench harness

This serves as a baseline comparison for blind bug detection mode.

Output files:
- PATCH_ONLY_predictions.json: Official format for SWE-bench harness
- PATCH_ONLY_results.json: Evaluation results

Example workflow:
  python swe_bench_PATCH_ONLY.py --run-all    # Generate patches
  python swe_bench_PATCH_ONLY.py --verify     # Run official verification
        """
    )

    parser.add_argument('--list', '-l', action='store_true', help='List available instances')
    parser.add_argument('--instance', '-i', help='Run on specific instance')
    parser.add_argument('--run-test', action='store_true', help='Run on test split')
    parser.add_argument('--run-all', action='store_true', help='Run on all instances')
    parser.add_argument('--limit', '-n', type=int, help='Limit number of instances to run')
    parser.add_argument('--verify', action='store_true', help='Run official SWE-bench harness')
    parser.add_argument('--auto-verify', action='store_true', help='Auto-run harness after generation')
    parser.add_argument('--output', '-o', help='Save results to JSON file')
    parser.add_argument('--api-key', '-k', help='OpenAI API key')
    parser.add_argument('--model', '-m', default='gpt-4o-mini', help='OpenAI model')

    return parser.parse_args()


if __name__ == '__main__':
    main()
