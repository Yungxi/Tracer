#!/usr/bin/env python3
"""
DSDBench Integration - Data Science Debugging Benchmark Evaluation.

Evaluates LLM ability to debug data science code across four dimensions:
1. Cause Lines - Identifying the source of errors
2. Effect Lines - Pinpointing where errors manifest
3. Error Types - Classifying the nature of bugs
4. Error Messages - Recognizing runtime error descriptions

Dataset: https://github.com/KevinCL16/DSDBench

Usage:
    python dsd_bench.py --list                         # List dataset statistics
    python dsd_bench.py --instance <id>                # Run on specific instance
    python dsd_bench.py --run-single                   # Run single-bug evaluation
    python dsd_bench.py --run-multi                    # Run multi-bug evaluation
    python dsd_bench.py --run-all                      # Run full evaluation
    python dsd_bench.py --compute-metrics <file>       # Compute metrics from results
"""

import argparse
import json
import os
import re
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from patcher import LLMPatcher


# =============================================================================
# CONFIGURATION
# =============================================================================
# Config is in parent directory (project root)
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
MODEL_NAME = "tracer-debugger"

# DSDBench data paths (relative to DSDBench repo or can be overridden)
SINGLE_ERROR_FILE = "bench_final_annotation_single_error.jsonl"
MULTI_ERROR_FILE = "bench_final_annotation_multi_errors.jsonl"


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config.json: {e}")
    return {}


def get_api_key(args_api_key: Optional[str] = None) -> str:
    """
    Get API key from (in order of priority):
    1. Command line argument
    2. Environment variable OPENAI_API_KEY
    3. config.json file
    """
    if args_api_key:
        return args_api_key

    env_key = os.environ.get('OPENAI_API_KEY')
    if env_key:
        return env_key

    config = load_config()
    config_key = config.get('openai_api_key', '')
    if config_key:
        return config_key

    return ""


def get_default_model() -> str:
    """Get default model from config or use fallback."""
    config = load_config()
    return config.get('default_model', 'gpt-4o-mini')


def get_results_dir() -> str:
    """Get results directory from config or use default."""
    config = load_config()
    dsd_config = config.get('dsd_bench', {})
    return dsd_config.get('results_dir', './dsd_bench_results')


def get_data_dir() -> str:
    """Get data directory from config or use default."""
    config = load_config()
    dsd_config = config.get('dsd_bench', {})
    return dsd_config.get('data_dir', '.')


# Load defaults from config
RESULTS_DIR = get_results_dir()
DEFAULT_MODEL = get_default_model()
# =============================================================================


@dataclass
class SingleErrorInstance:
    """A single-bug DSDBench instance."""
    id: int
    question: str
    error_versions: List[Dict[str, Any]]  # List of error variants


@dataclass
class MultiErrorInstance:
    """A multi-bug DSDBench instance."""
    id: int
    question: str
    modified_code: str
    error_count: int
    execution_outputs: List[str]
    effect_error_lines: List[str]
    cause_error_lines: List[str]
    original_sample_id: int


@dataclass
class DebugPrediction:
    """LLM prediction for a single error."""
    cause_line: str
    effect_line: str
    error_message: str
    raw_response: Optional[str] = None


@dataclass
class EvalResult:
    """Evaluation result for a single prediction."""
    cause_line_score: int  # 0 or 1
    effect_line_score: int  # 0 or 1
    error_type_score: int  # 0 or 1
    error_message_score: float  # 0.0, 0.25, 0.5, 0.75, 1.0
    error_message_eval_reason: str = ""


@dataclass
class DimensionMetrics:
    """Metrics for a single evaluation dimension."""
    TP: int = 0
    FP: int = 0
    FN: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0


# =============================================================================
# DATA LOADER
# =============================================================================

class DSDBenchLoader:
    """Loads DSDBench datasets from JSONL files."""

    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self._single_instances: Dict[int, SingleErrorInstance] = {}
        self._multi_instances: Dict[int, MultiErrorInstance] = {}

    def load_single_error(self, filepath: Optional[str] = None) -> None:
        """Load single-error benchmark data."""
        if filepath is None:
            filepath = os.path.join(self.data_dir, SINGLE_ERROR_FILE)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Single-error data not found: {filepath}")

        self._single_instances.clear()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                instance = SingleErrorInstance(
                    id=data['id'],
                    question=data['question'],
                    error_versions=data.get('error_versions', [])
                )
                self._single_instances[instance.id] = instance

        print(f"Loaded {len(self._single_instances)} single-error instances")

    def load_multi_error(self, filepath: Optional[str] = None) -> None:
        """Load multi-error benchmark data."""
        if filepath is None:
            filepath = os.path.join(self.data_dir, MULTI_ERROR_FILE)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Multi-error data not found: {filepath}")

        self._multi_instances.clear()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                instance = MultiErrorInstance(
                    id=data.get('id', 0),
                    question=data['question'],
                    modified_code=data['modified_code'],
                    error_count=data.get('error_count', 0),
                    execution_outputs=data.get('execution_outputs', []),
                    effect_error_lines=data.get('effect_error_lines', []),
                    cause_error_lines=data.get('cause_error_lines', []),
                    original_sample_id=data.get('original_sample_id', 0)
                )
                self._multi_instances[instance.id] = instance

        print(f"Loaded {len(self._multi_instances)} multi-error instances")

    def get_single_instance(self, instance_id: int) -> Optional[SingleErrorInstance]:
        return self._single_instances.get(instance_id)

    def get_multi_instance(self, instance_id: int) -> Optional[MultiErrorInstance]:
        return self._multi_instances.get(instance_id)

    def get_all_single_instances(self) -> List[SingleErrorInstance]:
        return list(self._single_instances.values())

    def get_all_multi_instances(self) -> List[MultiErrorInstance]:
        return list(self._multi_instances.values())

    def get_single_ids(self) -> List[int]:
        return list(self._single_instances.keys())

    def get_multi_ids(self) -> List[int]:
        return list(self._multi_instances.keys())


# =============================================================================
# BUG DEBUGGER (LLM-based)
# =============================================================================

class BugDebugger:
    """Uses LLM to analyze code and identify bugs."""

    SINGLE_BUG_SYSTEM_PROMPT = """You will be provided with an original query and a data analysis code. Your task is to:

1. Read the question carefully and identify if there are any logic errors injected into the code.
2. For each logic error:
  - Locate the Cause: Specify the exact line of code that causes the issue.
  - Locate the Effect: Identify the line of code where the error will be triggered and the interpreter will throw an error.
  - Error Description: Provide a concise description of the error message thrown by the Python Interpreter (not the full traceback).

Output Format:
```json
{
    "cause_line": "Specify the exact line of code causing the issue",
    "effect_line": "Specify the exact line of code where the error will be triggered",
    "error_message": "Provide a concise description of the error message thrown by the Python Interpreter (not the full traceback)"
}
```

There will be only one error in the code. Output only ONE json dict in your response."""

    SINGLE_BUG_USER_PROMPT = """You are given the following query and data analysis code.

### Original Query:
{query}


### Data Analysis Code:
{code}


1. Read the question carefully and identify if there are any logic error injected into the code.
2. For each logic error:
  - Locate the Cause: Specify the exact line of code that causes the issue.
  - Locate the Effect: Identify the line of code where the error will be triggered and the interpreter will throw an error.
  - Error Description: Provide a concise description of the error message thrown by the Python Interpreter (not the full traceback).

### Output Format:
```json
{{
    "cause_line": "Specify the exact line of code causing the issue",
    "effect_line": "Specify the exact line of code where the error will be triggered",
    "error_message": "Provide a concise description of the error message thrown by the Python Interpreter (not the full traceback)"
}}
```

There will be only one error in the code. Output only ONE json dict in your response."""

    MULTI_BUG_SYSTEM_PROMPT = """You will be provided with a data analysis code. Your task is to:

1. Read the code carefully and identify all logic errors injected into the code. There will be two or more logic errors in the code.
2. For each logic error you identify:
  - Locate the Cause: Specify the exact line of code that causes the issue.
  - Locate the Effect: Identify the line of code where the error will be triggered and the interpreter will throw an error or where the incorrect behavior is observed.
  - Error Description: Provide a concise description of the error message thrown by the Python Interpreter (not the full traceback). Focus on the *type* of error and the *reason* if possible from the output.

Output Format:
```json
[
    {
        "cause_line": "Specify the exact line of code causing error 1",
        "effect_line": "Specify the exact line of code where error 1 is triggered",
        "error_message": "Concise error message for error 1"
    },
    {
        "cause_line": "Specify the exact line of code causing error 2",
        "effect_line": "Specify the exact line of code where error 2 is triggered",
        "error_message": "Concise error message for error 2"
    }
]
```

There will be more than one error in the code. Output only ONE json block in your response."""

    MULTI_BUG_USER_PROMPT = """You are given the following query and data analysis code.

### Original Query:
{query}


### Data Analysis Code:
{code}


1. Read the code carefully and identify all logic errors injected into the code. There will be two or more logic errors in the code.
2. For each logic error you identify:
  - Locate the Cause: Specify the exact line of code that causes the issue.
  - Locate the Effect: Identify the line of code where the error will be triggered.
  - Error Description: Provide a concise description of the error message thrown by the Python Interpreter.

Output Format:
```json
[
    {{
        "cause_line": "Specify the exact line of code causing error 1",
        "effect_line": "Specify the exact line of code where error 1 is triggered",
        "error_message": "Concise error message for error 1"
    }}
]
```

Output only ONE json block in your response."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if OpenAI is None:
            raise ImportError("openai package required. Install: pip install openai")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Make an LLM API call with retry logic."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API error: {e}")
            return ""

    def _extract_json(self, response: str, expect_list: bool = False) -> Any:
        """Extract JSON from LLM response."""
        if not response:
            return [] if expect_list else {}

        # Remove markdown code blocks if present
        cleaned = response.strip()

        # Handle ```json ... ``` blocks
        if cleaned.startswith('```'):
            # Remove opening ```json or ```
            lines = cleaned.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            # Remove closing ```
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned = '\n'.join(lines)

        # Try to find JSON block
        if expect_list:
            match = re.search(r'\[\s*\{.*\}\s*\]', cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        else:
            # Find JSON object
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start:end + 1])
                except json.JSONDecodeError:
                    pass

        return [] if expect_list else {}

    def analyze_single_bug(self, query: str, code: str) -> DebugPrediction:
        """Analyze code for a single bug."""
        user_prompt = self.SINGLE_BUG_USER_PROMPT.format(query=query, code=code)
        response = self._call_llm(self.SINGLE_BUG_SYSTEM_PROMPT, user_prompt)

        result = self._extract_json(response, expect_list=False)

        return DebugPrediction(
            cause_line=result.get('cause_line', ''),
            effect_line=result.get('effect_line', ''),
            error_message=result.get('error_message', ''),
            raw_response=response
        )

    def analyze_multi_bug(self, query: str, code: str) -> List[DebugPrediction]:
        """Analyze code for multiple bugs."""
        user_prompt = self.MULTI_BUG_USER_PROMPT.format(query=query, code=code)
        response = self._call_llm(self.MULTI_BUG_SYSTEM_PROMPT, user_prompt)

        results = self._extract_json(response, expect_list=True)

        predictions = []
        for r in results:
            predictions.append(DebugPrediction(
                cause_line=r.get('cause_line', ''),
                effect_line=r.get('effect_line', ''),
                error_message=r.get('error_message', ''),
                raw_response=response
            ))

        return predictions


# =============================================================================
# EXACT MATCH EVALUATOR
# =============================================================================

class ExactMatchEvaluator:
    """Evaluates predictions using exact matching for code lines and error types."""

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        if text is None:
            return ""
        return text.strip().lower()

    @staticmethod
    def _exact_match(output: str, ground_truth: str) -> int:
        """Check if two strings match exactly (case-insensitive)."""
        out_norm = ExactMatchEvaluator._normalize(output)
        gt_norm = ExactMatchEvaluator._normalize(ground_truth)

        if not out_norm and not gt_norm:
            return 1
        if not out_norm or not gt_norm:
            return 0

        return 1 if out_norm == gt_norm else 0

    @staticmethod
    def _extract_error_type(error_message: str) -> str:
        """Extract error type from error message (text before first colon)."""
        if not error_message:
            return ""
        parts = error_message.split(':', 1)
        return parts[0].strip() if parts else ""

    def evaluate(
        self,
        prediction: DebugPrediction,
        gt_cause_line: str,
        gt_effect_line: str,
        gt_error_message: str
    ) -> EvalResult:
        """
        Evaluate a single prediction against ground truth.

        Returns evaluation scores for cause_line, effect_line, error_type.
        error_message_score should be computed separately via LLM.
        """
        pred_error_type = self._extract_error_type(prediction.error_message)
        gt_error_type = self._extract_error_type(gt_error_message)

        return EvalResult(
            cause_line_score=self._exact_match(prediction.cause_line, gt_cause_line),
            effect_line_score=self._exact_match(prediction.effect_line, gt_effect_line),
            error_type_score=self._exact_match(pred_error_type, gt_error_type),
            error_message_score=0.0  # To be filled by LLM evaluator
        )


# =============================================================================
# ERROR MESSAGE EVALUATOR (LLM-based)
# =============================================================================

class ErrorMessageEvaluator:
    """Uses LLM to evaluate error message similarity."""

    EVAL_PROMPT = """You are provided with the following error message analysis:

### Ground Truth Error Message:
{gt_error_message}

### LLM Output Error Message:
{pred_error_message}

### Evaluation Task:
Evaluate the LLM's error message against the Ground Truth error message.

### Evaluation Criteria:
- **1.0**: The error message in the LLM Output **exactly matches** the Ground Truth (including all key details).
- **0.75**: The error message is **mostly correct** but lacks minor details.
- **0.5**: The error message is **partially correct** but contains vague or incomplete information.
- **0.25**: The error message is **only loosely related** to the Ground Truth.
- **0.0**: The error message is **completely irrelevant or incorrect**.

### Output Format:
```json
{{
    "error_message_score": 0.0,
    "error_message_eval_reason": "Scoring justification (in English)"
}}
```"""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if OpenAI is None:
            raise ImportError("openai package required.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def evaluate(self, pred_error_message: str, gt_error_message: str) -> Tuple[float, str]:
        """Evaluate error message similarity using LLM."""
        prompt = self.EVAL_PROMPT.format(
            gt_error_message=gt_error_message,
            pred_error_message=pred_error_message
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            result_text = response.choices[0].message.content

            # Extract JSON
            start = result_text.rfind('{')
            end = result_text.rfind('}')
            if start != -1 and end != -1:
                result = json.loads(result_text[start:end + 1])
                return (
                    float(result.get('error_message_score', 0.0)),
                    result.get('error_message_eval_reason', '')
                )
        except Exception as e:
            print(f"Error message evaluation failed: {e}")

        return 0.0, "Evaluation failed"


# =============================================================================
# METRICS CALCULATOR
# =============================================================================

def extract_traceback_message(error_str: str) -> Optional[str]:
    """Extract the final error message from a traceback string."""
    if not error_str:
        return None
    pattern = r"Traceback \(most recent call last\):.*"
    match = re.search(pattern, error_str, re.DOTALL)
    if match:
        lines = match.group(0).split('\n')
        return lines[-2] if len(lines) >= 2 else lines[-1]
    return None


def calculate_dimension_metrics(
    eval_results: List[EvalResult],
    total_ground_truth: int,
    dimension: str
) -> DimensionMetrics:
    """Calculate precision, recall, F1, accuracy for a dimension."""
    metrics = DimensionMetrics()

    for result in eval_results:
        score = getattr(result, f"{dimension}_score")
        if dimension == "error_message":
            is_tp = score >= 0.75
        else:
            is_tp = score == 1

        if is_tp:
            metrics.TP += 1
        else:
            metrics.FP += 1

    metrics.FN = max(0, total_ground_truth - (metrics.TP + metrics.FP))

    # Calculate metrics
    if metrics.TP + metrics.FP > 0:
        metrics.precision = metrics.TP / (metrics.TP + metrics.FP)
    if metrics.TP + metrics.FN + metrics.FP > 0:
        metrics.recall = metrics.TP / (metrics.TP + metrics.FN + metrics.FP)
    if metrics.precision + metrics.recall > 0:
        metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
    if total_ground_truth > 0:
        metrics.accuracy = metrics.TP / total_ground_truth

    return metrics


def calculate_all_metrics(
    eval_results: List[EvalResult],
    total_ground_truth: int
) -> Dict[str, DimensionMetrics]:
    """Calculate metrics for all four dimensions."""
    dimensions = ["cause_line", "effect_line", "error_type", "error_message"]
    return {
        dim: calculate_dimension_metrics(eval_results, total_ground_truth, dim)
        for dim in dimensions
    }


# =============================================================================
# EVALUATOR
# =============================================================================

class DSDBenchEvaluator:
    """Orchestrates DSDBench evaluation."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model
        self.debugger = BugDebugger(api_key, model)
        self.exact_evaluator = ExactMatchEvaluator()
        self.message_evaluator = ErrorMessageEvaluator(api_key, model)
        self.results: List[Dict[str, Any]] = []

    def evaluate_single_instance(
        self,
        instance: SingleErrorInstance,
        verbose: bool = True
    ) -> List[EvalResult]:
        """Evaluate a single-error instance."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating instance {instance.id}")
            print(f"{'='*60}")

        eval_results = []

        for idx, error_version in enumerate(instance.error_versions):
            modified_code = error_version.get('modified_code', '')
            exec_output = error_version.get('execution_output', '')
            gt_cause = error_version.get('cause_error_line', '')
            gt_effect = error_version.get('effect_error_line', '')
            gt_error_msg = extract_traceback_message(exec_output) or ''

            if verbose:
                print(f"\n  Error version {idx + 1}/{len(instance.error_versions)}")

            # Get LLM prediction
            prediction = self.debugger.analyze_single_bug(instance.question, modified_code)

            if verbose:
                print(f"    Predicted cause: {prediction.cause_line[:50]}...")
                print(f"    GT cause: {gt_cause[:50]}...")

            # Exact match evaluation
            result = self.exact_evaluator.evaluate(
                prediction, gt_cause, gt_effect, gt_error_msg
            )

            # LLM evaluation for error message
            msg_score, msg_reason = self.message_evaluator.evaluate(
                prediction.error_message, gt_error_msg
            )
            result.error_message_score = msg_score
            result.error_message_eval_reason = msg_reason

            eval_results.append(result)

            if verbose:
                print(f"    Scores: cause={result.cause_line_score}, "
                      f"effect={result.effect_line_score}, "
                      f"type={result.error_type_score}, "
                      f"message={result.error_message_score}")

        return eval_results

    def evaluate_multi_instance(
        self,
        instance: MultiErrorInstance,
        verbose: bool = True
    ) -> List[EvalResult]:
        """Evaluate a multi-error instance."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating multi-error instance {instance.id}")
            print(f"{'='*60}")

        # Get LLM predictions for all errors
        predictions = self.debugger.analyze_multi_bug(
            instance.question, instance.modified_code
        )

        eval_results = []

        # Extract ground truth error messages
        gt_error_messages = []
        for exec_out in instance.execution_outputs:
            msg = extract_traceback_message(exec_out)
            gt_error_messages.append(msg or '')

        # Evaluate each prediction against ground truth
        for pred_idx, prediction in enumerate(predictions):
            best_result = None
            best_score = -1

            # Find best matching ground truth
            for gt_idx in range(len(instance.cause_error_lines)):
                gt_cause = instance.cause_error_lines[gt_idx] if gt_idx < len(instance.cause_error_lines) else ''
                gt_effect = instance.effect_error_lines[gt_idx] if gt_idx < len(instance.effect_error_lines) else ''
                gt_msg = gt_error_messages[gt_idx] if gt_idx < len(gt_error_messages) else ''

                result = self.exact_evaluator.evaluate(
                    prediction, gt_cause, gt_effect, gt_msg
                )

                total_score = result.cause_line_score + result.effect_line_score + result.error_type_score
                if total_score > best_score:
                    best_score = total_score
                    best_result = result
                    best_gt_msg = gt_msg

            if best_result is None:
                best_result = EvalResult(0, 0, 0, 0.0)
                best_gt_msg = ''

            # LLM evaluation for error message
            msg_score, msg_reason = self.message_evaluator.evaluate(
                prediction.error_message, best_gt_msg
            )
            best_result.error_message_score = msg_score
            best_result.error_message_eval_reason = msg_reason

            eval_results.append(best_result)

            if verbose:
                print(f"  Prediction {pred_idx + 1}: cause={best_result.cause_line_score}, "
                      f"effect={best_result.effect_line_score}, "
                      f"type={best_result.error_type_score}, "
                      f"message={best_result.error_message_score}")

        return eval_results

    def run_single_evaluation(
        self,
        loader: DSDBenchLoader,
        instance_ids: Optional[List[int]] = None,
        result_file: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run evaluation on single-error instances."""
        instances = loader.get_all_single_instances()
        if instance_ids:
            instances = [inst for inst in instances if inst.id in instance_ids]

        all_eval_results = []
        total_ground_truth = 0

        for instance in instances:
            eval_results = self.evaluate_single_instance(instance, verbose)
            all_eval_results.extend(eval_results)
            total_ground_truth += len(instance.error_versions)

            # Save intermediate result
            self.results.append({
                'id': instance.id,
                'eval_result': [
                    {
                        'cause_line_score': r.cause_line_score,
                        'effect_line_score': r.effect_line_score,
                        'error_type_score': r.error_type_score,
                        'error_message_score': r.error_message_score,
                        'error_message_eval_reason': r.error_message_eval_reason
                    }
                    for r in eval_results
                ]
            })

        # Calculate metrics
        metrics = calculate_all_metrics(all_eval_results, total_ground_truth)

        # Save results
        if result_file:
            self._save_results(result_file)

        return {
            'total_instances': len(instances),
            'total_errors': total_ground_truth,
            'metrics': {
                dim: {
                    'precision': m.precision,
                    'recall': m.recall,
                    'f1_score': m.f1_score,
                    'accuracy': m.accuracy,
                    'TP': m.TP,
                    'FP': m.FP,
                    'FN': m.FN
                }
                for dim, m in metrics.items()
            }
        }

    def run_multi_evaluation(
        self,
        loader: DSDBenchLoader,
        instance_ids: Optional[List[int]] = None,
        result_file: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run evaluation on multi-error instances."""
        instances = loader.get_all_multi_instances()
        if instance_ids:
            instances = [inst for inst in instances if inst.id in instance_ids]

        all_eval_results = []
        total_ground_truth = 0

        for instance in instances:
            eval_results = self.evaluate_multi_instance(instance, verbose)
            all_eval_results.extend(eval_results)
            total_ground_truth += instance.error_count

            self.results.append({
                'id': instance.id,
                'eval_result': [
                    {
                        'cause_line_score': r.cause_line_score,
                        'effect_line_score': r.effect_line_score,
                        'error_type_score': r.error_type_score,
                        'error_message_score': r.error_message_score,
                        'error_message_eval_reason': r.error_message_eval_reason
                    }
                    for r in eval_results
                ]
            })

        metrics = calculate_all_metrics(all_eval_results, total_ground_truth)

        if result_file:
            self._save_results(result_file)

        return {
            'total_instances': len(instances),
            'total_errors': total_ground_truth,
            'metrics': {
                dim: {
                    'precision': m.precision,
                    'recall': m.recall,
                    'f1_score': m.f1_score,
                    'accuracy': m.accuracy,
                    'TP': m.TP,
                    'FP': m.FP,
                    'FN': m.FN
                }
                for dim, m in metrics.items()
            }
        }

    def _save_results(self, filepath: str) -> None:
        """Save evaluation results to JSONL file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for result in self.results:
                f.write(json.dumps(result) + '\n')
        print(f"\nResults saved to: {filepath}")


# =============================================================================
# CLI
# =============================================================================

def print_metrics(summary: Dict[str, Any]) -> None:
    """Print evaluation metrics in a formatted way."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total instances: {summary['total_instances']}")
    print(f"Total errors evaluated: {summary['total_errors']}")

    print("\nDimension-wise Metrics:")
    print("-" * 60)

    for dim, m in summary['metrics'].items():
        print(f"\n{dim.upper().replace('_', ' ')}:")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1-Score:  {m['f1_score']:.4f}")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  (TP={m['TP']}, FP={m['FP']}, FN={m['FN']})")


def main():
    parser = argparse.ArgumentParser(
        description='DSDBench Evaluation - Data Science Debugging Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data options
    parser.add_argument('--data-dir', type=str, default=get_data_dir(),
                        help='Directory containing DSDBench data files')
    parser.add_argument('--single-file', type=str,
                        help='Path to single-error JSONL file')
    parser.add_argument('--multi-file', type=str,
                        help='Path to multi-error JSONL file')

    # Run options
    parser.add_argument('--list', action='store_true',
                        help='List dataset statistics')
    parser.add_argument('--instance', type=int,
                        help='Run on specific instance ID')
    parser.add_argument('--run-single', action='store_true',
                        help='Run single-bug evaluation')
    parser.add_argument('--run-multi', action='store_true',
                        help='Run multi-bug evaluation')
    parser.add_argument('--run-all', action='store_true',
                        help='Run full evaluation (single + multi)')
    parser.add_argument('--ids', type=str,
                        help='Comma-separated list of instance IDs to evaluate')
    parser.add_argument('--limit', type=int,
                        help='Limit number of instances to evaluate')

    # Output options
    parser.add_argument('--result-file', type=str,
                        help='Path to save results JSONL file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    # Model options
    parser.add_argument('--api-key', type=str,
                        help='OpenAI API key (or set in OPENAI_API_KEY env var or config.json)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f'Model to use (default: {DEFAULT_MODEL})')

    # Metrics computation
    parser.add_argument('--compute-metrics', type=str, metavar='FILE',
                        help='Compute metrics from existing results file')

    args = parser.parse_args()

    # Get API key (priority: CLI arg > env var > config.json)
    api_key = get_api_key(args.api_key)
    if not api_key and not args.list and not args.compute_metrics:
        print("Error: OpenAI API key required.")
        print("  Set via --api-key, OPENAI_API_KEY env var, or in config.json")
        sys.exit(1)

    # Initialize loader
    loader = DSDBenchLoader(args.data_dir)

    # Handle --list
    if args.list:
        try:
            loader.load_single_error(args.single_file)
        except FileNotFoundError as e:
            print(f"Warning: {e}")

        try:
            loader.load_multi_error(args.multi_file)
        except FileNotFoundError as e:
            print(f"Warning: {e}")

        print("\nDataset Statistics:")
        print(f"  Single-error instances: {len(loader.get_single_ids())}")
        print(f"  Multi-error instances: {len(loader.get_multi_ids())}")

        if loader.get_single_ids():
            total_errors = sum(
                len(inst.error_versions) for inst in loader.get_all_single_instances()
            )
            print(f"  Total single-error variants: {total_errors}")

        sys.exit(0)

    # Handle --compute-metrics
    if args.compute_metrics:
        if not os.path.exists(args.compute_metrics):
            print(f"Error: File not found: {args.compute_metrics}")
            sys.exit(1)

        with open(args.compute_metrics, 'r') as f:
            results = [json.loads(line) for line in f]

        all_eval_results = []
        total = 0
        for record in results:
            for er in record.get('eval_result', []):
                all_eval_results.append(EvalResult(
                    cause_line_score=er.get('cause_line_score', 0),
                    effect_line_score=er.get('effect_line_score', 0),
                    error_type_score=er.get('error_type_score', 0),
                    error_message_score=er.get('error_message_score', 0.0),
                    error_message_eval_reason=er.get('error_message_eval_reason', '')
                ))
                total += 1

        metrics = calculate_all_metrics(all_eval_results, total)
        summary = {
            'total_instances': len(results),
            'total_errors': total,
            'metrics': {
                dim: {
                    'precision': m.precision,
                    'recall': m.recall,
                    'f1_score': m.f1_score,
                    'accuracy': m.accuracy,
                    'TP': m.TP,
                    'FP': m.FP,
                    'FN': m.FN
                }
                for dim, m in metrics.items()
            }
        }
        print_metrics(summary)
        sys.exit(0)

    # Parse instance IDs
    instance_ids = None
    if args.ids:
        instance_ids = [int(x.strip()) for x in args.ids.split(',')]
    elif args.instance:
        instance_ids = [args.instance]

    # Create evaluator
    evaluator = DSDBenchEvaluator(api_key, args.model)
    verbose = not args.quiet

    # Run evaluations
    if args.run_single or args.run_all:
        try:
            loader.load_single_error(args.single_file)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

        ids = instance_ids
        if args.limit:
            all_ids = loader.get_single_ids()[:args.limit]
            ids = all_ids if not ids else [i for i in ids if i in all_ids]

        result_file = args.result_file or os.path.join(
            RESULTS_DIR, f'eval_{args.model.replace("/", "_")}_single_bug.jsonl'
        )

        summary = evaluator.run_single_evaluation(loader, ids, result_file, verbose)
        print_metrics(summary)

    if args.run_multi or args.run_all:
        try:
            loader.load_multi_error(args.multi_file)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

        ids = instance_ids
        if args.limit:
            all_ids = loader.get_multi_ids()[:args.limit]
            ids = all_ids if not ids else [i for i in ids if i in all_ids]

        result_file = args.result_file or os.path.join(
            RESULTS_DIR, f'eval_{args.model.replace("/", "_")}_multi_bug.jsonl'
        )

        evaluator.results = []  # Reset results for multi
        summary = evaluator.run_multi_evaluation(loader, ids, result_file, verbose)
        print_metrics(summary)

    if not (args.run_single or args.run_multi or args.run_all):
        parser.print_help()


if __name__ == "__main__":
    main()
