"""
LLM Judge - Evaluates function outputs using OpenAI API.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class Verdict(Enum):
    """Possible verdicts from the LLM judge."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class JudgmentResult:
    """Result of an LLM judgment."""
    verdict: Verdict
    explanation: str
    confidence: float  # 0.0 to 1.0


class LLMJudge:
    """Judges function outputs using OpenAI's API."""

    SYSTEM_PROMPT = """You are a code execution judge. Your task is to evaluate whether a function's output is correct and reasonable given:
1. The SCRIPT GOAL - the overall purpose of the entire script
2. The function's name, docstring, and implementation
3. The inputs provided and output returned

Analyze the function execution and determine if:
1. The output helps achieve the script's stated goal
2. The output makes logical sense given the function name and inputs
3. The output type is appropriate
4. There are any obvious errors, bugs, or incorrect results

Respond with a JSON object containing:
- "verdict": one of "correct", "incorrect", or "unknown"
- "explanation": brief explanation of your judgment (1-2 sentences)
- "confidence": a number between 0 and 1 indicating your confidence

Be strict but fair. Pay special attention to whether the function's behavior aligns with what the script is supposed to accomplish."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", script_goal: Optional[str] = None):
        if OpenAI is None:
            raise ImportError("openai package is required. Install with: pip install openai")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.script_goal = script_goal

    def judge_function_call(
        self,
        function_name: str,
        function_source: str,
        args: tuple,
        kwargs: dict,
        result: Any,
        docstring: Optional[str] = None,
        context: Optional[str] = None
    ) -> JudgmentResult:
        """
        Judge whether a function's output is correct.

        Args:
            function_name: Name of the function
            function_source: Source code of the function
            args: Positional arguments passed to the function
            kwargs: Keyword arguments passed to the function
            result: The actual return value
            docstring: Function docstring if available
            context: Additional context about what the code is doing

        Returns:
            JudgmentResult with verdict, explanation, and confidence
        """
        prompt = self._build_prompt(
            function_name, function_source, args, kwargs, result, docstring, context
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            content = response.choices[0].message.content
            return self._parse_response(content)

        except Exception as e:
            return JudgmentResult(
                verdict=Verdict.ERROR,
                explanation=f"Failed to get LLM judgment: {str(e)}",
                confidence=0.0
            )

    def judge_execution_state(
        self,
        code_snippet: str,
        variables: dict,
        expected_behavior: Optional[str] = None
    ) -> JudgmentResult:
        """
        Judge whether the current execution state looks correct.

        Args:
            code_snippet: The code that was just executed
            variables: Current variable state
            expected_behavior: Description of expected behavior

        Returns:
            JudgmentResult with verdict and explanation
        """
        # Format variables for display (handle non-serializable types)
        var_display = {}
        for k, v in variables.items():
            if k.startswith('_'):
                continue
            try:
                json.dumps(v)
                var_display[k] = v
            except (TypeError, ValueError):
                var_display[k] = f"<{type(v).__name__}>"

        prompt = f"""Evaluate this code execution state:

Code executed:
```python
{code_snippet}
```

Current variables after execution:
{json.dumps(var_display, indent=2, default=str)}

{"Expected behavior: " + expected_behavior if expected_behavior else ""}

Is this execution state reasonable and correct?"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            content = response.choices[0].message.content
            return self._parse_response(content)

        except Exception as e:
            return JudgmentResult(
                verdict=Verdict.ERROR,
                explanation=f"Failed to get LLM judgment: {str(e)}",
                confidence=0.0
            )

    def _build_prompt(
        self,
        function_name: str,
        function_source: str,
        args: tuple,
        kwargs: dict,
        result: Any,
        docstring: Optional[str],
        context: Optional[str]
    ) -> str:
        """Build the prompt for judging a function call."""
        # Format result for display
        try:
            result_str = json.dumps(result, default=str)
        except:
            result_str = repr(result)

        # Format arguments
        args_str = ", ".join(repr(a) for a in args)
        kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))

        # Build goal section
        goal_section = ""
        if self.script_goal:
            goal_section = f"""SCRIPT GOAL: {self.script_goal}

The function being evaluated should contribute to achieving this goal.

---

"""

        prompt = f"""{goal_section}Evaluate this function execution:

Function name: {function_name}
{"Docstring: " + docstring if docstring else ""}

Function source:
```python
{function_source}
```

Called with: {function_name}({all_args})
Returned: {result_str}

{"Additional context: " + context if context else ""}

Given the script's goal, is this output correct and does it help achieve the intended purpose?"""

        return prompt

    def _parse_response(self, content: str) -> JudgmentResult:
        """Parse the LLM response into a JudgmentResult."""
        try:
            # Try to extract JSON from the response
            # Handle case where LLM wraps JSON in markdown code blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            data = json.loads(content)

            verdict_str = data.get("verdict", "unknown").lower()
            verdict_map = {
                "correct": Verdict.CORRECT,
                "incorrect": Verdict.INCORRECT,
                "error": Verdict.ERROR,
                "unknown": Verdict.UNKNOWN
            }
            verdict = verdict_map.get(verdict_str, Verdict.UNKNOWN)

            return JudgmentResult(
                verdict=verdict,
                explanation=data.get("explanation", "No explanation provided"),
                confidence=float(data.get("confidence", 0.5))
            )

        except (json.JSONDecodeError, KeyError, ValueError):
            # If we can't parse JSON, try to infer from text
            content_lower = content.lower()
            if "incorrect" in content_lower or "wrong" in content_lower:
                verdict = Verdict.INCORRECT
            elif "correct" in content_lower or "right" in content_lower:
                verdict = Verdict.CORRECT
            else:
                verdict = Verdict.UNKNOWN

            return JudgmentResult(
                verdict=verdict,
                explanation=content[:200],
                confidence=0.3
            )
