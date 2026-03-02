"""
Patcher - Uses LLM to generate code fixes for detected issues.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from judge import JudgmentResult, Verdict
from executor import ExecutionResult, FunctionCall


@dataclass
class Patch:
    """A suggested code patch."""
    original_code: str
    fixed_code: str
    explanation: str
    confidence: float
    affected_lines: Optional[List[int]] = None


@dataclass
class PatchResult:
    """Result of a patching attempt."""
    success: bool
    patches: List[Patch]
    error_message: Optional[str] = None


class LLMPatcher:
    """Uses LLM to generate code fixes."""

    SYSTEM_PROMPT = """You are an expert code debugger and fixer. Your task is to analyze code that has bugs or issues and generate correct patches.

When given:
1. The original source code
2. The problem description or error information
3. The expected behavior (goal)

You must:
1. Identify the bug or issue
2. Generate a minimal fix that solves the problem
3. Explain what was wrong and how you fixed it

Respond with a JSON object containing:
- "analysis": brief analysis of the bug (1-2 sentences)
- "fixed_code": the complete fixed code (full file content, not just the changed lines)
- "explanation": what was changed and why
- "confidence": 0-1 indicating confidence in the fix

Important:
- Make minimal changes - only fix what's broken
- Preserve the original code style and formatting
- Don't add unnecessary features or refactoring
- The fixed_code must be complete and runnable"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if OpenAI is None:
            raise ImportError("openai package is required. Install with: pip install openai")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def patch_from_execution_result(
        self,
        source_code: str,
        execution_result: ExecutionResult,
        script_goal: Optional[str] = None
    ) -> PatchResult:
        """
        Generate patches based on execution trace results.

        Args:
            source_code: The original source code
            execution_result: Result from TracingExecutor
            script_goal: The intended goal of the script

        Returns:
            PatchResult with suggested fixes
        """
        # Find the problematic function calls
        failed_calls = [
            call for call in execution_result.function_calls
            if call.judgment and call.judgment.verdict == Verdict.INCORRECT
        ]

        if not failed_calls and not execution_result.error_message:
            return PatchResult(
                success=True,
                patches=[],
                error_message="No issues found to patch"
            )

        # Build problem description
        problems = []
        if execution_result.error_message:
            problems.append(f"Runtime error: {execution_result.error_message}")

        for call in failed_calls:
            problems.append(
                f"Function '{call.name}' returned incorrect result:\n"
                f"  Input: {call.args}, {call.kwargs}\n"
                f"  Output: {call.result}\n"
                f"  Issue: {call.judgment.explanation if call.judgment else 'Unknown'}"
            )

        problem_description = "\n\n".join(problems)

        return self.patch_code(
            source_code=source_code,
            problem_description=problem_description,
            expected_behavior=script_goal
        )

    def patch_code(
        self,
        source_code: str,
        problem_description: str,
        expected_behavior: Optional[str] = None,
        hints: Optional[str] = None
    ) -> PatchResult:
        """
        Generate a patch for code with a known problem.

        Args:
            source_code: The buggy source code
            problem_description: Description of the issue/bug
            expected_behavior: What the code should do
            hints: Optional hints for fixing

        Returns:
            PatchResult with suggested fix
        """
        prompt = self._build_patch_prompt(
            source_code, problem_description, expected_behavior, hints
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4000
            )

            content = response.choices[0].message.content
            return self._parse_patch_response(content, source_code)

        except Exception as e:
            return PatchResult(
                success=False,
                patches=[],
                error_message=f"Failed to generate patch: {str(e)}"
            )

    def patch_from_swe_bench(
        self,
        problem_statement: str,
        repo: str,
        hints_text: Optional[str] = None,
        base_code: Optional[str] = None
    ) -> PatchResult:
        """
        Generate a patch for a SWE-bench style problem.

        Args:
            problem_statement: The GitHub issue description
            repo: Repository name
            hints_text: Optional hints
            base_code: The code to patch (if available)

        Returns:
            PatchResult with suggested fix
        """
        prompt = f"""You are fixing a bug in the {repo} repository.

## Problem (GitHub Issue)
{problem_statement}

{f"## Hints{chr(10)}{hints_text}" if hints_text else ""}

{f"## Code to Fix{chr(10)}```python{chr(10)}{base_code}{chr(10)}```" if base_code else "Note: Base code not provided. Generate a conceptual patch."}

Generate a patch that fixes this issue. If code is provided, fix it. If not, describe the fix in diff format."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4000
            )

            content = response.choices[0].message.content
            return self._parse_patch_response(content, base_code or "")

        except Exception as e:
            return PatchResult(
                success=False,
                patches=[],
                error_message=f"Failed to generate patch: {str(e)}"
            )

    def _build_patch_prompt(
        self,
        source_code: str,
        problem_description: str,
        expected_behavior: Optional[str],
        hints: Optional[str]
    ) -> str:
        """Build the prompt for patch generation."""
        prompt = f"""Fix the following code:

## Source Code
```python
{source_code}
```

## Problem
{problem_description}

{f"## Expected Behavior{chr(10)}{expected_behavior}" if expected_behavior else ""}

{f"## Hints{chr(10)}{hints}" if hints else ""}

Generate a fix for this code. Return the complete fixed code."""

        return prompt

    def _parse_patch_response(self, content: str, original_code: str) -> PatchResult:
        """Parse the LLM response into a PatchResult."""
        try:
            # Try to extract JSON
            json_match = None
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
                json_match = json.loads(json_str)
            elif "{" in content and "}" in content:
                # Try to find JSON object
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                try:
                    json_match = json.loads(json_str)
                except json.JSONDecodeError:
                    pass

            if json_match:
                fixed_code = json_match.get("fixed_code", "")
                # Extract code from markdown if present
                if "```python" in fixed_code:
                    start = fixed_code.find("```python") + 9
                    end = fixed_code.find("```", start)
                    fixed_code = fixed_code[start:end].strip()
                elif "```" in fixed_code:
                    start = fixed_code.find("```") + 3
                    end = fixed_code.find("```", start)
                    fixed_code = fixed_code[start:end].strip()

                patch = Patch(
                    original_code=original_code,
                    fixed_code=fixed_code,
                    explanation=json_match.get("explanation", json_match.get("analysis", "")),
                    confidence=float(json_match.get("confidence", 0.5))
                )
                return PatchResult(success=True, patches=[patch])

            # Fallback: try to extract code block from response
            if "```python" in content:
                start = content.find("```python") + 9
                end = content.find("```", start)
                fixed_code = content[start:end].strip()

                patch = Patch(
                    original_code=original_code,
                    fixed_code=fixed_code,
                    explanation=content[:content.find("```python")].strip(),
                    confidence=0.5
                )
                return PatchResult(success=True, patches=[patch])

            # Last resort: return the whole response as explanation
            return PatchResult(
                success=False,
                patches=[],
                error_message=f"Could not parse patch from response: {content[:500]}"
            )

        except Exception as e:
            return PatchResult(
                success=False,
                patches=[],
                error_message=f"Error parsing patch response: {str(e)}"
            )

    def apply_patch(self, patch: Patch, filepath: str) -> bool:
        """
        Apply a patch to a file.

        Args:
            patch: The patch to apply
            filepath: Path to the file to patch

        Returns:
            True if successful
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(patch.fixed_code)
            return True
        except Exception:
            return False


def generate_diff(original: str, fixed: str) -> str:
    """Generate a unified diff between original and fixed code."""
    import difflib

    original_lines = original.splitlines(keepends=True)
    fixed_lines = fixed.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        fixed_lines,
        fromfile='original',
        tofile='fixed'
    )

    return ''.join(diff)
