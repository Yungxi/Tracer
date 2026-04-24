"""
PropertyAwareJudge - Tracer's LLM judge extended with a fixed list of properties.

Every per-call judgment prompt includes the derived properties. The judge is asked
to name which properties the observed I/O violates when issuing an INCORRECT verdict.

Variant 1 of two (see judge_test_property.py for variant 2 with executable checks).
"""
from __future__ import annotations

from typing import Any, List, Optional

from judge import LLMJudge, JudgmentResult
from properties import Property, format_properties


class PropertyAwareJudge(LLMJudge):
    """LLMJudge + a static property list baked into every prompt."""

    SYSTEM_PROMPT = LLMJudge.SYSTEM_PROMPT + """

In addition to the script goal, you will be given a list of PROPERTIES that any correct
execution must satisfy. For each judgment:
  - Explicitly consider which properties the observed input/output might violate.
  - If you mark the call INCORRECT, cite the property ids violated (e.g. "violates P2").
  - A call that satisfies every listed property AND advances the script goal is CORRECT.
"""

    def __init__(self, api_key: str, properties: List[Property],
                 model: str = "gpt-4o-mini", script_goal: Optional[str] = None):
        super().__init__(api_key=api_key, model=model, script_goal=script_goal)
        self.properties = properties

    def _build_prompt(
        self,
        function_name: str,
        function_source: str,
        args: tuple,
        kwargs: dict,
        result: Any,
        docstring: Optional[str],
        context: Optional[str],
    ) -> str:
        base = super()._build_prompt(
            function_name, function_source, args, kwargs, result, docstring, context
        )
        block = "\n\nPROPERTIES (necessary for correctness):\n" + format_properties(self.properties)
        return base + block
