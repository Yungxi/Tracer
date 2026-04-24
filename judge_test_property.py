"""
TestPropertyJudge - property-aware judge that ALSO runs each property's executable
assertion against the observed call and feeds the pass/fail outcomes into the prompt
as concrete evidence.

Variant 2 of two. Properties come from the same source as variant 1
(properties.PropertyGenerator); the difference is that this judge treats each
property's `assertion` as a one-line unit test and records its outcome per call.

The executor does NOT need to change: we intercept the evidence inside
judge_function_call, then delegate the actual verdict back to the LLM with richer
context.
"""
from __future__ import annotations

from typing import Any, List, Optional

from judge import JudgmentResult, Verdict
from judge_property import PropertyAwareJudge
from properties import TestOutcome, evaluate_property, format_outcomes


class TestPropertyJudge(PropertyAwareJudge):

    SYSTEM_PROMPT = PropertyAwareJudge.SYSTEM_PROMPT + """

You will ALSO be given TEST OUTCOMES: the result of evaluating each property's
executable assertion against the observed call. A FAIL outcome is direct evidence
the call is incorrect on that property. Treat test outcomes as primary evidence
and cite them by property id in your explanation.
"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Outcomes from the most recent judge call; useful for reporters/tests.
        self.last_outcomes: List[TestOutcome] = []
        # Accumulated outcomes keyed by function name; a simple audit log.
        self.history: List[tuple] = []  # (function_name, List[TestOutcome])

    def _run_property_tests(self, function_name: str, args: tuple, kwargs: dict,
                             result: Any) -> List[TestOutcome]:
        return [evaluate_property(p, function_name, args, kwargs, result)
                for p in self.properties]

    def judge_function_call(
        self,
        function_name: str,
        function_source: str,
        args: tuple,
        kwargs: dict,
        result: Any,
        docstring: Optional[str] = None,
        context: Optional[str] = None,
    ) -> JudgmentResult:
        outcomes = self._run_property_tests(function_name, args, kwargs, result)
        self.last_outcomes = outcomes
        self.history.append((function_name, outcomes))

        extra = "TEST OUTCOMES:\n" + format_outcomes(outcomes)
        context = (context + "\n\n" + extra) if context else extra

        judgment = super().judge_function_call(
            function_name=function_name,
            function_source=function_source,
            args=args,
            kwargs=kwargs,
            result=result,
            docstring=docstring,
            context=context,
        )

        # If any hard-assertion failed, force at least INCORRECT regardless of
        # what the LLM said. Executable FAILs are ground truth for that property.
        hard_fails = [o for o in outcomes if o.status == "fail"]
        if hard_fails and judgment.verdict != Verdict.INCORRECT:
            cited = ", ".join(o.property_id for o in hard_fails)
            return JudgmentResult(
                verdict=Verdict.INCORRECT,
                explanation=(f"Property assertion(s) failed: {cited}. "
                             f"LLM additional note: {judgment.explanation}"),
                confidence=max(judgment.confidence, 0.9),
            )
        return judgment
