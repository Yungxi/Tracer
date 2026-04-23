"""
Property-based localization: derive script-level invariants and check them per step.

A Property is a natural-language invariant plus (optionally) an executable Python
assertion that can be evaluated against a single function call's observed I/O.

Two consumers live next to this module:
  - judge_property.PropertyAwareJudge : places property descriptions in every
    judge prompt as context.
  - judge_test_property.TestPropertyJudge : additionally evaluates each
    assertion per call and feeds pass/fail outcomes to the judge as evidence.

Adapted from the property-derivation phase in
  C:/Research/CodeErrLoc/baselines/unit_test_prop_baseline.py
but scoped down for per-function Tracer-style checks (no coverage loop,
no test-suite synthesis, no subprocess execution).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class Property:
    """One invariant the script must satisfy for any correct execution."""
    id: str
    description: str
    applies_to: List[str] = field(default_factory=list)  # function names; [] == any
    assertion: Optional[str] = None                      # Python expr; None == narrative only

    def to_prompt_line(self) -> str:
        tgt = ", ".join(self.applies_to) if self.applies_to else "any step"
        tail = f" [check: {self.assertion}]" if self.assertion else ""
        return f"- {self.id} ({tgt}): {self.description}{tail}"


@dataclass
class TestOutcome:
    """Result of evaluating one Property.assertion against one function call."""
    property_id: str
    status: str  # "pass" | "fail" | "skip" | "error"
    detail: str = ""


PROP_DERIV_SYSTEM = """You are a test architect. Given a Python script and a one-sentence goal,
infer 3-6 implicit PROPERTIES that any correct execution must satisfy.

A property is:
  - NECESSARY for correctness (violating it => the script is buggy)
  - CHECKABLE from one function's observed inputs/outputs (not the whole program)
  - NON-TAUTOLOGICAL (it does not merely restate the code)

For each property, also supply a short Python expression that returns True iff the property
holds for one call. The expression may reference four names only:
    result, args, kwargs, function_name
Keep expressions one line, stdlib only (no imports). If no executable check is natural,
set "assertion" to null and rely on the description.

Return STRICT JSON with this schema (no prose, no markdown):
{
  "properties": [
    {
      "id": "P1",
      "description": "short natural-language invariant",
      "applies_to": ["pick_top_n"],
      "assertion": "len(result['top']) <= 3"
    }
  ]
}"""


class PropertyGenerator:
    """One-shot LLM caller that derives properties from (goal + script source)."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if OpenAI is None:
            raise ImportError("openai package required")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, script_goal: str, script_source: str,
                 max_props: int = 6) -> List[Property]:
        prompt = (
            f"SCRIPT GOAL: {script_goal}\n\n"
            f"SCRIPT SOURCE:\n```python\n{script_source}\n```\n\n"
            f"Derive up to {max_props} properties. Return JSON only."
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PROP_DERIV_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_completion_tokens=1200,
            )
            content = resp.choices[0].message.content or ""
            return self._parse(content, max_props)
        except Exception:
            return []

    @staticmethod
    def _parse(content: str, max_props: int) -> List[Property]:
        # Strip common fences if the model slips any in.
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.strip()
        # Locate the outer JSON object if any leading prose remains.
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1:
            return []
        try:
            data = json.loads(stripped[start:end + 1])
        except json.JSONDecodeError:
            return []
        out: List[Property] = []
        for item in (data.get("properties") or [])[:max_props]:
            if not isinstance(item, dict):
                continue
            out.append(Property(
                id=str(item.get("id") or f"P{len(out) + 1}"),
                description=str(item.get("description") or "").strip(),
                applies_to=[str(x) for x in (item.get("applies_to") or [])],
                assertion=(item.get("assertion") or None),
            ))
        return out


def evaluate_property(prop: Property, function_name: str, args: tuple,
                      kwargs: dict, result: Any) -> TestOutcome:
    """Run one property's assertion against a function call. Never raises."""
    if prop.applies_to and function_name not in prop.applies_to:
        return TestOutcome(prop.id, "skip", "not applicable to this call")
    if not prop.assertion:
        return TestOutcome(prop.id, "skip", "no executable assertion")
    scope = {
        "result": result,
        "args": args,
        "kwargs": kwargs,
        "function_name": function_name,
    }
    try:
        ok = bool(eval(prop.assertion, {"__builtins__": __builtins__}, scope))
        return TestOutcome(prop.id, "pass" if ok else "fail", prop.assertion)
    except Exception as e:
        return TestOutcome(prop.id, "error", f"{type(e).__name__}: {e}")


def format_properties(props: List[Property]) -> str:
    """Render properties as a bulleted text block for a prompt."""
    if not props:
        return "(none)"
    return "\n".join(p.to_prompt_line() for p in props)


def format_outcomes(outcomes: List[TestOutcome]) -> str:
    """Render per-call test outcomes as a bulleted text block for a prompt."""
    if not outcomes:
        return "(no property checks run)"
    tag = {"pass": "PASS", "fail": "FAIL", "error": "ERROR", "skip": "SKIP"}
    return "\n".join(f"  [{tag[o.status]}] {o.property_id}: {o.detail}" for o in outcomes)
