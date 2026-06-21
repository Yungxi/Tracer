#!/usr/bin/env python3
"""Smoke test for vanilla Tracer + the two new localization variants.

Runs the same buggy top-3-customers script three times:
  1. VANILLA         - LLMJudge only (existing behavior).
  2. PROPERTY        - PropertyAwareJudge: derive 3-6 properties once, keep them
                       in every judge prompt as context.
  3. TEST + PROPERTY - TestPropertyJudge: additionally evaluate each property's
                       one-line assertion per call and feed pass/fail to the judge.

For each variant we print the set of errors Tracer surfaced. All three should
localize the bug in pick_top_n (and possibly in the downstream format_result
whose output inherits the bug).

Run with:
    python materials/Tracer/test_scripts/test_variants.py
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACER_DIR = HERE.parent
REPO_ROOT = TRACER_DIR.parent.parent
sys.path.insert(0, str(TRACER_DIR))

from code_parser import parse_source                         # noqa: E402
from executor import TracingExecutor                    # noqa: E402
from judge import LLMJudge                              # noqa: E402
from judge_property import PropertyAwareJudge           # noqa: E402
from judge_test_property import TestPropertyJudge       # noqa: E402
from properties import PropertyGenerator                # noqa: E402


BUGGY_SCRIPT = '''
import pandas as pd

customers = pd.DataFrame({'customer_id': [1, 2, 3, 4, 5],
                          'name': ['Alice', 'Bob', 'Carol', 'David', 'Eve']})
orders = pd.DataFrame({'order_id':     [101, 102, 103, 104, 105, 106],
                       'customer_id':  [  1,   2,   1,   3,   2,   4],
                       'total_amount': [150.0, 200.0, 300.0, 175.5, 250.0, 400.0]})


def build_totals(state):
    """Aggregate total spending per customer. Adds state['totals']."""
    state['totals'] = (state['orders']
                       .groupby('customer_id')['total_amount']
                       .sum().reset_index())
    return state


def pick_top_n(state):
    """Pick the top 3 customers by total spending, HIGHEST first."""
    # BUG: ascending=True returns the LOWEST spenders.
    state['top'] = state['totals'].sort_values('total_amount', ascending=True).head(3)
    return state


def format_result(state):
    """Return the top-3 customer names as a list."""
    merged = state['customers'].merge(state['top'], on='customer_id')
    return merged['name'].tolist()


state = {'customers': customers, 'orders': orders}
state = build_totals(state)
state = pick_top_n(state)
state = format_result(state)
print('final:', state)
'''

GOAL = ("Find the top 3 customers by total spending and return their names, "
        "highest spender first.")

MODEL = os.environ.get("TRACER_TEST_MODEL", "gpt-5.4-mini-2026-03-17")


def _load_api_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    # Fallback: the course demo keeps the key here.
    key_file = REPO_ROOT / "examples" / "tracer_toy_agent" / "api_key.json"
    if key_file.exists():
        return json.loads(key_file.read_text())["api_key"]
    raise RuntimeError(
        "No OpenAI key: set OPENAI_API_KEY or create "
        "examples/tracer_toy_agent/api_key.json with {\"api_key\": \"sk-...\"}"
    )


def run_variant(label: str, judge):
    """Run the buggy script under one judge and return the ExecutionResult."""
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    parsed = parse_source(BUGGY_SCRIPT)
    exe = TracingExecutor(parsed, judge=judge, continue_on_error=True)
    result = exe.execute()
    print(f"final state: {result.final_variables.get('state')}")
    print(f"errors reported: {len(result.errors)}")
    for err in result.errors:
        msg = err.error_message
        print(f"  L{err.lineno:>3}  {err.error_type:<12}  "
              f"{msg[:180]}{'...' if len(msg) > 180 else ''}")
    return result


def _check_result_shape(result, label: str) -> None:
    """Assert the ExecutionResult has the contract the reporter/notebook depend on."""
    assert hasattr(result, "errors"), f"{label}: ExecutionResult missing .errors"
    assert hasattr(result, "final_variables"), f"{label}: missing .final_variables"
    assert isinstance(result.errors, list), f"{label}: .errors is not a list"
    for err in result.errors:
        assert isinstance(err.lineno, int), f"{label}: err.lineno is not int"
        assert isinstance(err.error_type, str), f"{label}: err.error_type is not str"
        assert isinstance(err.error_message, str), f"{label}: err.error_message is not str"


def main() -> None:
    api_key = _load_api_key()
    print(f"Using model: {MODEL}")

    failures: List[str] = []

    def check(cond: bool, msg: str) -> None:
        """Record a failure without aborting so we collect the full picture."""
        if cond:
            print(f"  [PASS] {msg}")
        else:
            failures.append(msg)
            print(f"  [FAIL] {msg}")

    # --- VARIANT 1/3: vanilla LLMJudge --------------------------------------
    # Regression guard: the recent judge.py change (max_tokens ->
    # max_completion_tokens) is part of the same working tree as the new
    # variants. If the rename broke vanilla on the configured model, this
    # variant will silently produce zero errors on a script with an obvious
    # logic bug. The assertions below catch that.
    vanilla = LLMJudge(api_key=api_key, model=MODEL, script_goal=GOAL)
    r1 = run_variant("VARIANT 1/3: vanilla LLMJudge (regression guard)", vanilla)
    print("\n-- vanilla assertions --")
    _check_result_shape(r1, "vanilla")
    check(len(r1.errors) >= 1,
          "vanilla: >=1 error surfaced on a script with a known logic bug "
          "(guards against judge.py OpenAI-param regression)")
    check(any(err.error_type == "LogicError" for err in r1.errors),
          "vanilla: at least one error is classified as LogicError")

    # --- property derivation (shared by variants 2 and 3) -------------------
    gen = PropertyGenerator(api_key=api_key, model=MODEL)
    props = gen.generate(GOAL, BUGGY_SCRIPT)
    print(f"\n--- Derived {len(props)} properties ---")
    for p in props:
        applies = ", ".join(p.applies_to) if p.applies_to else "any step"
        print(f"  {p.id} [{applies}]: {p.description}")
        if p.assertion:
            print(f"        check: {p.assertion}")
    print("\n-- property-derivation assertions --")
    check(len(props) >= 1,
          "PropertyGenerator produced >=1 property (guards against silent JSON-parse failure)")
    check(any(p.applies_to and "pick_top_n" in p.applies_to for p in props),
          "at least one derived property targets pick_top_n (the buggy step)")
    check(any(p.assertion for p in props),
          "at least one derived property has an executable assertion")

    # --- VARIANT 2/3: PropertyAwareJudge ------------------------------------
    prop_judge = PropertyAwareJudge(api_key=api_key, properties=props,
                                    model=MODEL, script_goal=GOAL)
    r2 = run_variant("VARIANT 2/3: PropertyAwareJudge", prop_judge)
    print("\n-- PropertyAwareJudge assertions --")
    _check_result_shape(r2, "property-aware")
    check(len(r2.errors) >= 1, "PropertyAwareJudge: >=1 error surfaced on the buggy script")

    # --- VARIANT 3/3: TestPropertyJudge -------------------------------------
    test_judge = TestPropertyJudge(api_key=api_key, properties=props,
                                   model=MODEL, script_goal=GOAL)
    r3 = run_variant("VARIANT 3/3: TestPropertyJudge", test_judge)
    print("\n--- per-call test outcomes recorded by TestPropertyJudge ---")
    for name, outcomes in test_judge.history:
        fails = [o.property_id for o in outcomes if o.status == "fail"]
        passes = [o.property_id for o in outcomes if o.status == "pass"]
        print(f"  {name}: pass={passes} fail={fails}")
    print("\n-- TestPropertyJudge assertions --")
    _check_result_shape(r3, "test-property")
    check(len(r3.errors) >= 1,
          "TestPropertyJudge: >=1 error surfaced on the buggy script")
    check(len(test_judge.history) >= 1,
          "TestPropertyJudge.history is populated (audit log not silently dropped)")
    all_outcomes = [o for _, outs in test_judge.history for o in outs]
    any_fail = any(o.status == "fail" for o in all_outcomes)
    check(any_fail,
          "TestPropertyJudge observed >=1 property FAIL on the buggy script "
          "(executable-assertion path is actually evaluating and detecting violations)")
    # Bug lives in pick_top_n -- a derived property targeting it should have failed.
    pick_top_n_fails = [o for name, outs in test_judge.history if name == "pick_top_n"
                         for o in outs if o.status == "fail"]
    check(len(pick_top_n_fails) >= 1,
          "TestPropertyJudge: at least one property FAILED for pick_top_n specifically")

    # --- interface-compatibility assertions ---------------------------------
    # The executor depends only on judge.judge_function_call returning a
    # JudgmentResult-shaped value. Confirm both subclasses produced results
    # the executor accepted (if not, we wouldn't reach this point).
    print("\n-- interface compatibility --")
    check(type(r2).__name__ == type(r1).__name__,
          "PropertyAwareJudge produces same ExecutionResult type as vanilla")
    check(type(r3).__name__ == type(r1).__name__,
          "TestPropertyJudge produces same ExecutionResult type as vanilla")

    # --- summary -----------------------------------------------------------
    print(f"\n{'=' * 72}")
    if failures:
        print(f"FAILED: {len(failures)} assertion(s) did not pass:")
        for f in failures:
            print(f"  - {f}")
        raise AssertionError(f"{len(failures)} assertion(s) failed")
    print(f"OK: all assertions passed.")


if __name__ == "__main__":
    main()
