"""
Reporter - Formats and displays execution trace results.
"""

import json
from typing import Optional, TextIO
import sys

from executor import ExecutionResult, ExecutionStep, FunctionCall, StopReason
from judge import Verdict
from parser import ParsedCode


class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'

    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'


class Reporter:
    """Formats and outputs execution trace results."""

    def __init__(self, use_colors: bool = True, output: TextIO = sys.stdout):
        self.use_colors = use_colors
        self.output = output

    def _c(self, text: str, *colors: str) -> str:
        """Apply colors to text if colors are enabled."""
        if not self.use_colors:
            return text
        return ''.join(colors) + text + Colors.RESET

    def report_parsed_code(self, parsed: ParsedCode):
        """Report the structure of parsed code."""
        self._print(self._c("\n=== Code Structure ===\n", Colors.BOLD, Colors.CYAN))

        # Functions
        if parsed.functions:
            self._print(self._c("Functions:", Colors.BOLD))
            for func in parsed.functions:
                args_str = ", ".join(func.args)
                self._print(f"  - {func.name}({args_str}) [line {func.lineno}]")
                if func.docstring:
                    doc_preview = func.docstring.split('\n')[0][:60]
                    self._print(self._c(f"      \"{doc_preview}...\"", Colors.DIM))

        # Classes
        if parsed.classes:
            self._print(self._c("\nClasses:", Colors.BOLD))
            for cls in parsed.classes:
                self._print(f"  - {cls.name} [line {cls.lineno}]")
                for method in cls.methods:
                    self._print(f"      .{method.name}()")

        # Main code
        if parsed.main_statements:
            self._print(self._c(f"\nMain code: {len(parsed.main_statements)} statement(s)", Colors.BOLD))

        self._print("")

    def report_execution_start(self):
        """Report that execution is starting."""
        self._print(self._c("=== Execution Trace ===\n", Colors.BOLD, Colors.CYAN))

    def report_step(self, step: ExecutionStep, step_num: int):
        """Report a single execution step."""
        prefix = self._c(f"[{step_num}]", Colors.DIM)
        line_info = self._c(f"L{step.lineno}", Colors.BLUE) if step.lineno else ""

        if step.step_type == 'import':
            self._print(f"{prefix} {line_info} {self._c('IMPORT', Colors.MAGENTA)}: {step.code}")

        elif step.step_type == 'statement':
            code_preview = step.code.strip()[:80]
            if len(step.code.strip()) > 80:
                code_preview += "..."
            self._print(f"{prefix} {line_info} {code_preview}")

        elif step.step_type == 'function_call':
            self._report_function_call(prefix, line_info, step.function_call)

        elif step.step_type == 'error':
            self._print(f"{prefix} {line_info} {self._c('ERROR', Colors.RED)}: {step.code}")
            self._print(f"    {self._c(str(step.result), Colors.RED)}")

    def _report_function_call(self, prefix: str, line_info: str, call: Optional[FunctionCall]):
        """Report a function call with judgment."""
        if not call:
            return

        # Function call header
        self._print(f"{prefix} {line_info} {self._c('CALL', Colors.YELLOW)}: {call.name}()")

        # Arguments
        if call.args or call.kwargs:
            args_preview = self._format_args_preview(call.args, call.kwargs)
            self._print(f"    Args: {args_preview}")

        # Result
        result_str = repr(call.result)[:100]
        self._print(f"    {self._c('Return:', Colors.GREEN)} {result_str}")

        # Judgment
        if call.judgment:
            verdict = call.judgment.verdict
            if verdict == Verdict.CORRECT:
                verdict_str = self._c("CORRECT", Colors.GREEN, Colors.BOLD)
            elif verdict == Verdict.INCORRECT:
                verdict_str = self._c("INCORRECT", Colors.RED, Colors.BOLD)
            else:
                verdict_str = self._c(verdict.value.upper(), Colors.YELLOW)

            confidence = f"{call.judgment.confidence:.0%}"
            self._print(f"    {self._c('LLM Judge:', Colors.CYAN)} {verdict_str} ({confidence})")
            self._print(f"    {self._c('Reason:', Colors.DIM)} {call.judgment.explanation}")

        if call.error:
            self._print(f"    {self._c('Error:', Colors.RED)} {call.error}")

        self._print("")

    def _format_args_preview(self, args: tuple, kwargs: dict) -> str:
        """Format arguments for preview."""
        parts = []
        for a in args:
            s = repr(a)
            parts.append(s[:30] + "..." if len(s) > 30 else s)
        for k, v in kwargs.items():
            s = repr(v)
            parts.append(f"{k}={s[:30]}..." if len(s) > 30 else f"{k}={s}")
        return ", ".join(parts) or "(none)"

    def report_result(self, result: ExecutionResult):
        """Report the final execution result."""
        self._print(self._c("\n=== Execution Result ===\n", Colors.BOLD, Colors.CYAN))

        # Status
        if result.success:
            status = self._c("SUCCESS", Colors.GREEN, Colors.BOLD)
        else:
            status = self._c("STOPPED", Colors.RED, Colors.BOLD)

        self._print(f"Status: {status}")
        self._print(f"Reason: {result.stop_reason.value}")
        self._print(f"Steps executed: {len(result.steps)}")
        self._print(f"Function calls: {len(result.function_calls)}")

        # Error details
        if result.error_message:
            self._print(f"\n{self._c('Error:', Colors.RED, Colors.BOLD)} {result.error_message}")

        if result.error_traceback:
            self._print(f"\n{self._c('Traceback:', Colors.RED)}")
            self._print(result.error_traceback)

        # Judgment summary
        if result.function_calls:
            self._print(f"\n{self._c('Function Call Summary:', Colors.BOLD)}")
            for call in result.function_calls:
                verdict_icon = "  "
                if call.judgment:
                    if call.judgment.verdict == Verdict.CORRECT:
                        verdict_icon = self._c("[OK]", Colors.GREEN)
                    elif call.judgment.verdict == Verdict.INCORRECT:
                        verdict_icon = self._c("[!!]", Colors.RED)
                    else:
                        verdict_icon = self._c("[??]", Colors.YELLOW)
                self._print(f"  {verdict_icon} {call.name}() -> {repr(call.result)[:50]}")

        # Final variables
        if result.final_variables:
            self._print(f"\n{self._c('Final Variables:', Colors.BOLD)}")
            for name, value in result.final_variables.items():
                if not callable(value) and not name.startswith('<'):
                    self._print(f"  {name} = {value}")

        self._print("")

    def report_json(self, result: ExecutionResult) -> str:
        """Generate a JSON report of the execution."""
        data = {
            "success": result.success,
            "stop_reason": result.stop_reason.value,
            "steps": [
                {
                    "lineno": s.lineno,
                    "code": s.code,
                    "type": s.step_type,
                    "result": str(s.result) if s.result else None
                }
                for s in result.steps
            ],
            "function_calls": [
                {
                    "name": c.name,
                    "args": [repr(a) for a in c.args],
                    "kwargs": {k: repr(v) for k, v in c.kwargs.items()},
                    "result": repr(c.result),
                    "judgment": {
                        "verdict": c.judgment.verdict.value,
                        "explanation": c.judgment.explanation,
                        "confidence": c.judgment.confidence
                    } if c.judgment else None,
                    "error": c.error
                }
                for c in result.function_calls
            ],
            "error_message": result.error_message,
            "final_variables": result.final_variables
        }
        return json.dumps(data, indent=2, default=str)

    def _print(self, text: str):
        """Print to output stream."""
        print(text, file=self.output)
