"""
Code Executor - Traces and executes Python code with function call interception.
"""

import ast
import sys
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from parser import ParsedCode, FunctionInfo
from judge import LLMJudge, JudgmentResult, Verdict


class StopReason(Enum):
    """Reasons for stopping execution."""
    COMPLETED = "completed"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    JUDGMENT_FAILED = "judgment_failed"
    USER_STOPPED = "user_stopped"


@dataclass
class FunctionCall:
    """Record of a function call."""
    name: str
    args: tuple
    kwargs: dict
    result: Any
    source: str
    judgment: Optional[JudgmentResult] = None
    error: Optional[str] = None


@dataclass
class ExecutionStep:
    """A single step in execution."""
    lineno: int
    code: str
    step_type: str  # 'statement', 'function_call', 'error'
    result: Any = None
    function_call: Optional[FunctionCall] = None
    variables_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Complete result of code execution."""
    success: bool
    stop_reason: StopReason
    steps: List[ExecutionStep]
    function_calls: List[FunctionCall]
    final_variables: Dict[str, Any]
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None


class TracingExecutor:
    """Executes Python code with tracing and LLM judgment."""

    def __init__(self, parsed_code: ParsedCode, judge: Optional[LLMJudge] = None):
        self.parsed_code = parsed_code
        self.judge = judge
        self.steps: List[ExecutionStep] = []
        self.function_calls: List[FunctionCall] = []
        self.globals: Dict[str, Any] = {}
        self.locals: Dict[str, Any] = {}
        self._function_sources: Dict[str, str] = {}
        self._function_docstrings: Dict[str, Optional[str]] = {}
        self._stopped = False
        self._stop_reason: Optional[StopReason] = None
        self._error_message: Optional[str] = None

    def execute(self) -> ExecutionResult:
        """Execute the parsed code with tracing."""
        try:
            # Set up execution environment
            self._setup_environment()

            # Execute imports first
            self._execute_imports()

            # Define all functions and classes (wrap them for tracing)
            self._define_functions()
            self._define_classes()

            # Execute main statements one by one
            self._execute_main_statements()

            if self._stopped:
                return self._create_result(False, self._stop_reason)

            return self._create_result(True, StopReason.COMPLETED)

        except SyntaxError as e:
            return self._create_result(
                False, StopReason.SYNTAX_ERROR,
                error_message=str(e),
                error_traceback=traceback.format_exc()
            )
        except Exception as e:
            return self._create_result(
                False, StopReason.RUNTIME_ERROR,
                error_message=str(e),
                error_traceback=traceback.format_exc()
            )

    def _setup_environment(self):
        """Set up the execution environment."""
        # Provide safe builtins
        self.globals = {
            '__builtins__': __builtins__,
            '__name__': '__main__',
        }
        self.locals = {}

    def _execute_imports(self):
        """Execute import statements."""
        for import_code in self.parsed_code.imports:
            try:
                exec(import_code, self.globals, self.locals)
                self.steps.append(ExecutionStep(
                    lineno=0,
                    code=import_code,
                    step_type='import',
                    variables_snapshot=self._safe_snapshot()
                ))
            except Exception as e:
                self._stop_with_error(StopReason.RUNTIME_ERROR, f"Import error: {e}")
                raise

    def _define_functions(self):
        """Define and wrap all functions for tracing."""
        for func_info in self.parsed_code.functions:
            self._function_sources[func_info.name] = func_info.source
            self._function_docstrings[func_info.name] = func_info.docstring

            # Execute the function definition
            exec(func_info.source, self.globals, self.locals)

            # Get the original function
            original_func = self.locals.get(func_info.name) or self.globals.get(func_info.name)

            # Wrap it for tracing
            if original_func and callable(original_func):
                wrapped = self._create_wrapper(func_info.name, original_func, func_info)
                self.locals[func_info.name] = wrapped
                self.globals[func_info.name] = wrapped

    def _define_classes(self):
        """Define classes in the execution environment."""
        for class_info in self.parsed_code.classes:
            exec(class_info.source, self.globals, self.locals)

            # Store method sources
            for method in class_info.methods:
                full_name = f"{class_info.name}.{method.name}"
                self._function_sources[full_name] = method.source
                self._function_docstrings[full_name] = method.docstring

    def _create_wrapper(
        self, name: str, func: Callable, func_info: FunctionInfo
    ) -> Callable:
        """Create a wrapper function that traces calls and judges output."""
        executor = self

        def wrapper(*args, **kwargs):
            if executor._stopped:
                raise RuntimeError("Execution stopped due to previous error")

            call_record = FunctionCall(
                name=name,
                args=args,
                kwargs=kwargs,
                result=None,
                source=func_info.source
            )

            try:
                result = func(*args, **kwargs)
                call_record.result = result

                # Judge the function output if we have a judge
                if executor.judge:
                    judgment = executor.judge.judge_function_call(
                        function_name=name,
                        function_source=func_info.source,
                        args=args,
                        kwargs=kwargs,
                        result=result,
                        docstring=func_info.docstring
                    )
                    call_record.judgment = judgment

                    if judgment.verdict == Verdict.INCORRECT:
                        executor._stop_with_error(
                            StopReason.JUDGMENT_FAILED,
                            f"Function '{name}' output judged incorrect: {judgment.explanation}"
                        )

                executor.function_calls.append(call_record)
                executor.steps.append(ExecutionStep(
                    lineno=func_info.lineno,
                    code=f"{name}({executor._format_args(args, kwargs)})",
                    step_type='function_call',
                    result=result,
                    function_call=call_record,
                    variables_snapshot=executor._safe_snapshot()
                ))

                return result

            except Exception as e:
                call_record.error = str(e)
                executor.function_calls.append(call_record)
                raise

        wrapper.__name__ = name
        wrapper.__doc__ = func_info.docstring
        return wrapper

    def _execute_main_statements(self):
        """Execute main (top-level) statements one by one."""
        for stmt in self.parsed_code.main_statements:
            if self._stopped:
                break

            code = ast.get_source_segment(self.parsed_code.source, stmt) or ""
            lineno = stmt.lineno

            try:
                # Compile and execute single statement
                module = ast.Module(body=[stmt], type_ignores=[])
                compiled = compile(module, '<traced>', 'exec')
                exec(compiled, self.globals, self.locals)

                self.steps.append(ExecutionStep(
                    lineno=lineno,
                    code=code,
                    step_type='statement',
                    variables_snapshot=self._safe_snapshot()
                ))

            except Exception as e:
                self.steps.append(ExecutionStep(
                    lineno=lineno,
                    code=code,
                    step_type='error',
                    result=str(e),
                    variables_snapshot=self._safe_snapshot()
                ))
                raise

    def _stop_with_error(self, reason: StopReason, message: str):
        """Stop execution with an error."""
        self._stopped = True
        self._stop_reason = reason
        self._error_message = message

    def _safe_snapshot(self) -> Dict[str, Any]:
        """Create a safe snapshot of current variables."""
        snapshot = {}
        for name, value in {**self.globals, **self.locals}.items():
            if name.startswith('_') or name == '__builtins__':
                continue
            try:
                # Try to get a safe representation
                if callable(value) and not isinstance(value, type):
                    snapshot[name] = f"<function {name}>"
                elif isinstance(value, type):
                    snapshot[name] = f"<class {value.__name__}>"
                else:
                    snapshot[name] = repr(value)[:100]
            except:
                snapshot[name] = "<unprintable>"
        return snapshot

    def _format_args(self, args: tuple, kwargs: dict) -> str:
        """Format function arguments for display."""
        parts = [repr(a)[:50] for a in args]
        parts.extend(f"{k}={repr(v)[:50]}" for k, v in kwargs.items())
        return ", ".join(parts)

    def _create_result(
        self,
        success: bool,
        stop_reason: StopReason,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None
    ) -> ExecutionResult:
        """Create the final execution result."""
        return ExecutionResult(
            success=success,
            stop_reason=self._stop_reason or stop_reason,
            steps=self.steps,
            function_calls=self.function_calls,
            final_variables=self._safe_snapshot(),
            error_message=error_message or self._error_message,
            error_traceback=error_traceback
        )
