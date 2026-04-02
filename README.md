# Tracer

A Python code tracing tool that executes code step-by-step and uses an LLM to evaluate function outputs for correctness.

Tracer can detect 3 types of errors:

1. Syntax Errors - Errors made in code syntax. Tracer will stop and report the first syntax error upon parsing the code and terminate the program. Note that syntax errors further along will not be detected in one pass.

2. Runtime Errors - Errors caught during program execution (E.g. 0 division, KeyError, TypeError). Tracer will find all runtime errors in one pass assuming there are no syntax errors.

3. Logical Errors - Errors in the design and function of the program. Tracer uses an LLM judge to evaluate whether the code given actually achieves the pugrpose of the program as specified by the user. Tracer will find all logical errors in one pass assuming there are no syntax errors.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        HOW TRACER WORKS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Code String ──► parser.py ──► executor.py ◄──► judge.py      │
│                                      │                          │
│                                      ▼                          │
│                                 reporter.py                     │
│                                      │                          │
│                                      ▼                          │
│                              Execution Report                   │
│                         (errors, judgments, trace)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Core Modules

| Module | Purpose |
|--------|---------|
| **parser.py** | Parses Python code into AST, extracts functions, classes, and statements |
| **executor.py** | Executes code step-by-step, wraps functions to intercept calls, collects errors |
| **judge.py** | LLM-based judge that evaluates function outputs for correctness |
| **reporter.py** | Formats and displays execution trace with colors |
| **patcher.py** | LLM-based code patching for fixing bugs (optional) |

### Entry Points

| File | Purpose |
|------|---------|
| **tracer.py** | CLI wrapper - runs tracer on `.py` files from command line |
| **Direct imports** | Import `parser`, `executor`, `judge`, `reporter` directly in your code |

**Important:** `tracer.py` is just a CLI convenience wrapper. The actual tracing logic lives in the core modules above. You can use them directly:

```python
from parser import parse_source
from executor import TracingExecutor
from judge import LLMJudge
from reporter import Reporter

# Parse code
parsed = parse_source(code_string)

# Create judge with goal
judge = LLMJudge(api_key="sk-...", script_goal="Calculate totals correctly")

# Execute with tracing (continue_on_error=True finds ALL bugs)
executor = TracingExecutor(parsed, judge=judge, continue_on_error=True)
result = executor.execute()

# Report results
reporter = Reporter(use_colors=True)
reporter.report_result(result)

# Access errors found
for err in result.errors:
    print(f"Line {err.lineno}: {err.error_type} - {err.error_message}")
```

## Project Structure

```
Tracer/
├── parser.py           # Core: AST-based code parser
├── executor.py         # Core: Step-by-step executor with tracing
├── judge.py            # Core: LLM judge for function outputs
├── reporter.py         # Core: Colored terminal output
├── patcher.py          # Core: LLM-based code fixer
├── tracer.py           # CLI: Command-line wrapper for core modules
│
├── config.json         # Your API key and settings (git-ignored)
├── config.example.json # Template for config.json
├── requirements.txt
├── README.md
│
└── test_scripts/       # Demos, benchmarks, and test data
    ├── demo_tracer.py      # Demo: Tracer on custom buggy code
    ├── demo_dsdbench.py    # Demo: Tracer on DSDBench instances
    ├── dsd_bench.py        # Benchmark: DSDBench evaluation
    ├── swe_bench.py        # Benchmark: SWE-bench evaluation
    ├── swe_bench_PATCH_ONLY.py  # Benchmark: SWE-bench baseline
    ├── dsd_bench_data/     # DSDBench dataset files
    ├── example.py
    └── scripts.json
```

## Installation

```bash
git clone https://github.com/Yungxi/Tracer.git
cd Tracer
pip install -r requirements.txt
```

## Configuration

Set your OpenAI API key:

```bash
cp config.example.json config.json
# Edit config.json and add your API key
```

```json
{
    "openai_api_key": "sk-your-api-key-here",
    "default_model": "gpt-4o-mini"
}
```

Or use environment variable:
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

## Quick Start

### Run the Demos

```bash
# Demo 1: Tracer finding bugs in pandas code (3 intentional bugs)
python test_scripts/demo_tracer.py

# Demo 2: Tracer on real DSDBench benchmark instances
python test_scripts/demo_dsdbench.py --instance 1
python test_scripts/demo_dsdbench.py --list  # See available instances
```

### Use the CLI

```bash
# Trace a Python file
python tracer.py example.py

# Trace with a goal (LLM judges against this)
python tracer.py example.py --goal "Calculate arithmetic operations correctly"
```

## Features

### 1. Step-by-Step Execution Tracing

Tracer executes code statement-by-statement, recording:
- Each line executed
- Function calls with arguments and return values
- Variable states at each step
- Errors encountered

### 2. LLM Judge

For each function call, the LLM evaluates:
- Is the output correct given the function's purpose?
- Does it match the script's overall goal?

Verdicts: `CORRECT`, `INCORRECT`, `UNSURE`

### 3. Multi-Error Detection

With `continue_on_error=True`, Tracer keeps executing after errors to find ALL bugs:

```python
executor = TracingExecutor(parsed, judge=judge, continue_on_error=True)
result = executor.execute()

# result.errors contains ALL errors found:
# - Runtime errors (KeyError, TypeError, etc.)
# - Logic errors (detected by LLM Judge)
```

### 4. Benchmark Integration

- **DSDBench**: 1,117 annotated data science debugging examples
- **SWE-bench**: 323 real-world GitHub issues

## Example Output

```
============================================================
QUESTION: Find the top 3 customers who spent the most money
============================================================

[1] Parsing code...
Functions: get_top_customers, calculate_average, format_output
Main code: 11 statements

[2] LLM Judge goal: Find top 3 customers by spending

[3] Executing with Tracer (continue_on_error=True)...

=== Execution Trace ===
[1]  IMPORT: import pandas as pd
[2]  L5 customers_data = {...}
[3]  L10 orders_data = {...}
...
[6]  L20 CALL: get_top_customers()
     Return: DataFrame with Carol, David, Alice
     LLM Judge: INCORRECT (90%)
     Reason: Returns LOWEST spenders due to ascending=True

[10] L30 CALL: calculate_average()
     Return: {1: 112.5, 3: 43.875, 4: 100.0}
     LLM Judge: INCORRECT (90%)
     Reason: Divides by total count instead of per-customer

[13] L38 ERROR: format_output()
     KeyError: 'customer_name'

=== TRACER'S FINDINGS ===

Tracer found 3 error(s):

  [1] Line 20: LogicError
      LLM Judge: ascending=True returns lowest spenders

  [2] Line 30: LogicError
      LLM Judge: Wrong average calculation

  [3] Line 38: KeyError
      'customer_name' column doesn't exist
```

## Benchmark Usage

### DSDBench

```bash
# List dataset statistics
python test_scripts/dsd_bench.py --list

# Run on specific instance
python test_scripts/dsd_bench.py --instance 1

# Run full evaluation
python test_scripts/dsd_bench.py --run-single
```

### SWE-bench

```bash
# List instances
python test_scripts/swe_bench.py --list

# Run on specific instance
python test_scripts/swe_bench.py --instance django__django-11179

# Verify with actual tests
python test_scripts/swe_bench.py --instance django__django-11179 --verify
```

## API Reference

### TracingExecutor

```python
executor = TracingExecutor(
    parsed_code,           # From parser.parse_source()
    judge=None,            # Optional LLMJudge instance
    continue_on_error=False  # Set True to find ALL errors
)

result = executor.execute()

# result.success: bool
# result.stop_reason: StopReason enum
# result.steps: List[ExecutionStep]
# result.errors: List[ErrorInfo]  # All errors found
# result.function_calls: List[FunctionCall]
```

### LLMJudge

```python
judge = LLMJudge(
    api_key="sk-...",
    model="gpt-4o-mini",
    script_goal="Description of what the code should do"
)

# Judge is called automatically by TracingExecutor for each function call
```

### ErrorInfo

```python
@dataclass
class ErrorInfo:
    lineno: int           # Line number
    code: str             # Code that caused error
    error_type: str       # "KeyError", "LogicError", etc.
    error_message: str    # Error description
    traceback: str        # Full traceback (for runtime errors)
```

## License

MIT
