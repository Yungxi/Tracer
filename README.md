# Tracer

A Python code tracing and patching tool that uses LLM to evaluate function outputs and automatically fix bugs.

## Features

- **Code Tracing**: Execute Python code step-by-step with full visibility
- **LLM Judge**: Evaluate function outputs for correctness using OpenAI
- **Goal-Based Evaluation**: Judge outputs against the script's intended purpose
- **Auto Patching**: Generate fixes for detected bugs using LLM
- **SWE-bench Integration**: Test against real-world bugs from the SWE-bench Lite dataset

## Installation

```bash
git clone https://github.com/Yungxi/Tracer.git
cd Tracer
pip install -r requirements.txt
```

## Configuration

Set your OpenAI API key in the respective files:

```python
# In tracer.py or swe_bench.py
OPENAI_API_KEY = "sk-your-key-here"
```

Or use the `--api-key` flag or `OPENAI_API_KEY` environment variable.

## Usage

### 1. Tracer - Trace and Judge Code Execution

Trace a Python file and have the LLM judge each function's output:

```bash
# Trace a file
python tracer.py example.py

# Trace with a goal (LLM judges against this goal)
python tracer.py example.py --goal "Calculate arithmetic operations correctly"

# Show code structure before execution
python tracer.py example.py --show-structure
```

#### Using scripts.json

Store scripts with their goals in `scripts.json`:

```json
{
  "scripts": [
    {
      "name": "my_script",
      "goal": "Description of what the script should do",
      "code": "def add(a, b):\n    return a + b\n\nresult = add(2, 3)\nprint(result)"
    }
  ]
}
```

Then run:

```bash
# List available scripts
python tracer.py --list

# Run a script by name
python tracer.py --script my_script
```

### 2. SWE-bench - Test Against Real Bugs

Evaluate the patcher against the SWE-bench Lite dataset (323 real-world GitHub issues):

```bash
# List all available instances
python swe_bench.py --list

# Run on a specific instance
python swe_bench.py --instance django__django-11179

# Run on dev split (23 instances)
python swe_bench.py --run-dev

# Run on all instances
python swe_bench.py --run-all
```

#### Verify Patches with Actual Tests

Use `--verify` to clone the repo and run the actual test suite:

```bash
# Verify a single instance
python swe_bench.py --instance django__django-11179 --verify

# Verify all dev instances
python swe_bench.py --run-dev --verify

# Save results to JSON
python swe_bench.py --run-dev --verify --output results.json
```

### 3. Patcher - Generate Bug Fixes

The patcher can be used programmatically:

```python
from patcher import LLMPatcher

patcher = LLMPatcher(api_key="sk-...")

# Fix code with a known problem
result = patcher.patch_code(
    source_code="def subtract(a, b):\n    return a + b",
    problem_description="subtract returns wrong result",
    expected_behavior="Should return a - b"
)

print(result.patches[0].fixed_code)
print(result.patches[0].explanation)
```

## Project Structure

```
Tracer/
├── tracer.py       # Main CLI - trace and judge code
├── parser.py       # AST-based code parser
├── executor.py     # Step-by-step code executor
├── judge.py        # LLM judge for function outputs
├── reporter.py     # Colored terminal output
├── patcher.py      # LLM-based code fixer
├── swe_bench.py    # SWE-bench dataset integration
├── scripts.json    # Example scripts with goals
└── example.py      # Sample code to trace
```

## How It Works

### Tracing Flow

1. **Parse**: Split code into functions and main statements using AST
2. **Execute**: Run main code step-by-step
3. **Intercept**: Wrap function calls to capture inputs/outputs
4. **Judge**: Send each function call to LLM for evaluation
5. **Report**: Display results with verdict (CORRECT/INCORRECT)

### SWE-bench Flow

1. **Load**: Fetch dataset from Hugging Face
2. **Generate**: Create patch from problem description using LLM
3. **Compare**: Show similarity to ground truth patch
4. **Verify** (optional): Clone repo, apply patch, run tests

## Options

### tracer.py

| Flag | Description |
|------|-------------|
| `--script, -S` | Run script from scripts.json by name |
| `--goal, -g` | Goal/purpose for LLM to judge against |
| `--list, -l` | List available scripts |
| `--show-structure, -s` | Show parsed code structure |
| `--model, -m` | OpenAI model (default: gpt-4o-mini) |
| `--json, -j` | Output results as JSON |
| `--no-color` | Disable colored output |

### swe_bench.py

| Flag | Description |
|------|-------------|
| `--instance, -i` | Run specific instance by ID |
| `--run-dev` | Run on dev split (23 instances) |
| `--run-all` | Run on all instances (323) |
| `--verify, -V` | Clone repos and run actual tests |
| `--output, -o` | Save results to JSON file |
| `--model, -m` | OpenAI model (default: gpt-4o-mini) |

## Example Output

```
============================================================
Script: buggy_calculator
Goal: A calculator that correctly performs addition, subtraction, and multiplication
============================================================

=== Execution Trace ===

[1] L1 IMPORT: (none)
[2] L10 CALL: add()
    Args: 10, 3
    Return: 13
    LLM Judge: CORRECT (95%)
    Reason: Addition of 10 + 3 = 13 is correct.

[3] L14 CALL: subtract()
    Args: 10, 3
    Return: 13
    LLM Judge: INCORRECT (98%)
    Reason: subtract(10, 3) should return 7, not 13. The function has a bug.

=== Execution Result ===

Status: STOPPED
Reason: judgment_failed
```

## License

MIT
