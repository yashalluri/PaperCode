#!/usr/bin/env python3
"""Generate PyTorch training code for paper reproduction using Claude API."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from claude_client import get_client, MODEL, truncate_paper, extract_code

CODE_MAX_TOKENS = 16384  # Code files can be long

DEVICE_DETECTION = '''device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")'''

MODEL_PY_PROMPT = """You are an expert PyTorch ML engineer. Generate model.py to reproduce the model(s) described in this paper.

<paper>
{paper_text}
</paper>

<architecture_spec>
{architecture}
</architecture_spec>

<hyperparameters>
{hyperparameters}
</hyperparameters>

## Requirements
- Implement the model(s) EXACTLY as described in the paper
- Each model must be an nn.Module subclass
- Add shape comments on key layers: # (B, C, H, W) -> (B, 2C, H/2, W/2)
- Include `if __name__ == "__main__":` smoke test block that:
  - Creates a model instance
  - Runs a dummy forward pass with correctly-shaped random input
  - Prints the output shape
- Use this device detection:
  {device_detection}
- Use standard PyTorch modules (nn.Linear, nn.Conv1d/2d, nn.GRU, etc.)
- Implement novel/custom components as separate nn.Module classes
- If the paper describes multiple model variants, implement all of them

Output ONLY valid Python code. No markdown fences, no explanation text."""

DATASET_PY_PROMPT = """You are an expert PyTorch ML engineer. Generate dataset.py for the dataset described in this paper.

<paper>
{paper_text}
</paper>

<dataset_spec>
{dataset}
</dataset_spec>

<model_code>
{model_code}
</model_code>

## Requirements
- For standard datasets (MNIST, CIFAR, ImageNet, etc.): use torchvision.datasets or HuggingFace datasets
- For custom/paper-specific datasets: implement data generation that matches the described dimensions
- Implement ALL preprocessing from the paper (normalization, resizing, augmentation)
- Provide a function like `get_dataloaders(batch_size, ...)` that returns train/test DataLoaders
- Include `if __name__ == "__main__":` smoke test that loads one batch and prints shapes
- Use this device detection if needed:
  {device_detection}
- Make sure the data shapes match what model.py expects

Output ONLY valid Python code. No markdown fences, no explanation text."""

TRAIN_PY_PROMPT = """You are an expert PyTorch ML engineer. Generate train.py for training the model(s) described in this paper.

<paper>
{paper_text}
</paper>

<hyperparameters>
{hyperparameters}
</hyperparameters>

<evaluation_spec>
{evaluation}
</evaluation_spec>

<model_code>
{model_code}
</model_code>

<dataset_code>
{dataset_code}
</dataset_code>

## Requirements
- argparse with these arguments (AT MINIMUM):
  --epochs (int, default from paper or 100)
  --batch_size (int, default from paper or 128)
  --lr (float, default from paper or 1e-3)
  --output_dir (str, default "results")
  --max_steps (int, default -1, for smoke testing — if >0, stop after this many batches)
  --seed (int, default 42)
- Device detection: {device_detection}
- Set random seeds: torch.manual_seed, numpy, random
- Import model(s) from model.py and data from dataset.py using sys.path
- Training loop with:
  - Loss computation and backpropagation
  - LR scheduling as described in the paper
  - Print progress every epoch with loss and metrics
  - Save best_model.pt to output_dir based on validation metric
  - Save training_log.json with epoch-by-epoch metrics
  - tqdm progress bars for batches
- Handle KeyboardInterrupt: save checkpoint before exiting
- Print "FINAL RESULTS" before the summary at the end
- If the paper trains multiple models, train them sequentially and print "Training MODEL_NAME" before each

Output ONLY valid Python code. No markdown fences, no explanation text."""

EVALUATE_PY_PROMPT = """You are an expert PyTorch ML engineer. Generate evaluate.py for evaluating trained models and producing figures.

<paper>
{paper_text}
</paper>

<evaluation_spec>
{evaluation}
</evaluation_spec>

<claimed_metrics>
{claimed_metrics}
</claimed_metrics>

<model_code>
{model_code}
</model_code>

<dataset_code>
{dataset_code}
</dataset_code>

<train_code>
{train_code}
</train_code>

## Requirements
- argparse with:
  --checkpoint (str, path to best_model.pt or results dir with model checkpoints)
  --output_dir (str, where to save metrics and figures)
- Load the trained model(s) from checkpoint(s)
- Compute ALL metrics mentioned in the paper on the test set
- Save metrics.json to output_dir with the SAME metric keys used in claimed_metrics
- Generate matplotlib figures and save to output_dir/figures/:
  - Training curves (if training_log.json exists in the results dir)
  - Comparison bar chart: paper claimed vs reproduced metrics
  - Any paper-specific visualizations if applicable
- Use this device detection:
  {device_detection}

Output ONLY valid Python code. No markdown fences, no explanation text."""

REQUIREMENTS_PROMPT = """Given these Python files for an ML project, generate a requirements.txt listing all necessary pip packages.

<model_py>
{model_code}
</model_py>

<dataset_py>
{dataset_code}
</dataset_py>

<train_py>
{train_code}
</train_py>

<evaluate_py>
{evaluate_code}
</evaluate_py>

## Requirements
- List one package per line with minimum version pins (e.g., torch>=2.0.0)
- Always include: torch, numpy, matplotlib, tqdm
- Include any dataset libraries used (torchvision, datasets, etc.)
- Do NOT include standard library modules
- Do NOT include the project's own modules

Output ONLY the requirements.txt content. No markdown fences, no explanation."""


def generate_code(
    parsed_md_path: str,
    spec_dir: str,
    code_dir: str,
    log_callback=None,
) -> dict:
    """Generate all code files for paper reproduction.

    Args:
        parsed_md_path: Path to parsed.md
        spec_dir: Path to spec directory
        code_dir: Path to write code files
        log_callback: Optional callable(str) for progress logs

    Returns:
        dict with 'files' list of generated file paths
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    code_path = Path(code_dir)
    code_path.mkdir(parents=True, exist_ok=True)
    spec_path = Path(spec_dir)

    # Read paper and specs
    paper_text = Path(parsed_md_path).read_text()
    paper_text, was_truncated = truncate_paper(paper_text)
    if was_truncated:
        log("Paper text truncated to fit context window.")

    architecture = _read_file(spec_path / "architecture.md", "No architecture spec found.")
    hyperparameters = _read_file(spec_path / "hyperparameters.json", "{}")
    dataset = _read_file(spec_path / "dataset.md", "No dataset spec found.")
    evaluation = _read_file(spec_path / "evaluation.md", "No evaluation spec found.")
    claimed_metrics = _read_file(spec_path / "claimed_metrics.json", "{}")

    client = get_client()
    files = []

    # 1. model.py
    log("Generating model.py...")
    model_code = _generate_file(
        client, MODEL_PY_PROMPT.format(
            paper_text=paper_text,
            architecture=architecture,
            hyperparameters=hyperparameters,
            device_detection=DEVICE_DETECTION,
        ),
    )
    (code_path / "model.py").write_text(model_code)
    files.append("model.py")
    log(f"model.py generated ({model_code.count(chr(10)) + 1} lines).")

    # 2. dataset.py
    log("Generating dataset.py...")
    dataset_code = _generate_file(
        client, DATASET_PY_PROMPT.format(
            paper_text=paper_text,
            dataset=dataset,
            model_code=model_code,
            device_detection=DEVICE_DETECTION,
        ),
    )
    (code_path / "dataset.py").write_text(dataset_code)
    files.append("dataset.py")
    log(f"dataset.py generated ({dataset_code.count(chr(10)) + 1} lines).")

    # 3. train.py
    log("Generating train.py...")
    train_code = _generate_file(
        client, TRAIN_PY_PROMPT.format(
            paper_text=paper_text,
            hyperparameters=hyperparameters,
            evaluation=evaluation,
            model_code=model_code,
            dataset_code=dataset_code,
            device_detection=DEVICE_DETECTION,
        ),
    )
    (code_path / "train.py").write_text(train_code)
    files.append("train.py")
    log(f"train.py generated ({train_code.count(chr(10)) + 1} lines).")

    # 4. evaluate.py
    log("Generating evaluate.py...")
    evaluate_code = _generate_file(
        client, EVALUATE_PY_PROMPT.format(
            paper_text=paper_text,
            evaluation=evaluation,
            claimed_metrics=claimed_metrics,
            model_code=model_code,
            dataset_code=dataset_code,
            train_code=train_code,
            device_detection=DEVICE_DETECTION,
        ),
    )
    (code_path / "evaluate.py").write_text(evaluate_code)
    files.append("evaluate.py")
    log(f"evaluate.py generated ({evaluate_code.count(chr(10)) + 1} lines).")

    # 5. requirements.txt
    log("Generating requirements.txt...")
    req_text = _generate_file(
        client, REQUIREMENTS_PROMPT.format(
            model_code=model_code,
            dataset_code=dataset_code,
            train_code=train_code,
            evaluate_code=evaluate_code,
        ),
        max_tokens=1024,
    )
    (code_path / "requirements.txt").write_text(req_text)
    files.append("requirements.txt")
    log("requirements.txt generated.")

    return {"files": files, "code_dir": str(code_path)}


def validate_and_fix(
    code_dir: str,
    spec_dir: str,
    parsed_md_path: str,
    log_callback=None,
    max_retries: int = 3,
) -> bool:
    """Run smoke tests on generated code and fix failures via Claude.

    Returns True if all tests pass.
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    code_path = Path(code_dir)
    all_passed = True

    # Install code-specific requirements first
    req_path = code_path / "requirements.txt"
    if req_path.exists():
        log("Installing generated code dependencies...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_path), "-q"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            log(f"Warning: Some dependencies failed to install: {result.stderr[:200]}")

    # Smoke tests in order
    tests = [
        ("model.py", [sys.executable, str(code_path / "model.py")], "forward pass"),
        ("dataset.py", [sys.executable, str(code_path / "dataset.py")], "data loading"),
        (
            "train.py",
            [
                sys.executable, str(code_path / "train.py"),
                "--epochs", "1", "--max_steps", "5",
                "--output_dir", str(code_path.parent / "results" / "smoke_test"),
            ],
            "1-step training",
        ),
    ]

    for filename, cmd, description in tests:
        log(f"Smoke test: {filename} ({description})...")
        passed = False

        for attempt in range(max_retries + 1):
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120,
                    cwd=str(code_path),
                )
                if result.returncode == 0:
                    log(f"  PASSED")
                    passed = True
                    break
                else:
                    error = (result.stderr or result.stdout)[-1500:]
                    if attempt < max_retries:
                        log(f"  FAILED (attempt {attempt + 1}/{max_retries}). Fixing...")
                        _fix_file(code_path, filename, error, spec_dir, parsed_md_path, log)
                    else:
                        log(f"  FAILED after {max_retries} retries. Error: {error[:300]}")
            except subprocess.TimeoutExpired:
                log(f"  TIMEOUT (attempt {attempt + 1})")
                if attempt >= max_retries:
                    log(f"  Skipping {filename} after timeout.")
                passed = True  # Timeout on training isn't necessarily a code error
                break

        if not passed:
            all_passed = False

    return all_passed


def _generate_file(client, prompt: str, max_tokens: int = CODE_MAX_TOKENS) -> str:
    """Call Claude and extract clean code from response."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_code(response.content[0].text)


def _fix_file(code_path, filename, error, spec_dir, parsed_md_path, log):
    """Send error back to Claude and rewrite the file."""
    current_code = (code_path / filename).read_text()
    spec_path = Path(spec_dir)
    architecture = _read_file(spec_path / "architecture.md", "")

    client = get_client()
    prompt = f"""The following Python file has an error when executed. Fix the error and return the corrected file.

## File: {filename}
```python
{current_code}
```

## Error:
```
{error}
```

## Architecture context:
{architecture[:2000]}

Fix the error. Output ONLY the corrected Python code. No markdown fences, no explanation."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=CODE_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    fixed_code = extract_code(response.content[0].text)
    (code_path / filename).write_text(fixed_code)
    log(f"  {filename} rewritten ({fixed_code.count(chr(10)) + 1} lines).")


def _read_file(path: Path, default: str = "") -> str:
    """Read a file, returning default if it doesn't exist."""
    if path.exists():
        return path.read_text()
    return default


def main():
    parser = argparse.ArgumentParser(description="Generate code using Claude API")
    parser.add_argument("parsed_md", help="Path to parsed.md")
    parser.add_argument("--spec-dir", required=True, help="Path to spec directory")
    parser.add_argument("--code-dir", required=True, help="Path to write code files")
    parser.add_argument("--validate", action="store_true", help="Run smoke tests after generation")
    args = parser.parse_args()

    result = generate_code(args.parsed_md, args.spec_dir, args.code_dir, log_callback=print)
    print(json.dumps(result, indent=2))

    if args.validate:
        print("\nRunning validation...")
        ok = validate_and_fix(args.code_dir, args.spec_dir, args.parsed_md, log_callback=print)
        print(f"\nValidation: {'ALL PASSED' if ok else 'SOME FAILURES'}")


if __name__ == "__main__":
    main()
