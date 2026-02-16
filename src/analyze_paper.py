#!/usr/bin/env python3
"""Analyze a parsed paper to extract specs using Claude API."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from claude_client import (
    get_client, MODEL, MAX_TOKENS,
    truncate_paper, parse_delimited_response, extract_json_block,
)

ARCH_HYPER_PROMPT = """You are a machine learning research paper analyst. Given the following research paper, extract the model architecture details and training hyperparameters.

<paper>
{paper_text}
</paper>

## Task 1: Architecture
Extract in markdown format:
- Model type (CNN, Transformer, GAN, RNN, etc.)
- Layer-by-layer architecture with dimensions where specified
- Activation functions, normalization layers, dropout rates
- Novel components or modifications to standard architectures
- Input/output shapes and data flow

## Task 2: Hyperparameters
Extract as a JSON object:
- optimizer: type, learning_rate, momentum, weight_decay, betas (for Adam)
- lr_schedule: type (cosine, step, warmup, etc.), parameters
- batch_size: integer
- epochs: integer
- loss_function: name and any parameters
- regularization: dropout, weight_decay, data augmentation
- seed: random seed (use 42 if not specified)

If a value is not mentioned in the paper, use reasonable defaults and note it.

Respond with EXACTLY this format (no text before or after):
===ARCHITECTURE===
[markdown content]
===HYPERPARAMETERS===
[valid JSON object]"""

DATASET_EVAL_PROMPT = """You are a machine learning research paper analyst. Given the following research paper, extract the dataset and evaluation details.

<paper>
{paper_text}
</paper>

## Task 1: Dataset
Extract in markdown format:
- Dataset name and source
- How to download/obtain it (torchvision, HuggingFace, custom URL, etc.)
- Preprocessing steps (normalization, resizing, tokenization)
- Data augmentation techniques
- Train/validation/test split sizes
- Input dimensions and data types

## Task 2: Evaluation
Extract in markdown format:
- All evaluation metrics used (accuracy, loss, F1, FID, BLEU, etc.)
- How metrics are computed (per-class, macro, micro, etc.)
- Evaluation protocol (k-fold, single split, etc.)
- Any special evaluation procedures

## Task 3: Claimed Metrics
Extract as a JSON object mapping metric names to their numeric values.
- Use descriptive keys like "model_name_metric_name" (e.g., "cnn_test_accuracy": 94.0)
- Only include concrete numeric values from the paper's results
- Include ALL reported numeric results from tables and text
- Values should be numbers, not strings

Respond with EXACTLY this format (no text before or after):
===DATASET===
[markdown content]
===EVALUATION===
[markdown content]
===CLAIMED_METRICS===
[valid JSON object]"""

PLAN_PROMPT = """Given these extracted specs from a research paper, write a brief reproduction plan.

## Architecture
{architecture}

## Hyperparameters
{hyperparameters}

## Dataset
{dataset}

## Evaluation
{evaluation}

## Claimed Metrics
{claimed_metrics}

Write a concise reproduction plan (under 500 words) covering:
1. What models to implement and key architectural decisions
2. What dataset to use and how to obtain it
3. Training strategy (epochs, batch size, LR schedule)
4. What metrics to track and target values
5. Potential challenges or necessary deviations (e.g., dataset unavailability, compute constraints)"""


def analyze_paper(
    parsed_md_path: str,
    spec_dir: str,
    log_callback=None,
) -> dict:
    """Analyze paper and produce spec files.

    Args:
        parsed_md_path: Path to parsed.md
        spec_dir: Directory to write spec files
        log_callback: Optional callable(str) for progress logs

    Returns:
        dict with 'files' list of generated file paths
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    spec_path = Path(spec_dir)
    spec_path.mkdir(parents=True, exist_ok=True)

    # Read paper
    paper_text = Path(parsed_md_path).read_text()
    paper_text, was_truncated = truncate_paper(paper_text)
    if was_truncated:
        log("Paper text truncated to fit context window.")

    client = get_client()
    files = []

    # Call 1: Architecture + Hyperparameters
    log("Extracting architecture and hyperparameters...")
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": ARCH_HYPER_PROMPT.format(paper_text=paper_text),
        }],
    )
    sections = parse_delimited_response(response.content[0].text)

    arch_text = sections.get("ARCHITECTURE", "No architecture details extracted.")
    (spec_path / "architecture.md").write_text(arch_text)
    files.append("architecture.md")
    log(f"Architecture extracted ({len(arch_text)} chars).")

    hyper_text = sections.get("HYPERPARAMETERS", "{}")
    try:
        hyper_json = json.loads(extract_json_block(hyper_text))
    except json.JSONDecodeError:
        log("Warning: Could not parse hyperparameters as JSON, saving raw text.")
        hyper_json = {"raw": hyper_text}
    (spec_path / "hyperparameters.json").write_text(json.dumps(hyper_json, indent=2))
    files.append("hyperparameters.json")
    log("Hyperparameters extracted.")

    # Call 2: Dataset + Evaluation + Claimed Metrics
    log("Extracting dataset, evaluation, and claimed metrics...")
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": DATASET_EVAL_PROMPT.format(paper_text=paper_text),
        }],
    )
    sections = parse_delimited_response(response.content[0].text)

    dataset_text = sections.get("DATASET", "No dataset details extracted.")
    (spec_path / "dataset.md").write_text(dataset_text)
    files.append("dataset.md")
    log("Dataset info extracted.")

    eval_text = sections.get("EVALUATION", "No evaluation details extracted.")
    (spec_path / "evaluation.md").write_text(eval_text)
    files.append("evaluation.md")
    log("Evaluation protocol extracted.")

    metrics_text = sections.get("CLAIMED_METRICS", "{}")
    try:
        metrics_json = json.loads(extract_json_block(metrics_text))
    except json.JSONDecodeError:
        log("Warning: Could not parse claimed metrics as JSON, saving raw text.")
        metrics_json = {"raw": metrics_text}
    (spec_path / "claimed_metrics.json").write_text(json.dumps(metrics_json, indent=2))
    files.append("claimed_metrics.json")
    log(f"Found {len(metrics_json)} claimed metrics.")

    # Call 3: Reproduction Plan
    log("Generating reproduction plan...")
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": PLAN_PROMPT.format(
                architecture=arch_text,
                hyperparameters=json.dumps(hyper_json, indent=2),
                dataset=dataset_text,
                evaluation=eval_text,
                claimed_metrics=json.dumps(metrics_json, indent=2),
            ),
        }],
    )
    plan_text = response.content[0].text
    (spec_path / "plan.md").write_text(plan_text)
    files.append("plan.md")
    log("Reproduction plan generated.")

    return {"files": files, "spec_dir": str(spec_path)}


def main():
    parser = argparse.ArgumentParser(description="Analyze a parsed paper using Claude API")
    parser.add_argument("parsed_md", help="Path to parsed.md")
    parser.add_argument("--spec-dir", required=True, help="Directory to write spec files")
    args = parser.parse_args()

    result = analyze_paper(args.parsed_md, args.spec_dir, log_callback=print)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
