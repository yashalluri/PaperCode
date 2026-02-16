#!/usr/bin/env python3
"""Generate a reproducibility report from workspace artifacts."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def generate_report(workspace_dir: str) -> str:
    ws = Path(workspace_dir)
    metadata = load_json(ws / "paper" / "metadata.json")
    comparison = load_json(ws / "results" / "comparison.json")
    metrics = load_json(ws / "results" / "metrics.json")
    training_log = load_json(ws / "results" / "training_log.json")

    lines = []
    lines.append("# Reproducibility Report")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Paper Summary
    lines.append("## Paper Summary\n")
    if metadata:
        lines.append(f"**Title:** {metadata.get('title', 'Unknown')}\n")
        authors = metadata.get("authors", [])
        if authors:
            lines.append(f"**Authors:** {', '.join(authors[:5])}")
            if len(authors) > 5:
                lines.append(f" et al. ({len(authors)} total)")
            lines.append("\n")
        lines.append(f"**arXiv ID:** {metadata.get('paper_id', 'Unknown')}\n")
        if metadata.get("abstract"):
            lines.append(f"**Abstract:** {metadata['abstract'][:500]}...\n")

    # Verdict
    lines.append("## Verdict\n")
    if comparison and "summary" in comparison:
        s = comparison["summary"]
        lines.append(f"### {s['verdict_emoji']} {s['verdict']}\n")
        lines.append(f"- **Metrics within tolerance:** {s['within_tolerance']}/{s['total_metrics']}")
        lines.append(f"- **Metrics outside tolerance:** {s['outside_tolerance']}/{s['total_metrics']}\n")
    else:
        lines.append("### ⚠️ INCONCLUSIVE\n*Comparison data not available.*\n")

    # Results Comparison
    lines.append("## Results Comparison\n")
    if comparison and "comparisons" in comparison:
        lines.append("| Metric | Claimed | Reproduced | Difference | Status |")
        lines.append("|--------|---------|------------|------------|--------|")
        for name, c in comparison["comparisons"].items():
            if c.get("status") == "MISSING":
                lines.append(f"| {name} | {c['claimed']} | N/A | N/A | MISSING |")
            else:
                status = "✅" if c["within_tolerance"] else "❌"
                lines.append(f"| {name} | {c['claimed']:.4f} | {c['reproduced']:.4f} | {c['difference_description']} | {status} |")
        lines.append("")

    # Training Summary
    lines.append("## Training Summary\n")
    if training_log and isinstance(training_log, dict):
        for key, value in training_log.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")
    elif metrics:
        lines.append(f"```json\n{json.dumps(metrics, indent=2)}\n```\n")

    # Figures
    figures_dir = ws / "results" / "figures"
    if figures_dir.exists():
        figures = sorted(list(figures_dir.glob("*.png")) + list(figures_dir.glob("*.jpg")))
        if figures:
            lines.append("## Reproduced Figures\n")
            for fig in figures:
                lines.append(f"### {fig.stem}\n![{fig.stem}](../results/figures/{fig.name})\n")

    # Code
    code_dir = ws / "code"
    if code_dir.exists():
        code_files = sorted(code_dir.glob("*.py"))
        if code_files:
            lines.append("## Generated Code Files\n")
            for f in code_files:
                lines.append(f"- `{f.name}`")
            lines.append("")

    lines.append("## Limitations\n")
    lines.append("- Results may vary due to hardware, random seeds, and library versions.")
    lines.append("- If training was scaled down, the verdict reflects partial training only.")
    lines.append("- Synthetic data was used if the original dataset was unavailable.\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ws = Path(args.workspace)
    if not ws.exists():
        print(f"Error: Workspace not found: {ws}", file=sys.stderr)
        sys.exit(1)

    report = generate_report(args.workspace)
    output_path = args.output or str(ws / "report" / "reproducibility_report.md")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()
