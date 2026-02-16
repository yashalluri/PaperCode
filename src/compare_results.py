#!/usr/bin/env python3
"""Compare reproduced results against paper's claimed results."""

import argparse
import json
from pathlib import Path
from tabulate import tabulate

TOLERANCES = {
    "accuracy": {"type": "absolute", "value": 2.0},
    "top1_accuracy": {"type": "absolute", "value": 2.0},
    "f1": {"type": "absolute", "value": 2.0},
    "precision": {"type": "absolute", "value": 2.0},
    "recall": {"type": "absolute", "value": 2.0},
    "bleu": {"type": "absolute", "value": 5.0},
    "rouge": {"type": "absolute", "value": 5.0},
    "fid": {"type": "relative", "value": 15.0},
    "loss": {"type": "relative", "value": 10.0},
    "mse": {"type": "relative", "value": 10.0},
    "mae": {"type": "relative", "value": 10.0},
    "perplexity": {"type": "relative", "value": 10.0},
}
FALLBACK = {"type": "relative", "value": 10.0}


def get_tolerance(name: str) -> dict:
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key in TOLERANCES:
        return TOLERANCES[key]
    for k, v in TOLERANCES.items():
        if k in key or key in k:
            return v
    return FALLBACK


def compare_results(claimed: dict, reproduced: dict) -> dict:
    comparisons = {}
    total = within = 0

    for metric, claimed_val in claimed.items():
        if metric not in reproduced:
            comparisons[metric] = {"claimed": claimed_val, "reproduced": None, "status": "MISSING"}
            continue

        repro_val = reproduced[metric]
        tol = get_tolerance(metric)
        abs_diff = abs(repro_val - claimed_val)

        if tol["type"] == "absolute":
            ok = abs_diff <= tol["value"]
            diff_str = f"{abs_diff:.4f} (tol: ±{tol['value']})"
        else:
            rel_diff = (abs_diff / abs(claimed_val) * 100) if claimed_val != 0 else (0 if repro_val == 0 else float("inf"))
            ok = rel_diff <= tol["value"]
            diff_str = f"{rel_diff:.1f}% (tol: ±{tol['value']}%)"

        comparisons[metric] = {
            "claimed": claimed_val, "reproduced": repro_val,
            "absolute_difference": abs_diff, "difference_description": diff_str,
            "within_tolerance": ok,
        }
        total += 1
        within += ok

    if total == 0:
        verdict, emoji = "INCONCLUSIVE", "⚠️"
    elif within == total:
        verdict, emoji = "REPRODUCED", "✅"
    elif within >= total * 0.5:
        verdict, emoji = "PARTIALLY_REPRODUCED", "🟡"
    else:
        verdict, emoji = "NOT_REPRODUCED", "❌"

    return {
        "comparisons": comparisons,
        "summary": {"total_metrics": total, "within_tolerance": within,
                     "outside_tolerance": total - within, "verdict": verdict, "verdict_emoji": emoji},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("claimed", help="Path to claimed_metrics.json")
    parser.add_argument("reproduced", help="Path to reproduced_metrics.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.claimed) as f:
        claimed = json.load(f)
    with open(args.reproduced) as f:
        reproduced = json.load(f)

    result = compare_results(claimed, reproduced)

    rows = []
    for name, c in result["comparisons"].items():
        if c.get("status") == "MISSING":
            rows.append([name, c["claimed"], "N/A", "N/A", "MISSING"])
        else:
            rows.append([name, f"{c['claimed']:.4f}", f"{c['reproduced']:.4f}",
                         c["difference_description"], "✅" if c["within_tolerance"] else "❌"])

    print(tabulate(rows, headers=["Metric", "Claimed", "Reproduced", "Difference", "Status"], tablefmt="github"))
    s = result["summary"]
    print(f"\nVerdict: {s['verdict_emoji']} {s['verdict']} ({s['within_tolerance']}/{s['total_metrics']} within tolerance)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
