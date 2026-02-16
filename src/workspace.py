#!/usr/bin/env python3
"""Workspace manager — creates per-paper directory structure."""

import argparse
import json
from pathlib import Path


def create_workspace(paper_id: str, base_dir: str = "outputs") -> Path:
    root = Path(base_dir) / paper_id
    for sub in ["paper", "spec", "code", "data",
                 "results/logs", "results/checkpoints", "results/figures",
                 "report"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_id")
    parser.add_argument("--base-dir", default="outputs")
    args = parser.parse_args()
    root = create_workspace(args.paper_id, args.base_dir)
    print(json.dumps({"workspace": str(root), "paper_id": args.paper_id}))


if __name__ == "__main__":
    main()
