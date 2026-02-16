#!/usr/bin/env python3
"""Fetch paper from arXiv — downloads PDF, attempts LaTeX source, saves metadata."""

import argparse
import json
import re
from pathlib import Path

import arxiv
import requests


def extract_paper_id(url_or_id: str) -> str:
    patterns = [
        r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",
        r"arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)",
        r"^(\d{4}\.\d{4,5}(?:v\d+)?)$",
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    raise ValueError(f"Could not extract arXiv ID from: {url_or_id}")


def fetch_paper(paper_id: str, output_dir: str) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client(page_size=1, delay_seconds=3.0)
    search = arxiv.Search(id_list=[paper_id])
    results = list(client.results(search))
    if not results:
        raise ValueError(f"Paper not found: {paper_id}")

    paper = results[0]

    metadata = {
        "paper_id": paper_id,
        "title": paper.title,
        "authors": [str(a) for a in paper.authors],
        "abstract": paper.summary,
        "categories": paper.categories,
        "published": paper.published.isoformat() if paper.published else None,
        "pdf_url": paper.pdf_url,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    paper.download_pdf(dirpath=str(output_path), filename="paper.pdf")

    result = {
        "paper_id": paper_id,
        "title": paper.title,
        "pdf_path": str(output_path / "paper.pdf"),
        "metadata_path": str(output_path / "metadata.json"),
    }

    try:
        resp = requests.get(f"https://arxiv.org/e-print/{paper_id}", timeout=30)
        if resp.status_code == 200:
            source_path = output_path / "source.tar.gz"
            with open(source_path, "wb") as f:
                f.write(resp.content)
            result["source_path"] = str(source_path)
    except Exception:
        pass

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_id", help="arXiv URL or paper ID")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    paper_id = extract_paper_id(args.paper_id)
    output_dir = args.output or f"outputs/{paper_id}/paper"
    result = fetch_paper(paper_id, output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
