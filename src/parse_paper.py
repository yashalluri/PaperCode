#!/usr/bin/env python3
"""Parse PDF to structured markdown using PyMuPDF."""

import argparse
import json
import re
import sys
from pathlib import Path

import pymupdf


def parse_pdf(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)

    # First pass: compute median font size
    all_sizes = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        all_sizes.append(span["size"])

    if not all_sizes:
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        return text

    all_sizes.sort()
    median_size = all_sizes[len(all_sizes) // 2]

    # Second pass: extract structured content
    parts = []
    for page_num, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") == 1:
                parts.append(f"\n[Figure on page {page_num + 1}]\n")
                continue
            if block.get("type") != 0:
                continue

            block_text = ""
            is_heading = False
            max_size = 0

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"]
                    if not text.strip():
                        continue
                    max_size = max(max_size, span["size"])
                    if span["size"] > median_size * 1.2 or (
                        "bold" in span.get("font", "").lower()
                        and span["size"] >= median_size * 1.05
                    ):
                        is_heading = True
                    block_text += text

            block_text = block_text.strip()
            if not block_text:
                continue

            if is_heading:
                level = "#" if max_size > median_size * 1.5 else "##" if max_size > median_size * 1.25 else "###"
                parts.append(f"\n{level} {block_text}\n")
            else:
                parts.append(block_text + "\n")

        parts.append(f"\n---\n*Page {page_num + 1}*\n")

    doc.close()
    return re.sub(r"\n{4,}", "\n\n\n", "\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    markdown = parse_pdf(str(pdf_path))
    output_path = args.output or str(pdf_path.parent / "parsed.md")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(markdown)

    print(json.dumps({"output": output_path, "length": len(markdown)}))


if __name__ == "__main__":
    main()
