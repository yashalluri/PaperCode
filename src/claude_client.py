"""Shared Claude API client for Paper Replicator."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import anthropic

MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 8192


def get_client() -> anthropic.Anthropic:
    """Create an Anthropic client. Raises RuntimeError if no API key."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. "
            "Set it in .env or export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=api_key)


def truncate_paper(text: str, max_chars: int = 150_000) -> tuple[str, bool]:
    """Truncate paper text if too long, keeping beginning and end.

    Returns (text, was_truncated).
    """
    if len(text) <= max_chars:
        return text, False

    head = max_chars * 2 // 3  # ~100k chars
    tail = max_chars // 3       # ~50k chars
    return (
        text[:head]
        + "\n\n[... content truncated for length ...]\n\n"
        + text[-tail:],
        True,
    )


def parse_delimited_response(response_text: str) -> dict[str, str]:
    """Parse a response that uses ===SECTION=== delimiters.

    Returns dict mapping section names to their content.
    """
    sections = {}
    current_key = None
    current_lines = []

    for line in response_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("===") and stripped.endswith("===") and len(stripped) > 6:
            if current_key:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = stripped.strip("=").strip()
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)

    if current_key:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections


def extract_code(response_text: str) -> str:
    """Extract Python code from response, stripping markdown fences if present."""
    text = response_text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def extract_json_block(text: str) -> str:
    """Extract JSON from text, handling markdown fences and surrounding prose."""
    import re
    # Try markdown-fenced JSON first
    match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(\{.*?\})```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try finding raw JSON object
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return text.strip()
