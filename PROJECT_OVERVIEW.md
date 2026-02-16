# Paper Replicator — Complete Project Overview

*An autonomous system that reproduces ML research papers from arXiv.*

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Tech Stack](#tech-stack)
5. [Pipeline Phases](#pipeline-phases)
6. [Backend — Python](#backend--python)
7. [Frontend — Next.js](#frontend--nextjs)
8. [Claude Code Agent](#claude-code-agent)
9. [How to Run](#how-to-run)
10. [Source Code Reference](#source-code-reference)

---

## What It Does

Paper Replicator takes an arXiv URL as input and autonomously:

1. Downloads and parses the research paper
2. Extracts architecture, hyperparameters, dataset, and evaluation specs
3. Generates PyTorch training code (model, dataset, training loop, evaluation)
4. Validates the generated code with dry-run smoke tests
5. Trains the model(s)
6. Evaluates results and compares against the paper's claimed metrics
7. Produces a reproducibility verdict and report

The system has two modes of operation:
- **Claude Code Agent mode** — the full autonomous pipeline driven by CLAUDE.md instructions
- **Web UI mode** — a FastAPI backend + Next.js frontend for interactive use with SSE-streamed progress

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    User / Browser                     │
│          (Next.js 16 + React 19 + Tailwind 4)        │
│    URL input → SSE progress → results table/figures   │
└─────────────────────┬────────────────────────────────┘
                      │ HTTP / SSE
                      ▼
┌──────────────────────────────────────────────────────┐
│                 FastAPI Backend (api.py)              │
│  POST /api/replicate → spawns async pipeline         │
│  GET  /api/jobs/{id}/stream → SSE event stream       │
│  GET  /api/jobs/{id}/figures/{name} → serve PNGs     │
└─────────────────────┬────────────────────────────────┘
                      │ subprocess calls
                      ▼
┌──────────────────────────────────────────────────────┐
│              Pipeline Scripts (src/)                  │
│  workspace.py → fetch_paper.py → parse_paper.py      │
│  → [Claude Code generates code] → train → evaluate   │
│  → compare_results.py → generate_report.py           │
└──────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────┐
│             Output Workspace (outputs/{paper_id}/)   │
│  paper/ spec/ code/ results/ report/                 │
└──────────────────────────────────────────────────────┘
```

---

## Project Structure

```
paperreplicator/
├── CLAUDE.md                    # Agent instructions for the full pipeline
├── requirements.txt             # Python dependencies (backend + pipeline)
├── api.py                       # FastAPI backend with SSE streaming
├── src/
│   ├── __init__.py              # Empty package init
│   ├── workspace.py             # Creates per-paper directory structure
│   ├── fetch_paper.py           # Downloads PDF + metadata from arXiv
│   ├── parse_paper.py           # Converts PDF → structured markdown (PyMuPDF)
│   ├── compare_results.py       # Compares claimed vs reproduced metrics
│   └── generate_report.py       # Generates reproducibility report markdown
├── frontend/
│   ├── package.json             # Next.js 16, React 19, Tailwind 4
│   ├── next.config.ts           # API proxy rewrite → localhost:8000
│   ├── tsconfig.json            # TypeScript config
│   ├── postcss.config.mjs       # Tailwind PostCSS plugin
│   ├── next-env.d.ts            # Next.js type references
│   └── app/
│       ├── globals.css          # Tailwind import
│       ├── layout.tsx           # Root layout (dark theme)
│       └── page.tsx             # Main UI (input, progress, logs, results)
├── outputs/                     # Generated workspaces (one per paper)
│   └── {paper_id}/
│       ├── paper/               # PDF, metadata, parsed markdown
│       ├── spec/                # Architecture, hyperparams, dataset, evaluation specs
│       ├── code/                # Generated model.py, dataset.py, train.py, evaluate.py
│       ├── results/             # Training logs, metrics, checkpoints, figures
│       └── report/              # Final reproducibility report
└── templates/                   # (Empty, reserved for future use)
```

---

## Tech Stack

### Backend
| Component | Technology |
|-----------|------------|
| API server | FastAPI + uvicorn |
| Real-time streaming | SSE via sse-starlette |
| Paper fetching | arxiv Python library + requests |
| PDF parsing | PyMuPDF (pymupdf) |
| ML framework | PyTorch + torchvision |
| Data/viz | numpy, pandas, matplotlib, scikit-learn, Pillow |
| Utilities | tqdm, tabulate, psutil |

### Frontend
| Component | Technology |
|-----------|------------|
| Framework | Next.js 16 (App Router, Turbopack) |
| UI library | React 19 |
| Styling | Tailwind CSS 4 |
| Language | TypeScript 5.9 |
| API proxy | Next.js rewrites → localhost:8000 |

### AI Agent
| Component | Technology |
|-----------|------------|
| Agent | Claude Code (driven by CLAUDE.md) |
| Sub-agents | Parallel Task agents for spec extraction |
| Code generation | Claude writes PyTorch code from paper specs |

---

## Pipeline Phases

| # | Phase | Script/Tool | Description |
|---|-------|-------------|-------------|
| 1 | Workspace | `src/workspace.py` | Creates `outputs/{paper_id}/` directory tree |
| 2 | Fetch | `src/fetch_paper.py` | Downloads PDF, metadata, and LaTeX source from arXiv |
| 3 | Parse | `src/parse_paper.py` | Converts PDF to structured markdown with heading detection |
| 4 | Analyze | Claude Code agents | Extracts architecture, hyperparameters, dataset, evaluation specs |
| 5 | Generate | Claude Code | Writes model.py, dataset.py, train.py, evaluate.py, requirements.txt |
| 6 | Validate | Bash agent | Smoke tests: forward pass, data loading, 1-step training |
| 7 | Train | subprocess | Full training run with checkpointing and logging |
| 8 | Evaluate | `evaluate.py` | Computes metrics, generates figures |
| 9 | Compare | `src/compare_results.py` | Compares against paper's claimed metrics with tolerances |
| 10 | Report | `src/generate_report.py` | Generates reproducibility_report.md with verdict |

### Verdict Categories
| Verdict | Criteria |
|---------|----------|
| REPRODUCED | All primary metrics within tolerance |
| PARTIALLY REPRODUCED | >50% of metrics within tolerance |
| NOT REPRODUCED | <50% of metrics within tolerance |
| INCONCLUSIVE | Insufficient training to determine |

### Metric Tolerances
| Metric | Tolerance Type | Value |
|--------|---------------|-------|
| accuracy, f1, precision, recall | Absolute | ±2.0 |
| bleu, rouge | Absolute | ±5.0 |
| fid | Relative | ±15% |
| loss, mse, mae, perplexity | Relative | ±10% |
| Default (unrecognized) | Relative | ±10% |

---

## Backend — Python

### `api.py` (FastAPI server)

The backend exposes these endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/replicate` | Starts a replication job. Accepts `{ "arxiv_url": "..." }`. Returns `{ "job_id": "..." }` |
| GET | `/api/jobs/{job_id}` | Returns full job state (status, phase, progress, logs, results) |
| GET | `/api/jobs/{job_id}/stream` | SSE stream of real-time updates (log lines, phase changes, completion) |
| GET | `/api/jobs/{job_id}/figures/{filename}` | Serves generated PNG figure files |
| GET | `/api/health` | Health check |

**Job lifecycle:**
1. `POST /api/replicate` creates a job and spawns `run_pipeline()` as an async task
2. The pipeline calls each `src/` script as a subprocess, streaming output
3. Training runs as an async subprocess with line-by-line log capture
4. SSE events are emitted every 300ms with new log lines and phase updates
5. On completion, final metrics/comparison/report are loaded and sent as the `complete` event

**Key design choices:**
- In-memory job store (dict) — no database needed for single-user use
- CORS enabled for all origins (development mode)
- 600-second timeout on subprocess calls
- Paper ID extraction supports `arxiv.org/abs/`, `arxiv.org/pdf/`, and bare IDs

### `src/workspace.py`

Creates the standard directory tree for a paper replication:
```
outputs/{paper_id}/
├── paper/    ├── spec/    ├── code/    ├── data/
├── results/logs/    ├── results/checkpoints/    ├── results/figures/
└── report/
```

### `src/fetch_paper.py`

1. Extracts paper ID from URL via regex
2. Uses the `arxiv` Python library to search and download
3. Saves `metadata.json` (title, authors, abstract, categories, dates)
4. Downloads `paper.pdf`
5. Attempts to download LaTeX source as `source.tar.gz` (best-effort)

### `src/parse_paper.py`

1. Opens PDF with PyMuPDF
2. First pass: computes median font size across all text spans
3. Second pass: classifies blocks as headings (larger/bold fonts) or body text
4. Produces markdown with `#`/`##`/`###` headings and page separators
5. Collapses excessive newlines

### `src/compare_results.py`

1. Loads claimed metrics (from paper) and reproduced metrics
2. For each metric, looks up tolerance (absolute or relative)
3. Determines if each metric is within tolerance
4. Computes overall verdict based on fraction of passing metrics
5. Outputs comparison table (tabulate) and JSON

### `src/generate_report.py`

Assembles a markdown report from workspace artifacts:
- Paper metadata (title, authors, abstract)
- Verdict banner with emoji
- Results comparison table
- Training summary
- Links to reproduced figures
- List of generated code files
- Limitations disclaimer

---

## Frontend — Next.js

### `app/page.tsx` (Main UI)

A single-page application with these sections:

1. **Header** — Gradient title "Paper Replicator" with tagline
2. **Input form** — Text input for arXiv URL + "Replicate" button
3. **Pipeline progress** — Shows current phase with icons, progress bar, phase indicator chips
4. **Live logs** — Scrolling terminal-style log viewer with color-coded lines
5. **Error display** — Red banner for pipeline failures
6. **Results panel** (on completion):
   - Verdict banner (green/yellow/red based on reproduction status)
   - Metrics comparison table (Paper vs Reproduced with tolerance status)
   - Reproduced figures (served from API)
   - Full report text

**Real-time updates via SSE:**
- Connects to `/api/jobs/{id}/stream` on job start
- Receives `update` events (new logs, phase, progress) and `complete` event (final results)
- Auto-scrolls log panel

**Phase display:**
12 named phases from "Starting" through "Complete", each with an icon and label. Phase chips show past (green check), current (purple highlight), and future (gray) states.

### `app/layout.tsx`

Root layout with:
- Dark theme (`className="dark"`)
- Background color `#0a0a0f` with gray text
- Page title "Paper Replicator"

### `next.config.ts`

Proxies all `/api/*` requests to `http://localhost:8000/api/*` so the frontend and backend can run on different ports during development.

---

## Claude Code Agent

The `CLAUDE.md` file configures Claude Code as an autonomous paper replication agent. Key behaviors:

**Trigger:** Any arXiv URL or paper ID in the user's message starts the pipeline immediately.

**Device detection:** Generated code always includes CUDA → MPS → CPU fallback.

**Agent strategy:**
- Main context handles: paper reading, spec synthesis, code generation, debugging, verdicts
- Sub-agents handle: parallel spec extraction (3 agents), validation dry-runs, long training runs, evaluation

**Error recovery:**
| Error | Automatic Fix |
|-------|---------------|
| CUDA OOM | Halve batch size, add gradient accumulation |
| NaN loss | Reduce LR by 10x, add gradient clipping |
| Shape mismatch | Re-read paper, fix dimensions |
| Import error | pip install the missing package |
| Training plateau | Check LR schedule, verify data shuffling |

Max 5 retries per file, max 3 full training restarts.

---

## How to Run

### Backend

```bash
# Install Python dependencies
pip install -r requirements.txt
pip install fastapi uvicorn sse-starlette

# Start the API server
python api.py
# → Runs on http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → Runs on http://localhost:3000 (proxies API to :8000)
```

### Claude Code Agent (CLI)

```bash
# From the project root, with Claude Code installed:
# Just provide an arXiv URL and the CLAUDE.md pipeline runs automatically
claude "https://arxiv.org/abs/2011.14439"
```

---

## Source Code Reference

### Python Dependencies (`requirements.txt`)

```
arxiv>=2.1.0
pymupdf>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
numpy>=1.24.0
pandas>=1.5.0
scikit-learn>=1.2.0
Pillow>=9.4.0
tabulate>=0.9.0
psutil>=5.9.0
requests>=2.28.0
tqdm>=4.65.0
```

### Frontend Dependencies (`frontend/package.json`)

```
@tailwindcss/postcss: ^4.1.18
@types/node: ^25.2.3
@types/react: ^19.2.14
next: ^16.1.6
postcss: ^8.5.6
react: ^19.2.4
react-dom: ^19.2.4
tailwindcss: ^4.1.18
typescript: ^5.9.3
```

---

### `api.py`

```python
"""FastAPI backend for Paper Replicator with SSE streaming."""

import asyncio
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="Paper Replicator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store
jobs: dict[str, dict] = {}

BASE_DIR = Path(__file__).parent


class ReplicateRequest(BaseModel):
    arxiv_url: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    phase: str
    progress: int
    logs: list[str]
    results: dict | None = None


def run_cmd(cmd: str, cwd: str | None = None) -> tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=cwd or str(BASE_DIR), timeout=600,
    )
    return result.returncode, result.stdout, result.stderr


def extract_paper_id(url: str) -> str:
    """Extract arXiv paper ID from URL."""
    import re
    patterns = [
        r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",
        r"arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)",
        r"^(\d{4}\.\d{4,5}(?:v\d+)?)$",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError(f"Invalid arXiv URL: {url}")


async def run_pipeline(job_id: str, arxiv_url: str):
    """Run the full replication pipeline, updating job state as we go."""
    job = jobs[job_id]

    def log(msg: str):
        job["logs"].append(msg)

    def set_phase(phase: str, progress: int):
        job["phase"] = phase
        job["progress"] = progress

    try:
        paper_id = extract_paper_id(arxiv_url)
        job["paper_id"] = paper_id
        log(f"Extracted paper ID: {paper_id}")

        # Phase 1: Create workspace
        set_phase("workspace", 5)
        log("Creating workspace...")
        rc, out, err = run_cmd(f"python -u src/workspace.py {paper_id}")
        if rc != 0:
            raise RuntimeError(f"Workspace creation failed: {err}")
        log("Workspace created.")

        # Phase 2: Fetch paper
        set_phase("fetching", 10)
        log("Fetching paper from arXiv...")
        rc, out, err = run_cmd(
            f"python -u src/fetch_paper.py {paper_id} --output outputs/{paper_id}/paper/"
        )
        if rc != 0:
            raise RuntimeError(f"Paper fetch failed: {err}")
        metadata = json.loads(out) if out.strip() else {}
        job["title"] = metadata.get("title", "Unknown")
        log(f"Paper: {job['title']}")

        # Phase 3: Parse paper
        set_phase("parsing", 20)
        log("Parsing PDF to markdown...")
        rc, out, err = run_cmd(
            f"python -u src/parse_paper.py outputs/{paper_id}/paper/paper.pdf "
            f"--output outputs/{paper_id}/paper/parsed.md"
        )
        if rc != 0:
            raise RuntimeError(f"PDF parse failed: {err}")
        log("Paper parsed successfully.")

        # Phase 4: Analysis
        set_phase("analyzing", 30)
        log("Analyzing paper — extracting architecture, hyperparameters, dataset info...")
        await asyncio.sleep(0.5)

        spec_dir = BASE_DIR / "outputs" / paper_id / "spec"
        if (spec_dir / "claimed_metrics.json").exists():
            log("Found existing paper analysis specs.")
        else:
            log("Note: Full paper analysis requires Claude Code agent. Using demo specs.")
            spec_dir.mkdir(parents=True, exist_ok=True)
            claimed = {
                "logistic_test_accuracy": 32.0,
                "mlp_test_accuracy": 68.0,
                "cnn_test_accuracy": 94.0,
                "gru_test_accuracy": 91.0,
            }
            with open(spec_dir / "claimed_metrics.json", "w") as f:
                json.dump(claimed, f, indent=2)
        log("Paper analysis complete.")
        set_phase("analyzing", 40)

        # Phase 5: Code generation
        set_phase("generating", 45)
        log("Checking for generated training code...")
        code_dir = BASE_DIR / "outputs" / paper_id / "code"
        if (code_dir / "train.py").exists():
            log("Found existing training code.")
        else:
            log("Note: Code generation requires Claude Code agent. No code found.")
            job["status"] = "error"
            job["error"] = "No training code generated. Run pipeline via Claude Code first."
            return
        set_phase("generating", 50)

        # Phase 6: Install dependencies
        set_phase("installing", 55)
        log("Installing code dependencies...")
        req_path = code_dir / "requirements.txt"
        if req_path.exists():
            rc, out, err = run_cmd(f"pip install -r {req_path}")
            if rc != 0:
                log(f"Warning: Some deps failed: {err[:200]}")
        log("Dependencies ready.")

        # Phase 7: Training
        set_phase("training", 60)
        log("Starting model training (all 4 models, 3 runs each)...")

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", str(code_dir / "train.py"),
            "--epochs", "100", "--batch_size", "128", "--lr", "1e-3",
            "--patience", "10", "--num_runs", "3",
            "--output_dir", str(BASE_DIR / "outputs" / paper_id / "results"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(BASE_DIR),
        )

        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode().strip()
            if text:
                log(text)
                if "Training LOGISTIC" in text:
                    set_phase("training", 62)
                elif "Training MLP" in text:
                    set_phase("training", 68)
                elif "Training CNN" in text:
                    set_phase("training", 74)
                elif "Training GRU" in text:
                    set_phase("training", 80)
                elif "FINAL RESULTS" in text:
                    set_phase("training", 88)

        await proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("Training failed")
        log("Training complete!")

        # Phase 8: Evaluation
        set_phase("evaluating", 90)
        log("Generating evaluation figures...")
        rc, out, err = run_cmd(
            f"python -u outputs/{paper_id}/code/evaluate.py "
            f"--output_dir outputs/{paper_id}/results"
        )
        if rc != 0:
            log(f"Warning: Evaluation had issues: {err[:200]}")
        log("Figures generated.")

        # Phase 9: Comparison
        set_phase("comparing", 95)
        log("Comparing results against paper...")
        rc, out, err = run_cmd(
            f"python -u src/compare_results.py "
            f"outputs/{paper_id}/spec/claimed_metrics.json "
            f"outputs/{paper_id}/results/metrics.json "
            f"--output outputs/{paper_id}/results/comparison.json"
        )
        if out:
            for line in out.strip().split("\n")[:10]:
                log(line)

        # Phase 10: Report
        set_phase("reporting", 98)
        log("Generating reproducibility report...")
        rc, out, err = run_cmd(f"python -u src/generate_report.py outputs/{paper_id}")
        log("Report generated.")

        # Load final results
        results = {}
        metrics_path = BASE_DIR / "outputs" / paper_id / "results" / "metrics.json"
        comparison_path = BASE_DIR / "outputs" / paper_id / "results" / "comparison.json"
        report_path = BASE_DIR / "outputs" / paper_id / "report" / "reproducibility_report.md"

        if metrics_path.exists():
            with open(metrics_path) as f:
                results["metrics"] = json.load(f)
        if comparison_path.exists():
            with open(comparison_path) as f:
                results["comparison"] = json.load(f)
        if report_path.exists():
            with open(report_path) as f:
                results["report"] = f.read()

        figures_dir = BASE_DIR / "outputs" / paper_id / "results" / "figures"
        if figures_dir.exists():
            results["figures"] = [f.name for f in sorted(figures_dir.glob("*.png"))]

        job["results"] = results
        job["status"] = "completed"
        set_phase("done", 100)
        log("Pipeline complete!")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        log(f"Error: {e}")


@app.post("/api/replicate")
async def start_replication(req: ReplicateRequest):
    """Start a new paper replication job."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "phase": "starting",
        "progress": 0,
        "logs": [],
        "results": None,
        "paper_id": None,
        "title": None,
        "error": None,
        "created_at": time.time(),
    }
    asyncio.create_task(run_pipeline(job_id, req.arxiv_url))
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get current job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    """SSE stream of job updates."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator() -> AsyncGenerator:
        last_log_idx = 0
        while True:
            job = jobs[job_id]
            new_logs = job["logs"][last_log_idx:]
            if new_logs:
                last_log_idx = len(job["logs"])
                yield {
                    "event": "update",
                    "data": json.dumps({
                        "status": job["status"],
                        "phase": job["phase"],
                        "progress": job["progress"],
                        "logs": new_logs,
                        "title": job.get("title"),
                        "paper_id": job.get("paper_id"),
                    }),
                }
            if job["status"] in ("completed", "error"):
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "status": job["status"],
                        "results": job.get("results"),
                        "error": job.get("error"),
                    }),
                }
                break
            await asyncio.sleep(0.3)

    return EventSourceResponse(event_generator())


@app.get("/api/jobs/{job_id}/figures/{filename}")
async def get_figure(job_id: str, filename: str):
    """Serve generated figure images."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    paper_id = jobs[job_id].get("paper_id")
    if not paper_id:
        raise HTTPException(status_code=404, detail="No paper ID")
    fig_path = BASE_DIR / "outputs" / paper_id / "results" / "figures" / filename
    if not fig_path.exists():
        raise HTTPException(status_code=404, detail="Figure not found")
    from fastapi.responses import FileResponse
    return FileResponse(str(fig_path), media_type="image/png")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### `src/workspace.py`

```python
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
```

---

### `src/fetch_paper.py`

```python
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
```

---

### `src/parse_paper.py`

```python
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
```

---

### `src/compare_results.py`

```python
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
        verdict, emoji = "INCONCLUSIVE", "\u26a0\ufe0f"
    elif within == total:
        verdict, emoji = "REPRODUCED", "\u2705"
    elif within >= total * 0.5:
        verdict, emoji = "PARTIALLY_REPRODUCED", "\U0001f7e1"
    else:
        verdict, emoji = "NOT_REPRODUCED", "\u274c"

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
                         c["difference_description"], "\u2705" if c["within_tolerance"] else "\u274c"])

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
```

---

### `src/generate_report.py`

```python
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
        lines.append("### \u26a0\ufe0f INCONCLUSIVE\n*Comparison data not available.*\n")

    # Results Comparison
    lines.append("## Results Comparison\n")
    if comparison and "comparisons" in comparison:
        lines.append("| Metric | Claimed | Reproduced | Difference | Status |")
        lines.append("|--------|---------|------------|------------|--------|")
        for name, c in comparison["comparisons"].items():
            if c.get("status") == "MISSING":
                lines.append(f"| {name} | {c['claimed']} | N/A | N/A | MISSING |")
            else:
                status = "\u2705" if c["within_tolerance"] else "\u274c"
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
```

---

### `frontend/app/layout.tsx`

```tsx
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Paper Replicator",
  description: "Autonomous ML paper reproduction powered by Claude Code",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#0a0a0f] text-gray-100 min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
```

---

### `frontend/app/page.tsx`

```tsx
"use client";

import { useState, useRef, useEffect, useCallback } from "react";

// ─── Types ──────────────────────────────────────────────────────────
interface ComparisonMetric {
  claimed: number;
  reproduced: number | null;
  absolute_difference?: number;
  difference_description?: string;
  within_tolerance?: boolean;
  status?: string;
}

interface Results {
  metrics?: Record<string, number>;
  comparison?: {
    comparisons: Record<string, ComparisonMetric>;
    summary: {
      total_metrics: number;
      within_tolerance: number;
      outside_tolerance: number;
      verdict: string;
      verdict_emoji: string;
    };
  };
  report?: string;
  figures?: string[];
}

// ─── Phase config ───────────────────────────────────────────────────
const PHASES: Record<string, { label: string; icon: string }> = {
  starting: { label: "Starting", icon: "⏳" },
  workspace: { label: "Creating workspace", icon: "📁" },
  fetching: { label: "Fetching paper", icon: "📥" },
  parsing: { label: "Parsing PDF", icon: "📄" },
  analyzing: { label: "Analyzing paper", icon: "🔬" },
  generating: { label: "Generating code", icon: "💻" },
  installing: { label: "Installing deps", icon: "📦" },
  training: { label: "Training models", icon: "🧠" },
  evaluating: { label: "Evaluating", icon: "📊" },
  comparing: { label: "Comparing results", icon: "⚖️" },
  reporting: { label: "Generating report", icon: "📝" },
  done: { label: "Complete", icon: "✅" },
};

// ─── Main Page ──────────────────────────────────────────────────────
export default function Home() {
  const [url, setUrl] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle");
  const [phase, setPhase] = useState<string>("starting");
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [title, setTitle] = useState<string | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [error, setError] = useState<string | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const connectSSE = useCallback((id: string) => {
    const es = new EventSource(`/api/jobs/${id}/stream`);

    es.addEventListener("update", (e) => {
      const data = JSON.parse(e.data);
      setStatus(data.status);
      setPhase(data.phase);
      setProgress(data.progress);
      if (data.title) setTitle(data.title);
      if (data.logs?.length) {
        setLogs((prev) => [...prev, ...data.logs]);
      }
    });

    es.addEventListener("complete", (e) => {
      const data = JSON.parse(e.data);
      setStatus(data.status);
      if (data.results) setResults(data.results);
      if (data.error) setError(data.error);
      setProgress(100);
      es.close();
    });

    es.onerror = () => {
      es.close();
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;

    setLogs([]);
    setResults(null);
    setError(null);
    setTitle(null);
    setStatus("running");
    setPhase("starting");
    setProgress(0);

    try {
      const res = await fetch("/api/replicate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arxiv_url: url }),
      });
      const data = await res.json();
      setJobId(data.job_id);
      connectSSE(data.job_id);
    } catch {
      setError("Failed to connect to API. Is the backend running?");
      setStatus("error");
    }
  };

  const isRunning = status === "running";
  const isDone = status === "completed";
  const isError = status === "error";

  return (
    <main className="max-w-5xl mx-auto px-6 py-12">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-5xl font-bold mb-3 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
          Paper Replicator
        </h1>
        <p className="text-gray-400 text-lg">
          Paste an arXiv URL. Get a reproducibility verdict.
        </p>
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="mb-10">
        <div className="flex gap-3">
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://arxiv.org/abs/2011.14439"
            disabled={isRunning}
            className="flex-1 px-5 py-4 bg-[#12121a] border border-gray-700/50 rounded-xl text-white text-lg placeholder-gray-500 focus:outline-none focus:border-purple-500/50 focus:ring-1 focus:ring-purple-500/30 transition-all disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={isRunning || !url.trim()}
            className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl hover:from-blue-500 hover:to-purple-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed text-lg"
          >
            {isRunning ? "Running..." : "Replicate"}
          </button>
        </div>
      </form>

      {/* Pipeline status */}
      {status !== "idle" && (
        <div className="space-y-6">
          {title && (
            <div className="bg-[#12121a] rounded-xl p-5 border border-gray-800/50">
              <p className="text-sm text-gray-400 mb-1">Paper</p>
              <p className="text-xl font-medium">{title}</p>
            </div>
          )}

          {/* Progress bar */}
          <div className="bg-[#12121a] rounded-xl p-5 border border-gray-800/50">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className="text-lg">
                  {PHASES[phase]?.icon || "⏳"}
                </span>
                <span className="font-medium">
                  {PHASES[phase]?.label || phase}
                </span>
              </div>
              <span className="text-gray-400 text-sm">{progress}%</span>
            </div>
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-500 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>

            <div className="flex flex-wrap gap-2 mt-4">
              {Object.entries(PHASES).map(([key, val]) => {
                const phaseKeys = Object.keys(PHASES);
                const currentIdx = phaseKeys.indexOf(phase);
                const thisIdx = phaseKeys.indexOf(key);
                const isPast = thisIdx < currentIdx;
                const isCurrent = key === phase;

                return (
                  <span
                    key={key}
                    className={`text-xs px-2.5 py-1 rounded-full transition-all ${
                      isCurrent
                        ? "bg-purple-500/20 text-purple-300 border border-purple-500/30"
                        : isPast
                        ? "bg-green-500/10 text-green-400/70"
                        : "bg-gray-800/50 text-gray-600"
                    }`}
                  >
                    {isPast ? "✓" : val.icon} {val.label}
                  </span>
                );
              })}
            </div>
          </div>

          {/* Live logs */}
          <div className="bg-[#12121a] rounded-xl border border-gray-800/50 overflow-hidden">
            <div className="px-5 py-3 border-b border-gray-800/50 flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isRunning ? "bg-green-400 animate-pulse" : isDone ? "bg-blue-400" : "bg-red-400"}`} />
              <span className="text-sm font-medium text-gray-300">Pipeline Output</span>
            </div>
            <div className="p-5 max-h-80 overflow-y-auto font-mono text-sm space-y-0.5">
              {logs.map((log, i) => (
                <div
                  key={i}
                  className={`${
                    log.startsWith("Error")
                      ? "text-red-400"
                      : log.includes("===")
                      ? "text-purple-400 font-bold"
                      : log.includes("final:")
                      ? "text-green-400"
                      : "text-gray-400"
                  }`}
                >
                  {log}
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>

          {isError && error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-5">
              <p className="text-red-400 font-medium">Error</p>
              <p className="text-red-300 text-sm mt-1">{error}</p>
            </div>
          )}

          {isDone && results && <ResultsPanel results={results} jobId={jobId!} />}
        </div>
      )}
    </main>
  );
}

// ─── Results Panel ──────────────────────────────────────────────────
function ResultsPanel({ results, jobId }: { results: Results; jobId: string }) {
  const comparison = results.comparison;
  const verdict = comparison?.summary;

  return (
    <div className="space-y-6">
      {verdict && (
        <div
          className={`rounded-xl p-6 border ${
            verdict.verdict === "REPRODUCED"
              ? "bg-green-500/10 border-green-500/30"
              : verdict.verdict === "PARTIALLY_REPRODUCED"
              ? "bg-yellow-500/10 border-yellow-500/30"
              : verdict.verdict === "NOT_REPRODUCED"
              ? "bg-red-500/10 border-red-500/30"
              : "bg-gray-500/10 border-gray-500/30"
          }`}
        >
          <div className="text-center">
            <p className="text-4xl mb-2">{verdict.verdict_emoji}</p>
            <h2 className="text-2xl font-bold">
              {verdict.verdict.replace(/_/g, " ")}
            </h2>
            <p className="text-gray-400 mt-1">
              {verdict.within_tolerance} of {verdict.total_metrics} metrics within tolerance
            </p>
          </div>
        </div>
      )}

      {comparison && (
        <div className="bg-[#12121a] rounded-xl border border-gray-800/50 overflow-hidden">
          <div className="px-5 py-3 border-b border-gray-800/50">
            <span className="font-medium text-gray-300">Results Comparison</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 text-left">
                  <th className="px-5 py-3 font-medium">Metric</th>
                  <th className="px-5 py-3 font-medium">Paper</th>
                  <th className="px-5 py-3 font-medium">Reproduced</th>
                  <th className="px-5 py-3 font-medium">Difference</th>
                  <th className="px-5 py-3 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(comparison.comparisons).map(([name, m]) => (
                  <tr key={name} className="border-t border-gray-800/30">
                    <td className="px-5 py-3 font-mono text-gray-300">
                      {name.replace(/_/g, " ")}
                    </td>
                    <td className="px-5 py-3">{m.claimed?.toFixed(1)}%</td>
                    <td className="px-5 py-3">
                      {m.reproduced !== null ? `${m.reproduced.toFixed(1)}%` : "N/A"}
                    </td>
                    <td className="px-5 py-3 text-gray-400 text-xs">
                      {m.difference_description || "—"}
                    </td>
                    <td className="px-5 py-3 text-lg">
                      {m.within_tolerance ? "✅" : m.status === "MISSING" ? "⚠️" : "❌"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {results.figures && results.figures.length > 0 && (
        <div className="bg-[#12121a] rounded-xl border border-gray-800/50 overflow-hidden">
          <div className="px-5 py-3 border-b border-gray-800/50">
            <span className="font-medium text-gray-300">Reproduced Figures</span>
          </div>
          <div className="p-5 grid gap-4">
            {results.figures.map((fig) => (
              <div key={fig} className="bg-white rounded-lg p-2">
                <img
                  src={`/api/jobs/${jobId}/figures/${fig}`}
                  alt={fig}
                  className="w-full rounded"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {results.report && (
        <div className="bg-[#12121a] rounded-xl border border-gray-800/50 overflow-hidden">
          <div className="px-5 py-3 border-b border-gray-800/50">
            <span className="font-medium text-gray-300">Full Report</span>
          </div>
          <div className="p-5 max-h-96 overflow-y-auto">
            <pre className="text-sm text-gray-400 whitespace-pre-wrap font-mono">
              {results.report}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
```

---

### `frontend/next.config.ts`

```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
```
