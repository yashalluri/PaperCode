"""FastAPI backend for Paper Replicator with SSE streaming."""

import asyncio
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
from analyze_paper import analyze_paper
from generate_code import generate_code, validate_and_fix

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

        # Phase 4: Analysis via Claude API
        set_phase("analyzing", 30)
        log("Analyzing paper with Claude — extracting architecture, hyperparameters, dataset info...")

        parsed_md_path = str(BASE_DIR / "outputs" / paper_id / "paper" / "parsed.md")
        spec_dir = str(BASE_DIR / "outputs" / paper_id / "spec")

        try:
            result = await asyncio.to_thread(
                analyze_paper, parsed_md_path, spec_dir, log_callback=log,
            )
            log(f"Analysis complete. Generated {len(result.get('files', []))} spec files.")
        except Exception as e:
            log(f"Warning: Claude analysis failed ({e}). Checking for existing specs...")
            if not (Path(spec_dir) / "claimed_metrics.json").exists():
                raise RuntimeError(f"Paper analysis failed and no existing specs found: {e}")
            log("Using existing spec files.")

        set_phase("analyzing", 40)

        # Phase 5: Code generation via Claude API
        set_phase("generating", 45)
        code_dir = str(BASE_DIR / "outputs" / paper_id / "code")

        if (Path(code_dir) / "train.py").exists():
            log("Found existing training code. Skipping generation.")
        else:
            log("Generating PyTorch code with Claude...")
            try:
                result = await asyncio.to_thread(
                    generate_code, parsed_md_path, spec_dir, code_dir, log_callback=log,
                )
                log(f"Code generation complete: {', '.join(result.get('files', []))}")
            except Exception as e:
                raise RuntimeError(f"Code generation failed: {e}")

            # Validation smoke tests
            log("Running validation smoke tests...")
            try:
                ok = await asyncio.to_thread(
                    validate_and_fix, code_dir, spec_dir, parsed_md_path, log_callback=log,
                )
                if ok:
                    log("All smoke tests passed.")
                else:
                    log("Warning: Some smoke tests failed. Proceeding with training attempt.")
            except Exception as e:
                log(f"Warning: Validation error: {e}. Proceeding anyway.")

        set_phase("generating", 50)

        # Phase 6: Install dependencies
        set_phase("installing", 55)
        log("Installing code dependencies...")
        req_path = Path(code_dir) / "requirements.txt"
        if req_path.exists():
            rc, out, err = run_cmd(f"pip install -r {req_path}")
            if rc != 0:
                log(f"Warning: Some deps failed: {err[:200]}")
        log("Dependencies ready.")

        # Phase 7: Training
        set_phase("training", 60)

        # Read hyperparameters from spec
        hp_path = Path(spec_dir) / "hyperparameters.json"
        hp = {}
        if hp_path.exists():
            try:
                with open(hp_path) as f:
                    hp = json.load(f)
            except json.JSONDecodeError:
                pass

        epochs = hp.get("epochs", 100)
        batch_size = hp.get("batch_size", 128)
        lr = hp.get("learning_rate", hp.get("lr", 1e-3))
        # Handle non-numeric values
        if not isinstance(epochs, (int, float)):
            epochs = 100
        if not isinstance(batch_size, (int, float)):
            batch_size = 128
        if not isinstance(lr, (int, float)):
            lr = 1e-3

        log(f"Starting training (epochs={epochs}, batch_size={batch_size}, lr={lr})...")

        # Build training command with standard args
        train_args = [
            sys.executable, "-u", str(Path(code_dir) / "train.py"),
            "--epochs", str(int(epochs)),
            "--batch_size", str(int(batch_size)),
            "--lr", str(lr),
            "--output_dir", str(BASE_DIR / "outputs" / paper_id / "results"),
        ]

        proc = await asyncio.create_subprocess_exec(
            *train_args,
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
                # Update progress based on which model we're on
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

        # Check for figures
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

    # Run pipeline in background
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

            # Send any new log lines
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
