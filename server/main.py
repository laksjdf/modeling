"""ZRT-Sim FastAPI service.

Wraps the three CLI modes (graph capture, spec estimate, grid search) as
async background jobs with a simple in-memory job store.

Launch (from project root):
    uvicorn server.main:app --host 0.0.0.0 --port 8000

    # with auto-reload during development:
    uvicorn server.main:app --reload --host 0.0.0.0 --port 8000

Poll a job:
    GET /jobs/{job_id}

Interactive docs:
    http://localhost:8000/docs
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException

from .schemas import EstimateRequest, JobResponse, JobStatus, SearchRequest, TraceRequest

# Ensure 'python/' is on sys.path so that zrt.* imports inside the training
# module (which uses 'from zrt.*') resolve correctly.
_python_dir = str(Path(__file__).parent.parent / "python")
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

app = FastAPI(
    title="ZRT-Sim API",
    description="LLM performance modelling and simulation service.",
    version="1.0.0",
)

# ── In-memory job store ───────────────────────────────────────────────────────
# Each entry: {id, status, result, error, created_at, finished_at}
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _new_job() -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": JobStatus.PENDING,
            "result": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
        }
    return job_id


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _lock:
        _jobs[job_id].update(kwargs)


def _snapshot(job_id: str) -> dict:
    with _lock:
        return dict(_jobs[job_id])


# ── Utility endpoints ─────────────────────────────────────────────────────────

@app.get("/health", tags=["utility"])
def health():
    return {"status": "ok"}


@app.get("/hardware", tags=["utility"], summary="List available hardware specs")
def list_hardware():
    from python.zrt.hardware.registry import list_available
    return {"hardware": list_available()}


@app.get("/models", tags=["utility"], summary="List available local model shorthands")
def list_models():
    from python.zrt.graph.main import _MODEL_DIRS
    return {"models": list(_MODEL_DIRS.keys())}


# ── Job polling ───────────────────────────────────────────────────────────────

@app.get("/jobs", tags=["jobs"], summary="List all submitted jobs")
def list_jobs() -> List[dict]:
    with _lock:
        return list(_jobs.values())


@app.get(
    "/jobs/{job_id}",
    tags=["jobs"],
    response_model=JobResponse,
    summary="Poll job status and result",
)
def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, detail=f"Job '{job_id}' not found")
    return job


# ── POST /trace ───────────────────────────────────────────────────────────────

@app.post(
    "/trace",
    tags=["jobs"],
    response_model=JobResponse,
    status_code=202,
    summary="Submit a graph-capture (+ optional perf modelling) job",
    description=(
        "Traces the operator sequence of an HF causal LM and optionally runs the "
        "inference or training performance modelling pipeline. "
        "Returns a job_id immediately; poll GET /jobs/{job_id} for completion."
    ),
)
def submit_trace(req: TraceRequest, bg: BackgroundTasks):
    job_id = _new_job()
    bg.add_task(_trace_worker, job_id, req)
    return _snapshot(job_id)


def _trace_worker(job_id: str, req: TraceRequest) -> None:
    _update_job(job_id, status=JobStatus.RUNNING)
    try:
        result = _do_trace(req)
        _update_job(
            job_id,
            status=JobStatus.DONE,
            result=result,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        _update_job(
            job_id,
            status=JobStatus.ERROR,
            error=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )


def _do_trace(req: TraceRequest) -> dict:
    from python.zrt.graph.main import run_trace_phases, _MODEL_DIRS

    # Resolve model_id: 'local:<shorthand>' → absolute hf_models/ path
    if req.model_id.startswith("local:"):
        shorthand = req.model_id[len("local:"):]
        if shorthand not in _MODEL_DIRS:
            raise ValueError(
                f"Unknown local model '{shorthand}'. "
                f"Available: {list(_MODEL_DIRS.keys())}"
            )
        model_id = str(
            Path(__file__).parent.parent / "hf_models" / _MODEL_DIRS[shorthand]
        )
    else:
        model_id = req.model_id

    phases = (
        ("train_forward", "train_backward")
        if req.train
        else tuple(req.phases or ["prefill", "decode"])
    )

    target_layers: Optional[List[int]] = None
    if req.target_layers:
        target_layers = [int(x.strip()) for x in req.target_layers.split(",")]

    # When target_layers is explicit, disable auto_layers
    auto_layers = req.auto_layers if target_layers is None else False

    out_dir = Path(req.output_dir) if req.output_dir else None

    trace_result = run_trace_phases(
        model_id=model_id,
        num_layers=req.layers,
        batch_size=req.batch_size,
        seq_len=req.seq_len,
        output_dir=out_dir,
        phases=phases,
        target_layers=target_layers,
        auto_layers=auto_layers,
        platform=req.platform,
        graph_mode=req.graph_mode,
        gradient_checkpointing=req.gradient_checkpointing,
    )

    result: dict[str, Any] = {
        "output_dir": str(trace_result.output_dir),
        "phases": list(trace_result.graphs.keys()),
        "summary": None,
    }

    if not req.hw:
        return result

    # Run perf modelling pipeline
    import python.zrt.hardware.registry as hw_registry
    from python.zrt.cli import _run_inference_pipeline, _run_training_modelling

    hw = hw_registry.load(req.hw)
    fake_args = SimpleNamespace(
        hw=req.hw,
        tp=req.tp,
        pp=req.pp,
        ep=req.ep,
        dp=req.dp,
        cp=req.cp,
        quant=req.quant,
        batch_size=req.batch_size,
        seq_len=req.seq_len,
        # Training extras (only used when req.train is True)
        total_params=req.total_params,
        hidden=req.hidden,
        layers=req.layers,
        num_layers_full=req.num_layers_full,
        zero_stage=req.zero_stage,
        optimizer=req.optimizer,
        muon_rotation=req.muon_rotation,
        muon_ns_steps=req.muon_ns_steps,
        micro_batch=req.micro_batch,
        global_batch=req.global_batch,
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        if req.train:
            _run_training_modelling(fake_args, model_id, hw, trace_result)
        else:
            _run_inference_pipeline(fake_args, model_id, hw, trace_result)

    summary = buf.getvalue().strip()
    if summary:
        result["summary"] = summary

    return result


# ── POST /estimate ────────────────────────────────────────────────────────────

@app.post(
    "/estimate",
    tags=["jobs"],
    response_model=JobResponse,
    status_code=202,
    summary="Submit a spec-based training estimate job",
    description=(
        "Runs spec-based training estimation from a YAML config — no model weights "
        "or graph capture required. "
        "Provide either config_path (server-side file) or config_content (raw YAML)."
    ),
)
def submit_estimate(req: EstimateRequest, bg: BackgroundTasks):
    if not req.config_path and not req.config_content:
        raise HTTPException(422, detail="Provide either config_path or config_content")
    job_id = _new_job()
    bg.add_task(_estimate_worker, job_id, req)
    return _snapshot(job_id)


def _estimate_worker(job_id: str, req: EstimateRequest) -> None:
    _update_job(job_id, status=JobStatus.RUNNING)
    try:
        result = _do_estimate(req)
        _update_job(
            job_id,
            status=JobStatus.DONE,
            result=result,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        _update_job(
            job_id,
            status=JobStatus.ERROR,
            error=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )


def _do_estimate(req: EstimateRequest) -> dict:
    from python.zrt.training.io.config_loader import load_specs
    from python.zrt.training.search.estimator import estimate
    from python.zrt.training.search.report import report_summary, report_to_dict

    config_path, tmp = _resolve_yaml(req.config_path, req.config_content)
    try:
        model, system, strategy = load_specs(config_path)
        report = estimate(model, system, strategy)
        return {
            "summary": report_summary(report),
            "data": report_to_dict(report),
        }
    finally:
        if tmp:
            Path(tmp).unlink(missing_ok=True)


# ── POST /search ──────────────────────────────────────────────────────────────

@app.post(
    "/search",
    tags=["jobs"],
    response_model=JobResponse,
    status_code=202,
    summary="Submit a parallel strategy grid-search job",
    description=(
        "Grid-searches parallel strategies (TP/CP/PP/EP/DP/ZeRO/PPSched) for a "
        "training config and returns the Pareto-optimal frontier. "
        "Provide either config_path (server-side file) or config_content (raw YAML)."
    ),
)
def submit_search(req: SearchRequest, bg: BackgroundTasks):
    if not req.config_path and not req.config_content:
        raise HTTPException(422, detail="Provide either config_path or config_content")
    job_id = _new_job()
    bg.add_task(_search_worker, job_id, req)
    return _snapshot(job_id)


def _search_worker(job_id: str, req: SearchRequest) -> None:
    _update_job(job_id, status=JobStatus.RUNNING)
    try:
        result = _do_search(req)
        _update_job(
            job_id,
            status=JobStatus.DONE,
            result=result,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        _update_job(
            job_id,
            status=JobStatus.ERROR,
            error=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )


def _do_search(req: SearchRequest) -> dict:
    from python.zrt.training.io.config_loader import load_specs
    from python.zrt.training.search.estimator import grid_search, pareto_frontier
    from python.zrt.training.search.space import SearchSpace
    from python.zrt.training.search.report import report_to_dict

    config_path, tmp = _resolve_yaml(req.config_path, req.config_content)
    try:
        model, system, strategy = load_specs(config_path)
        space = SearchSpace(
            micro_batch=strategy.micro_batch,
            global_batch=strategy.global_batch,
        )
        all_reports = grid_search(model, system, space)
        frontier = pareto_frontier(all_reports)
        pareto_data = [report_to_dict(r) for r in frontier]

        if req.output and frontier:
            out = Path(req.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(pareto_data, indent=2))

        return {
            "total_configs": len(all_reports),
            "pareto_count": len(frontier),
            "pareto_frontier": pareto_data,
        }
    finally:
        if tmp:
            Path(tmp).unlink(missing_ok=True)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _resolve_yaml(
    config_path: Optional[str],
    config_content: Optional[str],
) -> tuple[str, Optional[str]]:
    """Return (path_to_use, tmp_path_to_delete).

    If config_path is given, use it directly (tmp_path is None).
    Otherwise write config_content to a temp file.
    """
    if config_path:
        return config_path, None
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    tmp.write(config_content)
    tmp.close()
    return tmp.name, tmp.name
