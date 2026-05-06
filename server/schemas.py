"""Pydantic request / response schemas for ZRT-Sim API."""
from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: str
    finished_at: Optional[str] = None


class TraceRequest(BaseModel):
    """Graph capture + optional performance modelling job."""

    model_id: str = Field(
        ...,
        description=(
            "HF Hub ID, local path, or 'local:<shorthand>' "
            "(e.g. 'local:v3'). "
            "Shorthands are listed by GET /models."
        ),
        examples=["Qwen/Qwen2.5-7B-Instruct", "local:v3"],
    )
    layers: int = Field(4, ge=1, description="Number of transformer layers to trace")
    batch_size: int = Field(1, ge=1)
    seq_len: int = Field(128, ge=1, description="Prefill sequence length")
    phases: Optional[List[str]] = Field(
        None,
        description=(
            "Phases to trace. "
            "Inference: prefill | decode. "
            "Training: train_forward | train_backward. "
            "Default: ['prefill', 'decode']."
        ),
    )
    train: bool = Field(
        False,
        description="Shorthand: trace train_forward + train_backward phases.",
    )
    platform: str = Field("generic", description="cuda | ascend_npu | cpu | generic")
    graph_mode: bool = Field(
        False,
        description="Use torch.compile graph capture instead of TorchDispatchMode.",
    )
    gradient_checkpointing: bool = Field(
        False,
        description="Enable activation checkpointing during training phases.",
    )
    target_layers: Optional[str] = Field(
        None,
        description="Comma-separated layer indices to trace, e.g. '0,3'.",
    )
    auto_layers: bool = Field(
        True,
        description="Auto-select first dense and first sparse (MoE) layer.",
    )
    output_dir: Optional[str] = Field(
        None,
        description="Output directory. Defaults to output/<model_slug>.",
    )

    # Performance report (requires hw)
    hw: Optional[str] = Field(
        None,
        description="Hardware spec name for perf report (e.g. nvidia_h100_sxm). "
                    "Available specs are listed by GET /hardware.",
    )

    # Parallelism
    tp: int = Field(1, ge=1, description="Tensor-parallel degree")
    pp: int = Field(1, ge=1, description="Pipeline-parallel degree")
    ep: int = Field(1, ge=1, description="Expert-parallel degree")
    dp: int = Field(1, ge=1, description="Data-parallel degree")
    cp: int = Field(1, ge=1, description="Context-parallel degree")
    quant: Optional[str] = Field(None, description="Weight quant dtype: int4 | int8 | fp8")

    # Training extras
    zero_stage: int = Field(1, ge=0, le=3, description="ZeRO stage 0-3")
    optimizer: str = Field("adam", description="adam | adamw | muon")
    muon_rotation: bool = Field(True, description="Enable Moonshot rotation for Muon")
    muon_ns_steps: Optional[int] = Field(None, description="Newton-Schulz steps for Muon")
    micro_batch: int = Field(1, ge=1, description="Micro-batch size per GPU")
    global_batch: int = Field(32, ge=1, description="Global batch size across DP ranks")
    total_params: Optional[float] = Field(None, description="Full model param count, e.g. 671e9")
    hidden: int = Field(7168, ge=1, description="Hidden dimension for memory estimation")
    num_layers_full: Optional[int] = Field(
        None,
        description="Total layers in the full model (defaults to --layers if not set).",
    )


class EstimateRequest(BaseModel):
    """Spec-based training estimation (no graph capture)."""

    config_path: Optional[str] = Field(
        None,
        description="Filesystem path to a YAML training config.",
        examples=["python/zrt/training/configs/llama3_70b_3d.yaml"],
    )
    config_content: Optional[str] = Field(
        None,
        description="Raw YAML content (alternative to config_path).",
    )


class SearchRequest(BaseModel):
    """Grid-search parallel strategies for a training config."""

    config_path: Optional[str] = Field(
        None,
        description="Filesystem path to a YAML training config.",
    )
    config_content: Optional[str] = Field(
        None,
        description="Raw YAML content (alternative to config_path).",
    )
    output: Optional[str] = Field(
        None,
        description="Write Pareto-frontier JSON to this file path.",
    )
