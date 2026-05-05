"""Integration layer: apply transform pipeline to traced graphs and export results."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from python.zrt.ir.graph import OpGraph
from python.zrt.transform import (
    TransformPipeline, build_default_pipeline, TransformContext,
    ParallelConfig, StreamConfig, export_transformed_graph,
)
from python.zrt.hardware.spec import HardwareSpec

logger = logging.getLogger(__name__)


def run_transform(
    raw_graph: OpGraph,
    output_dir: Path,
    parallel_config: Optional[ParallelConfig] = None,
    stream_config: Optional[StreamConfig] = None,
    pipeline: Optional[TransformPipeline] = None,
    hw_spec: Optional[HardwareSpec] = None,
) -> Tuple[Path, OpGraph]:
    """Apply transform pipeline to a traced OpGraph and export results.

    This is the bridge between graph.tracing (raw ops) and transform.pipeline
    (optimized ops with parallelism, communication, and stream info).

    Parameters
    ----------
    raw_graph : OpGraph
        The raw computation graph from tracing phase
    output_dir : Path
        Directory for output files
    parallel_config : ParallelConfig, optional
        Parallelism strategy (TP/EP/DP/PP/SP). Defaults to single-device.
    stream_config : StreamConfig, optional
        Multi-stream configuration. Defaults to 1 compute stream, 1 comm stream.
    pipeline : TransformPipeline, optional
        Custom transform pipeline. Defaults to build_default_pipeline().
    hw_spec : HardwareSpec, optional
        Hardware specification for performance modeling.

    Returns
    -------
    (output_dir, transformed_graph) : Tuple[Path, OpGraph]
        Paths to generated files and the transformed graph object.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defaults
    if parallel_config is None:
        parallel_config = ParallelConfig()
    if stream_config is None:
        stream_config = StreamConfig()
    if pipeline is None:
        pipeline = build_default_pipeline()
    if hw_spec is None:
        # Minimal default: A100 40GB
        from python.zrt.hardware.gpu import GPU_SPECS
        hw_spec = GPU_SPECS.get("A100_40GB")
        if hw_spec is None:
            logger.warning("No default GPU spec found; using minimal spec")
            hw_spec = _minimal_hw_spec()

    # Build transform context (hw_spec may be None/dict-like for now)
    ctx = TransformContext(
        hw_spec=hw_spec or _minimal_hw_spec(),
        parallel=parallel_config,
        stream_config=stream_config,
    )

    # Apply pipeline
    logger.info(f"Applying transform pipeline: {pipeline}")
    logger.info(f"  Parallelism: {ctx.parallel.describe()}")
    logger.info(f"  Streams: {ctx.stream_config.num_compute_streams} compute + "
                f"{ctx.stream_config.num_comm_streams} comm")

    transformed_graph = pipeline.run(raw_graph, ctx)

    # Export
    logger.info(f"Exporting transformed graph to {output_dir}")
    export_paths = export_transformed_graph(transformed_graph, ctx, output_dir)

    # DOT export
    from python.zrt.report.dot_exporter import export_dot as _export_dot, render_dot as _render_dot
    base_name = raw_graph.name or "graph"
    dot_path = _export_dot(
        transformed_graph,
        output_dir / f"{base_name}_transformed_graph.dot",
    )
    _render_dot(dot_path)  # no-op when graphviz absent

    logger.info("Transform pipeline complete")
    logger.info(f"  Excel: {export_paths['excel']}")
    logger.info(f"  JSON:  {export_paths['json']}")
    logger.info(f"  DOT:   {dot_path}")

    return output_dir, transformed_graph


def _minimal_hw_spec() -> HardwareSpec:
    """Create a minimal hardware spec for fallback."""
    try:
        from python.zrt.hardware.spec import HardwareSpec
        return HardwareSpec(
            name="generic_gpu",
            device_type="gpu",
            peak_flops_fp32=312e12,  # 312 TFLOPS (generic upper bound)
            hbm_bandwidth=900e9,     # 900 GB/s (generic upper bound)
            hbm_capacity=40 * 1024**3,
        )
    except Exception as e:
        logger.warning(f"Could not create HardwareSpec: {e}")
        # Return a dict-like mock
        return {
            "name": "unknown",
            "peak_flops_fp32": 100e12,
            "hbm_bandwidth": 500e9,
        }
