"""ZRT capture pipeline: load model, trace phases, write outputs.

This is the **top-level orchestration module** for the ZRT tool.
It ties together model loading (``graph.model_loader``), dispatch/compile
capture (``graph.dispatch``, internal helpers), fusion
(``transform.fusion``), and reporting (``report.excel_writer``,
``report.onnx_exporter``, ``report.dot_exporter``) into a single pipeline.

Public API::

    from python.zrt.pipeline import run_trace_phases, run_trace

    # Inference: trace both prefill and decode in one call (recommended)
    output_dir, phase_records = run_trace_phases(
        model_id="deepseek-ai/DeepSeek-V3-0324",
        num_layers=4,
        batch_size=1,
        seq_len=128,
    )

    # Training: trace forward + backward (gradient ops included).
    # result.graphs["train_forward"] and result.graphs["train_backward"] hold
    # the captured raw OpGraph for each phase.
    # result.graphs["train"] holds the stitched fwd+bwd OpGraph — the two
    # phases merged into one connected graph, with cross-phase edges and
    # annotations["phase"] ∈ {"fwd", "bwd"}.
    # Module-boundary fusion is now applied later by the transform pipeline
    # (FusionPass — bucket → rule-fuse | structural-collapse), so no
    # separate fused graph is returned here.
    output_dir, phase_records = run_trace_phases(
        model_id="deepseek-ai/DeepSeek-V3-0324",
        num_layers=4,
        batch_size=1,
        seq_len=128,
        phases=("train_forward", "train_backward"),
    )
    result = run_trace_phases(...)
    train_graph = result.graphs["train"]            # OpGraph (stitched)
    fwd_graph   = result.graphs["train_forward"]    # OpGraph
    bwd_graph   = result.graphs["train_backward"]   # OpGraph

    # Single phase
    output_dir, records = run_trace(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        num_layers=4,
        phase="prefill",
    )

CLI usage::

    python -m python.zrt --model-id deepseek-ai/DeepSeek-V3-0324 --layers 4
    python -m python.zrt --model-id hf_models/deepseek_v3 --train --layers 2
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from python.zrt.graph.dispatch import RecordingDispatch, TensorTracker
from python.zrt.graph.graph_builder import build_op_graph, build_fused_op_graph
from python.zrt.graph.model_loader import load_model
from python.zrt.graph.tracker import ModuleTracker, NullModuleTracker
from python.zrt.ir.adapter import records_to_opgraph, stitch_fwd_bwd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

logger = logging.getLogger(__name__)


# ── Backward-compatible result types ──────────────────────────────────────────

class TracePhaseResult(tuple):
    """Return value of :func:`run_trace_phases`.

    Behaves as a 2-tuple ``(output_dir, phase_records)`` for backward
    compatibility.  New code can also access ``.graphs`` for the
    ``OpGraph`` IR objects.

    Attributes
    ----------
    graphs : Dict[str, OpGraph]
        Maps each phase name to the raw captured ``OpGraph``.  Module-level
        fusion is applied later by the transform pipeline, so only the raw
        graph is stored here.
    """

    def __new__(cls, output_dir: Path, phase_records: Dict, phase_graphs: Dict):
        instance = super().__new__(cls, (output_dir, phase_records))
        instance.graphs: Dict[str, Any] = phase_graphs
        return instance

    @property
    def output_dir(self) -> Path:
        return self[0]

    @property
    def phase_records(self) -> Dict:
        return self[1]


class TraceResult(tuple):
    """Return value of :func:`run_trace`.

    Behaves as a 2-tuple ``(output_dir, records)`` for backward
    compatibility.  New code can also access ``.graphs``.

    Attributes
    ----------
    graphs : OpGraph | None
        The raw captured ``OpGraph`` for the traced phase, or
        ``None`` if graph building was skipped.
    """

    def __new__(cls, output_dir: Path, records: List, graphs):
        instance = super().__new__(cls, (output_dir, records))
        instance.graphs = graphs  # OpGraph | None
        return instance

    @property
    def output_dir(self) -> Path:
        return self[0]

    @property
    def records(self) -> List:
        return self[1]


# ── Constants ─────────────────────────────────────────────────────────────────

# Backward-compat map for --model v3 / v3.2 shorthand
_MODEL_DIRS = {
    "v3":  "deepseek_v3",
    "v3.2": "deepseek_v3_2",
}

# Normalise legacy phase names
_PHASE_ALIASES = {"forward": "prefill", "train": "train_forward"}

# Phases that run with gradients enabled and model in train() mode
_TRAINING_PHASES = {"train_forward", "train_backward"}


# ── Record filtering ───────────────────────────────────────────────────────────

def _filter_records_by_layers(
    records: List[Dict[str, Any]],
    target_layers: List[int],
) -> List[Dict[str, Any]]:
    """Keep only op records belonging to the given layer indices.

    Records that have no layer tag (e.g. embedding lookup, final norm,
    lm_head) are *always* retained because they are model-level ops that
    appear exactly once regardless of which layers are traced.

    ``node_id`` is renumbered sequentially in the filtered list.
    """
    target_set: Set[str] = {str(i) for i in target_layers}
    filtered: List[Dict[str, Any]] = []
    for rec in records:
        layer_str = rec.get("layer", "")
        if layer_str == "" or layer_str in target_set:
            filtered.append(dict(rec))
    for i, rec in enumerate(filtered):
        rec["node_id"] = i
    return filtered


# ── Public helpers ─────────────────────────────────────────────────────────────

def build_config_summary(
    model_id: str,
    config: Any,
    num_hidden_layers: int,
    batch_size: int,
    seq_len: int,
    target_layers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Return a config dict suitable for the Excel Model Config sheet.

    Always includes ``model_id``, ``model_type``, ``hidden_size``,
    ``num_attention_heads``, and ``vocab_size``.  Architecture-specific
    fields (MLA ranks, MoE counts, etc.) are included when present.
    """
    def _get(attr: str) -> Any:
        return getattr(config, attr, None)

    summary: Dict[str, Any] = {
        "model_id": model_id,
        "model_type": _get("model_type") or "unknown",
        "hidden_size": _get("hidden_size"),
        "intermediate_size": _get("intermediate_size"),
        "num_hidden_layers (full)": getattr(
            config, "_full_num_hidden_layers", _get("num_hidden_layers")),
        "num_hidden_layers (traced)": num_hidden_layers,
        "num_attention_heads": _get("num_attention_heads"),
        "num_key_value_heads": _get("num_key_value_heads"),
        "vocab_size": _get("vocab_size"),
        "batch_size": batch_size,
        "seq_len": seq_len,
    }
    if target_layers is not None:
        summary["target_layers"] = str(target_layers)

    # Optional architecture-specific fields
    for field in (
        "moe_intermediate_size",
        "q_lora_rank", "kv_lora_rank",
        "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
        "n_routed_experts", "n_shared_experts", "num_experts_per_tok",
        "first_k_dense_replace",
        "num_local_experts",
        "sliding_window", "head_dim", "rope_theta",
        "index_head_dim", "index_n_heads", "index_topk",
    ):
        val = _get(field)
        if val is not None:
            summary[field] = val

    return {k: v for k, v in summary.items() if v is not None}


def _make_model_slug(model_id: str) -> str:
    """Create a filesystem-safe slug from a model ID."""
    return re.sub(r"[^\w]+", "_", Path(model_id).name).strip("_")


# ── Internal phase helpers ─────────────────────────────────────────────────────

def _trace_phase(
    model: Any,
    config: Any,
    batch_size: int,
    seq_len: int,
    phase: str,
    past_key_values: Any = None,
    target_layers: Optional[List[int]] = None,
    gradient_checkpointing: bool = False,
    tensor_tracker: Optional["TensorTracker"] = None,
) -> Tuple[List[Dict[str, Any]], "ModuleTracker", Any]:
    """Run one forward pass for *phase* and return ``(records, tracker, output)``.

    ``tracker.remove()`` is called before returning so forward hooks are
    cleaned up, but the tracker's metadata (``path_to_class``,
    ``path_to_children``) remains available for downstream writers.

    Parameters
    ----------
    phase:
        ``"prefill"`` — full-sequence pass (query_len == seq_len, no KV cache).
        ``"decode"``  — single-token pass (query_len == 1, with KV cache).
    past_key_values:
        KV cache returned by a prior prefill pass.  Should be the
        ``past_key_values`` attribute of the prefill model output.
        Ignored when *phase* is ``"prefill"``.
    """
    phase = _PHASE_ALIASES.get(phase, phase)

    if phase == "decode":
        query_len = 1
        pos_start = seq_len          # next token position after the prefill window
    else:
        query_len = seq_len
        pos_start = 0

    input_ids = torch.randint(0, config.vocab_size, (batch_size, query_len))
    position_ids = (
        torch.arange(pos_start, pos_start + query_len)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    if phase == "decode":
        if past_key_values is not None:
            # KV cache present: the model attends to all past + current token.
            # Mask shape (1, 1, 1, seq_len+1) covers the full context window.
            total_len = seq_len + 1
            mask = torch.zeros(1, 1, 1, total_len)
        else:
            # No KV cache: only the single current token is in K/V memory.
            # Use a (1, 1, 1, 1) mask so shapes are consistent.
            mask = torch.zeros(1, 1, 1, 1)
    else:
        # Standard causal mask for the full prefill window.
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)

    forward_kwargs: Dict[str, Any] = dict(
        input_ids=input_ids,
        attention_mask=mask,
        position_ids=position_ids,
        use_cache=True,
    )
    if past_key_values is not None:
        forward_kwargs["past_key_values"] = past_key_values

    is_training = phase in _TRAINING_PHASES
    # train_backward: pause recording during forward, only capture backward ops
    is_backward_only = phase == "train_backward"
    if is_training:
        model.train()
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")

    tensor_tracker = tensor_tracker or TensorTracker()
    tracker = ModuleTracker(model)
    recorder = RecordingDispatch(
        tensor_tracker=tensor_tracker,
        module_tracker=tracker,
        skip_reshapes=True,
        active=not is_backward_only,
        target_layers=target_layers,
    )
    try:
        if is_training:
            with recorder:
                output = model(**forward_kwargs)
                if phase == "train_backward":
                    tracker._forward_depth = 0       # reset any accumulated state
                    tracker._pre_backward_module = tracker.current_module  # snapshot stale module
                    tracker._in_backward_phase = True
                    recorder.active = True  # start capturing backward ops
                    logits = getattr(output, "logits", None)
                    if logits is None:
                        logits = getattr(output, "last_hidden_state", None)
                    if logits is not None:
                        loss = logits.sum()
                        try:
                            loss.backward()
                        except Exception as e:
                            logger.warning("backward() failed (FakeTensor limitation): %s", e)
                    else:
                        logger.warning("No logits/last_hidden_state found; skipping backward.")
        else:
            with recorder, torch.no_grad():
                output = model(**forward_kwargs)
    finally:
        tracker.remove()
        if is_training:
            if gradient_checkpointing:
                model.gradient_checkpointing_disable()
            model.eval()

    return recorder.records, tracker, output


# ── torch.compile graph-mode helpers ──────────────────────────────────────────

def _compile_graph_to_records(
    gm: Any,
    skip_reshapes: bool = True,
    id_to_path: Optional[Dict[int, str]] = None,
    target_layers: Optional[Set[int]] = None,
) -> List[Dict[str, Any]]:
    """Convert a torch.compile-captured GraphModule to op-record dicts.

    The output format is identical to :class:`RecordingDispatch` records so
    that all downstream writers work without modification.  When Dynamo
    preserved ``nn_module_stack`` metadata, ``module_path`` and
    ``module_class`` are populated; otherwise they are left empty.
    """
    import torch.fx
    from python.zrt.graph.tensor_utils import SKIP_OPS
    from python.zrt.graph.classifier import extract_layer_idx, classify_component

    name_to_id: Dict[str, int] = {}
    for node in gm.graph.nodes:
        if node.op == "call_function":
            name_to_id[node.name] = len(name_to_id)

    records: List[Dict[str, Any]] = []

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue

        target = node.target
        if hasattr(target, "_overloadpacket"):
            # Proper aten OpOverload → "aten.mm.default"
            target_str = str(target)
        elif hasattr(target, "__name__"):
            module = getattr(target, "__module__", "") or ""
            target_str = f"{module}.{target.__name__}".lstrip(".")
        else:
            try:
                target_str = str(target)
            except Exception:
                target_str = repr(target)

        if skip_reshapes and target_str in SKIP_OPS:
            continue

        parts = target_str.split(".")
        op_short = parts[1] if len(parts) >= 2 else target_str

        # Output tensors from meta['val']
        out_val = node.meta.get("val")
        if isinstance(out_val, torch.Tensor):
            out_vals = [out_val]
        elif isinstance(out_val, (tuple, list)):
            out_vals = [v for v in out_val if isinstance(v, torch.Tensor)]
        else:
            out_vals = []

        # Input tensors and IDs from arg nodes' meta['val']
        in_vals: List[Any] = []
        in_ids: List[int] = []

        def _collect(arg: Any) -> None:
            if isinstance(arg, torch.fx.Node):
                v = arg.meta.get("val")
                if isinstance(v, torch.Tensor):
                    in_vals.append(v)
                if arg.name in name_to_id:
                    in_ids.append(name_to_id[arg.name])
            elif isinstance(arg, (list, tuple)):
                for a in arg:
                    _collect(a)
            elif isinstance(arg, torch.Tensor):
                # Direct tensor input (not from another node)
                in_vals.append(arg)
            elif isinstance(arg, dict):
                # Handle dictionary inputs
                for value in arg.values():
                    _collect(value)

        for arg in node.args:
            _collect(arg)

        # Module context — Dynamo preserves nn_module_stack
        module_path, module_class = "", ""
        layer = ""
        stack = node.meta.get("nn_module_stack")
        if stack:
            try:
                items = list(stack.items())
                if items:
                    last_key, last_val = items[-1]
                    if isinstance(last_val, tuple) and len(last_val) >= 2:
                        raw_path, cls_obj = last_val[0], last_val[1]
                        module_class = getattr(cls_obj, "__name__", str(cls_obj))
                        # Prefer id_to_path resolution (full path with layers.N)
                        if id_to_path and isinstance(last_key, str):
                            try:
                                resolved = id_to_path.get(int(last_key))
                                if resolved:
                                    module_path = re.sub(
                                        r"^model\.", "", resolved
                                    )
                            except (ValueError, TypeError):
                                pass
                        # Fallback to raw_path from nn_module_stack
                        if not module_path and isinstance(raw_path, str):
                            module_path = re.sub(
                                r"^L\['self'\]\.", "", raw_path
                            )
                        layer = extract_layer_idx(module_path)
                    else:
                        module_class = str(last_val)
                        module_path = module_class
                        layer = extract_layer_idx(module_path)
            except Exception as exc:
                logger.debug("nn_module_stack parse failed: %s", exc)

        if target_layers is not None:
            layer_str = extract_layer_idx(module_path)
            if layer_str and int(layer_str) not in target_layers:
                continue

        out_id = [name_to_id[node.name]] if node.name in name_to_id else []

        records.append({
            "node_id": len(records),
            "op_short": op_short,
            "aten_op": target_str,
            "module_path": module_path,
            "module_class": module_class,
            "layer": layer,
            "component": classify_component(module_path, target_str),
            "src_file": "", "src_line": 0, "src_code": "", "src_func": "",
            "extra_args": "",
            "input_shapes":  ", ".join(str(list(v.shape)) for v in in_vals),
            "input_dtypes":  ", ".join(str(v.dtype) for v in in_vals),
            "output_shapes": ", ".join(str(list(v.shape)) for v in out_vals),
            "output_dtypes": ", ".join(str(v.dtype) for v in out_vals),
            "num_inputs":  len(in_vals),
            "num_outputs": len(out_vals),
            "_input_ids":  in_ids,
            "_output_ids": out_id,
        })

    return records


def _trace_compile_phase(
    model: Any,
    config: Any,
    batch_size: int,
    seq_len: int,
    phase: str,
    fake_mode: Any = None,
    target_layers: Optional[Set[int]] = None,
) -> Tuple[List[Dict[str, Any]], NullModuleTracker]:
    """Capture one phase via ``torch.compile`` with a graph-capturing backend.

    A custom backend collects every ``GraphModule`` that Dynamo produces
    (there may be multiple due to graph breaks).  For ``train_backward``,
    ``loss.backward()`` is called inside the same compile context so gradient
    subgraphs are captured as well.

    Returns ``(records, NullModuleTracker())`` — same shape as
    :func:`_trace_phase` minus the model output.
    """
    phase = _PHASE_ALIASES.get(phase, phase)
    is_training = phase in _TRAINING_PHASES

    query_len = 1 if phase == "decode" else seq_len
    pos_start = seq_len if phase == "decode" else 0

    input_ids = torch.randint(0, config.vocab_size, (batch_size, query_len))
    position_ids = (
        torch.arange(pos_start, pos_start + query_len)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    if phase == "decode":
        mask = torch.zeros(1, 1, 1, query_len)
    else:
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)

    captured: List[Any] = []

    _fm = fake_mode
    if _fm is not None and not getattr(_fm, "allow_non_fake_inputs", False):
        from torch._subclasses.fake_tensor import FakeTensorMode as _FTM
        _fm = _FTM(allow_non_fake_inputs=True)

    class _ValCapture(torch.fx.Interpreter):
        def run_node(self, node: Any) -> Any:
            if _fm is not None:
                with _fm:
                    result = super().run_node(node)
            else:
                result = super().run_node(node)
            node.meta["val"] = result
            return result

    def _backend(gm: Any, example_inputs: Any) -> Any:
        try:
            _ValCapture(gm).run(*example_inputs)
        except Exception as exc:
            logger.debug("Shape propagation failed for captured subgraph: %s", exc)
        captured.append(gm)
        return gm.forward

    if is_training:
        model.train()

    compiled = torch.compile(model, backend=_backend, fullgraph=False)
    fwd_kwargs: Dict[str, Any] = dict(
        input_ids=input_ids,
        attention_mask=mask,
        position_ids=position_ids,
        use_cache=False,
    )

    try:
        if is_training:
            with torch.enable_grad():
                out = compiled(**fwd_kwargs)
                if phase == "train_backward":
                    logits = getattr(out, "logits", None)
                    if logits is None:
                        logits = getattr(out, "last_hidden_state", None)
                    if logits is not None:
                        try:
                            logits.sum().backward()
                        except Exception as exc:
                            logger.warning("backward() skipped in compile mode: %s", exc)
                    else:
                        logger.warning("No logits found; skipping backward.")
        else:
            with torch.no_grad():
                compiled(**fwd_kwargs)
    finally:
        if is_training:
            model.eval()
        torch._dynamo.reset()

    id_to_path: Dict[int, str] = {
        id(mod): name for name, mod in model.named_modules()
    }

    all_records: List[Dict[str, Any]] = []
    for gm in captured:
        all_records.extend(_compile_graph_to_records(gm, id_to_path=id_to_path,
                                                     target_layers=target_layers))
    for i, rec in enumerate(all_records):
        rec["node_id"] = i

    logger.info(
        "  compile-mode captured %d ops from %d subgraph(s).",
        len(all_records), len(captured),
    )
    return all_records, NullModuleTracker()


def _save_stitched_graph(
    stitched_graph: Any,
    slug: str,
    output_dir: Path,
) -> None:
    """Persist a stitched (fwd+bwd) OpGraph as JSON to *output_dir*.

    Writes ``{slug}_train_stitched_graph.json`` using the canonical OpGraph
    serde format.  ONNX export is skipped — the stitched graph has no
    op-record list and the ONNX exporter requires one.
    """
    from python.zrt.ir.serde import save_json

    json_path = output_dir / f"{slug}_train_stitched_graph.json"
    save_json(stitched_graph, str(json_path))
    logger.info("Stitched fwd+bwd graph saved to %s", json_path)



# ── Public API ─────────────────────────────────────────────────────────────────

def run_trace_phases(
    model_id: str,
    num_layers: int = 4,
    batch_size: int = 1,
    seq_len: int = 128,
    output_dir: Optional[Any] = None,
    phases: Tuple[str, ...] = ("prefill", "decode"),
    target_layers: Optional[List[int]] = None,
    auto_layers: bool = True,
    platform: str = "generic",
    graph_mode: bool = False,
    gradient_checkpointing: bool = False,
) -> Tuple[Path, Dict[str, List[Dict[str, Any]]]]:
    """Load *model_id* once, trace each requested phase, write separate files.

    Prefill and decode phases share the same ``FakeTensorMode`` context so
    the KV cache produced by prefill (fake tensors) can be passed directly
    into the decode forward pass.

    Training phases (``train_forward``, ``train_backward``) run with
    ``model.train()`` and gradient computation enabled, so dropout and other
    training-specific ops are captured.  ``train_backward`` additionally
    triggers a ``loss.backward()`` call (via ``logits.sum()``) to capture
    all gradient aten ops.  Training phases are independent — they do not
    participate in KV-cache chaining and can be freely mixed with inference
    phases in the same call.

    Parameters
    ----------
    phases:
        Ordered sequence of phases to trace.  Supported values:

        * ``"prefill"`` — full-sequence inference pass (default KV-cache source).
        * ``"decode"``  — single-token inference pass; uses prefill KV cache
          when both phases are requested together.
        * ``"train_forward"`` — training forward pass (``model.train()``,
          gradients enabled).
        * ``"train_backward"`` — training forward + backward pass; captures
          all gradient ops emitted by ``loss.backward()``.
        * ``"forward"`` / ``"train"`` — aliases for ``"prefill"`` /
          ``"train_forward"`` respectively.
    target_layers:
        Optional explicit list of layer indices to keep in the output.
        For example ``[0, 3]`` retains only ops from layer 0 and layer 3.
        Ops with no layer tag (embedding, lm_head, final norm) are always
        retained.  The model is loaded with ``max(target_layers)+1`` layers
        so all requested indices exist.
    auto_layers:
        When ``True`` and *target_layers* is ``None``, automatically infer
        the first dense and first sparse (MoE) layer indices from the model
        config and use those as *target_layers*.  Ignored when
        *target_layers* is already set.
    graph_mode:
        When ``True``, use ``torch.fx``-based graph capture
        (:mod:`python.zrt.graph.fx_capture`) instead of the default
        ``TorchDispatchMode`` eager path.  Key differences:

        * Explicit producer-consumer edges from FX graph structure.
        * Unified GEMM representation (``aten.linear.default``).
        * Training backward ops captured in a single ``make_fx`` pass.
        * Module-path annotations may be absent (fusion still works, but
          component labels fall back to bare aten op names).

        The output format is identical to the eager path; all downstream
        writers (Excel, JSON, ONNX) work without modification.

    Returns
    -------
    TracePhaseResult
        A tuple ``(output_dir, {phase: records})``.  New code should use the
        ``.graphs`` attribute:

        * ``result.graphs["train_forward"]``  → raw ``OpGraph``
        * ``result.graphs["train_backward"]`` → raw ``OpGraph``
        * ``result.graphs["train"]``          → stitched ``OpGraph``
          — present only when **both** ``train_forward`` and ``train_backward``
          were requested.  The stitched graph has ``phase="train"``; every node
          carries ``annotations["phase"] ∈ {"fwd", "bwd"}`` and cross-phase
          data-flow edges connect forward producers to backward consumers.

    Output files (per phase)
    ------------------------
    ``<slug>_prefill_ops.xlsx``            — op table, Excel
    ``<slug>_decode_ops.xlsx``             — same for decode
    ``<slug>_train_forward_ops.xlsx``      — training forward op table
    ``<slug>_train_backward_ops.xlsx``     — training backward op table
    …
    """
    from python.zrt.graph.model_loader import _load_config
    from python.zrt.graph.patches import apply_compat_patches

    # ── Resolve target_layers before loading the model ────────────────────
    if auto_layers and target_layers is None:
        # Quick config-only load to infer layer types (no model weights).
        apply_compat_patches()
        cfg_tmp, _ = _load_config(model_id)
        # target_layers = auto_target_layers(cfg_tmp)
        logger.info("Auto-selected target layers: %s", target_layers)

    # When specific layers are requested, ensure we load enough layers.
    effective_num_layers = num_layers
    if target_layers:
        min_required = max(target_layers) + 1
        if min_required > effective_num_layers:
            logger.info(
                "Extending num_layers %d → %d to include target layer %d.",
                effective_num_layers, min_required, max(target_layers),
            )
            effective_num_layers = min_required

    # Detect if any requested phases are training phases — needed for DSV4
    # to apply patch_for_training_capture (removes @inference_mode, upgrades
    # kernel stubs to differentiable versions).
    is_training_mode = any(p in _TRAINING_PHASES for p in phases)

    logger.info(
        "Loading model %s (%d layers, graph_mode=%s, training=%s) …",
        model_id, effective_num_layers, graph_mode, is_training_mode,
    )

    model, config, fake_mode = load_model(
        model_id, num_hidden_layers=effective_num_layers, training=is_training_mode)

    slug = _make_model_slug(model_id)
    if output_dir is None:
        output_dir = Path("output") / slug
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: Dict[str, List[Dict[str, Any]]] = {}
    all_graphs: Dict[str, Any] = {}
    past_key_values: Any = None
    canonical_phases = [_PHASE_ALIASES.get(p, p) for p in phases]

    # Create a shared TensorTracker for training phases so that tensor IDs are
    # globally unique across train_forward and train_backward.  This enables
    # exact tensor-ID matching in stitch_fwd_bwd.
    has_training = any(p in _TRAINING_PHASES for p in canonical_phases)
    shared_tracker = TensorTracker() if has_training else None

    try:
        for phase in canonical_phases:
            if graph_mode:
                # ── torch.compile graph-mode path ─────────────────────────
                records, tracker = _trace_compile_phase(
                    model, config, batch_size, seq_len, phase, fake_mode,
                    target_layers=set(target_layers) if target_layers else None)
            else:
                # ── Eager TorchDispatchMode path ──────────────────────────
                query_len = 1 if phase == "decode" else seq_len
                logger.info(
                    "Tracing %s phase (batch=%d, query_len=%d, grad=%s) …",
                    phase, batch_size, query_len, phase in _TRAINING_PHASES,
                )
                # Pass shared tracker for training phases so IDs are unique
                phase_tracker = shared_tracker if phase in _TRAINING_PHASES else None
                records, tracker, output = _trace_phase(
                    model, config, batch_size, seq_len, phase, past_key_values,
                    target_layers=target_layers,
                    gradient_checkpointing=gradient_checkpointing,
                    tensor_tracker=phase_tracker)
                logger.info("  Captured %d ops.", len(records))

                # Pass KV cache from prefill into the subsequent decode pass.
                if phase == "prefill" and "decode" in canonical_phases:
                    past_key_values = getattr(output, "past_key_values", None)
                    if past_key_values is None:
                        logger.warning(
                            "Prefill output contains no past_key_values; "
                            "decode pass will run without KV cache "
                            "(attention shapes may differ from a real decode step)")

            raw_opgraph = records_to_opgraph(
                records, name=f"{slug}_{phase}", phase=phase)
            all_records[phase] = records
            all_graphs[phase] = raw_opgraph

    finally:
        fake_mode.__exit__(None, None, None)

    # ── Auto-stitch forward + backward into a unified training graph ──────────
    # Triggered automatically when both train_forward and train_backward are
    # captured in the same call.  The shared TensorTracker (built above) gives
    # stitch_fwd_bwd exact tensor-ID matching across phases.
    if "train_forward" in all_graphs and "train_backward" in all_graphs:
        fwd_raw = all_graphs["train_forward"]
        bwd_raw = all_graphs["train_backward"]
        stitched_raw = stitch_fwd_bwd(fwd_raw, bwd_raw, name=f"{slug}_train_raw")
        all_graphs["train"] = stitched_raw
        logger.info(
            "Stitched training graph: %d fwd + %d bwd = %d total nodes",
            len(fwd_raw), len(bwd_raw), len(stitched_raw),
        )

    logger.info("All outputs saved to %s", output_dir)
    return TracePhaseResult(output_dir, all_records, all_graphs)


def run_trace(
    model_id: str,
    num_layers: int = 4,
    batch_size: int = 1,
    seq_len: int = 128,
    output_dir: Optional[Any] = None,
    phase: str = "prefill",
    target_layers: Optional[List[int]] = None,
    auto_layers: bool = False,
    platform: str = "generic",
    graph_mode: bool = False,
) -> Tuple[Path, List[Dict[str, Any]]]:
    """Load *model_id*, trace a single phase, write Excel + graph outputs.

    For tracing both prefill and decode in one call prefer
    :func:`run_trace_phases`.

    Parameters
    ----------
    phase:
        ``"prefill"`` (default) — full-sequence forward pass.
        ``"decode"``            — single-token forward pass (no KV cache fed
                                  in; use :func:`run_trace_phases` to get the
                                  correct decode shapes with KV cache).
        ``"forward"``           — legacy alias for ``"prefill"``.
    target_layers:
        Optional list of layer indices to retain in the output.
    auto_layers:
        Auto-infer first dense + first sparse layer when ``target_layers``
        is not supplied.

    Returns
    -------
    (output_dir, records)
    """
    result = run_trace_phases(
        model_id=model_id,
        num_layers=num_layers,
        batch_size=batch_size,
        seq_len=seq_len,
        output_dir=output_dir,
        phases=(_PHASE_ALIASES.get(phase, phase),),
        target_layers=target_layers,
        auto_layers=auto_layers,
        platform=platform,
        graph_mode=graph_mode,
    )
    canonical = _PHASE_ALIASES.get(phase, phase)
    phase_graphs = result.graphs.get(canonical)
    return TraceResult(result.output_dir, result.phase_records[canonical], phase_graphs)
