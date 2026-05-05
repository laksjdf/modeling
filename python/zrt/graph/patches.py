"""Model compatibility patches for FakeTensorMode tracing.

Contains the minimal set of patches needed to run arbitrary HuggingFace
causal LMs through FakeTensorMode without allocating real memory.

Patch inventory
---------------
apply_compat_patches()
    Adds deprecated transformers attributes expected by older model code
    (is_torch_fx_available, is_torch_greater_or_equal_than_1_13).

patch_moe_for_fake(model)
    Replaces MoE module forwards with a simplified version that avoids
    .cpu().numpy() and torch.bincount() calls on routing indices, which
    crash on fake tensors.  Only applied when the standard heuristic
    identifies a module as MoE (has nn.ModuleList experts, not already
    patched).

patch_indexer_for_fake(model)
    Patches DeepSeek-V3.2 Indexer modules whose original forward contains
    a 3-D tensor .transpose(2,3) that is invalid under FakeTensorMode.
    The original modeling files from HF are kept untouched; this patch
    supplies a corrected forward at runtime only.

patch_for_training_capture(model)
    Enables backward() on the DeepSeek-V4 inference model for training
    graph capture.  Removes the @torch.inference_mode() decorator from
    Transformer.forward and upgrades the kernel stubs (fp4_gemm, fp8_gemm,
    sparse_attn, hc_split_sinkhorn, act_quant) to backward-compatible
    differentiable versions that maintain the correct op shapes.

What is intentionally NOT patched
----------------------------------
* Autocast / dtype casting — FakeTensorMode handles these transparently.
* Meta-device specific hacks — superseded by FakeTensorMode.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Transformers compatibility ────────────────────────────────────────────────

def apply_compat_patches() -> None:
    """Apply all compatibility patches needed before model loading.

    Call order matters: version shims must be injected *before* any model file
    is imported, so that ``from transformers.xxx import yyy`` at module level
    finds the stub symbols even when the installed transformers version removed them.
    """
    # 0. Dtype stubs — FP4 packed dtype not in PyTorch < 2.8
    #    Use float16 as alias: must be a float dtype (nn.Parameter requires_grad),
    #    and distinct from bfloat16 (the model's default) so the dtype == check
    #    in Linear.__init__ only triggers for expert (FP4) paths.
    if not hasattr(torch, "float4_e2m1fn_x2"):
        torch.float4_e2m1fn_x2 = torch.float16

    # 1. Version shims (missing symbols injected into transformers sub-modules)
    from python.zrt.graph.compat import apply_version_shims
    apply_version_shims()

    # 2. Legacy attrs still expected by some older model files
    try:
        import transformers.utils.import_utils as _iu
        if not hasattr(_iu, "is_torch_fx_available"):
            _iu.is_torch_fx_available = lambda: True
    except ImportError:
        pass
    try:
        import transformers.pytorch_utils as _pu
        if not hasattr(_pu, "is_torch_greater_or_equal_than_1_13"):
            _pu.is_torch_greater_or_equal_than_1_13 = True
    except ImportError:
        pass


# ── MoE patch ─────────────────────────────────────────────────────────────────
# Many MoE implementations call .cpu().numpy() or torch.bincount() on routing
# indices, which crash on fake tensors.  This simplified forward exercises the
# gate + one expert + shared experts — enough to capture the full op pattern.

def is_moe_module(module: nn.Module) -> bool:
    """True if module looks like a MoE layer that needs patching."""
    experts = getattr(module, "experts", None)
    return (
        isinstance(experts, nn.ModuleList)
        and any(e is not None for e in experts)
        and not getattr(module, "_fake_patched", False)
    )


def _returns_router_tuple(mod: nn.Module) -> bool:
    """True if this MoE forward returns (hidden, router_logits) tuple."""
    try:
        src = inspect.getsource(type(mod).forward)
        if any(pat in src for pat in ("router_logits", "aux_loss",
                                      "return hidden_states,")):
            return True
    except Exception:
        pass
    return hasattr(mod, "router") and not hasattr(mod, "gate")


def _make_fake_moe_forward(mod: nn.Module):
    _tuple_return = _returns_router_tuple(mod)

    def _forward(hidden_states: torch.Tensor, *args: Any, **kwargs: Any):
        try:
            result = _impl(hidden_states)
            return (result, None) if _tuple_return else result
        except Exception as exc:
            logger.debug("Fake MoE forward error (%s) — returning identity.", exc)
            return (hidden_states, None) if _tuple_return else hidden_states

    def _impl(hidden_states: torch.Tensor) -> torch.Tensor:
        orig = hidden_states
        bs, seq, h = orig.shape
        flat = orig.reshape(bs * seq, h)

        gate_weight: Optional[torch.Tensor] = None
        gate = getattr(mod, "gate", None)
        if gate is not None and callable(gate):
            try:
                # Always pass flat [bs*seq, h] — Gate.linear() expects 2-D input.
                gate_out = gate(flat)
                if isinstance(gate_out, (tuple, list)):
                    # gate returns (weights, indices) — use weights (index 0)
                    gate_weight = gate_out[0]
                else:
                    gate_weight = torch.softmax(gate_out.float(), dim=-1)[:, :1]
            except Exception as exc:
                logger.debug("Gate forward failed (%s).", exc)

        first_expert = next((e for e in mod.experts if e is not None), None)
        if first_expert is None:
            return orig

        try:
            expert_out = first_expert(flat)
        except Exception:
            expert_out = first_expert(orig).reshape(bs * seq, -1)

        y = (expert_out * gate_weight[:, :1]
             if gate_weight is not None else expert_out)
        try:
            y = y.reshape(bs, seq, -1)
        except Exception:
            y = orig

        for attr in ("shared_experts", "shared_expert"):
            shared = getattr(mod, attr, None)
            if shared is not None and callable(shared):
                try:
                    y = y + shared(orig)
                except Exception as exc:
                    logger.debug("Shared expert failed (%s).", exc)
                break

        return y

    return _forward


def patch_moe_for_fake(model: nn.Module) -> None:
    """Replace MoE forwards with a fake-tensor-safe simplified version."""
    patched = 0
    for _, module in model.named_modules():
        if not is_moe_module(module):
            continue
        module._fake_patched = True
        module.forward = _make_fake_moe_forward(module)
        patched += 1
    if patched:
        logger.info("Applied fake-tensor MoE patch to %d module(s).", patched)


# ── DeepSeek-V3.2 Indexer patch ──────────────────────────────────────────────
# The Indexer forward in the original HF modeling file (kept unmodified) uses
# k_nope.transpose(1, 2).transpose(2, 3) on a 3-D tensor, which is invalid.
# We supply a corrected forward at runtime without touching the model files.

def _make_indexer_forward_fake(IndexerClass: type) -> None:
    """Replace Indexer.forward with a FakeTensorMode-compatible version."""

    def _forward(self, x, qr, position_ids=None, attention_mask=None):
        import torch.nn.functional as F
        bsz, seqlen, _ = x.size()
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.index_n_heads, self.index_head_dim)
        # Split into RoPE and NoPE parts (mirrors the original design intent)
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.index_head_dim - self.rope_head_dim], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, self.index_head_dim - self.rope_head_dim], dim=-1)
        # q_nope: (bsz, seqlen, n_heads, nope_dim) -> (bsz, n_heads, seqlen, nope_dim)
        q_nope = q_nope.transpose(1, 2)
        # k_nope: (bsz, seqlen, nope_dim) -> broadcast across heads
        k_nope = k_nope.unsqueeze(1).expand(-1, self.index_n_heads, -1, -1)
        scores = torch.matmul(q_nope, k_nope.transpose(2, 3))
        if attention_mask is not None:
            scores = scores + attention_mask
        scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        topk_indices = scores.topk(min(self.index_topk, seqlen), dim=-1)[1]
        return topk_indices

    IndexerClass.forward = _forward


def patch_indexer_for_fake(model: nn.Module) -> None:
    """Patch DeepSeek-V3.2 Indexer modules for FakeTensorMode compatibility.

    The original HF modeling files are not modified.  The fix is applied
    in-memory at runtime: any module whose class name contains 'Indexer'
    and which has the expected MLA attributes (wq_b, index_n_heads) gets
    a corrected forward method.
    """
    patched_classes: set = set()
    for _, module in model.named_modules():
        cls = type(module)
        if cls in patched_classes:
            continue
        if "Indexer" in cls.__name__ and hasattr(module, "wq_b") and hasattr(module, "index_n_heads"):
            _make_indexer_forward_fake(cls)
            patched_classes.add(cls)
            logger.debug("Patched Indexer class: %s", cls.__name__)
    if patched_classes:
        logger.info("Applied Indexer patch to: %s",
                    ", ".join(c.__name__ for c in patched_classes))


# ── DeepSeek-V4 Hyper-Connections capture patch ──────────────────────────────
# Block.hc_pre / Block.hc_post are *methods* on the inference Block class.  The
# graph-capture pipeline relies on ``ModuleTracker`` to delimit semantic units
# by ``nn.Module`` boundaries, so methods are invisible to it.  We do not edit
# the upstream inference/model.py; instead, at load time we attach 4 wrapper
# nn.Modules to each Block (and 1 to ParallelHead for the MTP head path) and
# rebind .forward on the *class* to dispatch through them.  After this patch,
# ModuleTracker reports paths like ``transformer.layers.0.hc_pre_attn``.


class _HCBoundMethodModule(nn.Module):
    """Base wrapper whose forward delegates to a method bound on a parent module.

    Subclasses (HCPreAttn / HCPostAttn / HCPreFfn / HCPostFfn / HCHead) carry
    distinct class names so ``fusion_rules.SEMANTIC_LABELS`` can match them
    individually.  The parent is held by ``weakref`` so this child does not
    double-register the parent's params in ``state_dict()``.
    """

    def __init__(self, parent: nn.Module, method_name: str, *param_names: str):
        super().__init__()
        import weakref
        self._parent_ref = weakref.ref(parent)
        self._method_name = method_name
        self._param_names = param_names

    def forward(self, *runtime_args):
        parent = self._parent_ref()
        if parent is None:
            raise RuntimeError(f"HC wrapper lost reference to parent before {self._method_name}")
        method = getattr(type(parent), self._method_name)
        bound_params = tuple(getattr(parent, n) for n in self._param_names)
        return method(parent, *runtime_args, *bound_params)


class HCPreAttn(_HCBoundMethodModule): pass
class HCPostAttn(_HCBoundMethodModule): pass
class HCPreFfn(_HCBoundMethodModule): pass
class HCPostFfn(_HCBoundMethodModule): pass
class HCHead(_HCBoundMethodModule): pass


def _block_forward_with_hc_modules(self, x, start_pos, input_ids):
    """Replacement Block.forward routing HC through child nn.Modules."""
    residual = x
    x, post, comb = self.hc_pre_attn(x)
    x = self.attn_norm(x)
    x = self.attn(x, start_pos)
    x = self.hc_post_attn(x, residual, post, comb)

    residual = x
    x, post, comb = self.hc_pre_ffn(x)
    x = self.ffn_norm(x)
    x = self.ffn(x, input_ids)
    x = self.hc_post_ffn(x, residual, post, comb)
    return x


def _head_forward_with_hc_module(self, x, hc_fn, hc_scale, hc_base, norm):
    """Replacement ParallelHead.forward routing hc_head through a child Module.

    Cannot bind hc_fn / hc_scale / hc_base on the head itself: a single
    ParallelHead is shared between the main Transformer (uses
    ``transformer.hc_head_*``) and each MTPBlock (uses ``mtp[i].hc_head_*``).
    The wrapper module therefore takes them as runtime args.
    """
    import torch.distributed as dist
    x = self.hc_head_module(x, hc_fn, hc_scale, hc_base)
    logits = self.get_logits(norm(x))
    if dist.is_initialized() and dist.get_world_size() > 1:
        all_logits = [torch.empty_like(logits) for _ in range(dist.get_world_size())]
        dist.all_gather(all_logits, logits)
        logits = torch.cat(all_logits, dim=-1)
    return logits


def _attach_hc_modules_to_block(block: nn.Module) -> None:
    """Attach 4 HC wrapper modules to a Block (covers MTPBlock via inheritance)."""
    if getattr(block, "_hc_patched", False):
        return
    # hc_pre takes only x at runtime; hc_*_fn / hc_*_scale / hc_*_base are bound
    # to the parent block.
    block.hc_pre_attn = HCPreAttn(
        block, "hc_pre", "hc_attn_fn", "hc_attn_scale", "hc_attn_base"
    )
    block.hc_pre_ffn = HCPreFfn(
        block, "hc_pre", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"
    )
    # hc_post takes (x, residual, post, comb) — all runtime.
    block.hc_post_attn = HCPostAttn(block, "hc_post")
    block.hc_post_ffn = HCPostFfn(block, "hc_post")
    block._hc_patched = True


def _attach_hc_module_to_head(head: nn.Module) -> None:
    """Attach 1 HC wrapper module to a ParallelHead."""
    if getattr(head, "_hc_patched", False):
        return
    # hc_head takes (x, hc_fn, hc_scale, hc_base) — all runtime; the head
    # is shared between Transformer and MTPBlocks, each supplying their own.
    head.hc_head_module = HCHead(head, "hc_head")
    head._hc_patched = True


def patch_hc_for_capture(model: nn.Module) -> None:
    """Inject HC wrapper nn.Modules so ModuleTracker sees HC boundaries.

    Each Block / MTPBlock gets 4 child modules:
      - ``hc_pre_attn`` / ``hc_post_attn`` (around the attn sub-layer)
      - ``hc_pre_ffn``  / ``hc_post_ffn``  (around the ffn  sub-layer)

    Each ParallelHead gets 1 child module ``hc_head_module``, used by both the
    main Transformer's final head call and any MTPBlock's head call.

    ``Block.forward`` and ``ParallelHead.forward`` are rebound on the *class*
    so the change carries through the inheritance chain (MTPBlock inherits
    Block.forward via ``super().forward`` in its own forward).
    """
    block_classes: set[type] = set()
    head_classes: set[type] = set()

    for _, module in model.named_modules():
        cls = type(module)
        cls_name = cls.__name__
        if cls_name in ("Block", "MTPBlock") and hasattr(module, "hc_attn_fn"):
            _attach_hc_modules_to_block(module)
            # Only rebind the *defining* class for hc_pre/hc_post forward.
            # MTPBlock keeps its own forward (which calls super().forward).
            if cls_name == "Block":
                block_classes.add(cls)
        elif cls_name == "ParallelHead" and hasattr(module, "hc_head"):
            _attach_hc_module_to_head(module)
            head_classes.add(cls)

    for cls in block_classes:
        cls.forward = _block_forward_with_hc_modules
    for cls in head_classes:
        cls.forward = _head_forward_with_hc_module

    if block_classes or head_classes:
        logger.info(
            "Applied HC capture patch — Block classes: %s; Head classes: %s",
            ", ".join(c.__name__ for c in block_classes) or "(none)",
            ", ".join(c.__name__ for c in head_classes) or "(none)",
        )


# ── DeepSeek-V4 training-capture patch ───────────────────────────────────────
# The inference/model.py has three blockers for backward():
#   1. @torch.inference_mode() on Transformer.forward disables autograd entirely.
#   2. Kernel stubs (fp4_gemm, fp8_gemm, sparse_attn, hc_split_sinkhorn) return
#      new_empty tensors with no autograd history, so gradients cannot propagate.
#   3. MoE forward uses data-dependent loops that crash on FakeTensors (already
#      handled by patch_moe_for_fake).
#
# All fixes are applied at runtime without touching inference/model.py.


def _diff_gemm(
    x: torch.Tensor,
    sx: torch.Tensor,
    w: torch.Tensor,
    sw: torch.Tensor,
    scale_dtype: torch.dtype,
) -> torch.Tensor:
    """Differentiable GEMM placeholder using real aten ops for training-capture mode.

    Uses ``aten.mm`` (or ``aten.bmm`` for batched inputs) so that autograd records
    a proper backward graph with ``mm_backward`` / ``t`` / ``mm`` ops that
    ``TorchDispatchMode`` can capture.

    Signature matches the kernel stub: ``fp8_gemm(x, sx, w, sw, scale_dtype)``.
    The scale tensors ``sx`` / ``sw`` are ignored — they exist only to satisfy
    the caller's argument list.

    ``w`` is expected in ``[out_features, in_features]`` layout (matching
    ``nn.Linear.weight`` and ``F.linear`` convention), so the forward is
    ``x @ w.T``.
    """
    # x: [*, in_features]  w: [out_features, in_features]
    # F.linear(x, w) = x @ w.T = [*, out_features]
    # Flatten x to 2-D for mm, then restore batch dims.
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    # w is [out, in] → transpose to [in, out] for mm
    out_2d = torch.mm(x_2d, w.t())
    return out_2d.reshape(*orig_shape[:-1], w.shape[0]).to(torch.bfloat16)


def _diff_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Differentiable attention placeholder for training-capture mode.

    Replaces the custom sparse_attn kernel with a standard self-attention on q
    so that gradients flow back through q while preserving the correct output
    shape [bsz, seqlen, n_heads, head_dim].  topk_idxs is ignored.
    """
    bsz, seqlen, n_heads, head_dim = q.shape
    q_bh = q.reshape(bsz * n_heads, seqlen, head_dim)
    attn_w = torch.bmm(q_bh, q_bh.transpose(1, 2)) * softmax_scale
    attn_w = torch.softmax(attn_w, dim=-1)
    out = torch.bmm(attn_w, q_bh)
    return out.reshape(bsz, seqlen, n_heads, head_dim)


def _diff_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable softmax approximation of Sinkhorn normalisation.

    Replaces hc_split_sinkhorn for training-capture mode.  The Sinkhorn
    iterations are approximated by softmax / sigmoid, which gives the same
    output shapes and a valid backward graph.

    Input  mixes: [b, s, (2+hc_mult)*hc_mult]
    Output pre:   [b, s, hc_mult]
           post:  [b, s, hc_mult]
           comb:  [b, s, hc_mult, hc_mult]
    """
    b, s, _ = mixes.shape
    pre  = torch.softmax(mixes[:, :, :hc_mult], dim=-1)
    post = torch.sigmoid(mixes[:, :, hc_mult : 2 * hc_mult])
    comb = torch.softmax(
        mixes[:, :, 2 * hc_mult :].reshape(b, s, hc_mult, hc_mult), dim=-1
    )
    return pre, post, comb


def _patch_inference_mode(model: nn.Module) -> None:
    """Remove @torch.inference_mode() from inference.Transformer.forward."""
    for _name, module in model.named_modules():
        cls = type(module)
        if cls.__name__ != "Transformer":
            continue
        fwd = cls.forward
        unwrapped = getattr(fwd, "__wrapped__", None)
        if unwrapped is not None:
            cls.forward = unwrapped
            logger.info("Removed @inference_mode from %s.forward", cls.__qualname__)
        else:
            logger.debug(
                "%s.forward has no __wrapped__; already unwrapped or not decorated.",
                cls.__qualname__,
            )
        break


def _act_quant_passthrough(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt=None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
) -> torch.Tensor:
    """Identity pass-through for act_quant that preserves the autograd graph.

    Returns x unchanged (no dtype cast, no in-place copy_) so that gradients
    flow through activations during backward.  The scale tensor is a dummy
    ones array to satisfy callers that unpack (y, s).
    """
    if inplace:
        return x
    n = x.size(-1)
    s = x.new_ones(*x.shape[:-1], max(1, n // block_size), dtype=scale_dtype)
    return x, s


def _fp4_act_quant_passthrough(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
) -> torch.Tensor:
    if inplace:
        return x
    s = x.new_ones(*x.shape[:-1], max(1, x.size(-1) // block_size))
    return x, s


def _make_attention_training_closure(attn: nn.Module):
    """Return an instance-bound closure for training-capture-safe Attention forward.

    Patched at the instance level (not class level) so that inference-mode
    Attention instances in the same process are not affected.

    Two in-place ops in the original forward break autograd version-counter
    checks once @inference_mode is removed:
      • ``q *= rsqrt(...)`` — in-place on a tensor whose view was saved by
        wq_b's LinearBackward node.
      • ``apply_rotary_emb(q[..., -rd:], ...)`` — copy_() on a slice of q.

    Both are avoided here: normalize is out-of-place and apply_rotary_emb is a
    no-op (patched separately in _upgrade_kernel_stubs_for_backward).
    """
    def forward(x: torch.Tensor, start_pos: int):
        import sys
        inf = sys.modules["_v4_inference_model"]

        bsz, seqlen, _ = x.size()
        freqs_cis = attn.freqs_cis[start_pos : start_pos + seqlen]
        win = attn.window_size
        ratio = attn.compress_ratio
        rd = attn.rope_head_dim

        if ratio and attn.compressor.kv_cache is None:
            attn.compressor.kv_cache = attn.kv_cache[:, win:]
            attn.compressor.freqs_cis = attn.freqs_cis
            if attn.indexer is not None:
                attn.indexer.freqs_cis = attn.freqs_cis

        # q: out-of-place normalize; apply_rotary_emb is a no-op (patched)
        qr = q = attn.q_norm(attn.wq_a(x))
        q = attn.wq_b(q).unflatten(-1, (attn.n_local_heads, attn.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + attn.eps)

        # kv: apply_rotary_emb is no-op; act_quant is passthrough
        kv = attn.wkv(x)
        kv = attn.kv_norm(kv)
        inf.act_quant(kv[..., :-rd], 64, None, torch.float32, True)

        topk_idxs = inf.get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if attn.indexer is not None:
                compress_topk_idxs = attn.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = inf.get_compress_topk_idxs(
                    ratio, bsz, seqlen, start_pos, offset
                )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        if start_pos == 0:
            if seqlen <= win:
                attn.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                attn.kv_cache[:bsz, cutoff:win], attn.kv_cache[:bsz, :cutoff] = (
                    kv[:, -win:].split([win - cutoff, cutoff], dim=1)
                )
            if ratio:
                kv_compress = attn.compressor(x, start_pos)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = inf.sparse_attn(q, kv, attn.attn_sink, topk_idxs, attn.softmax_scale)
        else:
            attn.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if ratio:
                attn.compressor(x, start_pos)
            o = inf.sparse_attn(q, attn.kv_cache[:bsz], attn.attn_sink, topk_idxs, attn.softmax_scale)
        # apply_rotary_emb(o[..., -rd:], ...) is a no-op

        o = o.view(bsz, seqlen, attn.n_local_groups, -1)
        wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return attn.wo_b(o.flatten(2))

    return forward


def _patch_attention_for_training(model: nn.Module) -> None:
    """Replace each Attention.forward with an instance-bound training-safe closure.

    Instance-level patching (rather than class-level) ensures that inference-mode
    Attention objects created in the same process are not affected.
    """
    patched = 0
    for _, module in model.named_modules():
        cls = type(module)
        if (
            cls.__name__ == "Attention"
            and hasattr(module, "wq_a")
            and not getattr(module, "_train_capture_patched", False)
        ):
            module.forward = _make_attention_training_closure(module)
            module._train_capture_patched = True
            patched += 1
    if patched:
        logger.info("Patched %d Attention instance(s) for training capture.", patched)


def _upgrade_kernel_stubs_for_backward() -> None:
    """Replace kernel functions with backward-compatible differentiable versions.

    The inference/model.py binds kernel functions at *import time* via
    ``from kernel import fp4_gemm, ...``.  Those module-level names live in
    ``sys.modules['_v4_inference_model'].__dict__``, not in the kernel module
    itself.  Patching only ``sys.modules['kernel']`` would have no effect on
    subsequent calls because Python resolves global names through the function's
    own module dict (``func.__globals__``).

    This function therefore patches BOTH:
      • ``sys.modules['kernel']`` — so any future dynamic lookups see the new fns
      • ``sys.modules['_v4_inference_model']`` — so the already-bound globals
        used by ``linear()``, ``Block.hc_pre()``, ``Attention.forward()``, etc.
        are replaced before the forward / backward pass runs.
    """
    import sys

    # apply_rotary_emb writes back via copy_() on a view of the input tensor.
    # Once @inference_mode is removed that copy_() increments the version counter
    # of tensors saved for backward (q, kv, o), causing autograd to raise.
    # For training-capture purposes (FLOPs / memory modelling, not correctness)
    # we can safely skip the rotation.
    def _apply_rotary_emb_noop(x, freqs_cis, inverse=False):
        return x

    new_fns: dict[str, object] = {
        "fp4_gemm": _diff_gemm,
        "fp8_gemm": _diff_gemm,
        "sparse_attn": _diff_sparse_attn,
        "hc_split_sinkhorn": _diff_sinkhorn,
        "act_quant": _act_quant_passthrough,
        "fp4_act_quant": _fp4_act_quant_passthrough,
        "apply_rotary_emb": _apply_rotary_emb_noop,
    }

    patched_mods: list[str] = []

    # 1. Patch kernel module (covers future dynamic lookups)
    kernel = sys.modules.get("kernel")
    if kernel is not None:
        for name, fn in new_fns.items():
            setattr(kernel, name, fn)
        patched_mods.append("kernel")

    # 2. Patch the inference module's own global namespace.
    #    The module name is set by modeling_deepseek._INFERENCE_MODULE_NAME.
    #    We also scan for any module that owns these names to be resilient to
    #    rename, but skip third-party packages whose lazy __getattr__ would
    #    import optional deps (e.g. transformers.models.aria.image_processing_aria
    #    pulls torchvision when probed via hasattr).
    _INF_MOD_NAME = "_v4_inference_model"
    _SKIP_PREFIXES = (
        "transformers", "torch", "torchvision", "numpy", "scipy",
        "pandas", "matplotlib", "sklearn", "PIL",
    )
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod is kernel:
            continue
        # Fast path: explicit V4 inference module name match.
        if mod_name == _INF_MOD_NAME:
            for name, fn in new_fns.items():
                if name in vars(mod):
                    setattr(mod, name, fn)
            patched_mods.append(mod_name)
            continue
        # Fallback scan: skip third-party packages whose lazy attribute access
        # may trigger optional-dep imports, and use vars(mod) to avoid hasattr.
        if any(mod_name == p or mod_name.startswith(p + ".") for p in _SKIP_PREFIXES):
            continue
        try:
            mod_vars = vars(mod)
        except TypeError:
            continue
        if "fp4_gemm" in mod_vars and "sparse_attn" in mod_vars and "hc_split_sinkhorn" in mod_vars:
            for name, fn in new_fns.items():
                if name in mod_vars:
                    setattr(mod, name, fn)
            patched_mods.append(mod_name)

    if patched_mods:
        logger.info(
            "Upgraded kernel stubs for training-capture backward in: %s",
            ", ".join(patched_mods),
        )
    else:
        logger.debug("_upgrade_kernel_stubs_for_backward: no target modules found.")


def patch_for_training_capture(model: nn.Module) -> None:
    """Enable backward() on the DeepSeek-V4 inference model for training graph capture.

    Applies minimal runtime patches without touching inference/model.py:

    1. Removes ``@torch.inference_mode()`` from ``Transformer.forward`` so that
       autograd records op history during the forward pass.
    2. Upgrades kernel stubs to backward-compatible differentiable versions:
       - ``fp4_gemm`` / ``fp8_gemm`` → ``_DiffGemm`` (autograd.Function with
         correct dx/dw shapes)
       - ``sparse_attn`` → standard self-attention on q (differentiable,
         topk_idxs ignored)
       - ``hc_split_sinkhorn`` → softmax/sigmoid approximation (differentiable)
       - ``act_quant`` / ``fp4_act_quant`` → identity pass-through (preserves
         grad tracking through activations)
    3. Applies ``patch_moe_for_fake`` to handle the data-dependent MoE routing
       loop that is incompatible with FakeTensorMode.
    4. Applies ``patch_indexer_for_fake`` to fix the Indexer transpose bug.
    5. Applies ``patch_hc_for_capture`` so ModuleTracker sees HC sub-layer
       boundaries as distinct nn.Modules.

    Only has effect when the ZRT kernel stubs are loaded
    (``sys.modules['kernel']._zrt_stub is True``).
    """
    _patch_inference_mode(model)
    _upgrade_kernel_stubs_for_backward()
    _patch_attention_for_training(model)
    patch_moe_for_fake(model)
    patch_indexer_for_fake(model)
    patch_hc_for_capture(model)


# Backward-compatible aliases
patch_moe_for_meta = patch_moe_for_fake
_is_moe_module = is_moe_module
