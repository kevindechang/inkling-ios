#!/usr/bin/env python3
"""Stage 3 Working-Set Concentration Distillation (WSD-40) for inkling-ios.

Teacher: frozen full 128-expert 3-pass top-8 TML-d24-varloop-research (GPU int8 via --experts-int8-from-hf).
Student: full 128-expert 1-pass top-4 sharing the teacher's exact int8 base experts:
  - Keeps ALL 128 experts alive (40 RAM-resident HOT experts + 88 flash safety-net COLD experts).
  - Full-rank trainable router weight delta [128, H] + adaptive working-set bias delta [128].
  - Trainable Expert LoRA (r=32, alpha=64) on the 40 HOT experts' gate_up_proj and down_proj.
  - Trainable Attention LoRA (r=32, alpha=64) + lm_head LoRA (r=32, alpha=64).
  - Differentiable cold-mass loss + smoothly annealed PI bias controller targeting ~1.5% cold hits
    (~1.4 cold visits/token -> <=1.2 flash misses/tok -> 21+ tok/s & <1.0s warm TTFT on iPhone 15 Pro).
  - 50/50 packed all-token + completion-only RWT data mix with concise thinking supervision.

Dry run (CPU, ~30s):
    PYTHONPATH=train/.pydeps python3 train/distill_workingset.py --dry-run-tiny
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

log = logging.getLogger("distill_ws")

MODEL_ID = "thinkingmachines/TML-d24-varloop-research"
MODEL_REVISION = "c90921bd5f78df6c8168173b9f8a3a850320c594"

LORA_TARGETS = {
    "self_attn.q_proj.linear": "attn_q",
    "self_attn.k_proj.linear": "attn_k",
    "self_attn.v_proj.linear": "attn_v",
    "self_attn.o_proj.linear": "attn_output",
    "self_attn.o_proj.gate_proj": "attn_gate",
    "self_attn.relative_position_bias.linear": "attn_rel",
    "self_attn.sink_proj": "attn_sink_proj",
    "self_attn.k_proj.sconv.weight_proj": "attn_k_conv_proj",
    "self_attn.v_proj.sconv.weight_proj": "attn_v_conv_proj",
}

END_MESSAGE = "<|end_message|>"
MESSAGE_MODEL = "<|message_model|>"
STREAM_RETRIES = 5
STREAM_BACKOFF_S = 20

IDENTITY_RE = re.compile(
    r"\b(as an ai\b|an ai (language )?model|language model developed|i am an ai|i'm an ai|my name is|"
    r"i am (chatgpt|claude|gpt|llama|tulu|olmo|gemini)|i'm (chatgpt|claude|tulu|olmo))", re.IGNORECASE)
CODING_SOURCE_RE = re.compile(r"(code|python|coder|evol_code|starcoder|humaneval|mbpp|swe)", re.IGNORECASE)
CALC_ANN_RE = re.compile(r"<<[^<>]*>>")

MATH_SUBSETS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)


# --------------------------------------------------------------------------------------------- LoRA & Working-Set MoE

class LoRAState:
    enabled = True
    track_load = False
    collect_cold_loss = False


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not LoRAState.enabled:
            return out
        delta = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B) * self.scale
        return out + delta.to(out.dtype)


def _ws_router_forward(self, hidden_states: torch.Tensor):
    """Teacher (LoRAState.enabled=False): unmodified full-128 top-8 routing.
    Student (LoRAState.enabled=True): full-128 top-4 routing with full-rank weight delta + working-set bias delta."""
    flat = hidden_states.float().reshape(-1, self.hidden_dim)
    if not LoRAState.enabled:
        logits = F.linear(flat, self.weight.float())
        scores = self.score_fn(logits)
        indices = torch.topk(scores + self.e_score_correction_bias.float(), self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices

    w = self.weight.float() + self.student_weight_delta
    b = self.e_score_correction_bias.float() + self.student_bias_delta
    logits = F.linear(flat, w)  # [T, E]
    scores = self.score_fn(logits)  # sigmoid(logits) in (0, 1)
    indices = torch.topk(scores + b, self.top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)

    if LoRAState.collect_cold_loss:
        # Stash detached subsampled inputs so pop_cold_loss() recomputes scores WITH grad outside activation checkpointing
        n_t = flat.shape[0]
        if n_t > 512:
            stride = max(1, n_t // 512)
            self._last_flat = flat[::stride][:512].detach().to(torch.bfloat16)
        else:
            self._last_flat = flat.detach().to(torch.bfloat16)

    if LoRAState.track_load:
        with torch.no_grad():
            counts = torch.bincount(indices.reshape(-1), minlength=self.num_experts).float()
            self._load_accum.add_(counts)
            self._load_tokens += flat.shape[0]
    return logits, weights * self.routed_scaling_factor, indices


def _ws_experts_forward(
    self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
) -> torch.Tensor:
    """Teacher: unmodified eager forward over frozen QuantExperts/Parameters.
    Student: adds per-HOT-expert LoRA deltas on gate_up_proj and down_proj; COLD experts use frozen base weights."""
    final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1).tolist()

    use_lora = LoRAState.enabled and getattr(self, "exp_lora_rank", 0) > 0
    for e_glob in expert_hit:
        if e_glob == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[e_glob])
        current_state = hidden_states[token_idx]
        gu = F.linear(current_state, self.gate_up_proj[e_glob])
        loc = self._g2l_py.get(e_glob, -1) if use_lora else -1
        if loc >= 0:
            gu_d = F.linear(F.linear(current_state.float(), self.gu_lora_A[loc]), self.gu_lora_B[loc]) * self.exp_lora_scale
            gu = gu + gu_d.to(gu.dtype)
        act = self._apply_gate(gu)
        dn = F.linear(act, self.down_proj[e_glob])
        if loc >= 0:
            dn_d = F.linear(F.linear(act.float(), self.dn_lora_A[loc]), self.dn_lora_B[loc]) * self.exp_lora_scale
            dn = dn + dn_d.to(dn.dtype)
        weighted = dn * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, weighted.to(final_hidden_states.dtype))
    return final_hidden_states.to(hidden_states.dtype)


def inject_ws_student(
    text_model: nn.Module, lm_head_owner: nn.Module, hot_experts: list[list[int]], args: argparse.Namespace
) -> list[str]:
    wrapped = []
    # 1. Dense attention LoRA
    for name, module in list(text_model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(suffix) for suffix in LORA_TARGETS):
            continue
        parent_name, child = name.rsplit(".", 1)
        parent = text_model.get_submodule(parent_name)
        lora = LoRALinear(module, args.lora_rank, args.lora_alpha).to(module.weight.device)
        setattr(parent, child, lora)
        wrapped.append(name)
    # 2. LM head LoRA
    if args.lora_lm_head:
        lm_head_owner.lm_head = LoRALinear(lm_head_owner.lm_head, args.lora_rank, args.lora_alpha).to(
            lm_head_owner.lm_head.weight.device
        )
        wrapped.append("lm_head")
    # 3. Full-128 routers with working-set hot/cold masks + Expert LoRA on HOT experts
    for i, layer in enumerate(text_model.layers):
        hot = list(hot_experts[i])
        M = len(hot)
        rtr = layer.mlp.gate
        E = rtr.num_experts
        dev = rtr.weight.device
        hot_mask = torch.zeros(E, dtype=torch.bool, device=dev)
        hot_mask[torch.tensor(hot, dtype=torch.long, device=dev)] = True
        rtr.register_buffer("hot_mask", hot_mask)
        rtr.register_buffer("cold_mask_f", (~hot_mask).float())
        rtr.student_weight_delta = nn.Parameter(torch.zeros(E, rtr.hidden_dim, dtype=torch.float32, device=dev))
        rtr.register_buffer("student_bias_delta", torch.zeros(E, dtype=torch.float32, device=dev))
        rtr.register_buffer("_load_accum", torch.zeros(E, dtype=torch.float32, device=dev))
        rtr._load_tokens = 0
        rtr._last_flat = None
        rtr.forward = MethodType(_ws_router_forward, rtr)

        exp = layer.mlp.experts
        exp._g2l_py = {e_glob: loc for loc, e_glob in enumerate(hot)}
        exp.exp_lora_rank = args.expert_lora_rank
        exp.exp_lora_scale = (args.expert_lora_alpha / args.expert_lora_rank) if args.expert_lora_rank > 0 else 0.0
        if args.expert_lora_rank > 0:
            r = args.expert_lora_rank
            h, inter = exp.hidden_dim, exp.intermediate_dim
            gu_A = torch.empty(M, r, h, dtype=torch.float32, device=dev)
            dn_A = torch.empty(M, r, inter, dtype=torch.float32, device=dev)
            for loc in range(M):
                nn.init.kaiming_uniform_(gu_A[loc], a=math.sqrt(5))
                nn.init.kaiming_uniform_(dn_A[loc], a=math.sqrt(5))
            exp.gu_lora_A = nn.Parameter(gu_A)
            exp.gu_lora_B = nn.Parameter(torch.zeros(M, 2 * inter, r, dtype=torch.float32, device=dev))
            exp.dn_lora_A = nn.Parameter(dn_A)
            exp.dn_lora_B = nn.Parameter(torch.zeros(M, h, r, dtype=torch.float32, device=dev))
        exp.forward = MethodType(_ws_experts_forward, exp)
    return wrapped


# ------------------------------------------------------------------------------------ model access

@dataclass(frozen=True)
class Parts:
    full: nn.Module
    text: nn.Module
    head_owner: nn.Module
    loop_start: int
    loop_end: int


def get_parts(model: nn.Module) -> Parts:
    text = model.model.language_model if hasattr(model.model, "language_model") else model.model
    cfg = text.config
    return Parts(model, text, model, cfg.loop_layer_index_start, cfg.loop_layer_index_end)


def routers(parts: Parts) -> list[nn.Module]:
    return [layer.mlp.gate for layer in parts.text.layers]


def set_resting_shape(parts: Parts, passes: int, top_k: int) -> None:
    parts.text.config.loop_num_passes = passes
    for r in routers(parts):
        r.top_k = top_k
    LoRAState.enabled = True


@contextlib.contextmanager
def shape(parts: Parts, passes: int, top_k: int, lora: bool, track_load: bool = False, collect_cold_loss: bool = False):
    cfg = parts.text.config
    old = (cfg.loop_num_passes, [r.top_k for r in routers(parts)], LoRAState.enabled, LoRAState.track_load, LoRAState.collect_cold_loss)
    cfg.loop_num_passes = passes
    for r in routers(parts):
        r.top_k = top_k
    LoRAState.enabled = lora
    LoRAState.track_load = track_load
    LoRAState.collect_cold_loss = collect_cold_loss
    try:
        yield
    finally:
        cfg.loop_num_passes = old[0]
        for r, k in zip(routers(parts), old[1]):
            r.top_k = k
        LoRAState.enabled = old[2]
        LoRAState.track_load = old[3]
        LoRAState.collect_cold_loss = old[4]


def current_cold_target(step: int, args) -> float:
    if args.cold_ramp_steps <= 0 or step >= args.cold_ramp_steps:
        return args.target_cold_frac
    prog = step / float(args.cold_ramp_steps)
    # Smooth cosine ramp from start_cold_frac -> target_cold_frac
    return args.target_cold_frac + (args.start_cold_frac - args.target_cold_frac) * 0.5 * (1.0 + math.cos(math.pi * prog))


def current_cold_weight(step: int, args) -> float:
    if args.cold_loss <= 0:
        return 0.0
    if args.cold_loss_warmup <= 0 or step >= args.cold_loss_warmup:
        return args.cold_loss
    return args.cold_loss * ((step + 1) / float(args.cold_loss_warmup))


def update_router_workingset(parts: Parts, step: int, args) -> dict[str, float]:
    """Aux-loss-free load balance among HOT experts + PI bias controller on COLD experts."""
    tgt_cold = current_cold_target(step, args)
    cold_fracs, hot_imbals, min_hot_active, cold_biases = [], [], 10_000, []
    with torch.no_grad():
        for r in routers(parts):
            if r._load_tokens <= 0:
                continue
            tot_visits = float(r._load_tokens * r.top_k)
            hot_counts = r._load_accum[r.hot_mask]
            cold_counts = r._load_accum[~r.hot_mask]
            cold_frac = float(cold_counts.sum().item()) / max(tot_visits, 1.0)
            cold_fracs.append(cold_frac)

            # 1. Keep all HOT experts balanced and zero-centered
            M = int(r.hot_mask.sum().item())
            if args.balance_gamma > 0 and M > 0:
                target_h = (tot_visits * (1.0 - cold_frac)) / float(M)
                hot_b = r.student_bias_delta[r.hot_mask]
                hot_b = hot_b + torch.sign(target_h - hot_counts) * args.balance_gamma
                hot_b = (hot_b - hot_b.mean()).clamp(-0.5, 0.5)
                r.student_bias_delta[r.hot_mask] = hot_b
                hot_imbals.append(float((hot_counts.max() / max(target_h, 1e-6)).item()))
                min_hot_active = min(min_hot_active, int((hot_counts > 0).sum().item()))

            # 2. PI controller on COLD experts' bias to hit tgt_cold smoothly
            if args.cold_bias_gamma > 0:
                cold_b = r.student_bias_delta[~r.hot_mask]
                if cold_frac > tgt_cold:
                    # Proportional step scaled by how far above target we are, capped at 3x gamma
                    scale = min(3.0, max(0.5, (cold_frac - tgt_cold) / max(tgt_cold, 0.02)))
                    cold_b = (cold_b - args.cold_bias_gamma * scale).clamp(-2.5, 0.0)
                elif cold_frac < tgt_cold * 0.65:
                    cold_b = (cold_b + args.cold_bias_gamma * 0.5).clamp(-2.5, 0.0)
                r.student_bias_delta[~r.hot_mask] = cold_b
                cold_biases.append(float(cold_b.mean().item()))

            r._load_accum.zero_()
            r._load_tokens = 0
    if not cold_fracs:
        return {}
    mean_cf = sum(cold_fracs) / len(cold_fracs)
    return {
        "cold_pct": round(100.0 * mean_cf, 2),
        "cold_per_tok": round(mean_cf * len(routers(parts)) * args.student_topk, 2),
        "tgt_cold_pct": round(100.0 * tgt_cold, 2),
        "cold_bias": round(sum(cold_biases) / max(len(cold_biases), 1), 3),
        "hot_imbal": round(max(hot_imbals) if hot_imbals else 0.0, 2),
        "hot_min_act": float(min_hot_active if min_hot_active < 10_000 else 0),
    }


def head_logits(parts: Parts, hidden: torch.Tensor, use_lora: bool) -> torch.Tensor:
    head = parts.head_owner.lm_head
    if isinstance(head, LoRALinear) and not use_lora:
        logits = head.base(hidden).float()
    else:
        logits = head(hidden).float()
    return parts.head_owner.logit_softcap(logits)


def freeze_and_select(parts: Parts) -> dict[str, nn.Parameter]:
    for p in parts.full.parameters():
        p.requires_grad_(False)
    trainable = {}
    for name, p in parts.full.named_parameters():
        if any(tag in name for tag in ("lora_", "student_weight_delta")):
            p.requires_grad_(True)
            trainable[name] = p
    return trainable


class QuantExperts:
    """Frozen full-128 expert stack held as per-row int8 + bf16 scale on GPU (~24.5 GB total)."""

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
        self.q, self.scale, self.dtype = q, scale, dtype
        self.shape = q.shape

    def __getitem__(self, idx) -> torch.Tensor:
        return self.q[idx].to(self.dtype) * self.scale[idx]


def quantize_hf_experts_int8(parts: Parts, device: str | torch.device) -> float:
    worst_rel = 0.0
    for i, layer in enumerate(parts.text.layers):
        experts = layer.mlp.experts
        experts.config._experts_implementation = "eager"
        for attr in ("gate_up_proj", "down_proj"):
            param = getattr(experts, attr)
            dtype = param.dtype
            n_exp, rows, cols = param.shape
            q = torch.empty((n_exp, rows, cols), dtype=torch.int8, device=device)
            sc = torch.empty((n_exp, rows, 1), dtype=dtype, device=device)
            with torch.no_grad():
                for e in range(n_exp):
                    w = param.data[e].float()
                    rs = w.abs().amax(dim=-1, keepdim=True) / 127.0
                    rs.clamp_(min=1e-12)
                    wq = torch.round(w / rs).clamp(-127, 127).to(torch.int8)
                    if e == 0:
                        err = wq.float() * rs - w
                        rel = float(err.pow(2).mean().sqrt() / w.pow(2).mean().sqrt().clamp_min(1e-12))
                        worst_rel = max(worst_rel, rel)
                    q[e].copy_(wq, non_blocking=True)
                    sc[e].copy_(rs.to(dtype), non_blocking=True)
            del experts._parameters[attr]
            setattr(experts, attr, QuantExperts(q, sc, dtype))
        if (i + 1) % 6 == 0 or i == len(parts.text.layers) - 1:
            log.info("quantized HF experts up to layer %d to int8 on %s (worst rel rms %.4f)", i, device, worst_rel)
    return worst_rel


def expert_digest(parts: Parts) -> str:
    h = hashlib.sha256()
    for layer in parts.text.layers:
        for t in (layer.mlp.experts.gate_up_proj, layer.mlp.experts.down_proj):
            if isinstance(t, QuantExperts):
                h.update(t.q[0].cpu().numpy().tobytes()[:1 << 14])
                h.update(str(int(t.q.sum(dtype=torch.int64))).encode())
                continue
            h.update(t.detach().float().cpu().numpy().tobytes()[:1 << 14])
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------------- forward + loss

class MultiLayerRecorder:
    def __init__(self, layers: nn.ModuleList, tap_indices: list[int]):
        self.tap_indices = [i for i in tap_indices if 0 <= i < len(layers)]
        self.values: dict[int, torch.Tensor] = {}
        self.active = False
        self.handles = []
        for idx in self.tap_indices:
            self.handles.append(layers[idx].register_forward_hook(self._make_hook(idx)))

    def _make_hook(self, idx: int):
        def _hook(module, inputs, output):
            if self.active:
                self.values[idx] = output[0] if isinstance(output, tuple) else output
        return _hook

    def pop(self) -> dict[int, torch.Tensor]:
        out = dict(self.values)
        self.values.clear()
        return out


def chunked_kl_ce(
    parts: Parts,
    s_h: torch.Tensor,
    t_h: torch.Tensor,
    chunk: int,
    temp: float,
    with_reverse: bool = False,
    targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def one(s_part: torch.Tensor, t_part: torch.Tensor, tgt_part: torch.Tensor | None) -> torch.Tensor:
        with torch.no_grad():
            t_logp = F.log_softmax(head_logits(parts, t_part, use_lora=False) / temp, dim=-1)
        s_logits = head_logits(parts, s_part, use_lora=True)
        s_logp_t = F.log_softmax(s_logits / temp, dim=-1)
        forward = (t_logp.exp() * (t_logp - s_logp_t)).sum() * (temp * temp)
        reverse = ((s_logp_t.exp() * (s_logp_t - t_logp)).sum() * (temp * temp)) if with_reverse else forward.new_zeros(())
        if tgt_part is not None:
            s_logp1 = s_logp_t if temp == 1.0 else F.log_softmax(s_logits, dim=-1)
            ce = F.nll_loss(s_logp1, tgt_part, ignore_index=-100, reduction="sum")
        else:
            ce = forward.new_zeros(())
        return torch.stack([forward, reverse, ce])

    total = s_h.new_zeros((3,), dtype=torch.float32)
    n_tok = s_h.shape[0]
    n_ce = int((targets != -100).sum().item()) if targets is not None else 0
    for start in range(0, n_tok, chunk):
        s_part = s_h[start:start + chunk]
        t_part = t_h[start:start + chunk]
        tgt_part = targets[start:start + chunk] if targets is not None else None
        if tgt_part is not None:
            total = total + checkpoint(one, s_part, t_part, tgt_part, use_reentrant=False)
        else:
            total = total + checkpoint(lambda a, b: one(a, b, None), s_part, t_part, use_reentrant=False)
    fwd = total[0] / max(n_tok, 1)
    rev = total[1] / max(n_tok, 1)
    ce = total[2] / max(n_ce, 1) if n_ce > 0 else total[2] * 0.0
    return fwd, rev, ce


@dataclass(frozen=True)
class RolloutBatch:
    s_ids: torch.Tensor
    s_attn: torch.Tensor
    s_mask: torch.Tensor
    s_targets: torch.Tensor
    t_ids: torch.Tensor
    t_attn: torch.Tensor
    t_mask: torch.Tensor
    has_gold_ce: bool


def teacher_forward(
    parts: Parts, ids: torch.Tensor, args, recorder: MultiLayerRecorder, attention_mask: torch.Tensor | None = None
):
    recorder.values.clear()
    recorder.active = True
    try:
        with torch.no_grad(), shape(parts, args.teacher_passes, args.teacher_topk, lora=False, track_load=False, collect_cold_loss=False):
            out = parts.text(input_ids=ids, attention_mask=attention_mask, use_cache=False)
            taps = {k: v.detach() for k, v in recorder.pop().items()}
    finally:
        recorder.active = False
    return out.last_hidden_state.detach(), taps


def layer_mse_loss(
    s_taps: dict[int, torch.Tensor], t_taps: dict[int, torch.Tensor],
    s_mask: torch.Tensor | None = None, t_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    terms = []
    for k in sorted(t_taps):
        if k not in s_taps:
            continue
        sh = s_taps[k][s_mask].float() if s_mask is not None else s_taps[k].float()
        th = t_taps[k][t_mask].float() if t_mask is not None else t_taps[k].float()
        diff = (sh - th).pow(2).sum(-1)
        rel = (diff / th.pow(2).sum(-1).clamp_min(1e-6)).mean()
        terms.append(rel)
    if not terms:
        return None
    return sum(terms) / len(terms)


def pop_cold_loss(parts: Parts) -> torch.Tensor | None:
    terms = []
    for r in routers(parts):
        if r._last_flat is not None:
            flat = r._last_flat.float()
            r._last_flat = None
            logits = F.linear(flat, r.weight.float() + r.student_weight_delta)
            scores = r.score_fn(logits)
            cold_mass = (scores * r.cold_mask_f).sum(dim=-1)
            tot_mass = scores.sum(dim=-1).clamp_min(1e-6)
            terms.append((cold_mass / tot_mass).mean())
    if not terms:
        return None
    return sum(terms) / len(terms)


def distill_loss(
    parts: Parts, batch: torch.Tensor | RolloutBatch, args, recorder: MultiLayerRecorder, cold_w: float
) -> tuple[torch.Tensor, dict]:
    collect_cold = cold_w > 0.0
    if isinstance(batch, RolloutBatch):
        t_last, t_taps = teacher_forward(parts, batch.t_ids, args, recorder, attention_mask=batch.t_attn)
        recorder.values.clear()
        recorder.active = True
        LoRAState.track_load = True
        LoRAState.collect_cold_loss = collect_cold
        try:
            s_last = parts.text(input_ids=batch.s_ids, attention_mask=batch.s_attn, use_cache=False).last_hidden_state
            s_taps = recorder.pop()
            closs = pop_cold_loss(parts) if collect_cold else None
        finally:
            recorder.active = False
            LoRAState.track_load = False
            LoRAState.collect_cold_loss = False
        s_h, t_h = s_last[batch.s_mask], t_last[batch.t_mask]
        tgts = batch.s_targets[batch.s_mask] if (batch.has_gold_ce and args.sft_ce > 0) else None
        kl, rev, ce = chunked_kl_ce(
            parts, s_h, t_h, args.logit_chunk, args.kl_temp,
            with_reverse=(args.rollout_reverse_kl > 0), targets=tgts,
        )
        loss = args.rollout_forward_kl * kl + args.rollout_reverse_kl * rev
        stats = {"ro_kl": float(kl.detach()), "ro_rev_kl": float(rev.detach())}
        if tgts is not None:
            loss = loss + args.sft_ce * ce
            stats["sft_ce"] = float(ce.detach())
        if args.layer_mse > 0:
            lmse = layer_mse_loss(s_taps, t_taps, batch.s_mask, batch.t_mask)
            if lmse is not None:
                loss = loss + args.layer_mse * lmse
                stats["layer_mse"] = float(lmse.detach())
        if closs is not None:
            loss = loss + cold_w * closs
            stats["cold_mass"] = float(closs.detach())
        return loss, stats

    t_last, t_taps = teacher_forward(parts, batch, args, recorder)
    recorder.values.clear()
    recorder.active = True
    LoRAState.track_load = True
    LoRAState.collect_cold_loss = collect_cold
    try:
        s_last = parts.text(input_ids=batch, use_cache=False).last_hidden_state
        s_taps = recorder.pop()
        closs = pop_cold_loss(parts) if collect_cold else None
    finally:
        recorder.active = False
        LoRAState.track_load = False
        LoRAState.collect_cold_loss = False
    hdim = s_last.shape[-1]
    kl, rev, _ = chunked_kl_ce(
        parts, s_last.reshape(-1, hdim), t_last.reshape(-1, hdim),
        args.logit_chunk, args.kl_temp, with_reverse=(args.reverse_kl > 0), targets=None,
    )
    loss = kl + (args.reverse_kl * rev if args.reverse_kl > 0 else 0.0)
    stats = {"kl": float(kl.detach())}
    if args.reverse_kl > 0:
        stats["rev_kl"] = float(rev.detach())
    if args.layer_mse > 0:
        lmse = layer_mse_loss(s_taps, t_taps)
        if lmse is not None:
            loss = loss + args.layer_mse * lmse
            stats["layer_mse"] = float(lmse.detach())
    if closs is not None:
        loss = loss + cold_w * closs
        stats["cold_mass"] = float(closs.detach())
    return loss, stats


@torch.no_grad()
def evaluate(
    parts: Parts, packed_eval: list[torch.Tensor], ro_eval: list[RolloutBatch], args, recorder: MultiLayerRecorder
) -> dict:
    was_training = parts.full.training
    parts.full.eval()
    kls, lmses, ro_kls = [], [], []
    # Reset temp load accumulators cleanly (preserve none, eval doesn't update bias)
    for r in routers(parts):
        r._load_accum.zero_()
        r._load_tokens = 0
    for ids in packed_eval:
        t_last, t_taps = teacher_forward(parts, ids, args, recorder)
        recorder.values.clear()
        recorder.active = True
        try:
            with shape(parts, args.student_passes, args.student_topk, lora=True, track_load=True, collect_cold_loss=False):
                s_last = parts.text(input_ids=ids, use_cache=False).last_hidden_state
                s_taps = recorder.pop()
                hdim = s_last.shape[-1]
                kl, _, _ = chunked_kl_ce(parts, s_last.reshape(-1, hdim), t_last.reshape(-1, hdim), args.logit_chunk, 1.0)
                kls.append(float(kl))
                lm = layer_mse_loss(s_taps, t_taps)
                if lm is not None:
                    lmses.append(float(lm))
        finally:
            recorder.active = False
    for rb in ro_eval:
        t_last, _ = teacher_forward(parts, rb.t_ids, args, recorder, attention_mask=rb.t_attn)
        with shape(parts, args.student_passes, args.student_topk, lora=True, track_load=True, collect_cold_loss=False):
            s_last = parts.text(input_ids=rb.s_ids, attention_mask=rb.s_attn, use_cache=False).last_hidden_state
            kl, _, _ = chunked_kl_ce(parts, s_last[rb.s_mask], t_last[rb.t_mask], args.logit_chunk, 1.0)
            ro_kls.append(float(kl))
    cold_fracs = []
    for r in routers(parts):
        if r._load_tokens > 0:
            tot = float(r._load_tokens * r.top_k)
            cold_fracs.append(float(r._load_accum[~r.hot_mask].sum().item()) / max(tot, 1.0))
        r._load_accum.zero_()
        r._load_tokens = 0
    parts.full.train(was_training)
    mean_cf = sum(cold_fracs) / max(len(cold_fracs), 1)
    out = {
        "eval_kl": round(sum(kls) / max(len(kls), 1), 4),
        "eval_ro_kl": round(sum(ro_kls) / max(len(ro_kls), 1), 4) if ro_kls else 0.0,
        "eval_layer_mse": round(sum(lmses) / max(len(lmses), 1), 4) if lmses else 0.0,
        "eval_cold_pct": round(100.0 * mean_cf, 2),
        "eval_cold_per_tok": round(mean_cf * len(routers(parts)) * args.student_topk, 2),
    }
    return out


@torch.no_grad()
def greedy_samples(parts: Parts, tokenizer, prompts: list[str], args, device) -> list[dict]:
    out = []
    was_training = parts.full.training
    parts.full.eval()
    for prompt in prompts:
        if tokenizer is None:
            ids = torch.tensor([[int(t) for t in prompt.split()]], device=device)
        else:
            text = (
                f"<|message_system|><|content_text|>Thinking effort level: 0.9{END_MESSAGE}"
                f"<|message_user|><|content_text|>{prompt}{END_MESSAGE}"
                f"{MESSAGE_MODEL}<|content_thinking|>"
            )
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with shape(parts, args.student_passes, args.student_topk, lora=True, track_load=False, collect_cold_loss=False):
            gen = parts.full.generate(input_ids=ids, max_new_tokens=args.gen_tokens, do_sample=False)
        new = gen[0, ids.shape[1]:].tolist()
        out.append({"prompt": prompt, "student": tokenizer.decode(new, skip_special_tokens=False) if tokenizer else new})
    parts.full.train(was_training)
    return out


# --------------------------------------------------------------------------------------------- RWT Data Mix

def _extract_boxed(sol: str) -> str:
    idx = sol.rfind("\\boxed{")
    if idx < 0:
        return sol.strip().splitlines()[-1].strip()
    depth, out = 0, []
    for ch in sol[idx + 7:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                break
            depth -= 1
        out.append(ch)
    return "".join(out).strip()


def _format_gsm8k_row(row: dict) -> dict | None:
    q = str(row.get("question", "")).strip()
    a = str(row.get("answer", "")).strip()
    if not q or "####" not in a:
        return None
    steps, final = a.rsplit("####", 1)
    steps = CALC_ANN_RE.sub("", steps).strip()
    final = final.strip()
    if not steps or not final:
        return None
    text = (
        f"<|message_system|><|content_text|>Thinking effort level: 0.9{END_MESSAGE}"
        f"<|message_user|><|content_text|>{q}\n\nEnd with: Answer: <final answer as a number>{END_MESSAGE}"
        f"{MESSAGE_MODEL}<|content_thinking|>{steps}{END_MESSAGE}"
        f"{MESSAGE_MODEL}<|content_text|>Answer: {final}{END_MESSAGE}"
    )
    return {"text": text, "is_gold": True, "has_thinking": True}


def _format_math_row(row: dict) -> dict | None:
    prob = str(row.get("problem", "")).strip()
    sol = str(row.get("solution", "")).strip()
    if not prob or not sol:
        return None
    ans = _extract_boxed(sol)
    text = (
        f"<|message_system|><|content_text|>Thinking effort level: 0.9{END_MESSAGE}"
        f"<|message_user|><|content_text|>{prob}{END_MESSAGE}"
        f"{MESSAGE_MODEL}<|content_thinking|>{sol}{END_MESSAGE}"
        f"{MESSAGE_MODEL}<|content_text|>Answer: {ans}{END_MESSAGE}"
    )
    return {"text": text, "is_gold": True, "has_thinking": True}


def _format_hermes_row(row: dict) -> dict | None:
    convs = row.get("conversations") or []
    if len(convs) < 2:
        return None
    sys_parts = ["Thinking effort level: 0.0"]
    turns = []
    for m in convs:
        fr = str(m.get("from", "")).lower()
        val = str(m.get("value", "")).strip()
        if not val:
            continue
        if fr == "system":
            sys_parts.append(val)
        elif fr in ("human", "user"):
            turns.append(f"<|message_user|><|content_text|>{val}{END_MESSAGE}")
        elif fr in ("gpt", "assistant"):
            turns.append(f"{MESSAGE_MODEL}<|content_text|>{val}{END_MESSAGE}")
        elif fr == "tool":
            turns.append(f"<|message_tool|><|content_text|>{val}{END_MESSAGE}")
    if not any(t.startswith(MESSAGE_MODEL) for t in turns):
        return None
    text = f"<|message_system|><|content_text|>{chr(10).join(sys_parts)}{END_MESSAGE}" + "".join(turns)
    return {"text": text, "is_gold": True, "has_thinking": False}


def _stream_gsm8k(seed: int, holdout: int = 100) -> Iterator[dict]:
    from datasets import load_dataset
    ds = list(load_dataset("openai/gsm8k", "main", split="train"))
    train_rows = ds[:-holdout] if len(ds) > holdout * 2 else ds
    rng = random.Random(seed)
    while True:
        rng.shuffle(train_rows)
        for r in train_rows:
            out = _format_gsm8k_row(r)
            if out is not None:
                yield out


def _stream_math(seed: int, holdout_per_sub: int = 15) -> Iterator[dict]:
    from datasets import load_dataset
    train_rows = []
    for sub in MATH_SUBSETS:
        ds = list(load_dataset("EleutherAI/hendrycks_math", sub, split="train"))
        train_rows.extend(ds[:-holdout_per_sub] if len(ds) > holdout_per_sub * 2 else ds)
    rng = random.Random(seed + 17)
    while True:
        rng.shuffle(train_rows)
        for r in train_rows:
            out = _format_math_row(r)
            if out is not None:
                yield out


def _stream_hermes(seed: int) -> Iterator[dict]:
    from datasets import load_dataset
    rows = []
    for cfg in ("func_calling_singleturn", "glaive_func_calling"):
        try:
            ds = load_dataset("NousResearch/hermes-function-calling-v1", cfg, split="train")
            for idx, r in enumerate(ds):
                if idx >= 8000:
                    break
                rows.append(r)
        except Exception as err:
            log.warning("hermes cfg %s skipped: %s", cfg, err)
    rng = random.Random(seed + 31)
    while True:
        rng.shuffle(rows)
        for r in rows:
            out = _format_hermes_row(r)
            if out is not None:
                yield out


def _stream_tulu_noncode(seed: int) -> Iterator[dict]:
    from datasets import load_dataset
    epoch = 0
    while True:
        try:
            ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True).shuffle(
                seed=seed + 53 + epoch, buffer_size=5000
            )
            for row in ds:
                src = str(row.get("source", ""))
                if "hardcoded" in src.lower() or CODING_SOURCE_RE.search(src):
                    continue
                msgs = [
                    {"role": m["role"], "content": m["content"]}
                    for m in row.get("messages", [])
                    if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
                ]
                if len(msgs) < 2 or not any(m["role"] == "assistant" for m in msgs):
                    continue
                if any(m["role"] == "assistant" and IDENTITY_RE.search(m["content"]) for m in msgs):
                    continue
                yield {"messages": msgs}
        except Exception as err:
            log.warning("tulu stream retry: %s", err)
            time.sleep(STREAM_BACKOFF_S)
        epoch += 1


def _stream_local_jsonl(path: str, seed: int, is_gold: bool = False) -> Iterator[dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rng = random.Random(seed + 71)
    while True:
        rng.shuffle(rows)
        for r in rows:
            txt = r.get("text")
            if isinstance(txt, str) and MESSAGE_MODEL in txt:
                yield {"text": txt, "is_gold": is_gold, "has_thinking": "<|content_thinking|>" in txt}


def prepare_rollout(text: str, is_gold: bool, has_thinking: bool, tokenizer, seq_len: int, teacher_note: str):
    split_at = text.find(MESSAGE_MODEL)
    if split_at < 0:
        return None
    prefix_end = split_at + len(MESSAGE_MODEL)
    s_prefix, completion = text[:prefix_end], text[prefix_end:]
    if not completion:
        return None
    # Only inject concise-thinking note on thinking rows (never corrupt effort=0.0 tool calls!)
    use_note = teacher_note if (teacher_note and has_thinking) else ""
    t_prefix = (
        f"<|message_system|><|content_text|>{use_note}{END_MESSAGE}{s_prefix}"
        if use_note else s_prefix
    )
    s_pre = tokenizer(s_prefix, add_special_tokens=False).input_ids
    t_pre = s_pre if not use_note else tokenizer(t_prefix, add_special_tokens=False).input_ids
    comp = tokenizer(completion, add_special_tokens=False).input_ids
    avail = seq_len - len(s_pre)
    if not s_pre or not t_pre or avail <= 4:
        return None
    comp = comp[:avail]
    return s_pre + comp, t_pre + comp, len(s_pre) - 1, len(t_pre) - 1, len(comp), is_gold


def collate_rollouts(items, pad_id: int, device) -> RolloutBatch:
    bsz = len(items)
    max_s = max(len(x[0]) for x in items)
    max_t = max(len(x[1]) for x in items)
    s_ids = torch.full((bsz, max_s), pad_id, dtype=torch.long)
    t_ids = torch.full((bsz, max_t), pad_id, dtype=torch.long)
    s_attn = torch.zeros((bsz, max_s), dtype=torch.long)
    t_attn = torch.zeros((bsz, max_t), dtype=torch.long)
    s_mask = torch.zeros((bsz, max_s), dtype=torch.bool)
    t_mask = torch.zeros((bsz, max_t), dtype=torch.bool)
    s_targets = torch.full((bsz, max_s), -100, dtype=torch.long)
    any_gold = False
    for b, (si, ti, ss, ts, nc, is_gold) in enumerate(items):
        s_ids[b, :len(si)] = torch.tensor(si, dtype=torch.long)
        t_ids[b, :len(ti)] = torch.tensor(ti, dtype=torch.long)
        s_attn[b, :len(si)] = 1
        t_attn[b, :len(ti)] = 1
        s_mask[b, ss:ss + nc] = True
        t_mask[b, ts:ts + nc] = True
        if is_gold:
            any_gold = True
            s_targets[b, ss:ss + nc] = torch.tensor(si[ss + 1:ss + nc + 1], dtype=torch.long)
    return RolloutBatch(
        s_ids.to(device), s_attn.to(device), s_mask.to(device), s_targets.to(device),
        t_ids.to(device), t_attn.to(device), t_mask.to(device), any_gold,
    )


def rollout_batches(tokenizer, streams, weights, seq_len: int, batch: int, seed: int, teacher_note: str, pad_id: int, device):
    picker = random.Random(seed + 1)
    cur, skipped = [], 0
    while True:
        row = next(picker.choices(streams, weights=weights)[0])
        item = prepare_rollout(
            row["text"], bool(row.get("is_gold", False)), bool(row.get("has_thinking", False)),
            tokenizer, seq_len, teacher_note,
        )
        if item is None:
            skipped += 1
            if skipped > 500:
                raise RuntimeError("rollout_batches skipped > 500 rows consecutively")
            continue
        skipped = 0
        cur.append(item)
        if len(cur) == batch:
            yield collate_rollouts(cur, pad_id, device)
            cur = []


def packed_mixed_batches(tokenizer, streams, weights, seq_len: int, batch: int, seed: int, device):
    """Packs Tulu-3 non-code + GSM8K + MATH + Hermes into full seq_len all-token batches."""
    picker = random.Random(seed + 13)
    buf, cur = [], []
    while True:
        row = next(picker.choices(streams, weights=weights)[0])
        if "messages" in row:
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False, reasoning_effort=0.0)
        else:
            text = row["text"]
        buf.extend(tokenizer(text, add_special_tokens=False).input_ids)
        while len(buf) >= seq_len:
            cur.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(cur) == batch:
                yield torch.tensor(cur, dtype=torch.long, device=device)
                cur = []


def build_real_data_iters(tokenizer, args, pad_id: int):
    gsm_s = _stream_gsm8k(args.seed, holdout=100)
    math_s = _stream_math(args.seed, holdout_per_sub=15)
    hermes_s = _stream_hermes(args.seed)
    tulu_s = _stream_tulu_noncode(args.seed)

    ro_streams = [gsm_s, math_s, hermes_s]
    ro_weights = [0.30, 0.30, 0.22]
    if args.rollouts_jsonl and Path(args.rollouts_jsonl).is_file():
        ro_streams.append(_stream_local_jsonl(args.rollouts_jsonl, args.seed, is_gold=False))
        ro_weights.append(0.18)
    ro_it = rollout_batches(tokenizer, ro_streams, ro_weights, args.seq_len, args.batch, args.seed, args.teacher_note, pad_id, args.device)

    ord_streams = [tulu_s, _stream_gsm8k(args.seed + 101, holdout=100), _stream_math(args.seed + 101, holdout_per_sub=15), _stream_hermes(args.seed + 101)]
    ord_weights = [0.50, 0.20, 0.20, 0.10]
    ord_it = packed_mixed_batches(tokenizer, ord_streams, ord_weights, args.seq_len, args.batch, args.seed, args.device)

    def mixed():
        rng = random.Random(args.seed + 99)
        while True:
            src = ro_it if rng.random() < args.rollout_frac else ord_it
            for _ in range(args.grad_accum):
                yield next(src)

    # Held-out reasoning eval sets (tail of train splits — ZERO test overlap!)
    from datasets import load_dataset
    eval_rows = []
    for r in list(load_dataset("openai/gsm8k", "main", split="train"))[-64:]:
        f = _format_gsm8k_row(r)
        if f:
            eval_rows.append(f)
    for r in list(load_dataset("EleutherAI/hendrycks_math", "algebra", split="train"))[-15:]:
        f = _format_math_row(r)
        if f:
            eval_rows.append(f)

    ev_buf, ev_batches = [], []
    for row in eval_rows:
        ev_buf.extend(tokenizer(row["text"], add_special_tokens=False).input_ids)
        while len(ev_buf) >= args.seq_len and len(ev_batches) * args.batch < args.eval_seqs:
            if len(ev_batches) == 0 or len(ev_batches[-1]) == args.batch:
                ev_batches.append([])
            ev_batches[-1].append(ev_buf[:args.seq_len])
            ev_buf = ev_buf[args.seq_len:]
    packed_eval = [torch.tensor(b, dtype=torch.long, device=args.device) for b in ev_batches if len(b) == args.batch]

    # Completion-only held-out rollout eval batches (no teacher_note so pure teacher-vs-student KL!)
    ro_ev_items, ro_eval = [], []
    for row in eval_rows:
        it = prepare_rollout(row["text"], True, True, tokenizer, args.seq_len, "")
        if it is not None:
            ro_ev_items.append(it)
            if len(ro_ev_items) == args.batch:
                ro_eval.append(collate_rollouts(ro_ev_items, pad_id, args.device))
                ro_ev_items = []
                if len(ro_eval) * args.batch >= args.eval_seqs:
                    break
    log.info("built 50/50 RWT data iter + %d packed eval + %d completion eval batches", len(packed_eval), len(ro_eval))
    return mixed(), packed_eval, ro_eval


# ---------------------------------------------------------------------------------------- tiny dry-run

class _TinyTokenizer:
    pad_token_id = 298
    SPECIAL = {
        "<|message_system|>": 290, "<|message_user|>": 291, MESSAGE_MODEL: 292,
        "<|content_text|>": 293, "<|content_thinking|>": 294, END_MESSAGE: 295,
    }

    def __call__(self, text: str, add_special_tokens: bool = False):
        ids, i = [], 0
        while i < len(text):
            hit = next((tok for tok in self.SPECIAL if text.startswith(tok, i)), None)
            if hit is not None:
                ids.append(self.SPECIAL[hit])
                i += len(hit)
            else:
                ids.append(ord(text[i]) % 250)
                i += 1
        return argparse.Namespace(input_ids=ids)


def tiny_model(args):
    from transformers import Inkling2LoopForCausalLM, Inkling2LoopTextConfig
    cfg = Inkling2LoopTextConfig(
        vocab_size=320, unpadded_vocab_size=300, hidden_size=64, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=16,
        sliding_window_size=16, d_rel=4, rel_extent=32, log_scaling_n_floor=16,
        layer_types=["hybrid_sliding", "hybrid_sliding", "hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid"],
        n_layers_per_attn_res_block=2, moe_intermediate_size=32, n_routed_experts=16, num_experts_per_tok=8,
        loop_layer_index_start=2, loop_layer_index_end=4, loop_num_passes=3, loop_gate_rank=8,
        eos_token_id=299, pad_token_id=298,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(args.seed)
    model = Inkling2LoopForCausalLM(cfg)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("query") or n.endswith("e_score_correction_bias"):
                p.normal_(0, 0.5)
        model.lm_head.weight.normal_(0, 0.5)
    return model.float(), None


def random_blocks(vocab: int, seq_len: int, seed: int) -> Iterator[list[int]]:
    chain = torch.Generator().manual_seed(12345)
    succ = torch.randint(0, vocab, (vocab, 4), generator=chain)
    g = torch.Generator().manual_seed(seed)
    while True:
        tok = int(torch.randint(0, vocab, (1,), generator=g))
        seq = [tok]
        for _ in range(seq_len - 1):
            tok = int(succ[tok, torch.randint(0, 4, (1,), generator=g)])
            seq.append(tok)
        yield seq


# ---------------------------------------------------------------------------------------- ckpt & training

def save_ckpt(path: Path, parts: Parts, trainable: dict, hot_experts: list[list[int]], opt, sched, state: dict, args) -> None:
    path.mkdir(parents=True, exist_ok=True)
    bias_deltas = {f"blk.{i}.student_bias_delta": r.student_bias_delta.detach().cpu() for i, r in enumerate(routers(parts))}
    tmp = path / "ckpt.pt.tmp"
    torch.save({
        "trainable": {k: v.detach().cpu() for k, v in trainable.items()},
        "bias_deltas": bias_deltas,
        "hot_experts": hot_experts,
        "meta": {
            "mode": "workingset_128e",
            "n_hot": len(hot_experts[0]),
            "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
            "expert_lora_rank": args.expert_lora_rank, "expert_lora_alpha": args.expert_lora_alpha,
            "lora_lm_head": args.lora_lm_head,
            "model": args.model, "revision": args.revision,
        },
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "state": state,
    }, tmp)
    os.replace(tmp, path / "ckpt.pt")
    (path / "state.json").write_text(json.dumps(state, indent=1))


def load_ckpt(path: Path, parts: Parts, trainable: dict, opt, sched) -> dict:
    blob = torch.load(path / "ckpt.pt", map_location="cpu", weights_only=False)
    with torch.no_grad():
        for k, v in blob["trainable"].items():
            trainable[k].copy_(v.to(trainable[k].device, trainable[k].dtype))
        for i, r in enumerate(routers(parts)):
            bk = f"blk.{i}.student_bias_delta"
            if bk in blob.get("bias_deltas", {}):
                r.student_bias_delta.copy_(blob["bias_deltas"][bk].to(r.student_bias_delta.device))
    opt.load_state_dict(blob["opt"])
    sched.load_state_dict(blob["sched"])
    return blob["state"]


def build_optimizer(trainable: dict, args):
    exp_params = [p for n, p in trainable.items() if ("gu_lora_" in n or "dn_lora_" in n)]
    rtr_params = [p for n, p in trainable.items() if "student_weight_delta" in n]
    other = [p for n, p in trainable.items() if not ("gu_lora_" in n or "dn_lora_" in n or "student_weight_delta" in n)]
    groups = [
        {"params": other, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": exp_params, "lr": args.expert_lr, "weight_decay": args.weight_decay},
        {"params": rtr_params, "lr": args.router_lr, "weight_decay": 0.0},
    ]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, args.beta2), eps=1e-8)
    total = max(args.max_steps, 1)

    def lr_lambda(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / max(args.warmup, 1)
        prog = min(1.0, (step - args.warmup) / max(1, total - args.warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def train_step(parts, train_it, opt, sched, trainable, step: int, args, recorder) -> dict:
    opt.zero_grad(set_to_none=True)
    agg: dict[str, float] = {}
    counts: dict[str, int] = {}
    kinds: set[str] = set()
    cold_w = current_cold_weight(step, args)
    for _ in range(args.grad_accum):
        batch = next(train_it)
        kinds.add("rollout" if isinstance(batch, RolloutBatch) else "ordinary")
        loss, stats = distill_loss(parts, batch, args, recorder, cold_w)
        (loss / args.grad_accum).backward()
        agg["loss"] = agg.get("loss", 0.0) + float(loss.detach()) / args.grad_accum
        for k, v in stats.items():
            agg[k] = agg.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    for k, n in counts.items():
        agg[k] /= n
    agg["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(list(trainable.values()), args.clip))
    agg["batch_kind"] = next(iter(kinds)) if len(kinds) == 1 else "mixed"
    opt.step()
    sched.step()
    agg.update(update_router_workingset(parts, step, args))
    return agg


def dry_run_checks(parts: Parts, hot_experts: list[list[int]], args, recorder, packed_eval, ro_eval) -> None:
    ids = packed_eval[0]
    # 1. Teacher invariance: perturb ALL student deltas + bias and verify teacher diff == 0
    with torch.no_grad(), shape(parts, args.teacher_passes, args.teacher_topk, lora=False):
        a = parts.full(input_ids=ids, use_cache=False).logits
    saved = {}
    saved_b = {}
    with torch.no_grad():
        for n, p in parts.full.named_parameters():
            if any(tag in n for tag in ("lora_B", "student_weight_delta")):
                saved[n] = p.detach().clone()
                p.add_(0.25)
        for i, r in enumerate(routers(parts)):
            saved_b[i] = r.student_bias_delta.clone()
            r.student_bias_delta.add_(-0.75)
    with torch.no_grad(), shape(parts, args.teacher_passes, args.teacher_topk, lora=False):
        b = parts.full(input_ids=ids, use_cache=False).logits
    with torch.no_grad():
        for n, p in saved.items():
            dict(parts.full.named_parameters())[n].copy_(p)
        for i, r in enumerate(routers(parts)):
            r.student_bias_delta.copy_(saved_b[i])
    tdiff = float((a - b).abs().max())
    log.info("CHECK teacher invariant to ALL student deltas + cold bias: max|diff| = %.3g", tdiff)
    assert tdiff == 0.0, f"teacher leaked student delta: {tdiff}"

    # 2. Verify non-zero finite grads on EVERY group + cold-concentration controller drives cold_pct down!
    rargs = argparse.Namespace(**{**vars(args), "lr": 3e-3, "expert_lr": 3e-3, "router_lr": 5e-3,
                                  "max_steps": 40, "warmup": 2, "cold_loss": 0.5, "cold_loss_warmup": 1,
                                  "start_cold_frac": 0.20, "target_cold_frac": 0.05, "cold_ramp_steps": 15,
                                  "cold_bias_gamma": 0.08, "balance_gamma": 0.005, "reverse_kl": 0.0})
    set_resting_shape(parts, rargs.student_passes, rargs.student_topk)
    trainable = freeze_and_select(parts)
    opt, sched = build_optimizer(trainable, rargs)
    fixed_batch = packed_eval[0]
    ev_before = evaluate(parts, packed_eval, ro_eval, rargs, recorder)
    opt.zero_grad(set_to_none=True)
    loss0, stats0 = distill_loss(parts, fixed_batch, rargs, recorder, 0.5)
    assert "cold_mass" in stats0 and loss0.requires_grad, f"missing cold_mass or grad: {stats0}"
    loss0.backward()
    for tag in ("gu_lora_B", "dn_lora_B", "student_weight_delta", "lora_B"):
        gn = sum(float(p.grad.norm()) for n, p in trainable.items() if tag in n and p.grad is not None)
        log.info("CHECK grad norm [%s] = %.4f", tag, gn)
        assert gn > 0.0 and math.isfinite(gn), f"zero/bad grad on {tag}: {gn}"
    # Verify cold_loss alone produces non-zero grad on student_weight_delta even under --grad-ckpt
    opt.zero_grad(set_to_none=True)
    LoRAState.collect_cold_loss = True
    try:
        parts.text(input_ids=fixed_batch, use_cache=False)
        cl_only = pop_cold_loss(parts)
    finally:
        LoRAState.collect_cold_loss = False
    assert cl_only is not None and cl_only.requires_grad, "cold_loss lost grad under grad_ckpt!"
    cl_only.backward()
    rtr_cold_gn = sum(float(p.grad.norm()) for n, p in trainable.items() if "student_weight_delta" in n and p.grad is not None)
    log.info("CHECK cold_loss-only grad norm on student_weight_delta (with grad_ckpt) = %.4f", rtr_cold_gn)
    assert rtr_cold_gn > 0.0 and math.isfinite(rtr_cold_gn)
    opt.zero_grad(set_to_none=True)
    l0 = float(loss0.detach())
    for st in range(35):
        train_step(parts, itertools.repeat(fixed_batch), opt, sched, trainable, st, rargs, recorder)
    ev_after = evaluate(parts, [fixed_batch], [], rargs, recorder)
    with torch.no_grad():
        l1 = float(distill_loss(parts, fixed_batch, rargs, recorder, 0.5)[0])
    log.info("CHECK WSD overfit & concentration: loss %.4f->%.4f | cold_pct %.1f%%->%.1f%%",
             l0, l1, ev_before["eval_cold_pct"], ev_after["eval_cold_pct"])
    assert l1 < 0.55 * l0, f"overfit failed: {l0:.4f} -> {l1:.4f}"
    assert ev_after["eval_cold_pct"] <= 10.0, f"cold concentration failed: {ev_before} -> {ev_after}"

    # 3. Verify BF16 -> int8 HF quantizer + step works
    worst_rel = quantize_hf_experts_int8(parts, args.device)
    log.info("CHECK quantize_hf_experts_int8 worst_rel=%.4f", worst_rel)
    assert worst_rel < 0.03
    stats = train_step(parts, itertools.repeat(fixed_batch), opt, sched, trainable, 36, rargs, recorder)
    assert math.isfinite(stats["loss"]) and stats["grad_norm"] > 0
    log.info("CHECK int8 WSD train step: loss=%.4f cold_pct=%.1f%%: OK", stats["loss"], stats["cold_pct"])
    log.info("DRY RUN PASSED ALL CHECKS")


EVAL_PROMPTS = [
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? End with: Answer: <final answer as a number>",
    "If 3 machines make 3 widgets in 3 minutes, how many minutes does it take 100 machines to make 100 widgets? End with: Answer: <final answer as a number>",
    "Write a 2-sentence warm reply to a friend who just passed their driving test.",
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    a = p.add_argument
    a("--dry-run-tiny", action="store_true")
    a("--model", default=MODEL_ID); a("--revision", default=MODEL_REVISION)
    a("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a("--hot-plan", default=None, help="JSON with hot_experts [[...], ...]")
    a("--out", default="runs/distill_ws"); a("--resume", action="store_true")
    a("--init-from", default=None)
    a("--max-hours", type=float, default=4.0); a("--max-steps", type=int, default=550)
    a("--seq-len", type=int, default=1024); a("--batch", type=int, default=4); a("--grad-accum", type=int, default=2)
    a("--lr", type=float, default=1.5e-4)
    a("--expert-lr", type=float, default=2.0e-4)
    a("--router-lr", type=float, default=2.0e-4)
    a("--balance-gamma", type=float, default=1e-3)
    a("--cold-loss", type=float, default=0.25)
    a("--cold-loss-warmup", type=int, default=60)
    a("--start-cold-frac", type=float, default=0.25)
    a("--target-cold-frac", type=float, default=0.015)
    a("--cold-ramp-steps", type=int, default=160)
    a("--cold-bias-gamma", type=float, default=0.008)
    a("--rollout-frac", type=float, default=0.50)
    a("--warmup", type=int, default=25); a("--weight-decay", type=float, default=0.0); a("--clip", type=float, default=1.0)
    a("--beta2", type=float, default=0.99); a("--seed", type=int, default=42)
    a("--teacher-passes", type=int, default=3); a("--teacher-topk", type=int, default=8)
    a("--student-passes", type=int, default=1); a("--student-topk", type=int, default=4)
    a("--lora-rank", type=int, default=32); a("--lora-alpha", type=float, default=64.0)
    a("--expert-lora-rank", type=int, default=32); a("--expert-lora-alpha", type=float, default=64.0)
    a("--lora-lm-head", action="store_true")
    a("--experts-int8-from-hf", action="store_true")
    a("--reverse-kl", type=float, default=0.25)
    a("--rollout-forward-kl", type=float, default=1.0)
    a("--rollout-reverse-kl", type=float, default=0.35)
    a("--sft-ce", type=float, default=0.10)
    a("--layer-mse", type=float, default=0.25)
    a("--teacher-note", default="Think concisely in under 180 words and stop as soon as you verify the final answer.")
    a("--rollouts-jsonl", default=None)
    a("--kl-temp", type=float, default=1.0); a("--logit-chunk", type=int, default=512)
    a("--grad-ckpt", action="store_true")
    a("--eval-every", type=int, default=50); a("--eval-seqs", type=int, default=16)
    a("--save-every", type=int, default=50); a("--gen-tokens", type=int, default=192)
    args = p.parse_args(argv)
    if args.dry_run_tiny:
        args.device, args.max_steps, args.seq_len, args.batch, args.grad_accum = "cpu", 10, 32, 2, 1
        args.eval_every, args.save_every, args.eval_seqs, args.gen_tokens = 5, 10, 4, 8
        args.lr, args.expert_lr, args.router_lr, args.warmup, args.logit_chunk = 1e-3, 1e-3, 2e-3, 2, 16
        args.lora_rank, args.lora_alpha, args.expert_lora_rank, args.expert_lora_alpha = 4, 8.0, 4, 8.0
        args.lora_lm_head, args.grad_ckpt = True, True
        args.out = "runs/dry-run-ws"
    return args


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run_tiny:
        model, tok = tiny_model(args)
        parts = get_parts(model)
        hot_experts = [list(range(0, 6)) for _ in range(len(parts.text.layers))]
        tap_layers = [1, 3, 5]
    else:
        assert args.hot_plan, "--hot-plan is required for real runs"
        plan = json.loads(Path(args.hot_plan).read_text())
        hot_experts = plan["hot_experts"]
        from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration
        tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = Inkling2LoopForConditionalGeneration.from_pretrained(
            args.model, revision=args.revision, dtype=torch.bfloat16,
            device_map={"": "cpu" if args.experts_int8_from_hf else args.device},
            experts_implementation="eager",
        )
        model.model.vision_tower = None
        parts = get_parts(model)
        if args.experts_int8_from_hf:
            quantize_hf_experts_int8(parts, args.device)
        tap_layers = [5, 11, 17, 23]

    wrapped = inject_ws_student(parts.text, parts.head_owner, hot_experts, args)
    model.to(args.device)
    set_resting_shape(parts, args.student_passes, args.student_topk)
    trainable = freeze_and_select(parts)
    if args.grad_ckpt:
        model.activation_checkpointing_enable({"use_reentrant": False})
    model.train()

    n_train = sum(p.numel() for p in trainable.values())
    log.info("WSD-128e student ready: hot=%d/layer, %d wrapped linears, %d trainable tensors (%.2fM params)",
             len(hot_experts[0]), len(wrapped), len(trainable), n_train / 1e6)

    recorder = MultiLayerRecorder(parts.text.layers, tap_layers)
    opt, sched = build_optimizer(trainable, args)
    state = {"step": 0, "seqs": 0, "hours": 0.0}
    if args.resume and (out / "ckpt.pt").exists():
        state = load_ckpt(out, parts, trainable, opt, sched)
        log.info("resumed at step %d (%d seqs)", state["step"], state["seqs"])
    elif args.init_from:
        blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for k, v in blob["trainable"].items():
                if k in trainable and trainable[k].shape == v.shape:
                    trainable[k].copy_(v.to(trainable[k].device, trainable[k].dtype))
            for i, r in enumerate(routers(parts)):
                bk = f"blk.{i}.student_bias_delta"
                if bk in blob.get("bias_deltas", {}) and r.student_bias_delta.shape == blob["bias_deltas"][bk].shape:
                    r.student_bias_delta.copy_(blob["bias_deltas"][bk].to(r.student_bias_delta.device))
        log.info("initialized from %s (fresh optimizer)", args.init_from)

    digest0 = expert_digest(parts)
    pad_id = (tok.pad_token_id if (tok is not None and tok.pad_token_id is not None) else 0)
    if args.dry_run_tiny:
        vocab = parts.text.config.unpadded_vocab_size
        tiny_tok = _TinyTokenizer()
        def synth_ro():
            while True:
                yield {"text": f"<|message_system|><|content_text|>E:0.9{END_MESSAGE}<|message_user|><|content_text|>2+2?{END_MESSAGE}{MESSAGE_MODEL}<|content_thinking|>2+2=4{END_MESSAGE}{MESSAGE_MODEL}<|content_text|>Answer: 4{END_MESSAGE}", "is_gold": True, "has_thinking": True}
        ro_it = rollout_batches(tiny_tok, [synth_ro()], [1.0], args.seq_len, args.batch, args.seed, "Be brief.", pad_id, args.device)
        ord_it = (torch.tensor([next(random_blocks(vocab, args.seq_len, args.seed + s)) for _ in range(args.batch)], device=args.device) for s in itertools.count())
        train_it = (next(ro_it) if s % 2 == 0 else next(ord_it) for s in itertools.count())
        packed_eval = [torch.tensor([next(random_blocks(vocab, args.seq_len, args.seed + 100 + s)) for _ in range(args.batch)], device=args.device) for s in range(2)]
        ro_eval = [next(ro_it)]
    else:
        train_it, packed_eval, ro_eval = build_real_data_iters(tok, args, pad_id)

    t0, hours0, seqs0 = time.time(), state["hours"], state["seqs"]
    if state["step"] == 0:
        ev0 = evaluate(parts, packed_eval, ro_eval, args, recorder)
        log.info("step 0 eval %s", ev0)
        with open(out / "eval.jsonl", "a") as f:
            f.write(json.dumps({"step": 0, **ev0}) + "\n")

    while state["step"] < args.max_steps:
        stats = train_step(parts, train_it, opt, sched, trainable, state["step"], args, recorder)
        state["step"] += 1
        state["seqs"] += args.batch * args.grad_accum
        state["hours"] = hours0 + (time.time() - t0) / 3600.0
        tok_s = (state["seqs"] - seqs0) * args.seq_len / max(1e-6, time.time() - t0)
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        log.info("step %d [%s] loss %.4f %s lr %.2e tok/s %.0f gpu_peak_gb %.1f",
                 state["step"], stats["batch_kind"], stats["loss"],
                 {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items() if k not in ("loss", "batch_kind")},
                 sched.get_last_lr()[0], tok_s, peak_gb)
        if state["step"] % args.eval_every == 0:
            ev = evaluate(parts, packed_eval, ro_eval, args, recorder)
            gens = greedy_samples(parts, tok, EVAL_PROMPTS if tok else ["1 2 3 4"], args, args.device)
            log.info("eval step %d %s", state["step"], ev)
            with open(out / "eval.jsonl", "a") as f:
                f.write(json.dumps({"step": state["step"], **ev, "samples": gens}) + "\n")
        hit_cap = (time.time() - t0) / 3600.0 >= args.max_hours
        if state["step"] % args.save_every == 0 or hit_cap:
            save_ckpt(out, parts, trainable, hot_experts, opt, sched, state, args)
        if hit_cap:
            log.info("wall-clock cap %.2fh reached; saved & stopping", args.max_hours)
            break

    save_ckpt(out, parts, trainable, hot_experts, opt, sched, state, args)
    assert expert_digest(parts) == digest0, "frozen base experts changed!"
    log.info("base expert digest unchanged: %s", digest0)
    if args.dry_run_tiny:
        dry_run_checks(parts, hot_experts, args, recorder, packed_eval, ro_eval)


if __name__ == "__main__":
    main()
