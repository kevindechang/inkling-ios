#!/usr/bin/env python3
"""SOTA v2 QAT-base working-set concentration distillation (WSD-40 v2).

Teacher: frozen clean HF BF16 dense / norms / attention-residual queries and clean HF
per-row INT8 experts (128/128, nothing pruned). LoRAState.enabled is False, so the
teacher never sees student QAT weights, LoRAs, router deltas, bias deltas, or
small-dense deltas.

Student: 1-pass top-4. Hot-40 experts and the attention / lm_head matmuls come from
the phone GGUF (already run3-merged and expert-reordered). ``init-run3.pt`` is loaded
into the trainable LoRAs and subtracted back out of the GGUF dense weights so step 0
is the GGUF student, and the final merge adds the LoRA onto clean HF BF16 exactly once.

Dry run (CPU, no network, does not touch the real GGUF):

    PYTHONPATH=train/.pydeps:train/../llama.cpp/gguf-py \\
    python train/distill_v2_sota.py --dry-run-tiny
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
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

log = logging.getLogger("distill_v2")

MODEL_ID = "thinkingmachines/TML-d24-varloop-research"
MODEL_REVISION = "c90921bd5f78df6c8168173b9f8a3a850320c594"
TOKENIZER_DIR = Path(os.environ.get("INKLING_TOKENIZER_DIR", "models/inkling2mini/hf-tokenizer"))

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
INT8_MAX = 127

IDENTITY_RE = re.compile(
    r"\b(as an ai\b|an ai (language )?model|language model developed|i am an ai|i'm an ai|my name is|"
    r"i am (chatgpt|claude|gpt|llama|tulu|olmo|gemini)|i'm (chatgpt|claude|tulu|olmo))",
    re.IGNORECASE,
)
CODING_SOURCE_RE = re.compile(
    r"(\bcode\b|\bpython\b|\bcoder\b|\bevol[-_ ]?code\b|\bstarcoder\b|\bhumaneval\b|\bmbpp\b|\bswe(?:[-_ ]?bench)?\b)",
    re.IGNORECASE,
)
CALC_ANN_RE = re.compile(r"<<[^<>]*>>")
TOOL_BLOCK_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
FUNCTION_CALL_RE = re.compile(r"<functioncall>\s*(.*?)\s*</functioncall>", re.DOTALL | re.IGNORECASE)
GLAIVE_SPLIT_RE = re.compile(r"\n?(SYSTEM:|USER:|ASSISTANT:|FUNCTION RESPONSE:)\s*")

MATH_SUBSETS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)
EFFORT_MATH = (0.2, 0.45, 0.9)
EFFORT_LOGIC = (0.35, 0.5, 0.9)
# Domain mix sums to 1.0. Sub-splits are the streams inside each domain.
MIX_WEIGHTS = (
    ("gsm8k", 0.09),
    ("math", 0.09),
    ("tulu_math", 0.05),
    ("ot_puzzle", 0.05),
    ("logiqa2", 0.05),
    ("folio", 0.03),
    ("ot_science", 0.05),
    ("sciq", 0.03),
    ("arc_challenge", 0.02),
    ("mmlu_aux", 0.02),
    ("tulu_if", 0.20),
    ("hermes", 0.11),
    ("glaive", 0.09),
    ("tulu_mix", 0.07),
    ("rollouts", 0.05),
)


# --------------------------------------------------------------------------------------------- LoRA & QAT base


class LoRAState:
    enabled = True
    track_load = False
    collect_cold_loss = False


class LoRALinear(nn.Module):
    """Teacher uses the frozen clean HF linear. Student uses the optional QAT base plus the LoRA.

    ``student_kind`` is ``dense`` (BF16 attention) or ``int8`` (per-row INT8 lm_head).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.student_kind = "none"

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def set_dense_base(self, w: torch.Tensor) -> None:
        w = w.to(dtype=torch.bfloat16)
        if self.student_kind == "dense":
            self.student_base.copy_(w)
        else:
            self.register_buffer("student_base", w, persistent=False)
            self.student_kind = "dense"

    def set_int8_base(self, q: torch.Tensor, scale: torch.Tensor) -> None:
        scale = scale.to(dtype=torch.bfloat16)
        if self.student_kind == "int8":
            self.student_q.copy_(q)
            self.student_scale.copy_(scale)
        else:
            self.register_buffer("student_q", q, persistent=False)
            self.register_buffer("student_scale", scale, persistent=False)
            self.student_kind = "int8"

    def student_weight(self) -> torch.Tensor:
        if self.student_kind == "dense":
            return self.student_base
        if self.student_kind == "int8":
            return self.student_q.to(self.student_scale.dtype) * self.student_scale
        raise RuntimeError("student_weight requested without a QAT base")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not LoRAState.enabled) or self.student_kind == "none":
            out = self.base(x)
        else:
            w = self.student_weight()
            if w.dtype != x.dtype:
                w = w.to(x.dtype)
            out = F.linear(x, w, self.base.bias)
        if not LoRAState.enabled:
            return out
        delta = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B) * self.scale
        return out + delta.to(out.dtype)


def _ws_router_forward(self, hidden_states: torch.Tensor):
    """Teacher: unmodified full-expert routing. Student: weight delta + working-set bias delta."""
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
    logits = F.linear(flat, w)
    scores = self.score_fn(logits)
    indices = torch.topk(scores + b, self.top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)

    if LoRAState.collect_cold_loss:
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


def _ws_experts_forward(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
    """Cold experts and the teacher use the clean HF stack. Hot student experts use the QAT GGUF stack + LoRA."""
    final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
        hits = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten()
        expert_hit = hits.tolist()

    use_lora = LoRAState.enabled and getattr(self, "exp_lora_rank", 0) > 0
    use_student_hot = LoRAState.enabled and getattr(self, "student_hot_gu", None) is not None
    for e_glob in expert_hit:
        if e_glob == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[e_glob])
        current_state = hidden_states[token_idx]
        loc = self._g2l_py.get(e_glob, -1) if (use_lora or use_student_hot) else -1
        if use_student_hot and loc >= 0:
            gu_w = self.student_hot_gu[loc]
            if gu_w.dtype != current_state.dtype:
                gu_w = gu_w.to(current_state.dtype)
            gu = F.linear(current_state, gu_w)
        else:
            gu = F.linear(current_state, self.gate_up_proj[e_glob])
        if use_lora and loc >= 0:
            gu_d = F.linear(F.linear(current_state.float(), self.gu_lora_A[loc]), self.gu_lora_B[loc]) * self.exp_lora_scale
            gu = gu + gu_d.to(gu.dtype)
        act = self._apply_gate(gu)
        if use_student_hot and loc >= 0:
            dn_w = self.student_hot_dn[loc]
            if dn_w.dtype != act.dtype:
                dn_w = dn_w.to(act.dtype)
            dn = F.linear(act, dn_w)
        else:
            dn = F.linear(act, self.down_proj[e_glob])
        if use_lora and loc >= 0:
            dn_d = F.linear(F.linear(act.float(), self.dn_lora_A[loc]), self.dn_lora_B[loc]) * self.exp_lora_scale
            dn = dn + dn_d.to(dn.dtype)
        weighted = dn * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, weighted.to(final_hidden_states.dtype))
    return final_hidden_states.to(hidden_states.dtype)


def _attnres_forward(self, block_states: torch.Tensor, partial_block_states: torch.Tensor):
    """Teacher reads ``query`` only. Student adds the zero-init F32 delta, which is what receives the grad."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.inkling2_loop.modeling_inkling2_loop import eager_attention_forward

    stacked_states = torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim=-2).float()
    stacked_states = self.norm(stacked_states)
    hidden_size = stacked_states.shape[-1]
    query = self.query.float()
    if LoRAState.enabled:
        query = query + self.student_delta
    query_states = query.view(-1, hidden_size).expand(*stacked_states.shape[:-2], -1, -1)
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
    attn_output, _ = attention_interface(
        self, query_states, stacked_states, stacked_states, None, scaling=hidden_size**-0.5
    )
    return attn_output.transpose(1, 2).unbind(-2)


def _rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    weight = self.weight if not LoRAState.enabled else (self.weight.float() + self.student_delta)
    return (weight * hidden_states).to(input_dtype)


def _iter_small_dense(text: nn.Module):
    """The F32 tensors the phone graph cannot LoRA: attention-residual queries and RMSNorms."""
    for i, layer in enumerate(text.layers):
        attn_in = getattr(layer, "attention_input", None)
        if attn_in is not None:
            yield f"layers.{i}.attention_input", attn_in, "query"
        mlp_in = getattr(layer, "mlp_input", None)
        if mlp_in is not None:
            yield f"layers.{i}.mlp_input", mlp_in, "query"
        for attr in (
            "input_layernorm",
            "q_input_layernorm",
            "k_input_layernorm",
            "v_input_layernorm",
            "post_attention_layernorm",
        ):
            mod = getattr(layer, attr, None)
            if mod is not None and hasattr(mod, "weight"):
                yield f"layers.{i}.{attr}", mod, "weight"
    final_input = getattr(text, "final_input", None)
    if final_input is not None:
        yield "final_input", final_input, "query"
    norm = getattr(text, "norm", None)
    if norm is not None and hasattr(norm, "weight"):
        yield "norm", norm, "weight"


def install_small_dense_deltas(text: nn.Module) -> int:
    n = 0
    for _name, module, attr in _iter_small_dense(text):
        base = getattr(module, attr)
        module.student_delta = nn.Parameter(torch.zeros_like(base, dtype=torch.float32))
        if attr == "query":
            module.forward = MethodType(_attnres_forward, module)
        else:
            module.forward = MethodType(_rmsnorm_forward, module)
        n += 1
    return n


def inject_ws_student(text_model, lm_head_owner, hot_experts: list[list[int]], args) -> list[str]:
    wrapped = []
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
    if args.lora_lm_head:
        lm_head_owner.lm_head = LoRALinear(lm_head_owner.lm_head, args.lora_rank, args.lora_alpha).to(
            lm_head_owner.lm_head.weight.device
        )
        wrapped.append("lm_head")
    for i, layer in enumerate(text_model.layers):
        hot = list(hot_experts[i])
        m_hot = len(hot)
        rtr = layer.mlp.gate
        n_exp = rtr.num_experts
        dev = rtr.weight.device
        hot_mask = torch.zeros(n_exp, dtype=torch.bool, device=dev)
        hot_mask[torch.tensor(hot, dtype=torch.long, device=dev)] = True
        rtr.register_buffer("hot_mask", hot_mask)
        rtr.register_buffer("cold_mask_f", (~hot_mask).float())
        rtr.student_weight_delta = nn.Parameter(torch.zeros(n_exp, rtr.hidden_dim, dtype=torch.float32, device=dev))
        rtr.register_buffer("student_bias_delta", torch.zeros(n_exp, dtype=torch.float32, device=dev))
        rtr.register_buffer("_load_accum", torch.zeros(n_exp, dtype=torch.float32, device=dev))
        rtr._load_tokens = 0
        rtr._last_flat = None
        rtr.forward = MethodType(_ws_router_forward, rtr)

        exp = layer.mlp.experts
        exp._g2l_py = {e_glob: loc for loc, e_glob in enumerate(hot)}
        exp.exp_lora_rank = args.expert_lora_rank
        exp.exp_lora_scale = (args.expert_lora_alpha / args.expert_lora_rank) if args.expert_lora_rank > 0 else 0.0
        exp.student_hot_gu = None
        exp.student_hot_dn = None
        if args.expert_lora_rank > 0:
            r = args.expert_lora_rank
            h, inter = exp.hidden_dim, exp.intermediate_dim
            gu_A = torch.empty(m_hot, r, h, dtype=torch.float32, device=dev)
            dn_A = torch.empty(m_hot, r, inter, dtype=torch.float32, device=dev)
            for loc in range(m_hot):
                nn.init.kaiming_uniform_(gu_A[loc], a=math.sqrt(5))
                nn.init.kaiming_uniform_(dn_A[loc], a=math.sqrt(5))
            exp.gu_lora_A = nn.Parameter(gu_A)
            exp.gu_lora_B = nn.Parameter(torch.zeros(m_hot, 2 * inter, r, dtype=torch.float32, device=dev))
            exp.dn_lora_A = nn.Parameter(dn_A)
            exp.dn_lora_B = nn.Parameter(torch.zeros(m_hot, h, r, dtype=torch.float32, device=dev))
        exp.forward = MethodType(_ws_experts_forward, exp)
    if args.train_small_dense_deltas:
        n_small = install_small_dense_deltas(text_model)
        log.info("installed %d student-only F32 small-dense deltas", n_small)
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


def _is_edge_layer(i: int, n_layers: int) -> bool:
    return i in (0, 1, n_layers - 2, n_layers - 1)


def layer_final_target(i: int, n_layers: int, args) -> float:
    """Edge layers keep a slightly higher cold budget. The 24-layer mean of these targets is target_cold_frac."""
    if _is_edge_layer(i, n_layers):
        return min(0.25, args.target_cold_frac * args.edge_cold_mult)
    return args.target_cold_frac * 0.94


def current_layer_cold_target(step: int, i: int, n_layers: int, args) -> float:
    final = layer_final_target(i, n_layers, args)
    if args.cold_ramp_steps <= 0 or step >= args.cold_ramp_steps:
        return final
    prog = step / float(args.cold_ramp_steps)
    return final + (args.start_cold_frac - final) * 0.5 * (1.0 + math.cos(math.pi * prog))


def current_cold_weight(step: int, args) -> float:
    if args.cold_loss <= 0:
        return 0.0
    if args.cold_loss_warmup <= 0 or step >= args.cold_loss_warmup:
        return args.cold_loss
    return args.cold_loss * ((step + 1) / float(args.cold_loss_warmup))


def update_router_workingset(parts: Parts, step: int, args) -> dict[str, float]:
    n_layers = len(routers(parts))
    targets = [current_layer_cold_target(step, i, n_layers, args) for i in range(n_layers)]
    cold_fracs, hot_imbals, min_hot_active, cold_biases = [], [], 10_000, []
    with torch.no_grad():
        for i, r in enumerate(routers(parts)):
            if r._load_tokens <= 0:
                continue
            tgt_cold = targets[i]
            tot_visits = float(r._load_tokens * r.top_k)
            hot_counts = r._load_accum[r.hot_mask]
            cold_counts = r._load_accum[~r.hot_mask]
            cold_frac = float(cold_counts.sum().item()) / max(tot_visits, 1.0)
            cold_fracs.append(cold_frac)
            m_hot = int(r.hot_mask.sum().item())
            if args.balance_gamma > 0 and m_hot > 0:
                target_h = (tot_visits * (1.0 - cold_frac)) / float(m_hot)
                hot_b = r.student_bias_delta[r.hot_mask]
                hot_b = hot_b + torch.sign(target_h - hot_counts) * args.balance_gamma
                hot_b = (hot_b - hot_b.mean()).clamp(-0.5, 0.5)
                r.student_bias_delta[r.hot_mask] = hot_b
                hot_imbals.append(float((hot_counts.max() / max(target_h, 1e-6)).item()))
                min_hot_active = min(min_hot_active, int((hot_counts > 0).sum().item()))
            if args.cold_bias_gamma > 0:
                cold_b = r.student_bias_delta[~r.hot_mask]
                if cold_frac > tgt_cold:
                    scale = min(3.0, max(0.5, (cold_frac - tgt_cold) / max(tgt_cold, 0.02)))
                    cold_b = (cold_b - args.cold_bias_gamma * scale).clamp(args.cold_bias_min, 0.0)
                elif cold_frac < tgt_cold * 0.65:
                    cold_b = (cold_b + args.cold_bias_gamma * 0.5).clamp(args.cold_bias_min, 0.0)
                r.student_bias_delta[~r.hot_mask] = cold_b
                cold_biases.append(float(cold_b.mean().item()))
            r._load_accum.zero_()
            r._load_tokens = 0
    if not cold_fracs:
        return {}
    mean_cf = sum(cold_fracs) / len(cold_fracs)
    mean_tgt = sum(targets) / len(targets)
    return {
        "cold_pct": round(100.0 * mean_cf, 2),
        "cold_per_tok": round(mean_cf * n_layers * args.student_topk, 2),
        "tgt_cold_pct": round(100.0 * mean_tgt, 2),
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


def _is_small_delta_name(name: str) -> bool:
    return "student_delta" in name and "student_weight_delta" not in name


def freeze_and_select(parts: Parts) -> dict[str, nn.Parameter]:
    for p in parts.full.parameters():
        p.requires_grad_(False)
    trainable = {}
    for name, p in parts.full.named_parameters():
        if "lora_" in name or "student_weight_delta" in name or _is_small_delta_name(name):
            p.requires_grad_(True)
            trainable[name] = p
    return trainable


class QuantExperts:
    """Per-row INT8 + BF16 scale. Indexing returns one matrix in the compute dtype. Not a Parameter."""

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
        self.q, self.scale, self.dtype = q, scale, dtype
        self.shape = q.shape

    def __getitem__(self, idx) -> torch.Tensor:
        return self.q[idx].to(self.dtype) * self.scale[idx]


def row_int8_np(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = np.ascontiguousarray(w, dtype=np.float32)
    rs = np.max(np.abs(w), axis=1, keepdims=True) / float(INT8_MAX)
    rs = np.maximum(rs, 1e-12).astype(np.float32)
    q = np.clip(np.rint(w / rs), -INT8_MAX, INT8_MAX).astype(np.int8)
    return q, rs


def stack_row_int8(slices: list[np.ndarray], device) -> QuantExperts:
    qs, ss = [], []
    for s in slices:
        q, rs = row_int8_np(s)
        qs.append(q)
        ss.append(rs)
    q = torch.from_numpy(np.stack(qs)).to(device)
    scale = torch.from_numpy(np.stack(ss)).to(device=device, dtype=torch.bfloat16)
    return QuantExperts(q, scale, torch.bfloat16)


def quantize_hf_experts_int8(parts: Parts, device) -> float:
    worst_rel = 0.0
    for i, layer in enumerate(parts.text.layers):
        experts = layer.mlp.experts
        experts.config._experts_implementation = "eager"
        for attr in ("gate_up_proj", "down_proj"):
            param = getattr(experts, attr)
            if isinstance(param, QuantExperts):
                continue
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
                h.update(t.q[0].detach().cpu().numpy().tobytes()[:1 << 14])
                h.update(str(int(t.q.sum(dtype=torch.int64))).encode())
                continue
            h.update(t.detach().float().cpu().numpy().tobytes()[:1 << 14])
    return h.hexdigest()[:16]


def student_expert_gguf_index(loc: int, hot_row: list[int], reordered: bool) -> int:
    """Published hot-40 GGUFs already store hot experts in positions ``0 .. n_hot-1``."""
    return int(loc) if reordered else int(hot_row[loc])


def import_gguf(llama_cpp_dir: str):
    sys.path.insert(0, str(Path(llama_cpp_dir) / "gguf-py"))
    import gguf  # noqa: E402
    return gguf


def _dequant_array(data, qtype) -> np.ndarray:
    from gguf.quants import dequantize
    arr = np.ascontiguousarray(np.asarray(data))
    return np.ascontiguousarray(dequantize(arr, qtype).astype(np.float32))


def load_student_qat_from_gguf(parts: Parts, hot_experts: list[list[int]], gguf_path: str, args) -> None:
    """Hot expert slices + attention / lm_head QAT bases. Teacher HF weights are not written."""
    gguf = import_gguf(args.llama_cpp_dir)
    by_name = {t.name: t for t in gguf.GGUFReader(gguf_path).tensors}
    n_hot = len(hot_experts[0])
    n_hot_loaded = 0
    n_dense = 0
    for i, layer in enumerate(parts.text.layers):
        exp = layer.mlp.experts
        for stem, attr in (("ffn_gate_up_exps", "student_hot_gu"), ("ffn_down_exps", "student_hot_dn")):
            t = by_name[f"blk.{i}.{stem}.weight"]
            scale_t = by_name.get(f"blk.{i}.{stem}.scale")
            escale = np.asarray(scale_t.data, dtype=np.float32) if scale_t is not None else None
            slices = []
            for loc in range(n_hot):
                idx = student_expert_gguf_index(loc, hot_experts[i], args.gguf_experts_reordered)
                w = _dequant_array(t.data[idx], t.tensor_type)
                if escale is not None:
                    w = w * np.asarray(escale[idx], dtype=np.float32)
                slices.append(w)
            setattr(exp, attr, stack_row_int8(slices, args.device))
            n_hot_loaded += 1
        dev = next(layer.parameters()).device
        for suffix, stem in LORA_TARGETS.items():
            try:
                module = layer.get_submodule(suffix)
            except AttributeError:
                continue
            if not isinstance(module, LoRALinear):
                continue
            gname = f"blk.{i}.{stem}.weight"
            if gname not in by_name:
                continue
            t = by_name[gname]
            w = _dequant_array(t.data, t.tensor_type)
            if tuple(w.shape) != tuple(module.base.weight.shape):
                raise ValueError(f"{gname}: GGUF {w.shape} vs linear {tuple(module.base.weight.shape)}")
            # Keep the f32 GGUF value until run3 is subtracted, then drop it.
            module._gguf_f32 = torch.from_numpy(w.copy())
            module.set_dense_base(module._gguf_f32.to(device=dev))
            n_dense += 1
    head = parts.head_owner.lm_head
    if isinstance(head, LoRALinear) and "output.weight" in by_name:
        t = by_name["output.weight"]
        w = _dequant_array(t.data, t.tensor_type)[: head.base.out_features]
        if w.shape[1] != head.base.in_features:
            raise ValueError(f"output.weight GGUF {w.shape} vs lm_head in={head.base.in_features}")
        q, scale = row_int8_np(w)
        dev = head.lora_A.device
        head.set_int8_base(torch.from_numpy(q).to(dev), torch.from_numpy(scale).to(dev))
        n_dense += 1
    log.info(
        "student QAT base: %d hot expert stacks, %d dense tensors (experts reordered=%s)",
        n_hot_loaded, n_dense, args.gguf_experts_reordered,
    )


def _lora_pairs(trainable: dict) -> dict[str, dict[str, torch.Tensor]]:
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in trainable.items():
        if key.endswith(".lora_A"):
            pairs.setdefault(key[: -len(".lora_A")], {})["A"] = value
        elif key.endswith(".lora_B"):
            pairs.setdefault(key[: -len(".lora_B")], {})["B"] = value
    return pairs


def _meta_scale(meta: dict, alpha_key: str, rank_key: str, default: float = 1.0) -> float:
    if alpha_key in meta and rank_key in meta and float(meta[rank_key]) != 0.0:
        return float(meta[alpha_key]) / float(meta[rank_key])
    return default


def gguf_includes_run3_dense(args, blob: dict) -> bool:
    """run3-style checkpoints have no ``hot_experts``. The phone GGUF already contains that dense delta."""
    if not args.student_from_gguf or "hot_experts" in blob:
        return False
    if args.gguf_includes_init_dense is None:
        return True
    return bool(args.gguf_includes_init_dense)


def apply_run3_init(parts: Parts, blob: dict, args) -> dict:
    """Load attention LoRAs, convert router LoRAs to a full delta, and remove run3 from the QAT dense base.

    ``student_base = (W_gguf - B @ A * run3_scale).to(bf16)`` so step 0 reconstructs ``W_gguf`` and the
    later merge onto clean HF BF16 does not apply run3 twice. Router LoRAs become ``student_weight_delta``.
    """
    meta = blob.get("meta") or {}
    run3_scale = _meta_scale(meta, "lora_alpha", "lora_rank", 1.0)
    router_scale = _meta_scale(meta, "router_alpha", "router_rank", 1.0)
    pairs = _lora_pairs(blob["trainable"])
    mod_of = {module: name for name, module in parts.full.named_modules()}
    n_attn = 0
    max_diff = 0.0
    with torch.no_grad():
        for module in list(mod_of):
            if not isinstance(module, LoRALinear):
                continue
            full_name = mod_of[module]
            if full_name == "lm_head" or full_name.endswith(".lm_head"):
                continue
            pair = pairs.get(full_name)
            if not pair or "A" not in pair or "B" not in pair:
                continue
            A = pair["A"].detach().float()
            B = pair["B"].detach().float()
            if tuple(A.shape) != tuple(module.lora_A.shape) or tuple(B.shape) != tuple(module.lora_B.shape):
                raise ValueError(f"{full_name}: run3 LoRA {tuple(A.shape)}/{tuple(B.shape)} vs student {tuple(module.lora_A.shape)}/{tuple(module.lora_B.shape)}")
            # student_scale * B_loaded @ A == run3_scale * B @ A. When the student scale is 1 this is a copy.
            module.lora_A.copy_(A.to(device=module.lora_A.device))
            module.lora_B.copy_((B * (run3_scale / module.scale)).to(device=module.lora_B.device))
            if module.student_kind == "dense" and hasattr(module, "_gguf_f32"):
                delta = (B @ A) * run3_scale
                w_gguf_cpu = module._gguf_f32.detach().float().cpu()
                base = (w_gguf_cpu - delta.cpu()).to(device=module.student_base.device, dtype=torch.bfloat16)
                module.student_base.copy_(base)
                eff = (module.lora_B.float() @ module.lora_A.float()) * module.scale
                recon = module.student_base.float() + eff
                max_diff = max(max_diff, float((recon - w_gguf_cpu.to(recon.device)).abs().max()))
            n_attn += 1
        n_rtr = 0
        for router in routers(parts):
            rname = mod_of[router]
            pair = pairs.get(rname)
            if not pair or "A" not in pair or "B" not in pair:
                continue
            A = pair["A"].detach().float()
            B = pair["B"].detach().float()
            delta = (B @ A) * router_scale
            if tuple(delta.shape) != tuple(router.student_weight_delta.shape):
                raise ValueError(f"{rname}: router delta {tuple(delta.shape)} vs {tuple(router.student_weight_delta.shape)}")
            router.student_weight_delta.copy_(delta.to(device=router.student_weight_delta.device))
            n_rtr += 1
        for module in mod_of:
            if isinstance(module, LoRALinear) and hasattr(module, "_gguf_f32"):
                del module._gguf_f32
    log.info(
        "loaded %d attention LoRAs and converted %d router LoRAs (run3 dense scale %.4f, router scale %.4f, recon max|diff|=%.3g)",
        n_attn, n_rtr, run3_scale, router_scale, max_diff,
    )
    return {"recon": max_diff, "n_attn": n_attn, "n_rtr": n_rtr}


def copy_matching_trainable(parts: Parts, trainable: dict, blob: dict) -> None:
    with torch.no_grad():
        for key, value in blob.get("trainable", {}).items():
            if key in trainable and tuple(trainable[key].shape) == tuple(value.shape):
                trainable[key].copy_(value.to(device=trainable[key].device, dtype=trainable[key].dtype))
        for i, router in enumerate(routers(parts)):
            bk = f"blk.{i}.student_bias_delta"
            saved = blob.get("bias_deltas", {}).get(bk)
            if saved is not None and tuple(saved.shape) == tuple(router.student_bias_delta.shape):
                router.student_bias_delta.copy_(saved.to(router.student_bias_delta.device))


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


def chunked_kl_ce(parts, s_h, t_h, chunk, temp, with_reverse=False, targets=None):
    def one(s_part, t_part, tgt_part):
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


def teacher_forward(parts, ids, args, recorder, attention_mask=None):
    recorder.values.clear()
    recorder.active = True
    try:
        with torch.no_grad(), shape(parts, args.teacher_passes, args.teacher_topk, lora=False):
            out = parts.text(input_ids=ids, attention_mask=attention_mask, use_cache=False)
            taps = {k: v.detach() for k, v in recorder.pop().items()}
    finally:
        recorder.active = False
    return out.last_hidden_state.detach(), taps


def layer_mse_loss(s_taps, t_taps, s_mask=None, t_mask=None):
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


def pop_cold_loss(parts: Parts):
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


def distill_loss(parts, batch, args, recorder, cold_w: float):
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
def _kl_batch(parts, s_h, t_h, args) -> float:
    kl, _, _ = chunked_kl_ce(parts, s_h, t_h, args.logit_chunk, 1.0)
    return float(kl)


@torch.no_grad()
def evaluate(parts, packed_eval, ro_eval, args, recorder, tool_if_eval=None) -> dict:
    was_training = parts.full.training
    parts.full.eval()
    kls, lmses, ro_kls, tool_kls = [], [], [], []
    for r in routers(parts):
        r._load_accum.zero_()
        r._load_tokens = 0
    for ids in packed_eval:
        t_last, t_taps = teacher_forward(parts, ids, args, recorder)
        recorder.values.clear()
        recorder.active = True
        try:
            with shape(parts, args.student_passes, args.student_topk, lora=True, track_load=True):
                s_last = parts.text(input_ids=ids, use_cache=False).last_hidden_state
                s_taps = recorder.pop()
                hdim = s_last.shape[-1]
                kls.append(_kl_batch(parts, s_last.reshape(-1, hdim), t_last.reshape(-1, hdim), args))
                lm = layer_mse_loss(s_taps, t_taps)
                if lm is not None:
                    lmses.append(float(lm))
        finally:
            recorder.active = False
    for rb in ro_eval:
        t_last, _ = teacher_forward(parts, rb.t_ids, args, recorder, attention_mask=rb.t_attn)
        with shape(parts, args.student_passes, args.student_topk, lora=True, track_load=True):
            s_last = parts.text(input_ids=rb.s_ids, attention_mask=rb.s_attn, use_cache=False).last_hidden_state
            ro_kls.append(_kl_batch(parts, s_last[rb.s_mask], t_last[rb.t_mask], args))
    for rb in tool_if_eval or []:
        t_last, _ = teacher_forward(parts, rb.t_ids, args, recorder, attention_mask=rb.t_attn)
        with shape(parts, args.student_passes, args.student_topk, lora=True, track_load=True):
            s_last = parts.text(input_ids=rb.s_ids, attention_mask=rb.s_attn, use_cache=False).last_hidden_state
            tool_kls.append(_kl_batch(parts, s_last[rb.s_mask], t_last[rb.t_mask], args))
    cold_fracs = []
    for r in routers(parts):
        if r._load_tokens > 0:
            tot = float(r._load_tokens * r.top_k)
            cold_fracs.append(float(r._load_accum[~r.hot_mask].sum().item()) / max(tot, 1.0))
        r._load_accum.zero_()
        r._load_tokens = 0
    parts.full.train(was_training)
    mean_cf = sum(cold_fracs) / max(len(cold_fracs), 1)
    return {
        "eval_kl": round(sum(kls) / max(len(kls), 1), 4),
        "eval_ro_kl": round(sum(ro_kls) / max(len(ro_kls), 1), 4) if ro_kls else 0.0,
        "eval_tool_if_kl": round(sum(tool_kls) / max(len(tool_kls), 1), 4) if tool_kls else 0.0,
        "eval_layer_mse": round(sum(lmses) / max(len(lmses), 1), 4) if lmses else 0.0,
        "eval_cold_pct": round(100.0 * mean_cf, 2),
        "eval_cold_per_tok": round(mean_cf * len(routers(parts)) * args.student_topk, 2),
    }


@torch.no_grad()
def greedy_samples(parts, tokenizer, prompts, args, device) -> list[dict]:
    out = []
    was_training = parts.full.training
    parts.full.eval()
    for prompt in prompts:
        if tokenizer is None:
            ids = torch.tensor([[int(t) for t in prompt.split()]], device=device)
        else:
            text = render_chat(
                tokenizer,
                [{"role": "user", "content": prompt}, {"role": "assistant", "reasoning_content": " ", "content": ""}],
                reasoning_effort=0.9,
            )
            # Keep only the prompt side so generation continues the assistant turn.
            cut = text.find("<|content_thinking|>")
            text = text[: cut + len("<|content_thinking|>")] if cut >= 0 else text
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with shape(parts, args.student_passes, args.student_topk, lora=True):
            gen = parts.full.generate(input_ids=ids, max_new_tokens=args.gen_tokens, do_sample=False)
        new = gen[0, ids.shape[1]:].tolist()
        out.append({"prompt": prompt, "student": tokenizer.decode(new, skip_special_tokens=False) if tokenizer else new})
    parts.full.train(was_training)
    return out


# --------------------------------------------------------------------------------------------- chat render + data


def render_chat(tokenizer, messages, tools=None, reasoning_effort=0.0) -> str:
    """Always the tokenizer template. Never paste ``Thinking effort level`` into a system string."""
    kwargs = {"tokenize": False, "reasoning_effort": reasoning_effort}
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


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


def _bad_identity(text: str) -> bool:
    return bool(IDENTITY_RE.search(text or ""))


def _messages_text(messages) -> str:
    parts = []
    for m in messages or []:
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def format_gsm8k_row(row: dict, tokenizer, rng: random.Random) -> dict | None:
    q = str(row.get("question", "")).strip()
    a = str(row.get("answer", "")).strip()
    if not q or "####" not in a:
        return None
    steps, final = a.rsplit("####", 1)
    steps = CALC_ANN_RE.sub("", steps).strip()
    final = final.strip()
    if not steps or not final:
        return None
    eff = rng.choice(EFFORT_MATH)
    messages = [
        {"role": "user", "content": q + "\n\nEnd with: Answer: <final answer as a number>"},
        {"role": "assistant", "reasoning_content": steps, "content": f"Answer: {final}"},
    ]
    return {"text": render_chat(tokenizer, messages, reasoning_effort=eff), "is_gold": True, "has_thinking": True}


def format_math_row(row: dict, tokenizer, rng: random.Random) -> dict | None:
    prob = str(row.get("problem", "")).strip()
    sol = str(row.get("solution", "")).strip()
    if not prob or not sol:
        return None
    ans = _extract_boxed(sol)
    eff = rng.choice(EFFORT_MATH)
    messages = [
        {"role": "user", "content": prob},
        {"role": "assistant", "reasoning_content": sol, "content": f"Answer: {ans}"},
    ]
    return {"text": render_chat(tokenizer, messages, reasoning_effort=eff), "is_gold": True, "has_thinking": True}


def _clean_chat_messages(row: dict, drop_identity: bool, drop_code: bool) -> list[dict] | None:
    src = str(row.get("source", ""))
    if drop_code and (CODING_SOURCE_RE.search(src) or "hardcoded" in src.lower()):
        return None
    raw = row.get("messages") or []
    messages = []
    for m in raw:
        role = m.get("role")
        content = m.get("content")
        if role not in ("system", "user", "assistant", "tool") or not isinstance(content, str):
            continue
        item = {"role": role, "content": content}
        if m.get("reasoning_content"):
            item["reasoning_content"] = m["reasoning_content"]
        messages.append(item)
    if len(messages) < 2 or not any(m["role"] == "assistant" for m in messages):
        return None
    blob = _messages_text(messages)
    if drop_identity and _bad_identity(blob):
        return None
    if drop_code and CODING_SOURCE_RE.search(blob[:400]):
        return None
    return messages


def format_persona_math_row(row: dict, tokenizer, rng: random.Random) -> dict | None:
    messages = _clean_chat_messages(row, drop_identity=True, drop_code=True)
    if messages is None:
        return None
    text = render_chat(tokenizer, messages, reasoning_effort=rng.choice(EFFORT_MATH))
    return {"text": text, "is_gold": True, "has_thinking": "<|content_thinking|>" in text}


def format_if_row(row: dict, tokenizer, rng: random.Random) -> dict | None:
    messages = _clean_chat_messages(row, drop_identity=True, drop_code=False)
    if messages is None:
        return None
    eff = 0.0 if rng.random() < 0.80 else 0.2
    text = render_chat(tokenizer, messages, reasoning_effort=eff)
    return {"text": text, "is_gold": True, "has_thinking": False}


def format_tulu_mix_row(row: dict, tokenizer, rng: random.Random) -> dict | None:
    messages = _clean_chat_messages(row, drop_identity=True, drop_code=True)
    if messages is None:
        return None
    text = render_chat(tokenizer, messages, reasoning_effort=0.0)
    return {"text": text, "is_gold": False, "has_thinking": "<|content_thinking|>" in text}


def _json_objects(text: str) -> list:
    """Pull JSON objects out of a tool block. Strings are skipped so nested braces inside arguments stay intact."""
    raw = (text or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [obj for obj in parsed if isinstance(obj, dict)]
    objs = []
    i, n = 0, len(raw)
    while i < n:
        if raw[i] != "{":
            i += 1
            continue
        start, depth, in_str, esc = i, 0, False, False
        closed = False
        for k in range(i, n):
            ch = raw[k]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(raw[start:k + 1])
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict):
                        objs.append(obj)
                    i = k + 1
                    closed = True
                    break
        if not closed:
            break
    return objs


def _as_openai_tool(obj) -> dict | None:
    if not isinstance(obj, dict):
        return None
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return None
    params = fn.get("parameters") if "parameters" in fn else fn.get("params", {})
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            params = {}
    if not isinstance(params, dict):
        params = {}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": fn.get("description") or "",
            "parameters": params,
        },
    }


def _as_tool_call(obj, idx: int) -> dict | None:
    if not isinstance(obj, dict):
        return None
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
    name = obj.get("name") if isinstance(obj.get("name"), str) else fn.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments", obj.get("args", fn.get("arguments", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    return {"id": f"call_{idx}", "type": "function", "function": {"name": name, "arguments": args}}


def _tools_from_text(text: str) -> list[dict]:
    tools = []
    match = TOOL_BLOCK_RE.search(text or "")
    blob = match.group(1) if match else (text or "")
    parsed = None
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        parsed = _json_objects(blob)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if isinstance(parsed, list):
        for obj in parsed:
            tool = _as_openai_tool(obj)
            if tool is not None:
                tools.append(tool)
    return tools


def _tools_from_row(row: dict, system_text: str) -> list[dict]:
    raw = row.get("tools")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = None
    tools = []
    if isinstance(raw, list):
        for obj in raw:
            tool = _as_openai_tool(obj)
            if tool is not None:
                tools.append(tool)
    if not tools:
        tools = _tools_from_text(system_text)
    return tools


def _calls_from_text(text: str) -> list[dict]:
    calls = []
    blobs = TOOL_CALL_RE.findall(text or "")
    if not blobs:
        blobs = FUNCTION_CALL_RE.findall(text or "")
    if not blobs and "<functioncall>" in (text or "").lower():
        for part in re.split(r"<functioncall>", text, flags=re.IGNORECASE)[1:]:
            blobs.append(part.replace("<|endoftext|>", "").strip())
    for blob in blobs:
        for obj in _json_objects(blob):
            call = _as_tool_call(obj, len(calls))
            if call is not None:
                calls.append(call)
    return calls


def _strip_calls(text: str) -> str:
    text = TOOL_CALL_RE.sub("", text or "")
    text = FUNCTION_CALL_RE.sub("", text)
    if "<functioncall>" in text.lower():
        text = re.split(r"<functioncall>", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return text.replace("<|endoftext|>", "").strip()


def _native_and_prompt(tokenizer, system_text: str, turns: list[tuple[str, str]], tools: list[dict]):
    """50/50 is chosen by the caller. Both renders share one reasoning_effort of 0."""
    prompt_msgs = []
    if system_text.strip():
        prompt_msgs.append({"role": "system", "content": system_text.strip()})
    last_name = "tool"
    for role, val in turns:
        if role == "tool":
            prompt_msgs.append({"role": "tool", "name": last_name, "content": val})
        else:
            prompt_msgs.append({"role": role, "content": val})
            if role == "assistant":
                calls = _calls_from_text(val)
                if calls:
                    last_name = calls[-1]["function"]["name"]
    if not any(m["role"] == "assistant" for m in prompt_msgs):
        return None, None
    prompt = render_chat(tokenizer, prompt_msgs, reasoning_effort=0.0)

    native_msgs = []
    last_name = "tool"
    for role, val in turns:
        if role == "user":
            native_msgs.append({"role": "user", "content": val})
        elif role == "tool":
            native_msgs.append({"role": "tool", "name": last_name, "content": val})
        elif role == "assistant":
            calls = _calls_from_text(val)
            content = _strip_calls(val)
            msg = {"role": "assistant"}
            if content:
                msg["content"] = content
            if calls:
                msg["tool_calls"] = calls
                last_name = calls[-1]["function"]["name"]
            if "content" not in msg and "tool_calls" not in msg:
                continue
            native_msgs.append(msg)
    native = None
    if tools and any(m["role"] == "assistant" for m in native_msgs):
        native = render_chat(tokenizer, native_msgs, tools=tools, reasoning_effort=0.0)
    return native, prompt


def _pack_tool_row(native: str | None, prompt: str | None, which: str) -> dict | None:
    if which == "native" and native:
        return {"text": native, "is_gold": True, "has_thinking": False, "fmt": "native"}
    if which == "prompt" and prompt:
        return {"text": prompt, "is_gold": True, "has_thinking": False, "fmt": "prompt"}
    text = native or prompt
    if not text:
        return None
    return {"text": text, "is_gold": True, "has_thinking": False, "fmt": "native" if text == native else "prompt"}


def format_hermes_variants(row: dict, tokenizer) -> tuple[dict | None, dict | None]:
    convs = row.get("conversations") or row.get("messages") or []
    system_parts, turns = [], []
    for m in convs:
        fr = str(m.get("from", m.get("role", ""))).lower()
        val = m.get("value", m.get("content", ""))
        if not isinstance(val, str):
            continue
        val = val.strip()
        if not val:
            continue
        if fr in ("system",):
            system_parts.append(val)
        elif fr in ("human", "user"):
            turns.append(("user", val))
        elif fr in ("gpt", "assistant", "model"):
            turns.append(("assistant", val))
        elif fr in ("tool", "function"):
            turns.append(("tool", val))
    system_text = "\n".join(system_parts)
    tools = _tools_from_row(row, system_text)
    native, prompt = _native_and_prompt(tokenizer, system_text, turns, tools)
    return (
        _pack_tool_row(native, None, "native"),
        _pack_tool_row(None, prompt, "prompt"),
    )


def format_hermes_row(row: dict, tokenizer, rng: random.Random) -> dict | None:
    native, prompt = format_hermes_variants(row, tokenizer)
    if rng.random() < 0.5 and native is not None:
        return native
    return prompt or native


def _parse_glaive(text: str) -> tuple[str, list[tuple[str, str]]]:
    cleaned = (text or "").replace("<|endoftext|>", "").strip()
    parts = GLAIVE_SPLIT_RE.split(cleaned)
    system, turns = "", []
    i = 1
    while i + 1 < len(parts):
        marker, content = parts[i], parts[i + 1].strip()
        i += 2
        if not content:
            continue
        if marker == "SYSTEM:":
            system = content
        elif marker == "USER:":
            turns.append(("user", content))
        elif marker == "ASSISTANT:":
            turns.append(("assistant", content))
        elif marker == "FUNCTION RESPONSE:":
            turns.append(("tool", content))
    return system, turns


def format_glaive_variants(row: dict, tokenizer) -> tuple[dict | None, dict | None]:
    chat = row.get("chat") or row.get("text") or ""
    system = str(row.get("system") or "").replace("<|endoftext|>", "").strip()
    if system.startswith("SYSTEM:"):
        system = system[len("SYSTEM:"):].strip()
    if chat:
        parsed_system, turns = _parse_glaive(chat)
        system = parsed_system or system
    else:
        turns = []
    if not any(role == "assistant" for role, _ in turns):
        return None, None
    tools = _tools_from_text(system)
    native, prompt = _native_and_prompt(tokenizer, system, turns, tools)
    return (
        _pack_tool_row(native, None, "native"),
        _pack_tool_row(None, prompt, "prompt"),
    )


def format_glaive_row(row: dict, tokenizer, rng: random.Random) -> dict | None:
    native, prompt = format_glaive_variants(row, tokenizer)
    if rng.random() < 0.5 and native is not None:
        return native
    return prompt or native


def _label_text(label) -> str:
    if isinstance(label, str):
        low = label.strip().lower()
        if low in ("true", "false", "uncertain"):
            return low.capitalize() if low != "uncertain" else "Uncertain"
        return label.strip()
    if isinstance(label, bool):
        return "True" if label else "False"
    if label in (0, 1, 2):
        return ("False", "True", "Uncertain")[int(label)]
    return str(label)


def format_folio_row(row: dict, tokenizer, rng: random.Random) -> dict | None:
    premises = row.get("premises")
    if isinstance(premises, list):
        premises = "\n".join(str(p) for p in premises)
    premises = str(premises or "").strip()
    conclusion = str(row.get("conclusion", "")).strip()
    if not premises or not conclusion:
        return None
    label = _label_text(row.get("label", row.get("gold_label", "")))
    if label not in ("True", "False", "Uncertain"):
        return None
    p_fol = str(row.get("premises-FOL") or "").strip()
    c_fol = str(row.get("conclusion-FOL") or "").strip()
    reasoning = (
        f"Translate to first-order logic and check entailment.\nPremises FOL:\n{p_fol}\nConclusion FOL: {c_fol}\nTherefore the conclusion is {label}."
        if p_fol and c_fol
        else f"Evaluate whether the premises logically entail the conclusion: {conclusion}\nTherefore the conclusion is {label}."
    )
    messages = [
        {
            "role": "user",
            "content": (
                f"Premises:\n{premises}\n\nConclusion: {conclusion}\n\n"
                "Is the conclusion True, False, or Uncertain? End with Answer: True, Answer: False, or Answer: Uncertain."
            ),
        },
        {
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": f"Answer: {label}",
        },
    ]
    return {"text": render_chat(tokenizer, messages, reasoning_effort=rng.choice(EFFORT_LOGIC)), "is_gold": True, "has_thinking": True}


def format_sciq_row(row: dict, tokenizer, rng: random.Random) -> dict | None:
    question = str(row.get("question", "")).strip()
    correct = str(row.get("correct_answer", "")).strip()
    distractors = [str(row.get(k, "")).strip() for k in ("distractor1", "distractor2", "distractor3")]
    support = str(row.get("support", "")).strip()
    if not question or not correct or not all(distractors):
        return None
    options = [correct, *distractors]
    order = list(range(4))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    letter = "ABCD"[shuffled.index(correct)]
    lines = "\n".join(f"{'ABCD'[i]}. {opt}" for i, opt in enumerate(shuffled))
    messages = [
        {"role": "user", "content": f"{question}\n\n{lines}\n\nEnd with Answer: <letter>."},
        {"role": "assistant", "reasoning_content": support or "Use the scientific support to choose the option.", "content": f"Answer: {letter}"},
    ]
    return {"text": render_chat(tokenizer, messages, reasoning_effort=rng.choice(EFFORT_MATH)), "is_gold": True, "has_thinking": True}


def format_openthoughts_row(row: dict, tokenizer, rng: random.Random) -> dict | None:
    prob = str(row.get("problem") or "").strip()
    reas = str(row.get("deepseek_reasoning") or "").strip()
    sol = str(row.get("deepseek_solution") or row.get("ground_truth_solution") or "").strip()
    if not prob or not sol or _bad_identity(sol):
        return None
    if reas:
        # Cap overly verbose teacher reasoning traces so the completion fits comfortably inside seq_len.
        if len(reas) > 2200:
            reas = reas[:2200].rsplit(" ", 1)[0]
        messages = [
            {"role": "user", "content": prob},
            {"role": "assistant", "reasoning_content": reas, "content": sol},
        ]
        return {"text": render_chat(tokenizer, messages, reasoning_effort=rng.choice(EFFORT_LOGIC)), "is_gold": True, "has_thinking": True}
    messages = [{"role": "user", "content": prob}, {"role": "assistant", "content": sol}]
    return {"text": render_chat(tokenizer, messages, reasoning_effort=0.0), "is_gold": True, "has_thinking": False}


def format_logiqa2_row(row: dict, tokenizer, rng: random.Random) -> dict | None:
    prem = str(row.get("premise") or "").strip()
    hyp = str(row.get("hypothesis") or "").strip()
    lab_raw = str(row.get("label") or "").strip().lower()
    if not prem or not hyp or not lab_raw:
        return None
    label = "Entailment" if lab_raw in ("entailment", "entailed", "true", "1") else "Not Entailment"
    messages = [
        {
            "role": "user",
            "content": (
                f"Premise:\n{prem}\n\nHypothesis: {hyp}\n\n"
                "Does the hypothesis logically follow from the premise? End with Answer: Entailment or Answer: Not Entailment."
            ),
        },
        {
            "role": "assistant",
            "reasoning_content": f"Check strictly whether the premise entails the hypothesis without unstated assumptions.\nConclusion: {label}.",
            "content": f"Answer: {label}",
        },
    ]
    return {"text": render_chat(tokenizer, messages, reasoning_effort=rng.choice(EFFORT_LOGIC)), "is_gold": True, "has_thinking": True}


def format_arc_row(row: dict, tokenizer, rng: random.Random) -> dict | None:
    q = str(row.get("question") or "").strip()
    ch = row.get("choices") or {}
    texts = ch.get("text") if isinstance(ch, dict) else None
    labels = ch.get("label") if isinstance(ch, dict) else None
    ans = str(row.get("answerKey") or "").strip()
    if not q or not texts or not labels or not ans or len(texts) != len(labels):
        return None
    lines = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
    ans_txt = next((txt for lab, txt in zip(labels, texts) if str(lab).strip() == ans), "")
    messages = [
        {"role": "user", "content": f"{q}\n\n{lines}\n\nEnd with Answer: <letter>."},
        {
            "role": "assistant",
            "reasoning_content": f"Evaluate each scientific option carefully. Option {ans} ({ans_txt}) is best supported.",
            "content": f"Answer: {ans}",
        },
    ]
    return {"text": render_chat(tokenizer, messages, reasoning_effort=rng.choice(EFFORT_MATH)), "is_gold": True, "has_thinking": True}


def format_mmlu_row(row: dict, tokenizer, rng: random.Random) -> dict | None:
    q = str(row.get("question") or "").strip()
    choices = row.get("choices") or []
    ans_idx = row.get("answer")
    if not q or not isinstance(choices, list) or len(choices) < 2 or not isinstance(ans_idx, int) or not (0 <= ans_idx < len(choices)):
        return None
    letters = "ABCDEFGH"
    lines = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices[:len(letters)]))
    ans = letters[ans_idx]
    messages = [
        {"role": "user", "content": f"{q}\n\n{lines}\n\nEnd with Answer: <letter>."},
        {
            "role": "assistant",
            "reasoning_content": f"Analyze the question and eliminate incorrect options. Option {ans} ({choices[ans_idx]}) is correct.",
            "content": f"Answer: {ans}",
        },
    ]
    return {"text": render_chat(tokenizer, messages, reasoning_effort=rng.choice(EFFORT_LOGIC)), "is_gold": True, "has_thinking": True}


def load_dataset_retry(path: str, name: str | None = None, split: str = "train", streaming: bool = False):
    from datasets import load_dataset
    last = None
    for attempt in range(STREAM_RETRIES):
        try:
            if name is None:
                return load_dataset(path, split=split, streaming=streaming)
            return load_dataset(path, name, split=split, streaming=streaming)
        except Exception as err:
            last = err
            log.warning("HF load %s name=%s split=%s streaming=%s attempt %d/%d: %s", path, name, split, streaming, attempt + 1, STREAM_RETRIES, err)
            time.sleep(min(STREAM_BACKOFF_S, 2 * (attempt + 1)))
    log.error("HF load giving up %s: %s", path, last)
    return None


def _materialize(name: str, load_fn, limit: int | None = None) -> list:
    last = None
    for attempt in range(STREAM_RETRIES):
        rows: list = []
        try:
            ds = load_fn()
            if ds is None:
                raise RuntimeError(f"{name} loader returned None")
            for row in ds:
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
            log.info("stream %-16s loaded=%d", name, len(rows))
            return rows
        except Exception as err:
            last = err
            log.warning("stream %s attempt %d failed after %d rows: %s", name, attempt + 1, len(rows), err)
            if rows and attempt == STREAM_RETRIES - 1:
                log.info("stream %-16s loaded=%d (partial)", name, len(rows))
                return rows
            time.sleep(min(STREAM_BACKOFF_S, 2 * (attempt + 1)))
    log.error("stream %s unavailable: %s", name, last)
    return []


def _cycle(rows: list, seed: int, formatter) -> Iterator[dict]:
    if not rows:
        return
        yield  # pragma: no cover
    rng = random.Random(seed)
    while True:
        order = list(range(len(rows)))
        rng.shuffle(order)
        for i in order:
            out = formatter(rows[i], rng)
            if out is not None:
                yield out


def _holdout(rows: list, n: int) -> tuple[list, list]:
    if n <= 0 or len(rows) <= n * 2:
        return rows, []
    return rows[:-n], rows[-n:]


def prepare_rollout(text: str, is_gold: bool, has_thinking: bool, tokenizer, seq_len: int, teacher_note: str):
    split_at = text.find(MESSAGE_MODEL)
    if split_at < 0:
        return None
    prefix_end = split_at + len(MESSAGE_MODEL)
    s_prefix, completion = text[:prefix_end], text[prefix_end:]
    if not completion.strip():
        return None
    use_note = teacher_note if (teacher_note and has_thinking) else ""
    t_prefix = (
        f"<|message_system|><|content_text|>{use_note}{END_MESSAGE}{s_prefix}"
        if use_note else s_prefix
    )
    s_pre = tokenizer(s_prefix, add_special_tokens=False).input_ids
    t_pre = s_pre if not use_note else tokenizer(t_prefix, add_special_tokens=False).input_ids
    comp = tokenizer(completion, add_special_tokens=False).input_ids
    avail = seq_len - len(s_pre)
    if not s_pre or not t_pre or not comp or avail <= 4:
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


def rollout_batches(tokenizer, streams, weights, seq_len, batch, seed, teacher_note, pad_id, device):
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


def packed_mixed_batches(tokenizer, streams, weights, seq_len, batch, seed, device):
    picker = random.Random(seed + 13)
    buf, cur = [], []
    while True:
        row = next(picker.choices(streams, weights=weights)[0])
        text = row["text"]
        buf.extend(tokenizer(text, add_special_tokens=False).input_ids)
        while len(buf) >= seq_len:
            cur.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(cur) == batch:
                yield torch.tensor(cur, dtype=torch.long, device=device)
                cur = []


def _stream_local_jsonl(path: str, seed: int) -> Iterator[dict]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    log.info("stream %-16s loaded=%d", "rollouts", len(rows))
    rng = random.Random(seed + 71)
    while True:
        rng.shuffle(rows)
        for r in rows:
            txt = r.get("text")
            if isinstance(txt, str) and MESSAGE_MODEL in txt:
                yield {"text": txt, "is_gold": False, "has_thinking": "<|content_thinking|>" in txt}


def _eval_rollouts(rows: list[dict], tokenizer, args, pad_id: int) -> list[RolloutBatch]:
    items, batches = [], []
    for row in rows:
        item = prepare_rollout(row["text"], True, bool(row.get("has_thinking", False)), tokenizer, args.seq_len, "")
        if item is None:
            continue
        items.append(item)
        if len(items) == args.batch:
            batches.append(collate_rollouts(items, pad_id, args.device))
            items = []
            if len(batches) * args.batch >= args.eval_seqs:
                break
    return batches


def _eval_packed(rows: list[dict], tokenizer, args) -> list[torch.Tensor]:
    ev_buf, ev_batches = [], []
    for row in rows:
        ev_buf.extend(tokenizer(row["text"], add_special_tokens=False).input_ids)
        while len(ev_buf) >= args.seq_len and len(ev_batches) * args.batch < args.eval_seqs:
            if len(ev_batches) == 0 or len(ev_batches[-1]) == args.batch:
                ev_batches.append([])
            ev_batches[-1].append(ev_buf[:args.seq_len])
            ev_buf = ev_buf[args.seq_len:]
    return [torch.tensor(b, dtype=torch.long, device=args.device) for b in ev_batches if len(b) == args.batch]


def build_real_data_iters(tokenizer, args, pad_id: int):
    rng_eval = random.Random(args.seed + 7)

    def fmt_fixed(formatter):
        def _fmt(row, _rng):
            return formatter(row, tokenizer, rng_eval)
        return _fmt

    gsm_all = _materialize("gsm8k", lambda: load_dataset_retry("openai/gsm8k", "main", "train"))
    gsm_train, gsm_hold = _holdout(gsm_all, 100)
    math_all = []
    math_hold = []
    for sub in MATH_SUBSETS:
        rows = _materialize(f"math:{sub}", lambda sub=sub: load_dataset_retry("EleutherAI/hendrycks_math", sub, "train"))
        train, hold = _holdout(rows, 15)
        math_all.extend(train)
        math_hold.extend(hold)
    log.info("stream %-16s loaded=%d holdout=%d", "math", len(math_all), len(math_hold))
    tulu_math = _materialize(
        "tulu_math",
        lambda: load_dataset_retry("allenai/tulu-3-sft-personas-math", None, "train", streaming=True),
        limit=20000,
    )
    if_all = _materialize(
        "tulu_if",
        lambda: load_dataset_retry("allenai/tulu-3-sft-personas-instruction-following", None, "train"),
    )
    if_train, if_hold = _holdout(if_all, 64)
    hermes_train, hermes_hold = [], []
    for cfg in ("func_calling_singleturn", "func_calling", "glaive_func_calling"):
        rows = _materialize(
            f"hermes:{cfg}",
            lambda cfg=cfg: load_dataset_retry("NousResearch/hermes-function-calling-v1", cfg, "train"),
        )
        train, hold = _holdout(rows, 64)
        hermes_train.extend(train)
        hermes_hold.extend(hold)
    log.info("stream %-16s loaded=%d holdout=%d", "hermes", len(hermes_train), len(hermes_hold))
    glaive = _materialize(
        "glaive",
        lambda: load_dataset_retry("glaiveai/glaive-function-calling-v2", None, "train", streaming=True),
        limit=25000,
    )
    folio_all = _materialize("folio", lambda: load_dataset_retry("tasksource/folio", None, "train"))
    folio_train, folio_hold = _holdout(folio_all, 32)
    logiqa2_all = _materialize("logiqa2", lambda: load_dataset_retry("tasksource/logiqa-2.0-nli", None, "train"))
    logiqa2_train, logiqa2_hold = _holdout(logiqa2_all, 64)
    sciq_all = _materialize("sciq", lambda: load_dataset_retry("allenai/sciq", None, "train"))
    sciq_train, sciq_hold = _holdout(sciq_all, 64)
    arc_all = _materialize("arc_challenge", lambda: load_dataset_retry("allenai/ai2_arc", "ARC-Challenge", "train"))
    arc_train, _arc_hold = _holdout(arc_all, 32)
    mmlu_aux = _materialize(
        "mmlu_aux",
        lambda: load_dataset_retry("cais/mmlu", "all", "auxiliary_train", streaming=True),
        limit=15000,
    )
    ot_puzzle, ot_science = [], []
    ot_raw = _materialize(
        "openthoughts_tail",
        lambda: load_dataset_retry("open-thoughts/OpenThoughts-114k", "metadata", "train[104000:]", streaming=False),
    )
    for row in ot_raw:
        dom = str(row.get("domain") or "").lower()
        if dom == "puzzle":
            ot_puzzle.append(row)
        elif dom in ("science", "physics", "chemistry", "biology"):
            ot_science.append(row)
    log.info("stream %-16s loaded=%d | %-16s loaded=%d", "ot_puzzle", len(ot_puzzle), "ot_science", len(ot_science))
    tulu_mix = _materialize(
        "tulu_mix",
        lambda: load_dataset_retry("allenai/tulu-3-sft-mixture", None, "train", streaming=True),
        limit=20000,
    )

    banks = {
        "gsm8k": (gsm_train, lambda row, rng: format_gsm8k_row(row, tokenizer, rng)),
        "math": (math_all, lambda row, rng: format_math_row(row, tokenizer, rng)),
        "tulu_math": (tulu_math, lambda row, rng: format_persona_math_row(row, tokenizer, rng)),
        "ot_puzzle": (ot_puzzle, lambda row, rng: format_openthoughts_row(row, tokenizer, rng)),
        "logiqa2": (logiqa2_train, lambda row, rng: format_logiqa2_row(row, tokenizer, rng)),
        "folio": (folio_train, lambda row, rng: format_folio_row(row, tokenizer, rng)),
        "ot_science": (ot_science, lambda row, rng: format_openthoughts_row(row, tokenizer, rng)),
        "sciq": (sciq_train, lambda row, rng: format_sciq_row(row, tokenizer, rng)),
        "arc_challenge": (arc_train, lambda row, rng: format_arc_row(row, tokenizer, rng)),
        "mmlu_aux": (mmlu_aux, lambda row, rng: format_mmlu_row(row, tokenizer, rng)),
        "tulu_if": (if_train, lambda row, rng: format_if_row(row, tokenizer, rng)),
        "hermes": (hermes_train, lambda row, rng: format_hermes_row(row, tokenizer, rng)),
        "glaive": (glaive, lambda row, rng: format_glaive_row(row, tokenizer, rng)),
        "tulu_mix": (tulu_mix, lambda row, rng: format_tulu_mix_row(row, tokenizer, rng)),
    }
    streams, weights = [], []
    for i, (name, weight) in enumerate(MIX_WEIGHTS):
        if name == "rollouts":
            if args.rollouts_jsonl and Path(args.rollouts_jsonl).is_file():
                streams.append(_stream_local_jsonl(args.rollouts_jsonl, args.seed))
                weights.append(weight)
            else:
                log.warning("stream rollouts missing, dropping weight %.3f", weight)
            continue
        rows, formatter = banks[name]
        if not rows:
            log.warning("stream %s empty, dropping weight %.3f", name, weight)
            continue
        streams.append(_cycle(rows, args.seed + 17 * (i + 1), formatter))
        weights.append(weight)
    if not streams:
        raise RuntimeError("no training streams loaded")
    z = sum(weights)
    weights = [w / z for w in weights]
    log.info("training mix streams=%d weights=%s", len(streams), [round(w, 4) for w in weights])

    ro_it = rollout_batches(tokenizer, streams, weights, args.seq_len, args.batch, args.seed, args.teacher_note, pad_id, args.device)
    # A second pass of the same mix so packed and rollout draws do not share one generator.
    streams2, weights2 = [], []
    for i, (name, weight) in enumerate(MIX_WEIGHTS):
        if name == "rollouts":
            if args.rollouts_jsonl and Path(args.rollouts_jsonl).is_file():
                streams2.append(_stream_local_jsonl(args.rollouts_jsonl, args.seed + 3))
                weights2.append(weight)
            continue
        rows, formatter = banks[name]
        if not rows:
            continue
        streams2.append(_cycle(rows, args.seed + 100 + 17 * (i + 1), formatter))
        weights2.append(weight)
    z2 = sum(weights2)
    weights2 = [w / z2 for w in weights2]
    ord_it = packed_mixed_batches(tokenizer, streams2, weights2, args.seq_len, args.batch, args.seed, args.device)

    def mixed():
        rng = random.Random(args.seed + 99)
        while True:
            src = ro_it if rng.random() < args.rollout_frac else ord_it
            for _ in range(args.grad_accum):
                yield next(src)

    math_eval_rows = []
    for row in gsm_hold + math_hold + folio_hold + sciq_hold:
        if "question" in row and "answer" in row and "####" in str(row.get("answer", "")):
            out = format_gsm8k_row(row, tokenizer, rng_eval)
        elif "problem" in row and "solution" in row:
            out = format_math_row(row, tokenizer, rng_eval)
        elif "premises" in row:
            out = format_folio_row(row, tokenizer, rng_eval)
        elif "correct_answer" in row:
            out = format_sciq_row(row, tokenizer, rng_eval)
        else:
            out = None
        if out:
            math_eval_rows.append(out)
    packed_eval = _eval_packed(math_eval_rows, tokenizer, args)
    ro_eval = _eval_rollouts(math_eval_rows, tokenizer, args, pad_id)

    tool_rows = []
    for row in hermes_hold:
        native, prompt = format_hermes_variants(row, tokenizer)
        if native:
            tool_rows.append(native)
        if prompt:
            tool_rows.append(prompt)
    for row in if_hold:
        out = format_if_row(row, tokenizer, random.Random(0))
        if out:
            tool_rows.append(out)
    tool_if_eval = _eval_rollouts(tool_rows, tokenizer, args, pad_id)
    log.info(
        "eval batches packed=%d ro=%d tool_if=%d (math/reasoning rows=%d, tool/if rows=%d)",
        len(packed_eval), len(ro_eval), len(tool_if_eval), len(math_eval_rows), len(tool_rows),
    )
    return mixed(), packed_eval, ro_eval, tool_if_eval


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


def write_synthetic_qat(parts: Parts, hot_experts: list[list[int]], args, out: Path) -> tuple[Path, Path]:
    """Tiny F32 GGUF whose dense weights already include a run3-style LoRA, plus that LoRA as init-run3.pt."""
    gguf = import_gguf(args.llama_cpp_dir)
    gguf_path = out / "tiny_student.gguf"
    init_path = out / "tiny_run3_init.pt"
    gguf_path.unlink(missing_ok=True)
    init_path.unlink(missing_ok=True)
    mod_of = {module: name for name, module in parts.full.named_modules()}
    trainable = {}
    g = torch.Generator().manual_seed(args.seed + 99)
    writer = gguf.GGUFWriter(str(gguf_path), "inkling2_loop")
    for module, name in list(mod_of.items()):
        if not isinstance(module, LoRALinear) or name == "lm_head":
            continue
        # name is like model.layers.0.self_attn.q_proj.linear
        m = re.search(r"layers\.(\d+)\.(.+)$", name)
        if m is None:
            continue
        layer_idx = int(m.group(1))
        suffix = m.group(2)
        if suffix not in LORA_TARGETS:
            continue
        A = torch.zeros(module.lora_A.shape)
        B = torch.zeros(module.lora_B.shape)
        A[0].uniform_(-0.25, 0.25, generator=g)
        B[:, 0].uniform_(-0.25, 0.25, generator=g)
        trainable[name + ".lora_A"] = A
        trainable[name + ".lora_B"] = B
        w_bf = module.base.weight.detach().to(torch.bfloat16).float()
        # run3 scale is 1 (alpha == rank in the synthetic meta). The GGUF stores the merged weight.
        w_gguf = (w_bf + B @ A).numpy().astype(np.float32)
        writer.add_tensor(f"blk.{layer_idx}.{LORA_TARGETS[suffix]}.weight", np.ascontiguousarray(w_gguf))
    for i, layer in enumerate(parts.text.layers):
        exp = layer.mlp.experts
        hot = hot_experts[i]
        cold = [e for e in range(exp.num_experts) if e not in set(hot)]
        order = list(hot) + cold
        for stem, param in (("ffn_gate_up_exps", exp.gate_up_proj), ("ffn_down_exps", exp.down_proj)):
            base = param.detach().float().cpu().numpy()
            stack = np.empty_like(base)
            for new_i, old_i in enumerate(order):
                stack[new_i] = base[old_i]
            writer.add_tensor(f"blk.{i}.{stem}.weight", np.ascontiguousarray(stack))
            writer.add_tensor(f"blk.{i}.{stem}.scale", np.ones((base.shape[0],), np.float32))
    head = parts.head_owner.lm_head
    head_w = head.base.weight.detach().float().numpy()
    pad = np.zeros((parts.text.config.vocab_size - head_w.shape[0], head_w.shape[1]), np.float32)
    writer.add_tensor("output.weight", np.ascontiguousarray(np.concatenate([head_w, pad])))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    for i, router in enumerate(routers(parts)):
        rname = mod_of[router]
        rank = 4
        A = torch.zeros(rank, router.hidden_dim)
        B = torch.zeros(router.num_experts, rank)
        A[0].uniform_(-0.25, 0.25, generator=g)
        B[:, 0].uniform_(-0.25, 0.25, generator=g)
        trainable[rname + ".lora_A"] = A
        trainable[rname + ".lora_B"] = B
    blob = {
        "trainable": trainable,
        "meta": {"lora_rank": 4, "lora_alpha": 4.0, "router_rank": 4, "router_alpha": 4.0, "mode": "run3"},
        "state": {"step": 0, "seqs": 0, "hours": 0.0},
    }
    torch.save(blob, init_path)
    log.info("wrote synthetic QAT GGUF %s and run3 init %s (%d LoRA tensors)", gguf_path, init_path, len(trainable))
    return gguf_path, init_path


# ---------------------------------------------------------------------------------------- ckpt & training


def save_ckpt(path: Path, parts: Parts, trainable: dict, hot_experts, opt, sched, state: dict, args) -> None:
    path.mkdir(parents=True, exist_ok=True)
    bias_deltas = {f"blk.{i}.student_bias_delta": r.student_bias_delta.detach().cpu() for i, r in enumerate(routers(parts))}
    tmp = path / "ckpt.pt.tmp"
    torch.save({
        "trainable": {k: v.detach().cpu() for k, v in trainable.items()},
        "bias_deltas": bias_deltas,
        "hot_experts": hot_experts,
        "meta": {
            "mode": "workingset_v2_qat",
            "n_hot": len(hot_experts[0]),
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "expert_lora_rank": args.expert_lora_rank,
            "expert_lora_alpha": args.expert_lora_alpha,
            "train_small_dense_deltas": bool(args.train_small_dense_deltas),
            "lora_lm_head": bool(args.lora_lm_head),
            "model": args.model,
            "revision": args.revision,
            "gguf_experts_reordered": bool(args.gguf_experts_reordered),
            "student_from_gguf": args.student_from_gguf or "",
        },
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "state": state,
    }, tmp)
    os.replace(tmp, path / "ckpt.pt")
    (path / "state.json").write_text(json.dumps(state, indent=1))


def load_ckpt(path: Path, parts: Parts, trainable: dict, opt, sched) -> dict:
    blob = torch.load(path / "ckpt.pt", map_location="cpu", weights_only=False)
    missing = []
    with torch.no_grad():
        for k, v in blob["trainable"].items():
            if k not in trainable:
                missing.append(k)
                continue
            if tuple(trainable[k].shape) != tuple(v.shape):
                raise ValueError(f"resume shape mismatch {k}: ckpt {tuple(v.shape)} vs model {tuple(trainable[k].shape)}")
            trainable[k].copy_(v.to(device=trainable[k].device, dtype=trainable[k].dtype))
        for i, r in enumerate(routers(parts)):
            bk = f"blk.{i}.student_bias_delta"
            if bk in blob.get("bias_deltas", {}):
                r.student_bias_delta.copy_(blob["bias_deltas"][bk].to(r.student_bias_delta.device))
    if missing:
        log.warning("resume ckpt had %d keys not in this model (first=%s)", len(missing), missing[0])
    opt.load_state_dict(blob["opt"])
    sched.load_state_dict(blob["sched"])
    return blob["state"]


def _group_of(name: str) -> str:
    if _is_small_delta_name(name):
        return "small_dense"
    if "student_weight_delta" in name:
        return "router"
    if "gu_lora_" in name or "dn_lora_" in name:
        return "expert"
    if name == "lm_head.lora_A" or name == "lm_head.lora_B" or name.startswith("lm_head."):
        return "lm_head"
    if "lora_" in name:
        return "attn"
    return "other"


def build_optimizer(trainable: dict, args):
    buckets = {"attn": [], "lm_head": [], "expert": [], "router": [], "small_dense": []}
    for name, param in trainable.items():
        buckets[_group_of(name)].append(param)
    spec = (
        ("attn", args.lr, args.weight_decay),
        ("lm_head", args.lr, args.weight_decay),
        ("expert", args.expert_lr, args.weight_decay),
        ("router", args.router_lr, 0.0),
        ("small_dense", args.small_dense_lr, 0.0),
    )
    groups = [{"params": buckets[name], "lr": lr, "weight_decay": wd, "name": name} for name, lr, wd in spec if buckets[name]]
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


def _grad_norm(trainable: dict, pred) -> float:
    total = 0.0
    for name, param in trainable.items():
        if pred(name) and param.grad is not None:
            total += float(param.grad.detach().float().norm())
    return total


def dry_run_checks(parts, hot_experts, args, recorder, packed_eval, ro_eval, tool_if_eval, trainable, opt, sched, out: Path) -> None:
    ids = packed_eval[0]
    # Index helper: reordered GGUF position 0 is hot[0], not expert 0.
    assert student_expert_gguf_index(0, hot_experts[0], True) == 0
    assert student_expert_gguf_index(0, hot_experts[0], False) == hot_experts[0][0]
    exp0 = parts.text.layers[0].mlp.experts
    got = exp0.student_hot_gu[0].float()
    ref = exp0.gate_up_proj[hot_experts[0][0]].detach().float()
    other = exp0.gate_up_proj[0].detach().float()
    err_ref = float((got - ref).abs().mean())
    err_other = float((got - other).abs().mean())
    rel = err_ref / max(1e-6, float(ref.abs().mean()))
    log.info("CHECK hot-0 QAT slice matches HF expert %d (rel rms %.4f, closer than expert 0: %s)", hot_experts[0][0], rel, err_ref < err_other)
    assert rel < 0.05, rel
    assert err_ref < err_other

    # 24-layer edge relaxation averages back to the global cold target.
    probe = argparse.Namespace(target_cold_frac=0.065, edge_cold_mult=1.30, start_cold_frac=0.30, cold_ramp_steps=220)
    tgts = [layer_final_target(i, 24, probe) for i in range(24)]
    assert abs(sum(tgts) / 24 - 0.065) < 1e-12
    assert abs(tgts[0] - 0.0845) < 1e-12 and abs(tgts[2] - 0.0611) < 1e-12
    assert abs(current_layer_cold_target(0, 0, 24, probe) - 0.30) < 1e-12
    log.info("CHECK layer-aware cold targets mean=%.4f edge=%.4f mid=%.4f", sum(tgts) / 24, tgts[0], tgts[2])

    with torch.no_grad(), shape(parts, args.teacher_passes, args.teacher_topk, lora=False):
        teacher_before = parts.full(input_ids=ids, use_cache=False).logits
    with torch.no_grad(), shape(parts, args.student_passes, args.student_topk, lora=True):
        student_before = parts.full(input_ids=ids, use_cache=False).logits
    saved_delta = {}
    with torch.no_grad():
        for name, param in parts.full.named_parameters():
            if _is_small_delta_name(name):
                saved_delta[name] = param.detach().clone()
                param.add_(0.25)
    with torch.no_grad(), shape(parts, args.teacher_passes, args.teacher_topk, lora=False):
        teacher_delta = parts.full(input_ids=ids, use_cache=False).logits
    with torch.no_grad(), shape(parts, args.student_passes, args.student_topk, lora=True):
        student_delta = parts.full(input_ids=ids, use_cache=False).logits
    tdiff = float((teacher_before - teacher_delta).abs().max())
    sdiff = float((student_before - student_delta).abs().max())
    log.info("CHECK teacher invariant to student_delta += 0.25: max|diff| = %.3g (student moved %.3g)", tdiff, sdiff)
    assert tdiff == 0.0, tdiff
    assert sdiff > 0.0 and math.isfinite(sdiff)
    with torch.no_grad():
        named = dict(parts.full.named_parameters())
        for name, value in saved_delta.items():
            named[name].copy_(value)

    saved, saved_b = {}, {}
    saved_base, saved_hot = [], []
    with torch.no_grad():
        for name, param in parts.full.named_parameters():
            if "lora_B" in name or "student_weight_delta" in name or _is_small_delta_name(name):
                saved[name] = param.detach().clone()
                param.add_(0.25)
        for i, r in enumerate(routers(parts)):
            saved_b[i] = r.student_bias_delta.clone()
            r.student_bias_delta.add_(-0.5)
        for module in parts.full.modules():
            if isinstance(module, LoRALinear) and module.student_kind == "dense":
                saved_base.append((module, module.student_base.detach().clone()))
                module.student_base.add_(0.25)
        for layer in parts.text.layers:
            exp = layer.mlp.experts
            if getattr(exp, "student_hot_gu", None) is not None:
                saved_hot.append((exp, exp.student_hot_gu.q.detach().clone(), exp.student_hot_dn.q.detach().clone()))
                exp.student_hot_gu.q.bitwise_xor_(torch.tensor(0x0F, dtype=torch.int8, device=exp.student_hot_gu.q.device))
                exp.student_hot_dn.q.bitwise_xor_(torch.tensor(0x0F, dtype=torch.int8, device=exp.student_hot_dn.q.device))
    with torch.no_grad(), shape(parts, args.teacher_passes, args.teacher_topk, lora=False):
        teacher_all = parts.full(input_ids=ids, use_cache=False).logits
    with torch.no_grad():
        named = dict(parts.full.named_parameters())
        for name, value in saved.items():
            named[name].copy_(value)
        for i, r in enumerate(routers(parts)):
            r.student_bias_delta.copy_(saved_b[i])
        for module, value in saved_base:
            module.student_base.copy_(value)
        for exp, gu, dn in saved_hot:
            exp.student_hot_gu.q.copy_(gu)
            exp.student_hot_dn.q.copy_(dn)
    tdiff = float((teacher_before - teacher_all).abs().max())
    log.info("CHECK teacher invariant to QAT base + LoRA + router + bias + small-dense: max|diff| = %.3g", tdiff)
    assert tdiff == 0.0, tdiff

    rargs = argparse.Namespace(**{
        **vars(args),
        "lr": 3e-3, "expert_lr": 3e-3, "router_lr": 5e-3, "small_dense_lr": 3e-3,
        "max_steps": 40, "warmup": 2, "cold_loss": 0.5, "cold_loss_warmup": 1,
        "start_cold_frac": 0.20, "target_cold_frac": 0.05, "cold_ramp_steps": 15,
        "cold_bias_gamma": 0.08, "cold_bias_min": -0.75, "balance_gamma": 0.005, "reverse_kl": 0.0,
    })
    set_resting_shape(parts, rargs.student_passes, rargs.student_topk)
    trainable = freeze_and_select(parts)
    opt, sched = build_optimizer(trainable, rargs)
    assert len(opt.param_groups) == 5, [g.get("name") for g in opt.param_groups]
    fixed_batch = packed_eval[0]
    ev_before = evaluate(parts, packed_eval, ro_eval, rargs, recorder, tool_if_eval)
    assert "eval_tool_if_kl" in ev_before
    opt.zero_grad(set_to_none=True)
    loss0, stats0 = distill_loss(parts, fixed_batch, rargs, recorder, 0.5)
    assert "cold_mass" in stats0 and loss0.requires_grad
    loss0.backward()
    group_pred = {
        "attn": lambda n: _group_of(n) == "attn" and n.endswith("lora_B"),
        "lm_head": lambda n: _group_of(n) == "lm_head" and n.endswith("lora_B"),
        "expert": lambda n: "gu_lora_B" in n or "dn_lora_B" in n,
        "router": lambda n: "student_weight_delta" in n,
        "small_dense": _is_small_delta_name,
    }
    for name, pred in group_pred.items():
        gn = _grad_norm(trainable, pred)
        log.info("CHECK grad norm [%s] = %.4f", name, gn)
        assert gn > 0.0 and math.isfinite(gn), f"zero/bad grad on {name}: {gn}"
    for tag in ("gu_lora_B", "dn_lora_B"):
        gn = _grad_norm(trainable, lambda n, tag=tag: tag in n)
        assert gn > 0.0 and math.isfinite(gn), tag
    opt.zero_grad(set_to_none=True)
    LoRAState.collect_cold_loss = True
    try:
        parts.text(input_ids=fixed_batch, use_cache=False)
        cl_only = pop_cold_loss(parts)
    finally:
        LoRAState.collect_cold_loss = False
    assert cl_only is not None and cl_only.requires_grad
    cl_only.backward()
    rtr_cold_gn = _grad_norm(trainable, lambda n: "student_weight_delta" in n)
    log.info("CHECK cold_loss-only grad norm on student_weight_delta = %.4f", rtr_cold_gn)
    assert rtr_cold_gn > 0.0 and math.isfinite(rtr_cold_gn)
    opt.zero_grad(set_to_none=True)
    l0 = float(loss0.detach())
    for st in range(35):
        train_step(parts, itertools.repeat(fixed_batch), opt, sched, trainable, st, rargs, recorder)
    ev_after = evaluate(parts, [fixed_batch], [], rargs, recorder, [])
    with torch.no_grad():
        l1 = float(distill_loss(parts, fixed_batch, rargs, recorder, 0.5)[0])
    log.info(
        "CHECK WSD overfit & concentration: loss %.4f->%.4f | cold_pct %.1f%%->%.1f%%",
        l0, l1, ev_before["eval_cold_pct"], ev_after["eval_cold_pct"],
    )
    assert l1 < 0.55 * l0, f"overfit failed: {l0:.4f} -> {l1:.4f}"
    assert ev_after["eval_cold_pct"] <= 10.0, f"cold concentration failed: {ev_before} -> {ev_after}"
    assert float(routers(parts)[0].student_bias_delta.min()) >= args.cold_bias_min - 1e-6

    worst_rel = quantize_hf_experts_int8(parts, args.device)
    log.info("CHECK quantize_hf_experts_int8 worst_rel=%.4f", worst_rel)
    assert worst_rel < 0.03
    stats = train_step(parts, itertools.repeat(fixed_batch), opt, sched, trainable, 36, rargs, recorder)
    assert math.isfinite(stats["loss"]) and stats["grad_norm"] > 0
    log.info("CHECK int8 WSD train step: loss=%.4f cold_pct=%.1f%%", stats["loss"], stats["cold_pct"])

    resume_state = {"step": 123, "seqs": 456, "hours": 0.25}
    save_ckpt(out, parts, trainable, hot_experts, opt, sched, resume_state, args)
    snap = {k: v.detach().clone() for k, v in trainable.items()}
    bias_snap = [r.student_bias_delta.detach().clone() for r in routers(parts)]
    with torch.no_grad():
        for param in trainable.values():
            param.add_(0.5)
        for r in routers(parts):
            r.student_bias_delta.add_(0.25)
    restored = load_ckpt(out, parts, trainable, opt, sched)
    for key, value in snap.items():
        if not torch.equal(trainable[key], value):
            raise AssertionError(f"resume mismatch on {key}")
    for router, saved_bias in zip(routers(parts), bias_snap):
        assert torch.equal(router.student_bias_delta, saved_bias)
    assert restored["step"] == 123 and restored["seqs"] == 456
    log.info("CHECK resume restored step=%d seqs=%d tensors=%d bias=%d", restored["step"], restored["seqs"], len(snap), len(bias_snap))
    log.info("DRY RUN PASSED ALL CHECKS")


EVAL_PROMPTS = [
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "If 3 machines make 3 widgets in 3 minutes, how many minutes does it take 100 machines to make 100 widgets?",
    "Write a 2-sentence warm reply to a friend who just passed their driving test.",
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    a = p.add_argument
    a("--dry-run-tiny", action="store_true")
    a("--verify-formats", action="store_true")
    a("--model", default=MODEL_ID)
    a("--revision", default=MODEL_REVISION)
    a("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a("--hot-plan", default=None)
    a("--out", default="runs/distill_v2")
    a("--resume", action="store_true")
    a("--init-from", default=None)
    a("--student-from-gguf", default=None)
    a("--llama-cpp-dir", default=str(Path(__file__).resolve().parent.parent / "llama.cpp"))
    a("--gguf-experts-reordered", action=argparse.BooleanOptionalAction, default=True)
    a("--gguf-includes-init-dense", action=argparse.BooleanOptionalAction, default=None)
    a("--train-small-dense-deltas", action=argparse.BooleanOptionalAction, default=True)
    a("--experts-int8-from-hf", action=argparse.BooleanOptionalAction, default=True)
    a("--max-hours", type=float, default=4.0)
    a("--max-steps", type=int, default=550)
    a("--seq-len", type=int, default=1024)
    a("--batch", type=int, default=2)
    a("--grad-accum", type=int, default=4)
    a("--lr", type=float, default=8e-5)
    a("--expert-lr", type=float, default=1.8e-4)
    a("--router-lr", type=float, default=1.2e-4)
    a("--small-dense-lr", type=float, default=5e-5)
    a("--balance-gamma", type=float, default=0.001)
    a("--cold-loss", type=float, default=0.08)
    a("--cold-loss-warmup", type=int, default=60)
    a("--start-cold-frac", type=float, default=0.30)
    a("--target-cold-frac", type=float, default=0.065)
    a("--edge-cold-mult", type=float, default=1.30)
    a("--cold-ramp-steps", type=int, default=220)
    a("--cold-bias-gamma", type=float, default=0.005)
    a("--cold-bias-min", type=float, default=-0.75)
    a("--rollout-frac", type=float, default=0.50)
    a("--warmup", type=int, default=25)
    a("--weight-decay", type=float, default=0.0)
    a("--clip", type=float, default=1.0)
    a("--beta2", type=float, default=0.99)
    a("--seed", type=int, default=42)
    a("--teacher-passes", type=int, default=3)
    a("--teacher-topk", type=int, default=8)
    a("--student-passes", type=int, default=1)
    a("--student-topk", type=int, default=4)
    a("--lora-rank", type=int, default=32)
    a("--lora-alpha", type=float, default=32.0)
    a("--expert-lora-rank", type=int, default=24)
    a("--expert-lora-alpha", type=float, default=48.0)
    a("--lora-lm-head", action=argparse.BooleanOptionalAction, default=True)
    a("--reverse-kl", type=float, default=0.25)
    a("--rollout-forward-kl", type=float, default=1.0)
    a("--rollout-reverse-kl", type=float, default=0.35)
    a("--sft-ce", type=float, default=0.0)
    a("--layer-mse", type=float, default=0.20)
    a("--teacher-note", default="")
    a("--rollouts-jsonl", default=None)
    a("--kl-temp", type=float, default=1.0)
    a("--logit-chunk", type=int, default=512)
    a("--grad-ckpt", action=argparse.BooleanOptionalAction, default=True)
    a("--eval-every", type=int, default=50)
    a("--eval-seqs", type=int, default=16)
    a("--save-every", type=int, default=50)
    a("--gen-tokens", type=int, default=192)
    args = p.parse_args(argv)
    if args.dry_run_tiny:
        args.device = "cpu"
        args.max_steps, args.seq_len, args.batch, args.grad_accum = 10, 32, 2, 1
        args.eval_every, args.save_every, args.eval_seqs, args.gen_tokens = 5, 10, 4, 8
        args.lr, args.expert_lr, args.router_lr, args.small_dense_lr = 1e-3, 1e-3, 2e-3, 1e-3
        args.warmup, args.logit_chunk = 2, 16
        # scale = 2, while the synthetic run3 checkpoint uses scale 1, so B is rescaled.
        args.lora_rank, args.lora_alpha = 4, 8.0
        args.expert_lora_rank, args.expert_lora_alpha = 4, 8.0
        args.lora_lm_head = True
        args.grad_ckpt = True
        args.train_small_dense_deltas = True
        args.out = str(Path(__file__).resolve().parent / "runs" / "dry-run-v2-sota")
    return args


def _assert_effort_once(text: str, label: str) -> None:
    n = text.count("Thinking effort level:")
    if n != 1:
        raise AssertionError(f"{label}: expected one Thinking effort header, found {n}\n{text[:500]}")


def _assert_completion(tokenizer, text: str, label: str) -> None:
    item = prepare_rollout(text, True, "<|content_thinking|>" in text, tokenizer, 4096, "")
    if item is None or item[4] <= 0:
        raise AssertionError(f"{label}: empty completion span")
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        raise AssertionError(f"{label}: tokenizer returned no ids")


def verify_formats(tokenizer_dir: Path = TOKENIZER_DIR) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    rng = random.Random(0)

    gsm = format_gsm8k_row(
        {"question": "What is 2+2?", "answer": "Two plus two is four.\n#### 4"},
        tokenizer, rng,
    )
    assert gsm and "<|content_thinking|>" in gsm["text"] and "Answer: 4" in gsm["text"]
    _assert_effort_once(gsm["text"], "gsm8k")
    assert "Thinking effort level:" in gsm["text"].split("<|message_user|>")[0]
    assert "Thinking effort level:" not in gsm["text"].split("<|content_thinking|>", 1)[0].split("<|message_system|>")[-1].replace("Thinking effort level:", "", 1)
    _assert_completion(tokenizer, gsm["text"], "gsm8k")

    math_row = format_math_row(
        {"problem": "Compute 3*3.", "solution": "Multiply three by three.\n\\boxed{9}"},
        tokenizer, rng,
    )
    assert math_row and "Answer: 9" in math_row["text"] and "<|content_thinking|>" in math_row["text"]
    _assert_effort_once(math_row["text"], "math")
    _assert_completion(tokenizer, math_row["text"], "math")

    persona = format_persona_math_row(
        {"messages": [
            {"role": "user", "content": "A person has 3 apples and gets 2 more. How many apples?"},
            {"role": "assistant", "content": "Answer: 5"},
        ]},
        tokenizer, rng,
    )
    assert persona
    _assert_effort_once(persona["text"], "tulu-math")
    _assert_completion(tokenizer, persona["text"], "tulu-math")
    assert format_persona_math_row(
        {"source": "code", "messages": [
            {"role": "user", "content": "write python"},
            {"role": "assistant", "content": "```python\nprint(1)\n```"},
        ]},
        tokenizer, rng,
    ) is None

    if_row = format_if_row(
        {"messages": [
            {"role": "system", "content": "Follow the constraint exactly."},
            {"role": "user", "content": "Reply with exactly the word: ok"},
            {"role": "assistant", "content": "ok"},
        ]},
        tokenizer, random.Random(1),
    )
    assert if_row and "ok" in if_row["text"]
    _assert_effort_once(if_row["text"], "tulu-if")
    _assert_completion(tokenizer, if_row["text"], "tulu-if")

    hermes = {
        "conversations": [
            {"from": "system", "value": 'You call tools.\n<tools>[{"type":"function","function":{"name":"get_weather","description":"Weather lookup","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]</tools>'},
            {"from": "human", "value": "Weather in Paris?"},
            {"from": "gpt", "value": '<tool_call>{"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}</tool_call>'},
            {"from": "tool", "value": '{"temp": 18}'},
            {"from": "gpt", "value": "It is 18 degrees in Paris."},
        ]
    }
    native, prompt = format_hermes_variants(hermes, tokenizer)
    assert native and prompt
    assert "<|message_system|>tool_declare<|content_xml|>" in native["text"]
    assert "<|content_invoke_tool_json|>" in native["text"]
    assert "get_weather" in native["text"] and "Paris" in native["text"]
    assert native["text"].count("<|message_system|>tool_declare") == 1
    _assert_effort_once(native["text"], "hermes-native")
    declare = native["text"].split("<|end_message|>", 1)[0]
    assert "tool_declare" in declare and "Thinking effort level" not in declare
    _assert_completion(tokenizer, native["text"], "hermes-native")
    assert "tool_declare" not in prompt["text"]
    assert "<tool_call>" in prompt["text"]
    _assert_effort_once(prompt["text"], "hermes-prompt")
    _assert_completion(tokenizer, prompt["text"], "hermes-prompt")

    glaive_tool = {
        "chat": (
            'SYSTEM: You are a helpful assistant. Use functions if required -\n'
            '{"name": "add", "description": "add two numbers", "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}\n'
            'USER: What is 2+2?\n'
            'ASSISTANT: <functioncall> {"name": "add", "arguments": {"a": 2, "b": 2}} </functioncall>\n'
            'FUNCTION RESPONSE: {"result": 4}\n'
            'ASSISTANT: The sum is 4.'
        )
    }
    g_native, g_prompt = format_glaive_variants(glaive_tool, tokenizer)
    assert g_native and "<|message_system|>tool_declare<|content_xml|>" in g_native["text"]
    assert "<|content_invoke_tool_json|>" in g_native["text"]
    _assert_effort_once(g_native["text"], "glaive-native")
    _assert_completion(tokenizer, g_native["text"], "glaive-native")
    assert g_prompt and "tool_declare" not in g_prompt["text"] and "<functioncall>" in g_prompt["text"]
    _assert_effort_once(g_prompt["text"], "glaive-prompt")
    _assert_completion(tokenizer, g_prompt["text"], "glaive-prompt")

    glaive_irr = {
        "chat": (
            'SYSTEM: You are a helpful assistant. Use functions if required -\n'
            '{"name": "add", "description": "add two numbers", "parameters": {"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]}}\n'
            'USER: Tell me a short joke about books.\n'
            'ASSISTANT: Why did the book go to the library? It needed a new chapter.'
        )
    }
    _irr_native, irr_prompt = format_glaive_variants(glaive_irr, tokenizer)
    assert irr_prompt and "tool_declare" not in irr_prompt["text"]
    assert "<|content_invoke_tool_json|>" not in irr_prompt["text"]
    assert "joke" in irr_prompt["text"] or "chapter" in irr_prompt["text"]
    _assert_effort_once(irr_prompt["text"], "glaive-irrelevance")
    _assert_completion(tokenizer, irr_prompt["text"], "glaive-irrelevance")

    folio = format_folio_row(
        {"premises": ["All humans are mortal.", "Socrates is a human."], "conclusion": "Socrates is mortal.", "label": "True"},
        tokenizer, rng,
    )
    assert folio and "Answer: True" in folio["text"] and "<|content_thinking|>" in folio["text"]
    _assert_effort_once(folio["text"], "folio")
    _assert_completion(tokenizer, folio["text"], "folio")

    sciq = format_sciq_row(
        {"question": "What is H2O?", "correct_answer": "water", "distractor1": "gold", "distractor2": "salt", "distractor3": "iron", "support": "Water is the molecule H2O."},
        tokenizer, random.Random(2),
    )
    assert sciq and "Water is the molecule H2O." in sciq["text"] and "Answer: " in sciq["text"]
    _assert_effort_once(sciq["text"], "sciq")
    _assert_completion(tokenizer, sciq["text"], "sciq")

    ot = format_openthoughts_row(
        {"problem": "Three boxes are mislabeled...", "deepseek_reasoning": "Pick from the mixed box.", "deepseek_solution": "Open the mixed box first."},
        tokenizer, rng,
    )
    assert ot and "<|content_thinking|>" in ot["text"]
    _assert_effort_once(ot["text"], "openthoughts")
    _assert_completion(tokenizer, ot["text"], "openthoughts")

    lq2 = format_logiqa2_row(
        {"premise": "All cats sleep.", "hypothesis": "Some cats sleep.", "label": "entailment"},
        tokenizer, rng,
    )
    assert lq2 and "Answer: Entailment" in lq2["text"]
    _assert_effort_once(lq2["text"], "logiqa2")
    _assert_completion(tokenizer, lq2["text"], "logiqa2")

    arc = format_arc_row(
        {"question": "Which produces heat?", "choices": {"text": ["dry palms", "wet palms"], "label": ["A", "B"]}, "answerKey": "A"},
        tokenizer, rng,
    )
    assert arc and "Answer: A" in arc["text"]
    _assert_effort_once(arc["text"], "arc")
    _assert_completion(tokenizer, arc["text"], "arc")

    mmlu = format_mmlu_row(
        {"question": "Capital of France?", "choices": ["Berlin", "Paris", "Rome", "Madrid"], "answer": 1},
        tokenizer, rng,
    )
    assert mmlu and "Answer: B" in mmlu["text"]
    _assert_effort_once(mmlu["text"], "mmlu")
    _assert_completion(tokenizer, mmlu["text"], "mmlu")

    glaive_v2_real = {
        "system": 'SYSTEM: You are a helpful assistant with access to the following functions. Use them if required - {"name": "get_exchange_rate", "description": "Get rate", "parameters": {"type": "object", "properties": {"base": {"type": "string"}}, "required": ["base"]}}',
        "chat": 'USER: Rate for USD?\nASSISTANT: <functioncall> {"name": "get_exchange_rate", "arguments": "{\\"base\\": \\"USD\\"}"} <|endoftext|>\nFUNCTION RESPONSE: {"rate": 1.0}\nASSISTANT: The rate is 1.0. <|endoftext|>',
    }
    gv2_nat, gv2_prm = format_glaive_variants(glaive_v2_real, tokenizer)
    assert gv2_nat and "<|content_invoke_tool_json|>" in gv2_nat["text"] and "<|endoftext|>" not in gv2_nat["text"]
    assert gv2_prm and "<|endoftext|>" not in gv2_prm["text"]
    _assert_effort_once(gv2_nat["text"], "glaive-v2-real-native")
    _assert_completion(tokenizer, gv2_nat["text"], "glaive-v2-real-native")
    log.info("FORMAT CHECK PASSED (%s)", tokenizer_dir)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.verify_formats:
        verify_formats()
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run_tiny:
        model, tok = tiny_model(args)
        parts = get_parts(model)
        hot = [3, 1, 7, 0, 12, 9]
        hot_experts = [list(hot) for _ in range(len(parts.text.layers))]
        tap_layers = [1, 3, 5]
    else:
        if not args.hot_plan:
            raise SystemExit("--hot-plan is required for a real run")
        plan = json.loads(Path(args.hot_plan).read_text())
        hot_experts = plan["hot_experts"]
        from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration
        tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = Inkling2LoopForConditionalGeneration.from_pretrained(
            args.model, revision=args.revision, dtype=torch.bfloat16,
            device_map={"": "cpu"},
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

    if args.dry_run_tiny:
        gguf_path, init_path = write_synthetic_qat(parts, hot_experts, args, out)
        args.student_from_gguf = str(gguf_path)
        args.init_from = str(init_path)
        args.gguf_experts_reordered = True

    if args.student_from_gguf:
        load_student_qat_from_gguf(parts, hot_experts, args.student_from_gguf, args)

    trainable = freeze_and_select(parts)
    real_gguf = Path(os.environ.get("INKLING_V1_GGUF", "models/inkling2-26b-flashpin-1p-hot40-q2k-i8mm.gguf"))
    if args.student_from_gguf and Path(args.student_from_gguf).resolve() == real_gguf.resolve():
        raise SystemExit("refusing to use the reap40 i8mm file as a student base from this trainer entry")

    if args.init_from:
        blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if gguf_includes_run3_dense(args, blob):
            recon = apply_run3_init(parts, blob, args)
            if args.dry_run_tiny:
                n_attn_modules = sum(
                    1 for m in parts.full.modules() if isinstance(m, LoRALinear) and m is not parts.head_owner.lm_head
                )
                log.info(
                    "CHECK step0 student_base + init LoRA == W_gguf max|diff| = %.3g (%d attn LoRAs, %d router LoRAs)",
                    recon["recon"], recon["n_attn"], recon["n_rtr"],
                )
                assert recon["recon"] == 0.0, recon
                assert recon["n_attn"] == n_attn_modules, (recon["n_attn"], n_attn_modules)
                assert recon["n_rtr"] == len(routers(parts))
        elif "hot_experts" in blob and not (args.resume and (out / "ckpt.pt").exists()):
            copy_matching_trainable(parts, trainable, blob)
            log.info("initialized trainable tensors from v2 ckpt %s", args.init_from)
        elif not gguf_includes_run3_dense(args, blob):
            copy_matching_trainable(parts, trainable, blob)
            log.info("copied matching tensors from %s", args.init_from)
    elif args.student_from_gguf:
        for module in parts.full.modules():
            if isinstance(module, LoRALinear) and hasattr(module, "_gguf_f32"):
                del module._gguf_f32

    if args.grad_ckpt:
        model.activation_checkpointing_enable({"use_reentrant": False})
    model.train()
    n_train = sum(p.numel() for p in trainable.values())
    log.info(
        "WSD-40 v2 student ready: hot=%d/layer, %d wrapped linears, %d trainable tensors (%.2fM params)",
        len(hot_experts[0]), len(wrapped), len(trainable), n_train / 1e6,
    )
    recorder = MultiLayerRecorder(parts.text.layers, tap_layers)
    opt, sched = build_optimizer(trainable, args)
    state = {"step": 0, "seqs": 0, "hours": 0.0}
    if args.resume and (out / "ckpt.pt").exists():
        state = load_ckpt(out, parts, trainable, opt, sched)
        log.info("resumed at step %d (%d seqs)", state["step"], state["seqs"])
    elif args.resume and args.student_from_gguf and not args.init_from:
        log.warning("resume with --student-from-gguf and no run3 --init-from; dense QAT base still includes the merged run3 delta")

    digest0 = expert_digest(parts)
    pad_id = tok.pad_token_id if (tok is not None and getattr(tok, "pad_token_id", None) is not None) else 0
    if args.dry_run_tiny:
        vocab = parts.text.config.unpadded_vocab_size
        tiny_tok = _TinyTokenizer()

        def synth_ro():
            while True:
                yield {
                    "text": (
                        f"<|message_system|><|content_text|>0{END_MESSAGE}"
                        f"<|message_user|><|content_text|>2{END_MESSAGE}"
                        f"{MESSAGE_MODEL}<|content_thinking|>4{END_MESSAGE}"
                        f"{MESSAGE_MODEL}<|content_text|>4{END_MESSAGE}"
                    ),
                    "is_gold": True,
                    "has_thinking": True,
                }

        ro_it = rollout_batches(tiny_tok, [synth_ro()], [1.0], args.seq_len, args.batch, args.seed, "", pad_id, args.device)
        ord_it = (
            torch.tensor([next(random_blocks(vocab, args.seq_len, args.seed + s)) for _ in range(args.batch)], device=args.device)
            for s in itertools.count()
        )
        train_it = (next(ro_it) if s % 2 == 0 else next(ord_it) for s in itertools.count())
        packed_eval = [
            torch.tensor([next(random_blocks(vocab, args.seq_len, args.seed + 100 + s)) for _ in range(args.batch)], device=args.device)
            for s in range(2)
        ]
        ro_eval = [next(ro_it)]
        tool_if_eval = [next(ro_it)]
        if state["step"] > 0:
            stats = train_step(parts, train_it, opt, sched, trainable, state["step"], args, recorder)
            log.info("RESUME CHECK PASSED step=%d loss=%.4f grad_norm=%.4f", state["step"], stats["loss"], stats["grad_norm"])
            assert math.isfinite(stats["loss"]) and stats["grad_norm"] > 0
            return
    else:
        train_it, packed_eval, ro_eval, tool_if_eval = build_real_data_iters(tok, args, pad_id)

    t0, hours0, seqs0 = time.time(), state["hours"], state["seqs"]
    if state["step"] == 0:
        ev0 = evaluate(parts, packed_eval, ro_eval, args, recorder, tool_if_eval)
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
        log.info(
            "step %d [%s] loss %.4f %s lr %.2e tok/s %.0f gpu_peak_gb %.1f",
            state["step"], stats["batch_kind"], stats["loss"],
            {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items() if k not in ("loss", "batch_kind")},
            sched.get_last_lr()[0], tok_s, peak_gb,
        )
        if state["step"] % args.eval_every == 0:
            ev = evaluate(parts, packed_eval, ro_eval, args, recorder, tool_if_eval)
            gens = greedy_samples(parts, tok, EVAL_PROMPTS if tok else ["1 2 3 4"], args, args.device)
            log.info("eval step %d %s", state["step"], ev)
            with open(out / "eval.jsonl", "a") as f:
                f.write(json.dumps({"step": state["step"], **ev, "samples": gens}) + "\n")
        hit_cap = (time.time() - t0) / 3600.0 >= args.max_hours
        if state["step"] % args.save_every == 0 or hit_cap:
            save_ckpt(out, parts, trainable, hot_experts, opt, sched, state, args)
        if hit_cap:
            log.info("wall-clock cap %.2fh reached; saved and stopping", args.max_hours)
            break

    save_ckpt(out, parts, trainable, hot_experts, opt, sched, state, args)
    assert expert_digest(parts) == digest0, "frozen base experts changed"
    log.info("base expert digest unchanged: %s", digest0)
    if args.dry_run_tiny:
        dry_run_checks(parts, hot_experts, args, recorder, packed_eval, ro_eval, tool_if_eval, trainable, opt, sched, out)


if __name__ == "__main__":
    main()
