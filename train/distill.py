#!/usr/bin/env python3
"""Self-distillation of Inkling-2-Mini into its phone shape (1 loop pass, top-4 experts).

Teacher: the frozen model, 3 loop passes, top-8 experts, LoRA off.
Student: the same module and the same frozen tensors (one copy in memory), 1 pass, top-4, LoRA on.
Loss:    KL(teacher || student) on next-token distributions (+ optional relative MSE of the hidden
         state at the loop exit). Experts never train. Only LoRA on matmuls that llama.cpp routes
         through build_lora_mm (see FACTS.md) and, optionally, small dense tensors.

Run `python distill.py --dry-run-tiny` for a 20-step CPU smoke test with a random tiny model.
Needs transformers >= 5.24.0 (first release with inkling2_loop). `datasets` only for real runs.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

log = logging.getLogger("distill")

MODEL_ID = "thinkingmachines/Inkling-2-Mini"
MODEL_REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"

# transformers module suffix -> llama.cpp GGUF tensor stem (blk.{i}.<stem>.weight). All of these are
# applied with build_lora_mm in src/models/inkling2.cpp, so a GGUF LoRA adapter can change them.
LORA_TARGETS = {
    "self_attn.q_proj.linear": "attn_q",
    "self_attn.k_proj.linear": "attn_k",
    "self_attn.v_proj.linear": "attn_v",
    "self_attn.o_proj.linear": "attn_output",
    "self_attn.o_proj.gate_proj": "attn_gate",
    "self_attn.relative_position_bias.linear": "attn_rel",
    "self_attn.sink_proj": "attn_sink_proj",
    "self_attn.k_proj.sconv.weight_proj": "attn_k_conv_proj",  # full-attention layers only
    "self_attn.v_proj.sconv.weight_proj": "attn_v_conv_proj",  # full-attention layers only
}
ROUTER_GGUF_STEM = "ffn_gate_inp"  # mlp.gate.weight, also through build_lora_mm (llama-graph.cpp build_moe_ffn)

# Small dense tensors that llama.cpp does NOT route through build_lora_mm: they can only ship by
# rewriting their (F32) values in the GGUF. Trained only with --train-small-dense.
SMALL_DENSE_PATTERNS = (
    r"layernorm\.weight$", r"\.norm\.weight$", r"attention_input\.query$", r"mlp_input\.query$",
    r"final_input\.query$", r"q_proj\.norm\.scale$", r"relative_position_bias\.norm\.scale$",
)

END_MESSAGE = "<|end_message|>"
MESSAGE_MODEL = "<|message_model|>"
STREAM_RETRIES = 5
STREAM_BACKOFF_S = 30

IDENTITY_RE = re.compile(
    r"\b(as an ai\b|an ai (language )?model|language model developed|i am an ai|i'm an ai|my name is|"
    r"i am (chatgpt|claude|gpt|llama|tulu|olmo|gemini)|i'm (chatgpt|claude|tulu|olmo))", re.IGNORECASE)


# --------------------------------------------------------------------------------------------- LoRA

class LoRAState:
    """Global switch: LoRA deltas apply only while `enabled` (student); off for the teacher."""
    enabled = True


class LoRALinear(nn.Module):
    """y = base(x) + scale * B(A(x)); base frozen and shared with the teacher."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self) -> torch.Tensor:  # some callers read `.weight` (e.g. dtype checks)
        return self.base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not LoRAState.enabled:
            return out
        delta = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B) * self.scale
        return out + delta.to(out.dtype)


def _router_forward_with_lora(self, hidden_states: torch.Tensor):
    """Copy of Inkling2LoopTopKRouter.forward (modeling_inkling2_loop.py:1163-1172) plus a LoRA delta on the logits."""
    flat = hidden_states.float().reshape(-1, self.hidden_dim)
    logits = F.linear(flat, self.weight)
    if LoRAState.enabled:
        logits = logits + F.linear(F.linear(flat, self.lora_A), self.lora_B) * self.lora_scale
    scores = self.score_fn(logits)
    indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * self.routed_scaling_factor, indices


def add_router_lora(router: nn.Module, rank: int, alpha: float) -> None:
    router.lora_A = nn.Parameter(torch.empty(rank, router.hidden_dim, dtype=torch.float32))
    router.lora_B = nn.Parameter(torch.zeros(router.num_experts, rank, dtype=torch.float32))
    nn.init.kaiming_uniform_(router.lora_A, a=math.sqrt(5))
    router.lora_scale = alpha / rank
    router.forward = MethodType(_router_forward_with_lora, router)


def inject_lora(text_model: nn.Module, lm_head_owner: nn.Module, args: argparse.Namespace) -> list[str]:
    """Wrap target linears in LoRALinear; returns the wrapped module names (relative to text_model)."""
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
    if args.lora_router:
        for layer in text_model.layers:
            add_router_lora(layer.mlp.gate, args.router_rank, args.router_alpha)
            layer.mlp.gate.to(layer.mlp.gate.weight.device)
    if args.lora_lm_head:
        lm_head_owner.lm_head = LoRALinear(lm_head_owner.lm_head, args.lora_rank, args.lora_alpha).to(
            lm_head_owner.lm_head.weight.device)
    return wrapped


# ------------------------------------------------------------------------------------ model access

@dataclass(frozen=True)
class Parts:
    full: nn.Module          # the HF model (ForConditionalGeneration or ForCausalLM)
    text: nn.Module          # Inkling2LoopTextModel
    head_owner: nn.Module    # module holding .lm_head and .logit_softcap
    loop_start: int
    loop_end: int


def get_parts(model: nn.Module) -> Parts:
    text = model.model.language_model if hasattr(model.model, "language_model") else model.model
    cfg = text.config
    return Parts(model, text, model, cfg.loop_layer_index_start, cfg.loop_layer_index_end)


def routers(parts: Parts) -> list[nn.Module]:
    return [layer.mlp.gate for layer in parts.text.layers]


def set_resting_shape(parts: Parts, passes: int, top_k: int) -> None:
    """The student shape is the module's resting state. Activation checkpointing re-runs the student
    layers during backward, outside any context manager, so they must see the student's top-k then."""
    parts.text.config.loop_num_passes = passes
    for r in routers(parts):
        r.top_k = top_k
    LoRAState.enabled = True


@contextlib.contextmanager
def shape(parts: Parts, passes: int, top_k: int, lora: bool):
    """Temporarily run the shared module as teacher (or student without LoRA, for baselines)."""
    cfg = parts.text.config
    old = (cfg.loop_num_passes, [r.top_k for r in routers(parts)], LoRAState.enabled)
    cfg.loop_num_passes = passes  # read at forward time: modeling_inkling2_loop.py:1398
    for r in routers(parts):
        r.top_k = top_k              # read at forward time: modeling_inkling2_loop.py:1169
    LoRAState.enabled = lora
    try:
        yield
    finally:
        cfg.loop_num_passes = old[0]
        for r, k in zip(routers(parts), old[1]):
            r.top_k = k
        LoRAState.enabled = old[2]


def head_logits(parts: Parts, hidden: torch.Tensor, use_lora: bool) -> torch.Tensor:
    head = parts.head_owner.lm_head
    if isinstance(head, LoRALinear) and not use_lora:
        logits = head.base(hidden).float()
    else:
        logits = head(hidden).float()
    return parts.head_owner.logit_softcap(logits)


def is_small_dense(name: str) -> bool:
    return any(re.search(p, name) for p in SMALL_DENSE_PATTERNS)


def freeze_and_select(parts: Parts, train_small_dense: bool) -> dict[str, nn.Parameter]:
    """Freeze everything, then return the trainable parameters by name (relative to the full model)."""
    for p in parts.full.parameters():
        p.requires_grad_(False)
    trainable = {}
    for name, p in parts.full.named_parameters():
        if "lora_" in name or (train_small_dense and is_small_dense(name) and "vision" not in name):
            if train_small_dense and is_small_dense(name):
                p.data = p.data.float()  # train small dense tensors in fp32
            p.requires_grad_(True)
            trainable[name] = p
    return trainable


class QuantExperts:
    """Frozen expert stack held as int8 with one scale per output row (about half the memory of bf16, so the
    26B model fits a 48 GB GPU). Indexing returns that expert's weight in the compute dtype. Not a Parameter:
    it can never receive a gradient or an optimizer update."""

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
        self.q, self.scale, self.dtype = q, scale, dtype
        self.shape = q.shape

    def __getitem__(self, idx) -> torch.Tensor:
        return self.q[idx].to(self.dtype) * self.scale[idx]


def expert_digest(parts: Parts) -> str:
    """Hash of every expert tensor; must not change during training."""
    h = hashlib.sha256()
    for layer in parts.text.layers:
        for t in (layer.mlp.experts.gate_up_proj, layer.mlp.experts.down_proj):
            if isinstance(t, QuantExperts):
                h.update(t.q[0].cpu().numpy().tobytes()[:1 << 16])
                h.update(str(int(t.q.sum(dtype=torch.int64))).encode())
                h.update(str(float(t.scale.float().sum())).encode())
                continue
            h.update(t.detach().float().cpu().numpy().tobytes()[:1 << 16])
            h.update(str(float(t.detach().float().sum())).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------------- forward + loss

class LoopExitRecorder:
    """Forward hook on the last loop layer; keeps the output of its last call (= the final pass)."""

    def __init__(self, layer: nn.Module):
        self.value = None
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.value = output[0] if isinstance(output, tuple) else output


def chunked_kl(parts: Parts, s_h: torch.Tensor, t_h: torch.Tensor, chunk: int, temp: float,
               with_reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean per-token (KL(teacher || student), KL(student || teacher)) over a [N, H] batch; the vocab-sized logits
    live one chunk at a time. The reverse term is the on-policy distillation loss: on text the student wrote itself
    it penalises probability the student puts where the teacher puts little (loops). It is zero unless asked for."""

    def one(s_part: torch.Tensor, t_part: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t_logp = F.log_softmax(head_logits(parts, t_part, use_lora=False) / temp, dim=-1)
        s_logp = F.log_softmax(head_logits(parts, s_part, use_lora=True) / temp, dim=-1)
        forward = (t_logp.exp() * (t_logp - s_logp)).sum()
        reverse = (s_logp.exp() * (s_logp - t_logp)).sum() if with_reverse else forward.new_zeros(())
        return torch.stack([forward, reverse]) * (temp * temp)

    total = s_h.new_zeros((2,), dtype=torch.float32)
    for start in range(0, s_h.shape[0], chunk):
        s_part, t_part = s_h[start:start + chunk], t_h[start:start + chunk]
        total = total + checkpoint(one, s_part, t_part, use_reentrant=False)
    total = total / s_h.shape[0]
    return total[0], total[1]


@dataclass(frozen=True)
class RolloutBatch:
    s_ids: torch.Tensor
    s_attn: torch.Tensor
    s_mask: torch.Tensor
    t_ids: torch.Tensor
    t_attn: torch.Tensor
    t_mask: torch.Tensor


def teacher_forward(parts: Parts, ids: torch.Tensor, args, recorder: LoopExitRecorder,
                    attention_mask: torch.Tensor | None = None):
    with torch.no_grad(), shape(parts, args.teacher_passes, args.teacher_topk, lora=False):
        out = parts.text(input_ids=ids, attention_mask=attention_mask, use_cache=False)
        exit_h = recorder.value.detach() if recorder.value is not None else None
    return out.last_hidden_state.detach(), exit_h


def distill_loss(parts: Parts, batch: torch.Tensor | RolloutBatch, args,
                 recorder: LoopExitRecorder) -> tuple[torch.Tensor, dict]:
    if isinstance(batch, RolloutBatch):
        t_last, t_exit = teacher_forward(parts, batch.t_ids, args, recorder, attention_mask=batch.t_attn)
        s_last = parts.text(input_ids=batch.s_ids, attention_mask=batch.s_attn, use_cache=False).last_hidden_state
        s_exit = recorder.value
        s_h, t_h = s_last[batch.s_mask], t_last[batch.t_mask]
        kl, rev = chunked_kl(parts, s_h, t_h, args.logit_chunk, args.kl_temp, with_reverse=True)
        loss = kl.new_zeros(())
        if args.rollout_forward_kl > 0:
            loss = loss + args.rollout_forward_kl * kl
        if args.rollout_reverse_kl > 0:
            loss = loss + args.rollout_reverse_kl * rev
        if args.rollout_forward_kl == 0 and args.rollout_reverse_kl == 0:
            loss = loss + 0.0 * kl
        stats = {"ro_kl": float(kl.detach()), "ro_rev_kl": float(rev.detach())}
        if args.hidden_mse > 0 and s_exit is not None and t_exit is not None:
            s_ex, t_ex = s_exit[batch.s_mask].float(), t_exit[batch.t_mask].float()
            diff = (s_ex - t_ex).pow(2).sum(-1)
            rel = (diff / t_ex.pow(2).sum(-1).clamp_min(1e-6)).mean()
            loss = loss + args.hidden_mse * rel
            stats["exit_rel_mse"] = float(rel.detach())
        return loss, stats
    t_last, t_exit = teacher_forward(parts, batch, args, recorder)
    s_last = parts.text(input_ids=batch, use_cache=False).last_hidden_state  # resting shape = student
    s_exit = recorder.value
    hdim = s_last.shape[-1]
    kl, rev = chunked_kl(parts, s_last.reshape(-1, hdim), t_last.reshape(-1, hdim), args.logit_chunk, args.kl_temp,
                         with_reverse=args.reverse_kl > 0)
    loss, stats = kl, {"kl": float(kl.detach())}
    if args.reverse_kl > 0:
        loss = loss + args.reverse_kl * rev
        stats["rev_kl"] = float(rev.detach())
    if args.hidden_mse > 0 and s_exit is not None and t_exit is not None:
        diff = (s_exit.float() - t_exit.float()).pow(2).sum(-1)
        rel = (diff / t_exit.float().pow(2).sum(-1).clamp_min(1e-6)).mean()
        loss = loss + args.hidden_mse * rel
        stats["exit_rel_mse"] = float(rel.detach())
    return loss, stats


@torch.no_grad()
def evaluate(parts: Parts, batches: list[torch.Tensor], args, recorder: LoopExitRecorder) -> dict:
    was_training = parts.full.training
    parts.full.eval()
    kls, base_kls = [], []
    for ids in batches:
        t_last, _ = teacher_forward(parts, ids, args, recorder)
        for use_lora, sink in ((True, kls), (False, base_kls)):
            with shape(parts, args.student_passes, args.student_topk, lora=use_lora):
                s_last = parts.text(input_ids=ids, use_cache=False).last_hidden_state
                hdim = s_last.shape[-1]
                sink.append(float(chunked_kl(parts, s_last.reshape(-1, hdim), t_last.reshape(-1, hdim),
                                             args.logit_chunk, 1.0)[0]))
    parts.full.train(was_training)
    return {"eval_kl": sum(kls) / len(kls), "eval_kl_untrained_student": sum(base_kls) / len(base_kls)}


@torch.no_grad()
def greedy_samples(parts: Parts, tokenizer, prompts: list[str], args, device) -> list[dict]:
    """A few greedy generations from teacher and student (HF generate, with cache)."""
    out = []
    was_training = parts.full.training
    parts.full.eval()
    for prompt in prompts:
        if tokenizer is None:
            ids = torch.tensor([[int(t) for t in prompt.split()]], device=device)
        else:
            text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                 add_generation_prompt=True, reasoning_effort=args.reasoning_effort)
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        row = {"prompt": prompt}
        for tag, (passes, k, lora) in {"teacher": (args.teacher_passes, args.teacher_topk, False),
                                       "student": (args.student_passes, args.student_topk, True)}.items():
            with shape(parts, passes, k, lora):
                gen = parts.full.generate(input_ids=ids, max_new_tokens=args.gen_tokens, do_sample=False)
            new = gen[0, ids.shape[1]:].tolist()
            row[tag] = tokenizer.decode(new, skip_special_tokens=False) if tokenizer is not None else new
        out.append(row)
    parts.full.train(was_training)
    return out


# --------------------------------------------------------------------------------------------- data

def _messages_ok(row: dict) -> list[dict] | None:
    if "hardcoded" in str(row.get("source", "")).lower():  # tulu-3 identity subset
        return None
    msgs = [{"role": m["role"], "content": m["content"]} for m in row.get("messages", [])
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
    if len(msgs) < 2 or not any(m["role"] == "assistant" for m in msgs):
        return None
    if any(m["role"] == "assistant" and IDENTITY_RE.search(m["content"]) for m in msgs):
        return None
    return msgs


def _source_rows(name: str, split: str, seed: int, order: int) -> Iterator[dict]:
    """One source as an endless row iterator (re-shuffled on each pass); plain dicts, so sources with
    different columns can be mixed (interleave_datasets refuses that)."""
    from datasets import load_dataset  # real runs only

    def open_stream():
        if name.endswith(".jsonl"):  # local rows; the split field is ignored
            return load_dataset("json", data_files=name, split="train", streaming=True)
        return load_dataset(name, split=split, streaming=True)

    epoch, failures = 0, 0
    while True:
        n = 0
        try:
            for row in open_stream().shuffle(seed=seed + order + epoch, buffer_size=10_000):
                n += 1
                yield row
        except (OSError, RuntimeError) as err:  # a Hub stream can drop mid-run: reopen, do not lose the job
            failures += 1
            if failures > STREAM_RETRIES or name.endswith(".jsonl"):  # a local file error is a real bug
                raise
            log.warning("data source %s failed (%s: %s); reopening in %d s (attempt %d of %d)", name,
                        type(err).__name__, str(err)[:200], STREAM_BACKOFF_S * failures, failures, STREAM_RETRIES)
            time.sleep(STREAM_BACKOFF_S * failures)
            epoch += 1
            continue
        if n == 0:
            raise ValueError(f"data source {name} has no rows")
        epoch, failures = epoch + 1, 0  # the retry count resets only after a full clean pass


def _pack_blocks(tokenizer, streams: list[Iterator[dict]], weights: list[float], seq_len: int, seed: int,
                 effort: str | float) -> Iterator[list[int]]:
    picker = random.Random(seed)
    mixed = (next(picker.choices(streams, weights=weights)[0]) for _ in iter(int, 1))
    buf: list[int] = []
    for row in mixed:
        if isinstance(row.get("text"), str) and row["text"]:
            text = row["text"]
            if not text.endswith(END_MESSAGE):
                text += END_MESSAGE
        else:
            msgs = _messages_ok(row)
            if msgs is None:
                continue
            text = tokenizer.apply_chat_template(msgs, tokenize=False, reasoning_effort=effort)
        buf.extend(tokenizer(text, add_special_tokens=False).input_ids)
        while len(buf) >= seq_len:
            yield buf[:seq_len]
            buf = buf[seq_len:]


def token_blocks(tokenizer, sources: list[tuple[str, str, float]], seq_len: int, seed: int,
                 effort: str | float) -> Iterator[list[int]]:
    """Stream chat rows, render with the model's chat template, pack into seq_len blocks."""
    streams = [_source_rows(name, split, seed, i) for i, (name, split, _) in enumerate(sources)]
    weights = [w for _, _, w in sources]
    return _pack_blocks(tokenizer, streams, weights, seq_len, seed, effort)


def prepare_rollout(text: str, tokenizer, seq_len: int,
                    teacher_note: str) -> tuple[list[int], list[int], int, int, int] | None:
    """Split a pre-rendered rollout at the first <|message_model|>, optionally append teacher_note to the
    teacher input as its own leading system message, truncate the student side to seq_len, and
    return (s_ids, t_ids, s_start, t_start, n_comp) where [start : start + n_comp] predicts the completion."""
    split_at = text.find(MESSAGE_MODEL)
    if split_at < 0:
        return None
    prefix_end = split_at + len(MESSAGE_MODEL)
    s_prefix, completion = text[:prefix_end], text[prefix_end:]
    if not completion:
        return None
    if teacher_note:
        # its own leading system message, the same layout the bench renders for an arm's system_note
        t_prefix = f"<|message_system|><|content_text|>{teacher_note}{END_MESSAGE}{s_prefix}"
    else:
        t_prefix = s_prefix
    s_pre = tokenizer(s_prefix, add_special_tokens=False).input_ids
    t_pre = s_pre if not teacher_note else tokenizer(t_prefix, add_special_tokens=False).input_ids
    comp = tokenizer(completion, add_special_tokens=False).input_ids
    avail = seq_len - len(s_pre)
    if not s_pre or not t_pre or avail <= 0:
        return None
    comp = comp[:avail]
    if not comp:
        return None
    return s_pre + comp, t_pre + comp, len(s_pre) - 1, len(t_pre) - 1, len(comp)


def collate_rollouts(items: list[tuple[list[int], list[int], int, int, int]], pad_id: int, device) -> RolloutBatch:
    bsz = len(items)
    max_s = max(len(x[0]) for x in items)
    max_t = max(len(x[1]) for x in items)
    s_ids = torch.full((bsz, max_s), pad_id, dtype=torch.long)
    t_ids = torch.full((bsz, max_t), pad_id, dtype=torch.long)
    s_attn = torch.zeros((bsz, max_s), dtype=torch.long)
    t_attn = torch.zeros((bsz, max_t), dtype=torch.long)
    s_mask = torch.zeros((bsz, max_s), dtype=torch.bool)
    t_mask = torch.zeros((bsz, max_t), dtype=torch.bool)
    for b, (si, ti, ss, ts, nc) in enumerate(items):
        s_ids[b, :len(si)] = torch.tensor(si, dtype=torch.long)
        t_ids[b, :len(ti)] = torch.tensor(ti, dtype=torch.long)
        s_attn[b, :len(si)] = 1
        t_attn[b, :len(ti)] = 1
        s_mask[b, ss:ss + nc] = True
        t_mask[b, ts:ts + nc] = True
    return RolloutBatch(s_ids.to(device), s_attn.to(device), s_mask.to(device),
                        t_ids.to(device), t_attn.to(device), t_mask.to(device))


def rollout_batches(tokenizer, streams: list[Iterator[dict]], weights: list[float], seq_len: int, batch: int,
                    seed: int, teacher_note: str, pad_id: int, device) -> Iterator[RolloutBatch]:
    picker = random.Random(seed + 1)
    mixed = (next(picker.choices(streams, weights=weights)[0]) for _ in iter(int, 1))
    cur, skipped = [], 0
    for row in mixed:
        text = row.get("text")
        item = prepare_rollout(text, tokenizer, seq_len, teacher_note) if isinstance(text, str) and text else None
        if item is None:
            skipped += 1
            if skipped >= 1000:
                raise ValueError(f"rollout source yielded no valid rows with {MESSAGE_MODEL!r} and completion")
            continue
        skipped = 0
        cur.append(item)
        if len(cur) == batch:
            yield collate_rollouts(cur, pad_id, device)
            cur = []


def mix_batch_streams(ord_it, ro_it, ord_weight: float, ro_weight: float, grad_accum: int, seed: int):
    """Choose rollout vs ordinary batches per optimizer step at the mix frequency given in --data."""
    if ord_weight + ro_weight <= 0:
        raise ValueError("--data weights must sum to > 0")
    picker = random.Random(seed + 2)
    p_ro = ro_weight / (ord_weight + ro_weight)
    while True:
        src = ro_it if picker.random() < p_ro else ord_it
        for _ in range(grad_accum):
            yield next(src)


def train_batches(tokenizer, sources: list[tuple[str, str, float]], seq_len: int, batch: int, grad_accum: int,
                  seed: int, effort: str | float, teacher_note: str, pad_id: int, device):
    ord_streams, ord_weights, ro_streams, ro_weights = [], [], [], []
    for i, (name, split, weight) in enumerate(sources):
        it = _source_rows(name, split, seed, i)
        first = next(it)
        stream = itertools.chain([first], it)
        if isinstance(first.get("text"), str) and first["text"]:
            ro_streams.append(stream)
            ro_weights.append(weight)
        else:
            ord_streams.append(stream)
            ord_weights.append(weight)
    ord_it = (batches(_pack_blocks(tokenizer, ord_streams, ord_weights, seq_len, seed, effort), batch, device)
              if ord_streams else None)
    ro_it = (rollout_batches(tokenizer, ro_streams, ro_weights, seq_len, batch, seed, teacher_note, pad_id, device)
             if ro_streams else None)
    if ord_it is not None and ro_it is not None:
        return mix_batch_streams(ord_it, ro_it, sum(ord_weights), sum(ro_weights), grad_accum, seed)
    return ro_it if ro_it is not None else ord_it


class _TinyTokenizer:
    """Deterministic character/special-token tokenizer for --dry-run-tiny rollout tests."""
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


def _synthetic_rollout_rows(seed: int) -> Iterator[dict]:
    rng = random.Random(seed)
    alphabet = "abcdefghij0123456789 "
    while True:
        u = "".join(rng.choice(alphabet) for _ in range(rng.randint(2, 5)))
        th = "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 18)))
        ans = "".join(rng.choice(alphabet) for _ in range(rng.randint(2, 10)))
        yield {"text": (f"<|message_system|><|content_text|>E:0{END_MESSAGE}"
                        f"<|message_user|><|content_text|>{u}{END_MESSAGE}"
                        f"{MESSAGE_MODEL}<|content_thinking|>{th}{END_MESSAGE}"
                        f"{MESSAGE_MODEL}<|content_text|>{ans}{END_MESSAGE}")}


def batches(blocks: Iterator[list[int]], batch: int, device) -> Iterator[torch.Tensor]:
    cur = []
    for b in blocks:
        cur.append(b)
        if len(cur) == batch:
            yield torch.tensor(cur, device=device)
            cur = []


def random_blocks(vocab: int, seq_len: int, seed: int) -> Iterator[list[int]]:
    """Dry run data: walks on one fixed sparse bigram chain (same chain for train and eval, different
    walks), so there is a shared distribution on which held-out KL can actually improve."""
    chain = torch.Generator().manual_seed(12345)
    successors = torch.randint(0, vocab, (vocab, 4), generator=chain)
    g = torch.Generator().manual_seed(seed)
    while True:
        tok = int(torch.randint(0, vocab, (1,), generator=g))
        seq = [tok]
        for _ in range(seq_len - 1):
            tok = int(successors[tok, torch.randint(0, 4, (1,), generator=g)])
            seq.append(tok)
        yield seq


# ------------------------------------------------------------------------------------- GGUF helpers

def import_gguf(llama_cpp_dir: str):
    sys.path.insert(0, str(Path(llama_cpp_dir) / "gguf-py"))
    import gguf  # noqa: E402
    return gguf


def lora_gguf_tensors(parts: Parts, n_vocab_padded: int | None) -> dict[str, tuple]:
    """GGUF adapter tensors: name -> (lora_a [r, in], lora_b [out, r] pre-multiplied by its scale), numpy order.

    llama.cpp checks ne0(base) == ne0(a) and ne1(base) == ne1(b) (llama-adapter.cpp:370); in numpy order
    a is [r, n_in] and b is [n_out, r], which is exactly the torch layout used here.
    """
    import numpy as np

    out = {}
    for name, module in parts.text.named_modules():
        if isinstance(module, LoRALinear):
            layer, suffix = int(name.split(".")[1]), name.split(".", 2)[2]
            out[f"blk.{layer}.{LORA_TARGETS[suffix]}.weight"] = (
                module.lora_A.detach().float().cpu().numpy(),
                module.lora_B.detach().float().cpu().numpy() * module.scale)
    for i, router in enumerate(routers(parts)):
        if hasattr(router, "lora_A"):
            out[f"blk.{i}.{ROUTER_GGUF_STEM}.weight"] = (
                router.lora_A.detach().float().cpu().numpy(),
                router.lora_B.detach().float().cpu().numpy() * router.lora_scale)
    head = parts.head_owner.lm_head
    if isinstance(head, LoRALinear):
        b = head.lora_B.detach().float().cpu().numpy() * head.scale
        if n_vocab_padded and b.shape[0] < n_vocab_padded:  # GGUF output.weight is padded to vocab_size
            b = np.concatenate([b, np.zeros((n_vocab_padded - b.shape[0], b.shape[1]), b.dtype)])
        out["output.weight"] = (head.lora_A.detach().float().cpu().numpy(), b)
    return out


def export_gguf_lora(parts: Parts, path: str, args) -> None:
    """Write a llama.cpp LoRA adapter (general.type=adapter, adapter.type=lora).

    Every lora_b is stored pre-scaled and adapter.lora.alpha = 0, so llama.cpp uses scale = the user's
    adapter scale (1.0 by default) for every tensor (llama-adapter.h:53-56, llama-graph.cpp build_lora_mm).
    """
    gguf = import_gguf(args.llama_cpp_dir)
    tensors = lora_gguf_tensors(parts, parts.text.config.vocab_size)
    w = gguf.GGUFWriter(path, "inkling2_loop")
    w.add_type("adapter")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", 0.0)
    for name, (a, b) in tensors.items():
        w.add_tensor(name + ".lora_a", a.astype("float32"))
        w.add_tensor(name + ".lora_b", b.astype("float32"))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    log.info("wrote GGUF LoRA adapter %s (%d target tensors)", path, len(tensors))


def check_adapter_against_gguf(adapter_path: str, base_path: str, args) -> list[str]:
    """Apply llama.cpp's adapter checks (llama-adapter.cpp:337-375) using only the two GGUF headers."""
    gguf = import_gguf(args.llama_cpp_dir)
    base = {t.name: [int(x) for x in t.shape] for t in gguf.GGUFReader(base_path).tensors}
    errors, pairs = [], {}
    for t in gguf.GGUFReader(adapter_path).tensors:
        stem, kind = t.name.rsplit(".", 1)
        pairs.setdefault(stem, {})[kind] = [int(x) for x in t.shape]
    for stem, ab in pairs.items():
        if stem not in base:
            errors.append(f"{stem}: not in base model")
            continue
        a, b, m = ab.get("lora_a"), ab.get("lora_b"), base[stem]
        if a is None or b is None or m[0] != a[0] or m[1] != b[1] or a[1] != b[0]:
            errors.append(f"{stem}: shape mismatch base={m} a={a} b={b}")
    return errors


def load_experts_from_gguf(parts: Parts, gguf_path: str, args) -> None:
    """Replace every expert tensor with the phone's dequantized expert (q * per-expert .scale)."""
    gguf = import_gguf(args.llama_cpp_dir)
    from gguf.quants import dequantize
    import numpy as np

    reader = gguf.GGUFReader(gguf_path)
    by_name = {t.name: t for t in reader.tensors}
    for i, layer in enumerate(parts.text.layers):
        for stem, param in (("ffn_gate_up_exps", layer.mlp.experts.gate_up_proj),
                            ("ffn_down_exps", layer.mlp.experts.down_proj)):
            t = by_name[f"blk.{i}.{stem}.weight"]
            scale_t = by_name.get(f"blk.{i}.{stem}.scale")
            scale = np.asarray(scale_t.data, dtype=np.float32) if scale_t is not None else None
            for e in range(param.shape[0]):  # one expert at a time keeps host RAM small
                w = dequantize(np.asarray(t.data[e]), t.tensor_type).astype(np.float32)
                if scale is not None:
                    w = w * scale[e]
                if tuple(w.shape) != tuple(param.shape[1:]):
                    raise ValueError(f"blk.{i}.{stem}[{e}]: GGUF {w.shape} vs model {tuple(param.shape[1:])}")
                param.data[e].copy_(torch.from_numpy(w).to(param.dtype))
        log.info("experts of layer %d loaded from GGUF (%s / %s)", i,
                 by_name[f"blk.{i}.ffn_gate_up_exps.weight"].tensor_type.name,
                 by_name[f"blk.{i}.ffn_down_exps.weight"].tensor_type.name)


INT8_MAX = 127


def load_experts_int8_from_gguf(parts: Parts, gguf_path: str, args) -> None:
    """Like load_experts_from_gguf, but stores each expert stack as QuantExperts directly on args.device and
    drops the bf16 parameter, so the full-precision experts never reach the GPU."""
    gguf = import_gguf(args.llama_cpp_dir)
    from gguf.quants import dequantize
    import numpy as np

    by_name = {t.name: t for t in gguf.GGUFReader(gguf_path).tensors}
    worst = 0.0
    for i, layer in enumerate(parts.text.layers):
        experts = layer.mlp.experts
        experts.config._experts_implementation = "eager"  # the only implementation that indexes one expert at a time
        for stem, attr in (("ffn_gate_up_exps", "gate_up_proj"), ("ffn_down_exps", "down_proj")):
            param = getattr(experts, attr)
            t = by_name[f"blk.{i}.{stem}.weight"]
            scale_t = by_name.get(f"blk.{i}.{stem}.scale")
            escale = np.asarray(scale_t.data, dtype=np.float32) if scale_t is not None else None
            n_exp, rows, cols = param.shape
            q = torch.empty((n_exp, rows, cols), dtype=torch.int8, device=args.device)
            scale = torch.empty((n_exp, rows, 1), dtype=param.dtype, device=args.device)
            for e in range(n_exp):
                w = dequantize(np.asarray(t.data[e]), t.tensor_type).astype(np.float32)
                if escale is not None:
                    w = w * escale[e]
                if w.shape != (rows, cols):
                    raise ValueError(f"blk.{i}.{stem}[{e}]: GGUF {w.shape} vs model {(rows, cols)}")
                if not np.isfinite(w).all():
                    raise ValueError(f"blk.{i}.{stem}[{e}]: non-finite values after dequantization")
                row_scale = np.abs(w).max(axis=1, keepdims=True) / INT8_MAX
                row_scale[row_scale == 0] = 1.0
                wq = np.round(w / row_scale).astype(np.int8)
                if e == 0:
                    err = wq.astype(np.float32) * row_scale - w
                    worst = max(worst, float(np.sqrt((err ** 2).mean()) / max(1e-12, np.sqrt((w ** 2).mean()))))
                q[e].copy_(torch.from_numpy(wq))
                scale[e].copy_(torch.from_numpy(row_scale).to(param.dtype))
            dtype = param.dtype
            del experts._parameters[attr]
            setattr(experts, attr, QuantExperts(q, scale, dtype))
        log.info("experts of layer %d loaded from GGUF as int8 (%s / %s), worst rel rms err so far %.4f", i,
                 by_name[f"blk.{i}.ffn_gate_up_exps.weight"].tensor_type.name,
                 by_name[f"blk.{i}.ffn_down_exps.weight"].tensor_type.name, worst)


def _gguf_dense(by_name: dict, name: str, rows: int | None = None):
    """Dequantize one whole (dense, 2-D) GGUF tensor to float32 numpy in torch layout [out, in]."""
    from gguf.quants import dequantize
    import numpy as np

    t = by_name[name]
    w = dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)
    return w[:rows] if rows is not None else w


def load_dense_from_gguf(parts: Parts, gguf_path: str, args) -> None:
    """Replace the LoRA-target matmuls, routers and output head with the phone's (quantized) values, so the
    LoRA learns against exactly what llama.cpp multiplies. Call before inject_lora."""
    gguf = import_gguf(args.llama_cpp_dir)
    by_name = {t.name: t for t in gguf.GGUFReader(gguf_path).tensors}
    n = 0
    with torch.no_grad():
        for i, layer in enumerate(parts.text.layers):
            for suffix, stem in LORA_TARGETS.items():
                gname = f"blk.{i}.{stem}.weight"
                if gname not in by_name:
                    continue
                param = layer.get_submodule(suffix).weight
                param.copy_(torch.from_numpy(_gguf_dense(by_name, gname)).to(param.dtype))
                n += 1
            gate = layer.mlp.gate.weight
            gate.copy_(torch.from_numpy(_gguf_dense(by_name, f"blk.{i}.{ROUTER_GGUF_STEM}.weight")).to(gate.dtype))
            n += 1
        head = parts.head_owner.lm_head.weight
        head.copy_(torch.from_numpy(_gguf_dense(by_name, "output.weight", head.shape[0])).to(head.dtype))
    log.info("loaded %d dense tensors (+ output head) from GGUF", n)


def write_tiny_gguf(parts: Parts, path: str, args) -> None:
    """Dry-run only: a GGUF holding the tiny model's experts (Q8_0 + per-expert scale) and the
    LoRA-target matmuls, so the GGUF expert loader and the adapter checks can be tested end to end."""
    gguf = import_gguf(args.llama_cpp_dir)
    from gguf.quants import quantize
    import numpy as np

    w = gguf.GGUFWriter(path, "inkling2_loop")
    for i, layer in enumerate(parts.text.layers):
        for stem, param in (("ffn_gate_up_exps", layer.mlp.experts.gate_up_proj),
                            ("ffn_down_exps", layer.mlp.experts.down_proj)):
            x = param.detach().float().cpu().numpy()
            s = np.abs(x).reshape(x.shape[0], -1).max(1).astype(np.float32) / 100.0
            q = quantize(x / s[:, None, None], gguf.GGMLQuantizationType.Q8_0)
            w.add_tensor(f"blk.{i}.{stem}.weight", q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            w.add_tensor(f"blk.{i}.{stem}.scale", s)
        def q8(name: str, x: np.ndarray) -> None:  # dense matmuls are quantized on the phone too
            w.add_tensor(name, quantize(x, gguf.GGMLQuantizationType.Q8_0), raw_dtype=gguf.GGMLQuantizationType.Q8_0)

        for name, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                q8(f"blk.{i}.{LORA_TARGETS[name]}.weight", module.base.weight.detach().float().numpy())
        q8(f"blk.{i}.{ROUTER_GGUF_STEM}.weight", layer.mlp.gate.weight.detach().float().numpy())
    head = parts.head_owner.lm_head
    head_w = (head.base if isinstance(head, LoRALinear) else head).weight.detach().float().numpy()
    pad = np.zeros((parts.text.config.vocab_size - head_w.shape[0], head_w.shape[1]), np.float32)
    w.add_tensor("output.weight", np.concatenate([head_w, pad]))  # padded like the real GGUF
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


# ---------------------------------------------------------------------------------------- building

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
    with torch.no_grad():  # the init leaves the attention-residual and message queries at zero; perturb them
        for n, p in model.named_parameters():
            if n.endswith("query") or n.endswith("e_score_correction_bias"):
                p.normal_(0, 0.5)
        model.lm_head.weight.normal_(0, 0.5)  # peaky next-token distributions, so the KL is not ~0
    return model.float(), None


def real_model(args):
    from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = Inkling2LoopForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        device_map={"": "cpu" if args.experts_int8 else args.device},  # int8 mode: experts never go to the GPU as bf16
        experts_implementation=args.experts_impl,
    )
    model.model.vision_tower = None  # text-only distillation; frees ~1 GB
    return model, tok


# ------------------------------------------------------------------------------------- checkpoints

def save_ckpt(path: Path, trainable: dict, opt, sched, state: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / "ckpt.pt.tmp"
    torch.save({"trainable": {k: v.detach().cpu() for k, v in trainable.items()},
                "opt": opt.state_dict(), "sched": sched.state_dict(), "state": state}, tmp)
    os.replace(tmp, path / "ckpt.pt")
    (path / "state.json").write_text(json.dumps(state, indent=1))


def load_ckpt(path: Path, trainable: dict, opt, sched) -> dict:
    blob = torch.load(path / "ckpt.pt", map_location="cpu", weights_only=False)
    with torch.no_grad():
        for k, v in blob["trainable"].items():
            trainable[k].copy_(v.to(trainable[k].device, trainable[k].dtype))
    opt.load_state_dict(blob["opt"])
    sched.load_state_dict(blob["sched"])
    return blob["state"]


# -------------------------------------------------------------------------------------------- main

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--dry-run-tiny", action="store_true", help="random tiny model, CPU, 20 steps, plus GGUF round trips")
    a("--model", default=MODEL_ID); a("--revision", default=MODEL_REVISION)
    a("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a("--experts-impl", default="eager", choices=["eager", "grouped_mm", "batched_mm"])
    a("--out", default="runs/distill"); a("--resume", action="store_true")
    a("--max-hours", type=float, default=12.0); a("--max-steps", type=int, default=100_000)
    a("--seq-len", type=int, default=2048); a("--batch", type=int, default=2); a("--grad-accum", type=int, default=8)
    a("--lr", type=float, default=1e-4); a("--small-dense-lr", type=float, default=1e-5)
    a("--warmup", type=int, default=50); a("--weight-decay", type=float, default=0.0); a("--clip", type=float, default=1.0)
    a("--seed", type=int, default=0)
    a("--teacher-passes", type=int, default=3); a("--teacher-topk", type=int, default=8)
    a("--student-passes", type=int, default=1); a("--student-topk", type=int, default=4)
    a("--lora-rank", type=int, default=32); a("--lora-alpha", type=float, default=32.0)
    a("--lora-router", action="store_true", help="also LoRA the expert routers (llama.cpp can apply it)")
    a("--router-rank", type=int, default=8); a("--router-alpha", type=float, default=8.0)
    a("--lora-lm-head", action="store_true", help="also LoRA the output head (GGUF output, padded vocab)")
    a("--train-small-dense", action="store_true",
      help="also train norms / residual queries / clip scales (needs a GGUF rewrite to ship, not an adapter)")
    a("--init-from", default=None,
      help="ckpt.pt of an earlier run: load its trained tensors only, with a fresh optimizer, schedule and step count")
    a("--reverse-kl", type=float, default=0.0,
      help="weight of KL(student || teacher) added to the forward KL (on-policy distillation; use with rollout rows)")
    a("--teacher-note", default="",
      help="text appended (after \\n\\n) to the system message for the teacher on rollout rows only")
    a("--rollout-forward-kl", type=float, default=1.0, help="forward KL weight used on rollout rows")
    a("--rollout-reverse-kl", type=float, default=None,
      help="reverse KL weight used on rollout rows (default: value of --reverse-kl)")
    a("--beta2", type=float, default=0.99)
    a("--kl-temp", type=float, default=1.0); a("--hidden-mse", type=float, default=0.0,
                                                   help="weight of relative MSE at the loop exit (0 = off)")
    a("--logit-chunk", type=int, default=1024, help="tokens per vocab-sized logits chunk")
    a("--grad-ckpt", action="store_true", help="activation checkpointing of decoder layers")
    a("--eval-every", type=int, default=200); a("--eval-seqs", type=int, default=32)
    a("--save-every", type=int, default=200); a("--gen-tokens", type=int, default=128)
    a("--reasoning-effort", default="none",
      help="chat-template effort; the SFT datasets carry no thinking traces, so 'none' keeps the data on-format")
    a("--data", default="allenai/tulu-3-sft-mixture:train:0.6,HuggingFaceH4/ultrachat_200k:train_sft:0.4")
    a("--eval-data", default="HuggingFaceH4/ultrachat_200k:test_sft:1")
    a("--llama-cpp-dir", default=str(Path(__file__).resolve().parent.parent / "llama.cpp"))
    a("--experts-from-gguf", default=None, help="replace all experts with the dequantized phone GGUF experts")
    a("--experts-int8", action="store_true",
      help="hold the GGUF experts as int8 + per-row scale on the GPU (fits 48 GB); needs --experts-impl eager")
    a("--dense-from-gguf", default=None,
      help="replace LoRA-target matmuls, routers and output head with the GGUF's dequantized values")
    a("--export-gguf-lora", default=None, help="after training (or with --resume and --max-steps 0), write adapter")
    args = p.parse_args(argv)
    if args.rollout_reverse_kl is None:
        args.rollout_reverse_kl = args.reverse_kl
    if args.dry_run_tiny:
        args.device, args.max_steps, args.seq_len, args.batch, args.grad_accum = "cpu", 20, 32, 2, 1
        args.eval_every, args.save_every, args.eval_seqs, args.gen_tokens = 10, 10, 4, 8
        args.lr, args.warmup, args.logit_chunk, args.lora_rank, args.lora_alpha = float(os.environ.get("DRY_LR", 1e-3)), 2, 24, 4, 8.0
        args.lora_router, args.lora_lm_head, args.hidden_mse, args.grad_ckpt = True, True, 0.01, True
        args.out = args.out if args.out != "runs/distill" else "runs/dry-run-tiny"
    return args


def parse_sources(spec: str) -> list[tuple[str, str, float]]:
    out = []
    for item in spec.split(","):
        name, split, weight = item.split(":")
        out.append((name, split, float(weight)))
    return out


def build_optimizer(trainable: dict, args):
    lora = [p for n, p in trainable.items() if "lora_" in n]
    dense = [p for n, p in trainable.items() if "lora_" not in n]
    groups = [{"params": lora, "lr": args.lr, "weight_decay": args.weight_decay}]
    if dense:
        groups.append({"params": dense, "lr": args.small_dense_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, args.beta2), eps=1e-8)
    total = max(args.max_steps, 1)

    def lr_lambda(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - args.warmup) / max(1, total - args.warmup))))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def setup(args):
    model, tok = tiny_model(args) if args.dry_run_tiny else real_model(args)
    parts = get_parts(model)
    for name, (lo, hi) in {"student_passes": (1, None), "teacher_passes": (1, None)}.items():
        if getattr(args, name) < lo:
            raise ValueError(f"--{name} must be >= {lo}")
    n_exp = parts.text.config.n_routed_experts
    if not (1 <= args.student_topk <= n_exp and 1 <= args.teacher_topk <= n_exp):
        raise ValueError(f"top-k must be in [1, {n_exp}]")
    if args.rollout_forward_kl < 0 or args.rollout_reverse_kl < 0:
        raise ValueError("--rollout-forward-kl and --rollout-reverse-kl must be >= 0")
    if args.experts_int8:
        if not args.experts_from_gguf:
            raise ValueError("--experts-int8 needs --experts-from-gguf")
        if args.experts_impl != "eager":
            raise ValueError("--experts-int8 needs --experts-impl eager (it indexes one expert at a time)")
        load_experts_int8_from_gguf(parts, args.experts_from_gguf, args)
    elif args.experts_from_gguf:
        load_experts_from_gguf(parts, args.experts_from_gguf, args)
    if args.dense_from_gguf:
        load_dense_from_gguf(parts, args.dense_from_gguf, args)
    wrapped = inject_lora(parts.text, parts.head_owner, args)
    model.to(args.device)
    set_resting_shape(parts, args.student_passes, args.student_topk)
    trainable = freeze_and_select(parts, args.train_small_dense)
    if args.grad_ckpt:
        model.activation_checkpointing_enable({"use_reentrant": False})
    model.train()
    n_train = sum(p.numel() for p in trainable.values())
    log.info("LoRA on %d linears; %d trainable tensors, %.2fM params", len(wrapped), len(trainable), n_train / 1e6)
    return parts, tok, trainable


def data_iters(args, tok, parts):
    pad_id = (tok.pad_token_id if (tok is not None and tok.pad_token_id is not None)
              else (parts.text.config.pad_token_id if parts.text.config.pad_token_id is not None else 0))
    if args.dry_run_tiny:
        vocab = parts.text.config.unpadded_vocab_size
        ord_it = batches(random_blocks(vocab, args.seq_len, args.seed), args.batch, args.device)
        if args.teacher_note:
            ro_it = rollout_batches(_TinyTokenizer(), [_synthetic_rollout_rows(args.seed)], [1.0],
                                    args.seq_len, args.batch, args.seed, args.teacher_note, pad_id, args.device)
            train = mix_batch_streams(ord_it, ro_it, 0.7, 0.3, args.grad_accum, args.seed)
        else:
            train = ord_it
        ev_it = batches(random_blocks(vocab, args.seq_len, args.seed + 1), args.batch, args.device)
        return train, [next(ev_it) for _ in range(max(1, args.eval_seqs // args.batch))]
    train = train_batches(tok, parse_sources(args.data), args.seq_len, args.batch, args.grad_accum,
                          args.seed, args.reasoning_effort, args.teacher_note, pad_id, args.device)
    ev_blocks = token_blocks(tok, parse_sources(args.eval_data), args.seq_len, 1234, args.reasoning_effort)
    evalb = []
    for b in batches(ev_blocks, args.batch, args.device):
        evalb.append(b)
        if len(evalb) * args.batch >= args.eval_seqs:
            break
    return train, evalb


EVAL_PROMPTS = [
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "Write a short, warm reply to a friend who just told you they failed their driving test.",
    "Explain why the sky is blue in three sentences.",
]


def train(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts, tok, trainable = setup(args)
    recorder = LoopExitRecorder(parts.text.layers[parts.loop_end - 1])
    opt, sched = build_optimizer(trainable, args)
    state = {"step": 0, "seqs": 0, "hours": 0.0}
    if args.resume and (out / "ckpt.pt").exists():
        state = load_ckpt(out, trainable, opt, sched)
        log.info("resumed at step %d (%d sequences)", state["step"], state["seqs"])
    elif args.init_from:
        blob = torch.load(args.init_from, map_location="cpu", weights_only=False)["trainable"]
        missing = sorted(set(trainable) - set(blob))
        if missing or set(blob) - set(trainable):
            raise ValueError(f"--init-from tensors do not match this run's trainable set (missing {missing[:3]})")
        with torch.no_grad():
            for k, v in blob.items():
                trainable[k].copy_(v.to(trainable[k].device, trainable[k].dtype))
        log.info("initialised %d trainable tensors from %s (fresh optimizer and schedule)", len(blob), args.init_from)
    digest0 = expert_digest(parts)
    train_it, eval_batches = data_iters(args, tok, parts)
    for _ in range(state["seqs"] // args.batch):  # deterministic stream: skip what was consumed
        next(train_it)
    prompts = EVAL_PROMPTS if tok is not None else ["1 2 3 4 5", "7 7 7 9"]
    t0, hours0, seqs0 = time.time(), state["hours"], state["seqs"]
    if state["step"] == 0:
        log.info("step 0 eval %s", evaluate(parts, eval_batches, args, recorder))
    while state["step"] < args.max_steps:
        stats = train_step(parts, train_it, opt, sched, trainable, args, recorder)
        state["step"] += 1
        state["seqs"] += args.batch * args.grad_accum
        state["hours"] = hours0 + (time.time() - t0) / 3600
        tok_s = (state["seqs"] - seqs0) * args.seq_len / max(1e-6, time.time() - t0)
        peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
        log.info("step %d [%s] loss %.4f %s lr %.2e tok/s %.0f gpu_peak_gb %.1f", state["step"], stats["batch_kind"],
                 stats["loss"], {k: round(v, 4) for k, v in stats.items() if k not in ("loss", "batch_kind")},
                 sched.get_last_lr()[0], tok_s, peak_gb)
        if state["step"] % args.eval_every == 0:
            ev = evaluate(parts, eval_batches, args, recorder)
            gens = greedy_samples(parts, tok, prompts[:2 if args.dry_run_tiny else 3], args, args.device)
            log.info("eval step %d %s", state["step"], ev)
            with open(out / "eval.jsonl", "a") as f:
                f.write(json.dumps({"step": state["step"], **ev, "samples": gens}) + "\n")
        hit_cap = (time.time() - t0) / 3600 >= args.max_hours  # this run only; state["hours"] is the running total
        if state["step"] % args.save_every == 0 or hit_cap:
            save_ckpt(out, trainable, opt, sched, state)
        if hit_cap:
            log.info("wall-clock cap %.2f h reached; checkpoint saved, stopping", args.max_hours)
            break
    save_ckpt(out, trainable, opt, sched, state)
    if expert_digest(parts) != digest0:
        raise RuntimeError("expert tensors changed during training")
    log.info("expert digest unchanged: %s", digest0)
    if args.export_gguf_lora:
        export_gguf_lora(parts, args.export_gguf_lora, args)
    if args.dry_run_tiny:
        dry_run_checks(parts, out, args, recorder, eval_batches)


def train_step(parts, train_it, opt, sched, trainable, args, recorder) -> dict:
    opt.zero_grad(set_to_none=True)
    agg: dict = {}
    counts: dict[str, int] = {}
    kinds: set[str] = set()
    for _ in range(args.grad_accum):
        batch = next(train_it)
        kinds.add("rollout" if isinstance(batch, RolloutBatch) else "ordinary")
        loss, stats = distill_loss(parts, batch, args, recorder)
        (loss / args.grad_accum).backward()
        agg["loss"] = agg.get("loss", 0.0) + float(loss.detach()) / args.grad_accum
        for k, v in stats.items():
            agg[k] = agg.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    for k, n in counts.items():
        agg[k] /= n
    params = list(trainable.values())
    agg["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
    agg["batch_kind"] = next(iter(kinds)) if len(kinds) == 1 else "mixed"
    opt.step()
    sched.step()
    return agg


def dry_run_checks(parts: Parts, out: Path, args, recorder, eval_batches) -> None:
    """Invariants the real run relies on, checked on the tiny model."""
    ids = eval_batches[0]
    # 1. teacher logits do not depend on LoRA (LoRA off in teacher shape)
    with torch.no_grad(), shape(parts, args.teacher_passes, args.teacher_topk, lora=False):
        a = parts.full(input_ids=ids, use_cache=False).logits
    saved = {n: p.detach().clone() for n, p in parts.full.named_parameters() if "lora_B" in n}
    with torch.no_grad():
        for n, p in parts.full.named_parameters():
            if "lora_B" in n:
                p.add_(1.0)
    with torch.no_grad(), shape(parts, args.teacher_passes, args.teacher_topk, lora=False):
        b = parts.full(input_ids=ids, use_cache=False).logits
    with torch.no_grad():
        for n, p in parts.full.named_parameters():
            if n in saved:
                p.copy_(saved[n])
    log.info("CHECK teacher independent of LoRA: max|diff| = %.3g", float((a - b).abs().max()))
    # 2. GGUF round trips: experts loader and adapter export, checked with llama.cpp's own shape rules
    tiny_gguf, adapter = str(out / "tiny_base.gguf"), str(out / "tiny_lora.gguf")
    write_tiny_gguf(parts, tiny_gguf, args)
    before = [l.mlp.experts.gate_up_proj.detach().clone() for l in parts.text.layers]
    load_experts_from_gguf(parts, tiny_gguf, args)
    err = max(float((l.mlp.experts.gate_up_proj - b0).abs().max() / b0.abs().max())
              for l, b0 in zip(parts.text.layers, before))
    log.info("CHECK experts from GGUF (Q8_0 round trip): max relative error %.4f", err)
    q_before = parts.text.layers[3].self_attn.q_proj.linear.base.weight.detach().clone()
    load_dense_from_gguf(parts, tiny_gguf, args)
    q_after = parts.text.layers[3].self_attn.q_proj.linear.base.weight
    dense_err = float((q_after - q_before).abs().max() / q_before.abs().max())
    log.info("CHECK dense from GGUF (Q8_0 round trip, layer 3 attn_q): max relative error %.4f", dense_err)
    err = max(err, dense_err)
    export_gguf_lora(parts, adapter, args)
    errors = check_adapter_against_gguf(adapter, tiny_gguf, args)
    log.info("CHECK adapter vs base GGUF shapes: %s", "OK" if not errors else errors)
    if errors or err > 0.02:
        raise RuntimeError("dry-run check failed")
    recovery_check(parts, args, recorder, eval_batches)
    int8_check(parts, tiny_gguf, args, recorder, eval_batches)
    rollout_check(parts, args, recorder)
    log.info("DRY RUN PASSED")


def rollout_check(parts: Parts, args, recorder) -> None:
    """Rollout batch with a teacher note: alignment of scored completion tokens, padding/prompt masking,
    teacher-note effect, and a backward step (including pure reverse-KL weights)."""
    set_resting_shape(parts, args.student_passes, args.student_topk)
    tok = _TinyTokenizer()
    pad_id = parts.text.config.pad_token_id if parts.text.config.pad_token_id is not None else 0
    note = args.teacher_note or "Think briefly."
    rows = [
        f"<|message_system|><|content_text|>E:0{END_MESSAGE}<|message_user|><|content_text|>2+2?{END_MESSAGE}"
        f"{MESSAGE_MODEL}<|content_thinking|>abc{END_MESSAGE}{MESSAGE_MODEL}<|content_text|>42{END_MESSAGE}",
        f"<|message_system|><|content_text|>E:0{END_MESSAGE}<|message_user|><|content_text|>hi{END_MESSAGE}"
        f"{MESSAGE_MODEL}<|content_thinking|>step one two three four five{END_MESSAGE}"
        f"{MESSAGE_MODEL}<|content_text|>done{END_MESSAGE}",
    ]
    items = [prepare_rollout(r, tok, args.seq_len, note) for r in rows]
    plain_items = [prepare_rollout(r, tok, args.seq_len, "") for r in rows]
    if any(x is None for x in items) or any(x is None for x in plain_items):
        raise RuntimeError("rollout check: prepare_rollout returned None")
    batch = collate_rollouts(items, pad_id, args.device)
    plain = collate_rollouts(plain_items, pad_id, args.device)
    extra_toks = batch.t_ids.shape[1] - batch.s_ids.shape[1]
    if batch.s_ids.shape[1] != args.seq_len or extra_toks <= 0:
        raise RuntimeError(f"rollout check: unexpected shapes s={tuple(batch.s_ids.shape)} t={tuple(batch.t_ids.shape)}")
    if int(batch.s_mask.sum()) != int(batch.t_mask.sum()) or int(batch.s_mask.sum()) == 0:
        raise RuntimeError("rollout check: scored token counts differ")
    if not torch.equal(batch.s_ids[batch.s_mask], batch.t_ids[batch.t_mask]):
        raise RuntimeError("rollout check: teacher/student input ids misaligned at scored positions")
    for b in range(len(rows)):
        s_pos = batch.s_mask[b].nonzero(as_tuple=True)[0]
        t_pos = batch.t_mask[b].nonzero(as_tuple=True)[0]
        if (len(s_pos) != len(t_pos) or s_pos[0] <= 0 or t_pos[0] <= s_pos[0]
                or int(batch.s_ids[b, s_pos[0]]) != tok.SPECIAL[MESSAGE_MODEL]
                or not torch.equal(batch.s_ids[b, s_pos + 1], batch.t_ids[b, t_pos + 1])
                or batch.s_mask[b][batch.s_attn[b] == 0].any() or batch.t_mask[b][batch.t_attn[b] == 0].any()):
            raise RuntimeError(f"rollout check: alignment/masking failed for sequence {b}")
    if not torch.equal(batch.s_ids, plain.s_ids) or not torch.equal(batch.s_mask, plain.s_mask):
        raise RuntimeError("rollout check: teacher note changed student inputs")
    t_note, _ = teacher_forward(parts, batch.t_ids, args, recorder, attention_mask=batch.t_attn)
    t_plain, _ = teacher_forward(parts, plain.t_ids, args, recorder, attention_mask=plain.t_attn)
    note_diff = float((t_note[batch.t_mask] - t_plain[plain.t_mask]).abs().max())
    log.info("CHECK rollout alignment (note +%d tok, scored %d tok, teacher diff %.3g): OK",
             extra_toks, int(batch.s_mask.sum()), note_diff)
    if note_diff <= 1e-4:
        raise RuntimeError("rollout check: teacher outputs did not change with teacher note")
    digest = expert_digest(parts)
    trainable = {n: p for n, p in parts.full.named_parameters() if p.requires_grad}
    opt, sched = build_optimizer(trainable, args)
    stats = train_step(parts, itertools.repeat(batch), opt, sched, trainable, args, recorder)
    rev_args = argparse.Namespace(**{**vars(args), "rollout_forward_kl": 0.0, "rollout_reverse_kl": 1.0})
    stats_rev = train_step(parts, itertools.repeat(batch), opt, sched, trainable, rev_args, recorder)
    log.info("CHECK rollout train step: %s (rev-only loss %.4f grad_norm %.4f)",
             {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()},
             stats_rev["loss"], stats_rev["grad_norm"])
    if (stats["batch_kind"] != "rollout" or "ro_kl" not in stats or "ro_rev_kl" not in stats
            or not math.isfinite(stats["loss"]) or not math.isfinite(stats_rev["loss"])
            or stats_rev["grad_norm"] <= 0 or expert_digest(parts) != digest
            or ((args.rollout_forward_kl > 0 or args.rollout_reverse_kl > 0) and stats["grad_norm"] <= 0)):
        raise RuntimeError("rollout train step check failed")


def int8_check(parts: Parts, gguf_path: str, args, recorder, eval_batches) -> None:
    """--experts-int8 path: outputs stay close to the dequantized experts, and a training step still runs."""
    ids = eval_batches[0]
    load_experts_from_gguf(parts, gguf_path, args)
    ref, _ = teacher_forward(parts, ids, args, recorder)
    load_experts_int8_from_gguf(parts, gguf_path, args)
    got, _ = teacher_forward(parts, ids, args, recorder)
    rel = float((got.float() - ref.float()).norm() / ref.float().norm())
    log.info("CHECK int8 experts vs dequantized experts: relative output error %.4f", rel)
    digest = expert_digest(parts)
    trainable = {n: p for n, p in parts.full.named_parameters() if p.requires_grad}
    opt, sched = build_optimizer(trainable, args)
    it = batches(random_blocks(parts.text.config.unpadded_vocab_size, args.seq_len, 11), args.batch, args.device)
    stats = train_step(parts, it, opt, sched, trainable, args, recorder)
    log.info("CHECK int8 train step: loss %.4f grad_norm %.4f", stats["loss"], stats["grad_norm"])
    if rel > 0.05 or not math.isfinite(stats["loss"]) or stats["grad_norm"] <= 0 or expert_digest(parts) != digest:
        raise RuntimeError("int8 expert check failed")


def recovery_check(parts: Parts, args, recorder, eval_batches, steps: int = 30) -> None:
    """Gradient/generalization sanity: give the student the teacher's shape plus a random low-rank
    perturbation (non-zero lora_B), then check that training shrinks the HELD-OUT KL. On a random tiny
    model the 1-pass/top-4 gap is noise-like and does not generalize in 20 steps; this test has a
    learnable target, so it exercises the same loss, backward, checkpointing and optimizer path."""
    from argparse import Namespace

    rargs = Namespace(**{**vars(args), "student_passes": args.teacher_passes, "student_topk": args.teacher_topk,
                         "lr": 3e-3, "hidden_mse": 0.0, "max_steps": steps})
    set_resting_shape(parts, rargs.student_passes, rargs.student_topk)
    trainable = {n: p for n, p in parts.full.named_parameters() if p.requires_grad}
    torch.manual_seed(1)
    with torch.no_grad():
        for n, p in trainable.items():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    opt, sched = build_optimizer(trainable, rargs)
    train_it = batches(random_blocks(parts.text.config.unpadded_vocab_size, args.seq_len, 7), args.batch, args.device)
    before = evaluate(parts, eval_batches, rargs, recorder)["eval_kl"]
    for _ in range(steps):
        train_step(parts, train_it, opt, sched, trainable, rargs, recorder)
    after = evaluate(parts, eval_batches, rargs, recorder)["eval_kl"]
    log.info("CHECK recover a known low-rank perturbation: held-out KL %.4f -> %.4f", before, after)
    if not after < 0.8 * before:
        raise RuntimeError("recovery check failed: held-out KL did not drop by 20%")


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
