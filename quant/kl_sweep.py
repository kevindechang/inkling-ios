#!/usr/bin/env python3
"""Targeted 1-pass / 4-expert KL sensitivity sweeper for TML-d24-varloop-research (inkling2_loop).

Built on train/distill.py's Inkling2LoopForConditionalGeneration loader and ctypes calls to
libggml-base (ggml_quantize_chunk + ggml_get_type_traits().to_float). Measures exact single-tensor
full-vocab KL divergence at the phone execution shape (default 1 pass, top-4 experts) with zero
intermediate GGUF files on disk.

Dry run on Mac CPU (no GPU, no network):
    PYTHONPATH=train/.pydeps python3 train/kl_sweep.py --dry-run --out runs/agy/dry_sens.json
"""

from __future__ import annotations

import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = Path(__file__).resolve().parent
PYDEPS = TRAIN_DIR / ".pydeps"
if PYDEPS.is_dir() and str(PYDEPS) not in sys.path:
    sys.path.insert(0, str(PYDEPS))
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import distill as D  # noqa: E402

log = logging.getLogger("kl_sweep")

VARLOOP_MODEL_ID = "thinkingmachines/TML-d24-varloop-research"
VARLOOP_REVISION = "c90921bd5f78df6c8168173b9f8a3a850320c594"

# Default menus from runs/agy/COMPILER_SCOPE.md Section 4 Step 1.
DEFAULT_GATE_UP_MENU = ["Q2_K", "Q2_0", "Q3_K", "IQ2_S", "Q4_K"]
DEFAULT_DOWN_MENU = ["Q2_0", "MXFP4", "Q4_0", "IQ4_NL", "NVFP4"]
DEFAULT_ATTN_MENU = ["Q4_K", "Q5_K", "Q6_K", "Q8_0"]


# -------------------------------------------------------------------------------- ggml ctypes bridge

class _GgmlTypeTraits(ctypes.Structure):
    _fields_ = [
        ("type_name", ctypes.c_char_p),
        ("blck_size", ctypes.c_int64),
        ("blck_size_interleave", ctypes.c_int64),
        ("type_size", ctypes.c_size_t),
        ("is_quantized", ctypes.c_bool),
        ("to_float", ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)),
        ("from_float_ref", ctypes.c_void_p),
    ]


class GgmlQuantizer:
    """In-memory llama.cpp quantize + dequantize via libggml-base."""

    def __init__(self, lib_path: str | Path, llama_cpp_dir: str | Path | None = None, n_threads: int = 1):
        self.lib_path = str(lib_path)
        self.n_threads = max(1, int(n_threads))
        self._pool = ThreadPoolExecutor(max_workers=self.n_threads) if self.n_threads > 1 else None
        self.lib = ctypes.CDLL(self.lib_path)

        self.lib.ggml_type_name.argtypes = [ctypes.c_int]
        self.lib.ggml_type_name.restype = ctypes.c_char_p
        self.lib.ggml_blck_size.argtypes = [ctypes.c_int]
        self.lib.ggml_blck_size.restype = ctypes.c_int64
        self.lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self.lib.ggml_row_size.restype = ctypes.c_size_t
        self.lib.ggml_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ggml_quantize_init.restype = None
        self.lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
        self.lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        self.lib.ggml_quantize_chunk.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
        self.lib.ggml_get_type_traits.restype = ctypes.POINTER(_GgmlTypeTraits)

        type_count = self._detect_type_count(lib_path, llama_cpp_dir)
        self.name_to_id: dict[str, int] = {}
        self.blck_sizes: dict[str, int] = {}
        for tid in range(type_count):
            raw = self.lib.ggml_type_name(tid)
            if not raw:
                continue
            name = raw.decode("ascii", errors="ignore").upper()
            if name and name != "NONE":
                self.name_to_id[name] = tid
                self.blck_sizes[name] = int(self.lib.ggml_blck_size(tid))
        self._inited: set[int] = set()

    @staticmethod
    def _detect_type_count(lib_path: str | Path, llama_cpp_dir: str | Path | None) -> int:
        import re as _re

        candidates: list[Path] = []
        if llama_cpp_dir is not None:
            candidates.append(Path(llama_cpp_dir) / "ggml" / "include" / "ggml.h")
        p = Path(lib_path).resolve()
        for parent in p.parents:
            candidates.append(parent / "ggml" / "include" / "ggml.h")
        for hdr in candidates:
            if hdr.is_file():
                m = _re.search(r"GGML_TYPE_COUNT\s*=\s*(\d+)", hdr.read_text(encoding="utf-8", errors="ignore"))
                if m:
                    return int(m.group(1))
        return 44

    @staticmethod
    def find_library(llama_cpp_dir: str | Path, explicit: str | None = None) -> Path:
        if explicit:
            p = Path(explicit)
            if not p.is_file():
                raise FileNotFoundError(f"libggml-base not found at {p}")
            return p
        ext = "dylib" if platform.system() == "Darwin" else "so"
        base = Path(llama_cpp_dir)
        for cand in (
            base / "build" / "bin" / f"libggml-base.{ext}",
            base / "build" / "ggml" / "src" / f"libggml-base.{ext}",
            base / "build" / f"libggml-base.{ext}",
        ):
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"libggml-base.{ext} not found under {base / 'build'}")

    def blck_size(self, type_name: str) -> int:
        return self.blck_sizes[type_name.upper()]

    def is_legal(self, type_name: str, ne0: int) -> bool:
        t = type_name.upper()
        if t not in self.name_to_id:
            return False
        bs = self.blck_sizes[t]
        return bs > 0 and (ne0 % bs == 0)

    def requires_imatrix(self, type_name: str) -> bool:
        tid = self.name_to_id[type_name.upper()]
        return bool(self.lib.ggml_quantize_requires_imatrix(tid))

    def _ensure_init(self, tid: int) -> None:
        if tid not in self._inited:
            self.lib.ggml_quantize_init(tid)
            self._inited.add(tid)

    def quant_dequant_2d(
        self,
        w_f32: np.ndarray,
        type_name: str,
        imatrix: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        """Quantize [nrows, ncols] float32 array and dequantize back to float32."""
        t = type_name.upper()
        if w_f32.ndim != 2:
            raise ValueError(f"expected 2D array, got {w_f32.shape}")
        nrows, ncols = w_f32.shape
        if t == "F32":
            return w_f32.astype(np.float32, copy=True), nrows * ncols * 4
        if t == "F16":
            f16 = w_f32.astype(np.float16)
            return f16.astype(np.float32), nrows * ncols * 2
        if t == "BF16":
            t_bf16 = torch.from_numpy(w_f32).to(torch.bfloat16).float().numpy()
            return t_bf16, nrows * ncols * 2

        tid = self.name_to_id[t]
        bs = self.blck_sizes[t]
        if ncols % bs != 0:
            raise ValueError(f"{t} requires ne[0] ({ncols}) divisible by blck_size ({bs})")
        self._ensure_init(tid)

        src = np.ascontiguousarray(w_f32, dtype=np.float32)
        row_bytes = int(self.lib.ggml_row_size(tid, ncols))
        n_bytes = row_bytes * nrows
        dst = np.empty(n_bytes, dtype=np.uint8)

        imat_ptr = ctypes.POINTER(ctypes.c_float)()
        if imatrix is not None:
            im = np.ascontiguousarray(imatrix, dtype=np.float32)
            if im.size != ncols:
                raise ValueError(f"imatrix size {im.size} != ncols {ncols}")
            imat_ptr = im.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        elif self.requires_imatrix(t):
            im = np.ones(ncols, dtype=np.float32)
            imat_ptr = im.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        traits = self.lib.ggml_get_type_traits(tid).contents
        out = np.empty((nrows, ncols), dtype=np.float32)

        def _run_rows(r0: int, nr: int) -> int:
            src_ptr = src[r0:].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            dst_ptr = ctypes.c_void_p(dst.ctypes.data + r0 * row_bytes)
            written = int(self.lib.ggml_quantize_chunk(tid, src_ptr, dst_ptr, 0, nr, ncols, imat_ptr))
            if written != nr * row_bytes:
                raise RuntimeError(f"ggml_quantize_chunk({t}) wrote {written} != {nr * row_bytes}")
            out_ptr = out[r0:].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            traits.to_float(dst_ptr, out_ptr, nr * ncols)
            return written

        if self._pool is not None and nrows >= 64:
            chunk = max(32, (nrows + self.n_threads - 1) // self.n_threads)
            ranges = [(r0, min(chunk, nrows - r0)) for r0 in range(0, nrows, chunk)]
            futs = [self._pool.submit(_run_rows, r0, nr) for r0, nr in ranges]
            for f in futs:
                f.result()
        else:
            _run_rows(0, nrows)
        return out, n_bytes

    def quant_dequant_3d(
        self,
        w_f32: np.ndarray,
        type_name: str,
        imatrix: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        """Quantize [n_exp, nrows, ncols] 3D expert stack slice-by-slice (parallel across experts)."""
        if w_f32.ndim != 3:
            raise ValueError(f"expected 3D array, got {w_f32.shape}")
        t = type_name.upper()
        n_exp, nrows, ncols = w_f32.shape
        if t in self.name_to_id:
            self._ensure_init(self.name_to_id[t])
        out = np.empty_like(w_f32, dtype=np.float32)

        def _im_for(e: int) -> np.ndarray | None:
            if imatrix is None:
                return None
            if imatrix.ndim == 2 and imatrix.shape[0] == n_exp:
                return imatrix[e]
            if imatrix.size == n_exp * ncols:
                return imatrix.reshape(n_exp, ncols)[e]
            if imatrix.size == ncols:
                return imatrix
            return None

        def _one_expert(e: int) -> int:
            res, nb = self.quant_dequant_2d(w_f32[e], t, _im_for(e))
            out[e] = res
            return nb

        if self._pool is not None and n_exp > 1:
            # Parallelize across experts (inner 2D runs single-threaded on worker to avoid pool deadlock)
            saved_pool = self._pool
            self._pool = None
            try:
                total_bytes = sum(saved_pool.map(_one_expert, range(n_exp)))
            finally:
                self._pool = saved_pool
        else:
            total_bytes = sum(_one_expert(e) for e in range(n_exp))
        return out, total_bytes


def load_imatrix_gguf(path: str | None, llama_cpp_dir: str) -> dict[str, np.ndarray]:
    """Load GGUF imatrix normalized exactly like llama.cpp/tools/quantize/quantize.cpp:199-215."""
    if not path:
        return {}
    gguf = D.import_gguf(llama_cpp_dir)
    reader = gguf.GGUFReader(path)
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for t in reader.tensors:
        if t.name.endswith(".in_sum2"):
            sums[t.name[: -len(".in_sum2")]] = np.asarray(t.data, dtype=np.float32).reshape(-1)
        elif t.name.endswith(".counts"):
            counts[t.name[: -len(".counts")]] = np.asarray(t.data, dtype=np.float32).reshape(-1)
    out: dict[str, np.ndarray] = {}
    for name, s in sums.items():
        c = counts.get(name)
        if c is None or c.size == 0:
            out[name] = s.copy()
            continue
        nc = int(c.size)
        ne0 = s.size // nc
        arr = np.empty_like(s, dtype=np.float32)
        for j in range(nc):
            cj = float(c[j])
            sl = slice(j * ne0, (j + 1) * ne0)
            arr[sl] = (s[sl] / cj) if cj > 0.0 else 1.0
        out[name] = arr
    log.info("loaded imatrix %s (%d tensors)", path, len(out))
    return out


# -------------------------------------------------------------------------------- tensor inventory

@dataclass
class SweepItem:
    gguf_name: str
    kind: str        # "experts" or "linear"
    role: str        # e.g. "ffn_gate_up_exps", "ffn_down_exps", "attn_q", ...
    layer: int
    module: nn.Module
    attr: str        # "weight", "gate_up_proj", or "down_proj"
    shape: tuple[int, ...]
    options: list[str]
    baseline_type: str | None = None


def collect_sweep_items(
    parts: D.Parts,
    quantizer: GgmlQuantizer,
    gate_up_menu: list[str],
    down_menu: list[str],
    attn_menu: list[str],
    baseline_types: dict[str, str],
    only_wide_attn: bool = False,
    max_layers: int | None = None,
) -> list[SweepItem]:
    items: list[SweepItem] = []
    n_layers = len(parts.text.layers)
    if max_layers is not None:
        n_layers = min(n_layers, max_layers)

    for i in range(n_layers):
        layer = parts.text.layers[i]
        experts = layer.mlp.experts
        for stem, attr, menu in (
            ("ffn_gate_up_exps", "gate_up_proj", gate_up_menu),
            ("ffn_down_exps", "down_proj", down_menu),
        ):
            t = getattr(experts, attr)
            gname = f"blk.{i}.{stem}.weight"
            ne0 = int(t.shape[-1])
            legal = [opt for opt in menu if quantizer.is_legal(opt, ne0)]
            if legal:
                items.append(SweepItem(
                    gguf_name=gname, kind="experts", role=stem, layer=i,
                    module=experts, attr=attr, shape=tuple(int(x) for x in t.shape),
                    options=legal, baseline_type=baseline_types.get(gname),
                ))

        for suffix, stem in D.LORA_TARGETS.items():
            if stem in ("attn_sink_proj", "attn_k_conv_proj", "attn_v_conv_proj"):
                continue  # fixed BF16 -> F16 (lossless)
            try:
                mod = layer.get_submodule(suffix)
            except AttributeError:
                continue
            gname = f"blk.{i}.{stem}.weight"
            btype = baseline_types.get(gname)
            if only_wide_attn and btype and btype not in ("Q8_0", "Q6_K", "Q5_K"):
                continue
            ne0 = int(mod.weight.shape[-1])
            legal = [opt for opt in attn_menu if quantizer.is_legal(opt, ne0)]
            if legal:
                items.append(SweepItem(
                    gguf_name=gname, kind="linear", role=stem, layer=i,
                    module=mod, attr="weight", shape=tuple(int(x) for x in mod.weight.shape),
                    options=legal, baseline_type=btype,
                ))
    return items


def read_gguf_tensor_types(gguf_path: str | None, llama_cpp_dir: str) -> dict[str, str]:
    if not gguf_path:
        return {}
    p = Path(gguf_path)
    if p.suffix.lower() == ".json":
        return {k: str(v).upper() for k, v in json.loads(p.read_text()).items()}
    gguf = D.import_gguf(llama_cpp_dir)
    return {t.name: t.tensor_type.name.upper() for t in gguf.GGUFReader(gguf_path).tensors}


# -------------------------------------------------------------------------------- KL evaluation

@torch.no_grad()
def forward_hidden(parts: D.Parts, windows: list[torch.Tensor]) -> list[torch.Tensor]:
    outs = []
    for ids in windows:
        h = parts.text(input_ids=ids, use_cache=False).last_hidden_state
        outs.append(h.detach())
    return outs


@torch.no_grad()
def compute_kl_against_ref(
    parts: D.Parts,
    windows: list[torch.Tensor],
    ref_hiddens: list[torch.Tensor],
    logit_chunk: int,
) -> float:
    """Mean per-token KL(ref || cand) over all windows using chunked full-vocab logits."""
    total_kl = 0.0
    total_toks = 0
    for ids, ref_h in zip(windows, ref_hiddens):
        cand_h = parts.text(input_ids=ids, use_cache=False).last_hidden_state
        hdim = cand_h.shape[-1]
        r_flat = ref_h.reshape(-1, hdim)
        c_flat = cand_h.reshape(-1, hdim)
        n_tok = r_flat.shape[0]
        for start in range(0, n_tok, logit_chunk):
            end = min(start + logit_chunk, n_tok)
            r_logp = F.log_softmax(D.head_logits(parts, r_flat[start:end], use_lora=False), dim=-1)
            c_logp = F.log_softmax(D.head_logits(parts, c_flat[start:end], use_lora=False), dim=-1)
            total_kl += float((r_logp.exp() * (r_logp - c_logp)).sum())
        total_toks += n_tok
    return total_kl / max(total_toks, 1)


def convert_experts_to_int8(parts: D.Parts, device: str) -> dict[tuple[int, str], torch.Tensor]:
    """Keep BF16 expert masters on CPU (128 GB host RAM) and put int8 QuantExperts on GPU (48 GB L40S)."""
    cpu_masters: dict[tuple[int, str], torch.Tensor] = {}
    for i, layer in enumerate(parts.text.layers):
        experts = layer.mlp.experts
        experts.config._experts_implementation = "eager"
        for attr in ("gate_up_proj", "down_proj"):
            param = getattr(experts, attr)
            cpu_w = param.detach().cpu()
            cpu_masters[(i, attr)] = cpu_w
            w_f32 = cpu_w.float()
            row_scale = w_f32.abs().amax(dim=-1, keepdim=True) / D.INT8_MAX
            row_scale = torch.where(row_scale == 0, torch.ones_like(row_scale), row_scale)
            q = torch.round(w_f32 / row_scale).to(torch.int8).to(device)
            sc = row_scale.to(cpu_w.dtype).to(device)
            del experts._parameters[attr]
            setattr(experts, attr, D.QuantExperts(q, sc, cpu_w.dtype))
    return cpu_masters


def swap_and_measure(
    parts: D.Parts,
    item: SweepItem,
    w_deq_f32: np.ndarray,
    windows: list[torch.Tensor],
    ref_hiddens: list[torch.Tensor],
    logit_chunk: int,
    device: str,
    floor_kl: float,
) -> float:
    cur = getattr(item.module, item.attr)
    if isinstance(cur, D.QuantExperts):
        # Swap a single layer's expert tensor as BF16 on GPU (~0.34-0.68 GB) while others stay int8.
        tmp = torch.from_numpy(w_deq_f32).to(dtype=cur.dtype, device=device)
        setattr(item.module, item.attr, tmp)
        try:
            kl = compute_kl_against_ref(parts, windows, ref_hiddens, logit_chunk)
        finally:
            setattr(item.module, item.attr, cur)
            del tmp
    else:
        backup = cur.data.detach().clone()
        cur.data.copy_(torch.from_numpy(w_deq_f32).to(dtype=cur.dtype, device=cur.device))
        try:
            kl = compute_kl_against_ref(parts, windows, ref_hiddens, logit_chunk)
        finally:
            cur.data.copy_(backup)
            del backup
    return max(0.0, kl - floor_kl)


# -------------------------------------------------------------------------------- dry-run tiny model

class _DryRunConfig:
    def __init__(self):
        self.vocab_size = 320
        self.unpadded_vocab_size = 300
        self.hidden_size = 256
        self.num_hidden_layers = 4
        self.moe_intermediate_size = 128
        self.n_routed_experts = 8
        self.num_experts_per_tok = 4
        self.loop_layer_index_start = 1
        self.loop_layer_index_end = 3
        self.loop_num_passes = 1


class _DryRunRouter(nn.Module):
    def __init__(self, hidden: int, n_exp: int, top_k: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_exp, hidden) * 0.05)
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(x, self.weight)
        vals, idx = torch.topk(logits, self.top_k, dim=-1)
        return F.softmax(vals, dim=-1), idx


class _DryRunExperts(nn.Module):
    def __init__(self, n_exp: int, hidden: int, inter: int):
        super().__init__()
        # Exact inkling2_loop shapes: gate_up [E, 2*inter, hidden], down [E, hidden, inter]
        self.gate_up_proj = nn.Parameter(torch.randn(n_exp, 2 * inter, hidden) * 0.04)
        self.down_proj = nn.Parameter(torch.randn(n_exp, hidden, inter) * 0.04)


class _DryRunMLP(nn.Module):
    def __init__(self, hidden: int, inter: int, n_exp: int, top_k: int):
        super().__init__()
        self.gate = _DryRunRouter(hidden, n_exp, top_k)
        self.experts = _DryRunExperts(n_exp, hidden, inter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, h = x.shape
        flat = x.reshape(-1, h)
        weights, idx = self.gate(flat)
        out = torch.zeros_like(flat)
        gu_all = self.experts.gate_up_proj
        dn_all = self.experts.down_proj
        inter = dn_all.shape[2]
        for k in range(idx.shape[1]):
            ek = idx[:, k]
            wk = weights[:, k : k + 1]
            for e in ek.unique().tolist():
                mask = ek == e
                xe = flat[mask]
                gu = F.linear(xe, gu_all[e])
                act = F.silu(gu[:, :inter]) * gu[:, inter:]
                out[mask] = out[mask] + wk[mask] * F.linear(act, dn_all[e])
        return out.reshape(bsz, seq, h)


class _LinearWrap(nn.Module):
    def __init__(self, in_f: int, out_f: int, with_gate: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=False)
        if with_gate:
            self.gate_proj = nn.Linear(in_f, out_f, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _DryRunAttn(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        # Exact inkling2_loop submodule paths from train/distill.py LORA_TARGETS
        self.q_proj = _LinearWrap(hidden, hidden)
        self.k_proj = _LinearWrap(hidden, hidden)
        self.v_proj = _LinearWrap(hidden, hidden)
        self.o_proj = _LinearWrap(hidden, hidden, with_gate=True)
        self.relative_position_bias = _LinearWrap(hidden, hidden // 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scale = 1.0 / math.sqrt(x.shape[-1])
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        t = x.shape[1]
        mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device), diagonal=1)
        probs = F.softmax(scores + mask, dim=-1)
        ctx = torch.matmul(probs, v) * torch.sigmoid(self.o_proj.gate_proj(x))
        rel = self.relative_position_bias(x).mean(dim=-1, keepdim=True)
        return self.o_proj(ctx + 0.05 * rel)


class _DryRunLayer(nn.Module):
    def __init__(self, cfg: _DryRunConfig):
        super().__init__()
        self.self_attn = _DryRunAttn(cfg.hidden_size)
        self.mlp = _DryRunMLP(cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_routed_experts, cfg.num_experts_per_tok)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(F.layer_norm(x, (x.shape[-1],)))
        x = x + self.mlp(F.layer_norm(x, (x.shape[-1],)))
        return x


class _DryRunTextOut:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class _DryRunTextModel(nn.Module):
    def __init__(self, cfg: _DryRunConfig):
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([_DryRunLayer(cfg) for _ in range(cfg.num_hidden_layers)])

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> _DryRunTextOut:
        h = self.embed_tokens(input_ids)
        for idx, layer in enumerate(self.layers):
            if self.config.loop_layer_index_start <= idx < self.config.loop_layer_index_end:
                for _ in range(self.config.loop_num_passes):
                    h = layer(h)
            else:
                h = layer(h)
        return _DryRunTextOut(F.layer_norm(h, (h.shape[-1],)))


class _DryRunCausalLM(nn.Module):
    def __init__(self, cfg: _DryRunConfig):
        super().__init__()
        self.model = _DryRunTextModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def logit_softcap(self, logits: torch.Tensor) -> torch.Tensor:
        return 30.0 * torch.tanh(logits / 30.0)


def dry_run_model(seed: int):
    """Small inkling2_loop model with hidden_size=256 (QK_K-divisible) and moe_intermediate=128
    (NOT QK_K-divisible, mirroring the real model's 2304 / 1152 divisibility asymmetry)."""
    torch.manual_seed(seed)
    try:
        from transformers import Inkling2LoopForCausalLM, Inkling2LoopTextConfig

        cfg = Inkling2LoopTextConfig(
            vocab_size=320, unpadded_vocab_size=300, hidden_size=256, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=64,
            swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=64,
            sliding_window_size=32, d_rel=4, rel_extent=32, log_scaling_n_floor=16,
            layer_types=["hybrid_sliding", "hybrid_sliding", "hybrid_sliding", "hybrid"],
            n_layers_per_attn_res_block=2, moe_intermediate_size=128, n_routed_experts=8,
            num_experts_per_tok=4, loop_layer_index_start=1, loop_layer_index_end=3,
            loop_num_passes=1, loop_gate_rank=8, eos_token_id=299, pad_token_id=298,
        )
        cfg._attn_implementation = "eager"
        model = Inkling2LoopForCausalLM(cfg)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.endswith("query") or n.endswith("e_score_correction_bias"):
                    p.normal_(0, 0.5)
            model.lm_head.weight.normal_(0, 0.5)
        return model.float().eval(), None
    except Exception as exc:
        log.info("transformers Inkling2Loop unavailable (%s); using built-in dry-run inkling2_loop mock", exc)
        cfg = _DryRunConfig()
        model = _DryRunCausalLM(cfg)
        with torch.no_grad():
            model.lm_head.weight.normal_(0, 0.5)
        return model.float().eval(), None


# -------------------------------------------------------------------------------- CLI + main

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--dry-run", action="store_true", help="CPU smoke test on a handful of tensors without GPU or network")
    a("--model", default=VARLOOP_MODEL_ID)
    a("--revision", default=VARLOOP_REVISION)
    a("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a("--experts-impl", default="eager", choices=["eager", "grouped_mm", "batched_mm"])
    a("--experts-int8", action="store_true", help="keep non-swapped experts as int8 on GPU (for 48 GB L40S)")
    a("--passes", type=int, default=1, help="loop passes during KL sweep (phone shape = 1)")
    a("--top-k", type=int, default=4, help="routed experts per token during KL sweep (phone shape = 4)")
    a("--windows", type=int, default=16, help="number of calibration/eval windows")
    a("--seq-len", type=int, default=512, help="tokens per window")
    a("--eval-batch", type=int, default=4, help="windows batched per GPU forward pass")
    a("--logit-chunk", type=int, default=512, help="tokens per vocab-logit chunk")
    a("--seed", type=int, default=1)
    a("--eval-data", default="HuggingFaceH4/ultrachat_200k:test_sft:0.5,allenai/tulu-3-sft-mixture:train:0.5")
    a("--reasoning-effort", default="none")
    a("--imatrix", default=None, help="path to imatrix GGUF (imatrix-TML-d24-varloop-research.gguf)")
    a("--baseline-gguf", default=None, help="optional baseline GGUF to read existing per-tensor types")
    a("--only-wide-attn", action="store_true", help="only sweep Q8_0/Q6_K/Q5_K attention tensors (per scope)")
    a("--gate-up-menu", default=",".join(DEFAULT_GATE_UP_MENU))
    a("--down-menu", default=",".join(DEFAULT_DOWN_MENU))
    a("--attn-menu", default=",".join(DEFAULT_ATTN_MENU))
    a("--max-layers", type=int, default=None, help="limit number of layers swept (for quick tests)")
    a("--max-tensors", type=int, default=None, help="limit total tensors swept")
    a("--llama-cpp-dir", default=str(ROOT / "llama.cpp"))
    a("--libggml-base", default=None, help="explicit path to libggml-base.{dylib,so}")
    a("--quant-threads", type=int, default=min(16, os.cpu_count() or 1), help="CPU threads for ggml_quantize_chunk")
    a("--out", default="runs/agy/kl_sens_1p_k4.json", help="output sensitivity JSON (resumable via .jsonl)")
    args = p.parse_args(argv)
    if args.dry_run:
        args.device = "cpu"
        args.windows = min(args.windows, 4)
        args.eval_batch = 1
        args.seq_len = min(args.seq_len, 64)
        args.logit_chunk = 64
        args.max_layers = args.max_layers or 2
        args.max_tensors = args.max_tensors or 8
    return args


def build_windows(args, tok, parts: D.Parts) -> list[torch.Tensor]:
    bsz = max(1, min(args.eval_batch, args.windows))
    if args.dry_run:
        vocab = parts.text.config.unpadded_vocab_size
        it = D.batches(D.random_blocks(vocab, args.seq_len, args.seed), bsz, args.device)
    else:
        blocks = D.token_blocks(tok, D.parse_sources(args.eval_data), args.seq_len, args.seed, args.reasoning_effort)
        it = D.batches(blocks, bsz, args.device)
    out = []
    remaining = args.windows
    while remaining > 0:
        b = next(it)
        if b.shape[0] > remaining:
            b = b[:remaining]
        out.append(b)
        remaining -= b.shape[0]
    return out


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    lib_path = GgmlQuantizer.find_library(args.llama_cpp_dir, args.libggml_base)
    quantizer = GgmlQuantizer(lib_path, n_threads=args.quant_threads)
    log.info("loaded %s (%d ggml types, quant_threads=%d)", lib_path, len(quantizer.name_to_id), quantizer.n_threads)

    if args.dry_run:
        model, tok = dry_run_model(args.seed)
    else:
        model, tok = D.real_model(args)
        model.eval()
    parts = D.get_parts(model)

    cpu_masters: dict[tuple[int, str], torch.Tensor] = {}
    if args.experts_int8 and not args.dry_run:
        cpu_masters = convert_experts_to_int8(parts, args.device)
        model.to(args.device)
    else:
        model.to(args.device)

    D.set_resting_shape(parts, args.passes, args.top_k)
    D.LoRAState.enabled = False

    imatrix = load_imatrix_gguf(args.imatrix, args.llama_cpp_dir)
    baseline_types = read_gguf_tensor_types(args.baseline_gguf, args.llama_cpp_dir)

    gate_up_menu = [x.strip().upper() for x in args.gate_up_menu.split(",") if x.strip()]
    down_menu = [x.strip().upper() for x in args.down_menu.split(",") if x.strip()]
    attn_menu = [x.strip().upper() for x in args.attn_menu.split(",") if x.strip()]

    # Verify block-size divisibility filtering explicitly.
    items = collect_sweep_items(
        parts, quantizer, gate_up_menu, down_menu, attn_menu,
        baseline_types, only_wide_attn=args.only_wide_attn, max_layers=args.max_layers,
    )
    if args.max_tensors is not None:
        items = items[: args.max_tensors]
    total_evals = sum(len(it.options) for it in items)
    log.info("sweeping %d tensors (%d tensor-type evaluations) at passes=%d top_k=%d",
             len(items), total_evals, args.passes, args.top_k)

    windows = build_windows(args, tok, parts)
    ref_hiddens = forward_hidden(parts, windows)
    floor_kl = compute_kl_against_ref(parts, windows, ref_hiddens, args.logit_chunk)
    log.info("cached %d windows x %d tokens; identity floor KL = %.3e", args.windows, args.seq_len, floor_kl)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path.with_suffix(".jsonl")

    done: dict[tuple[str, str], dict] = {}
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done[(rec["gguf tensor"], rec["option"])] = rec
        log.info("resumed %d completed evaluations from %s", len(done), jsonl_path)

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    n_run = 0
    for idx, item in enumerate(items):
        if (item.layer, item.attr) in cpu_masters:
            w_f32 = cpu_masters[(item.layer, item.attr)].float().numpy()
        else:
            w_f32 = getattr(item.module, item.attr).detach().float().cpu().numpy()
        im = imatrix.get(item.gguf_name)
        if im is None and imatrix:
            log.warning("imatrix missing entry for %s (falling back to ones if required)", item.gguf_name)
        if im is None and args.dry_run:
            im = (1.0 + 0.1 * rng.random(item.shape[-1])).astype(np.float32)

        for opt in item.options:
            key = (item.gguf_name, opt)
            if key in done:
                continue
            ts = time.time()
            if w_f32.ndim == 3:
                w_deq, n_bytes = quantizer.quant_dequant_3d(w_f32, opt, im)
            else:
                w_deq, n_bytes = quantizer.quant_dequant_2d(w_f32, opt, im)
            dkl = swap_and_measure(parts, item, w_deq, windows, ref_hiddens, args.logit_chunk, args.device, floor_kl)
            dt = time.time() - ts
            n_weights = int(w_f32.size)
            rec = {
                "item": item.gguf_name,
                "gguf tensor": item.gguf_name,
                "kind": item.kind,
                "role": item.role,
                "layer": item.layer,
                "shape": list(item.shape),
                "ne0": int(item.shape[-1]),
                "ne1": int(item.shape[-2]),
                "n_experts": int(item.shape[0]) if w_f32.ndim == 3 else 1,
                "exec_count": 1,
                "option": opt,
                "bits": float(8 * n_bytes),
                "bytes": int(n_bytes),
                "weights": n_weights,
                "bpw": round(8.0 * n_bytes / n_weights, 4),
                "dkl": float(dkl),
                "baseline_type": item.baseline_type,
                "elapsed_s": round(dt, 3),
            }
            done[key] = rec
            with open(jsonl_path, "a") as jf:
                jf.write(json.dumps(rec) + "\n")
            n_run += 1
            log.info("[%d/%d] %-34s %-7s bpw=%.3f dkl=%.6f (%.2fs)",
                     len(done), total_evals, item.gguf_name, opt, rec["bpw"], dkl, dt)

    cfg = parts.text.config
    payload = {
        "meta": {
            "model": "dry-run-tiny-256" if args.dry_run else args.model,
            "revision": None if args.dry_run else args.revision,
            "passes": args.passes,
            "top_k": args.top_k,
            "n_routed_experts": int(cfg.n_routed_experts),
            "n_layers": int(cfg.num_hidden_layers),
            "windows": args.windows,
            "seq_len": args.seq_len,
            "floor_kl": floor_kl,
            "imatrix": args.imatrix,
            "dry_run": args.dry_run,
            "elapsed_s": round(time.time() - t0, 2),
        },
        "records": sorted(done.values(), key=lambda r: (r["layer"], r["gguf tensor"], r["option"])),
    }
    out_path.write_text(json.dumps(payload, indent=1))
    log.info("wrote %d records to %s (ran %d new in %.1fs)", len(done), out_path, n_run, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
