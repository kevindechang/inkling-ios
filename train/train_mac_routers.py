#!/usr/bin/env python3
"""Project #1: OMP-Guided Function-Matched Layer-Wise Router Distillation (FMRD) on Apple M4 Pro GPU ($0).

Keeps ALL 128/128 experts intact (ZERO experts pruned!).
Trains ONLY the 24 routers (blk.L.ffn_gate_inp.weight [128, 2304] + blk.L.exp_probs_b.bias [128], 7.5 MB total!)
layer-by-layer (< 2.5 GB peak unified RAM!) against the EXACT quantized expert weights
(Q2_K / Q4_K / Q2_0 / Q4_0 / IQ4_NL) dequantized directly via libggml-base.dylib.

Algorithm per layer L in 0..23:
  1. Dequantize exact quantized gate_up [128, 2304, 2304], down [128, 2304, 1152], router [128, 2304], bias [128].
  2. Compute exact baseline top-4 MoE output y_teacher(x_i) on N=8192 dumped normalized router inputs x_L
     (7168 train / 1024 contiguous-tail val).
  3. On MPS GPU, evaluate all 40 HOT experts (0..39) on every token x_i in chunks of 256.
  4. For EVERY cold expert visit (c >= 40) in y_teacher(x_i), find the BEST unused HOT expert h* in 0..39
     that minimizes ||alpha * FFN_{h*}(x_i) - w_c * FFN_c(x_i)||^2, recording exact visit_err(i, c).
  5. Keep the top target_cold_L% most irreplaceable cold visits (largest visit_err) intact as COLD,
     and redirect all redundant cold visits to their function-matched HOT substitute h*, solving exact
     positive least-squares weights (sum = 8.0) on the resulting 4-expert set S*(i).
  6. Optimize W_rtr[L] + b_rtr[L] on MPS GPU with smooth top-4 margin ranking loss + weight MSE + L2 anchor
     + gentle cold-bias calibration to match target_cold_L.
  7. Quantize trained W_rtr[L] back to Q8_0 via libggml-base.dylib and verify EXACT quantized held-out
     MoE output MSE, cosine similarity, and cold hit rate vs baseline & blind CECAR=0.25/0.35!

Subcommands:
  train   : Train all 24 layers and save reversible 7.5 MB patch (runs/fmrd/routers_fmrd.pt)
  apply   : Back up original 24 routers+biases once and patch GGUF in-place (0.2s, 0 extra GB disk!)
  restore : Restore exact bit-for-bit original 24 routers+biases into GGUF
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402

SITU_BETA = 4.0
SITU_LB = 25.0
ROUTE_SCALE = 8.0
N_HOT = 40
N_EXPERT = 128
TOP_K = 4
H_DIM = 2304
FF_DIM = 1152

# Edge + full-attention layers carry disproportionate reasoning/output load -> keep more cold headroom!
CRITICAL_LAYERS = {0, 1, 5, 11, 17, 22, 23}


# --------------------------------------------------------------------------- Exact libggml-base dequant / quant

TO_FLOAT_FN = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)


class GGMLTypeTraits(ctypes.Structure):
    _fields_ = [
        ("type_name", ctypes.c_char_p),
        ("blck_size", ctypes.c_int64),
        ("blck_size_interleave", ctypes.c_int64),
        ("type_size", ctypes.c_size_t),
        ("is_quantized", ctypes.c_bool),
        ("to_float", TO_FLOAT_FN),
        ("from_float_ref", ctypes.c_void_p),
    ]


class GGMLCodec:
    """Bit-exact dequantizer + quantizer wrapping llama.cpp/build/bin/libggml-base.dylib."""

    def __init__(self, lib_path: Path):
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
        self.lib.ggml_get_type_traits.restype = ctypes.POINTER(GGMLTypeTraits)
        self.lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self.lib.ggml_row_size.restype = ctypes.c_size_t
        self.lib.ggml_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ggml_quantize_init.restype = None
        self.lib.ggml_quantize_chunk.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
        ]
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self._inited: set[int] = set()

    def dequantize(self, raw_u8: np.ndarray, qtype: int, n_elements: int) -> np.ndarray:
        if qtype == int(gguf.GGMLQuantizationType.F32):
            return np.frombuffer(raw_u8.tobytes(), dtype=np.float32).copy()
        if qtype == int(gguf.GGMLQuantizationType.F16):
            return np.frombuffer(raw_u8.tobytes(), dtype=np.float16).astype(np.float32)
        traits = self.lib.ggml_get_type_traits(int(qtype)).contents
        assert bool(traits.to_float), f"no to_float for qtype={qtype}"
        src = np.ascontiguousarray(raw_u8, dtype=np.uint8)
        dst = np.empty(n_elements, dtype=np.float32)
        traits.to_float(src.ctypes.data, dst.ctypes.data, int(n_elements))
        return dst

    def quantize_2d(self, f32: np.ndarray, qtype: int) -> np.ndarray:
        if qtype == int(gguf.GGMLQuantizationType.F32):
            return np.ascontiguousarray(f32, dtype=np.float32).view(np.uint8)
        if qtype not in self._inited:
            self.lib.ggml_quantize_init(qtype)
            self._inited.add(qtype)
        src = np.ascontiguousarray(f32, dtype=np.float32)
        nrows, ncols = src.shape
        row_sz = int(self.lib.ggml_row_size(qtype, ncols))
        dst = np.empty((nrows, row_sz), dtype=np.uint8)
        got = int(self.lib.ggml_quantize_chunk(qtype, src.ctypes.data, dst.ctypes.data, 0, nrows, ncols, None))
        assert got == dst.nbytes, f"quantized {got} != expected {dst.nbytes}"
        return dst


# --------------------------------------------------------------------------- Exact Inkling-2 SITU + MoE ops

def situ_act(gu: torch.Tensor) -> torch.Tensor:
    """Exact llama-graph.cpp:2613 situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)."""
    gate, up = gu.split(FF_DIM, dim=-1)
    act = (SITU_BETA * torch.tanh(gate * (1.0 / SITU_BETA))) * torch.sigmoid(gate)
    up_c = SITU_LB * torch.tanh(up * (1.0 / SITU_LB))
    return act * up_c


def route_topk(
    x: torch.Tensor,
    w_rtr: torch.Tensor,
    b_rtr: torch.Tensor,
    cecar_bonus: float = 0.0,
    n_hot: int = N_HOT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Inkling-2 sigmoid+bias top-4 routing (with optional CECAR bonus on 0..n_hot-1)."""
    logits = F.linear(x, w_rtr)  # [N, 128]
    probs = torch.sigmoid(logits)
    sel = probs + b_rtr.unsqueeze(0)
    if cecar_bonus != 0.0:
        sel = sel.clone()
        sel[:, :n_hot] += cecar_bonus
    idx = torch.topk(sel, TOP_K, dim=-1).indices  # [N, 4]
    w = probs.gather(1, idx)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20) * ROUTE_SCALE
    return idx, w


@torch.no_grad()
def eval_moe_exact(
    x_cpu: torch.Tensor,
    idx_cpu: torch.Tensor,
    w_cpu: torch.Tensor,
    gu_f16: torch.Tensor,
    dn_f16: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Exact MoE output [N, H] F32 given pre-selected top-4 experts & weights."""
    n = x_cpu.shape[0]
    out = torch.zeros((n, H_DIM), dtype=torch.float32, device=device)
    x_dev = x_cpu.to(device=device, dtype=torch.float16)
    idx_np = idx_cpu.numpy()
    w_dev = w_cpu.to(device=device, dtype=torch.float32)

    # Group tokens by expert id for fast batched matmul on MPS!
    for e in np.unique(idx_np):
        tok_pos, slot_pos = np.nonzero(idx_np == e)
        if len(tok_pos) == 0:
            continue
        tp = torch.from_numpy(tok_pos).to(device)
        sp = torch.from_numpy(slot_pos).to(device)
        xe = x_dev[tp]  # [M, 2304] f16
        w_gu_e = gu_f16[int(e)].to(device)  # [2304, 2304] f16
        w_dn_e = dn_f16[int(e)].to(device)  # [2304, 1152] f16
        gu = F.linear(xe, w_gu_e).float()
        act = situ_act(gu).half()
        ye = F.linear(act, w_dn_e).float()
        out.index_add_(0, tp, ye * w_dev[tp, sp].unsqueeze(-1))
    return out


@torch.no_grad()
def solve_pos_weights_4(Y: torch.Tensor, y_tgt: torch.Tensor) -> torch.Tensor:
    """Solve min_{w >= 0, sum(w) = ROUTE_SCALE} ||sum_k w_k Y_k - y_tgt||^2 for batch [B, 4, H].
    Uses 20 steps of fast projected gradient / multiplicative updates on the 4x4 Gram matrix on MPS!
    """
    # Y: [B, 4, H] f32, y_tgt: [B, H] f32
    G = torch.bmm(Y, Y.transpose(1, 2))  # [B, 4, 4]
    c = torch.bmm(Y, y_tgt.unsqueeze(-1)).squeeze(-1)  # [B, 4]
    # Normalize scale for stable step size
    diag_max = G.diagonal(dim1=1, dim2=2).max(dim=-1, keepdim=True).values.clamp(min=1e-6)  # [B, 1]
    Gn = G / diag_max.unsqueeze(-1)
    cn = c / diag_max
    w = torch.full((Y.shape[0], TOP_K), ROUTE_SCALE / TOP_K, device=Y.device, dtype=torch.float32)
    lr = 0.25
    for _ in range(30):
        grad = torch.bmm(Gn, w.unsqueeze(-1)).squeeze(-1) - cn  # [B, 4]
        # Project onto simplex sum(w) = ROUTE_SCALE, w >= 0.15 (prevent dead slots!)
        w = w - lr * grad
        w = w.clamp(min=0.15)
        w = w * (ROUTE_SCALE / w.sum(dim=-1, keepdim=True))
    return w


# --------------------------------------------------------------------------- Layer-wise FMRD trainer

def train_one_layer(
    il: int,
    reader: gguf.GGUFReader,
    codec: GGMLCodec,
    x_all: torch.Tensor,  # [N, 2304] f32 CPU
    n_hot: int,
    target_cold: float,
    max_visit_err: float,
    steps: int,
    lr: float,
    l2_anchor: float,
    device: torch.device,
) -> dict:
    t0 = time.time()
    tmap = {t.name: t for t in reader.tensors}
    t_gu = tmap[f"blk.{il}.ffn_gate_up_exps.weight"]
    t_dn = tmap[f"blk.{il}.ffn_down_exps.weight"]
    t_rt = tmap[f"blk.{il}.ffn_gate_inp.weight"]
    t_bi = tmap[f"blk.{il}.exp_probs_b.bias"]

    # Exact dequantization of all 128 experts + router for layer L (CPU RAM ~2.0 GB FP16)
    gu_f16 = torch.from_numpy(
        codec.dequantize(np.asarray(t_gu.data), int(t_gu.tensor_type), N_EXPERT * (2 * FF_DIM) * H_DIM)
        .reshape(N_EXPERT, 2 * FF_DIM, H_DIM)
    ).half()
    dn_f16 = torch.from_numpy(
        codec.dequantize(np.asarray(t_dn.data), int(t_dn.tensor_type), N_EXPERT * H_DIM * FF_DIM)
        .reshape(N_EXPERT, H_DIM, FF_DIM)
    ).half()
    w0_f32 = torch.from_numpy(
        codec.dequantize(np.asarray(t_rt.data), int(t_rt.tensor_type), N_EXPERT * H_DIM)
        .reshape(N_EXPERT, H_DIM)
    ).float()
    b0_f32 = torch.from_numpy(np.asarray(t_bi.data, dtype=np.float32).copy()).float()
    assert w0_f32.shape == (N_EXPERT, H_DIM) and b0_f32.shape == (N_EXPERT,)

    n_tot = x_all.shape[0]
    n_val = min(1024, max(256, n_tot // 8))
    n_tr = n_tot - n_val
    x_tr = x_all[:n_tr]
    x_va = x_all[n_tr:]

    # 1. Exact teacher routing & exact quantized MoE outputs
    idx0_tr, w0_tr = route_topk(x_tr, w0_f32, b0_f32, cecar_bonus=0.0)
    idx0_va, w0_va = route_topk(x_va, w0_f32, b0_f32, cecar_bonus=0.0)
    y_tr_tgt = eval_moe_exact(x_tr, idx0_tr, w0_tr, gu_f16, dn_f16, device).cpu()
    y_va_tgt = eval_moe_exact(x_va, idx0_va, w0_va, gu_f16, dn_f16, device).cpu()

    # 2. Preload ONLY the n_hot HOT experts on MPS GPU (~700 MB!) to compute exact functional substitutes!
    gu_hot_gpu = gu_f16[:n_hot].to(device)  # [n_hot, 2304, 2304] f16
    dn_hot_gpu = dn_f16[:n_hot].to(device)  # [n_hot, 2304, 1152] f16

    # For every training token, evaluate all n_hot HOT experts and find exact per-cold-visit best HOT substitute
    best_hot_sub = np.full((n_tr, TOP_K), -1, dtype=np.int32)
    visit_err = np.full((n_tr, TOP_K), -1.0, dtype=np.float32)  # < 0 means already hot!

    idx0_np = idx0_tr.numpy()
    w0_np = w0_tr.numpy()

    chunk = 128
    with torch.no_grad():
        for s0 in range(0, n_tr, chunk):
            s1 = min(n_tr, s0 + chunk)
            bc = s1 - s0
            sub_idx = idx0_np[s0:s1]
            # Skip hot-only rows if none in chunk have cold visits (rare, usually ~50% have >=1 cold)
            cold_mask = sub_idx >= n_hot
            if not cold_mask.any():
                continue
            xc = x_tr[s0:s1].to(device=device, dtype=torch.float16)  # [bc, 2304]
            yt = y_tr_tgt[s0:s1].to(device=device, dtype=torch.float32)  # [bc, 2304]
            yt_norm2 = (yt * yt).sum(dim=-1).clamp(min=1e-6)  # [bc]

            # Compute Y_hot [bc, n_hot, 2304] f16
            Y_hot = torch.empty((bc, n_hot, H_DIM), dtype=torch.float16, device=device)
            for h in range(n_hot):
                gu_h = F.linear(xc, gu_hot_gpu[h]).float()
                act_h = situ_act(gu_h).half()
                Y_hot[:, h, :] = F.linear(act_h, dn_hot_gpu[h])
            Y_hot_f = Y_hot.float()  # [bc, n_hot, 2304]
            hot_n2 = (Y_hot_f * Y_hot_f).sum(dim=-1).clamp(min=1e-6)  # [bc, n_hot]

            # Compute exact cold expert output for every cold visit in this chunk
            cold_rows, cold_slots = np.nonzero(cold_mask)
            cold_exps = sub_idx[cold_rows, cold_slots]
            # Compute v_c = w_c * FFN_c(x_i) grouped by cold expert id
            vc_all = torch.empty((len(cold_rows), H_DIM), dtype=torch.float32, device=device)
            for ce in np.unique(cold_exps):
                m = np.nonzero(cold_exps == ce)[0]
                rows_m = torch.from_numpy(cold_rows[m]).to(device)
                slots_m = cold_slots[m]
                xe = xc[rows_m]
                gu_c = F.linear(xe, gu_f16[int(ce)].to(device)).float()
                act_c = situ_act(gu_c).half()
                yc = F.linear(act_c, dn_f16[int(ce)].to(device)).float()
                wc = torch.from_numpy(w0_np[s0 + cold_rows[m], slots_m]).to(device).unsqueeze(-1)
                vc_all[torch.from_numpy(m).to(device)] = yc * wc

            # For each cold visit (r, sl), score all n_hot hot experts by min_{a>=0} ||a*Y_hot[r, h] - v_c||^2
            cr_t = torch.from_numpy(cold_rows).to(device)
            Yh_sel = Y_hot_f[cr_t]  # [M, n_hot, 2304]
            hn2_sel = hot_n2[cr_t]  # [M, n_hot]
            vc_n2 = (vc_all * vc_all).sum(dim=-1, keepdim=True)  # [M, 1]
            dots = torch.bmm(Yh_sel, vc_all.unsqueeze(-1)).squeeze(-1).clamp(min=0.0)  # [M, n_hot]
            err_mat = (vc_n2 - (dots * dots) / hn2_sel).clamp(min=0.0) / yt_norm2[cr_t].unsqueeze(-1)  # [M, n_hot]

            # Mask out hot experts already in token r's teacher top-4 so we never duplicate an expert!
            err_np = err_mat.cpu().numpy()
            used_hot_per_row: dict[int, set[int]] = {}
            for k_idx in range(len(cold_rows)):
                r = int(cold_rows[k_idx])
                sl = int(cold_slots[k_idx])
                if r not in used_hot_per_row:
                    used_hot_per_row[r] = {int(e) for e in sub_idx[r] if e < n_hot}
                row_err = err_np[k_idx].copy()
                for uh in used_hot_per_row[r]:
                    row_err[uh] = 1e9
                h_best = int(np.argmin(row_err))
                used_hot_per_row[r].add(h_best)
                best_hot_sub[s0 + r, sl] = h_best
                visit_err[s0 + r, sl] = float(row_err[h_best])

    # 3. Lagrange water-filling: replace cold visits with visit_err < err_thresh
    # If max_visit_err > 0, use universal distortion cap (bounded between [0.04, target_cold_max])!
    orig_cold_mask = visit_err >= 0.0
    orig_cold_frac = float(orig_cold_mask.mean())
    all_cold_errs = visit_err[orig_cold_mask]
    if max_visit_err > 0.0:
        err_thresh = max_visit_err
        # Also compute effective target cold rate produced by this distortion cap (clamped to [0.04, 0.35])
        cand_cold = float((orig_cold_mask & (visit_err >= err_thresh)).mean())
        eff_target_cold = min(0.35, max(0.04, cand_cold))
        n_keep_cold = int(round(eff_target_cold * n_tr * TOP_K))
        if len(all_cold_errs) > n_keep_cold and n_keep_cold > 0:
            err_thresh = float(np.partition(all_cold_errs, len(all_cold_errs) - n_keep_cold)[len(all_cold_errs) - n_keep_cold])
    else:
        eff_target_cold = target_cold
        n_keep_cold = int(round(eff_target_cold * n_tr * TOP_K))
        if len(all_cold_errs) > n_keep_cold and n_keep_cold > 0:
            err_thresh = float(np.partition(all_cold_errs, len(all_cold_errs) - n_keep_cold)[len(all_cold_errs) - n_keep_cold])
        elif n_keep_cold <= 0:
            err_thresh = 1e9
        else:
            err_thresh = -1.0

    tgt_idx = idx0_np.copy()
    replaced_mask = orig_cold_mask & (visit_err < err_thresh)
    tgt_idx[replaced_mask] = best_hot_sub[replaced_mask]
    tgt_idx_t = torch.from_numpy(tgt_idx).long()
    tgt_cold_actual = float((tgt_idx >= n_hot).mean())

    # 4. Solve exact optimal positive mixing weights w*(i) for every token whose expert set changed!
    tgt_w_t = w0_tr.clone()
    changed_rows = np.nonzero(replaced_mask.any(axis=1))[0]
    with torch.no_grad():
        for c0 in range(0, len(changed_rows), 256):
            c_ids = changed_rows[c0:c0 + 256]
            bc = len(c_ids)
            xc = x_tr[c_ids].to(device=device, dtype=torch.float16)
            Y4 = torch.empty((bc, TOP_K, H_DIM), dtype=torch.float32, device=device)
            sub_e = tgt_idx[c_ids]
            for e in np.unique(sub_e):
                rr, ss = np.nonzero(sub_e == e)
                rt = torch.from_numpy(rr).to(device)
                st_t = torch.from_numpy(ss).to(device)
                w_gu_e = gu_hot_gpu[int(e)] if e < n_hot else gu_f16[int(e)].to(device)
                w_dn_e = dn_hot_gpu[int(e)] if e < n_hot else dn_f16[int(e)].to(device)
                gu_e = F.linear(xc[rt], w_gu_e).float()
                act_e = situ_act(gu_e).half()
                Y4[rt, st_t, :] = F.linear(act_e, w_dn_e).float()
            yt = y_tr_tgt[c_ids].to(device=device, dtype=torch.float32)
            w_opt = solve_pos_weights_4(Y4, yt)
            tgt_w_t[torch.from_numpy(c_ids)] = w_opt.cpu()

    # Measure oracle upper-bound FFN MSE of our OMP target assignment on train tokens!
    y_oracle = eval_moe_exact(x_tr[:1024], tgt_idx_t[:1024], tgt_w_t[:1024], gu_f16, dn_f16, device).cpu()
    oracle_rel_mse = float(((y_oracle - y_tr_tgt[:1024]).pow(2).sum() / y_tr_tgt[:1024].pow(2).sum().clamp(min=1e-6)).item())

    # Free hot GPU experts before router optimization so MPS stays super fast & clean
    del gu_hot_gpu, dn_hot_gpu

    # 5. Train W_rtr[L] + b_rtr[L] on MPS GPU!
    x_tr_g = x_tr.to(device=device, dtype=torch.float32)
    x_va_g = x_va.to(device=device, dtype=torch.float32)
    tgt_idx_g = tgt_idx_t.to(device=device)
    tgt_w_g = tgt_w_t.to(device=device, dtype=torch.float32)
    w0_g = w0_f32.to(device=device)
    b0_g = b0_f32.to(device=device)

    # Multi-hot mask [n_tr, 128] of target top-4 experts
    pos_mask = torch.zeros((n_tr, N_EXPERT), dtype=torch.bool, device=device)
    pos_mask.scatter_(1, tgt_idx_g, True)
    # Token importance: upweight tokens that had cold-replacement or irreplaceable cold visits!
    tok_has_keep_cold = (tgt_idx_g >= n_hot).any(dim=1)
    tok_was_replaced = torch.from_numpy(replaced_mask.any(axis=1)).to(device)
    tok_wt = torch.ones(n_tr, device=device)
    tok_wt[tok_was_replaced] = 1.5
    tok_wt[tok_has_keep_cold] = 2.5

    W = w0_g.clone().detach().requires_grad_(True)
    b = b0_g.clone().detach().requires_grad_(True)
    opt = torch.optim.AdamW([
        {"params": [W], "lr": lr, "weight_decay": 0.0},
        {"params": [b], "lr": lr * 2.0, "weight_decay": 0.0},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)

    margin = 0.06
    tau = 0.025
    batch_sz = 1024

    for step in range(steps):
        if n_tr > batch_sz:
            perm = torch.randint(0, n_tr, (batch_sz,), device=device)
            xb = x_tr_g[perm]
            tidx = tgt_idx_g[perm]
            tw = tgt_w_g[perm]
            pm = pos_mask[perm]
            wt = tok_wt[perm]
        else:
            xb, tidx, tw, pm, wt = x_tr_g, tgt_idx_g, tgt_w_g, pos_mask, tok_wt

        logits = F.linear(xb, W)  # [B, 128]
        probs = torch.sigmoid(logits)
        sel = probs + b.unsqueeze(0)

        # (a) Hard-negative margin ranking loss: min(pos_sel) > top-k(neg_sel) + margin
        pos_sel = sel.gather(1, tidx)  # [B, 4]
        neg_sel = sel.masked_fill(pm, -1e9)
        top_neg = torch.topk(neg_sel, 8, dim=-1).values  # [B, 8] hardest non-targets!
        diff = top_neg.unsqueeze(1) - pos_sel.unsqueeze(2) + margin
        rank_loss = (F.softplus(diff / tau) * tau).mean(dim=(1, 2))  # [B]
        loss_rank = (rank_loss * wt).mean()

        # (b) Exact weight regression loss on target top-4 experts
        p_pos = probs.gather(1, tidx)
        w_pred = p_pos / (p_pos.sum(dim=-1, keepdim=True) + 1e-20) * ROUTE_SCALE
        loss_w = ((w_pred - tw).pow(2).mean(dim=-1) * wt).mean()

        # (c) L2 anchor to pretrained router (prevents overfitting on unseen eval domains!)
        loss_anchor = (W - w0_g).pow(2).mean() + 0.1 * (b - b0_g).pow(2).mean()

        loss = loss_rank + 0.05 * loss_w + l2_anchor * loss_anchor
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([W, b], 1.0)
        opt.step()
        sched.step()

    # 6. Gentle scalar bias calibration on COLD experts [n_hot:] so actual cold rate hits eff_target_cold cleanly!
    W_tr_f32 = W.detach().cpu().numpy().astype(np.float32)
    W_q8_raw = codec.quantize_2d(W_tr_f32, int(gguf.GGMLQuantizationType.Q8_0))
    W_q8_f32 = torch.from_numpy(
        codec.dequantize(W_q8_raw, int(gguf.GGMLQuantizationType.Q8_0), N_EXPERT * H_DIM)
        .reshape(N_EXPERT, H_DIM)
    ).float()
    b_tr_f32 = b.detach().cpu().float()

    with torch.no_grad():
        logits_q8_tr = F.linear(x_tr, W_q8_f32)
        probs_q8_tr = torch.sigmoid(logits_q8_tr)
        lo, hi = -0.25, 0.25
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            b_try = b_tr_f32.clone()
            b_try[n_hot:] += mid
            idx_try = torch.topk(probs_q8_tr + b_try.unsqueeze(0), TOP_K, dim=-1).indices
            cf = float((idx_try >= n_hot).float().mean().item())
            if cf > eff_target_cold:
                hi = mid
            else:
                lo = mid
        b_tr_f32[n_hot:] += 0.5 * (lo + hi)

    # 7. Exact held-out validation comparing baseline, blind CECAR=0.25, blind CECAR=0.35, FMRD(0.0), FMRD(0.10)!
    def eval_cfg(w_m: torch.Tensor, b_m: torch.Tensor, cecar: float) -> tuple[float, float, float]:
        idx_v, w_v = route_topk(x_va, w_m, b_m, cecar_bonus=cecar, n_hot=n_hot)
        cold_pct = 100.0 * float((idx_v >= n_hot).float().mean().item())
        y_v = eval_moe_exact(x_va, idx_v, w_v, gu_f16, dn_f16, device).cpu()
        rel_mse = float(((y_v - y_va_tgt).pow(2).sum() / y_va_tgt.pow(2).sum().clamp(min=1e-6)).item())
        cos = float(F.cosine_similarity(y_v, y_va_tgt, dim=-1).mean().item())
        return cold_pct, rel_mse, cos

    m_orig0   = eval_cfg(w0_f32, b0_f32, 0.00)
    m_c025    = eval_cfg(w0_f32, b0_f32, 0.25)
    m_c035    = eval_cfg(w0_f32, b0_f32, 0.35)
    m_fmrd0   = eval_cfg(W_q8_f32, b_tr_f32, 0.00)
    m_fmrd010 = eval_cfg(W_q8_f32, b_tr_f32, 0.10)

    dt = time.time() - t0
    print(
        f"L{il:2d} ({'CRIT' if il in CRITICAL_LAYERS else 'SWA '}, tgt={100*eff_target_cold:4.1f}%): "
        f"orig={m_orig0[0]:4.1f}%c | "
        f"C025={m_c025[0]:4.1f}%c mse={m_c025[1]:.4f} cos={m_c025[2]:.4f} | "
        f"FMRD={m_fmrd0[0]:4.1f}%c mse={m_fmrd0[1]:.4f} cos={m_fmrd0[2]:.4f} | "
        f"FMRD+0.10={m_fmrd010[0]:4.1f}%c mse={m_fmrd010[1]:.4f} | "
        f"oracle_mse={oracle_rel_mse:.4f} ({dt:.1f}s)",
        flush=True,
    )

    return {
        "layer": il,
        "target_cold": eff_target_cold,
        "orig_cold_tr": orig_cold_frac,
        "tgt_cold_tr": tgt_cold_actual,
        "oracle_rel_mse": oracle_rel_mse,
        "orig_w_q8": np.ascontiguousarray(np.asarray(t_rt.data)).copy(),
        "orig_b_f32": b0_f32.numpy().copy(),
        "new_w_q8": W_q8_raw.copy(),
        "new_b_f32": b_tr_f32.numpy().astype(np.float32).copy(),
        "metrics": {
            "orig0": m_orig0, "cecar025": m_c025, "cecar035": m_c035,
            "fmrd0": m_fmrd0, "fmrd010": m_fmrd010,
        },
    }


def cmd_train(args: argparse.Namespace) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[fmrd] device={device} model={args.model} data={args.data}")
    meta = json.loads((Path(args.data) / "meta.json").read_text())
    n_layer = int(meta["n_layer"])
    n_fmrd = int(meta["n_fmrd"])
    n_embd = int(meta["n_embd"])
    assert n_embd == H_DIM, f"unexpected n_embd={n_embd}"

    reader = gguf.GGUFReader(str(args.model), "r")
    codec = GGMLCodec(Path(args.lib))

    results = []
    t_all = time.time()
    for il in range(n_layer):
        tgt = args.cold_crit if il in CRITICAL_LAYERS else args.cold_swa
        x_path = Path(args.data) / f"x_L{il:02d}.f16"
        x_np = np.fromfile(x_path, dtype=np.float16).reshape(n_fmrd, H_DIM)
        x_t = torch.from_numpy(x_np.astype(np.float32))
        res = train_one_layer(
            il=il, reader=reader, codec=codec, x_all=x_t,
            n_hot=args.n_hot, target_cold=tgt, max_visit_err=args.max_visit_err,
            steps=args.steps, lr=args.lr, l2_anchor=args.l2_anchor, device=device,
        )
        results.append(res)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(args.model),
        "cold_crit": args.cold_crit,
        "cold_swa": args.cold_swa,
        "steps": args.steps,
        "lr": args.lr,
        "l2_anchor": args.l2_anchor,
        "layers": results,
    }
    torch.save(payload, out_path)

    # Print aggregate summary
    def avg_m(key: str, idx: int) -> float:
        return sum(r["metrics"][key][idx] for r in results) / len(results)

    summary = {
        "elapsed_s": round(time.time() - t_all, 1),
        "orig0":    {"cold_pct": round(avg_m("orig0", 0), 2),    "rel_mse": round(avg_m("orig0", 1), 5),    "cos": round(avg_m("orig0", 2), 5)},
        "cecar025": {"cold_pct": round(avg_m("cecar025", 0), 2), "rel_mse": round(avg_m("cecar025", 1), 5), "cos": round(avg_m("cecar025", 2), 5)},
        "cecar035": {"cold_pct": round(avg_m("cecar035", 0), 2), "rel_mse": round(avg_m("cecar035", 1), 5), "cos": round(avg_m("cecar035", 2), 5)},
        "fmrd0":    {"cold_pct": round(avg_m("fmrd0", 0), 2),    "rel_mse": round(avg_m("fmrd0", 1), 5),    "cos": round(avg_m("fmrd0", 2), 5)},
        "fmrd010":  {"cold_pct": round(avg_m("fmrd010", 0), 2),  "rel_mse": round(avg_m("fmrd010", 1), 5),  "cos": round(avg_m("fmrd010", 2), 5)},
        "oracle_rel_mse": round(sum(r["oracle_rel_mse"] for r in results) / len(results), 5),
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== FMRD 24-LAYER HELD-OUT SUMMARY ===")
    for k in ("orig0", "cecar025", "cecar035", "fmrd0", "fmrd010"):
        s = summary[k]
        print(f"  {k:10s}: cold={s['cold_pct']:5.2f}% ({0.96*s['cold_pct']:4.2f} visits/tok) | rel_mse={s['rel_mse']:.5f} | cos={s['cos']:.5f}")
    print(f"  oracle_mse: {summary['oracle_rel_mse']:.5f} | elapsed={summary['elapsed_s']}s -> saved {out_path}")


def _patch_gguf(model_path: Path, patch_path: Path, restore: bool) -> None:
    payload = torch.load(patch_path, map_location="cpu", weights_only=False)
    reader = gguf.GGUFReader(str(model_path), "r")
    tmap = {t.name: t for t in reader.tensors}
    # gguf-py t.data_offset is ALREADY the absolute file byte offset! Verify on blk.0 before touching disk.
    t0_chk = tmap["blk.0.ffn_gate_inp.weight"]
    off0 = int(t0_chk.data_offset)
    expect0 = np.ascontiguousarray(np.asarray(t0_chk.data)).tobytes()[:64]
    with model_path.open("rb") as f:
        f.seek(off0)
        got0 = f.read(64)
    assert got0 == expect0, f"GGUF absolute offset sanity check failed at {off0}!"
    offsets = {k: (int(t.data_offset), int(t.n_bytes)) for k, t in tmap.items()}
    del reader

    # On first apply, save immutable standalone backup next to patch if not present
    bak_path = patch_path.with_name("routers_orig_backup.pt")
    if not restore and not bak_path.exists():
        orig_bak = {}
        with model_path.open("rb") as f:
            for r in payload["layers"]:
                il = int(r["layer"])
                for key in (f"blk.{il}.ffn_gate_inp.weight", f"blk.{il}.exp_probs_b.bias"):
                    off, nb = offsets[key]
                    f.seek(off)
                    orig_bak[key] = f.read(nb)
        torch.save(orig_bak, bak_path)
        print(f"[patch] saved immutable original router backup -> {bak_path} ({len(orig_bak)} tensors)")

    if restore:
        src_bak = torch.load(bak_path, map_location="cpu", weights_only=False)
        with model_path.open("r+b") as f:
            for key, raw in src_bak.items():
                off, nb = offsets[key]
                assert len(raw) == nb
                f.seek(off)
                f.write(raw)
        print(f"[patch] RESTORED {len(src_bak)} original router/bias tensors into {model_path.name}!")
        return

    with model_path.open("r+b") as f:
        n_patched = 0
        for r in payload["layers"]:
            il = int(r["layer"])
            off_w, nb_w = offsets[f"blk.{il}.ffn_gate_inp.weight"]
            off_b, nb_b = offsets[f"blk.{il}.exp_probs_b.bias"]
            w_bytes = np.ascontiguousarray(r["new_w_q8"], dtype=np.uint8).tobytes()
            b_bytes = np.ascontiguousarray(r["new_b_f32"], dtype=np.float32).tobytes()
            assert len(w_bytes) == nb_w, f"L{il} W size {len(w_bytes)} != {nb_w}"
            assert len(b_bytes) == nb_b, f"L{il} b size {len(b_bytes)} != {nb_b}"
            f.seek(off_w)
            f.write(w_bytes)
            f.seek(off_b)
            f.write(b_bytes)
            n_patched += 2
    print(f"[patch] APPLIED {n_patched} FMRD router/bias tensors in-place to {model_path.name}!")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--model", default="models/varloop-stage2-v1-run3-reap40-i8mm.gguf")
    p_tr.add_argument("--lib", default=str(ROOT / "llama.cpp/build/bin/libggml-base.dylib"))
    p_tr.add_argument("--data", default="/tmp/fmrd_data")
    p_tr.add_argument("--out", default=str(ROOT / "runs/fmrd/routers_fmrd_v1.pt"))
    p_tr.add_argument("--n-hot", type=int, default=40, help="Number of hot experts 0..n_hot-1")
    p_tr.add_argument("--cold-crit", type=float, default=0.055, help="Target cold frac for 7 critical layers (L0,1,5,11,17,22,23)")
    p_tr.add_argument("--cold-swa", type=float, default=0.025, help="Target cold frac for 17 interior SWA layers")
    p_tr.add_argument("--max-visit-err", type=float, default=0.0, help="If >0, Lagrange distortion cap per cold visit (overrides fixed cold targets)")
    p_tr.add_argument("--steps", type=int, default=450)
    p_tr.add_argument("--lr", type=float, default=2.0e-3)
    p_tr.add_argument("--l2-anchor", type=float, default=2.0e-2)

    p_ap = sub.add_parser("apply")
    p_ap.add_argument("--model", default="models/varloop-stage2-v1-run3-reap40-i8mm.gguf")
    p_ap.add_argument("--patch", default=str(ROOT / "runs/fmrd/routers_fmrd_v1.pt"))

    p_rs = sub.add_parser("restore")
    p_rs.add_argument("--model", default="models/varloop-stage2-v1-run3-reap40-i8mm.gguf")
    p_rs.add_argument("--patch", default=str(ROOT / "runs/fmrd/routers_fmrd_v1.pt"))

    args = ap.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "apply":
        _patch_gguf(Path(args.model), Path(args.patch), restore=False)
    elif args.cmd == "restore":
        _patch_gguf(Path(args.model), Path(args.patch), restore=True)


if __name__ == "__main__":
    main()
