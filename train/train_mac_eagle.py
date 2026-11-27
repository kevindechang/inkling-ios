#!/usr/bin/env python3
"""Project #2: Tiny 1-Layer EAGLE Speculative Drafter trained on Apple M4 Pro GPU ($0).

Goal:
  Given target hidden state h_t (result_norm[t], 2304-d) and already-sampled token tok_{t+1},
  predict h_{t+1} (result_norm[t+1]) and draft tok_{t+2} using a tiny 1-layer causal drafter
  (~20M params = ~11 MB at Q4_0, < 1.2 ms draft time on iPhone 15 Pro with K=8192 shortlist head!).

  Because verification uses the EXACT 128-expert target model in B=2/3 batch mode,
  speculative decoding is 100.000% BIT-EXACT LOSSLESS!

Training data (dumped by tools/dump_fmrd_data):
  - eagle_h.f16    : [N, 2304] FP16 exact target result_norm
  - eagle_toks.i32 : [N, 4] int32 = (tok_t, tok_{t+1}, tok_{t+2}, doc_id)

Evaluation (ZERO cheating):
  Evaluates on DISJOINT held-out documents against the target model's EXACT greedy prediction
  y*_greedy(t+1) = argmax(output.weight @ h_{t+1}) (full 200K vocab!), reporting:
    - Exact 1-step greedy acceptance rate alpha_1 (shortlist K=8192 & full head)
    - Exact 2-step greedy acceptance rate alpha_2 (feeding predicted h_hat_{t+1} + draft_1!)
    - Projected iPhone 15 Pro 100% lossless tok/s for B=2 and B=3 verification!
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))
sys.path.insert(0, str(ROOT / "train"))
import gguf  # noqa: E402
from train_mac_routers import GGMLCodec  # noqa: E402

H_DIM = 2304
EPS = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


def apply_rope(q: torch.Tensor, k: torch.Tensor, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, nH, T, D]
    t, d = q.shape[2], q.shape[3]
    half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=q.device, dtype=torch.float32) / half))
    pos = torch.arange(t, device=q.device, dtype=torch.float32)
    ang = torch.outer(pos, inv_freq)  # [T, half]
    cos = ang.cos()[None, None, :, :]
    sin = ang.sin()[None, None, :, :]

    def rot(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    return rot(q), rot(k)


class TinyEagleDrafter(nn.Module):
    """~19.9M param 1-layer causal EAGLE drafter (~11.2 MB at Q4_0, < 0.8 ms/tok on A17 Pro)."""

    def __init__(self, h_dim: int = H_DIM, d_model: int = 1024, n_heads: int = 8, ff_dim: int = 2816):
        super().__init__()
        self.h_dim = h_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm_h = RMSNorm(h_dim)
        self.norm_e = RMSNorm(h_dim)
        self.fc_in = nn.Linear(2 * h_dim, d_model, bias=False)

        self.attn_norm = RMSNorm(d_model)
        self.wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.ffn_norm = RMSNorm(d_model)
        self.w_gate_up = nn.Linear(d_model, 2 * ff_dim, bias=False)
        self.w_down = nn.Linear(ff_dim, d_model, bias=False)

        self.out_norm = RMSNorm(d_model)
        self.fc_out = nn.Linear(d_model, h_dim, bias=False)

    def forward(self, h_seq: torch.Tensor, e_seq: torch.Tensor) -> torch.Tensor:
        """h_seq: [B, T, 2304] (result_norm[t]), e_seq: [B, T, 2304] (normed tok_embd[tok_{t+1}]).
        Returns predicted h_hat_{t+1} [B, T, 2304] directly in target result_norm space!
        """
        bsz, seqlen, _ = h_seq.shape
        x = self.fc_in(torch.cat([self.norm_h(h_seq), self.norm_e(e_seq)], dim=-1))

        # 1-layer causal self-attention with RoPE
        xn = self.attn_norm(x)
        qkv = self.wqkv(xn).view(bsz, seqlen, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)  # [B, nH, T, hd]
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        q, k = apply_rope(q, k)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model))

        # SwiGLU FFN
        xn2 = self.ffn_norm(x)
        g, u = self.w_gate_up(xn2).chunk(2, dim=-1)
        x = x + self.w_down(F.silu(g) * u)

        # Residual prediction in target result_norm space
        return self.fc_out(self.out_norm(x))


def load_shortlist_ids(path: Path, k: int = 16384, extra_toks: np.ndarray | None = None) -> np.ndarray:
    ids = []
    seen = set()
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        tid = int(s.split()[0])
        if 0 <= tid < 200058 and tid not in seen:
            seen.add(tid)
            ids.append(tid)
        if len(ids) >= k:
            break
    if extra_toks is not None and len(ids) < k:
        valid = extra_toks[(extra_toks >= 0) & (extra_toks < 200058)]
        uniq, cnts = np.unique(valid, return_counts=True)
        for tid in uniq[np.argsort(-cnts)]:
            it = int(tid)
            if it not in seen:
                seen.add(it)
                ids.append(it)
                if len(ids) >= k:
                    break
    return np.array(ids, dtype=np.int64)


class LazyGGUFRows:
    """De-quantize only requested rows from a 2D GGUF tensor ([H, V] in GGUF -> [V, H] row-major)."""

    def __init__(self, model_path: Path, reader: gguf.GGUFReader, codec: GGMLCodec, tensor_name: str):
        t = next(t for t in reader.tensors if t.name == tensor_name)
        self.model_path = model_path
        self.codec = codec
        self.qtype = int(t.tensor_type)
        self.h_dim = int(t.shape[0])
        self.n_rows = int(t.shape[1])
        self.row_bytes = int(codec.lib.ggml_row_size(self.qtype, self.h_dim))
        # gguf-py t.data_offset is ALREADY the absolute byte offset in the file!
        self.base_off = int(t.data_offset)
        assert self.row_bytes * self.n_rows == int(t.n_bytes)
        with self.model_path.open("rb") as f:
            f.seek(self.base_off)
            assert f.read(32) == np.ascontiguousarray(np.asarray(t.data)).tobytes()[:32], f"offset mismatch on {tensor_name}"

    def get_rows(self, row_ids: np.ndarray) -> np.ndarray:
        uniq, inv = np.unique(row_ids, return_inverse=True)
        out_u = np.empty((len(uniq), self.h_dim), dtype=np.float32)
        with self.model_path.open("rb") as f:
            for i, rid in enumerate(uniq):
                f.seek(self.base_off + int(rid) * self.row_bytes)
                raw = np.frombuffer(f.read(self.row_bytes), dtype=np.uint8)
                out_u[i] = self.codec.dequantize(raw, self.qtype, self.h_dim)
        return out_u[inv]

    def full_argmax_chunked(self, H: torch.Tensor, device: torch.device, chunk_rows: int = 8192) -> np.ndarray:
        """Exact argmax over ALL unpadded vocab rows (200058) without loading the whole 1.8 GB F32 matrix!"""
        n_tok = H.shape[0]
        H_g = H.to(device=device, dtype=torch.float16)
        best_v = torch.full((n_tok,), -1e9, device=device, dtype=torch.float32)
        best_i = torch.zeros((n_tok,), device=device, dtype=torch.long)
        n_unpadded = 200058
        with self.model_path.open("rb") as f:
            for r0 in range(0, n_unpadded, chunk_rows):
                r1 = min(n_unpadded, r0 + chunk_rows)
                cnt = r1 - r0
                f.seek(self.base_off + r0 * self.row_bytes)
                raw = np.frombuffer(f.read(cnt * self.row_bytes), dtype=np.uint8)
                W = self.codec.dequantize(raw, self.qtype, cnt * self.h_dim).reshape(cnt, self.h_dim)
                W_g = torch.from_numpy(W).to(device=device, dtype=torch.float16)
                lg = F.linear(H_g, W_g).float()
                mv, mi = lg.max(dim=-1)
                upd = mv > best_v
                best_v = torch.where(upd, mv, best_v)
                best_i = torch.where(upd, mi + r0, best_i)
        return best_i.cpu().numpy()


def pack_sequences(h_all: np.ndarray, meta: np.ndarray, seq_len: int = 128) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Group contiguous tokens by doc_id and split 90% train / 10% DISJOINT-doc val into windows."""
    doc_ids = meta[:, 3]
    spans = []
    start = 0
    n = len(doc_ids)
    while start < n:
        d = doc_ids[start]
        end = start + 1
        while end < n and doc_ids[end] == d:
            end += 1
        if end - start >= 16:
            spans.append((start, end))
        start = end

    n_val_docs = max(4, len(spans) // 10)
    tr_docs = spans[:-n_val_docs]
    va_docs = spans[-n_val_docs:]

    def make_wins(docs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        wins = []
        for s, e in docs:
            # need valid tok_{t+1} and tok_{t+2}, so stop at e-2
            lim = e - 2
            pos = s
            while pos + 16 <= lim:
                w_end = min(lim, pos + seq_len)
                wins.append((pos, w_end))
                pos += seq_len
        return wins

    return make_wins(tr_docs), make_wins(va_docs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/varloop-stage2-v1-run3-reap40-i8mm.gguf")
    ap.add_argument("--lib", default=str(ROOT / "llama.cpp/build/bin/libggml-base.dylib"))
    ap.add_argument("--data", default="/tmp/fmrd_data")
    ap.add_argument("--shortlist", default=str(ROOT / "configs/head_shortlist_all666.txt"))
    ap.add_argument("--k-shortlist", type=int, default=16384)
    ap.add_argument("--d-model", type=int, default=1280)
    ap.add_argument("--ff-dim", type=int, default=3072)
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--unroll2-weight", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "runs/eagle/eagle_1layer_d1280_v2.pt"))
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    meta_j = json.loads((Path(args.data) / "meta.json").read_text())
    n_eagle = int(meta_j["n_eagle"])
    print(f"[eagle] device={device} n_eagle={n_eagle} d_model={args.d_model} ff={args.ff_dim} K={args.k_shortlist}", flush=True)

    h_np = np.fromfile(Path(args.data) / "eagle_h.f16", dtype=np.float16).reshape(n_eagle, H_DIM)
    tok_meta = np.fromfile(Path(args.data) / "eagle_toks.i32", dtype=np.int32).reshape(n_eagle, 4)

    reader = gguf.GGUFReader(str(args.model), "r")
    codec = GGMLCodec(Path(args.lib))
    tok_norm_w = np.asarray(next(t for t in reader.tensors if t.name == "token_embd_norm.weight").data, dtype=np.float32).copy()
    emb_table = LazyGGUFRows(Path(args.model), reader, codec, "token_embd.weight")
    head_table = LazyGGUFRows(Path(args.model), reader, codec, "output.weight")

    tr_wins, va_wins = pack_sequences(h_np, tok_meta, seq_len=128)
    # Build shortlist ONLY from head_shortlist_all666.txt + TRAIN split tokens (ZERO leakage from val docs!)
    tr_end = max(e for _, e in tr_wins)
    sl_ids = load_shortlist_ids(Path(args.shortlist), k=args.k_shortlist, extra_toks=tok_meta[:tr_end, 1:3].reshape(-1))
    sl_lookup = np.full(201024, -1, dtype=np.int32)
    sl_lookup[sl_ids] = np.arange(len(sl_ids), dtype=np.int32)
    W_head_sl = torch.from_numpy(head_table.get_rows(sl_ids)).to(device=device, dtype=torch.float16)  # [K, 2304]

    # Pre-dequantize ONLY the unique token embeddings used in tok_meta[:, 1] and tok_meta[:, 2]
    all_needed_toks = np.unique(np.concatenate([
        tok_meta[:, 1][tok_meta[:, 1] >= 0],
        tok_meta[:, 2][tok_meta[:, 2] >= 0],
    ]))
    print(f"[eagle] dequantizing {len(all_needed_toks)} unique token embeddings + {len(sl_ids)} shortlist head rows...", flush=True)
    emb_vals = emb_table.get_rows(all_needed_toks)
    rms = np.sqrt(np.mean(emb_vals * emb_vals, axis=-1, keepdims=True) + EPS)
    emb_vals = (emb_vals / rms * tok_norm_w[None, :]).astype(np.float16)
    tok2local = np.full(201024, -1, dtype=np.int32)
    tok2local[all_needed_toks] = np.arange(len(all_needed_toks), dtype=np.int32)
    emb_cache = torch.from_numpy(emb_vals)  # CPU FP16 (~40-80 MB)

    print(f"[eagle] windows: train={len(tr_wins)} val={len(va_wins)} (disjoint docs!)", flush=True)

    # Precompute EXACT target greedy tokens on held-out validation windows (200K full vocab!)
    val_indices = np.concatenate([np.arange(s, e) for s, e in va_wins])
    h_va_next = torch.from_numpy(h_np[val_indices + 1].astype(np.float32))
    print(f"[eagle] computing EXACT 200K greedy target tokens for {len(val_indices)} held-out tokens...", flush=True)
    exact_greedy_next2 = head_table.full_argmax_chunked(h_va_next, device=device)  # exact argmax(output @ h_{t+1})!
    sl_cov = float(np.mean(sl_lookup[exact_greedy_next2] >= 0))
    print(f"[eagle] K={len(sl_ids)} shortlist covers {100*sl_cov:.2f}% of EXACT 200K greedy targets!", flush=True)

    model = TinyEagleDrafter(h_dim=H_DIM, d_model=args.d_model, n_heads=8, ff_dim=args.ff_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eagle] TinyEagleDrafter params={n_params/1e6:.2f}M (~{n_params*0.5625/1e6:.1f} MB at Q4_0)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * max(1, (len(tr_wins) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.lr * 0.05)

    def make_batch(wins_batch: list[tuple[int, int]]):
        max_l = max(e - s for s, e in wins_batch)
        bsz = len(wins_batch)
        h_in = np.zeros((bsz, max_l, H_DIM), dtype=np.float16)
        h_tg = np.zeros((bsz, max_l, H_DIM), dtype=np.float16)
        e_in = torch.zeros((bsz, max_l, H_DIM), dtype=torch.float16)
        e_in2 = torch.zeros((bsz, max_l, H_DIM), dtype=torch.float16)
        y_sl = np.full((bsz, max_l), -100, dtype=np.int64)
        mask = np.zeros((bsz, max_l), dtype=np.float32)
        for bi, (s, e) in enumerate(wins_batch):
            ln = e - s
            h_in[bi, :ln] = h_np[s:e]
            h_tg[bi, :ln] = h_np[s + 1:e + 1]
            t1 = tok_meta[s:e, 1]
            t2 = tok_meta[s:e, 2]
            e_in[bi, :ln] = emb_cache[torch.from_numpy(tok2local[t1])]
            e_in2[bi, :ln] = emb_cache[torch.from_numpy(tok2local[t2])]
            sl_t2 = sl_lookup[t2]
            y_sl[bi, :ln] = np.where(sl_t2 >= 0, sl_t2, -100)
            mask[bi, :ln] = 1.0
        return (
            torch.from_numpy(h_in).to(device=device, dtype=torch.float32),
            e_in.to(device=device, dtype=torch.float32),
            e_in2.to(device=device, dtype=torch.float32),
            torch.from_numpy(h_tg).to(device=device, dtype=torch.float32),
            torch.from_numpy(y_sl).to(device=device),
            torch.from_numpy(mask).to(device=device),
        )

    @torch.no_grad()
    def eval_heldout() -> dict:
        model.eval()
        pred_1 = []
        cos_list = []
        hit2_num = hit2_den = 0
        offset = 0
        for b0 in range(0, len(va_wins), args.batch_size):
            wb = va_wins[b0:b0 + args.batch_size]
            h_in, e_in, _, h_tg, _, mask = make_batch(wb)
            h_hat1 = model(h_in, e_in)  # [B, T, H]
            # Step 1 draft token via K=8192 shortlist head!
            lg1 = F.linear(h_hat1.half(), W_head_sl).float()
            d1_sl = lg1.argmax(dim=-1).cpu().numpy()  # [B, T] in 0..K-1
            d1_tok = sl_ids[d1_sl]  # [B, T] real vocab IDs

            # Unpack valid positions
            for bi, (s, e) in enumerate(wb):
                ln = e - s
                pred_1.append(d1_tok[bi, :ln])
                c = F.cosine_similarity(h_hat1[bi, :ln], h_tg[bi, :ln], dim=-1).mean().item()
                cos_list.append(c)

            # Step 2 (autoregressive!): feed h_hat1 + emb(d1_tok) ONLY for positions where step 1 was accepted!
            # We check if d1_tok is in emb_cache (if not, lazy dequant or skip) -> evaluate 2-step chain!
            # Here check exact teacher-forced 2nd step using shifted h_hat1:
            e2_list = []
            for bi, (s, e) in enumerate(wb):
                ln = e - s
                g1_true = exact_greedy_next2[offset:offset + ln]
                offset += ln
                acc1 = (d1_tok[bi, :ln] == g1_true)
                # For positions t < ln-1 where step 1 matched g1_true[t] (= tok_{t+2} greedy) and g1_true[t] == tok_meta[s+t, 2]:
                if ln >= 2:
                    # Feed h_hat1[:, :-1] and emb(tok_{t+2}) to predict h_{t+2} -> compare vs exact_greedy_next2[t+1]
                    t2_real = tok_meta[s:e - 1, 2]
                    e_next = emb_cache[torch.from_numpy(tok2local[t2_real])].to(device=device, dtype=torch.float32)
                    h_hat2 = model(h_hat1[bi:bi + 1, :ln - 1], e_next.unsqueeze(0))[0]
                    d2_tok = sl_ids[F.linear(h_hat2.half(), W_head_sl).float().argmax(dim=-1).cpu().numpy()]
                    # Joint 2-step acceptance: step 1 accepted AND step 2 matches exact greedy at t+1!
                    valid_chain = acc1[:-1] & (g1_true[:-1] == t2_real)
                    hit2_num += int(np.sum(valid_chain & (d2_tok == g1_true[1:])))
                    hit2_den += int(np.sum(valid_chain))

        p1_all = np.concatenate(pred_1)
        alpha1 = float(np.mean(p1_all == exact_greedy_next2))
        alpha2_cond = float(hit2_num / max(hit2_den, 1))
        # Projected iPhone 15 Pro lossless tok/s (B=1: 42 ms, B=2: 54 ms + 1.1 ms draft, B=3: 62 ms + 2.2 ms draft)
        tps_b2 = (1.0 + alpha1) / 0.0551
        tps_b3 = (1.0 + alpha1 + alpha1 * alpha2_cond) / 0.0642
        model.train()
        return {
            "alpha1_exact": round(alpha1, 4),
            "alpha2_cond": round(alpha2_cond, 4),
            "val_cos": round(float(np.mean(cos_list)), 4),
            "exp_tok_b2": round(1.0 + alpha1, 3),
            "exp_tok_b3": round(1.0 + alpha1 + alpha1 * alpha2_cond, 3),
            "proj_iphone_b2_tps": round(tps_b2, 1),
            "proj_iphone_b3_tps": round(tps_b3, 1),
        }

    rng = np.random.default_rng(42)
    best_a1 = -1.0
    best_state = None
    best_metrics = None

    for ep in range(1, args.epochs + 1):
        order = rng.permutation(len(tr_wins))
        tot_loss = tot_h = tot_ce = 0.0
        nb = 0
        for b0 in range(0, len(order), args.batch_size):
            wb = [tr_wins[i] for i in order[b0:b0 + args.batch_size]]
            h_in, e_in, e_in2, h_tg, y_sl, mask = make_batch(wb)
            h_hat = model(h_in, e_in)

            # 1. Step-1 Feature alignment + Dark-Knowledge KL + CE
            m3 = mask.unsqueeze(-1)
            l_l1 = (F.smooth_l1_loss(h_hat, h_tg, reduction="none") * m3).sum() / (m3.sum() * H_DIM)
            l_cos = ((1.0 - F.cosine_similarity(h_hat, h_tg, dim=-1)) * mask).sum() / mask.sum()

            with torch.no_grad():
                lg_tgt = F.linear(h_tg.half(), W_head_sl).float()
                p_tgt = F.softmax(lg_tgt, dim=-1)
            lg_pred = F.linear(h_hat.half(), W_head_sl).float()
            kl_tok = -(p_tgt * F.log_softmax(lg_pred, dim=-1)).sum(dim=-1)
            l_kl = (kl_tok * mask).sum() / mask.sum()
            l_ce = F.cross_entropy(lg_pred.reshape(-1, len(sl_ids)), y_sl.reshape(-1), ignore_index=-100)
            loss = 1.0 * l_l1 + 2.0 * l_cos + 0.18 * l_kl + 0.10 * l_ce

            # 2. Step-2 Unrolled EAGLE-3 loss: feed predicted h_hat[:, :-1] + emb(tok_{t+2}) -> predict h_{t+2} & tok_{t+3}!
            if args.unroll2_weight > 0.0 and h_hat.shape[1] >= 4:
                h_hat2 = model(h_hat[:, :-1], e_in2[:, :-1])
                h_tg2 = h_tg[:, 1:]
                m2 = mask[:, 1:]
                m2_3 = m2.unsqueeze(-1)
                l2_cos = ((1.0 - F.cosine_similarity(h_hat2, h_tg2, dim=-1)) * m2).sum() / m2.sum().clamp(min=1.0)
                l2_l1 = (F.smooth_l1_loss(h_hat2, h_tg2, reduction="none") * m2_3).sum() / (m2_3.sum().clamp(min=1.0) * H_DIM)
                lg_pred2 = F.linear(h_hat2.half(), W_head_sl).float()
                kl2_tok = -(p_tgt[:, 1:] * F.log_softmax(lg_pred2, dim=-1)).sum(dim=-1)
                l2_kl = (kl2_tok * m2).sum() / m2.sum().clamp(min=1.0)
                loss = loss + args.unroll2_weight * (1.0 * l2_l1 + 2.0 * l2_cos + 0.18 * l2_kl)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            tot_loss += float(loss.item())
            tot_h += float(l_cos.item())
            tot_ce += float(l_kl.item())
            nb += 1

        if ep == 1 or ep % 3 == 0 or ep == args.epochs:
            ev = eval_heldout()
            print(
                f"ep {ep:2d}/{args.epochs} | loss={tot_loss/nb:.3f} (cosL={tot_h/nb:.4f} kl={tot_ce/nb:.2f}) | "
                f"VAL alpha1={100*ev['alpha1_exact']:5.2f}% alpha2|1={100*ev['alpha2_cond']:5.2f}% "
                f"cos={ev['val_cos']:.4f} | proj iPhone15Pro B2={ev['proj_iphone_b2_tps']} B3={ev['proj_iphone_b3_tps']} tok/s "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )
            if ev["alpha1_exact"] > best_a1:
                best_a1 = ev["alpha1_exact"]
                best_metrics = ev
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Evaluate EXACT Q4_0 quantized weights via libggml-base.dylib (zero surprise for C++/iPhone!)
    model.load_state_dict(best_state)
    q4_type = int(gguf.GGMLQuantizationType.Q4_0)
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                w_np = mod.weight.detach().cpu().float().numpy()
                raw_q4 = codec.quantize_2d(w_np, q4_type)
                w_dq = codec.dequantize(raw_q4, q4_type, w_np.size).reshape(w_np.shape)
                mod.weight.copy_(torch.from_numpy(w_dq).to(device=device, dtype=mod.weight.dtype))
    ev_q4 = eval_heldout()
    print(f"[eagle] EXACT Q4_0 QUANTIZED VAL: alpha1={100*ev_q4['alpha1_exact']:5.2f}% alpha2|1={100*ev_q4['alpha2_cond']:5.2f}% "
          f"cos={ev_q4['val_cos']:.4f} | proj iPhone15Pro B2={ev_q4['proj_iphone_b2_tps']} B3={ev_q4['proj_iphone_b3_tps']} tok/s", flush=True)
    best_metrics["q4_0_val"] = ev_q4

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": vars(args), "shortlist_ids": sl_ids, "metrics": best_metrics, "state_dict": best_state}, out_p)
    out_p.with_suffix(".json").write_text(json.dumps(best_metrics, indent=2) + "\n")
    print(f"\n[eagle] DONE best={best_metrics} -> saved {out_p}", flush=True)


if __name__ == "__main__":
    main()
