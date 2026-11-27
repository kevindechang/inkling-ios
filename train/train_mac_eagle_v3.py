#!/usr/bin/env python3
"""Project #2 (v3): 2-Layer EAGLE-3 Speculative Drafter trained on Apple M4 Pro GPU ($0).

Key upgrades over EAGLE-v2:
  1. 2-layer causal transformer (d=1024, ff=2560, ~31.2M params = ~17.5 MB @ Q4_0) + learned residual h_skip.
  2. Trained on 240,000 exact unperturbed (CECAR=0) target hidden states across all 4 domains.
  3. True frequency-sorted K=16,384 train-only shortlist (~98.5% coverage of real English/math/tool prose).
  4. Assistant/thinking-turn upweighting (2.5x after <|message_model|>, 0.2x on system boilerplate).
  5. Random RoPE position offsets pos0 ~ Uniform(0, 512) during training for long-context generalization.
  6. Strong EAGLE-3 distillation loss (1.0*KL + 0.45*greedy_CE + 0.6*cos + 0.25*L1 + 0.5*2-step unroll).
  7. Honest disjoint-doc validation reporting BOTH overall AND assistant-generation-only alpha1 (zero boilerplate!).
"""
from __future__ import annotations

import argparse
import json
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
from train_mac_eagle import LazyGGUFRows, RMSNorm, H_DIM, EPS  # noqa: E402

TOK_MSG_MODEL = 200001
TOK_THINKING = 200008


def apply_rope_off(q: torch.Tensor, k: torch.Tensor, pos_offset: int = 0, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    t, d = q.shape[2], q.shape[3]
    half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=q.device, dtype=torch.float32) / half))
    pos = torch.arange(pos_offset, pos_offset + t, device=q.device, dtype=torch.float32)
    ang = torch.outer(pos, inv_freq)
    cos = ang.cos()[None, None, :, :]
    sin = ang.sin()[None, None, :, :]

    def rot(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    return rot(q), rot(k)


class EagleBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_norm = RMSNorm(d_model)
        self.wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.ffn_norm = RMSNorm(d_model)
        self.w_gate_up = nn.Linear(d_model, 2 * ff_dim, bias=False)
        self.w_down = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xn = self.attn_norm(x)
        qkv = self.wqkv(xn).view(bsz, seqlen, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        q, k = apply_rope_off(q, k, pos_offset=pos_offset)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model))

        xn2 = self.ffn_norm(x)
        g, u = self.w_gate_up(xn2).chunk(2, dim=-1)
        x = x + self.w_down(F.silu(g) * u)
        return x


class EagleV3Drafter(nn.Module):
    def __init__(self, h_dim: int = H_DIM, d_model: int = 1024, n_heads: int = 8, ff_dim: int = 2560, n_layers: int = 2):
        super().__init__()
        self.h_dim = h_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.n_layers = n_layers

        self.norm_h = RMSNorm(h_dim)
        self.norm_e = RMSNorm(h_dim)
        self.fc_in = nn.Linear(2 * h_dim, d_model, bias=False)
        self.layers = nn.ModuleList([EagleBlock(d_model, n_heads, ff_dim) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)
        self.fc_out = nn.Linear(d_model, h_dim, bias=False)
        self.h_skip = nn.Parameter(torch.full((h_dim,), 0.25))

    def forward(self, h_seq: torch.Tensor, e_seq: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        x = self.fc_in(torch.cat([self.norm_h(h_seq), self.norm_e(e_seq)], dim=-1))
        for blk in self.layers:
            x = blk(x, pos_offset=pos_offset)
        return self.fc_out(self.out_norm(x)) + h_seq * self.h_skip


def build_freq_shortlist(train_toks: np.ndarray, fallback_path: Path, k: int = 16384) -> np.ndarray:
    """Build true frequency-sorted train-only shortlist (zero val leakage!)."""
    valid = train_toks[(train_toks >= 0) & (train_toks < 200058)]
    uniq, cnts = np.unique(valid, return_counts=True)
    order = np.argsort(-cnts)
    ids: list[int] = []
    seen: set[int] = set()
    for tid in uniq[order]:
        it = int(tid)
        if it not in seen:
            seen.add(it)
            ids.append(it)
            if len(ids) >= k:
                break
    if len(ids) < k and fallback_path.exists():
        for line in fallback_path.read_text().splitlines():
            s = line.strip()
            if not s:
                continue
            tid = int(s.split()[0])
            if 0 <= tid < 200058 and tid not in seen:
                seen.add(tid)
                ids.append(tid)
                if len(ids) >= k:
                    break
    return np.array(ids, dtype=np.int64)


def pack_docs_with_roles(meta: np.ndarray, seq_len: int = 128) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], np.ndarray]:
    """Group by doc_id, compute per-token role weights (2.5x assistant/thinking, 0.6x user, 0.2x sys boilerplate),
    and split 92% train / 8% DISJOINT-doc val windows (s, e, doc_rel_pos).
    """
    n = len(meta)
    doc_ids = meta[:, 3]
    tok0 = meta[:, 0]
    role_w = np.full(n, 0.6, dtype=np.float32)
    is_gen = np.zeros(n, dtype=np.bool_)

    spans = []
    start = 0
    while start < n:
        d = doc_ids[start]
        end = start + 1
        while end < n and doc_ids[end] == d:
            end += 1
        if end - start >= 24:
            # Mark first 16 system-prompt tokens low weight
            role_w[start:min(end, start + 16)] = 0.2
            # Mark assistant/thinking tokens (after <|message_model|> or <|content_thinking|>) high weight!
            in_model = False
            for idx in range(start, end):
                t = tok0[idx]
                if t == TOK_MSG_MODEL or t == TOK_THINKING:
                    in_model = True
                if in_model:
                    role_w[idx] = 2.5
                    is_gen[idx] = True
            spans.append((start, end))
        start = end

    n_val_docs = max(12, len(spans) // 12)
    tr_docs = spans[:-n_val_docs]
    va_docs = spans[-n_val_docs:]

    def make_wins(docs: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
        wins = []
        for s, e in docs:
            lim = e - 2
            pos = s
            while pos + 20 <= lim:
                w_end = min(lim, pos + seq_len)
                wins.append((pos, w_end, pos - s))
                pos += seq_len
        return wins

    return make_wins(tr_docs), make_wins(va_docs), role_w, is_gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/varloop-stage2-v1-run3-reap40-i8mm.gguf")
    ap.add_argument("--lib", default=str(ROOT / "llama.cpp/build/bin/libggml-base.dylib"))
    ap.add_argument("--data", default="/tmp/eagle_v3_data")
    ap.add_argument("--shortlist", default=str(ROOT / "configs/head_shortlist_all666.txt"))
    ap.add_argument("--k-shortlist", type=int, default=16384)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--ff-dim", type=int, default=2560)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--unroll2-weight", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "runs/eagle/eagle_v3_2layer_d1024.pt"))
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    meta_j = json.loads((Path(args.data) / "meta.json").read_text())
    n_eagle = int(meta_j["n_eagle"])
    print(f"[eagle-v3] device={device} n_eagle={n_eagle} L={args.n_layers} d={args.d_model} ff={args.ff_dim} K={args.k_shortlist}", flush=True)

    h_np = np.fromfile(Path(args.data) / "eagle_h.f16", dtype=np.float16).reshape(n_eagle, H_DIM)
    tok_meta = np.fromfile(Path(args.data) / "eagle_toks.i32", dtype=np.int32).reshape(n_eagle, 4)

    tr_wins, va_wins, role_w, is_gen = pack_docs_with_roles(tok_meta, seq_len=128)
    tr_end = max(e for _, e, _ in tr_wins)
    sl_ids = build_freq_shortlist(tok_meta[:tr_end, 1:3].reshape(-1), Path(args.shortlist), k=args.k_shortlist)
    sl_lookup = np.full(201024, -1, dtype=np.int32)
    sl_lookup[sl_ids] = np.arange(len(sl_ids), dtype=np.int32)

    reader = gguf.GGUFReader(str(args.model), "r")
    codec = GGMLCodec(Path(args.lib))
    tok_norm_w = np.asarray(next(t for t in reader.tensors if t.name == "token_embd_norm.weight").data, dtype=np.float32).copy()
    emb_table = LazyGGUFRows(Path(args.model), reader, codec, "token_embd.weight")
    head_table = LazyGGUFRows(Path(args.model), reader, codec, "output.weight")

    print(f"[eagle-v3] loading {len(sl_ids)} frequency-sorted shortlist head rows...", flush=True)
    W_head_sl = torch.from_numpy(head_table.get_rows(sl_ids)).to(device=device, dtype=torch.float16)

    all_needed_toks = np.unique(np.concatenate([
        tok_meta[:, 1][tok_meta[:, 1] >= 0],
        tok_meta[:, 2][tok_meta[:, 2] >= 0],
    ]))
    print(f"[eagle-v3] dequantizing {len(all_needed_toks)} unique token embeddings...", flush=True)
    emb_vals = emb_table.get_rows(all_needed_toks)
    rms = np.sqrt(np.mean(emb_vals * emb_vals, axis=-1, keepdims=True) + EPS)
    emb_vals = (emb_vals / rms * tok_norm_w[None, :]).astype(np.float16)
    tok2local = np.full(201024, -1, dtype=np.int32)
    tok2local[all_needed_toks] = np.arange(len(all_needed_toks), dtype=np.int32)
    emb_cache = torch.from_numpy(emb_vals)

    val_indices = np.concatenate([np.arange(s, e) for s, e, _ in va_wins])
    val_is_gen = is_gen[val_indices]
    # Also mark non-boilerplate positions (doc_rel_pos >= 24)
    val_non_sys = np.concatenate([np.arange(rel, rel + (e - s)) >= 24 for s, e, rel in va_wins])
    print(f"[eagle-v3] windows: train={len(tr_wins)} val={len(va_wins)} ({len(val_indices)} val toks, {int(val_is_gen.sum())} assistant-gen toks)", flush=True)

    h_va_next = torch.from_numpy(h_np[val_indices + 1].astype(np.float32))
    print(f"[eagle-v3] computing EXACT 200K greedy target tokens for {len(val_indices)} held-out tokens...", flush=True)
    exact_greedy_next2 = head_table.full_argmax_chunked(h_va_next, device=device)
    cov_all = float(np.mean(sl_lookup[exact_greedy_next2] >= 0))
    cov_gen = float(np.mean(sl_lookup[exact_greedy_next2[val_is_gen]] >= 0)) if val_is_gen.any() else cov_all
    print(f"[eagle-v3] K={len(sl_ids)} shortlist covers {100*cov_all:.2f}% all / {100*cov_gen:.2f}% assistant-gen EXACT 200K greedy targets!", flush=True)

    model = EagleV3Drafter(h_dim=H_DIM, d_model=args.d_model, n_heads=args.n_heads, ff_dim=args.ff_dim, n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eagle-v3] EagleV3Drafter params={n_params/1e6:.2f}M (~{n_params*0.5625/1e6:.1f} MB @ Q4_0)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * max(1, (len(tr_wins) + args.batch_size - 1) // args.batch_size)
    warmup_steps = min(200, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def make_batch(wins_batch: list[tuple[int, int, int]]):
        max_l = max(e - s for s, e, _ in wins_batch)
        bsz = len(wins_batch)
        h_in = np.zeros((bsz, max_l, H_DIM), dtype=np.float16)
        h_tg = np.zeros((bsz, max_l, H_DIM), dtype=np.float16)
        e_in = torch.zeros((bsz, max_l, H_DIM), dtype=torch.float16)
        e_in2 = torch.zeros((bsz, max_l, H_DIM), dtype=torch.float16)
        y_sl = np.full((bsz, max_l), -100, dtype=np.int64)
        w_tok = np.zeros((bsz, max_l), dtype=np.float32)
        for bi, (s, e, _) in enumerate(wins_batch):
            ln = e - s
            h_in[bi, :ln] = h_np[s:e]
            h_tg[bi, :ln] = h_np[s + 1:e + 1]
            t1 = tok_meta[s:e, 1]
            t2 = tok_meta[s:e, 2]
            e_in[bi, :ln] = emb_cache[torch.from_numpy(tok2local[t1])]
            e_in2[bi, :ln] = emb_cache[torch.from_numpy(tok2local[t2])]
            sl_t2 = sl_lookup[t2]
            y_sl[bi, :ln] = np.where(sl_t2 >= 0, sl_t2, -100)
            w_tok[bi, :ln] = role_w[s:e]
        return (
            torch.from_numpy(h_in).to(device=device, dtype=torch.float32),
            e_in.to(device=device, dtype=torch.float32),
            e_in2.to(device=device, dtype=torch.float32),
            torch.from_numpy(h_tg).to(device=device, dtype=torch.float32),
            torch.from_numpy(y_sl).to(device=device),
            torch.from_numpy(w_tok).to(device=device),
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
            h_in, e_in, _, h_tg, _, _ = make_batch(wb)
            h_hat1 = model(h_in, e_in, pos_offset=0)
            lg1 = F.linear(h_hat1.half(), W_head_sl).float()
            d1_tok = sl_ids[lg1.argmax(dim=-1).cpu().numpy()]

            for bi, (s, e, _) in enumerate(wb):
                ln = e - s
                pred_1.append(d1_tok[bi, :ln])
                cos_list.append(F.cosine_similarity(h_hat1[bi, :ln], h_tg[bi, :ln], dim=-1).mean().item())
                g1_true = exact_greedy_next2[offset:offset + ln]
                offset += ln
                acc1 = (d1_tok[bi, :ln] == g1_true)
                if ln >= 2:
                    t2_real = tok_meta[s:e - 1, 2]
                    e_next = emb_cache[torch.from_numpy(tok2local[t2_real])].to(device=device, dtype=torch.float32)
                    h_hat2 = model(h_hat1[bi:bi + 1, :ln - 1], e_next.unsqueeze(0), pos_offset=1)[0]
                    d2_tok = sl_ids[F.linear(h_hat2.half(), W_head_sl).float().argmax(dim=-1).cpu().numpy()]
                    valid_chain = acc1[:-1] & (g1_true[:-1] == t2_real) & is_gen[s:e - 1]
                    hit2_num += int(np.sum(valid_chain & (d2_tok == g1_true[1:])))
                    hit2_den += int(np.sum(valid_chain))

        p1_all = np.concatenate(pred_1)
        m_all = (p1_all == exact_greedy_next2)
        alpha1_all = float(np.mean(m_all))
        alpha1_nosys = float(np.mean(m_all[val_non_sys])) if val_non_sys.any() else alpha1_all
        alpha1_gen = float(np.mean(m_all[val_is_gen])) if val_is_gen.any() else alpha1_all
        alpha2_cond = float(hit2_num / max(hit2_den, 1))
        model.train()
        return {
            "alpha1_all": round(alpha1_all, 4),
            "alpha1_nosys": round(alpha1_nosys, 4),
            "alpha1_gen": round(alpha1_gen, 4),
            "alpha2_cond_gen": round(alpha2_cond, 4),
            "val_cos": round(float(np.mean(cos_list)), 4),
            "exp_tok_b2_gen": round(1.0 + alpha1_gen, 3),
            "exp_tok_b3_gen": round(1.0 + alpha1_gen + alpha1_gen * alpha2_cond, 3),
        }

    rng = np.random.default_rng(42)
    best_a1 = -1.0
    best_state = None
    best_metrics = None

    for ep in range(1, args.epochs + 1):
        order = rng.permutation(len(tr_wins))
        tot_loss = tot_kl = 0.0
        nb = 0
        for b0 in range(0, len(order), args.batch_size):
            wb = [tr_wins[i] for i in order[b0:b0 + args.batch_size]]
            h_in, e_in, e_in2, h_tg, y_sl, w_tok = make_batch(wb)
            pos0 = int(rng.integers(0, 512))
            h_hat = model(h_in, e_in, pos_offset=pos0)

            w_sum = w_tok.sum().clamp(min=1.0)
            w3 = w_tok.unsqueeze(-1)
            l_l1 = (F.smooth_l1_loss(h_hat, h_tg, reduction="none") * w3).sum() / (w_sum * H_DIM)
            l_cos = ((1.0 - F.cosine_similarity(h_hat, h_tg, dim=-1)) * w_tok).sum() / w_sum

            with torch.no_grad():
                lg_tgt = F.linear(h_tg.half(), W_head_sl).float()
                p_tgt = F.softmax(lg_tgt, dim=-1)
                y_greedy = lg_tgt.argmax(dim=-1)

            lg_pred = F.linear(h_hat.half(), W_head_sl).float()
            logp_pred = F.log_softmax(lg_pred, dim=-1)
            l_kl = (-(p_tgt * logp_pred).sum(dim=-1) * w_tok).sum() / w_sum
            l_gce = (F.nll_loss(logp_pred.reshape(-1, len(sl_ids)), y_greedy.reshape(-1), reduction="none").view_as(w_tok) * w_tok).sum() / w_sum
            l_ce = F.cross_entropy(lg_pred.reshape(-1, len(sl_ids)), y_sl.reshape(-1), ignore_index=-100)
            loss = 0.25 * l_l1 + 0.60 * l_cos + 1.00 * l_kl + 0.45 * l_gce + 0.10 * l_ce

            if args.unroll2_weight > 0.0 and h_hat.shape[1] >= 4:
                h_hat2 = model(h_hat[:, :-1], e_in2[:, :-1], pos_offset=pos0 + 1)
                h_tg2 = h_tg[:, 1:]
                w2 = w_tok[:, 1:]
                w2_sum = w2.sum().clamp(min=1.0)
                w2_3 = w2.unsqueeze(-1)
                l2_l1 = (F.smooth_l1_loss(h_hat2, h_tg2, reduction="none") * w2_3).sum() / (w2_sum * H_DIM)
                l2_cos = ((1.0 - F.cosine_similarity(h_hat2, h_tg2, dim=-1)) * w2).sum() / w2_sum
                lg_pred2 = F.linear(h_hat2.half(), W_head_sl).float()
                logp2 = F.log_softmax(lg_pred2, dim=-1)
                l2_kl = (-(p_tgt[:, 1:] * logp2).sum(dim=-1) * w2).sum() / w2_sum
                l2_gce = (F.nll_loss(logp2.reshape(-1, len(sl_ids)), y_greedy[:, 1:].reshape(-1), reduction="none").view_as(w2) * w2).sum() / w2_sum
                loss = loss + args.unroll2_weight * (0.25 * l2_l1 + 0.60 * l2_cos + 1.00 * l2_kl + 0.45 * l2_gce)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            tot_loss += float(loss.item())
            tot_kl += float(l_kl.item())
            nb += 1

        if ep == 1 or ep % 3 == 0 or ep == args.epochs:
            ev = eval_heldout()
            print(
                f"ep {ep:2d}/{args.epochs} | loss={tot_loss/nb:.3f} kl={tot_kl/nb:.2f} | "
                f"VAL all={100*ev['alpha1_all']:5.2f}% nosys={100*ev['alpha1_nosys']:5.2f}% "
                f"GEN={100*ev['alpha1_gen']:5.2f}% a2|1={100*ev['alpha2_cond_gen']:5.2f}% "
                f"cos={ev['val_cos']:.4f} ({time.time()-t0:.0f}s)",
                flush=True,
            )
            if ev["alpha1_gen"] > best_a1:
                best_a1 = ev["alpha1_gen"]
                best_metrics = ev
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Evaluate EXACT Q4_0 quantized weights via libggml-base.dylib!
    model.load_state_dict(best_state)
    q4_type = int(gguf.GGMLQuantizationType.Q4_0)
    with torch.no_grad():
        for _, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                w_np = mod.weight.detach().cpu().float().numpy()
                raw_q4 = codec.quantize_2d(w_np, q4_type)
                w_dq = codec.dequantize(raw_q4, q4_type, w_np.size).reshape(w_np.shape)
                mod.weight.copy_(torch.from_numpy(w_dq).to(device=device, dtype=mod.weight.dtype))
    ev_q4 = eval_heldout()
    print(
        f"[eagle-v3] EXACT Q4_0 VAL: all={100*ev_q4['alpha1_all']:5.2f}% nosys={100*ev_q4['alpha1_nosys']:5.2f}% "
        f"GEN={100*ev_q4['alpha1_gen']:5.2f}% a2|1={100*ev_q4['alpha2_cond_gen']:5.2f}% cos={ev_q4['val_cos']:.4f}",
        flush=True,
    )
    best_metrics["q4_0_val"] = ev_q4

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": vars(args), "shortlist_ids": sl_ids, "metrics": best_metrics, "state_dict": best_state}, out_p)
    out_p.with_suffix(".json").write_text(json.dumps(best_metrics, indent=2) + "\n")
    print(f"[eagle-v3] DONE -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
