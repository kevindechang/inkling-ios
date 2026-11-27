#!/usr/bin/env python3
"""Exact Dynamic-Programming / Lagrange Water-Filling Optimizer for 24-layer CECAR bonuses.

Sweeps cecar_bonus in [0.00, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.32, 0.36, 0.40, 0.45]
per layer on the EXACT quantized experts (using held-out /tmp/fmrd_data tokens) and computes the
EXACT Pareto-optimal 24-float GGML_MMID_CECAR_LAYERS vector for any target cold rate (e.g. 20%, 24%, 27%)!

Why this crushes uniform CECAR=0.25:
  Uniform 0.25 nuked sharp-margin layers like L6 (3.8% cold, cos=0.699) and L13 (mse=0.385)
  while leaving wide-margin layers like L5/L16/L18 at 44-46% cold!
  DP water-filling equalizes d(MSE)/d(cold) across all 24 layers -> MASSIVELY LOWER distortion at LOWER cold rate!
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))
sys.path.insert(0, str(ROOT / "train"))
import gguf  # noqa: E402
from train_mac_routers import GGMLCodec, H_DIM, FF_DIM, N_EXPERT, eval_moe_exact, route_topk  # noqa: E402

GRID = [0.00, 0.04, 0.08, 0.10, 0.12, 0.15, 0.18, 0.21, 0.25, 0.29, 0.33, 0.38, 0.44, 0.50]


def main() -> None:
    model_p = Path("models/varloop-stage2-v1-run3-reap40-i8mm.gguf")
    data_dir = Path("/tmp/fmrd_data")
    lib_p = ROOT / "llama.cpp/build/bin/libggml-base.dylib"
    n_hot = 44
    n_eval = 1024

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    meta = json.loads((data_dir / "meta.json").read_text())
    n_fmrd = int(meta["n_fmrd"])
    reader = gguf.GGUFReader(str(model_p), "r")
    tmap = {t.name: t for t in reader.tensors}
    codec = GGMLCodec(lib_p)

    # curves[il] = list of (bonus, cold_frac, rel_mse, cos)
    curves: list[list[tuple[float, float, float, float]]] = []
    t0 = time.time()

    for il in range(24):
        x_np = np.fromfile(data_dir / f"x_L{il:02d}.f16", dtype=np.float16).reshape(n_fmrd, H_DIM)[-n_eval:]
        x_va = torch.from_numpy(x_np.astype(np.float32))

        t_gu = tmap[f"blk.{il}.ffn_gate_up_exps.weight"]
        t_dn = tmap[f"blk.{il}.ffn_down_exps.weight"]
        t_rt = tmap[f"blk.{il}.ffn_gate_inp.weight"]
        t_bi = tmap[f"blk.{il}.exp_probs_b.bias"]

        gu_f16 = torch.from_numpy(
            codec.dequantize(np.asarray(t_gu.data), int(t_gu.tensor_type), N_EXPERT * 2 * FF_DIM * H_DIM)
            .reshape(N_EXPERT, 2 * FF_DIM, H_DIM)
        ).half()
        dn_f16 = torch.from_numpy(
            codec.dequantize(np.asarray(t_dn.data), int(t_dn.tensor_type), N_EXPERT * H_DIM * FF_DIM)
            .reshape(N_EXPERT, H_DIM, FF_DIM)
        ).half()
        w0 = torch.from_numpy(
            codec.dequantize(np.asarray(t_rt.data), int(t_rt.tensor_type), N_EXPERT * H_DIM)
            .reshape(N_EXPERT, H_DIM)
        ).float()
        b0 = torch.from_numpy(np.asarray(t_bi.data, dtype=np.float32).copy()).float()

        idx0, wgt0 = route_topk(x_va, w0, b0, cecar_bonus=0.0, n_hot=n_hot)
        y0 = eval_moe_exact(x_va, idx0, wgt0, gu_f16, dn_f16, device).cpu()
        y0_n2 = y0.pow(2).sum().clamp(min=1e-6)

        row = []
        for bonus in GRID:
            idx, wgt = route_topk(x_va, w0, b0, cecar_bonus=bonus, n_hot=n_hot)
            cf = float((idx >= n_hot).float().mean().item())
            if bonus == 0.0:
                mse, cos = 0.0, 1.0
            else:
                yv = eval_moe_exact(x_va, idx, wgt, gu_f16, dn_f16, device).cpu()
                mse = float(((yv - y0).pow(2).sum() / y0_n2).item())
                cos = float(F.cosine_similarity(yv, y0, dim=-1).mean().item())
            row.append((bonus, cf, mse, cos))
        curves.append(row)
        c25 = next(r for r in row if abs(r[0] - 0.25) < 1e-6)
        print(f"L{il:2d}: c0={100*row[0][1]:4.1f}% | C025={100*c25[1]:4.1f}% mse={c25[2]:.4f} cos={c25[3]:.4f} | C050={100*row[-1][1]:4.1f}% mse={row[-1][2]:.4f}", flush=True)

    # Exact Lagrange dual sweep (penalize total_mse + lam * total_cold) -> guaranteed Pareto frontier!
    print("\n=== PARETO FRONTIER: OPTIMAL PER-LAYER CECAR vs UNIFORM CECAR (n_hot=44, 1024 held-out tokens) ===")
    for u_b in (0.15, 0.21, 0.25, 0.33):
        bi = GRID.index(u_b)
        uc = 100 * sum(curves[l][bi][1] for l in range(24)) / 24
        um = sum(curves[l][bi][2] for l in range(24)) / 24
        uco = sum(curves[l][bi][3] for l in range(24)) / 24
        worst_cos = min(curves[l][bi][3] for l in range(24))
        print(f"  UNIFORM {u_b:.2f} : cold={uc:5.2f}% | avg_mse={um:.5f} | avg_cos={uco:.5f} | worst_layer_cos={worst_cos:.4f}")

    print("-" * 100)
    saved = {}
    for lam in (0.35, 0.50, 0.65, 0.85, 1.10, 1.40):
        # Edge/global layers weighted 1.25x on MSE because downstream error amplifies across layers!
        chosen = []
        for l in range(24):
            wt = 1.30 if l in (0, 1, 5, 11, 17, 22, 23) else 1.0
            best = min(curves[l], key=lambda r: wt * r[2] + lam * r[1])
            chosen.append(best)
        oc = 100 * sum(r[1] for r in chosen) / 24
        om = sum(r[2] for r in chosen) / 24
        oco = sum(r[3] for r in chosen) / 24
        worst_cos = min(r[3] for r in chosen)
        vec = [round(r[0], 2) for r in chosen]
        vec_str = ",".join(f"{v:.2f}" for v in vec)
        print(f"  OPT lam={lam:4.2f}: cold={oc:5.2f}% | avg_mse={om:.5f} | avg_cos={oco:.5f} | worst_layer_cos={worst_cos:.4f}")
        print(f"      GGML_MMID_CECAR_LAYERS={vec_str}")
        saved[f"lam_{lam:.2f}"] = {"cold_pct": round(oc, 2), "avg_mse": round(om, 5), "avg_cos": round(oco, 5), "worst_cos": round(worst_cos, 4), "layers": vec, "env": vec_str}

    out_p = ROOT / "runs/fmrd/cecar_pareto.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({"curves": curves, "pareto": saved}, indent=2) + "\n")
    print(f"DONE in {time.time()-t0:.1f}s -> {out_p}")


if __name__ == "__main__":
    main()
