#!/usr/bin/env python3
"""Latency + Size Multiple-Choice Knapsack DP Allocator for inkling2_loop GGUF builds.

Implements runs/agy/COMPILER_SCOPE.md Section 2.2 on top of LookAtMyQuant's 1D DP knapsack:
  - Uses our measured ARM NEON kernel table (ns/weight for repacked MUL_MAT and un-repacked or
    mmid-repacked MUL_MAT_ID) combined with an A17 Pro / M4 Pro DRAM bandwidth roofline.
  - Enforces exact active-weight accounting (4/128 for routed experts, 0 for token_embd GET_ROWS,
    1.0 for dense attention / LM head).
  - Enforces block-size divisibility (ne[0] % blck_size == 0) so llama-quantize never falls back.
  - Enforces BOTH a hard latency budget (ms/tok via 1D DP knapsack) AND expert/dense RAM budgets
    (via outer Lagrangian multiplier iteration).
  - Emits both a plan JSON and an anchored ^name$=type file for `llama-quantize --tensor-type-file`.

Dry run:
    python3 train/dp_allocate.py --dry-run
    python3 train/dp_allocate.py --sens runs/agy/dry_sens.json --max-ms 1.0 --out-plan runs/agy/dry_plan.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# -------------------------------------------------------------------------------- measured ARM table
# Measured on Apple M4 Pro (1T, cold ring) in runs/agy/COMPILER_SCOPE.md Section 3.1/3.2
# and updated with GGML_MMID_REPACK 4x4 kernel measurements.
# Fields: (bpw, blck_size, uses_imatrix, mm_repackable, ns_mm_unrep, ns_mm_rep, ns_mmid_unrep, ns_mmid_rep, mmid_rep_mode)
@dataclass(frozen=True)
class KernelSpec:
    bpw: float
    blck_size: int
    uses_imatrix: bool
    mm_repackable: bool
    ns_mm_unrep: float
    ns_mm_rep: float | None
    ns_mmid_unrep: float
    ns_mmid_rep: float | None
    mmid_rep_mode: int  # 0=none, 1=same-size repack, 2=IQ unpack-repack
    mmid_rep_bpw: float | None = None  # cached bpw when mode 2 expands slice


KERNEL_TABLE: dict[str, KernelSpec] = {
    "BF16":    KernelSpec(16.000,   1, False, False, 0.5303, None,   0.5303, None,   0),
    "F16":     KernelSpec(16.000,   1, False, False, 0.0458, None,   0.0458, None,   0),
    "F32":     KernelSpec(32.000,   1, False, False, 0.0551, None,   0.0551, None,   0),
    "IQ3_XXS": KernelSpec( 3.0625,256, True,  False, 0.0664, None,   0.0664, None,   0),
    "IQ3_S":   KernelSpec( 3.4375,256, True,  False, 0.0726, None,   0.0726, None,   0),
    "IQ2_S":   KernelSpec( 2.5625,256, True,  False, 0.0530, None,   0.0507, 0.0169, 2, 4.3125),
    "IQ2_XXS": KernelSpec( 2.0625,256, True,  False, 0.0450, None,   0.0460, 0.0171, 2, 4.3125),
    "IQ2_XS":  KernelSpec( 2.3125,256, True,  False, 0.0429, None,   0.0446, 0.0169, 2, 4.3125),
    "IQ1_S":   KernelSpec( 1.5625,256, True,  False, 0.0408, None,   0.0407, 0.0136, 2, 2.3125),
    "Q2_0":    KernelSpec( 2.2500, 64, False, False, 0.0323, None,   0.0318, 0.0171, 1),
    "Q3_K":    KernelSpec( 3.4375,256, True,  False, 0.0327, None,   0.0316, None,   0),
    "Q1_0":    KernelSpec( 1.1250,128, False, True,  0.0319, 0.0249, 0.0319, 0.0241, 1),
    "Q6_K":    KernelSpec( 6.5625,256, True,  True,  0.0313, 0.0258, 0.0313, 0.0246, 1),
    "Q5_0":    KernelSpec( 5.5000, 32, True,  False, 0.0307, None,   0.0307, None,   0),
    "NVFP4":   KernelSpec( 4.5000, 64, False, False, 0.0273, None,   0.0271, 0.0183, 1),
    "Q5_K":    KernelSpec( 5.5000,256, True,  True,  0.0267, 0.0262, 0.0267, 0.0248, 1),
    "IQ4_NL":  KernelSpec( 4.5000, 32, True,  True,  0.0260, 0.0158, 0.0252, 0.0153, 1),
    "Q2_K":    KernelSpec( 2.6250,256, True,  False, 0.0256, None,   0.0259, 0.0173, 1),
    "IQ4_XS":  KernelSpec( 4.2500,256, True,  False, 0.0243, None,   0.0243, None,   0),
    "MXFP4":   KernelSpec( 4.2500, 32, False, True,  0.0239, 0.0191, 0.0241, 0.0180, 1),
    "Q4_0":    KernelSpec( 4.5000, 32, True,  True,  0.0236, 0.0142, 0.0227, 0.0133, 1),
    "Q8_0":    KernelSpec( 8.5000, 32, False, True,  0.0204, 0.0136, 0.0204, 0.0116, 1),
    "Q4_K":    KernelSpec( 4.5000,256, True,  True,  0.0174, 0.0152, 0.0163, 0.0139, 1),
}


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    b_eff_gbs: float       # sustained DRAM bandwidth across active threads (GB/s)
    cpu_scale: float       # wall-time multiplier vs Mac M4 Pro 1T kernel table
    mmid_repack_mode: int  # 0=off, 1=same-size repack, 2=+IQ unpack-repack
    cache_mb: float = 3000.0


PROFILES: dict[str, DeviceProfile] = {
    # A17 Pro 2 P-cores: ~28 GB/s sustained DRAM, ~1.36x slower single-thread / 2T -> ~0.72x Mac 1T wall
    "iphone15pro_2t":         DeviceProfile("iphone15pro_2t",         28.0, 0.72, 0, 3000.0),
    "iphone15pro_2t_repack1": DeviceProfile("iphone15pro_2t_repack1", 28.0, 0.72, 1, 3000.0),
    "iphone15pro_2t_repack2": DeviceProfile("iphone15pro_2t_repack2", 28.0, 0.72, 2, 3000.0),
    "mac_m4pro_1t":           DeviceProfile("mac_m4pro_1t",           75.0, 1.00, 0, 3000.0),
    "mac_m4pro_2t":           DeviceProfile("mac_m4pro_2t",           90.0, 0.54, 0, 3000.0),
    "mac_m4pro_2t_repack1":   DeviceProfile("mac_m4pro_2t_repack1",   90.0, 0.54, 1, 3000.0),
    "mac_m4pro_2t_repack2":   DeviceProfile("mac_m4pro_2t_repack2",   90.0, 0.54, 2, 3000.0),
}


def price_option(
    rec: dict,
    profile: DeviceProfile,
    k_used: int = 4,
    n_experts_default: int = 128,
) -> tuple[float, float, float]:
    """Return (ms_per_tok, expert_bytes, dense_bytes) for one sensitivity record."""
    opt = str(rec["option"]).upper()
    if opt not in KERNEL_TABLE:
        raise KeyError(f"unknown quant type {opt!r} not in KERNEL_TABLE")
    spec = KERNEL_TABLE[opt]
    gname = str(rec.get("gguf tensor", rec["item"]))
    weights = int(rec["weights"])
    exec_count = int(rec.get("exec_count", 1))
    disk_bytes = float(rec["bytes"]) if "bytes" in rec else float(rec["bits"]) / 8.0

    is_exp = gname.endswith("_exps.weight") or rec.get("kind") == "experts"
    is_embd = gname == "token_embd.weight"

    if is_embd:
        return 0.0, 0.0, disk_bytes

    if is_exp:
        n_exp = int(rec.get("n_experts", n_experts_default))
        active_w = weights * (k_used / max(n_exp, 1))
        if profile.mmid_repack_mode >= spec.mmid_rep_mode > 0 and spec.ns_mmid_rep is not None:
            t_compute_ns = spec.ns_mmid_rep * profile.cpu_scale
            read_bpw = spec.mmid_rep_bpw if spec.mmid_rep_bpw is not None else spec.bpw
        else:
            t_compute_ns = spec.ns_mmid_unrep * profile.cpu_scale
            read_bpw = spec.bpw
        exp_b, dense_b = disk_bytes, 0.0
    else:
        active_w = float(weights)
        if spec.mm_repackable and spec.ns_mm_rep is not None:
            t_compute_ns = spec.ns_mm_rep * profile.cpu_scale
        else:
            t_compute_ns = spec.ns_mm_unrep * profile.cpu_scale
        read_bpw = spec.bpw
        exp_b, dense_b = 0.0, disk_bytes

    # Roofline: max(compute ns/w, bytes/w / B_eff_GB_s)  (note: (B/w) / (GB/s) = ns/w!)
    t_bw_ns = (read_bpw / 8.0) / profile.b_eff_gbs
    ms = exec_count * active_w * max(t_compute_ns, t_bw_ns) * 1e-6
    return ms, exp_b, dense_b


def is_legal_option(rec: dict) -> bool:
    opt = str(rec["option"]).upper()
    spec = KERNEL_TABLE.get(opt)
    if spec is None:
        return False
    ne0 = rec.get("ne0")
    if ne0 is None and "shape" in rec and rec["shape"]:
        ne0 = int(rec["shape"][-1])
    if ne0 is not None and int(ne0) % spec.blck_size != 0:
        return False
    return True


# -------------------------------------------------------------------------------- 1D DP knapsack

def knapsack_1d(groups: list[list[tuple[float, float]]], budget: float, cells: int = 100_000) -> list[int]:
    """Exact multiple-choice knapsack DP (matching LookAtMyQuant compiler.py:693-721).
    groups[i] = [(cost_ms, objective_loss), ...]. Returns chosen option index per item."""
    lo = [min(c for c, _ in g) for g in groups]
    room = budget - sum(lo)
    if room < -1e-9:
        raise ValueError(f"latency budget {budget:.4f} ms is {-room:.4f} ms below fastest possible plan ({sum(lo):.4f} ms)")
    room = max(room, 0.0)
    unit = room / cells if room > 0 else 1.0
    steps = [[int(math.ceil((c - m) / unit - 1e-9)) if room > 0 else (0 if c <= m + 1e-12 else cells + 1)
              for c, _ in g] for g, m in zip(groups, lo)]

    dp = np.zeros(cells + 1, dtype=np.float64)
    pick = np.zeros((len(groups), cells + 1), dtype=np.int16)
    for i, g in enumerate(groups):
        new = np.full(cells + 1, np.inf, dtype=np.float64)
        arg = np.zeros(cells + 1, dtype=np.int16)
        for o, ((_, loss), w) in enumerate(zip(g, steps[i])):
            if w > cells:
                continue
            cand = np.full(cells + 1, np.inf, dtype=np.float64)
            cand[w:] = dp[: cells + 1 - w] + loss
            better = cand < new
            new[better] = cand[better]
            arg[better] = o
        dp, pick[i] = new, arg

    best_c = int(np.argmin(dp))
    if not np.isfinite(dp[best_c]):
        raise ValueError(f"no feasible plan found under latency budget {budget:.4f} ms")
    c = best_c
    choice = []
    for i in range(len(groups) - 1, -1, -1):
        o = int(pick[i, c])
        choice.append(o)
        c -= steps[i][o]
    return choice[::-1]


# -------------------------------------------------------------------------------- Lagrangian solver

def solve_latency_size_dp(
    items_by_name: dict[str, dict[str, dict]],
    profile: DeviceProfile,
    max_ms: float,
    max_expert_gb: float | None = None,
    max_dense_gb: float | None = None,
    k_used: int = 4,
    n_experts_default: int = 128,
    cells: int = 100_000,
    max_iters: int = 30,
) -> dict:
    """Minimize sum(dkl) s.t. sum(ms) <= max_ms, sum(expert_gb) <= max_expert_gb, sum(dense_gb) <= max_dense_gb."""
    names = sorted(items_by_name.keys())
    opts: dict[str, list[str]] = {}
    priced: dict[str, list[tuple[str, float, float, float, float]]] = {}  # (opt, ms, exp_gb, dense_gb, dkl)

    for n in names:
        valid = []
        for opt, rec in sorted(items_by_name[n].items()):
            if not is_legal_option(rec):
                continue
            ms, exp_b, dense_b = price_option(rec, profile, k_used=k_used, n_experts_default=n_experts_default)
            valid.append((opt, ms, exp_b / 1e9, dense_b / 1e9, max(0.0, float(rec["dkl"]))))
        if not valid:
            raise ValueError(f"no legal options for item {n}")
        opts[n] = [v[0] for v in valid]
        priced[n] = valid

    min_ms = sum(min(v[1] for v in priced[n]) for n in names)
    min_exp = sum(min(v[2] for v in priced[n]) for n in names)
    min_den = sum(min(v[3] for v in priced[n]) for n in names)
    if max_ms < min_ms - 1e-6:
        raise ValueError(f"--max-ms {max_ms:.3f} < minimum achievable {min_ms:.3f} ms")
    if max_expert_gb is not None and max_expert_gb < min_exp - 1e-6:
        raise ValueError(f"--max-expert-gb {max_expert_gb:.3f} < minimum achievable {min_exp:.3f} GB")
    if max_dense_gb is not None and max_dense_gb < min_den - 1e-6:
        raise ValueError(f"--max-dense-gb {max_dense_gb:.3f} < minimum achievable {min_den:.3f} GB")

    def solve_once(lam_exp: float, lam_den: float):
        groups = [
            [(ms, dkl + lam_exp * e_gb + lam_den * d_gb) for (_, ms, e_gb, d_gb, dkl) in priced[n]]
            for n in names
        ]
        pick = knapsack_1d(groups, max_ms, cells=cells)
        chosen = {}
        tot_ms = tot_exp = tot_den = tot_kl = 0.0
        for n, k in zip(names, pick):
            opt, ms, e_gb, d_gb, dkl = priced[n][k]
            chosen[n] = opt
            tot_ms += ms
            tot_exp += e_gb
            tot_den += d_gb
            tot_kl += dkl
        return chosen, tot_ms, tot_exp, tot_den, tot_kl

    # Estimate natural KL/GB scale for step sizing.
    dkl_span = sum(max(v[4] for v in priced[n]) - min(v[4] for v in priced[n]) for n in names)
    exp_span = max(1e-6, sum(max(v[2] for v in priced[n]) - min(v[2] for v in priced[n]) for n in names))
    den_span = max(1e-6, sum(max(v[3] for v in priced[n]) - min(v[3] for v in priced[n]) for n in names))
    scale_exp = max(1e-4, dkl_span / exp_span)
    scale_den = max(1e-4, dkl_span / den_span)

    lam_exp = 0.0
    lam_den = 0.0
    best_feasible = None
    best_any = None

    # Coordinate / subgradient Lagrangian loop with bracketed refinement.
    for it in range(max_iters):
        chosen, tot_ms, tot_exp, tot_den, tot_kl = solve_once(lam_exp, lam_den)
        viol_exp = max(0.0, tot_exp - max_expert_gb) if max_expert_gb is not None else 0.0
        viol_den = max(0.0, tot_den - max_dense_gb) if max_dense_gb is not None else 0.0
        viol = viol_exp + viol_den

        cand_any = (viol, tot_kl, chosen, tot_ms, tot_exp, tot_den, lam_exp, lam_den)
        if best_any is None or cand_any[:2] < best_any[:2]:
            best_any = cand_any
        if viol <= 1e-6:
            if best_feasible is None or tot_kl < best_feasible[0]:
                best_feasible = (tot_kl, chosen, tot_ms, tot_exp, tot_den, lam_exp, lam_den)
            if lam_exp == 0.0 and lam_den == 0.0:
                break

        step = 2.0 / math.sqrt(it + 1)
        if max_expert_gb is not None:
            diff_e = tot_exp - max_expert_gb
            if diff_e > 0:
                lam_exp = max(lam_exp * 1.6, lam_exp + step * scale_exp * (diff_e / max(exp_span, 1e-3) + 0.1))
            elif lam_exp > 0:
                lam_exp *= 0.65
        if max_dense_gb is not None:
            diff_d = tot_den - max_dense_gb
            if diff_d > 0:
                lam_den = max(lam_den * 1.6, lam_den + step * scale_den * (diff_d / max(den_span, 1e-3) + 0.1))
            elif lam_den > 0:
                lam_den *= 0.65

    if best_feasible is not None:
        tot_kl, chosen, tot_ms, tot_exp, tot_den, lam_exp, lam_den = best_feasible
        feasible = True
    else:
        _, tot_kl, chosen, tot_ms, tot_exp, tot_den, lam_exp, lam_den = best_any
        feasible = False

    # Summarize by role.
    role_summary: dict[str, dict] = {}
    for n, opt in chosen.items():
        rec = items_by_name[n][opt]
        role = rec.get("role", rec.get("kind", "other"))
        ms, eb, db = price_option(rec, profile, k_used=k_used, n_experts_default=n_experts_default)
        rs = role_summary.setdefault(role, {"counts": {}, "ms": 0.0, "gb": 0.0, "dkl": 0.0})
        rs["counts"][opt] = rs["counts"].get(opt, 0) + 1
        rs["ms"] += ms
        rs["gb"] += (eb + db) / 1e9
        rs["dkl"] += max(0.0, float(rec["dkl"]))
    for rs in role_summary.values():
        rs["ms"] = round(rs["ms"], 4)
        rs["gb"] = round(rs["gb"], 4)
        rs["dkl"] = round(rs["dkl"], 6)

    return {
        "profile": profile.name,
        "feasible": feasible,
        "budgets": {"max_ms": max_ms, "max_expert_gb": max_expert_gb, "max_dense_gb": max_dense_gb},
        "totals": {
            "ms_per_tok": round(tot_ms, 4),
            "expert_gb": round(tot_exp, 4),
            "dense_gb": round(tot_den, 4),
            "predicted_dkl": round(tot_kl, 6),
            "cache_coverage_pct": round(min(100.0, 100.0 * (profile.cache_mb / 1000.0) / max(tot_exp, 1e-6)), 1)
            if tot_exp > 0 else 100.0,
        },
        "lambdas": {"expert": round(lam_exp, 6), "dense": round(lam_den, 6)},
        "role_summary": role_summary,
        "tensor_types": dict(sorted(chosen.items())),
    }


def write_anchored_tensor_types(plan: dict, out_path: Path, fixed: dict[str, str] | None = None) -> None:
    """Emit anchored ^name$=type overrides for llama-quantize --tensor-type-file."""
    merged = dict(plan["tensor_types"])
    if fixed:
        merged.update(fixed)
    lines = []
    for name, tname in sorted(merged.items()):
        lines.append(f"^{re.escape(name)}$={tname.lower()}\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines))


# -------------------------------------------------------------------------------- dry-run fixture

def build_varloop_synthetic_sens() -> tuple[dict[str, dict[str, dict]], dict]:
    """Full 24-layer TML-d24-varloop-research geometry synthetic fixture for solver verification."""
    rng = np.random.default_rng(7)
    items: dict[str, dict[str, dict]] = {}
    for layer in range(24):
        edge = 1.8 if (layer < 3 or layer >= 21) else 1.0
        # gate_up: [128, 2304, 2304], ne0=2304
        gname = f"blk.{layer}.ffn_gate_up_exps.weight"
        items[gname] = {}
        for opt, base_kl in (("Q2_0", 0.018), ("IQ2_S", 0.008), ("Q2_K", 0.006), ("Q3_K", 0.003), ("Q4_K", 0.001)):
            w = 128 * 2304 * 2304
            b = int(w * KERNEL_TABLE[opt].bpw / 8.0)
            items[gname][opt] = {
                "item": gname, "gguf tensor": gname, "kind": "experts", "role": "ffn_gate_up_exps",
                "layer": layer, "shape": [128, 2304, 2304], "ne0": 2304, "n_experts": 128,
                "exec_count": 1, "option": opt, "bits": 8.0 * b, "bytes": b, "weights": w,
                "dkl": float(base_kl * edge * (0.85 + 0.3 * rng.random())),
            }
        # down: [128, 2304, 1152], ne0=1152 (deliberately inject illegal Q2_K to test filtering!)
        dname = f"blk.{layer}.ffn_down_exps.weight"
        items[dname] = {}
        for opt, base_kl in (("Q2_K", 0.0001), ("Q2_0", 0.012), ("MXFP4", 0.004), ("Q4_0", 0.0035), ("IQ4_NL", 0.003), ("NVFP4", 0.0032)):
            w = 128 * 2304 * 1152
            b = int(w * KERNEL_TABLE[opt].bpw / 8.0)
            items[dname][opt] = {
                "item": dname, "gguf tensor": dname, "kind": "experts", "role": "ffn_down_exps",
                "layer": layer, "shape": [128, 2304, 1152], "ne0": 1152, "n_experts": 128,
                "exec_count": 1, "option": opt, "bits": 8.0 * b, "bytes": b, "weights": w,
                "dkl": float(base_kl * edge * (0.85 + 0.3 * rng.random())),
            }
        # dense attention projections
        for stem, rows in (("attn_q", 3072), ("attn_k", 1024), ("attn_v", 1024), ("attn_gate", 3072), ("attn_rel", 384)):
            aname = f"blk.{layer}.{stem}.weight"
            items[aname] = {}
            sens = 1.6 if stem in ("attn_v", "attn_k") else 1.0
            for opt, base_kl in (("Q4_K", 0.004), ("Q5_K", 0.0018), ("Q6_K", 0.0007), ("Q8_0", 0.0001)):
                w = rows * 2304
                b = int(w * KERNEL_TABLE[opt].bpw / 8.0)
                items[aname][opt] = {
                    "item": aname, "gguf tensor": aname, "kind": "linear", "role": stem,
                    "layer": layer, "shape": [rows, 2304], "ne0": 2304, "n_experts": 1,
                    "exec_count": 1, "option": opt, "bits": 8.0 * b, "bytes": b, "weights": w,
                    "dkl": float(base_kl * edge * sens * (0.85 + 0.3 * rng.random())),
                }
    return items, {"top_k": 4, "n_routed_experts": 128}


def load_sens_json(path: str | Path) -> tuple[dict[str, dict[str, dict]], dict]:
    blob = json.loads(Path(path).read_text())
    records = blob["records"] if isinstance(blob, dict) and "records" in blob else blob
    meta = blob.get("meta", {}) if isinstance(blob, dict) else {}
    items: dict[str, dict[str, dict]] = {}
    for r in records:
        name = str(r.get("gguf tensor", r["item"]))
        items.setdefault(name, {})[str(r["option"]).upper()] = r
    return items, meta


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--sens", default=None, help="sensitivity JSON produced by train/kl_sweep.py")
    a("--dry-run", action="store_true", help="verify solver on full 24L varloop synthetic fixture (and --sens if given)")
    a("--profile", default="iphone15pro_2t", choices=sorted(PROFILES.keys()))
    a("--max-ms", type=float, default=32.0, help="hard matmul latency budget in ms/tok")
    a("--max-expert-gb", type=float, default=8.25, help="expert pool disk/cache size budget in GB")
    a("--max-dense-gb", type=float, default=0.45, help="swept dense attention RAM budget in GB")
    a("--cells", type=int, default=100_000, help="DP discretization bins")
    a("--out-plan", default=None, help="write plan JSON")
    a("--out-tensor-types", default=None, help="write anchored --tensor-type-file for llama-quantize")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    profile = PROFILES[args.profile]

    if args.dry_run and not args.sens:
        items, meta = build_varloop_synthetic_sens()
        plan = solve_latency_size_dp(
            items, profile,
            max_ms=args.max_ms,
            max_expert_gb=args.max_expert_gb,
            max_dense_gb=args.max_dense_gb,
            k_used=int(meta.get("top_k", 4)),
            n_experts_default=int(meta.get("n_routed_experts", 128)),
            cells=args.cells,
        )
        # Verify illegal Q2_K was never picked on ne[0]=1152 ffn_down_exps.
        for name, opt in plan["tensor_types"].items():
            if "ffn_down_exps" in name and KERNEL_TABLE[opt].blck_size == 256:
                raise RuntimeError(f"illegal QK_K=256 type {opt} chosen for {name}")
        assert plan["feasible"], f"dry-run plan not feasible: {plan['totals']}"
        assert plan["totals"]["ms_per_tok"] <= args.max_ms + 1e-3
        assert plan["totals"]["expert_gb"] <= args.max_expert_gb + 1e-3
        assert plan["totals"]["dense_gb"] <= args.max_dense_gb + 1e-3
        print(json.dumps({k: v for k, v in plan.items() if k != "tensor_types"}, indent=2))
        if args.out_plan:
            Path(args.out_plan).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_plan).write_text(json.dumps(plan, indent=1))
        if args.out_tensor_types:
            write_anchored_tensor_types(plan, Path(args.out_tensor_types))
        return 0

    if not args.sens:
        raise SystemExit("pass --sens <file.json> or --dry-run")

    items, meta = load_sens_json(args.sens)
    plan = solve_latency_size_dp(
        items, profile,
        max_ms=args.max_ms,
        max_expert_gb=args.max_expert_gb,
        max_dense_gb=args.max_dense_gb,
        k_used=int(meta.get("top_k", 4)),
        n_experts_default=int(meta.get("n_routed_experts", 128)),
        cells=args.cells,
    )
    print(json.dumps({k: v for k, v in plan.items() if k != "tensor_types"}, indent=2))
    if args.out_plan:
        Path(args.out_plan).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_plan).write_text(json.dumps(plan, indent=1))
    if args.out_tensor_types:
        write_anchored_tensor_types(plan, Path(args.out_tensor_types))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
