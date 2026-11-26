#!/usr/bin/env python3
"""Losslessly prune routed experts (128 -> M) directly from an inkling2_loop GGUF (and optional LoRA).

Slices raw quantized bytes of blk.{L}.ffn_{gate_up,down}_exps.weight verbatim (zero re-quantization!),
slices F32 blk.{L}.ffn_gate_inp.weight and blk.{L}.exp_probs_b.bias, updates inkling2_loop.expert_count=M,
and preserves every other KV metadata field and tensor byte-for-byte.

Optionally slices a matching GGUF LoRA adapter (blk.{L}.ffn_gate_inp.weight.lora_b).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
import numpy as np

LLAMA_CPP = Path(os.environ.get("LLAMA_CPP_DIR", "third_party/llama.cpp"))
sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
import gguf  # noqa: E402


def select_kept_experts(reap_json: dict, keep_per_layer: list[int], metric: str = "reap_mean_weighted") -> list[list[int]]:
    n_layer = int(reap_json["n_layer"])
    n_exp = int(reap_json["n_expert"])
    assert len(keep_per_layer) == n_layer
    freq = reap_json["freq"]
    w_sum = reap_json["w_sum"]
    reap_sum = reap_json["reap_sum"]
    kept = []
    for l in range(n_layer):
        scores = []
        for e in range(n_exp):
            f = max(0, int(freq[l][e]))
            r = float(reap_sum[l][e])
            w = float(w_sum[l][e])
            if metric == "reap_sum":
                s = r
            elif metric == "reap_mean":
                s = (r / f) if f > 0 else 0.0
            elif metric == "freq":
                s = float(f)
            elif metric == "w_sum":
                s = w
            else:
                # Default REAP-sum (total activation-weighted contribution across calibration tokens):
                # mathematically equals freq * E[|w_e| * ||down_e(x)||_2 | selected], naturally weighting
                # both how often an expert fires on reasoning/writing/tools AND its output magnitude.
                s = r
            scores.append((s, e))
        # Top-keep by score, then sort indices ascending to keep router ordering deterministic
        top = sorted([e for _, e in sorted(scores, key=lambda x: (-x[0], x[1]))[:keep_per_layer[l]]])
        kept.append(top)
    return kept


def copy_kv(reader: gguf.GGUFReader, writer: gguf.GGUFWriter, overrides: dict[str, tuple[gguf.GGUFValueType, object]]) -> None:
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for key, field in reader.fields.items():
        if key in skip:
            continue
        if key in overrides:
            vtype, val = overrides[key]
            writer.add_key_value(key, val, vtype)
            continue
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            sub_type = field.types[1]
            val = field.contents()
            writer.add_key_value(key, list(val), vtype, sub_type=sub_type)
        else:
            val = field.contents()
            writer.add_key_value(key, val, vtype)


def prune_model_gguf(in_path: Path, out_path: Path, kept: list[list[int]]) -> None:
    keep_max = max(len(k) for k in kept)
    keep_min = min(len(k) for k in kept)
    reader = gguf.GGUFReader(str(in_path), "r")
    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(str(out_path), arch, endianess=reader.endianess)

    copy_kv(reader, writer, {
        f"{arch}.expert_count": (gguf.GGUFValueType.UINT32, keep_max),
    })

    pat_exp = re.compile(r"^blk\.(\d+)\.(ffn_gate_up_exps|ffn_down_exps)\.weight$")
    pat_router = re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$")
    pat_bias = re.compile(r"^blk\.(\d+)\.exp_probs_b\.bias$")

    pruned_count = 0
    for t in reader.tensors:
        m_exp = pat_exp.match(t.name)
        m_rtr = pat_router.match(t.name)
        m_bia = pat_bias.match(t.name)
        if m_exp:
            il = int(m_exp.group(1))
            idx = kept[il]
            arr = np.asarray(t.data)[idx].copy()
            writer.add_tensor(t.name, arr, raw_dtype=t.tensor_type)
            pruned_count += 1
        elif m_rtr:
            il = int(m_rtr.group(1))
            idx = kept[il]
            arr = np.asarray(t.data)[idx].copy()
            writer.add_tensor(t.name, arr, raw_dtype=t.tensor_type)
            pruned_count += 1
        elif m_bia:
            il = int(m_bia.group(1))
            idx = kept[il]
            arr = np.asarray(t.data)[idx].copy()
            writer.add_tensor(t.name, arr, raw_dtype=t.tensor_type)
            pruned_count += 1
        else:
            arr = np.asarray(t.data)
            writer.add_tensor(t.name, arr, raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[prune] wrote {out_path} ({out_path.stat().st_size / (1024**3):.2f} GiB, keep={keep_min}..{keep_max}, avg={sum(len(k) for k in kept)/len(kept):.1f}, pruned_tensors={pruned_count})")


def prune_lora_gguf(in_path: Path, out_path: Path, kept: list[list[int]]) -> None:
    keep_max = max(len(k) for k in kept)
    reader = gguf.GGUFReader(str(in_path), "r")
    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(str(out_path), arch, endianess=reader.endianess)
    copy_kv(reader, writer, {})
    pat = re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight\.lora_b$")
    pruned = 0
    for t in reader.tensors:
        m = pat.match(t.name)
        if m:
            il = int(m.group(1))
            arr = np.asarray(t.data)[kept[il]].copy()
            writer.add_tensor(t.name, arr, raw_dtype=t.tensor_type)
            pruned += 1
        else:
            arr = np.asarray(t.data)
            writer.add_tensor(t.name, arr, raw_dtype=t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[prune] wrote pruned LoRA {out_path} (keep_max={keep_max}, pruned_router_lora_b={pruned})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-gguf", required=True, type=Path)
    ap.add_argument("--out-gguf", required=True, type=Path)
    ap.add_argument("--reap-json", required=True, type=Path)
    ap.add_argument("--keep", type=int, default=None, help="uniform experts to keep per layer (e.g. 48, 64, 96)")
    ap.add_argument("--keep-per-layer", type=str, default=None, help="comma-separated per-layer counts (24 ints)")
    ap.add_argument("--metric", default="reap_sum", choices=["reap_sum", "reap_mean", "freq", "w_sum"])
    ap.add_argument("--in-lora", type=Path, default=None)
    ap.add_argument("--out-lora", type=Path, default=None)
    ap.add_argument("--save-plan", type=Path, default=None)
    args = ap.parse_args()

    reap = json.loads(args.reap_json.read_text())
    n_layer = int(reap["n_layer"])
    if args.keep_per_layer:
        kpl = [int(x.strip()) for x in args.keep_per_layer.split(",") if x.strip()]
    elif args.keep is not None:
        kpl = [args.keep] * n_layer
    else:
        raise ValueError("must pass --keep or --keep-per-layer")
    kept = select_kept_experts(reap, kpl, args.metric)
    if args.save_plan:
        args.save_plan.write_text(json.dumps({
            "keep_per_layer": kpl, "metric": args.metric, "n_layer": len(kept), "kept_experts": kept,
        }, indent=1))
    prune_model_gguf(args.in_gguf, args.out_gguf, kept)
    if args.in_lora and args.out_lora:
        prune_lora_gguf(args.in_lora, args.out_lora, kept)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
