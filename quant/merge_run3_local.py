#!/usr/bin/env python3
"""Merge run3-1p-k4-step379.gguf (176 attention LoRA + 24 F32 router LoRA) into varloop-stage2-v1.gguf
directly from HuggingFace BF16 (~0.80 GiB fetched, ~90s, $0) + reorder experts by hot_plan_40.json (0..39 = HOT).

Produces one standalone 9.29 GiB GGUF with 0 ms LoRA runtime overhead and bit-exact unmodified experts!
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import merge_wsd_local as M  # noqa: E402
import gguf  # noqa: E402


def load_lora_gguf(lora_path: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[int, np.ndarray]]:
    """Load pre-scaled LoRA pairs from a GGUF adapter (train/distill.py stores lora_B * scale)."""
    r = gguf.GGUFReader(str(lora_path), "r")
    alpha = float(r.fields["adapter.lora.alpha"].contents()) if "adapter.lora.alpha" in r.fields else 0.0
    pairs: dict[str, dict[str, np.ndarray]] = {}
    for t in r.tensors:
        arr = np.asarray(t.data, dtype=np.float32)
        if t.name.endswith(".lora_a"):
            pairs.setdefault(t.name[:-7], {})["A"] = arr
        elif t.name.endswith(".lora_b"):
            pairs.setdefault(t.name[:-7], {})["B"] = arr
    dense_deltas: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    router_deltas: dict[int, np.ndarray] = {}
    pat_rtr = re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$")
    for base_name, d in pairs.items():
        A, B = d["A"], d["B"]  # GGUF shape: A=[rank, in] in numpy, B=[out, rank] in numpy
        rank = A.shape[0]
        sc = (alpha / rank) if alpha > 0 else 1.0
        if m := pat_rtr.match(base_name):
            router_deltas[int(m.group(1))] = (B @ A) * sc
        else:
            dense_deltas[base_name] = (A, B * sc)
    return dense_deltas, router_deltas


def main() -> None:
    from huggingface_hub import hf_hub_url
    base_gguf = Path("./models/varloop-stage2-v1.gguf")
    lora_gguf = Path("./models/adapters/run3-1p-k4-step379.gguf")
    im_path = Path("./models/imatrix-TML-d24-varloop-research.gguf")
    plan_path = ROOT / "runs/reap/hot_plan_40.json"
    lib_path = ROOT / "llama.cpp/build/bin/libggml-base.dylib"
    out_gguf = Path("./models/varloop-stage2-v1-run3-reap40.gguf")

    t0 = time.time()
    url = hf_hub_url(M.BF16_REPO, M.BF16_FILE)
    data_off, remote = M.parse_remote_gguf_header(url)
    im = M.load_imatrix_norm(im_path)
    q = M.GGMLQuantizer(lib_path)

    dense_deltas, router_deltas = load_lora_gguf(lora_gguf)
    plan = json.loads(plan_path.read_text())
    perms = M.build_layer_perms(plan["hot_experts"], n_expert=128)
    print(f"Loaded run3 LoRA: dense={len(dense_deltas)}, router={len(router_deltas)}, hot=40/layer", flush=True)

    reader = gguf.GGUFReader(str(base_gguf), "r")
    arch = reader.fields["general.architecture"].contents()
    out_tmp = out_gguf.with_suffix(".gguf.tmp")
    writer = gguf.GGUFWriter(str(out_tmp), arch, endianess=reader.endianess)
    M.copy_kv(reader, writer)

    tensor_list = list(reader.tensors)
    for t in tensor_list:
        raw = np.asarray(t.data)
        writer.add_tensor_info(t.name, raw.shape, raw.dtype, raw.nbytes, raw_dtype=t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    pat_exp = re.compile(r"^blk\.(\d+)\.(ffn_gate_up_exps|ffn_down_exps)\.weight$")
    pat_rtr = re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$")
    pat_bia = re.compile(r"^blk\.(\d+)\.exp_probs_b\.bias$")

    # Prefetch all 176 BF16 attention tensors in parallel (~0.80 GiB total)!
    pool = ThreadPoolExecutor(max_workers=8)
    futs = {}
    for name in dense_deltas:
        dims, rtype, toff = remote[name]
        assert rtype == int(gguf.GGMLQuantizationType.BF16)
        ne0, ne1 = dims[0], dims[1]
        futs[name] = (ne0, ne1, pool.submit(M.fetch_range, url, data_off + toff, ne0 * ne1 * 2))

    merged_dense, reord_exp = 0, 0
    try:
        for t in tensor_list:
            name = t.name
            raw_local = np.asarray(t.data)
            if m := pat_exp.match(name):
                il = int(m.group(1))
                out_arr = np.ascontiguousarray(raw_local[perms[il]])
                writer.write_tensor_data(out_arr)
                reord_exp += 1
            elif m := pat_rtr.match(name):
                il = int(m.group(1))
                w = raw_local.astype(np.float32).copy()
                if il in router_deltas:
                    w += router_deltas[il]
                writer.write_tensor_data(np.ascontiguousarray(w[perms[il]], dtype=np.float32))
            elif m := pat_bia.match(name):
                il = int(m.group(1))
                b = raw_local.astype(np.float32)[perms[il]]
                writer.write_tensor_data(np.ascontiguousarray(b, dtype=np.float32))
            elif name in dense_deltas:
                A, B = dense_deltas[name]
                ne0, ne1, fut = futs[name]
                f32 = M.bf16_bytes_to_f32(fut.result(), (ne1, ne0)).copy()
                f32 += B @ A
                im_v = im[name][0] if name in im else None
                qarr = q.quantize_2d(f32, int(t.tensor_type), im_v)
                assert qarr.nbytes == t.n_bytes
                writer.write_tensor_data(qarr)
                merged_dense += 1
                if merged_dense % 44 == 0:
                    print(f"  merged {merged_dense}/{len(dense_deltas)} dense tensors ({time.time()-t0:.1f}s)", flush=True)
            else:
                writer.write_tensor_data(raw_local)
    finally:
        pool.shutdown(wait=True)

    writer.close()
    assert merged_dense == len(dense_deltas) and reord_exp == 48
    os.replace(out_tmp, out_gguf)
    print(f"DONE -> {out_gguf} ({out_gguf.stat().st_size/1024**3:.3f} GiB) in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
