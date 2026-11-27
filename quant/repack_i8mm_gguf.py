#!/usr/bin/env python3
"""Upgrade slow non-i8mm tensors in varloop-stage2-v1-run3-reap40.gguf (ALL 128/128 experts intact, 0 pruned)
directly from HF BF16 + official normalized imatrix ($0, ~2 min):
  - blk.23.ffn_gate_up_exps.weight: Q3_K (un-repacked, 1.01 ms) -> Q4_K (HIGHER precision + i8mm inter=8 repacked!)
  - blk.{4,5,6,7,8,10,11,16,17}.ffn_down_exps.weight: NVFP4/MXFP4 (inter=4) -> Q4_0 (SAME 4.5 bpw + i8mm inter=8!)
    (keeps L0, L1, L22 IQ4_NL edge layers 100% untouched!)
  - blk.{0..23}.ffn_gate_inp.weight: F32 (28.3 MB) -> Q8_0 (7.5 MB, CPU_REPACK i8mm inter=8, -20.8 MB RAM!)
  - All other tensors (Q2_K, Q2_0, merged attention, token_embd, output, exp_probs_b) copied 100% bit-for-bit!
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "quant"))
import merge_wsd_local as M  # noqa: E402
import gguf  # noqa: E402

Q4_0_DOWN_LAYERS = {4, 5, 6, 7, 8, 10, 11, 16, 17}


def main() -> None:
    from huggingface_hub import hf_hub_url

    src_gguf = Path("models/varloop-stage2-v1-run3-reap40.gguf")
    im_path = Path("models/imatrix-TML-d24-varloop-research.gguf")
    plan_path = ROOT / "runs/reap/hot_plan_40.json"
    lib_path = ROOT / "third_party/llama.cpp/build/bin/libggml-base.dylib"
    out_gguf = Path("models/inkling2-26b-flashpin-1p-hot40-q2k-i8mm.gguf")

    t0 = time.time()
    url = hf_hub_url(M.BF16_REPO, M.BF16_FILE)
    data_off, remote = M.parse_remote_gguf_header(url)
    im = M.load_imatrix_norm(im_path)
    q = M.GGMLQuantizer(lib_path)

    plan = json.loads(plan_path.read_text())
    perms = M.build_layer_perms(plan["hot_experts"], n_expert=128)

    reader = gguf.GGUFReader(str(src_gguf), "r")
    arch = reader.fields["general.architecture"].contents()
    out_tmp = out_gguf.with_suffix(".gguf.tmp")
    writer = gguf.GGUFWriter(str(out_tmp), arch, endianess=reader.endianess)
    M.copy_kv(reader, writer)

    pat_down = re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$")
    pat_rtr = re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$")

    tensor_list = list(reader.tensors)
    plan_ops: list[tuple[object, str, int | None]] = []
    for t in tensor_list:
        name = t.name
        if name == "blk.23.ffn_gate_up_exps.weight":
            qt = int(gguf.GGMLQuantizationType.Q4_K)
            row_sz = int(q.lib.ggml_row_size(qt, 2304))
            shape = (128, 2304, row_sz)
            writer.add_tensor_info(name, shape, np.uint8, 128 * 2304 * row_sz, raw_dtype=gguf.GGMLQuantizationType.Q4_K)
            plan_ops.append((t, "bf16_exp", qt))
        elif (m := pat_down.match(name)) and int(m.group(1)) in Q4_0_DOWN_LAYERS:
            qt = int(gguf.GGMLQuantizationType.Q4_0)
            row_sz = int(q.lib.ggml_row_size(qt, 1152))
            shape = (128, 2304, row_sz)
            writer.add_tensor_info(name, shape, np.uint8, 128 * 2304 * row_sz, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
            plan_ops.append((t, "bf16_exp", qt))
        elif pat_rtr.match(name):
            qt = int(gguf.GGMLQuantizationType.Q8_0)
            row_sz = int(q.lib.ggml_row_size(qt, 2304))
            shape = (128, row_sz)
            writer.add_tensor_info(name, shape, np.uint8, 128 * row_sz, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            plan_ops.append((t, "rtr_q8", qt))
        else:
            raw = np.asarray(t.data)
            writer.add_tensor_info(name, raw.shape, raw.dtype, raw.nbytes, raw_dtype=t.tensor_type)
            plan_ops.append((t, "copy", None))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    def fetch_and_quant_expert(url_s: str, off_b: int, n_rows: int, n_cols: int, qt: int, im_vec: np.ndarray | None) -> np.ndarray:
        raw_bf16 = M.fetch_range(url_s, off_b, n_rows * n_cols * 2)
        f32 = M.bf16_bytes_to_f32(raw_bf16, (n_rows, n_cols))
        return q.quantize_2d(f32, qt, im_vec)

    # Pre-init quantizers on main thread before parallel calls
    for qt_init in (int(gguf.GGMLQuantizationType.Q4_K), int(gguf.GGMLQuantizationType.Q4_0), int(gguf.GGMLQuantizationType.Q8_0)):
        q.lib.ggml_quantize_init(qt_init)
        q._inited.add(qt_init)

    n_repacked_exp = 0
    n_rtr_q8 = 0
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        for t, kind, qt in plan_ops:
            name = t.name
            if kind == "copy":
                writer.write_tensor_data(np.ascontiguousarray(np.asarray(t.data)))
            elif kind == "rtr_q8":
                w = np.asarray(t.data, dtype=np.float32)  # already merged + REAP-40 reordered!
                q_arr = q.quantize_2d(w, qt, None)
                writer.write_tensor_data(np.ascontiguousarray(q_arr))
                n_rtr_q8 += 1
            elif kind == "bf16_exp":
                il = int(name.split(".")[1])
                dims, rtype, toff = remote[name]
                assert rtype == int(gguf.GGMLQuantizationType.BF16), f"unexpected {rtype} for {name}"
                ne0, ne1, ne2 = dims[0], dims[1], dims[2]  # [in, out, 128]
                assert ne2 == 128
                exp_bytes = ne0 * ne1 * 2
                im_vec = im.get(name)
                if im_vec is not None and im_vec.size != ne0:
                    im_vec = None
                perm = perms[il]  # perm[new_idx] = orig_idx
                t1 = time.time()
                futs = [
                    pool.submit(fetch_and_quant_expert, url, data_off + toff + int(orig_idx) * exp_bytes, ne1, ne0, qt, im_vec)
                    for orig_idx in perm
                ]
                slices = [f.result() for f in futs]
                out_arr = np.ascontiguousarray(np.stack(slices, axis=0))
                writer.write_tensor_data(out_arr)
                n_repacked_exp += 1
                print(
                    f"  [{n_repacked_exp:2d}/10] {name:32s} -> {gguf.GGMLQuantizationType(qt).name:4s} "
                    f"({out_arr.nbytes/1e6:5.1f} MB) in {time.time()-t1:.1f}s",
                    flush=True,
                )
    finally:
        pool.shutdown(wait=True)

    writer.close()
    out_tmp.replace(out_gguf)
    print(
        f"DONE: {out_gguf} ({out_gguf.stat().st_size/1e9:.2f} GB) | "
        f"repacked_exp={n_repacked_exp} rtr_q8={n_rtr_q8} | total={time.time()-t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
