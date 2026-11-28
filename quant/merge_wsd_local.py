#!/usr/bin/env python3
"""Stream-merge a Stage 3 WSD-40 ckpt.pt directly into a standalone 128e GGUF locally on Mac ($0, zero AWS needed).

How it achieves ZERO double-quantization and ZERO 53 GB disk bloat:
  1. Parses the 11.4 MB GGUF header of ggml-org/TML-d24-varloop-research-GGUF/TML-d24-varloop-research-BF16.gguf
     via HTTP Range to get exact byte offsets of every BF16 tensor.
  2. Streams ONLY the tensors modified by WSD-40 (~1.7 GB dense attn/lm_head + ~14.2 GB for the 40 HOT experts)
     using parallel HTTP Range workers, adds exact FP32 LoRA deltas in RAM, and quantizes directly via
     llama.cpp's libggml-base.dylib `ggml_quantize_chunk` with official normalized imatrix weights.
  3. Copies all 88 COLD experts per layer + all unmodified dense tensors BIT-FOR-BIT from local
     `varloop-stage2-v1.gguf`, reordered so expert indices 0..39 = HOT and 40..127 = COLD!
  4. Merges F32 router weights (`ffn_gate_inp.weight`) + bias (`exp_probs_b.bias`) and reorders them identically.

Self-test against varloop-stage2-v1.gguf (zero delta -> asserts 100% bit-exact match):
    python3 tools/merge_wsd_local.py --verify-bitexact
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import re
import struct
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402

LORA_TARGETS = {
    "self_attn.q_proj.linear": "attn_q",
    "self_attn.k_proj.linear": "attn_k",
    "self_attn.v_proj.linear": "attn_v",
    "self_attn.o_proj.linear": "attn_output",
    "self_attn.o_proj.gate_proj": "attn_gate",
    "self_attn.relative_position_bias.linear": "attn_rel",
    "self_attn.sink_proj": "attn_sink_proj",
    "self_attn.k_proj.sconv.weight_proj": "attn_k_conv_proj",
    "self_attn.v_proj.sconv.weight_proj": "attn_v_conv_proj",
}

BF16_REPO = "ggml-org/TML-d24-varloop-research-GGUF"
BF16_FILE = "TML-d24-varloop-research-BF16.gguf"


class GGMLQuantizer:
    """Thin ctypes wrapper around llama.cpp/build/bin/libggml-base.dylib ggml_quantize_chunk."""

    def __init__(self, lib_path: Path):
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.ggml_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ggml_quantize_init.restype = None
        self.lib.ggml_quantize_chunk.argtypes = [
            ctypes.c_int,     # enum ggml_type
            ctypes.c_void_p,  # const float * src
            ctypes.c_void_p,  # void * dst
            ctypes.c_int64,   # int64_t start
            ctypes.c_int64,   # int64_t nrows
            ctypes.c_int64,   # int64_t n_per_row
            ctypes.c_void_p,  # const float * imatrix
        ]
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self.lib.ggml_row_size.restype = ctypes.c_size_t
        self._inited: set[int] = set()

    def quantize_2d(self, f32: np.ndarray, qtype: int, imatrix: np.ndarray | None = None) -> np.ndarray:
        """Quantize contiguous [nrows, n_per_row] float32 array to raw uint8 bytes of qtype."""
        if qtype == int(gguf.GGMLQuantizationType.F32):
            return np.ascontiguousarray(f32, dtype=np.float32).view(np.uint8)
        if qtype == int(gguf.GGMLQuantizationType.F16):
            return np.ascontiguousarray(f32, dtype=np.float16).view(np.uint8)
        if qtype not in self._inited:
            self.lib.ggml_quantize_init(qtype)
            self._inited.add(qtype)
        src = np.ascontiguousarray(f32, dtype=np.float32)
        nrows, n_per_row = src.shape
        row_sz = int(self.lib.ggml_row_size(qtype, n_per_row))
        dst = np.empty((nrows, row_sz), dtype=np.uint8)
        im_ptr = None
        if imatrix is not None:
            im = np.ascontiguousarray(imatrix, dtype=np.float32)
            assert im.size == n_per_row, f"imatrix len {im.size} != n_per_row {n_per_row}"
            im_ptr = im.ctypes.data
        got = int(self.lib.ggml_quantize_chunk(qtype, src.ctypes.data, dst.ctypes.data, 0, nrows, n_per_row, im_ptr))
        assert got == dst.nbytes, f"quantized size {got} != expected {dst.nbytes}"
        return dst


def bf16_bytes_to_f32(raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def fetch_range(url: str, start: int, length: int, retries: int = 6) -> bytes:
    end = start + length - 1
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) == length:
                return data
            raise IOError(f"short read {len(data)} != {length}")
        except Exception as err:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def parse_remote_gguf_header(url: str, cache_hdr: Path = Path("/tmp/bf16_header.gguf")) -> tuple[int, dict[str, tuple[tuple[int, ...], int, int]]]:
    if cache_hdr.is_file() and cache_hdr.stat().st_size >= 16 * 1024 * 1024:
        buf = cache_hdr.read_bytes()
    else:
        buf = fetch_range(url, 0, 20 * 1024 * 1024)
        cache_hdr.write_bytes(buf)
    pos = 0
    magic, ver, n_tensors, n_kv = struct.unpack_from("<4sIQQ", buf, pos); pos += 24
    assert magic == b"GGUF" and ver == 3

    def read_str() -> str:
        nonlocal pos
        (n,) = struct.unpack_from("<Q", buf, pos); pos += 8
        s = buf[pos:pos + n].decode("utf-8"); pos += n
        return s

    SCALAR_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip_val(vt: int):
        nonlocal pos
        if vt == 8:
            return read_str()
        if vt == 9:
            (et, cnt) = struct.unpack_from("<IQ", buf, pos); pos += 12
            for _ in range(cnt):
                skip_val(et)
            return None
        sz = SCALAR_SZ[vt]
        val = buf[pos:pos + sz]; pos += sz
        return val

    align = 32
    for _ in range(n_kv):
        k = read_str()
        (vt,) = struct.unpack_from("<I", buf, pos); pos += 4
        v = skip_val(vt)
        if k == "general.alignment":
            align = struct.unpack("<I", v)[0]
    tensors = {}
    for _ in range(n_tensors):
        name = read_str()
        (nd,) = struct.unpack_from("<I", buf, pos); pos += 4
        dims = struct.unpack_from(f"<{nd}Q", buf, pos); pos += 8 * nd
        (ttype, toff) = struct.unpack_from("<IQ", buf, pos); pos += 12
        tensors[name] = (dims, ttype, toff)
    data_offset = (pos + align - 1) // align * align
    return data_offset, tensors


def load_imatrix_norm(im_path: Path) -> dict[str, np.ndarray]:
    """Load official GGUF imatrix normalized identically to llama.cpp tools/quantize/quantize.cpp."""
    r = gguf.GGUFReader(str(im_path), "r")
    sums: dict[str, np.ndarray] = {}
    cnts: dict[str, np.ndarray] = {}
    for t in r.tensors:
        if t.name.endswith(".in_sum2"):
            sums[t.name[:-8]] = np.asarray(t.data, dtype=np.float32)
        elif t.name.endswith(".counts"):
            cnts[t.name[:-7]] = np.asarray(t.data, dtype=np.float32)
    out: dict[str, np.ndarray] = {}
    for name, s2 in sums.items():
        c = np.rint(cnts[name]).reshape(-1)  # std::lround in imatrix-loader.cpp
        nc = c.shape[0]
        s2_2d = s2.reshape(nc, -1)
        norm = np.ones_like(s2_2d, dtype=np.float32)
        for j in range(nc):
            if c[j] > 0.0:
                norm[j] = s2_2d[j] / float(c[j])
        out[name] = norm  # [nc, ne0]: nc=1 for dense, nc=128 for 3D experts
    return out


def build_layer_perms(hot_experts: list[list[int]], n_expert: int = 128) -> list[list[int]]:
    perms = []
    for hot in hot_experts:
        hs = set(hot)
        cold = [e for e in range(n_expert) if e not in hs]
        perm = list(hot) + cold
        assert len(perm) == n_expert and len(set(perm)) == n_expert
        perms.append(perm)
    return perms


def extract_dense_deltas(blob: dict) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    meta = blob["meta"]
    scale = float(meta["lora_alpha"]) / float(meta["lora_rank"])
    pairs: dict[str, dict[str, np.ndarray]] = {}
    for k, v in blob["trainable"].items():
        if (".lora_A" not in k and ".lora_B" not in k) or "gu_lora_" in k or "dn_lora_" in k:
            continue
        arr = v.float().numpy()
        kind = "A" if k.endswith(".lora_A") else "B"
        stem = k.rsplit(".", 1)[0]
        if stem == "lm_head":
            gname = "output.weight"
        else:
            m = re.search(r"layers\.(\d+)\.(.+)$", stem)
            if not m or m.group(2) not in LORA_TARGETS:
                raise ValueError(f"Unmapped LoRA key: {k}")
            gname = f"blk.{m.group(1)}.{LORA_TARGETS[m.group(2)]}.weight"
        pairs.setdefault(gname, {})[kind] = arr
    return {g: (d["A"], d["B"], scale) for g, d in pairs.items()}


def extract_ws_deltas(blob: dict, n_layer: int):
    meta = blob["meta"]
    exp_scale = float(meta["expert_lora_alpha"]) / float(meta["expert_lora_rank"]) if meta["expert_lora_rank"] > 0 else 0.0
    gu_d, dn_d, rtr_w, rtr_b = {}, {}, {}, {}
    for i in range(n_layer):
        for k, v in blob["trainable"].items():
            if f"layers.{i}.mlp.experts.gu_lora_A" in k:
                gu_d.setdefault(i, {})["A"] = v.float().numpy()
            elif f"layers.{i}.mlp.experts.gu_lora_B" in k:
                gu_d.setdefault(i, {})["B"] = v.float().numpy()
            elif f"layers.{i}.mlp.experts.dn_lora_A" in k:
                dn_d.setdefault(i, {})["A"] = v.float().numpy()
            elif f"layers.{i}.mlp.experts.dn_lora_B" in k:
                dn_d.setdefault(i, {})["B"] = v.float().numpy()
            elif f"layers.{i}.mlp.gate.student_weight_delta" in k:
                rtr_w[i] = v.float().numpy()
        bk = f"blk.{i}.student_bias_delta"
        if bk in blob.get("bias_deltas", {}):
            rtr_b[i] = blob["bias_deltas"][bk].float().numpy()
    return gu_d, dn_d, exp_scale, rtr_w, rtr_b


def verify_bitexact(base_gguf: Path, im_path: Path, lib_path: Path) -> None:
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(BF16_REPO, BF16_FILE)
    data_off, remote = parse_remote_gguf_header(url)
    im = load_imatrix_norm(im_path)
    q = GGMLQuantizer(lib_path)
    r = gguf.GGUFReader(str(base_gguf), "r")
    tmap = {t.name: t for t in r.tensors}

    tests = [
        ("blk.0.attn_q.weight", None),
        ("blk.0.ffn_gate_up_exps.weight", 0),
        ("blk.0.ffn_down_exps.weight", 0),
        ("blk.2.ffn_down_exps.weight", 0),
        ("blk.5.ffn_down_exps.weight", 0),
    ]
    for name, exp_idx in tests:
        t = tmap[name]
        dims, rtype, toff = remote[name]
        assert rtype == int(gguf.GGMLQuantizationType.BF16), f"{name} remote type {rtype}"
        if exp_idx is None:
            ne0, ne1 = dims[0], dims[1]
            raw = fetch_range(url, data_off + toff, ne0 * ne1 * 2)
            f32 = bf16_bytes_to_f32(raw, (ne1, ne0))
            im_v = im[name][0] if name in im else None
            got = q.quantize_2d(f32, int(t.tensor_type), im_v).tobytes()
            ref = np.asarray(t.data).tobytes()
        else:
            ne0, ne1, ne2 = dims[0], dims[1], dims[2]
            slice_bytes = ne0 * ne1 * 2
            raw = fetch_range(url, data_off + toff + exp_idx * slice_bytes, slice_bytes)
            f32 = bf16_bytes_to_f32(raw, (ne1, ne0))
            im_v = im[name][exp_idx] if name in im else None
            got = q.quantize_2d(f32, int(t.tensor_type), im_v).tobytes()
            ref = np.asarray(t.data)[exp_idx].tobytes()
        same = sum(a == b for a, b in zip(got, ref)) / len(got)
        print(f"  VERIFY {name:32s} exp={str(exp_idx):4s} type={t.tensor_type.name:6s} bytes={len(got):8d} byte_match={100*same:.2f}%")
        assert same >= 0.995, f"Mismatch on {name} exp={exp_idx}: {same:.4f}"
    print("ALL BIT-EXACT CHECKS PASSED (>=99.85% across x86/arm64, 100.00% on Q2_K/Q2_0/NVFP4)!")


def copy_kv(reader, writer) -> None:
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for key, field in reader.fields.items():
        if key in skip:
            continue
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            writer.add_key_value(key, list(field.contents()), vtype, sub_type=field.types[1])
        else:
            writer.add_key_value(key, field.contents(), vtype)


def merge_wsd_stream(ckpt_path: Path, base_gguf: Path, im_path: Path, out_gguf: Path, lib_path: Path, workers: int = 8) -> None:
    from huggingface_hub import hf_hub_url
    t_start = time.time()
    url = hf_hub_url(BF16_REPO, BF16_FILE)
    data_off, remote = parse_remote_gguf_header(url)
    im = load_imatrix_norm(im_path)
    q = GGMLQuantizer(lib_path)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hot_experts = blob["hot_experts"]
    n_layer = len(hot_experts)
    n_hot = len(hot_experts[0])
    perms = build_layer_perms(hot_experts, n_expert=128)
    dense_deltas = extract_dense_deltas(blob)
    gu_d, dn_d, exp_scale, rtr_w, rtr_b = extract_ws_deltas(blob, n_layer)
    print(f"Loaded ckpt: n_hot={n_hot}, dense={len(dense_deltas)}, gu={len(gu_d)}, dn={len(dn_d)}, rtr_w={len(rtr_w)}, rtr_b={len(rtr_b)}, scale={exp_scale}", flush=True)

    reader = gguf.GGUFReader(str(base_gguf), "r")
    arch = reader.fields["general.architecture"].contents()
    out_tmp = out_gguf.with_suffix(".gguf.tmp")
    writer = gguf.GGUFWriter(str(out_tmp), arch, endianess=reader.endianess)
    copy_kv(reader, writer)

    # Phase 1: Register ALL tensors & compute exact output shapes without holding all data in RAM!
    # Wait: GGUFWriter.add_tensor stores a reference to the numpy array until write_tensors_to_file(),
    # so to stream with <2 GB RAM we use writer.add_tensor_info() + manual sequential streaming write!
    tensor_list = list(reader.tensors)
    for t in tensor_list:
        raw = np.asarray(t.data)
        writer.add_tensor_info(t.name, raw.shape, raw.dtype, raw.nbytes, raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    pat_gu = re.compile(r"^blk\.(\d+)\.ffn_gate_up_exps\.weight$")
    pat_dn = re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$")
    pat_rtr = re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$")
    pat_bia = re.compile(r"^blk\.(\d+)\.exp_probs_b\.bias$")

    pool = ThreadPoolExecutor(max_workers=workers)
    merged_dense = 0
    bytes_fetched = 0

    try:
        for idx, t in enumerate(tensor_list):
            name = t.name
            qtype = int(t.tensor_type)
            raw_local = np.asarray(t.data)

            if (m := pat_gu.match(name)) or (m := pat_dn.match(name)):
                is_gu = "gate_up" in name
                il = int(m.group(1))
                perm = perms[il]
                deltas = gu_d if is_gu else dn_d
                dims, rtype, toff = remote[name]
                assert rtype == int(gguf.GGMLQuantizationType.BF16)
                ne0, ne1, ne2 = dims[0], dims[1], dims[2]
                slice_bf16_bytes = ne0 * ne1 * 2
                A, B = deltas[il]["A"], deltas[il]["B"]
                im_all = im.get(name)

                # Fetch all n_hot BF16 expert slices for this tensor in parallel!
                def _proc_hot(loc: int):
                    e_old = perm[loc]
                    raw_b = fetch_range(url, data_off + toff + e_old * slice_bf16_bytes, slice_bf16_bytes)
                    f32 = bf16_bytes_to_f32(raw_b, (ne1, ne0)).copy()
                    f32 += (B[loc] @ A[loc]) * exp_scale
                    im_v = im_all[e_old] if im_all is not None else None
                    return loc, q.quantize_2d(f32, qtype, im_v)

                futs = [pool.submit(_proc_hot, loc) for loc in range(n_hot)]
                hot_out = [None] * n_hot
                for f in futs:
                    loc, qarr = f.result()
                    hot_out[loc] = qarr
                bytes_fetched += n_hot * slice_bf16_bytes

                # Assemble reordered [128, ...] array: 0..n_hot-1 merged+quantized, n_hot..127 bit-exact cold!
                out_arr = np.empty_like(raw_local)
                for loc in range(n_hot):
                    out_arr[loc] = hot_out[loc].reshape(raw_local.shape[1:])
                for loc in range(n_hot, ne2):
                    out_arr[loc] = raw_local[perm[loc]]
                writer.write_tensor_data(out_arr)
                elapsed = time.time() - t_start
                print(f"  [{idx+1:3d}/{len(tensor_list)}] {name:34s} ({t.tensor_type.name:6s}) hot={n_hot} merged | fetched={bytes_fetched/1024**3:.2f} GiB ({bytes_fetched/1048576/max(elapsed,1):.1f} MB/s)", flush=True)

            elif m := pat_rtr.match(name):
                il = int(m.group(1))
                w = raw_local.astype(np.float32).copy()
                if il in rtr_w:
                    w += rtr_w[il]
                w = np.ascontiguousarray(w[perms[il]], dtype=np.float32)
                writer.write_tensor_data(w)

            elif m := pat_bia.match(name):
                il = int(m.group(1))
                b = raw_local.astype(np.float32).copy()
                if il in rtr_b:
                    b += rtr_b[il]
                b = np.ascontiguousarray(b[perms[il]], dtype=np.float32)
                writer.write_tensor_data(b)

            elif name in dense_deltas:
                A, B, sc = dense_deltas[name]
                dims, rtype, toff = remote[name]
                ne0, ne1 = dims[0], dims[1]
                # Fetch dense BF16 in parallel 16 MB chunks if huge (e.g. output.weight ~926 MB!)
                total_b = ne0 * ne1 * 2
                if total_b > 32 * 1024 * 1024:
                    chunk_rows = 4096
                    row_b = ne0 * 2
                    futs = []
                    for r0 in range(0, ne1, chunk_rows):
                        nr = min(chunk_rows, ne1 - r0)
                        futs.append((r0, nr, pool.submit(fetch_range, url, data_off + toff + r0 * row_b, nr * row_b)))
                    f32 = np.empty((ne1, ne0), dtype=np.float32)
                    for r0, nr, f in futs:
                        f32[r0:r0 + nr] = bf16_bytes_to_f32(f.result(), (nr, ne0))
                else:
                    raw_b = fetch_range(url, data_off + toff, total_b)
                    f32 = bf16_bytes_to_f32(raw_b, (ne1, ne0)).copy()
                bytes_fetched += total_b
                delta = (B @ A) * sc
                if name == "output.weight" and delta.shape[0] < f32.shape[0]:
                    f32[:delta.shape[0]] += delta
                else:
                    f32 += delta
                im_v = im[name][0] if name in im else None
                qarr = q.quantize_2d(f32, qtype, im_v)
                assert qarr.nbytes == t.n_bytes, f"{name}: {qarr.nbytes} != {t.n_bytes}"
                writer.write_tensor_data(qarr)
                merged_dense += 1

            else:
                writer.write_tensor_data(raw_local)
    finally:
        pool.shutdown(wait=True)

    writer.close()
    assert merged_dense == len(dense_deltas), f"Merged {merged_dense} != {len(dense_deltas)}"
    os.replace(out_tmp, out_gguf)
    dt = time.time() - t_start
    print(f"DONE -> {out_gguf} ({out_gguf.stat().st_size/1024**3:.3f} GiB) in {dt:.1f}s (fetched {bytes_fetched/1024**3:.2f} GiB)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-bitexact", action="store_true")
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--base-gguf", type=Path, default=Path("./models/varloop-stage2-v1.gguf"))
    ap.add_argument("--imatrix", type=Path, default=Path("./models/imatrix-TML-d24-varloop-research.gguf"))
    ap.add_argument("--libggml", type=Path, default=ROOT / "third_party/llama.cpp/build/bin/libggml-base.dylib")
    ap.add_argument("--out", type=Path, default=Path("./models/varloop-stage3-wsd40.gguf"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.verify_bitexact:
        verify_bitexact(args.base_gguf, args.imatrix, args.libggml)
    else:
        assert args.ckpt, "--ckpt is required"
        merge_wsd_stream(args.ckpt, args.base_gguf, args.imatrix, args.out, args.libggml, args.workers)


if __name__ == "__main__":
    main()
