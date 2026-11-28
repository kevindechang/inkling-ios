#!/usr/bin/env python3
"""One-pass merge of a WSD-40 v2 QAT checkpoint into an i8mm GGUF.

The base file (varloop-stage2-v1-run3-reap40-i8mm.gguf) is already expert-reordered
and already has run3 inside the dense attention tensors. This merge:

* copies cold expert slices (positions ``n_hot .. 127``) bit-exact from that base
* fetches clean HF BF16 for the 40 hot experts, the dense LoRA targets, and the routers
* adds the v2 delta once, then quantizes once into the base tensor's own type
* adds F32 small-dense and router-bias deltas onto the base F32 values

``--dry-run-map-check`` only reads the base header. It does not fetch BF16 and does not write a GGUF.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import merge_wsd_local as M  # noqa: E402

H = 2304
INTER = 1152
N_EXP = 128
N_HOT = 40
N_LAYER = 24
RANK = 32
ERANK = 24
UNPAD = 200058
HEADS = 24
KV_HEADS = 8
HEAD_DIM = 128
D_REL = 16
CONV_K = 4
FULL_LAYERS = {5, 11, 17, 23}
PREFIX = "model.language_model"

DENSE_STEMS = set(M.LORA_TARGETS.values())
SMALL_ATTR = {
    "attention_input": "attn_res_query",
    "mlp_input": "ffn_res_query",
    "input_layernorm": "attn_norm",
    "q_input_layernorm": "attn_q_inp_norm",
    "k_input_layernorm": "attn_k_inp_norm",
    "v_input_layernorm": "attn_v_inp_norm",
    "post_attention_layernorm": "ffn_norm",
}
EXPERT_TYPES = {"Q2_K", "Q4_K", "Q2_0", "Q4_0", "IQ4_NL"}

DEFAULT_BASE = Path(os.environ.get("INKLING_V1_GGUF", "models/inkling2-26b-flashpin-1p-hot40-q2k-i8mm.gguf"))
DEFAULT_HOT = HERE.parent / "train" / "hot_plan_40.json"
DEFAULT_OUT = Path("models/inkling2-26b-flashpin-v2-1p-hot40-q2k-i8mm.gguf")


def torch_shape(tensor) -> tuple[int, ...]:
    return tuple(int(x) for x in reversed(tensor.shape))


def dense_out_in(stem: str) -> tuple[int, int]:
    q = HEADS * HEAD_DIM
    kv = KV_HEADS * HEAD_DIM
    rel = D_REL * HEADS
    conv = KV_HEADS * CONV_K
    table = {
        "attn_q": (q, H),
        "attn_k": (kv, H),
        "attn_v": (kv, H),
        "attn_output": (H, q),
        "attn_gate": (q, H),
        "attn_rel": (rel, H),
        "attn_sink_proj": (HEADS, H),
        "attn_k_conv_proj": (conv, H),
        "attn_v_conv_proj": (conv, H),
    }
    return table[stem]


def small_gguf_name(key: str) -> str | None:
    if not key.endswith(".student_delta") or key.endswith(".student_weight_delta"):
        return None
    layer = re.search(r"layers\.(\d+)\.([^.]+)\.student_delta$", key)
    if layer:
        attr = layer.group(2)
        if attr not in SMALL_ATTR:
            raise ValueError(f"unmapped small-dense key {key}")
        return f"blk.{layer.group(1)}.{SMALL_ATTR[attr]}.weight"
    if key.endswith(".final_input.student_delta") or key == "final_input.student_delta":
        return "output_res_query.weight"
    if key.endswith(".norm.student_delta") or key == "norm.student_delta":
        return "output_norm.weight"
    raise ValueError(f"unmapped small-dense key {key}")


def dense_gguf_name(stem_key: str) -> str:
    if stem_key == "lm_head" or stem_key.endswith(".lm_head"):
        return "output.weight"
    layer = re.search(r"layers\.(\d+)\.(.+)$", stem_key)
    if layer is None or layer.group(2) not in M.LORA_TARGETS:
        raise ValueError(f"unmapped dense LoRA {stem_key}")
    return f"blk.{layer.group(1)}.{M.LORA_TARGETS[layer.group(2)]}.weight"


def _empty(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.float32)


def synthesize_probe(path: Path, hot_plan: Path) -> None:
    """Full-scale keys and shapes from the architecture constants, not from a GGUF read."""
    plan = json.loads(hot_plan.read_text())
    hot = plan["hot_experts"]
    if len(hot) != N_LAYER or any(len(row) != N_HOT for row in hot):
        raise SystemExit(f"hot plan {hot_plan} is not {N_LAYER}x{N_HOT}")
    trainable: dict[str, torch.Tensor] = {}
    suffix_of = {stem: suffix for suffix, stem in M.LORA_TARGETS.items()}
    for i in range(N_LAYER):
        layer = f"{PREFIX}.layers.{i}"
        for stem, suffix in suffix_of.items():
            if stem in ("attn_k_conv_proj", "attn_v_conv_proj") and i not in FULL_LAYERS:
                continue
            out, inn = dense_out_in(stem)
            trainable[f"{layer}.{suffix}.lora_A"] = _empty((RANK, inn))
            trainable[f"{layer}.{suffix}.lora_B"] = _empty((out, RANK))
        trainable[f"{layer}.mlp.experts.gu_lora_A"] = _empty((N_HOT, ERANK, H))
        trainable[f"{layer}.mlp.experts.gu_lora_B"] = _empty((N_HOT, 2 * INTER, ERANK))
        trainable[f"{layer}.mlp.experts.dn_lora_A"] = _empty((N_HOT, ERANK, INTER))
        trainable[f"{layer}.mlp.experts.dn_lora_B"] = _empty((N_HOT, H, ERANK))
        trainable[f"{layer}.mlp.gate.student_weight_delta"] = _empty((N_EXP, H))
        if i > 0:
            qshape = (3, H) if i in FULL_LAYERS else (H,)
            trainable[f"{layer}.attention_input.student_delta"] = _empty(qshape)
        trainable[f"{layer}.mlp_input.student_delta"] = _empty((H,))
        if i in FULL_LAYERS:
            for attr in ("q_input_layernorm", "k_input_layernorm", "v_input_layernorm"):
                trainable[f"{layer}.{attr}.student_delta"] = _empty((H,))
        else:
            trainable[f"{layer}.input_layernorm.student_delta"] = _empty((H,))
        trainable[f"{layer}.post_attention_layernorm.student_delta"] = _empty((H,))
    trainable[f"{PREFIX}.final_input.student_delta"] = _empty((H,))
    trainable[f"{PREFIX}.norm.student_delta"] = _empty((H,))
    trainable["lm_head.lora_A"] = _empty((RANK, H))
    trainable["lm_head.lora_B"] = _empty((UNPAD, RANK))
    bias = {f"blk.{i}.student_bias_delta": _empty((N_EXP,)) for i in range(N_LAYER)}
    blob = {
        "trainable": trainable,
        "bias_deltas": bias,
        "hot_experts": hot,
        "meta": {
            "mode": "workingset_v2_qat",
            "n_hot": N_HOT,
            "lora_rank": RANK,
            "lora_alpha": float(RANK),
            "expert_lora_rank": ERANK,
            "expert_lora_alpha": float(ERANK * 2),
            "train_small_dense_deltas": True,
            "lora_lm_head": True,
            "gguf_experts_reordered": True,
        },
        "state": {"step": 0, "seqs": 0, "hours": 0.0},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    n_small = sum(1 for k in trainable if k.endswith(".student_delta") and not k.endswith(".student_weight_delta"))
    n_router = sum(1 for k in trainable if k.endswith(".student_weight_delta"))
    n_expert = sum(1 for k in trainable if k.endswith(("gu_lora_A", "dn_lora_A")))
    n_dense = sum(1 for k in trainable if k.endswith(".lora_A") and "lm_head" not in k and "experts" not in k)
    print(
        f"synthesized {path} ({path.stat().st_size / 1024**2:.1f} MiB) "
        f"dense_attn_pairs={n_dense} lm_head=1 expert_pairs={n_expert} routers={n_router} "
        f"biases={len(bias)} small_dense={n_small} trainable_keys={len(trainable)}",
        flush=True,
    )


def _discover(reader) -> dict[str, set[str]]:
    found = {"dense": set(), "expert": set(), "router": set(), "bias": set(), "small": set(), "lm_head": set()}
    for t in reader.tensors:
        name = t.name
        layer = re.match(r"^blk\.(\d+)\.([^.]+)\.(weight|bias)$", name)
        if layer:
            stem, kind = layer.group(2), layer.group(3)
            if kind == "weight" and stem in DENSE_STEMS:
                found["dense"].add(name)
            elif stem in ("ffn_gate_up_exps", "ffn_down_exps") and kind == "weight":
                found["expert"].add(name)
            elif stem == "ffn_gate_inp" and kind == "weight":
                found["router"].add(name)
            elif stem == "exp_probs_b" and kind == "bias":
                found["bias"].add(name)
            elif kind == "weight" and stem in set(SMALL_ATTR.values()):
                found["small"].add(name)
        elif name == "output.weight":
            found["lm_head"].add(name)
        elif name in ("output_res_query.weight", "output_norm.weight"):
            found["small"].add(name)
    return found


def _check_pair(gname: str, A: torch.Tensor, B: torch.Tensor, gguf_t, rank: int, prefix_rows: int | None) -> None:
    if tuple(A.shape)[:1] != (rank,) or B.shape[-1] != rank:
        raise AssertionError(f"{gname}: rank {tuple(A.shape)}/{tuple(B.shape)} != {rank}")
    out, inn = int(B.shape[0]), int(A.shape[1])
    got = torch_shape(gguf_t)
    if prefix_rows is not None:
        if out != prefix_rows or inn != got[1] or got[0] <= out:
            raise AssertionError(f"{gname}: lora {(out, inn)} vs GGUF {got} (prefix {prefix_rows})")
        return
    if (out, inn) != got:
        raise AssertionError(f"{gname}: lora weight {(out, inn)} != GGUF torch {got}")


def dry_run_map_check(ckpt_path: Path, base_gguf: Path, hot_plan: Path) -> None:
    """Every trainable key maps onto one base-GGUF tensor, and the shapes agree."""
    before = base_gguf.stat()
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = blob["meta"]
    if meta.get("mode") != "workingset_v2_qat":
        raise AssertionError(f"meta.mode={meta.get('mode')}")
    if int(meta["lora_rank"]) != RANK or float(meta["lora_alpha"]) != float(RANK):
        raise AssertionError(f"dense lora meta {meta['lora_rank']}/{meta['lora_alpha']}")
    if int(meta["expert_lora_rank"]) != ERANK or float(meta["expert_lora_alpha"]) != float(ERANK * 2):
        raise AssertionError(f"expert lora meta {meta['expert_lora_rank']}/{meta['expert_lora_alpha']}")
    plan_hot = json.loads(hot_plan.read_text())["hot_experts"]
    if blob["hot_experts"] != plan_hot:
        raise AssertionError("ckpt hot_experts != hot_plan_40.json")
    reader = M.gguf.GGUFReader(str(base_gguf), "r")
    tmap = {t.name: t for t in reader.tensors}
    discovered = _discover(reader)
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    routers, small, experts = {}, {}, {}
    unknown = []
    for key, value in blob["trainable"].items():
        if key.endswith(".student_weight_delta"):
            m = re.search(r"layers\.(\d+)\.mlp\.gate\.student_weight_delta$", key)
            if m is None:
                unknown.append(key)
                continue
            routers[f"blk.{m.group(1)}.ffn_gate_inp.weight"] = value
        elif key.endswith(".student_delta"):
            small[small_gguf_name(key)] = value
        elif re.search(r"\.(gu|dn)_lora_[AB]$", key):
            m = re.search(r"layers\.(\d+)\.mlp\.experts\.(gu|dn)_lora_([AB])$", key)
            if m is None:
                unknown.append(key)
                continue
            stem = "ffn_gate_up_exps" if m.group(2) == "gu" else "ffn_down_exps"
            experts.setdefault(f"blk.{m.group(1)}.{stem}.weight", {})[m.group(3)] = value
        elif key.endswith(".lora_A") or key.endswith(".lora_B"):
            kind = "A" if key.endswith(".lora_A") else "B"
            stem = key[: -len(".lora_A")] if kind == "A" else key[: -len(".lora_B")]
            pairs.setdefault(dense_gguf_name(stem), {})[kind] = value
        else:
            unknown.append(key)
    if unknown:
        raise AssertionError(f"{len(unknown)} unmapped keys, first={unknown[0]}")
    biases = {}
    for key, value in blob.get("bias_deltas", {}).items():
        m = re.fullmatch(r"blk\.(\d+)\.student_bias_delta", key)
        if m is None:
            raise AssertionError(f"unmapped bias {key}")
        biases[f"blk.{m.group(1)}.exp_probs_b.bias"] = value

    dense_names = {name for name in pairs if name != "output.weight"}
    if dense_names != discovered["dense"]:
        missing = sorted(discovered["dense"] - dense_names)[:5]
        extra = sorted(dense_names - discovered["dense"])[:5]
        raise AssertionError(f"dense set mismatch missing={missing} extra={extra} ckpt={len(dense_names)} gguf={len(discovered['dense'])}")
    if set(experts) != discovered["expert"]:
        raise AssertionError(f"expert set ckpt={len(experts)} gguf={len(discovered['expert'])}")
    if set(routers) != discovered["router"] or set(biases) != discovered["bias"] or set(small) != discovered["small"]:
        raise AssertionError(
            f"set mismatch router {len(routers)}/{len(discovered['router'])} "
            f"bias {len(biases)}/{len(discovered['bias'])} small {len(small)}/{len(discovered['small'])}"
        )
    if set(pairs) & {"output.weight"} != discovered["lm_head"]:
        raise AssertionError(f"lm_head set {set(pairs) & {'output.weight'}} vs {discovered['lm_head']}")

    for name, pair in pairs.items():
        if "A" not in pair or "B" not in pair:
            raise AssertionError(f"{name} missing A/B")
        prefix = UNPAD if name == "output.weight" else None
        _check_pair(name, pair["A"], pair["B"], tmap[name], RANK, prefix)
    for name, pair in experts.items():
        A, B = pair["A"], pair["B"]
        if tuple(A.shape) != (N_HOT, ERANK, A.shape[-1]) or tuple(B.shape) != (N_HOT, B.shape[1], ERANK):
            raise AssertionError(f"{name}: expert lora {tuple(A.shape)} {tuple(B.shape)}")
        got = torch_shape(tmap[name])
        if got[0] != N_EXP or got[1:] != (int(B.shape[1]), int(A.shape[2])):
            raise AssertionError(f"{name}: slice {(int(B.shape[1]), int(A.shape[2]))} vs GGUF {got}")
        if tmap[name].tensor_type.name not in EXPERT_TYPES:
            raise AssertionError(f"{name} type {tmap[name].tensor_type.name}")
    for name, value in routers.items():
        got = torch_shape(tmap[name])
        if tuple(value.shape) != got:
            raise AssertionError(f"{name}: {tuple(value.shape)} != {got}")
        if tmap[name].tensor_type.name != "Q8_0":
            raise AssertionError(f"{name} type {tmap[name].tensor_type.name}")
    for name, value in biases.items():
        got = torch_shape(tmap[name])
        if tuple(value.shape) != got or tmap[name].tensor_type.name != "F32":
            raise AssertionError(f"{name}: {tuple(value.shape)} vs {got} {tmap[name].tensor_type.name}")
    for name, value in small.items():
        got = torch_shape(tmap[name])
        if tuple(value.shape) != got or tmap[name].tensor_type.name != "F32":
            raise AssertionError(f"{name}: {tuple(value.shape)} vs {got} {tmap[name].tensor_type.name}")

    after = base_gguf.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise AssertionError("base GGUF was modified during the map check")
    counts = (
        f"dense_attn={len(dense_names)} lm_head=1 expert_pairs={len(experts)} "
        f"routers={len(routers)} biases={len(biases)} small_dense={len(small)}"
    )
    if (len(dense_names), len(experts), len(routers), len(biases)) != (176, 48, 24, 24) or len(small) != 105:
        raise AssertionError(f"unexpected counts {counts}")
    print(f"MAP CHECK PASSED {counts} base={base_gguf} bytes={after.st_size}", flush=True)


def _remote_f32(url: str, data_off: int, remote: dict, name: str, pool=None) -> np.ndarray:
    dims, rtype, toff = remote[name]
    ne = [int(x) for x in dims]
    n_elem = 1
    for d in ne:
        n_elem *= d
    if rtype == int(M.gguf.GGMLQuantizationType.BF16):
        total_b = n_elem * 2
        if pool is not None and len(ne) == 2 and total_b > 32 * 1024 * 1024:
            ne0, ne1 = ne[0], ne[1]
            chunk_rows = 4096
            row_b = ne0 * 2
            futs = [
                (r0, min(chunk_rows, ne1 - r0), pool.submit(M.fetch_range, url, data_off + toff + r0 * row_b, min(chunk_rows, ne1 - r0) * row_b))
                for r0 in range(0, ne1, chunk_rows)
            ]
            f32 = np.empty((ne1, ne0), dtype=np.float32)
            for r0, nr, fut in futs:
                f32[r0:r0 + nr] = M.bf16_bytes_to_f32(fut.result(), (nr, ne0))
            return f32
        raw = M.fetch_range(url, data_off + toff, total_b)
        return M.bf16_bytes_to_f32(raw, tuple(reversed(ne))).copy()
    if rtype == int(M.gguf.GGMLQuantizationType.F32):
        raw = M.fetch_range(url, data_off + toff, n_elem * 4)
        return np.frombuffer(raw, dtype=np.float32).reshape(tuple(reversed(ne))).copy()
    raise RuntimeError(f"{name}: remote type {rtype} is not BF16/F32")


def _add_delta(f32: np.ndarray, delta: np.ndarray) -> np.ndarray:
    if delta.shape == f32.shape:
        f32 += delta
        return f32
    if delta.ndim == 2 and f32.ndim == 2 and delta.shape[0] < f32.shape[0] and delta.shape[1] == f32.shape[1]:
        f32[: delta.shape[0]] += delta
        return f32
    raise ValueError(f"delta {delta.shape} does not match {f32.shape}")


def merge_v2(ckpt_path: Path, base_gguf: Path, im_path: Path, out_gguf: Path, lib_path: Path, workers: int) -> None:
    if out_gguf.resolve() == base_gguf.resolve():
        raise SystemExit("refusing to overwrite --base-gguf")
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_url

    t_start = time.time()
    url = hf_hub_url(M.BF16_REPO, M.BF16_FILE)
    data_off, remote = M.parse_remote_gguf_header(url)
    im = M.load_imatrix_norm(im_path)
    quant = M.GGMLQuantizer(lib_path)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hot_experts = blob["hot_experts"]
    n_layer = len(hot_experts)
    n_hot = len(hot_experts[0])
    perms = M.build_layer_perms(hot_experts, n_expert=N_EXP)
    dense = M.extract_dense_deltas(blob)
    gu_d, dn_d, exp_scale, rtr_w, rtr_b = M.extract_ws_deltas(blob, n_layer)
    small = {}
    for key, value in blob["trainable"].items():
        if key.endswith(".student_delta") and not key.endswith(".student_weight_delta"):
            small[small_gguf_name(key)] = value.detach().float().numpy()
    print(
        f"ckpt n_hot={n_hot} dense={len(dense)} gu={len(gu_d)} dn={len(dn_d)} "
        f"rtr_w={len(rtr_w)} rtr_b={len(rtr_b)} small={len(small)} exp_scale={exp_scale}",
        flush=True,
    )

    reader = M.gguf.GGUFReader(str(base_gguf), "r")
    arch = reader.fields["general.architecture"].contents()
    out_tmp = out_gguf.with_suffix(".gguf.tmp")
    writer = M.gguf.GGUFWriter(str(out_tmp), arch, endianess=reader.endianess)
    M.copy_kv(reader, writer)
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
    # Pre-init all GGML quantizers on main thread before spawning parallel ThreadPoolExecutor workers.
    for qt_init in sorted({int(t.tensor_type) for t in tensor_list}):
        if qt_init not in (int(M.gguf.GGMLQuantizationType.F32), int(M.gguf.GGMLQuantizationType.F16)):
            quant.lib.ggml_quantize_init(qt_init)
            quant._inited.add(qt_init)
    pool = ThreadPoolExecutor(max_workers=workers)
    merged_dense = 0
    try:
        for t in tensor_list:
            name = t.name
            qtype = int(t.tensor_type)
            raw_local = np.asarray(t.data)
            if (m := pat_gu.match(name)) or (m := pat_dn.match(name)):
                il = int(m.group(1))
                is_gu = "gate_up" in name
                deltas = gu_d[il] if is_gu else dn_d[il]
                A, B = deltas["A"], deltas["B"]
                dims, rtype, toff = remote[name]
                if rtype != int(M.gguf.GGMLQuantizationType.BF16):
                    raise RuntimeError(f"{name} remote type {rtype}")
                ne0, ne1, ne2 = int(dims[0]), int(dims[1]), int(dims[2])
                slice_bytes = ne0 * ne1 * 2
                im_all = im.get(name)

                def _hot(loc: int, il=il, toff=toff, slice_bytes=slice_bytes, ne0=ne0, ne1=ne1, A=A, B=B, im_all=im_all, qtype=qtype):
                    orig = int(hot_experts[il][loc])
                    raw_b = M.fetch_range(url, data_off + toff + orig * slice_bytes, slice_bytes)
                    f32 = M.bf16_bytes_to_f32(raw_b, (ne1, ne0)).copy()
                    f32 += (B[loc] @ A[loc]) * exp_scale
                    im_v = im_all[orig] if im_all is not None else None
                    return loc, quant.quantize_2d(f32, qtype, im_v)

                hot_out = [None] * n_hot
                for fut in [pool.submit(_hot, loc) for loc in range(n_hot)]:
                    loc, qarr = fut.result()
                    hot_out[loc] = qarr
                # Base experts are already hot-then-cold. Do not permute the cold tail again.
                out_arr = np.empty_like(raw_local)
                for loc in range(n_hot):
                    out_arr[loc] = hot_out[loc].reshape(raw_local.shape[1:])
                for loc in range(n_hot, ne2):
                    out_arr[loc] = raw_local[loc]
                writer.write_tensor_data(out_arr)
                print(f"  {name} hot={n_hot} cold={ne2 - n_hot} type={t.tensor_type.name}", flush=True)
            elif m := pat_rtr.match(name):
                il = int(m.group(1))
                f32 = _remote_f32(url, data_off, remote, name)
                if il in rtr_w:
                    f32 = _add_delta(f32, rtr_w[il])
                f32 = np.ascontiguousarray(f32[perms[il]], dtype=np.float32)
                qarr = quant.quantize_2d(f32, qtype, None)
                if qarr.nbytes != t.n_bytes:
                    raise RuntimeError(f"{name}: {qarr.nbytes} != {t.n_bytes}")
                writer.write_tensor_data(qarr)
            elif m := pat_bia.match(name):
                il = int(m.group(1))
                if name in remote:
                    f32 = _remote_f32(url, data_off, remote, name).reshape(-1)
                else:
                    base = raw_local.astype(np.float32).reshape(-1)
                    f32 = np.empty_like(base)
                    for loc, orig in enumerate(perms[il]):
                        f32[orig] = base[loc]
                if il in rtr_b:
                    f32 = f32 + rtr_b[il].reshape(-1)
                f32 = np.ascontiguousarray(f32[perms[il]], dtype=np.float32)
                if f32.shape != raw_local.shape:
                    raise RuntimeError(f"{name}: {f32.shape} != {raw_local.shape}")
                writer.write_tensor_data(f32)
            elif name in small:
                if t.tensor_type.name != "F32":
                    raise RuntimeError(f"{name} base type {t.tensor_type.name} is not F32")
                base = np.array(raw_local, dtype=np.float32, copy=True)
                delta = small[name]
                if base.shape != delta.shape:
                    raise RuntimeError(f"{name}: base {base.shape} != delta {delta.shape}")
                writer.write_tensor_data(np.ascontiguousarray(base + delta, dtype=np.float32))
            elif name in dense:
                A, B, scale = dense[name]
                f32 = _remote_f32(url, data_off, remote, name, pool=pool)
                f32 = _add_delta(f32, (B @ A) * scale)
                im_v = im[name][0] if name in im else None
                qarr = quant.quantize_2d(f32, qtype, im_v)
                if qarr.nbytes != t.n_bytes:
                    raise RuntimeError(f"{name}: {qarr.nbytes} != {t.n_bytes}")
                writer.write_tensor_data(qarr)
                merged_dense += 1
                if name == "output.weight" or merged_dense % 44 == 0:
                    print(f"  merged dense {merged_dense}/{len(dense)} ({name}) in {time.time() - t_start:.1f}s", flush=True)
            else:
                writer.write_tensor_data(raw_local)
    finally:
        pool.shutdown(wait=True)
    writer.close()
    if merged_dense != len(dense):
        raise RuntimeError(f"merged {merged_dense} dense tensors, expected {len(dense)}")
    os.replace(out_tmp, out_gguf)
    print(f"DONE -> {out_gguf} in {time.time() - t_start:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--base-gguf", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--imatrix", type=Path, default=Path("models/imatrix-TML-d24-varloop-research.gguf"))
    ap.add_argument("--libggml", type=Path, default=HERE.parent / "third_party/llama.cpp/build/bin/libggml-base.dylib")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hot-plan", type=Path, default=DEFAULT_HOT)
    ap.add_argument("--synthesize-probe", type=Path, default=None)
    ap.add_argument("--dry-run-map-check", type=Path, default=None)
    args = ap.parse_args()
    if args.synthesize_probe:
        synthesize_probe(args.synthesize_probe, args.hot_plan)
    if args.dry_run_map_check:
        dry_run_map_check(args.dry_run_map_check, args.base_gguf, args.hot_plan)
        return
    if args.synthesize_probe and not args.ckpt:
        return
    if not args.ckpt:
        ap.error("--ckpt is required unless --dry-run-map-check or --synthesize-probe is set")
    merge_v2(args.ckpt, args.base_gguf, args.imatrix, args.out, args.libggml, args.workers)


if __name__ == "__main__":
    main()
