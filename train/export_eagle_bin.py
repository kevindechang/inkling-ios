#!/usr/bin/env python3
"""Export trained TinyEagleDrafter checkpoint (.pt) to a compact Q4_0 binary (.bin) for ARM64 NEON C++ inference.

Supports:
  - v1 (1-layer EAGLE-v2 checkpoint)
  - v2 (N-layer EAGLE-v3 checkpoint with optional learned residual h_skip)
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))
sys.path.insert(0, str(ROOT / "train"))
import gguf  # noqa: E402
from train_mac_routers import GGMLCodec  # noqa: E402
from train_mac_eagle import LazyGGUFRows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/eagle/eagle_v3_2layer_d1024.pt"))
    ap.add_argument("--model", default="models/varloop-stage2-v1-run3-reap40-i8mm.gguf")
    ap.add_argument("--lib", default=str(ROOT / "llama.cpp/build/bin/libggml-base.dylib"))
    ap.add_argument("--k-export", type=int, default=16384, help="Shortlist rows to export (8192=~10.6MB head, 16384=~21.2MB head)")
    ap.add_argument("--out", default=str(ROOT / "runs/eagle/eagle_v3_2layer_d1024_k16384_q4_0.bin"))
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    sd = ck["state_dict"]
    sl_ids = np.asarray(ck["shortlist_ids"], dtype=np.int32)
    if args.k_export > 0 and args.k_export < len(sl_ids):
        sl_ids = sl_ids[:args.k_export].copy()

    h_dim = 2304
    d_model = int(cfg["d_model"])
    n_heads = int(cfg.get("n_heads", 8))
    ff_dim = int(cfg["ff_dim"])
    n_layers = int(cfg.get("n_layers", 1))
    has_h_skip = "h_skip" in sd
    is_v2 = ("layers.0.wqkv.weight" in sd) or n_layers > 1 or has_h_skip
    k_sl = int(len(sl_ids))
    n_vocab = 201024

    reader = gguf.GGUFReader(str(args.model), "r")
    codec = GGMLCodec(Path(args.lib))
    tok_norm_w = np.asarray(next(t for t in reader.tensors if t.name == "token_embd_norm.weight").data, dtype=np.float32).copy()
    head_table = LazyGGUFRows(Path(args.model), reader, codec, "output.weight")
    W_head_f32 = head_table.get_rows(sl_ids.astype(np.int64))

    q4_type = int(gguf.GGMLQuantizationType.Q4_0)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with out_p.open("wb") as f:
        ver = 2 if is_v2 else 1
        rsv = [n_layers if is_v2 else 0, 1 if has_h_skip else 0, 0, 0, 0, 0, 0, 0]
        hdr = struct.pack("<4s7I8I", b"EAGL", ver, h_dim, d_model, n_heads, ff_dim, k_sl, n_vocab, *rsv)
        assert len(hdr) == 64
        f.write(hdr)
        f.write(np.ascontiguousarray(sl_ids, dtype=np.int32).tobytes())
        f.write(np.ascontiguousarray(tok_norm_w, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(sd["norm_h.weight"].float().numpy(), dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(sd["norm_e.weight"].float().numpy(), dtype=np.float32).tobytes())

        def write_q4(key: str) -> None:
            w = sd[key].float().numpy()
            q = codec.quantize_2d(w, q4_type)
            f.write(np.ascontiguousarray(q, dtype=np.uint8).tobytes())
            print(f"  wrote {key:28s} shape={list(w.shape)} Q4_0={q.nbytes/1e6:5.2f} MB")

        if not is_v2:
            for name in ("attn_norm.weight", "ffn_norm.weight", "out_norm.weight"):
                f.write(np.ascontiguousarray(sd[name].float().numpy(), dtype=np.float32).tobytes())
            for name in ("fc_in.weight", "wqkv.weight", "wo.weight", "w_gate_up.weight", "w_down.weight", "fc_out.weight"):
                write_q4(name)
        else:
            f.write(np.ascontiguousarray(sd["out_norm.weight"].float().numpy(), dtype=np.float32).tobytes())
            if has_h_skip:
                f.write(np.ascontiguousarray(sd["h_skip"].float().numpy(), dtype=np.float32).tobytes())
            for l in range(n_layers):
                f.write(np.ascontiguousarray(sd[f"layers.{l}.attn_norm.weight"].float().numpy(), dtype=np.float32).tobytes())
                f.write(np.ascontiguousarray(sd[f"layers.{l}.ffn_norm.weight"].float().numpy(), dtype=np.float32).tobytes())
            write_q4("fc_in.weight")
            for l in range(n_layers):
                for sub in ("wqkv.weight", "wo.weight", "w_gate_up.weight", "w_down.weight"):
                    write_q4(f"layers.{l}.{sub}")
            write_q4("fc_out.weight")

        q_head = codec.quantize_2d(W_head_f32, q4_type)
        f.write(np.ascontiguousarray(q_head, dtype=np.uint8).tobytes())
        print(f"  wrote {'head_sl.weight':28s} shape={list(W_head_f32.shape)} Q4_0={q_head.nbytes/1e6:5.2f} MB")

    print(f"EXPORTED v{ver} (L={n_layers}, skip={has_h_skip}, K={k_sl}) -> {out_p} ({out_p.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
