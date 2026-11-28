# Architecture — `inkling-ios`

Running a **26.4B-parameter looped Mixture-of-Experts model** (`thinkingmachines/TML-d24-varloop-research`, 24 blocks / 128 experts per layer, `inkling2_loop` architecture) at **~20 tok/s** inside an iPhone 15 Pro's **6 GiB iOS memory entitlement** requires solving four coupled bottlenecks: **memory, NVMe flash I/O, ARM CPU kernel efficiency, and quality recovery after shape/bit cuts**.

---

## 1. Why CPU-Only Streaming (Not Metal)

Early device experiments (`P5–P12`, `M1–M3`) established three hard constraints on iPhone 15 Pro (A17 Pro, 6 GiB entitlement):
- **Full model doesn't fit in RAM:** Even at ~2.8–3.1 bpw, the 26B model is `8.5–9.3 GiB`. Resident Metal buffers fail with `kIOGPUCommandBufferCallbackErrorOutOfMemory` (`P7`).
- **Unmapped Metal host pointers thrash flash:** `mmap`-backed Metal buffers re-fault every token at ~15 s/tok (`P6`).
- **Hybrid dense-Metal / experts-CPU splits the graph ~900 times/token** and ran `2.8×` slower than pure CPU (`M2`).
- **Pure CPU with resident dense weights + explicit RAM expert cache** fits cleanly in `4.4–4.7 GB` total process footprint, avoids Metal OOMs completely, and hits `~43–54 ms` zero-miss decode on A17 Pro's 2 Performance cores (`-t 2`).

---

## 2. Model Shape: 1-Pass, Top-4 (`TML-d24-varloop-research`)

`Inkling 2 Mini` runs layers `6–17` for 3 loop passes (`48` layer visits/tok) with top-8 of 128 experts (`384` expert reads/tok ≈ `1.02 GB/tok`). Cutting passes on standard `Inkling 2 Mini` immediately collapses coherence (`0%` GSM8K untrained).

Switching to its Apache-2.0 variable-loop sibling **`thinkingmachines/TML-d24-varloop-research`** (trained with 1–5 passes) makes **`1 pass, top-4 experts`** (`24` layer visits/tok, `96` expert pairs/tok) coherent out of the box and cuts compute by ~`4×`:
- Enabled in patched `llama.cpp` via `INKLING_ALLOW_ONE_PASS=1` + KV overrides `inkling2_loop.loop.count=1, inkling2_loop.expert_used_count=4`.
- Repairing the `K=8 → K=4` and `3p → 1p` gap via self-distillation (Stage 3) recovers reasoning, writing, and tool-use quality while keeping the `1p K4` speed envelope.

---

## 3. Streaming Expert Cache & Known-Route Overlap

All dense weights (`~1.0–1.2 GB`) load eagerly into RAM (`-lm none -lzm on`); only the 48 `ffn_{gate_up,down}_exps.weight` tensors stay lazy/memory-mapped and are streamed through an explicit anonymous-memory LRU cache (current recommended `v35d` env on the v2 GGUF: `GGML_MMID_CACHE_MB=2950`, `GGML_MMID_PIN_N=40`, `GGML_MMID_CECAR_BONUS=0.35`, with `GGML_MMID_OVERLAP=1`, `GGML_MMID_REPACK=1`, `GGML_MMID_LRU=1`, `GGML_MMID_IO_THREADS=4`, `GGML_MMID_CECAR_PREFILL=1`; the `v35a` env uses `2700` / `36`; earlier versions of this doc listed `3000..3250`, which crosses the iOS memory-compressor cliff at `-ub 128`, see §7):

1. **Direct `F_NOCACHE` parallel `pread`:** Replaces 16 KB page-fault `memcpy` (~0.2 GB/s) with direct uncached `pread` across reader threads (~0.8–1.2 GB/s), giving a `4.5×` immediate speedup (`961 → 212 ms` at 3p K8).
2. **Known-Route Paired-`down` Overlap (`GGML_MMID_OVERLAP=1`):** Ported from Atomic Engine's CPU-portable design. Once `blk.L.ffn_gate_up_exps` routes its top-4 experts, the exact same 4 expert IDs are guaranteed to be needed by `blk.L.ffn_down_exps`. Two background reader threads prefetch all missing `down` slices *while* the 2 P-cores compute `gate_up` (`100%` prefetch hit rate, `0` wasted reads). Cut iPhone `-t 2` median latency by **`-40%`** (`119–124 → 72–73 ms`) and CPU heat by **`-33%`**.
3. **Decode Threading (`-t 2`) vs. Batch/Prefill Threading (`-tb 4`):** On A17 Pro's 2P+4E topology, running decode on `-t 4` or `-t 6` burns `1.9–2.5×` CPU heat for no sustained mean win. Locking decode to `-t 2` keeps the SoC cool (`nominal → fair`), while batch/prefill uses `-tb 4` with `-ub 64` in every `v35`/`v36` run (earlier versions of this doc said `-tb 6`; no `v35`/`v36` run uses it).

---

## 4. ARMv8.6 `i8mm` + Zero-Bloat NEON `4×4` On-Miss Repacking

1. **`IOS_I8MM=1` (`armv8.6-a+dotprod+fp16+i8mm`):** Unlocks `284` `smmla` / wider `8×8` & `4×8` GEMV/GEMM kernels on A17 Pro across all `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0` dense attention & LM-head tensors (`-6.0 ms/tok` pure zero-miss decode win, `-11.5 ms` CPU/tok).
2. **Zero-Bloat On-Miss Repacking (`GGML_MMID_REPACK=1`):** When an expert slice misses flash, reader threads repack it in-memory into an interleaved `4×4` SIMD layout of the **exact same byte size** (zero cache RAM bloat; `mode=2` IQ unpacking was rejected because it doubled cache misses).
   - Custom bit-exact (`0/2304` bit diffs) ARM NEON `4×4` GEMV kernels added for `Q2_K`, `Q2_0`, `NVFP4`, and `MXFP4`, alongside upstream `Q4_0`/`IQ4_NL` repacking.
   - `ggml_arm_repack_q2_K_4x4` vectorized with NEON `vst4q_u8` + `vst4q_u32` (`2.23×` faster on Mac `162 → 72.6 µs`, `>2.3×` faster on iPhone, `0` byte diffs).

---

## 5. KL-Optimal, Latency-Constrained Quantization (`quant/`)

Standard community quants (e.g. `Q2_K_XL`) use `IQ1_S`/`IQ2_S`/`IQ2_XXS` lookup-table formats on `gate_up` experts — tiny on disk, but slow on ARM CPU (`0.040–0.049 ns/w`, un-repackable without expanding cache size).

Using a two-stage compiler inspired by LookAtMyQuant's `gguf_compiler.py`, specialized for `inkling2_loop` at **`1p K4`** on ARM CPU:
1. **`quant/kl_sweep.py`:** Sweeps every expert (`gate_up`, `down`) and wide attention tensor against real `libggml-base` `ggml_quantize_chunk` + official `imatrix`, measuring exact `1p K4` logit KL divergence (`635/656` records swept on 1× L40S in 1.5 h).
   - *Key empirical findings:* imatrix `Q2_K` matches `IQ2_S` within `0.000–0.005` KL on every `gate_up` layer while running **`1.8×` faster** + `REPACK=1`-accelerated (`Q2_0` on `gate_up` is catastrophic: `+0.04–0.08` KL); `down` experts have sharp sensitivity spikes at loop boundaries **`L05` (`+0.073`)** and **`L17` (`+0.096`)**; late attention (`L09–L21`) moves `Q8_0 → Q4_K` for virtually zero KL (`+0.0000–0.0025`).
2. **`quant/dp_allocate.py`:** Solves a latency-and-size-constrained multiple-choice knapsack DP calibrated to A17 Pro's measured `i8mm` + `REPACK=1` roofline (`lm_head` predicted `9.30 ms` vs. measured `9.79 ms`).
   - Emits `quant/tensor_types_stage2_v1.txt` (226 anchored `^name$=type` rules): `23× Q2_K + L23 Q3_K` `gate_up`, `11× Q2_0 + 6× NVFP4 + 3× IQ4_NL + 3× MXFP4 + L23 Q4_0` `down` (protecting `L05` & `L17`), `58× Q4_K + 22× Q5_K + 12× Q6_K + 9× Q8_0` wide `attn`, `32× BF16 → F16` (`attn_sink_proj`, `attn_{k,v}_conv_proj`).
   - **Result (`varloop-stage2-v1.gguf`, `9.28 GiB`, `3.06 bpw`):** Pure expert compute dropped **`>2.3×`** (`~35 → 14.9 ms/tok`), attention dropped **`-5.6 ms`**, RAM footprint dropped **`-200 MB`**, and iPhone 15 Pro zero-miss floor dropped from **`69 ms → 43–44 ms` (`22.7–23.3 tok/s`)**.

---

## 6. Quality Recovery & Short Thinking (`train/`)

- **Why Stage 2/3 distillation + merge is required:** Cutting `3p K8 → 1p K4` + re-quantizing dense attention to `Q4_K` introduces repetition loops unless LoRA + AttnRes depth queries are calibrated against the exact quantized phone weights. Furthermore, unmerged LoRA adds `+7 ms/tok` at decode time (`58` vs `53 ms`).
- **Bit-exact local streaming BF16 merge (`quant/merge_run3_local.py`, `quant/merge_wsd_local.py`):** Streams only modified BF16 tensors via HTTP `Range` from Hugging Face, applies FP32 LoRA/router deltas, reorders experts `0..39 = HOT` (retaining all `128/128` experts), and quantizes directly via `libggml-base` `ggml_quantize_chunk` + official `imatrix` (`0 ms` LoRA decode cost, `0` double-quantization).

---

## 7. `Inkling-2-FlashPin-26B` (`v35`) — Closing the Unpruned `128e` `20+ tok/s` Gap

Eliminating the gap between pruned RAM-resident runs (`v30`) and the unpruned `128/128`-expert model (`v35a`: **`23.43 tok/s`**, `v35f`: **`21.20 tok/s`** 375-tok, `v35d`: **`21.78 tok/s`** & `0.761 s` warm TTFT, `v35f_mix`: **`18.36 tok/s`** 400-tok 3-topic mix, all exact `200K` vocab & `nominal → nominal`) came from six coupled systems optimizations:

1. **Hot-40 REAP Reordering & Pinning (`GGML_MMID_PIN_N=36..40`, `GGML_MMID_LRU=1`):** Permanently pins hot expert indices `0..PIN_N-1` in RAM while cold safety-net experts stream through the remaining dynamic LRU ring.
2. **Per-Layer Cache-Aware Routing (`GGML_MMID_CECAR_LAYERS`, `GGML_MMID_CECAR_PREFILL=1`):** Replaces uniform routing bias with per-layer sensitivity budgets (`tools/optimize_cecar_layers.py`) so cache-tolerant layers absorb more hits while sensitive layers stay closer to the unperturbed router.
3. **Compute-Buffer Halving (`-ub 64`) & iOS Memory-Compressor Avoidance:** At `-ub 128`, `llama.cpp` reserves a `242.7 MiB` CPU compute buffer, spiking `tok 0` footprint over iOS's `~4,450 MB` compressor cliff when `CACHE_MB >= 3000`. Using `-ub 64` (matching our 64-tok tiled FA and batched MoE chunk size) cuts the compute buffer to **`121.4 MiB`** (`-121 MiB` peak RAM), keeping every `v35` run in `3,975–4,364 MB` peak (`nominal → nominal`).
4. **`i8mm` Tensor Upgrades (`quant/repack_i8mm_gguf.py`):** Replaces the remaining non-`i8mm` expert tensors (`blk.23` `Q3_K → Q4_K`, middle-layer `NVFP4`/`MXFP4` `down → Q4_0`) and quantizes F32 routers to `Q8_0` (`-20.8 MB`), dropping zero-miss decode from `43 ms → 36 ms`.
5. **Lossless ARM64 NEON / Accelerate Compute Kernels:**
   - **Multi-token 64-tok tiled GQA `Q8_0` Flash-Attention** (`ops.cpp`, `neq1 >= 1`, `max_diff <= 1.34e-7`).
   - **Batched MoE repacked GEMM** (`ggml_mmid_repack_gemm`: zero-copy NEON `Q2_K`/`Q2_0` for `M=2,3,4` + gated `Q4_0`/`Q4_K` `i8mm`).
   - **Row-broadcast NEON `ADD`/`SUB`/`MUL`/`DIV`** (`row-bcast-neon.h`): replaces length-1 per-element `vDSP` calls with IEEE `vaddq`/`vsubq`/`vmulq`/`vdivq` (`2,121,548` checks, `0` mismatches; `8.4×` per-call `MUL` speedup on Mac graph profile).
   - **Fused `SiTU` (`GGML_GLU_OP_SITU`, `situ-neon.h`, `GGML_SITU_FAST=accel`):** Collapses Inkling 2's 9-node `SiTU` (`beta * tanh(gate/beta) * sigmoid(gate) * lb * tanh(up/lb)`) chain per layer into 1 op using Apple Accelerate `vvtanhf`/`vvexpf` (`3.10×` faster `GLU`), bringing A17 Pro zero-miss decode down to **`30–31 ms`** (`33 tok/s` compute ceiling) with exact `200K` vocab (`LLAMA_LM_HEAD_SHORTLIST=off`).
6. **Reversible Offline Pre-Repacking (`tools/prerepack_exps.cpp`, `GGML_MMID_PREREPACKED=1`):** Every expert tensor (`Q2_K`, `Q2_0`, `Q4_0`, `Q4_K`, `IQ4_NL`) has the exact same byte count in raw and repacked layouts (`static_assert` in `repack.h`). Pre-repacking offline zeroes runtime `repack_ms` (`438–907 ms → 0.0 ms` in decode) and speeds up prefill (`5.17 → 4.14–4.50 s`) and miss-heavy multi-topic decode (`17.46 → 18.36 tok/s`) with a journaled bit-exact `--unpack` path.

## 8. v2 — QAT Working-Set Distillation and the Pareto Cliff

v2 ([`docs/V2.md`](V2.md)) keeps the v1 file layout and runtime unchanged, and retrains on top of the shipped quantized weights (QAT).

The main systems lesson is a **Pareto cliff** in router working-set concentration. With a fixed Hot-40 set per layer, a gently bounded cold-expert bias (cold traffic 62% → 37% on the training distribution) improved every distillation metric. Driving cold traffic to 5–6% raised the tool/instruction KL by about 3×, because specialist experts outside the pinned set become unreachable.

A second lesson: training-time cold fraction did not predict runtime cache misses in `llama.cpp`. v2 still relies on per-layer CECAR (`OPT λ=0.95`) for its miss rate. The trained router bias alone is not enough.
