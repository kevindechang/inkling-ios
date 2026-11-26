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

All dense weights (`~1.0–1.2 GB`) load eagerly into RAM (`-lm none -lzm on`); only the 48 `ffn_{gate_up,down}_exps.weight` tensors stay lazy/memory-mapped and are streamed through an explicit anonymous-memory LRU cache (`GGML_MMID_CACHE_MB=3000..3250`):

1. **Direct `F_NOCACHE` parallel `pread`:** Replaces 16 KB page-fault `memcpy` (~0.2 GB/s) with direct uncached `pread` across reader threads (~0.8–1.2 GB/s), giving a `4.5×` immediate speedup (`961 → 212 ms` at 3p K8).
2. **Known-Route Paired-`down` Overlap (`GGML_MMID_OVERLAP=1`):** Ported from Atomic Engine's CPU-portable design. Once `blk.L.ffn_gate_up_exps` routes its top-4 experts, the exact same 4 expert IDs are guaranteed to be needed by `blk.L.ffn_down_exps`. Two background reader threads prefetch all missing `down` slices *while* the 2 P-cores compute `gate_up` (`100%` prefetch hit rate, `0` wasted reads). Cut iPhone `-t 2` median latency by **`-40%`** (`119–124 → 72–73 ms`) and CPU heat by **`-33%`**.
3. **Decode Threading (`-t 2`) vs. Prefill Threading (`-tb 6`):** On A17 Pro's 2P+4E topology, running decode on `-t 4` or `-t 6` burns `1.9–2.5×` CPU heat for no sustained mean win. Locking decode to `-t 2` keeps the SoC cool (`nominal → fair`), while `-tb 6` unlocks all 6 cores with `smmla` GEMM for fast prompt prefill.

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

- **Why Stage 3 distillation + merge is required:** Cutting `3p K8 → 1p K4` + re-quantizing dense attention to `Q4_K` introduces repetition loops unless LoRA + AttnRes depth queries are calibrated against the exact quantized phone weights. Furthermore, unmerged LoRA adds `+7 ms/tok` at decode time on Stage 2 (`58` vs `53 ms`).
- **Stage 3 design (next):** Train high-rank attention/router/LM-head LoRA + student-only F32 small-dense (`attn_res_query`, `ffn_res_query`, `output_res_query`, norms) against a clean BF16-dense 3p K8 teacher on concise reasoning + writing + tool-use trajectories (dropping coding data), then **merge directly into `varloop-stage2-v1.gguf`** (`0 ms` decode cost, `0 MB` extra RAM).
