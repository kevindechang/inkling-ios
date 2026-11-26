# Engineering Log — `inkling-ios`

Complete, transparent chronological record of every experiment, measured number, and rejected dead end on the path to **20 tok/s, <= 2 s TTFT, low RAM, and intact reasoning/writing/tool use** for Inkling 2 (`26.4B` looped MoE) on **iPhone 15 Pro** (A17 Pro, 6 GiB memory entitlement, iOS 27.0.1).

---

## Summary Timeline of Speed Milestones (iPhone 15 Pro)

| Milestone | Config | Zero-Miss Floor (`z0`) | Sustained Decode | RAM Footprint | Thermal | Key Change |
|---|---|---:|---:|---:|---|---|
| **P5** (Nov 22) | Mini `Q2_K_XL`, 3p K8, `mmap` | — | `0.64 tok/s` (`1432 ms`) | `1.30 GB` | nominal | First working CPU run (`LLAMA_MMAP_NO_PREFETCH=1`) |
| **P15** (Nov 22) | Mini `Q2_K_XL`, 3p K8, `F_NOCACHE pread` | — | `4.72 tok/s` (`212 ms`) | `4.79 GB` | fair | Explicit `3000 MB` LRU cache + parallel `pread` (`7.5×`) |
| **P21** (Nov 23) | Mini `Q2_K_XL`, 1p K4, `-t 4` | `~55 ms` | `15.4 tok/s` (`61 ms`) | `4.6 GB` | nominal | `INKLING_ALLOW_ONE_PASS=1` (Mini untrained loops) |
| **P29** (Nov 23) | Sibling `varloop` `Q2_K_XL`, 1p K4, `-t 4` | `50 ms` | `14.2–15.4 tok/s` (`65–70 ms`) | `4.6 GB` | fair | Switched to `TML-d24-varloop-research` (coherent at 1p K4!) |
| **P37** (Nov 24) | Sibling `Q2_K_XL`, 1p K4, `-t 2`, `OVERLAP=1` | `~66 ms` | `13.0–13.2 tok/s` (`72–73 ms`) | `4.68 GB` | fair | Atomic known-route paired-`down` overlap (`-40%` ms, `-33%` CPU) |
| **v21** (Nov 26) | + `IOS_I8MM=1` (`armv8.6+i8mm`), `-t 2` | **`69.0 ms`** (`cpu 138`) | **`13.3 tok/s`** (`73.0 / 75.2 ms`) | `4.66 GB` | fair | Reproducible `-6.0 ms/tok` zero-miss win on A17 Pro |
| **v26** (Nov 27) | **`varloop-stage2-v1.gguf`** + `REPACK=1` + NEON `Q2_K`, `-t 2`, `3250 MB` | **`43–44 ms`** (`22.7–23.3 t/s`, `cpu 86`) | `~18–20 tok/s` *(projected warm merged)* | `4.69 GB` | nominal | KL+DP quant (`23× Q2_K` `gate_up`, `Q4_K` `attn`) + `2.23×` faster NEON repack |

---

## Detailed Chronological Log

### 2026-11-22 — Initial Bring-up, Metal Failures, and True CPU Streaming (`P0–P16`)
- **P0–P4:** Stock `llama.cpp` hung on iPhone load because `mmap` issued `madvise(WILLNEED)` across the full `9.1 GB` file. Added `LLAMA_MMAP_NO_PREFETCH=1`.
- **P5:** First successful iPhone decode (CPU, `mmap`, 3p K8): `0.64 tok/s` (`1432 ms/tok`), `1.30 GB` footprint.
- **P6 (Metal, residency off):** `~15 s/tok` — host-pointer Metal buffers over file-backed pages faulted every token. **Rejected.**
- **P7 (Metal, residency on):** `kIOGPUCommandBufferCallbackErrorOutOfMemory` crash (signal 10). **Rejected.**
- **M1–M3 (Mac stand-in):** All-Metal `21.0 tok/s`, all-CPU `19.4 tok/s` (within 8%), hybrid dense-Metal/experts-CPU `7.5 tok/s` (`2.8×` slower due to ~900 graph splits). **Decided: CPU-only engine on phone.**
- **P9–P14 (Explicit LRU cache):** OS page cache never warmed on iOS (`0.63 tok/s` flat). Added `TENSOR_READ_LAZY` on `ffn_{gate_up,down}_exps` + explicit `GGML_MMID_CACHE_MB` anonymous-RAM LRU cache. `memcpy` from `mmap` still ran `869–961 ms/tok` (16 KB faults ≈ `0.2 GB/s`). Note: fixed bug where `-ngl 0` still offloaded graph ops to a `2.7 GB` Metal buffer (`op_offload=false`, empty device list).
- **P15–P16 (`F_NOCACHE` parallel `pread`):** Switched cache miss path to direct `F_NOCACHE` `pread` via `dispatch_apply`. Jumped `7.5×` to **`4.72 tok/s` (`212 ms/tok`, `69 MB/tok` missed, `4.79 GB` footprint)** at full 3p K8.

### 2026-11-23 — Shape Cuts, Variable-Loop Sibling (`TML-d24-varloop-research`), and First Distillation (`P18–P35`, `T1–T6`)
- **P18–P21 (Shape sweep on Inkling 2 Mini):**
  - `3p K4`: `141–152 ms` (`6.4–7.0 tok/s`).
  - `1p K4` (`INKLING_ALLOW_ONE_PASS=1`): **`57–61 ms` (`14–16 tok/s`)**, proving the 15+ tok/s compute envelope exists, but standard Mini loops degenerately at 1p K4 (`0/30` GSM8K untrained).
- **Found `thinkingmachines/TML-d24-varloop-research` (Apache 2.0):** Variable-loop sibling trained at 1–5 passes. Quantized from `ggml-org` BF16 + imatrix to the `Q2_K_XL` recipe (`varloop-Q2_K_XL.gguf`, `8.45 GiB`, `2.78 bpw`).
- **P24–P32 (Phone sustained runs on `varloop-Q2_K_XL`):**
  - Every shape (`1p K4`, `2p K4`, `1p K8`) produced coherent, properly terminating text out of the box!
  - Sustained 600-token runs: **`1p K4` = `65–68 ms` (`14.7–15.4 tok/s`, `thermal: fair`)** in 3/3 runs; `2p K4` throttled to `serious` (`125–146 ms`) in 2/3 long runs. **Locked `1p K4` as the phone shape.**
  - Cache sizing (`P28`, `P40`): Increasing `3000 → 3600/4000 MB` on `Q2_K_XL` raised footprint to `5.5–5.6 GB` (risking iOS OOM) for negligible speedup.
- **P33 (Forced no-thinking check):** Skipping `<|content_thinking|>` at 1p K4 caused loops on instruction prompts; kept free-channel thinking and targeted training to shorten thinking.
- **T1–T3 / T8 (First self-distillation runs on 1× L40S 48 GB):**
  - Built `train/distill.py` with `--experts-int8` (holds dequantized phone experts as int8 + per-row scale in `24.4 GB` VRAM so 26B fits on 48 GB L40S; `0.8%` rel RMS vs dequantized experts).
  - `smoke1` (202 steps, forward KL, r32 attn + r8 router) cut held-out KL `-50%` (`0.3805 → 0.190`).
  - `run3` (379 steps, `batch 8×1024`, `reverse-kl 0.5` + on-policy student rollouts) lifted 100-item composite score from `0.460 → 0.620` (`GSM8K-Hard 90%`, `BFCL 90%`, `IFEval 75%`), cut thinking `800 → 535 tok`, and cut capped-loop items `23 → 11`. Held-out KL plateaued at `~0.190` with rank-32 LoRA.

### 2026-11-24 — Community Replication, Atomic Overlap Port, and Rejected Dead Ends (`P36–P41`, `B9–B12`)
- **Inference thinking caps (`B9`, `B12`):** Hard cap 256 (`-0.04`), hard cap 128 (`-0.16`, reasoning collapsed `90% → 40%`), LookAtMyQuant sentence-boundary soft cap 384 (`-0.05`) and soft cap 256 (`-0.08`). **Rejected:** inference caps degrade quality; short thinking must be trained.
- **Sampling (`B6`):** `temp 0.6, top_k 50` scored lower than greedy (`0.537` vs `0.558`). **Rejected.**
- **Atomic Engine replication & CPU port (`P37`):**
  - Full Atomic Engine is Metal-only (forbids CPU streaming for `inkling2_loop`).
  - Ported its 3 CPU-portable ideas (`GGML_MMID_OVERLAP`, `GGML_MMID_LRU`, `GGML_MMID_COALESCE`, `GGML_MMID_STATS`, all default OFF).
  - **Phone A/B (`P37`, `-t 2`, 450 tok):** Known-route paired-`down` overlap dropped median **`119–124 → 72–73 ms` (`-40%`)**, mean `135–172 → 76–77 ms`, CPU `-33%` (`196–208 → 133–135 ms`), ending `fair` instead of `serious`. **Adopted.** `LRU2` and `COALESCE` were neutral at decode.
  - Confirmed `-t 2` matches `-t 4` in sustained speed at half the CPU heat (`135` vs `279 ms`).
- **LookAtMyQuant 3.9 GB MLX replication (`B11`):**
  - Full shape (3p K8, Mac M4 Pro MLX): `27–29 tok/s`, `0.650` composite (`GSM8K 95%`, `BFCL 100%`, `IFEval 70%`).
  - Cut shapes on Mac MLX: `1p K8 = 57 tok/s` (degraded/loops; est. ~14 tok/s on A17 Pro GPU), `1p K4 = 23 tok/s` (slower — kernel hardcoded for 8 experts).
  - **Verdict:** Do not port their MLX runtime; **adopt their KL-guided quantization allocation method** over `llama.cpp`'s native ARM CPU formats.
- **Naive dense-Q4 requant (`B10`):** Requantizing 68 `Q8_0` attention tensors to `Q4_K` without imatrix or LoRA recalibration cost `-0.077` (`IFEval 75% → 60%`). Motivated principled KL-guided allocation.
- **`K=3` vs `K=4` (`P41`):** Saved only `1–6 ms` on phone. **Rejected.**

### 2026-11-25 — Thread Scaling, Exact LM-Head Shortlist Proof, and Lossless ARM Kernels
- **Thread scaling study:** Emulated 2P+4E switches (`GGML_CPU_FINE_CHUNKS`, `GGML_CPU_SMALL_NODE`) gave only ~`6%` estimated gain at `-t 4/6`; confirmed `-t 2` is optimal for phone thermal/sustained decode.
- **Exact LM-head shortlist (`LLAMA_LM_HEAD_SHORTLIST`):** Proved mathematically and empirically (`100.0%` fallback across `3,728` decode steps on Mac & iPhone) that `output.weight` rows lie on a unit hypersphere (`||w_i||_2 ≈ 1.0`, top-1 median cosine `0.215`), requiring `>2.2–2.7 bpw` by Shannon rate-distortion before any provably-exact upper bound can prune. Lossy 8k shortlists diverged on 4/4 prompts. **Rejected.**
- **Lossless ARM optimizations:**
  - Added `IOS_I8MM=1` (`armv8.6-a+dotprod+fp16+i8mm`, `284` `smmla` ops).
  - Added `GGML_BF16_NEON=1` (`4.21×` faster `ggml_vec_dot_bf16`, `0/32` bit diffs).
  - Added `GGML_MMID_REPACK=1|2` with custom bit-exact (`0/2304` bit diffs) NEON `4×4` GEMV kernels for `Q2_0` (`1.80×`), `NVFP4` (`1.42×`), `MXFP4` (`1.26×`), and `Q2_K` (`1.38×`).
  - Wrote `quant/kl_sweep.py` and `quant/dp_allocate.py`.

### 2026-11-26 — Phone `i8mm` Verification, CECAR Test, Stage 2 KL Sweep, and `varloop-stage2-v1.gguf` (`v20–v22`)
- **Phone `v20`/`v21` (clean 100%-battery `-t 2`, 450 tok):**
  - **`IOS_I8MM=1` Adopted:** Reproducible **`-6.0 ms/tok`** on zero-miss decode floor (`75.0 → 69.0 ms`, CPU `149 → 137.5 ms`), `13.3 tok/s` sustained (`nominal → fair`).
  - **`GGML_MMID_REPACK=2` Rejected:** Expanding IQ2 slices doubled phone misses (`7.0 → 16.9/tok`) and slowed decode to `110 / 121 ms` (`wait 23.8 ms/t`).
  - **Decode-only `[mmid]` stats wired on iOS** (`ggml_cpu_mmid_stats_reset/dump` called directly before `_exit`): measured `7.2–14.0 ms/tok` flash wait even with `OVERLAP=1`.
- **CECAR cache-bonus routing (`GGML_MMID_CECAR_BONUS`, `v22`):**
  - `bonus=0.15` lifted `Q2_K_XL` phone decode to `70.0 / 71.6 ms` (`13.97 tok/s`, wait `7.3 → 4.2 ms/t`, `6.7%` swaps), **but dropped quality `-0.10`** (`0.620 → 0.520`, `IFEval 75% → 55%`). **Rejected `0.15`.**
- **Stage 2 `1p K4` KL Sweep (`quant/kl_sweep.py`, 1× L40S, `5,510 s`):**
  - Parallelized `ggml_quantize_chunk` across CPU threads (`IQ2_S` dropped `148.4 s → 4.7 s`, `31×` faster, bit-exact) + `--eval-batch 4`.
  - Swept `635 / 656` records (`L00–L22` complete; `L23` pinned conservatively).
  - Calibrated `quant/dp_allocate.py` against real A17 Pro `i8mm` floor (`lm_head` pred `9.30` vs meas `9.79 ms`) → emitted `quant/tensor_types_stage2_v1.txt` (predicted `51.3 ms` zero-miss = `19.5 tok/s`).
- **Built `varloop-stage2-v1.gguf` (`ml.m7i.4xlarge`, `418 s` quantize):** `9.28 GiB` (`9,979,246,016` B, `3.06 bpw`), `sha256 c4b79cd7...`. Verified all 14 tensor types dequantize cleanly in `gguf-py`.

### 2026-11-27 — Stage 2 iPhone Verification (`v23–v27`), NEON `Q2_K` Repack Speedup, and Public Reasoning Bench
- **Stage 2 iPhone A/B (`v23–v26`, `-t 2`, `IOS_I8MM=1`, `OVERLAP=1`, `REPACK=1`):**
  - **Pure zero-miss compute hit `43–44 ms` (`22.7–23.3 tok/s`, `cpu 86 ms` = `-37%` CPU heat)** vs `Q2_K_XL` control `69–72 ms` (`cpu 133–142 ms`).
  - `GGML_NODE_PROF` (`v24`, clock-normalized by `lm_head`): pure expert compute fell `>2.3×` (`~35 → 14.9 ms/tok`), attention `-5.6 ms` (`23.2 → 17.6 ms/tok`), total zero-miss floor `~49.5 ms` (`20.2 tok/s`).
  - Unmerged `run3` LoRA added `+7 ms/tok` (`58` vs `53 ms`) → Stage 3 will merge LoRA directly into the GGUF.
  - **Vectorized `ggml_arm_repack_q2_K_4x4`** with NEON `vst4q_u8`/`vst4q_u32`: `0` byte diffs, **`2.23×` faster** (`162 → 72.6 µs` on Mac; `>2.3×` faster on iPhone: `420 → 179 µs/slice` avg).
  - Bumped cache `3000 → 3250 MB` using dense-weight RAM savings (`4,694 MB` footprint = same as old `Q2_K_XL` `4,663 MB`), cutting flash reads by `-1.05 GB` and wait by `-5.6 ms/tok`.
- **Prefill TTFT optimization (`v27`):** Added `-tb N` (`n_threads_batch`) and `-wf` (warm turn-2 TTFT) + `-state-load` benchmarking to `ios/inkling_run.cpp`.
  - Cold 70-tok baseline (`-tb 2`): `6.03 s` (`4.01 s` flash wait on `3.75 GB` cold expert reads); warm 40-tok Turn 2: `3.08 s` (`-tb 2`) → **`2.67 s`** (`-tb 4`, optimal) vs `6.15 s` (`-tb 6`, starved I/O threads).
  - System-prompt KV state caching (`-state-save` / `-state-load`, `3.10 MB`, loads in **`0.01 s`**) + `-tb 4` + `GGML_MMID_IO_THREADS=4`: cold 70→58 tok dropped to **`4.28 s`**; **honest cross-prompt warm Turn 2 (`long1t → chat1t`, 28 new tok) dropped to `2.13 s`** (`thermal: fair → nominal`); hot-cache Turn 2 hit **`0.81 s` (`34.5 tok/s`)**.
- **Public 271-item reasoning/tool/instruction benchmark suite:** Early Stage 2 (*before* Stage 3 retrain, old mismatched LoRA) hit **`GSM8K-Hard 25/30 (83.3%)`**, **`MATH-500 18/30 (60.0%)`** (ahead of `Gemma 4 26B Q4` `43.3%` and `Qwen 3.6 35B` `50.0%`), and `Logic 4/13 (30.8%)`, with high thinking tokens (`596–1,118 tok`) confirming Stage 3 short-thinking distillation as the final key milestone.
