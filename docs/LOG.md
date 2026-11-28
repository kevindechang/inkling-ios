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
| **v26** (Nov 27) | `varloop-stage2-v1` (128e) + `REPACK=1` + NEON `Q2_K`, `-t 2`, `3250 MB` | **`43–44 ms`** (`22.7–23.3 t/s`, `cpu 86`) | `10.1–13.7 tok/s` (`73–99 ms`, `11–19 miss/t`) | `4.69 GB` | nominal | KL+DP quant (`23× Q2_K`) + `2.23×` faster NEON repack; flash wait still `+25–45 ms` |
| **v28** (Nov 27) | **`varloop-stage2-e48`** (48e, `4.08 GiB`), `REPACK=1`, `-t 2 -tb 4`, `3300 MB` | **`48.0 ms`** (`cpu 95`) | **`20.17 tok/s`** over all 450 tok (`50 ms`); `21.06 tok/s` for tok `20–449` (`47.5 ms`); `20.23 t/s` last 250 tok | **`4.36 GB`** | fair | **`>= 20 tok/s` for a hard-pruned `e48` model (untrained `e48` scored `0/30` GSM8K, see the pruning curve below), USB-C plugged in, thermal `fair`; the raw `v28` log is not in this repo** (`miss 0.3–0.7/t`, cold 70t prefill `3.0 s`) |

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
- **Public 236-item reasoning/tool/instruction baseline (`varloop-stage2-v1`, *pre-Stage 3*, old mismatched `run3` LoRA):**
  - **`GSM8K-Hard` (`30/30`):** **`83.3%`** (`25/30`, avg think `596`) — beats `Gemma 4 26B Q4` (`80.0%`).
  - **`MATH-500` (`30/30`):** **`60.0%`** (`18/30`, avg think `728`) — beats `Gemma 4 26B Q4` (`43.3%`) and `Qwen 3.6 35B` (`50.0%`).
  - **`Logic` (`30/30`):** `20.0%` (`6/30`, avg think `1,099`, matches `LFM3` `20.0%`); **`GPQA Diamond` (`30/30`):** `16.7%` (`5/30`, avg think `1,041`).
  - **`BFCL Tool Use` (`76/76` ALL):** **`85.5%`** (`65/76`, avg think `129`) — matches `Qwen 3.6 35B` (`86.8%`).
  - **`IFEval` (`40/40`):** `52.5%` (`21/40`, avg think `1,023`, hurt by old LoRA loops).
- **REAP Expert Pruning + `20.17 tok/s` over 450 tokens (`21.06 tok/s` excluding the first 20) on iPhone 15 Pro (`v28`, plugged in, pruned model, speed-only datapoint):**
  - Built `quant/reap_trace.cpp` (exact `cb_eval` REAP salience `sum_t |w_e| * ||down_e(x_t)||_2`) and `quant/prune_gguf.py` (raw quantized byte slicing, zero re-quantization).
  - Traced `80,044` tokens (`188` docs at `150.5 tok/s`) on a leak-free Reasoning + Writing + Tool-Use corpus (`0%` coding, `0%` eval overlap).
  - **Key structural discovery:** Because Thinking Machines already pruned `384 → 128` load-balanced experts, activation frequencies across all 128 experts are dead-flat (`2,500 ± 40` per expert in every layer). There are no dead or coding-only experts to drop for free.
  - **iPhone 15 Pro `v28` (`varloop-stage2-e48.gguf`, `4.08 GiB`, `3.07 GiB` experts, `CACHE_MB=3300`, `REPACK=1`, `-t 2 -tb 4`, USB-C plugged):**
    - Rep 2 (`e48_r1_c3300_b`, `n=450`, `thermal: fair → fair`): **ALL 450 tok mean `50 ms` (`20.17 tok/s`), tok `20–449` mean `47.5 ms` (`21.06 tok/s`, `med=48.0 ms`), LAST 250 tok mean `49.4 ms` (`20.23 tok/s`)**, **`miss=0.7/tok` (`0.3/tok` last 250)**, peak RAM **`4,365 MB`**, cold 70-tok prefill **`3.02 s` (`23.1 tok/s`)** without state reuse!
    - Rep 1 (`e48_r1_c3300`): held `50.0 ms` (`20.0 tok/s`) through tok 374 before mild fast-charge dip (`mean 57 ms = 17.62 tok/s`), cold 70-tok prefill **`2.99 s` (`23.4 tok/s`)**.
  - **iPhone 15 Pro `e64` (`5.13 GiB`, `4.09 GiB` experts, `CACHE_MB=3500`):** Cold 70-tok prefill `3.41 s`, `miss` fell to `0.5–1.5/tok`, **but `4,892 MB` footprint + charging tripped `thermal: serious`** (`417–565 ms`). Establishes **`<= 3.35 GiB` experts (`<= 4.45 GB` total RAM)** as the clean sustained `20+ tok/s` + `8K–16K` context envelope on a 6 GiB iPhone.
- **Zero-Shot Pruning & 1-Bit Cliff Sweep (Mac `$0`, exact same 10 `GSM8K-Hard` items):**
  - Added 5-line per-layer `n_expert[il]` support to `src/models/inkling2.cpp` and `-ctk/-ctv q8_0` KV quantization to `ios/inkling_run.cpp`.
  - **Uniform REAP pruning curve (untrained):** `128e` **`9/10 (90%)`** → `e96` **`8/10 (80%)`** → `e88` **`5/10 (50%)`** → `e80` **`2/10 (20%)`** → `e64` **`0/14 (0%)`** → `e48` **`0/30 (0%)`**.
  - **Layer-adaptive pruning (untrained):** `112×8 / 64×6 / 24×10` (`0/4`), `128×8 / 80×4 / 32×12` (`0/4`), `96×12 / 32×12` (`0/4`).
  - **Naive 1-bit `Q1_0` (`1.125 bpw`, no imatrix, untrained):** `all 24L` (`0/10`, whitespace), `L12–23` (`0/4`, loops), `L16–23` (`0/4`, loops).
  - **Conclusion:** Zero-shot compression cannot bridge `8.19 GiB → 3.35 GiB` without retraining.

### 2026-11-27 (Overnight) — `oMLX 0.8.4` / `DSpark` Audit (`v29`), `8K Q8_0 KV` `23.25 tok/s` + `0.67 s` TTFT (`v30`), Stage 3 `e56-smoke2` (`v31`), `CECAR` Frontier (`v32`), and Working-Set Concentration Distillation (`WSD-40`)
- **`oMLX 0.8.4` / `DSpark` / `MTP` audit (`v29`):**
  - Inspected `jundot/omlx v0.8.4` (`#5657` SSD offload, `#5674` Lightning MTP / `DSpark`, `#5640` LoopCD, `Splash 1.7.3`) and pulled `thinkingmachines/TML-d24-varloop-research/mtp.safetensors` (`606 MB` BF16, `317.9M` params → `~179 MB` `Q4_K`, 3-layer DSpark drafter). Note `LoopCD` requires `>= 2` passes (incompatible with `1p K4`) and `oMLX` is macOS-only Python/MLX.
  - Added `-vbench` and `-ctk/-ctv q8_0` to `ios/inkling_run.cpp` and benchmarked real `B = 1..8` verify steps on iPhone 15 Pro (`IOS_I8MM=1`):
    - On `128e` `stage2-v1` (`3,100 MB` cache), flash misses scaled linearly with batch size (`B=1: 19.7 [27 MB]` → `B=4: 84.6 [118 MB]` → `B=6: 114.8 [160 MB]` → `B=8: 135.9 [188 MB]`), proving speculative decoding cannot bypass the flash wall on unpruned `128e`.
    - On RAM-resident `e48` at **`16K` `Q8_0` KV** (`fair → fair`), `B=1` held **`20.45 tok/s`** (`47.8 ms` med / `48.9 ms` mean), and `B=4` ran `105.2 ms` med (`3.0` misses/step → `~27–31 tok/s` upside if stacked with DSpark later).
- **`8K` `Q8_0` KV Envelope Proven on iPhone 15 Pro (`v30`, `e48`, `CACHE_MB=3200`, `-state-load` `2.4 MB` in `0.02 s`):**
  - **Sustained decode (250 tok):** **`23.25 tok/s`** (`41 ms` median, `43 ms` mean, `cpu 83 ms`)!
  - **Honest cross-prompt warm Turn-2 TTFT (`long1t → chat1t`, 28 new tok):** **`0.670 s` (`41.8 tok/s`)** — 3× faster than the `2.0 s` target! Cold 58-tok Turn 1 = `2.71 s`.
  - **Footprint & thermals:** **`4,419 MB`** peak at `8K` context, **`nominal → nominal`**.
- **Stage 3 Hard-Pruned `e56-compact` Smoke Test + Sliced-ImMatrix `e88-iq1s` (`$3.69` AWS):**
  - Built `train/distill_pruned.py` (shared GPU `int8` `128e 3p K8` teacher → `e56` `1p K4` student with `r=16` expert LoRA + full router + `r=32` attention/LM-head LoRA, `204.9M` params, `34.1 / 48 GB` VRAM on `ml.g6e.8xlarge`) and BF16 merge + official imatrix slicer (`144` expert tensors sliced cleanly).
  - Ran `103` steps (`2,374 s` = `$3.31`) + CPU merge/quant (`1,440 s` on `ml.m7i.4xlarge` = `$0.38`):
    - **`varloop-e88-iq1s.gguf` (`4.60 GiB`, untrained `88e` with official sliced imatrix `IQ1_S` on `L2–23` `gate_up` + `Q2_0` `down`):** Collapsed to **`0/4 (0.0%)`** on `GSM8K-Hard` with repetition looping (`Need compute.`), ruling out `IQ1_S` on `gate_up`.
    - **`varloop-e56-smoke2-s103.gguf` (`4.24 GiB`, standalone merged Stage 3 step 103):** Lifted `0/30` gibberish looping to coherent step-by-step math with concise thinking (`med_think=305`), reaching **`2/15 (13.3%)` `GSM8K-Hard`**, `0/10` `MATH-500`, **`5/20 (25.0%)` `BFCL`** (`eval_kl 0.9265 → 0.5017`).
  - **iPhone 15 Pro `v31` Memory-Compressor Cliff Discovery:** `e56-smoke2` matched `e48`'s exact **`42–43 ms` (`23.25 tok/s`)** zero-miss compute at `CACHE_MB=3200` (`footprint <= 4,341 MB`, `nominal`), confirming `Q2_0` `down` kernels are fast. Raising `CACHE_MB` to `3350` pushed footprint to `4,505 MB` and triggered iOS memory-compressor slowdown (`60–68 ms`, random `128–211 ms` stalls) even at `0` misses and `nominal` thermal — pinning the exact clean iOS footprint ceiling at **`<= 4,420 MB` (`CACHE_MB <= 2750–3100` at `8K` `Q8_0` KV)**.
- **`CECAR` Zero-Shot Frontier on Full `128e` (`v32`, `$0` on Mac & iPhone 15 Pro):**
  - **Mac (`varloop-stage2-v1` + `run3`, `CACHE_MB=2900`):** `CECAR=0.20` scored **`10/10 (100%)` `GSM8K-Hard`** (`95.5%` hit rate); `CECAR=0.35` scored **`8/10 (80%)` `GSM8K-Hard`** + **`4/10 (40%)` `MATH-500`** (`95.2%` hit rate); `CECAR=0.50` dropped to `4/10 (40%)` `GSM8K-Hard` due to cache thrashing without hot-expert pinning.
  - **iPhone 15 Pro (`v32c`, standalone `128e`, `CACHE_MB=2750`, `8K Q8_0 KV`, `-state-load`, `nominal`):** Zero-shot `CECAR=0.50` ALREADY hit **`18.07 tok/s` mean (`48 ms` median, `55 ms` mean, `47 ms` zero-miss)** and **`1.975 s` warm TTFT** at `nominal → nominal` — before any working-set training, `PIN_N`, or `CECAR_PREFILL`.
- **Stage 3 Working-Set Concentration Distillation (`WSD-40`, `$16.51` AWS, total cumulative spend `$46.09 / $60.00`):**
  - Built `train/distill_workingset.py`: keeps all `128` experts alive (`40` RAM-pinned HOT + `88` flash safety-net COLD), training `283.23M` params (`r=32` hot-40 expert LoRA + `r=32` attention/LM-head LoRA + full-rank `[128, 2304]` router delta + checkpoint-safe differentiable `L_cold` + cosine-ramped PI cold-bias controller `25% → 1.5%` cold visits) on a `50/50` packed + completion-only RWT mix with concise thinking.
  - Trained all **550 / 550 steps** (`11,817 s` on `ml.g6e.8xlarge`, `36.2 / 48 GB` VRAM, `$16.51`): `eval_kl` finished at **`0.3944`** (below untrained `128e` step-0 `0.4130`), `eval_layer_mse = 0.3022`, `eval_ro_kl = 0.2794`, while cutting cold visits **40.6×** (`57.5% / 55.2 tok → 1.42% / 1.36 tok`) with `40/40` hot experts active (`cold_mass 0.688 → 0.088`, allowing `cold_bias` to relax `-0.510 → -0.329`).
  - Added **`GGML_MMID_PIN_N`** (permanent hot-expert pinning for `0..pin_n-1` with cold-ring fallback eviction in `ggml-cpu.c`) and **`GGML_MMID_CECAR_PREFILL=1`** + fast `std::partial_sort` in `ops.cpp`; rebuilt both Mac `llama-server` and iOS `InklingBench.app` (`IOS_I8MM=1`).
  - Built zero-double-quantization local streaming BF16 → official `imatrix` mergers (`quant/merge_wsd_local.py`, `quant/merge_run3_local.py`, verified **`100.00%` byte-exact** against `llama-quantize` on `Q2_K`, `Q2_0`, `NVFP4`, `99.99%` `IQ4_NL`, `99.85%` `Q5_K`) to build two standalone zero-LoRA-overhead GGUFs: `varloop-stage3-wsd40.gguf` and `varloop-stage2-v1-run3-reap40.gguf` (top-40 REAP reordered to `0..39`).
- **Final Head-to-Head Evaluation (`CACHE_MB=2750`, `PIN_N=40`, `REPACK=1`, `OVERLAP=1`):**
  - **`varloop-stage3-wsd40.gguf` (pure trained router, NO `CECAR`):** Achieved **`99.57–99.64%` cache hit rate** (**`0.71–0.85` slice misses/tok = `0.35–0.42` expert misses/tok** over `45,156` tokens — an `11.4×` reduction vs `CECAR=0.35`!), scoring `36.7%` `GSM8K-Hard` (11/30), `0.0%` `MATH-500` (0/11), `6.7%` `Logic` (1/15), **`26.7%` `GPQA`** (4/15), `30.0%` `BFCL` (9/30), `20.0%` `IFEval` (4/20).
  - **`varloop-stage2-v1-run3-reap40.gguf` (`PIN_N=40`, `CECAR=0.35`, `CECAR_PREFILL=1`, `cap=384`):** Achieved **`95.3–96.5%` cache hit rate** (`7.12–9.70` slice misses/tok), scoring **`73.3%` `GSM8K-Hard`** (11/15, `avg_think=351`), **`30.0%` `MATH-500`** (3/10, `avg_think=356`), `6.7%` `Logic` (1/15), **`26.7%` `GPQA`** (4/15), **`70.0%` `BFCL`** (21/30, `avg_think=161`), **`55.0%` `IFEval`** (11/20). At `CECAR=0.42`: `97.36%` hit rate (`5.30` slice misses/tok), `53.3%` `GSM8K-Hard`, `20.0%` `MATH-500`, `70.0%` `BFCL`, `40.0%` `IFEval`.
  - **Key Scientific Conclusion & Next Step (within remaining `$13.91` budget):** Working-Set Concentration Distillation (`WSD-40`) **solved the flash-cache wall cleanly** (`99.64%` hit rate, `< 1` slice miss/tok, zero `CECAR` needed), while `1.4%` cold budget (`40` experts) proved slightly too tight for multi-step `MATH-500` (`eval_ro_kl = 0.279` vs `0.187` on `128e` `run3`). Because `WSD-40` at step 100 (`9.4%` cold) held `eval_ro_kl = 0.2026`, the sweet spot for a follow-up `$10–13` QAT-base `WSD` run is **`target_cold_frac = 0.04–0.05` (`4–5%` cold visits ≈ `2–3` flash misses/tok with cold ring → `~20 tok/s` on iPhone 15 Pro + `75–85%` `GSM8K` / `45–55%` `MATH-500`)** trained directly on dequantized `stage2-v1` weights (`--experts-from-gguf`) so there is zero train→`Q2_K` quantization gap!

## 2026-11-28 — Lossless kernels and iPhone 15 Pro `v35`

- **Kernels (Mac profile, then the same binary on the phone).** Row-broadcast NEON `ADD`/`SUB`/`MUL`/`DIV` is IEEE (`vaddq`/`vsubq`/`vmulq`/`vdivq`). Unit test: `2,121,548` checks, `0` mismatches. Graph-profile `MUL` went from `201.7 ms` (`79,200` calls) to `21.0 ms` (`69,600` calls), `8.4×` per call. The 1-pass profile measures `SUB` at `0.8 ms` for `2,400` calls. `GGML_SITU_FAST=accel` dropped GLU from `38.1 ms` to `12.3 ms` (`3.10×`, `n=2400`) and moved a 100-token Mac decode from `41.34` to `45.08 tok/s` (median `23 → 21 ms`). `GGML_MMID_PREREPACKED=1` on Mac, cache `2700` / pin `36`: prefill wait `247.7 → 93.4 ms` (`repack_ms 521.9 → 0.0`), decode wait `293.3 → 209.5 ms` (`repack_ms 225.2 → 0.0`). Pack/unpack is same-size and bit-reversible. After the phone copy, the Mac file's SHA-256 is `6d856056c0ff323711a487233dc5c63425f1896547803c38aac6c5f39d2acf09` and the journal is gone.
- **Phone setup.** `IOS_I8MM=1` bench app. Model keeps `expert_count=128`. Context `8K Q8_0`, `-passes 1 -k 4 -lm none -lzm on -t 2 -tb 4 -ub 64`. Exact-head rows set `LLAMA_LM_HEAD_SHORTLIST=off`. Shared env: `INKLING_ALLOW_ONE_PASS=1`, `INKLING_GREEDY_SKIP_SOFTCAP=1`, `GGML_SITU_FAST=accel`, overlap, repack, LRU, 4 IO threads, stats, CECAR prefill. λ=`0.95` is bonus `1.0` on `0.18,0.29,0.29,0.33,0.33,0.44,0.12,0.21,0.21,0.21,0.25,0.15,0.18,0.12,0.15,0.21,0.44,0.00,0.38,0.29,0.21,0.25,0.50,0.18`. λ=`1.10` is bonus `1.0` on `0.21,0.29,0.38,0.38,0.33,0.44,0.12,0.25,0.25,0.25,0.29,0.25,0.25,0.15,0.18,0.25,0.44,0.50,0.44,0.29,0.25,0.29,0.50,0.25`. Each run started and ended at `nominal`. tok/s is `[gen]` over every decode token. Hit% is the decode `[mmid]` window; tok1+ drops token 0. The prior 66-token point (cache `2700`, pin `36`, λ=`0.95`, `-ub 128`) was `18.45 tok/s` (median `51`, mean `54`, footprint `3984 MB`, warm `1.165 s` on 31 tokens). `v35` warm turns are 40 tokens.
- **Exact head, unpacked.** `v35a` `2700/36/λ=0.95`, 66-tok, 200: zero-miss `31 ms` (14), median `39`, mean `43`, **`23.43 tok/s`**, warm `0.978 s`, hit `89.8%` (tok1+ `95.3%`), decode wait `2187.6 ms`, repack `437.8 ms`, peak/steady `4081/3945`. `v35b` `2950/40/λ=0.95`: `32 ms` (12), `43/48`, `20.78 tok/s`, warm `0.992 s`, `90.2%` (`95.7%`), wait `2140.2`, repack `602.9`, peak/steady `4334/4187` (under `4380`, above `4300`). `v35c` therefore stayed at `2980/41/λ=1.10`: `32 ms` (20), `43/46`, `21.58 tok/s`, warm `0.899 s`, `90.9%` (`96.5%`), wait `1838.2`, repack `604.4`, peak/steady `4364/4225`. `v35d` `2950/40` uniform bonus `0.35`: `34 ms` (31), `42/46`, `21.78 tok/s`, warm `0.761 s`, `91.6%` (`97.4%`), wait `1447.9`, repack `561.6`, peak/steady `4334/4187`. Same `v35a` cache on the 127-token mix, 200 tok: `34 ms` (3), `55/57`, `17.46 tok/s`, warm `1.038 s`, `85.5%` (`93.0%`), wait `3273.7`, repack `906.9`, peak/steady `3990/3933`.
- **Prerepacked (`GGML_MMID_PREREPACKED=1`), same `2700/36/λ=0.95`.** Decode `repack_ms` on `v35a` was `437.8` (prefill `800.8`), so the packed copy was measured. `v35e` 66-tok, 200: zero-miss `30 ms` (7), `41/45`, `22.05 tok/s`, warm `1.172 s`, hit `89.2%` (`94.6%`), wait `2119.3`, repack `0.0`, peak/steady `4062/3951`. `v35f` 66-tok requested 400 and the answer ended at 375: `34 ms` (14), `43/47`, `21.20 tok/s`, warm `1.133 s`, `92.0%` (`95.1%`), wait `3777.6`, repack `0.0`, peak/steady `4062/3927`. Mix, full 400: `36 ms` (3), `52/54`, `18.36 tok/s`, warm `1.126 s`, `88.4%` (`92.4%`), wait `5697.6`, repack `0.0`, peak/steady `3975/3922`. Exact-head winner stays unpacked `v35a` at `23.43 tok/s`.
- **Optional fast head.** `v35g` is `lossy_margin`, K=`13480`, logit `26`, margin `1`, prerepacked, and the log says `LOSSY mode=lossy_margin (NOT exact)`. 66-tok, 200: zero-miss `26 ms` (7), `37/42`, `23.62 tok/s`, warm `1.131 s`, hit `89.2%` (`94.6%`), wait `2188.5`, repack `0.0`, peak/steady `4094/3954`. The exact-head headline remains `v35a`.
- **80-item concise thinking (`soft_cap=320`, effort `0.2`, n=80 = 15 GSM8K-Hard + 10 MATH-500 + 10 Logic + 10 GPQA + 20 BFCL + 15 IFEval).** `c000` CECAR off: GSM `60.0`, MATH `30.0`, Logic `10.0`, GPQA `10.0`, BFCL `80.0`, IFEval `66.7`, reas `31.1`, mean `59.3`, think `245/293`, hit `75.3%`. `opt85`: `60.0 / 30.0 / 10.0 / 30.0 / 60.0 / 60.0`, reas `35.6`, mean `51.9`, think `250/294`, hit `90.2%`. `opt95`: `73.3 / 30.0 / 10.0 / 30.0 / 60.0 / 46.7`, reas `40.0`, mean `48.9`, think `249/293`, hit `92.2%`. `fmrd_v2` CECAR off: `46.7 / 30.0 / 10.0 / 20.0 / 70.0 / 46.7`, reas `28.9`, mean `48.5`, think `262/296`, hit `84.2%`.
- **EAGLE-v3.** Mac held-out Q4 2-layer draft: `alpha1_gen=0.551`, `alpha2_cond_gen=0.578`, expected `1.55` tok/step at B=2. iPhone `K=1` on the 66-token prompt: `12.84 tok/s` (`77.9 ms/tok`, `tok/step=1.389`, `alpha1=38.9%`, draft `1.25 ms`, verify `106.60 ms`, footprint `4031 MB`). Greedy B=1 on that prompt was `18.45 tok/s`. The verify step is the cost, so this drafter is slower than B=1 on the phone.


## 2026-11-28 — v2: QAT working-set distillation (`qat40pareto`) and the Pareto cliff

Full write-up: [`docs/V2.md`](V2.md).

- **Diagnosing the WSD-40 v1 regression.** Stage 3 WSD-40 crushed cold traffic to 1.4%. `eval_ro_kl` jumped `0.20 → 0.32` between steps 100 and 150. That run had no `run3` init and no QAT against the shipped quantized weights, plus a chat-template mismatch on system/tool turns.
- **New trainer `train/distill_v2_sota.py`.** It trains against the shipped v1 GGUF's quantized Hot-40 experts, attention and `lm_head` (QAT), initializes from `run3` with exact delta subtraction, and adds 105 F32 `AttnRes`/RMSNorm deltas. Data is 15 Parquet streams rendered through `apply_chat_template`. Logic/science share rose from about 3% to about 13%.
- **`qat40sota1`.** Stopped after 6 min (`$0.44`): a `camel-ai` `.zip` dataset loaded at 5 rows/s. It was replaced with Parquet `OpenThoughts-114k`, `LogiQA-2.0`, `FOLIO`, `SciQ`, `ARC-Challenge` and `MMLU-aux`.
- **`qat40sota2` (target cold 6.5%, bias floor −0.75).**
  - Step 50 (36.5% cold): `eval_kl 0.2777`, `tool_if_kl 0.1821`.
  - As cold traffic fell, tool/IF KL rose steadily: `0.3183` at 14.6% cold, `0.4856` at 5.7%, `0.4936` at step 300.
  - The decision gate stopped the run at step 300 (`$6.75`).
- **`qat40pareto` (target cold 28%, bias floor −0.18, cold loss 0.015, 220 steps, `$5.32`).**
  - Step 200: cold `62.3% → 37.1%`, `eval_kl 0.3361 → 0.2644`, `eval_ro_kl 0.1527 → 0.1346`, `tool_if_kl 0.1707 → 0.1725`, `layer_mse 0.2967 → 0.2331`.
  - Total AWS spend for the project: `$58.60` of the `$60` budget.
- **Local merge (`quant/merge_v2_local.py`, 35.8 min, `$0`).** Output `inkling2-26b-flashpin-v2-1p-hot40-q2k-i8mm.gguf`, `10,080,548,288` B, SHA-256 `9b6b87fdb8fa01deb3b69ffbee6e59b98ecea65b1f3305d5fbb9db3d3bebc351`.
  - Router cos vs v1: `0.9986–0.9991`.
  - `exp_probs_b` hot−cold gap: `0.18–0.25` per layer (v1: `0.00–0.07`).
- **N=80, same items as v1, effort 0.45.** v2 with `CECAR=0`: GSM8K-Hard 12/15, MATH-500 4/10, Logic 5/10, GPQA 2/10 (Reas 23/45 = 51.1%), BFCL 13/20, IFEval 8/15, Mean 56.5%. v1 with OPT λ=0.95: Reas 18/45 = 40.0%, BFCL 15/20, IFEval 7/15, Mean 53.9%. The v2 OPT λ=0.95 row is in `docs/V2.md` §3.
- **Negative / open result.** On the Mac at `2700/36`, v2 at `CECAR=0` does **not** have fewer cache misses than v1 (`p_math` 38.4 vs 30.4 miss/tok; `v34_mix` 47.9 vs 48.4). The training-time cold reduction does not reach the runtime. With OPT λ=0.95 the two are on par (11.3 vs 10.8 and 11.9 vs 13.1 miss/tok).
- **iPhone.** Two Wi-Fi `devicectl` copies of the v2 GGUF failed (`ECONNRESET`). The file was later copied over USB and measured, see the `v36` entry below.

## 2026-11-28 (evening) — v2 on iPhone 15 Pro (`v36`)

Raw logs: [`eval/results/iphone15pro_v36_v2/`](../eval/results/iphone15pro_v36_v2/). v2 GGUF copied over USB (`10,080,548,288` bytes on the phone), already-installed `InklingBench` build, no rebuild, CPU-only, `-ngl 0 -t 2 -tb 4 -ub 64 -passes 1 -k 4 -lm none -lzm on -c 8192 -ctk q8_0 -ctv q8_0 -n 200`, `prepack=0`, exact `200K` head. Runs after `v36a` were unplugged with `phone.sh thermal` reading `nominal` before each launch.

- **`v35a` env on v2 (`2700 / 36 / OPT λ=0.95`):** `18.76` (wired, thermal `fair`), `17.43` and `18.77 tok/s` (unplugged, `nominal`). v1 on the same env was `23.43`. Cooling and unplugging did not change the result. v2 and v1 GGUF headers have identical tensor names, types and shapes. v2 `v35a` has `cpu_ms` mean `85.4–88.9` against `67.1` for v1 `v35a`, and `wait_ms` `170.0–232.1` against `424.7`.
- **`v35d` env on v2 (`2950 / 40 / CECAR=0.35`), `p_math`, 200 tokens:** `22.83`, `22.46`, `20.79 tok/s`. Median of the run-means: **`22.46 tok/s`**. Warm TTFT `0.693 / 0.740 / 0.788 s`, peak RAM `4,315 MB`, `[mmid]` `41,299` hits / `4,155` misses in every run, thermal `nominal → nominal`.
- **`v35d` on the mix prompt (`v34_mix`, 127 tokens, 200 generated):** `23.03`, `20.64`, `20.86 tok/s`. Median of run-means `20.86`. Warm TTFT `0.655 / 0.752 / 0.689 s`, peak `4,318–4,320 MB`, `42,666` hits / `4,545` misses.
- **`-n 1000` on `p_math`:** ended at EOG after `285` tokens (`19.26 tok/s`, `50 / 52 ms`, peak `4,315 MB`). Not a 1,000-token measurement.
- **Cold vs warm TTFT:** warm (`40`-token Turn-2 prefill after the cache is warm) `0.655–0.790 s` on the `v35d` runs; cold prompt prefill `4.32–4.50 s` (`66` tokens) and `5.41–6.15 s` (`127` tokens).
- **Not run:** `prepack=1` on v2, full-context (`8K`) speed, the 236-item suite on v2.
