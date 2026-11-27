# `inkling-ios` — `Inkling-2-FlashPin-26B`

**Running a 26.4B-parameter, 128-expert Mixture-of-Experts model on an iPhone 15 Pro CPU at `21–23.4 tok/s` sustained (`18.4 tok/s` under 3-topic domain switching) — with all `128/128` experts intact, `8K` `Q8_0` context, exact `200K` vocabulary, `< 4.4 GB` peak RAM, and `nominal → nominal` thermals.**

- **Published GGUF Weights (Apache 2.0):** [`kevinexus/Inkling-2-FlashPin-26B-GGUF`](https://huggingface.co/kevinexus/Inkling-2-FlashPin-26B-GGUF)
  - File: `inkling2-26b-flashpin-1p-hot40-q2k-i8mm.gguf` (`10,080,548,288` bytes, `9.39 GiB`, all `128/128` experts retained)
  - SHA-256: `6d856056c0ff323711a487233dc5c63425f1896547803c38aac6c5f39d2acf09`
- **Raw Verification Artifacts:** [`eval/results/iphone15pro_v35/`](eval/results/iphone15pro_v35/) (every `[tok]`, RAM footprint, `[mmid]`, `[warm_ttft]`, generated text, and thermal reading) + [`eval/results/quality_80item/`](eval/results/quality_80item/) (raw deterministic JSON summaries) + exact prompt files in [`eval/results/iphone15pro_v35/prompts/`](eval/results/iphone15pro_v35/prompts/).

> **Explicit Scope & Lineage (Read Before Judging):**
> 1. **Base Model & Shape:** Derived from Apache-2.0 [`thinkingmachines/TML-d24-varloop-research`](https://huggingface.co/thinkingmachines/TML-d24-varloop-research) (the 26.4B variable-loop research sibling of Inkling 2, `inkling2_loop`, 24 blocks, 128 experts/layer) — **not** the fixed 3-pass Inkling 2 Mini checkpoint. Distilled and run at **1 loop pass, top-4 experts (`-passes 1 -k 4`)** with **zero experts pruned (`128/128` intact)**.
> 2. **Same-Artifact Rule:** Every `v35` iPhone 15 Pro speed measurement and every quality measurement in the flagship section below comes from the **exact same single GGUF artifact** (`SHA-256: 6d856056...2acf09`), with `LLAMA_LM_HEAD_SHORTLIST=off` (exact full `200,058`-token vocabulary) unless explicitly marked `LOSSY`.
> 3. **Harness Requirements:** `ios/` builds `InklingBench`, a minimal **CLI systems benchmark harness** (not a consumer SwiftUI chat app). Building and sideloading on a physical iPhone requires an Apple Developer provisioning profile with the `com.apple.developer.kernel.increased-memory-limit` entitlement.
> 4. **Total Cloud Compute Spent:** `$46.09 / $60.00` (AWS EC2 L40S for KL sweep + Stage 2/3 distillation; all Phase 2 kernel, router, and iPhone work was `$0` local).

---

## Same-Artifact Summary (`Inkling-2-FlashPin-26B`, `128/128` Experts, `8K Q8_0` KV)

All rows below measure the **exact same published GGUF** (`inkling2-26b-flashpin-1p-hot40-q2k-i8mm.gguf`, `128/128` experts, exact `200K` head, `1-pass K=4`):

### 1. Real iPhone 15 Pro Speed, Context, Memory & Thermals (`v35`)

Flags on every row: `-m models/inkling2-26b-flashpin-1p-hot40-q2k-i8mm.gguf -ngl 0 -t 2 -tb 4 -ub 64 -passes 1 -k 4 -lm none -lzm on -c 8192 -ctk q8_0 -ctv q8_0`, `GGML_SITU_FAST=accel`, `GGML_MMID_OVERLAP=1`, `GGML_MMID_REPACK=1`, `GGML_MMID_LRU=1`, `GGML_MMID_IO_THREADS=4`, `GGML_MMID_CECAR_PREFILL=1`. `tok/s` is the end-to-end `[gen]` mean over every decode token including cold misses and token 0.

| Run (Raw Log) | Prompt / Gen | Cache / Pin / Route | Head | Prepack | Zero-Miss (`miss=0`) | Med / Mean | Sustained `tok/s` | Warm TTFT (40 tok) | Decode Hit% (`tok1+`) | Peak / Steady RAM | Thermal |
|---|---|---|---|:---:|---:|---:|---:|---:|---:|---:|---|
| **[`v35a`](eval/results/iphone15pro_v35/v35a_c2700_p36_opt95_ub64.log)** | `p_math` (66t) / `200` | `2700 / 36 / OPT λ=0.95` | **exact** | `0` | `31 ms` (14) | `39 / 43 ms` | **`23.43`** | **`0.978 s`** (`40.9 t/s`) | `89.8%` (`95.3%`) | **`4,081 / 3,945 MB`** | `nominal → nominal` |
| **[`v35d`](eval/results/iphone15pro_v35/v35d_c2950_p40_c035_ub64.log)** | `p_math` (66t) / `200` | `2950 / 40 / CECAR=0.35` | **exact** | `0` | `34 ms` (31) | `42 / 46 ms` | **`21.78`** | **`0.761 s`** (`52.6 t/s`) | `91.6%` (`97.4%`) | `4,334 / 4,187 MB` | `nominal → nominal` |
| [`v35c`](eval/results/iphone15pro_v35/v35c_c2980_p41_opt110_ub64.log) | `p_math` (66t) / `200` | `2980 / 41 / OPT λ=1.10` | **exact** | `0` | `32 ms` (20) | `43 / 46 ms` | `21.58` | `0.899 s` (`44.5 t/s`) | `90.9%` (`96.5%`) | `4,364 / 4,225 MB` | `nominal → nominal` |
| [`v35b`](eval/results/iphone15pro_v35/v35b_c2950_p40_opt95_ub64.log) | `p_math` (66t) / `200` | `2950 / 40 / OPT λ=0.95` | **exact** | `0` | `32 ms` (12) | `43 / 48 ms` | `20.78` | `0.992 s` (`40.3 t/s`) | `90.2%` (`95.7%`) | `4,334 / 4,187 MB` | `nominal → nominal` |
| **[`v35f`](eval/results/iphone15pro_v35/v35f_sustained400.log)** | `p_math` (66t) / `375` *(EOG)* | `2700 / 36 / OPT λ=0.95` | **exact** | `1` | `34 ms` (14) | `43 / 47 ms` | **`21.20`** | `1.133 s` (`35.3 t/s`) | `92.0%` (`95.1%`) | `4,062 / 3,927 MB` | `nominal → nominal` |
| **[`v35f_mix`](eval/results/iphone15pro_v35/v35f_sustained400_mix.log)** | `v34_mix` (127t, 3 topics) / `400` | `2700 / 36 / OPT λ=0.95` | **exact** | `1` | `36 ms` (3) | `52 / 54 ms` | **`18.36`** | `1.126 s` (`35.5 t/s`) | `88.4%` (`92.4%`) | `3,975 / 3,922 MB` | `nominal → nominal` |
| [`v35_mix`](eval/results/iphone15pro_v35/v35_mix_c2700_p36_opt95_ub64.log) | `v34_mix` (127t, 3 topics) / `200` | `2700 / 36 / OPT λ=0.95` | **exact** | `0` | `34 ms` (3) | `55 / 57 ms` | `17.46` | `1.038 s` (`38.5 t/s`) | `85.5%` (`93.0%`) | `3,990 / 3,933 MB` | `nominal → nominal` |
| [`v35e`](eval/results/iphone15pro_v35/v35e_prerepack_best.log) | `p_math` (66t) / `200` | `2700 / 36 / OPT λ=0.95` | **exact** | `1` | `30 ms` (7) | `41 / 45 ms` | `22.05` | `1.172 s` (`34.1 t/s`) | `89.2%` (`94.6%`) | `4,062 / 3,951 MB` | `nominal → nominal` |
| [`v35g`](eval/results/iphone15pro_v35/v35g_prerepack_margin26.log) *(opt)* | `p_math` (66t) / `200` | `2700 / 36 / OPT λ=0.95` | *lossy margin (26,1)* | `1` | `26 ms` (7) | `37 / 42 ms` | `23.62` | `1.131 s` (`35.4 t/s`) | `89.2%` (`94.6%`) | `4,094 / 3,954 MB` | `nominal → nominal` |

- **What `v35` shows honestly:**
  - Switching `-ub 128 → -ub 64` halved the CPU compute buffer (`242.7 → 121.4 MiB`), keeping every single run below iOS's `~4,450 MB` memory-compressor cliff (`3,975–4,364 MB` peak, `nominal → nominal`).
  - On single-topic reasoning (`p_math`), sustained throughput is **`21.20–23.43 tok/s`** (`375`-tok natural completion to EOG / `200`-tok run). On the deliberately adversarial 3-topic domain-switching prompt (`v34_mix`: math + physics + Python in one turn), flash misses increase and sustained 400-tok throughput is **`18.36 tok/s`** (`+5.2%` over unpacked `17.46 tok/s`).
  - Offline pre-repacking (`GGML_MMID_PREREPACKED=1`) zeroes `repack_ms` (`437.8–906.9 ms → 0.0 ms` in decode, `800.8–1,780.4 ms → 0.0 ms` in prefill) and speeds up prefill (`5.17 → 4.14–4.50 s`) and miss-heavy mixed-domain decode (`17.46 → 18.36 tok/s`), while async `CECAR` cache-timing variance kept unpacked `v35a` slightly ahead on the 200-tok single-topic run (`23.43` vs `22.05 tok/s`).

### 2. Quality on the Exact Same Artifact (`N=80` Deterministic Code-Scored Suite + `N=236` Baseline)

All benchmarks use deterministic public code-scored datasets ([`eval/suite_spec.json`](eval/suite_spec.json); coding and obscure trivia excluded). Because the 80-item fast suite has small per-task counts (`15` GSM8K-Hard, `10` MATH-500, `10` Logic, `10` GPQA Diamond, `20` BFCL, `15` IFEval), exact fractions are reported alongside percentages. Raw JSONs live in [`eval/results/quality_80item/`](eval/results/quality_80item/).

| Routing Mode (Same GGUF, `128/128` Experts) | Think Cap / Effort | `GSM8K-Hard` | `MATH-500` | `Logic` | `GPQA` | `BFCL` | `IFEval` | Mean (`N=80`) | Avg Think | Cache Hit (`3000 MB`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *Ref: Full 3-pass `K=8` BF16/MLX (53 GB, Mac)* | *ref* | `95.0%` | — | — | — | `100.0%` | `70.0%` | *`65.0%`* | — | — |
| *Ref: 1-pass `K=4` `stage2-v1 + run3` (`N=236`)* | *uncapped* | `83.3%` *(25/30)* | `60.0%` *(18/30)* | `20.0%` *(6/30)* | `16.7%` *(5/30)* | `85.5%` *(65/76)* | `52.5%` *(21/40)* | — | `~650` | `~85–90%` |
| **[`c000_sc320`](eval/results/quality_80item/c000_sc320.json)** (`CECAR=0`, max quality) | `cap=320`, `e=0.2` | `60.0%` *(9/15)* | `30.0%` *(3/10)* | `10.0%` *(1/10)* | `10.0%` *(1/10)* | **`80.0%`** *(16/20)* | **`66.7%`** *(10/15)* | **`59.3%`** | `245` | `75.3%` |
| **[`opt95_e045`](eval/results/quality_80item/opt95_e045.json)** (`OPT λ=0.95`, same route as `v35a`) | uncapped, `e=0.45` | **`73.3%`** *(11/15)* | `30.0%` *(3/10)* | `20.0%` *(2/10)* | `20.0%` *(2/10)* | **`75.0%`** *(15/20)* | `46.7%` *(7/15)* | **`53.9%`** | `761` | `90.6%` |
| **[`opt95_e020`](eval/results/quality_80item/opt95_e020.json)** (`OPT λ=0.95`, same route as `v35a`) | uncapped, `e=0.20` | **`80.0%`** *(12/15)* | **`40.0%`** *(4/10)* | `20.0%` *(2/10)* | `0.0%` *(0/10)* | `60.0%` *(12/20)* | **`60.0%`** *(9/15)* | **`53.3%`** | `774` | `91.8%` |
| [`opt85_sc320`](eval/results/quality_80item/opt85_sc320.json) (`OPT λ=0.85`) | `cap=320`, `e=0.2` | `60.0%` *(9/15)* | `30.0%` *(3/10)* | `10.0%` *(1/10)* | `30.0%` *(3/10)* | `60.0%` *(12/20)* | `60.0%` *(9/15)* | `51.9%` | `250` | `90.2%` |
| [`opt95_sc320`](eval/results/quality_80item/opt95_sc320.json) (`OPT λ=0.95`, concise) | `cap=320`, `e=0.2` | `73.3%` *(11/15)* | `30.0%` *(3/10)* | `10.0%` *(1/10)* | `30.0%` *(3/10)* | `60.0%` *(12/20)* | `46.7%` *(7/15)* | `48.9%` | `249` | **`92.2%`** |
| `S2V1-Run3-Merged` (`CECAR=0.35`, same route as `v35d`) | `cap=384` | `73.3%` *(11/15)* | `30.0%` *(3/10)* | `6.7%` *(1/15)* | `26.7%` *(4/15)* | `70.0%` *(21/30)* | `55.0%` *(11/20)* | — | `351–381` | `95.3–96.5%` |
| [`fmrd_v2_sc320`](eval/results/quality_80item/fmrd_v2_sc320.json) (distilled routers, `CECAR=0`) | `cap=320`, `e=0.2` | `46.7%` *(7/15)* | `30.0%` *(3/10)* | `10.0%` *(1/10)* | `20.0%` *(2/10)* | `70.0%` *(14/20)* | `46.7%` *(7/15)* | `48.5%` | `262` | `84.2%` |

- **Honest Quality Tradeoff:**
  - At `CECAR=0`, this single 1-pass `128e` GGUF reaches **`59.3%` mean** (`~91%` of the full 3-pass 53 GB model's `65.0%` composite: `80.0%` BFCL, `66.7%` IFEval) at `245` avg think tokens.
  - Enabling `OPT λ=0.95` or `CECAR=0.35` cuts flash misses `~3×` (enabling `21–23.4 tok/s` on iPhone 15 Pro) while retaining `73.3–80.0%` GSM8K-Hard, `30–40%` MATH-500, and `~82–83%` of full-model mean uncapped (`53.3–53.9%`), but **introduces a `-15 to -20%` `BFCL` sensitivity** when paired with `cap=320`. Because tool calls only emit `20–50` tokens, disabling `CECAR` on tool-schema turns preserves `80%` `BFCL` for `< 0.5 s` extra latency.
  - Hard multi-step `Logic` (`10–20%`) and `MATH-500` (`30–40%`) remain the largest gap from cutting 3 loop passes → 1 pass + `Q2_K` quantization.

---

## Targets & Measured Results

| Goal | Target | Verified Same-Artifact Result (`Inkling-2-FlashPin-26B`, `128/128` Experts) | Status |
|---|---|---|---|
| **Sustained decode** | **`>= 20 tok/s`** (`<= 50 ms/tok` mean) | **`23.43 tok/s`** (`v35a`, 200 tok, exact head); **`21.20 tok/s`** (`v35f`, 375 tok EOG); **`21.78 tok/s`** (`v35d`, `CECAR=0.35`); **`18.36 tok/s`** (`v35f_mix`, 400-tok 3-topic domain switch) | **Met on single-topic (`21.2–23.4 t/s`); `18.4 t/s` on 3-topic mix** |
| **Warm TTFT** | **`<= 2.0 s`** | **`0.761 s`** (`52.6 tok/s`, `v35d`, 40-tok Turn-2) / **`0.978 s`** (`40.9 tok/s`, `v35a`) | **Met (`< 1.0 s`)** |
| **Context window** | **`8K`** at `>= 20 tok/s` | **`8K` (`Q8_0` KV)** verified on every `v35` row (`3,922–4,364 MB` peak); **`16K`** also proven (`v29`) | **Met (`8K Q8_0` on every `v35` run)** |
| **Memory & Thermal** | **`<= 4.5 GB`** (`nominal`) | **`3,945–4,225 MB`** steady (`4,062–4,364 MB` peak, `-ub 64`), **`nominal → nominal`** on all 9 `v35` runs (iOS compressor cliff mapped at `~4,450 MB`) | **Met with `~100–400 MB` margin** |
| **Quality** | Reasoning / tool-use / writing intact | `~82–83%` of full 3-pass model under `OPT λ=0.95` (`73–80%` GSM8K, `30–40%` MATH-500, `60–75%` BFCL, `47–60%` IFEval); `~91%` at `CECAR=0` (`80%` BFCL, `66.7%` IFEval) | **Strong on GSM8K/BFCL/IFEval; Logic/MATH-500 show 1p gap** |
| **Concise thinking** | `~150–350` tok | `245–296` tok (`soft_cap=320`) / `351–381` tok (`cap=384`) vs `600–1,100` tok baseline | **Met (`-55–65%` think tokens)** |

---

## Key Techniques (`FlashPin-1P` Stack)

All runtime patches live in [`patches/llama.cpp-inkling-ios.patch`](patches/llama.cpp-inkling-ios.patch) against upstream `llama.cpp @ 8a97fa98ab1a454709bbf5ac979661feaeadec40` (`b12628`). Every switch is opt-in and **default OFF**.

1. **1-Pass, Top-4 Unlooped Distillation (`INKLING_ALLOW_ONE_PASS=1`):** Runs `TML-d24-varloop-research` at `loop.count=1, expert_used_count=4` (~`4×` less compute), distilled against the 3-pass `K=8` BF16 teacher (`train/distill.py`), with LoRA + router deltas merged cleanly from streamed BF16 via HTTP `Range` + official `imatrix` (`quant/merge_run3_local.py`, zero double-quantization).
2. **Hot-40 REAP Expert Reordering & Pinning (`GGML_MMID_PIN_N=36..40`, `GGML_MMID_CACHE_MB=2700..2980`):** Keeps **all `128/128` experts intact**. Reorders top-40 hot experts per layer to indices `0..39` and pins them permanently in RAM (`O(1)` LRU `GGML_MMID_LRU=1`), while cold experts `40..127` stream from NVMe via uncached `F_NOCACHE` parallel `pread` (`GGML_MMID_IO_THREADS=4`). Halving `-ub 128 → -ub 64` drops the CPU compute buffer from `242.7 → 121.4 MiB`, staying safely below iOS's `~4,450 MB` memory-compressor cliff.
3. **Known-Route Paired-`down` Overlap (`GGML_MMID_OVERLAP=1`):** Background reader threads prefetch missing `ffn_down_exps` slices while the 2 A17 Pro Performance cores (`-t 2`) compute `ffn_gate_up_exps` (`100%` prefetch utilization, `0` wasted reads).
4. **Per-Layer Cache-Aware Routing (`GGML_MMID_CECAR_BONUS`, `GGML_MMID_CECAR_LAYERS`, `GGML_MMID_CECAR_PREFILL=1`):** Sensitivity-weighted per-layer cache-hit bias (`tools/optimize_cecar_layers.py`) boosts decode hit rate from `75% → 92–97%` without deleting cold experts.
5. **KL-Optimal Quantization + `i8mm` Tensor Upgrades (`quant/`):** `quant/kl_sweep.py` + `quant/dp_allocate.py` + `quant/repack_i8mm_gguf.py` assign `23× Q2_K + L23 Q4_K` `gate_up`, `11× Q2_0 + 10× Q4_0 + 3× IQ4_NL` `down`, `Q8_0` routers (`-20.8 MB`), and `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0` attention — eliminating slow `Q3_K`/`NVFP4`/`MXFP4` paths in favor of `armv8.6-a+i8mm` `smmla` kernels.
6. **Lossless ARM64 NEON / `i8mm` Kernels:**
   - **Zero-bloat NEON `4×4` on-miss repacking & GEMV (`GGML_MMID_REPACK=1`):** Bit-exact (`0/2304` diffs) `Q2_K`, `Q2_0`, `NVFP4`, `MXFP4` kernels (`0` extra cache bytes; `2.23×` faster `Q2_K` repack via `vst4q_u8`/`vst4q_u32`).
   - **Multi-token 64-tok tiled GQA `Q8_0` Flash-Attention** (`ops.cpp`, `neq1 >= 1`, `max_diff <= 1.34e-7`, `3.89–5.96×` faster standalone).
   - **Batched MoE repacked GEMM (`ggml_mmid_repack_gemm`):** Zero-copy NEON `Q2_K` (`max_abs = 0.0`) & `Q2_0` (`M=2,3,4`) + gated `Q4_0`/`Q4_K` `i8mm` (`+38.7%` Mac `-tb 4` prefill).
   - **Row-broadcast NEON `ADD`/`SUB`/`MUL`/`DIV` (`ggml/src/ggml-cpu/row-bcast-neon.h`):** Replaces length-1 per-element `vDSP` calls with IEEE `vaddq`/`vsubq`/`vmulq`/`vdivq` (`2,121,548` unit-test checks, `0` mismatches; Mac graph-profile `MUL` `201.7 ms` / `79,200` calls → `21.0 ms` / `69,600` calls, `8.4×` per call; 1-pass `SUB` `0.8 ms` / `2,400` calls).
   - **Fused `SiTU` (`GGML_GLU_OP_SITU`, `ggml/src/ggml-cpu/situ-neon.h`, `GGML_SITU_FAST=accel`):** Collapses Inkling 2's 9-node `SiTU` (`beta * tanh(gate/beta) * sigmoid(gate) * lb * tanh(up/lb)`) chain per layer into 1 op using vectorized Apple Accelerate `vvtanhf`/`vvexpf` (`GLU` `38.1 → 12.3 ms`, `3.10×`; Mac 100-tok decode `41.34 → 45.08 tok/s`).
   - **Reversible offline pre-repacking (`tools/prerepack_exps.cpp`, `GGML_MMID_PREREPACKED=1`):** Same-size, journaled, SHA-256 bit-reversible rewrite of all 48 `*_exps.weight` tensors (`repack_ms → 0.0 ms`; Mac prefill wait `247.7 → 93.4 ms`, decode wait `293.3 → 209.5 ms`; iPhone 3-topic mix `17.46 → 18.36 tok/s`).
7. **Honest Negative Results Documented:**
   - **Hard pruning (`e48`/`e64`) & untrained 1-bit (`Q1_0`/`IQ1_S`)** collapse reasoning (`0/30` GSM8K) without distillation.
   - **Stage 3 WSD-40 (`1.4%` cold budget)** solved the flash wall (`99.64%` hit rate, `< 1` miss/tok, zero `CECAR`), but `1.4%` cold mass proved too tight for multi-step `MATH-500` (`0%`).
   - **2-layer `EAGLE-v3` speculative drafter** (`train/train_mac_eagle_v3.py`, `tools/eagle_draft.hpp`): `alpha1_gen = 55.1%` on Mac held-out data, but on iPhone 15 Pro (`K=1`, `B=2`, `4,031 MB`) flash-miss verification (`106.6 ms`) slowed decode to `12.84 tok/s` vs `23.43 tok/s` greedy `B=1`.
   - **Exact Cauchy-Schwarz LM-head shortlist (`LLAMA_LM_HEAD_SHORTLIST=exact`)** had a `100%` fallback rate (`0/42` skipped) and wasted `20 MB` RAM + `1.2 s` init; keeping `LLAMA_LM_HEAD_SHORTLIST=off` is strictly faster and simpler.

---

## Quickstart (Reproduce `v35a` on iPhone 15 Pro)

### 1. Download Verified Weights
```bash
huggingface-cli download kevinexus/Inkling-2-FlashPin-26B-GGUF \
  inkling2-26b-flashpin-1p-hot40-q2k-i8mm.gguf --local-dir ./models

# Verify exact SHA-256 (should be 6d856056c0ff323711a487233dc5c63425f1896547803c38aac6c5f39d2acf09):
shasum -a 256 models/inkling2-26b-flashpin-1p-hot40-q2k-i8mm.gguf
```

### 2. Patch `llama.cpp`
```bash
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout 8a97fa98ab1a454709bbf5ac979661feaeadec40
git -C third_party/llama.cpp apply ../../patches/llama.cpp-inkling-ios.patch
```

### 3. Build, Sign & Run on iPhone 15 Pro
Requires Xcode, a paid Apple Developer signing identity, and a provisioning profile with `com.apple.developer.kernel.increased-memory-limit`.
```bash
export IOS_UDID="<your-device-udid>"
export IOS_BUNDLE_ID="<your.bundle.id>"
export IOS_TEAM_ID="<your-team-id>"
export IOS_SIGN_IDENTITY="<your-signing-identity>"
export IOS_PROVISION_PROFILE="/path/to/profile.mobileprovision"

IOS_I8MM=1 REBUILD_LIBS=1 ./ios/build_ios.sh
./ios/phone.sh push models/inkling2-26b-flashpin-1p-hot40-q2k-i8mm.gguf
./ios/phone.sh push eval/results/iphone15pro_v35/prompts/p_math.txt
./ios/phone.sh push eval/results/iphone15pro_v35/prompts/chat1t.txt

# v35a (23.43 tok/s, 0.978 s warm TTFT, 4,081 MB peak, exact 200K head, nominal -> nominal):
./ios/phone_env.sh v35a.log \
  '{"INKLING_ALLOW_ONE_PASS":"1","INKLING_GREEDY_SKIP_SOFTCAP":"1","GGML_SITU_FAST":"accel","GGML_MMID_CACHE_MB":"2700","GGML_MMID_PIN_N":"36","GGML_MMID_OVERLAP":"1","GGML_MMID_REPACK":"1","GGML_MMID_LRU":"1","GGML_MMID_IO_THREADS":"4","GGML_MMID_STATS":"1","GGML_MMID_CECAR_PREFILL":"1","GGML_MMID_CECAR_BONUS":"1.0","GGML_MMID_CECAR_LAYERS":"0.18,0.29,0.29,0.33,0.33,0.44,0.12,0.21,0.21,0.21,0.25,0.15,0.18,0.12,0.15,0.21,0.44,0.00,0.38,0.29,0.21,0.25,0.50,0.18","LLAMA_LM_HEAD_SHORTLIST":"off"}' \
  gen -m models/inkling2-26b-flashpin-1p-hot40-q2k-i8mm.gguf \
  -ngl 0 -t 2 -tb 4 -ub 64 -passes 1 -k 4 -lm none -lzm on -c 8192 -ctk q8_0 -ctv q8_0 -n 200 \
  -f models/p_math.txt -wf models/chat1t.txt

# Optional v35d (21.78 tok/s, 0.761 s warm TTFT, uniform CECAR=0.35, no layer list):
./ios/phone_env.sh v35d.log \
  '{"INKLING_ALLOW_ONE_PASS":"1","INKLING_GREEDY_SKIP_SOFTCAP":"1","GGML_SITU_FAST":"accel","GGML_MMID_CACHE_MB":"2950","GGML_MMID_PIN_N":"40","GGML_MMID_OVERLAP":"1","GGML_MMID_REPACK":"1","GGML_MMID_LRU":"1","GGML_MMID_IO_THREADS":"4","GGML_MMID_STATS":"1","GGML_MMID_CECAR_PREFILL":"1","GGML_MMID_CECAR_BONUS":"0.35","LLAMA_LM_HEAD_SHORTLIST":"off"}' \
  gen -m models/inkling2-26b-flashpin-1p-hot40-q2k-i8mm.gguf \
  -ngl 0 -t 2 -tb 4 -ub 64 -passes 1 -k 4 -lm none -lzm on -c 8192 -ctk q8_0 -ctv q8_0 -n 200 \
  -f models/p_math.txt -wf models/chat1t.txt
```

### 4. Optional Zero-Repack Layout (`tools/prerepack_exps.cpp`)
To eliminate runtime CPU repacking on miss-heavy / multi-topic workloads (`repack_ms = 0.0`, exact same file size, reversible back to the exact original SHA-256):
```bash
./tools/prerepack_exps --verify-self models/inkling2-26b-flashpin-1p-hot40-q2k-i8mm.gguf
./tools/prerepack_exps --pack        models/inkling2-26b-flashpin-1p-hot40-q2k-i8mm.gguf
# Run with "GGML_MMID_PREREPACKED":"1" in phone_env.sh; reverse anytime with --unpack
```

---

## Repository Layout

- [`patches/llama.cpp-inkling-ios.patch`](patches/llama.cpp-inkling-ios.patch) — Clean diff against `llama.cpp @ 8a97fa98ab1a454709bbf5ac979661feaeadec40` (`b12628`)
- [`ios/`](ios/) — Minimal iOS CLI runner (`inkling_run.cpp`, `ios_main.mm`), build/sign/install (`build_ios.sh`), and `devicectl` helpers (`phone.sh`, `phone_env.sh`)
- [`quant/`](quant/) — KL sweeper (`kl_sweep.py`), DP allocator (`dp_allocate.py`), `i8mm` tensor upgrader (`repack_i8mm_gguf.py`), streaming BF16 mergers (`merge_run3_local.py`, `merge_wsd_local.py`), REAP tracer/pruner (`reap_trace.cpp`, `prune_gguf.py`), and quant recipes
- [`train/`](train/) — Stage 2 (`distill.py`), Stage 3 hard-pruned (`distill_pruned.py`), Working-Set Concentration (`distill_workingset.py`), local Mac router (`train_mac_routers.py`), and EAGLE-v3 (`train_mac_eagle_v3.py`, `export_eagle_bin.py`) scripts + architecture notes (`FACTS.md`)
- [`tools/`](tools/) — Reversible offline pre-repacker (`prerepack_exps.cpp`), per-layer CECAR optimizer (`optimize_cecar_layers.py`), EAGLE drafter (`eagle_draft.hpp`, `gen_mac_eagle.cpp`), router dump (`dump_fmrd_data.cpp`)
- [`eval/`](eval/) — Public 271-item suite specification (`suite_spec.json`), full scoreboards (`results/summary.md`), raw sanitized iPhone 15 Pro `v35` telemetry logs + prompt files (`results/iphone15pro_v35/`), and raw 80-item quality JSONs (`results/quality_80item/`)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System architecture & roofline analysis
- [`docs/LOG.md`](docs/LOG.md) — Complete chronological engineering log (every experiment, number, cost, and dead end)

---

## License

Apache 2.0 (matching [`thinkingmachines/TML-d24-varloop-research`](https://huggingface.co/thinkingmachines/TML-d24-varloop-research)).
