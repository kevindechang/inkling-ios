# `inkling-ios`

**Running a 26.4B-parameter looped Mixture-of-Experts model (`TML-d24-varloop-research` / Inkling 2) at ~20 tok/s on iPhone 15 Pro CPU — under 4.7 GB RAM, keeping reasoning, writing, and tool use intact.**

> **Status:** Active engineering log & reproducible build. Speed shape (`varloop-stage2-v1.gguf`) is **locked** at **`43–44 ms` (`22.7–23.3 tok/s`)** zero-miss compute on A17 Pro (`-t 2`). Public 271-item reasoning/tool/instruction baseline is running now; Stage 3 short-thinking distillation + direct GGUF merge is next.

---

## Targets & Live Progress

| Goal | Target | Measured Today (iPhone 15 Pro / Mac) | Status |
|---|---|---|---|
| **Decode speed** | **`20 tok/s`** (`50 ms/tok` sustained) | **`43–44 ms` (`22.7–23.3 t/s`)** zero-miss compute floor (`-t 2`, `cpu 86 ms`); **`10.1–13.7 t/s`** (`73–99 ms`) on real tokens with `11–19` flash misses (`v26`) | **Compute met; flash wait (`+25–45 ms`) still blocks true sustained 20 t/s → testing REAP expert pruning (`128 → 48–64`) next** |
| **Memory** | **`<= 4.7 GB`** (under 6 GiB iOS limit) | **`4.47–4.69 GB`** (`3000–3250 MB` explicit expert cache) | **Met** |
| **Thermals** | Cool (`nominal → fair`) | **`-37%` CPU/tok** vs baseline (`86 ms` vs `138 ms` on 2 P-cores) | **Met** |
| **TTFT** | **`<= 2.0 s`** | **`2.13 s`** honest cross-prompt warm Turn 2 (`28` new tok, `-tb 4`, `IO=4`, `-state-load`); **`0.81 s`** (`34.5 t/s`) hot cache; `4.28 s` cold 58-tok (`v27`) | **Essentially met** (`-65%` from `6.03 s` baseline) |
| **Reasoning / Writing / Tool Use** | **Mostly intact** *(coding & trivia allowed to drop)* | Live Stage 2 (*pre-retrain*, old mismatched LoRA): **`GSM8K-Hard 83.3%`**, **`MATH-500 60.0%`**, `Logic 30.8%` *(13)*, **`BFCL 90%`** *(20)* | **271-item baseline running; Stage 3 heal + short thinking next** |
| **Thinking length** | **Concise** (`~150–250 tok`) | `535–840 tok` today | **Stage 3 focus** (inference caps broke reasoning; must be trained) |

---

## Key Techniques

All runtime patches live in [`patches/llama.cpp-inkling-ios.patch`](patches/llama.cpp-inkling-ios.patch) against upstream `llama.cpp @ 8a97fa98` (every switch is opt-in and **default OFF**):

1. **1-Pass, Top-4 Variable-Loop Shape (`INKLING_ALLOW_ONE_PASS=1`):** Uses Apache-2.0 [`thinkingmachines/TML-d24-varloop-research`](https://huggingface.co/thinkingmachines/TML-d24-varloop-research) (`inkling2_loop`, 24 blocks, 128 experts/layer) at `loop.count=1, expert_used_count=4` instead of 3p K8 (~`4×` less compute, coherent out of the box).
2. **CPU-Only Resident Dense + Streaming Expert LRU Cache (`GGML_MMID_CACHE_MB=3250`):** Avoids iOS Metal OOMs (`6 GiB` limit) and hybrid GPU/CPU graph splits; streams lazy-mapped expert slices via direct `F_NOCACHE` parallel `pread`.
3. **Known-Route Paired-`down` Overlap (`GGML_MMID_OVERLAP=1`):** Ported from Atomic Engine — background reader threads prefetch missing `ffn_down_exps` slices while the 2 P-cores compute `ffn_gate_up_exps` (`100%` prefetch hit rate, `-40%` phone decode latency, `-33%` CPU).
4. **ARMv8.6 `i8mm` + Zero-Bloat NEON `4×4` On-Miss Repacking (`IOS_I8MM=1`, `GGML_MMID_REPACK=1`):**
   - `armv8.6-a+dotprod+fp16+i8mm` (`-6.0 ms/tok` zero-miss win on A17 Pro).
   - Bit-exact (`0/2304` bit diffs) custom NEON `4×4` repack + GEMV kernels for `Q2_K`, `Q2_0`, `NVFP4`, and `MXFP4` with **0 bytes of cache bloat**; `ggml_arm_repack_q2_K_4x4` vectorized with `vst4q_u8`/`vst4q_u32` (`2.23×` faster).
5. **KL-Optimal, Latency-Constrained Quantization (`quant/`):**
   - [`quant/kl_sweep.py`](quant/kl_sweep.py) measures per-tensor `1p K4` logit KL divergence using real `libggml-base` quantization + official `imatrix`.
   - [`quant/dp_allocate.py`](quant/dp_allocate.py) solves a latency + size knapsack DP calibrated to A17 Pro's measured roofline → [`quant/tensor_types_stage2_v1.txt`](quant/tensor_types_stage2_v1.txt) (`23× Q2_K` `gate_up`, protected `L05`/`L17` loop-boundary `down` layers, `Q4_K` late attention, `F16` small projections).
   - Result: **`>2.3×` faster expert compute** (`~35 → 14.9 ms/tok`), **`-5.6 ms` attention**, **`-200 MB` RAM**, dropping zero-miss decode from **`69 ms → 43–44 ms`**.

---

## Repository Layout

- [`patches/llama.cpp-inkling-ios.patch`](patches/llama.cpp-inkling-ios.patch) — Clean `git diff` against `llama.cpp @ 8a97fa98ab1a454709bbf5ac979661feaeadec40`
- [`ios/`](ios/) — Minimal iOS CLI runner (`inkling_run.cpp`, `ios_main.mm`), build/sign/install (`build_ios.sh`), and `devicectl` helpers (`phone.sh`, `phone_env.sh`)
- [`quant/`](quant/) — 1p K4 KL sensitivity sweeper (`kl_sweep.py`), roofline knapsack DP compiler (`dp_allocate.py`), and locked Stage 2 recipe (`tensor_types_stage2_v1.txt`, `stage2_v1_plan.json`)
- [`train/`](train/) — Self-distillation trainer (`distill.py`, fits 26B on 1× 48 GB GPU via `--experts-int8`) and architecture notes (`FACTS.md`)
- [`eval/`](eval/) — Public 271-item reasoning / tool-use / instruction suite definition (`suite_spec.json`) and live scoreboards (`results/summary.md`)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System design & roofline analysis
- [`docs/LOG.md`](docs/LOG.md) — Complete chronological engineering log (every experiment, number, and rejected dead end)

---

## Quickstart

### 1. Patch & Build `llama.cpp`
```bash
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout 8a97fa98ab1a454709bbf5ac979661feaeadec40
git -C third_party/llama.cpp apply ../../patches/llama.cpp-inkling-ios.patch
```

### 2. Quantize `TML-d24-varloop-research` with the Stage 2 Recipe
Download `TML-d24-varloop-research-BF16.gguf` and `imatrix-TML-d24-varloop-research.gguf` from `ggml-org/TML-d24-varloop-research-GGUF`, then run:
```bash
llama-quantize \
  --imatrix imatrix-TML-d24-varloop-research.gguf \
  --tensor-type-file quant/tensor_types_stage2_v1.txt \
  TML-d24-varloop-research-BF16.gguf varloop-stage2-v1.gguf Q2_K 16
```

### 3. Build & Run on iPhone 15 Pro (requires `increased-memory-limit` entitlement)
```bash
export IOS_UDID="<your-device-udid>"
export IOS_BUNDLE_ID="<your.bundle.id>"
export IOS_TEAM_ID="<your-team-id>"
export IOS_SIGN_IDENTITY="<your-signing-identity>"
export IOS_PROVISION_PROFILE="/path/to/profile.mobileprovision"

IOS_I8MM=1 REBUILD_LIBS=1 ./ios/build_ios.sh
./ios/phone.sh push /path/to/varloop-stage2-v1.gguf

./ios/phone_env.sh run.log \
  '{"GGML_MMID_CACHE_MB":"3250","INKLING_ALLOW_ONE_PASS":"1","GGML_MMID_OVERLAP":"1","GGML_MMID_REPACK":"1","GGML_MMID_STATS":"1"}' \
  gen -m models/varloop-stage2-v1.gguf -ngl 0 -n 450 -lm none -lzm on -passes 1 -k 4 -t 2 -tb 4
```

---

## License

Apache 2.0 (matching `thinkingmachines/TML-d24-varloop-research`).
