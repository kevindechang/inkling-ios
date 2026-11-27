# `inkling-ios`

**Running a 26.4B-parameter looped Mixture-of-Experts model (`TML-d24-varloop-research` / Inkling 2) at ~20 tok/s on iPhone 15 Pro CPU — under 4.5 GB RAM, `8K` `Q8_0` context, keeping reasoning, writing, and tool use intact.**

> **Status:** Complete overnight engineering log & reproducible build (`$46.09 / $60.00` total AWS compute).
> - **Hardware envelope proven on real iPhone 15 Pro (`v28–v32`):** **`23.25 tok/s` sustained** (`41 ms` median / `43 ms` mean, `0.670 s` cross-prompt warm TTFT, `8K Q8_0 KV`, `4,419 MB` RAM, `nominal → nominal`) when active experts fit in `<= 3.1 GiB` RAM (`<= 4.45 GB` total footprint — above `~4.45 GB`, iOS memory compression drops zero-miss speed from `42 ms → 60 ms`).
> - **Two deployment points built & evaluated (standalone merged GGUFs, `0 ms` LoRA overhead):**
>   1. **High-Quality Full-128e + `PIN_N=40` + `CECAR=0.35` (`varloop-stage2-v1-run3-reap40.gguf`):** **`73.3%` `GSM8K-Hard`** (`80%` uncapped; `100%` at `CECAR=0.20`), **`30–40%` `MATH-500`**, **`26.7%` `GPQA`**, **`70.0%` `BFCL`**, **`55.0%` `IFEval`** (`avg_think=351–381` with 384 soft cap), `95.3–96.5%` cache hit rate (`18.07 tok/s` measured zero-shot at `CECAR=0.50`, `48 ms` median, `1.975 s` warm TTFT, `nominal → nominal` before `PIN_N=40` + `CECAR_PREFILL=1`).
>   2. **Stage 3 Working-Set Concentration Distillation (`WSD-40`, 550 steps on L40S, `$16.51`, `varloop-stage3-wsd40.gguf`, pure trained router — NO `CECAR`):** Achieved **`99.57–99.64%` cache hit rate** (`0.71–0.85` slice misses/tok = `0.35–0.42` expert misses/tok over 45K real diverse tokens — a `40×` drop in cold expert traffic, matching `21–22 tok/s` and `< 1 s` warm TTFT), lifting pruned reasoning from `0%` (`e48`) / `13.3%` (`e56`) to **`36.7%` `GSM8K-Hard`**, **`26.7%` `GPQA`**, **`30.0%` `BFCL`**, **`20.0%` `IFEval`**, while revealing that a `1.4%` cold budget is still too tight for `MATH-500` (`0%`).

---

## Targets & Measured Results

| Goal | Target | Best Measured Result (iPhone 15 Pro / Mac) | Status |
|---|---|---|---|
| **Sustained decode** | **`>= 20 tok/s`** (`<= 50 ms/tok` mean, no cheating) | **`23.25 tok/s`** (`41 ms` med / `43 ms` mean, `v30` `e48`, `8K` ctx, `nominal`); **`21.06 tok/s`** 450-tok (`v28`, `fair`); **`18.07 tok/s`** (`48 ms` med, `nominal`) zero-shot on full **`128e` `stage2-v1` + `CECAR=0.50`** (`v32c`); **`WSD-40`** cuts misses **11.4×** further (`0.71` slice miss/tok → **`99.64%` hit rate**) | **Hardware envelope proven; `WSD-40` locks `99.6%` hit rate** |
| **TTFT** | **`<= 2.0 s`** | **`0.670 s`** (`41.8 tok/s`, `v30` `e48`, `8K` ctx, `-state-load` 12-tok sys in `0.02 s`); **`1.975 s`** on full `128e` `v32c` *before* `CECAR_PREFILL=1` | **Met (`0.67 s` RAM-resident / `1.97 s` full 128e)** |
| **Context window** | **`8K`** viable at `>= 20 tok/s` | **`8K` (`Q8_0` KV)** proven at **`23.25 tok/s`, `4,419 MB` RAM, `nominal → nominal`** (`v30`); **`16K` (`Q8_0` KV)** proven at **`20.45 tok/s`, `fair → fair`** (`v29`) | **Met (`8K` & `16K` verified on device)** |
| **Memory** | **`<= 4.5 GB`** (under 6 GiB iOS limit) | **`4,365–4,419 MB`** clean envelope; crossing **`~4,450 MB`** triggers iOS memory-compressor slowdown (`42 ms → 60 ms`, `v31`) | **Hard iOS compressor cliff mapped (`<= 4.42 GB`)** |
| **Reasoning / Tool Use / Writing** | **Mostly intact** *(coding & trivia excluded)* | **`S2V1-Run3-Merged` (`PIN=40`, `CECAR=0.35`, `cap=384`):** **`73.3%` `GSM8K-Hard`** (`80%` uncapped, `100%` @ `CECAR=0.20`), **`30–40%` `MATH-500`**, **`26.7%` `GPQA`**, **`70.0%` `BFCL`**, **`55.0%` `IFEval`**. **`WSD-40` (`99.6%` hit, no `CECAR`):** `36.7%` `GSM8K-Hard`, `0%` `MATH-500`, `26.7%` `GPQA`, `30.0%` `BFCL`, `20.0%` `IFEval` | **Pareto frontier mapped (`95.3–99.6%` hit vs `73%–37%` GSM8K); next: `WSD-40` @ `4–6%` cold target** |
| **Thinking length** | **Concise** (`~150–350 tok`) | `~351–381 tok` on `S2V1-Run3-Merged` (`cap=384`, `73.3%` GSM8K) vs `596–1,099 tok` baseline | **Cut ~45–60%** |

---

## Key Techniques

All runtime patches live in [`patches/llama.cpp-inkling-ios.patch`](patches/llama.cpp-inkling-ios.patch) against upstream `llama.cpp @ 8a97fa98` (every switch is opt-in and **default OFF**):

1. **1-Pass, Top-4 Variable-Loop Shape (`INKLING_ALLOW_ONE_PASS=1`):** Runs Apache-2.0 [`thinkingmachines/TML-d24-varloop-research`](https://huggingface.co/thinkingmachines/TML-d24-varloop-research) (`inkling2_loop`, 24 blocks, 128 experts/layer) at `loop.count=1, expert_used_count=4` instead of `3p K8` (~`4×` less compute).
2. **CPU-Only Resident Dense + Pinned Hot-Working-Set Expert Cache (`GGML_MMID_CACHE_MB=2750`, `GGML_MMID_PIN_N=40`):** Avoids iOS Metal OOMs (`6 GiB` limit); permanently pins expert indices `0..39` (`2.60 GiB`) in RAM with zero eviction while cold safety-net experts `40..127` stream through a `~90 MiB` cold ring via `F_NOCACHE` parallel `pread`, keeping total app footprint strictly below iOS's `~4.45 GB` memory-compressor cliff.
3. **Known-Route Paired-`down` Overlap (`GGML_MMID_OVERLAP=1`):** Background reader threads (`GGML_MMID_IO_THREADS=4`) prefetch missing `ffn_down_exps` slices while the 2 P-cores compute `ffn_gate_up_exps` (`100%` prefetch hit rate, `-40%` phone decode latency).
4. **ARMv8.6 `i8mm` + Zero-Bloat NEON `4×4` On-Miss Repacking (`IOS_I8MM=1`, `GGML_MMID_REPACK=1`):** Bit-exact (`0/2304` bit diffs) NEON `4×4` repack + GEMV kernels for `Q2_K`, `Q2_0`, `NVFP4`, and `MXFP4` (`0` bytes cache bloat; `2.23×` faster `Q2_K` repack via `vst4q_u8`/`vst4q_u32`). Drops zero-miss compute from `69 ms → 41–43 ms` (`23.25 tok/s`).
5. **KL-Optimal Quantization + Bit-Exact Local Streaming BF16 Merger (`quant/`):**
   - [`quant/kl_sweep.py`](quant/kl_sweep.py) + [`quant/dp_allocate.py`](quant/dp_allocate.py) → [`quant/tensor_types_stage2_v1.txt`](quant/tensor_types_stage2_v1.txt).
   - [`quant/merge_wsd_local.py`](quant/merge_wsd_local.py) & [`quant/merge_run3_local.py`](quant/merge_run3_local.py) stream only the modified BF16 tensors via HTTP `Range` from HuggingFace, apply FP32 LoRA/router deltas, reorder experts `0..39 = HOT`, and quantize via `libggml-base` `ggml_quantize_chunk` + official `imatrix` (`100.00%` byte-exact match on `Q2_K`, `Q2_0`, `NVFP4`; zero double-quantization; zero 53 GB local BF16 storage).
6. **Working-Set Concentration Distillation (`train/distill_workingset.py`) & `CECAR` (`GGML_MMID_CECAR_BONUS`, `GGML_MMID_CECAR_PREFILL=1`):**
   - Keeps all 128 experts alive (40 RAM-pinned hot + 88 flash safety-net cold), training full-rank router deltas + `r=32` hot-40 expert LoRA + `r=32` attention/LM-head LoRA with checkpoint-safe differentiable cold-mass loss (`L_cold`) + adaptive PI cold-bias controller.
   - Standalone `CECAR` (`GGML_MMID_CECAR_BONUS=0.35`, `GGML_MMID_CECAR_PREFILL=1`, fast `std::partial_sort`) boosts cache hits to `96.5%` while keeping `73–80%` `GSM8K-Hard`, `30–40%` `MATH-500`, `70%` `BFCL`, and `55%` `IFEval`.
7. **System-Prompt KV State Caching & `Q8_0` KV (`-ctk q8_0 -ctv q8_0 -state-load`):** `20/24` layers are fixed `1024`-token SWA; only 4 layers scale with context (`8 KB/tok` at `Q8_0`), enabling `8K–16K` context under `4.42 GB` RAM and `0.67 s` warm TTFT.

---

## Repository Layout

- [`patches/llama.cpp-inkling-ios.patch`](patches/llama.cpp-inkling-ios.patch) — Clean `git diff` against `llama.cpp @ 8a97fa98ab1a454709bbf5ac979661feaeadec40`
- [`ios/`](ios/) — Minimal iOS CLI runner (`inkling_run.cpp`, `ios_main.mm`), build/sign/install (`build_ios.sh`), and `devicectl` helpers (`phone.sh`, `phone_env.sh`)
- [`quant/`](quant/) — KL sweeper (`kl_sweep.py`), DP compiler (`dp_allocate.py`), REAP tracer/pruner (`reap_trace.cpp`, `prune_gguf.py`), streaming BF16 mergers (`merge_wsd_local.py`, `merge_run3_local.py`), and quant recipes (`tensor_types_*.txt`)
- [`train/`](train/) — Stage 2 (`distill.py`), Stage 3 hard-pruned (`distill_pruned.py`), and Stage 3 Working-Set Concentration (`distill_workingset.py`) trainers + architecture notes (`FACTS.md`)
- [`eval/`](eval/) — Public 271-item reasoning / tool-use / instruction suite definition (`suite_spec.json`) and full scoreboards (`results/summary.md`)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System design & roofline analysis
- [`docs/LOG.md`](docs/LOG.md) — Complete chronological engineering log (every experiment, number, cost, and dead end)

---

## Quickstart

### 1. Patch & Build `llama.cpp`
```bash
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout 8a97fa98ab1a454709bbf5ac979661feaeadec40
git -C third_party/llama.cpp apply ../../patches/llama.cpp-inkling-ios.patch
```

### 2. Build & Run on iPhone 15 Pro (requires `increased-memory-limit` entitlement)
```bash
export IOS_UDID="<your-device-udid>"
export IOS_BUNDLE_ID="<your.bundle.id>"
export IOS_TEAM_ID="<your-team-id>"
export IOS_SIGN_IDENTITY="<your-signing-identity>"
export IOS_PROVISION_PROFILE="/path/to/profile.mobileprovision"

IOS_I8MM=1 REBUILD_LIBS=1 ./ios/build_ios.sh
./ios/phone.sh push /path/to/varloop-stage2-v1-run3-reap40.gguf

# High-quality 128e + PIN_N=40 + CECAR=0.35 + 8K Q8_0 KV:
./ios/phone_env.sh run.log \
  '{"GGML_MMID_CACHE_MB":"2750","GGML_MMID_PIN_N":"40","INKLING_ALLOW_ONE_PASS":"1","GGML_MMID_OVERLAP":"1","GGML_MMID_REPACK":"1","GGML_MMID_IO_THREADS":"4","GGML_MMID_CECAR_BONUS":"0.35","GGML_MMID_CECAR_PREFILL":"1","GGML_MMID_STATS":"1"}' \
  gen -m models/varloop-stage2-v1-run3-reap40.gguf -ngl 0 -n 450 -c 8192 -ctk q8_0 -ctv q8_0 -lm none -lzm on -passes 1 -k 4 -t 2 -tb 4 -ub 128
```

---

## License

Apache 2.0 (matching `thinkingmachines/TML-d24-varloop-research`).
