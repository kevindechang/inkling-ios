# Benchmark Results — `inkling-ios` (`Inkling-2-FlashPin-26B`)

All quality benchmarks use deterministic public code-scored datasets ([`eval/suite_spec.json`](../suite_spec.json)):
- **Reasoning (`120` full / `45` fast):** `GSM8K-Hard` (30 / 15), `MATH-500` (30 / 10), `Logic` (30 / 10), `GPQA Diamond` (30 / 10)
- **Tool Use (`76` full / `20` fast):** `BFCL` (`simple`, `multiple`, `parallel`, `parallel_multiple`, `irrelevance` + `live_*`)
- **Writing / Instructions (`75` full / `15` fast):** `IFEval` (40 / 15), `IFBench` (35)

Coding (`HumanEval`, `SWEBench`) and obscure trivia (`SimpleQA`, `SuperGPQA`, `HLE`) are intentionally excluded.

**Published Same-Artifact GGUF:** [`kevinexus/Inkling-2-FlashPin-26B-GGUF`](https://huggingface.co/kevinexus/Inkling-2-FlashPin-26B-GGUF) — `inkling2-26b-flashpin-1p-hot40-q2k-i8mm.gguf` (`10,080,548,288` bytes, SHA-256 `6d856056c0ff323711a487233dc5c63425f1896547803c38aac6c5f39d2acf09`, all `128/128` experts intact, derived from `thinkingmachines/TML-d24-varloop-research` distilled to 1-pass `K=4`).

---

## 1. Public Quality Scoreboard & Cache-Hit Pareto Frontier

### 1a. Same-Artifact Deterministic 80-Item Suite (`Inkling-2-FlashPin-26B`, `128/128` Experts)

Subset: 15 `GSM8K-Hard`, 10 `MATH-500`, 10 `Logic`, 10 `GPQA Diamond`, 20 `BFCL`, 15 `IFEval` (`N=80`). Raw JSON outputs are committed in [`quality_80item/`](quality_80item/). Cache hit is each run's `[mmid]` tail at `3000 MB` budget / `PIN=32`. `OPT` arms use `cecar_bonus=1.0` with the 24-layer sensitivity schedule; `c000` and `fmrd_v2` use `cecar_bonus=0`.

| Arm (Raw JSON) | Think Cap / Effort | `GSM8K-Hard` | `MATH-500` | `Logic` | `GPQA` | `BFCL` | `IFEval` | Reas | Mean | Avg Think (`all / reas`) | Cache Hit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`c000_sc320`](quality_80item/c000_sc320.json) (`CECAR=0`) | `cap=320`, `e=0.20` | `60.0%` (9/15) | `30.0%` (3/10) | `10.0%` (1/10) | `10.0%` (1/10) | **`80.0%`** (16/20) | **`66.7%`** (10/15) | `31.1%` | **`59.3%`** | `245 / 293` | `75.3%` |
| [`opt95_e045`](quality_80item/opt95_e045.json) (`OPT λ=0.95`) | uncapped, `e=0.45` | **`73.3%`** (11/15) | `30.0%` (3/10) | `20.0%` (2/10) | `20.0%` (2/10) | **`75.0%`** (15/20) | `46.7%` (7/15) | **`40.0%`** | **`53.9%`** | `761 / 857` | `90.6%` |
| [`opt95_e020`](quality_80item/opt95_e020.json) (`OPT λ=0.95`) | uncapped, `e=0.20` | **`80.0%`** (12/15) | **`40.0%`** (4/10) | `20.0%` (2/10) | `0.0%` (0/10) | `60.0%` (12/20) | **`60.0%`** (9/15) | **`40.0%`** | **`53.3%`** | `774 / 780` | `91.8%` |
| [`opt85_sc320`](quality_80item/opt85_sc320.json) (`OPT λ=0.85`) | `cap=320`, `e=0.20` | `60.0%` (9/15) | `30.0%` (3/10) | `10.0%` (1/10) | `30.0%` (3/10) | `60.0%` (12/20) | `60.0%` (9/15) | `35.6%` | `51.9%` | `250 / 294` | `90.2%` |
| [`opt95_sc320`](quality_80item/opt95_sc320.json) (`OPT λ=0.95`) | `cap=320`, `e=0.20` | `73.3%` (11/15) | `30.0%` (3/10) | `10.0%` (1/10) | `30.0%` (3/10) | `60.0%` (12/20) | `46.7%` (7/15) | **`40.0%`** | `48.9%` | `249 / 293` | **`92.2%`** |
| [`fmrd_v2_sc320`](quality_80item/fmrd_v2_sc320.json) (`CECAR=0`) | `cap=320`, `e=0.20` | `46.7%` (7/15) | `30.0%` (3/10) | `10.0%` (1/10) | `20.0%` (2/10) | `70.0%` (14/20) | `46.7%` (7/15) | `28.9%` | `48.5%` | `262 / 296` | `84.2%` |

### 1b. Larger Baselines, Reference Models & Stage 3 WSD-40 Ablation

| Model / Configuration | Experts (RAM / Total) | Cache Hit Rate (`2750 MB`, `PIN=40`) | `GSM8K-Hard` | `MATH-500` | `Logic` | `GPQA` | `BFCL` | `IFEval` | Avg Think |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `claude-sonnet-5-5` *(ref)* | — | — | `100.0%` | `100.0%` | `100.0%` | `76.7%` | `89.5%` | `90.0%` | `0` |
| `gemma4-26b-q4` *(ref)* | — | — | `80.0%` | `43.3%` | `33.3%` | `56.7%` | `89.5%` | `95.0%` | `315` |
| `qwen36-35b-a3b-fp8` *(ref)* | — | — | `100.0%` | `50.0%` | `46.7%` | `43.3%` | `86.8%` | `90.0%` | `339` |
| `lfm3-r1c-q4` *(ref)* | — | — | `90.0%` | `66.7%` | `20.0%` | `40.0%` | `77.6%` | `72.5%` | `275` |
| Full `TML-d24-varloop-research` (`3p K8`, Mac MLX ref) | `128 / 128` | — | `95.0%` | — | — | — | `100.0%` | `70.0%` | — |
| `stage2-v1 + run3` LoRA *(1p K4 baseline, 236 items)* | `128 / 128` | `~85–90%` | **`83.3%`** *(25/30)* | **`60.0%`** *(18/30)* | `20.0%` *(6/30)* | `16.7%` *(5/30)* | **`85.5%`** *(65/76)* | `52.5%` *(21/40)* | `~600–700` |
| `stage2-v1 + run3` + `CECAR=0.20` | `128 / 128` | `95.50%` | `100.0%` *(10/10)* | — | — | — | — | — | `521` |
| `stage2-v1 + run3` + `CECAR=0.35` *(uncapped)* | `128 / 128` | `95.24%` | `80.0%` *(8/10)* | `40.0%` *(4/10)* | — | — | — | — | `480–798` |
| `S2V1-Run3-Merged` (`PIN=40`, `CECAR=0.35`, `cap=384`) | `40` pinned / `128` | `95.3–96.5%` (`7.1–9.7` miss/t) | `73.3%` *(11/15)* | `30.0%` *(3/10)* | `6.7%` *(1/15)* | `26.7%` *(4/15)* | `70.0%` *(21/30)* | `55.0%` *(11/20)* | `351–381` |
| `S2V1-Run3-Merged` (`PIN=40`, `CECAR=0.42`, `cap=384`) | `40` pinned / `128` | `97.36%` (`5.30` miss/t) | `53.3%` *(8/15)* | `20.0%` *(2/10)* | — | — | `70.0%` *(14/20)* | `40.0%` *(8/20)* | `366–370` |
| `Stage 3 WSD-40` (550 steps, pure trained router, NO `CECAR`) | `40` pinned / `128` | **`99.57–99.64%`** (`0.71–0.85` miss/t) | `36.7%` *(11/30)* | `0.0%` *(0/11)* | `6.7%` *(1/15)* | `26.7%` *(4/15)* | `30.0%` *(9/30)* | `20.0%` *(4/20)* | `209–525` |
| `Stage 3 e56-smoke2` (103 steps, hard pruned) | `56 / 56` | `~99.5%` | `13.3%` *(2/15)* | `0.0%` *(0/10)* | — | — | `25.0%` *(5/20)* | — | `305–440` |

### 1c. Zero-Shot Pruning & 1-Bit Compression Cliff (`GSM8K-Hard`)

| Candidate (Untrained / Zero-Shot) | Experts / Layer | Expert RAM (`REPACK=1`) | `GSM8K-Hard` | Avg Think |
|---|---:|---:|---:|---:|
| `stage2-v1` (baseline) | `128` | `8.31 GiB` | `9/10` (`90.0%`) *(full 30: `83.3%`)* | `474` |
| `e96` (REAP uniform) | `96` (`75%`) | `6.14 GiB` | `8/10` (`80.0%`) | `476` |
| `e88` (REAP uniform, `Q2_K`) | `88` (`69%`) | `5.63 GiB` | `5/10` (`50.0%`) | `490` |
| `e88-iq1s` (sliced imatrix `IQ1_S` `L2–23` + `Q2_0`) | `88` (`69%`) | `3.62 GiB` | `0/4` (`0.0%`) | `1024` |
| `e80` (REAP uniform) | `80` (`62.5%`) | `5.12 GiB` | `2/10` (`20.0%`) | `682` |
| `e64` (REAP uniform) | `64` (`50%`) | `4.09 GiB` | `0/14` (`0.0%`) | `848` |
| `e48` (REAP uniform) | `48` (`37.5%`) | `3.07 GiB` | `0/30` (`0.0%`) | `944` |
| Layer-adaptive (`112/64/24`, `128/80/32`, `96/32`) | `24..128` | `4.0–4.6 GiB` | `0/4` each (`0.0%`) | `793–938` |
| `128e` `Q1_0` (`L16–23`, `L12–23`, `all 24L`, untrained) | `128` (`1.125 bpw`) | `3.2–5.9 GiB` | `0/4` each (`0.0%`) | `1024` |

---

## 2. iPhone 15 Pro Speed, Context & Memory Telemetry

### 2a. `v35` Flagship Sweep — Full `128/128` Experts, Exact `200K` Head, `-ub 64`, `GGML_SITU_FAST=accel`

Every row uses the exact published artifact (`SHA-256: 6d856056...2acf09`) with `-c 8192 -ctk q8_0 -ctv q8_0 -passes 1 -k 4 -lm none -lzm on -t 2 -tb 4 -ub 64`, overlap, repack, LRU, 4 IO threads, and CECAR prefill. Raw sanitized per-token logs and prompts are committed under [`iphone15pro_v35/`](iphone15pro_v35/).

| Run (Raw Log) | Prompt / Gen | Cache / Pin / Route | Head | Prepack | Zero-Miss (n) | Med / Mean ms | `tok/s` | Warm TTFT (40t) | Hit% (`tok1+`) | Decode Wait / Repack ms | Peak / Steady MB | Thermal |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **[`v35a`](iphone15pro_v35/v35a_c2700_p36_opt95_ub64.log)** | `p_math` (66t) / `200` | `2700 / 36 / λ=0.95` | exact | 0 | `31` (14) | `39 / 43` | **`23.43`** | `0.978 s` | `89.8%` (`95.3%`) | `2187.6 / 437.8` | `4081 / 3945` | `nominal → nominal` |
| [`v35b`](iphone15pro_v35/v35b_c2950_p40_opt95_ub64.log) | `p_math` (66t) / `200` | `2950 / 40 / λ=0.95` | exact | 0 | `32` (12) | `43 / 48` | `20.78` | `0.992 s` | `90.2%` (`95.7%`) | `2140.2 / 602.9` | `4334 / 4187` | `nominal → nominal` |
| [`v35c`](iphone15pro_v35/v35c_c2980_p41_opt110_ub64.log) | `p_math` (66t) / `200` | `2980 / 41 / λ=1.10` | exact | 0 | `32` (20) | `43 / 46` | `21.58` | `0.899 s` | `90.9%` (`96.5%`) | `1838.2 / 604.4` | `4364 / 4225` | `nominal → nominal` |
| **[`v35d`](iphone15pro_v35/v35d_c2950_p40_c035_ub64.log)** | `p_math` (66t) / `200` | `2950 / 40 / bonus 0.35` | exact | 0 | `34` (31) | `42 / 46` | **`21.78`** | **`0.761 s`** | `91.6%` (`97.4%`) | `1447.9 / 561.6` | `4334 / 4187` | `nominal → nominal` |
| [`v35_mix`](iphone15pro_v35/v35_mix_c2700_p36_opt95_ub64.log) | `v34_mix` (127t) / `200` | `2700 / 36 / λ=0.95` | exact | 0 | `34` (3) | `55 / 57` | `17.46` | `1.038 s` | `85.5%` (`93.0%`) | `3273.7 / 906.9` | `3990 / 3933` | `nominal → nominal` |
| [`v35e`](iphone15pro_v35/v35e_prerepack_best.log) | `p_math` (66t) / `200` | `2700 / 36 / λ=0.95` | exact | 1 | `30` (7) | `41 / 45` | `22.05` | `1.172 s` | `89.2%` (`94.6%`) | `2119.3 / 0.0` | `4062 / 3951` | `nominal → nominal` |
| **[`v35f`](iphone15pro_v35/v35f_sustained400.log)** | `p_math` (66t) / `375` *(EOG)* | `2700 / 36 / λ=0.95` | exact | 1 | `34` (14) | `43 / 47` | **`21.20`** | `1.133 s` | `92.0%` (`95.1%`) | `3777.6 / 0.0` | `4062 / 3927` | `nominal → nominal` |
| **[`v35f_mix`](iphone15pro_v35/v35f_sustained400_mix.log)** | `v34_mix` (127t) / `400` | `2700 / 36 / λ=0.95` | exact | 1 | `36` (3) | `52 / 54` | **`18.36`** | `1.126 s` | `88.4%` (`92.4%`) | `5697.6 / 0.0` | `3975 / 3922` | `nominal → nominal` |
| [`v35g`](iphone15pro_v35/v35g_prerepack_margin26.log) *(opt)* | `p_math` (66t) / `200` | `2700 / 36 / λ=0.95` | *lossy margin (26,1)* | 1 | `26` (7) | `37 / 42` | `23.62` | `1.131 s` | `89.2%` (`94.6%`) | `2188.5 / 0.0` | `4094 / 3954` | `nominal → nominal` |

Mac verification of the same Phase 2 kernels before deploying to iPhone:
- Row-broadcast NEON unit test (`ADD`/`SUB`/`MUL`/`DIV`): `2,121,548` checks, `0` mismatches. Graph-profile `MUL`: `201.7 ms` (`79,200` calls) → `21.0 ms` (`69,600` calls), `8.4×` per call; 1-pass `SUB`: `0.8 ms` (`2,400` calls).
- Fused `SiTU` (`GGML_SITU_FAST=accel`): `GLU` `38.1 → 12.3 ms` (`3.10×`), Mac 100-tok decode `41.34 → 45.08 tok/s`.
- Offline pre-repacking (`GGML_MMID_PREREPACKED=1`): Mac prefill wait `247.7 → 93.4 ms`, decode wait `293.3 → 209.5 ms`, `repack_ms = 0.0`; pack/unpack SHA-256 roundtrip verified bit-identical on the full `10.08 GB` GGUF.

### 2b. Historical Progression (`v21–v33`, iPhone 15 Pro, `-t 2`, `1p K4`, `IOS_I8MM=1`)

| Run / Build | Context (`KV`) | Zero-Miss (`miss=0`) | Sustained Decode | Warm Turn-2 TTFT | RAM Footprint | Thermal |
|---|---|---:|---|---:|---:|---|
| `v21` `varloop-Q2_K_XL` (`3000 MB`) | `2K` (`F16`) | `69.0 ms` (`14.5 t/s`) | `13.3 t/s` (450 tok) | — | `4,663 MB` | `nominal → fair` |
| `v26` `stage2-v1` + NEON `Q2_K` repack (`3250 MB`) | `2K` (`F16`) | `43–44 ms` (`22.7–23.3 t/s`) | `10–14 t/s` (`11–19` miss/t) | `2.13 s` (`v27`, `-state-load`) | `4,694 MB` | `fair → nominal` |
| `v28` `e48` (`3300 MB`, 450 tok) | `2K` (`F16`) | `48.0 ms` (`20.8 t/s`) | `21.06 t/s` (`47.5 ms` tok `20–449`) | — | `4,365 MB` | `fair → fair` |
| `v28` `e64` (`3500 MB`, 450 tok) | `2K` (`F16`) | `~55 ms` | throttled (`417 ms`) | — | `4,892 MB` | `serious` *(too tight)* |
| `v29` `e48` batch-verify (`3200 MB`) | `16K` (`Q8_0`) | `47.8 ms` med | `20.45 t/s` (`B=1`); `B=4` `105.2 ms` | — | `~4,460 MB` | `fair → fair` |
| `v30` `e48` (`3200 MB`, `-state-load`, 250 tok) | `8K` (`Q8_0`) | `41–42 ms` (`cpu 83 ms`) | `23.25 t/s` (`41 / 43 ms`) *(pruned `0/30` GSM8K)* | `0.670 s` (`41.8 t/s`, 28t) | `4,419 MB` | `nominal → nominal` |
| `v31` `e56-smoke2` (`3200 MB` vs `3350 MB`) | `8K` (`Q8_0`) | `42–43 ms` at `<= 4,341 MB` | Drops to `60–68 ms` at `4,505 MB` *(iOS compressor cliff)* | — | `4,341` / `4,583 MB` | `nominal → nominal` |
| `v32c` `128e` `stage2-v1` + `CECAR=0.50` (`2750 MB`) | `8K` (`Q8_0`) | `47 ms` (`cpu 93 ms`) | `18.07 t/s` (`48 / 55 ms`, *pre-`PIN_N`*) | `1.975 s` | `4,408–4,482 MB` | `nominal → nominal` |
| `v33a` `128e` `reap40-i8mm` (`2700 MB`, `ub=128`) | `8K` (`Q8_0`) | `36 ms` (`cpu 72 ms`) | `18.45 t/s` (`51 / 54 ms`, 200 tok) | `1.165 s` (`26.6 t/s`, 31t) | `3,984 MB` | `nominal → nominal` |

---

## 3. Speculative Decoding Audits (`EAGLE-v3` & `DSpark` / `MTP`, iPhone 15 Pro)

### 3a. 2-Layer `EAGLE-v3` Drafter (`v33b`)
Trained 2-layer `d=1024` Q4 drafter (`train/train_mac_eagle_v3.py`, `28.3 MB` @ `K=8192`). On Mac held-out validation: `alpha1_gen = 55.1%`, `alpha2|1 = 57.8%` (`~1.55` tok/step expected). On iPhone 15 Pro (`K=1`, `B=2`, `t=2`, `tb=4`, `CACHE=2650`): **`12.84 tok/s`** (`77.9 ms/tok`, `tok/step=1.389`, `alpha1=38.9%`, draft `1.25 ms`, verify `106.60 ms`, footprint `4,031 MB`) — slower than greedy `B=1` (`23.43 tok/s`) because `B=2` doubles cold flash misses and activates E-cores.

### 3b. `MTP` / `DSpark` Batch-Verify Scaling (`v29`)
Measured real `B = 1, 2, 4, 6, 8` verify steps via `-vbench` (`thinkingmachines/TML-d24-varloop-research/mtp.safetensors`):

| Model | `B=1` | `B=2` | `B=4` | `B=6` | `B=8` | Finding |
|---|---:|---:|---:|---:|---:|---|
| `128e` `stage2-v1` (`3100 MB` cache) | `19.7` miss (`27 MB`) | — | `84.6` miss (`118 MB`) | `114.8` miss (`160 MB`) | `135.9` miss (`188 MB`) | Flash misses scale `4.3×–6.9×` linearly with batch size — **speculative decoding loses to the flash wall on unpruned `128e`** |
| `e48` (`16K Q8_0 KV`, `3200 MB` cache) | `47.8 ms` med (`0.5` miss) | — | `105.2 ms` med (`3.0` miss) | — | — | Only beats `B=1` once experts are `>= 99%` RAM-resident |
