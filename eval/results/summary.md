# Benchmark Results — `inkling-ios`

All quality benchmarks use deterministic public code-scored datasets (`eval/suite_spec.json`):
- **Reasoning (`120`):** `GSM8K-Hard` (30), `MATH-500` (30), `Logic` (30), `GPQA Diamond` (30)
- **Tool Use (`76`):** `BFCL` (`simple`, `multiple`, `parallel`, `parallel_multiple`, `irrelevance` + `live_*`)
- **Writing / Instructions (`75`):** `IFEval` (40), `IFBench` (35)

Coding (`HumanEval`, `SWEBench`) and obscure trivia (`SimpleQA`, `SuperGPQA`, `HLE`) are intentionally excluded.

---

## 1. Public Quality Scoreboard & Cache-Hit Pareto Frontier

| Model / Configuration | Experts (RAM / Total) | Cache Hit Rate (`2750 MB`, `PIN=40`) | `GSM8K-Hard` | `MATH-500` | `Logic` | `GPQA` | `BFCL` | `IFEval` | Avg Think |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `claude-sonnet-5-5` *(ref)* | — | — | `100.0%` | `100.0%` | `100.0%` | `76.7%` | `89.5%` | `90.0%` | `0` |
| `gemma4-26b-q4` *(ref)* | — | — | `80.0%` | `43.3%` | `33.3%` | `56.7%` | `89.5%` | `95.0%` | `315` |
| `qwen36-35b-a3b-fp8` *(ref)* | — | — | `100.0%` | `50.0%` | `46.7%` | `43.3%` | `86.8%` | `90.0%` | `339` |
| `lfm3-r1c-q4` *(ref)* | — | — | `90.0%` | `66.7%` | `20.0%` | `40.0%` | `77.6%` | `72.5%` | `275` |
| `varloop-stage2-v1` + `run3` LoRA *(128e baseline, 236 items)* | `128 / 128` | `~85–90%` | **`83.3%`** *(30)* | **`60.0%`** *(30)* | `20.0%` *(30)* | `16.7%` *(30)* | **`85.5%`** *(76)* | `52.5%` *(40)* | `~600–700` |
| `varloop-stage2-v1` + `run3` + `CECAR=0.20` | `128 / 128` | `95.50%` | **`100.0%`** *(10)* | — | — | — | — | — | `521` |
| `varloop-stage2-v1` + `run3` + `CECAR=0.35` *(uncapped)* | `128 / 128` | `95.24%` | **`80.0%`** *(10)* | **`40.0%`** *(10)* | — | — | — | — | `480–798` |
| **`S2V1-Run3-Merged` (`PIN=40`, `CECAR=0.35`, `CECAR_PREFILL=1`, `cap=384`)** | **`40` pinned / `128`** | **`95.3–96.5%`** (`7.1–9.7` slice miss/t) | **`73.3%`** *(15)* | **`30.0%`** *(10)* | `6.7%` *(15)* | **`26.7%`** *(15)* | **`70.0%`** *(30)* | **`55.0%`** *(20)* | **`351–381`** |
| `S2V1-Run3-Merged` (`PIN=40`, `CECAR=0.42`, `CECAR_PREFILL=1`, `cap=384`) | `40` pinned / `128` | **`97.36%`** (`5.30` slice miss/t) | `53.3%` *(15)* | `20.0%` *(10)* | — | — | `70.0%` *(20)* | `40.0%` *(20)* | `366–370` |
| **`Stage 3 WSD-40` (550 steps, pure trained router, `PIN=40`, NO `CECAR`)** | **`40` pinned / `128`** | **`99.57–99.64%`** (**`0.71–0.85` slice miss/t!**) | `36.7%` *(30)* | `0.0%` *(11)* | `6.7%` *(15)* | **`26.7%`** *(15)* | `30.0%` *(30)* | `20.0%` *(20)* | `209–525` |
| `Stage 3 e56-smoke2` (103 steps, hard pruned) | `56 / 56` | `~99.5%` | `13.3%` *(15)* | `0.0%` *(10)* | — | — | `25.0%` *(20)* | — | `305–440` |

### 1b. Zero-Shot Pruning & 1-Bit Compression Cliff (`GSM8K-Hard`)

| Candidate (Untrained / Zero-Shot) | Experts / Layer | Expert RAM (`REPACK=1`) | `GSM8K-Hard` | Avg Think |
|---|---:|---:|---:|---:|
| `stage2-v1` (baseline) | `128` | `8.31 GiB` | **`9/10` (`90.0%`)** *(full 30: `83.3%`)* | `474` |
| `e96` (REAP uniform) | `96` (`75%`) | `6.14 GiB` | **`8/10` (`80.0%`)** | `476` |
| `e88` (REAP uniform, `Q2_K`) | `88` (`69%`) | `5.63 GiB` | `5/10` (`50.0%`) | `490` |
| `e88-iq1s` (sliced imatrix `IQ1_S` `L2–23` + `Q2_0`) | `88` (`69%`) | `3.62 GiB` | `0/4` (`0.0%`) | `1024` |
| `e80` (REAP uniform) | `80` (`62.5%`) | `5.12 GiB` | `2/10` (`20.0%`) | `682` |
| `e64` (REAP uniform) | `64` (`50%`) | `4.09 GiB` | `0/14` (`0.0%`) | `848` |
| `e48` (REAP uniform) | `48` (`37.5%`) | **`3.07 GiB`** | `0/30` (`0.0%`) | `944` |
| Layer-adaptive (`112/64/24`, `128/80/32`, `96/32`) | `24..128` | `4.0–4.6 GiB` | `0/4` each (`0.0%`) | `793–938` |
| `128e` `Q1_0` (`L16–23`, `L12–23`, `all 24L`, untrained) | `128` (`1.125 bpw`) | `3.2–5.9 GiB` | `0/4` each (`0.0%`) | `1024` |

---

## 2. iPhone 15 Pro Speed, Context & Memory Telemetry (`-t 2`, `1p K4`, `IOS_I8MM=1`, `OVERLAP=1`, `REPACK=1`)

| Run / Build | Context (`KV`) | Zero-Miss (`miss=0`) | Sustained Decode | Warm Turn-2 TTFT | RAM Footprint | Thermal |
|---|---|---:|---|---:|---:|---|
| `v21` `varloop-Q2_K_XL` (`3000 MB`) | `2K` (`F16`) | `69.0 ms` (`14.5 t/s`) | `13.3 t/s` (450 tok) | — | `4,663 MB` | `nominal → fair` |
| `v26` `stage2-v1` + NEON `Q2_K` repack (`3250 MB`) | `2K` (`F16`) | **`43–44 ms` (`22.7–23.3 t/s`)** | `10–14 t/s` (`11–19` miss/t) | **`2.13 s`** (`v27`, `-state-load`) | `4,694 MB` | `fair → nominal` |
| `v28` `e48` (`3300 MB`, 450 tok) | `2K` (`F16`) | `48.0 ms` (`20.8 t/s`) | **`21.06 t/s`** (`47.5 ms` tok `20–449`; **`20.23 t/s`** last 250) | — | **`4,365 MB`** | `fair → fair` |
| `v28` `e64` (`3500 MB`, 450 tok) | `2K` (`F16`) | `~55 ms` | throttled (`417 ms`) | — | `4,892 MB` | **`serious` (too tight)** |
| `v29` `e48` batch-verify (`3200 MB`) | **`16K` (`Q8_0`)** | **`47.8 ms` med** | **`20.45 t/s`** (`48.9 ms` mean, `B=1`); `B=4` `105.2 ms` (`~27–31 t/s` w/ drafter) | — | `~4,460 MB` | `fair → fair` |
| **`v30` `e48` (`3200 MB`, `-state-load`, 250 tok)** | **`8K` (`Q8_0`)** | **`41–42 ms`** (`cpu 83 ms`) | **`23.25 t/s`** (`41 ms` med / `43 ms` mean) | **`0.670 s`** (`41.8 t/s`, 28 new tok) | **`4,419 MB`** | **`nominal → nominal`** |
| `v31` `e56-smoke2` (`3200 MB` vs `3350 MB`) | `8K` (`Q8_0`) | **`42–43 ms`** at `<= 4,341 MB` | Drops to `60–68 ms` once footprint hits `4,505 MB` (iOS compressor cliff!) | — | `4,341` / `4,583 MB` | `nominal → nominal` |
| **`v32c` `128e` `stage2-v1` standalone + `CECAR=0.50` (`2750 MB`, `-state-load`, 250 tok)** | **`8K` (`Q8_0`)** | **`47 ms`** (`cpu 93 ms`) | **`18.07 t/s`** (`48 ms` med / `55 ms` mean, *zero training, pre-`PIN_N`*) | **`1.975 s`** (*pre-`CECAR_PREFILL`*) | **`4,408–4,482 MB`** | **`nominal → nominal`** |

---

## 3. Speculative Decoding (`oMLX 0.8.4` / `DSpark` / `MTP`) Batch-Verify Audit (`v29`, iPhone 15 Pro)

Downloaded `thinkingmachines/TML-d24-varloop-research/mtp.safetensors` (`606 MB` BF16, `317.9M` params → `~179 MB` `Q4_K`, 3-layer DSpark drafter) and measured real `B = 1, 2, 4, 6, 8` verify steps via `-vbench`:

| Model | `B=1` | `B=2` | `B=4` | `B=6` | `B=8` | Finding |
|---|---:|---:|---:|---:|---:|---|
| **`128e` `stage2-v1` (`3100 MB` cache)** | `19.7` miss (`27 MB`) | — | **`84.6` miss (`118 MB`)** | **`114.8` miss (`160 MB`)** | **`135.9` miss (`188 MB`)** | Flash misses scale `4.3×–6.9×` linearly with batch size — **speculative decoding cannot beat the flash wall on unpruned `128e`** |
| **`e48` (`16K Q8_0 KV`, `3200 MB` cache)** | `47.8 ms` med (`0.5` miss) | — | **`105.2 ms` med (`3.0` miss)** | — | — | Once RAM-resident (or `99.6%` hot-40), `B=4` verify is only `2.2×` slower → `~27–31 tok/s` effective bonus if stacked with DSpark later |
