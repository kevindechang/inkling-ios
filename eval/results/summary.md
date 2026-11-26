# Benchmark Results — `inkling-ios`

All quality benchmarks use deterministic public code-scored datasets (`eval/suite_spec.json`):
- **Reasoning (`120`):** `GSM8K-Hard` (30), `MATH-500` (30), `Logic` (30), `GPQA Diamond` (30)
- **Tool Use (`76`):** `BFCL` (`simple`, `multiple`, `parallel`, `parallel_multiple`, `irrelevance` + `live_*`)
- **Writing / Instructions (`75`):** `IFEval` (40), `IFBench` (35)

Coding (`HumanEval`, `SWEBench`) and obscure trivia (`SimpleQA`, `SuperGPQA`, `HLE`) are intentionally excluded.

---

## 1. Public 271-Item Suite (Reference Models + Live Stage 2 Baseline)

| Model | N | Reas (120) | Tool (76) | Writ/Instr (75) | Pillar Mean | `gsm8k` | `math500` | `logic` | `gpqa` | `bfcl` | `ifeval` | `ifbench` | avg think |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `claude-sonnet-5-5` | 271 | `94.2%` | `89.5%` | `62.7%` | `82.1%` | `100.0%` | `100.0%` | `100.0%` | `76.7%` | `89.5%` | `90.0%` | `31.4%` | `0` |
| `gemma4-26b-q4` | 271 | `53.3%` | `89.5%` | `73.3%` | `72.0%` | `80.0%` | `43.3%` | `33.3%` | `56.7%` | `89.5%` | `95.0%` | `48.6%` | `315` |
| `qwen36-35b-a3b-fp8` | 271 | `60.0%` | `86.8%` | `65.3%` | `70.7%` | `100.0%` | `50.0%` | `46.7%` | `43.3%` | `86.8%` | `90.0%` | `37.1%` | `339` |
| `lfm3-r1c-q4` | 271 | `54.2%` | `77.6%` | `48.0%` | `59.9%` | `90.0%` | `66.7%` | `20.0%` | `40.0%` | `77.6%` | `72.5%` | `20.0%` | `275` |
| `varloop-s-p3` (3p K8 teacher) | *running* | — | — | — | — | `100.0%` | `50.0%` *(10)* | *pending* | *pending* | `85.0%` *(40)* | `90.0%` | *pending* | `438` |
| `varloop-stage2-v1` + old `run3` LoRA *(pre-Stage 3)* | *running* | — | — | — | — | `83.3%` *(30)* | **`60.0%`** *(30)* | `30.8%` *(13)* | *pending* | `90.0%` *(20)* | `60.0%` *(20)* | *pending* | `~700+` |

*(Updated automatically as the 271-item baseline and Stage 3 merged checkpoint finish.)*

---

## 2. iPhone 15 Pro Speed & Memory Telemetry (`-t 2`, `1p K4`, `IOS_I8MM=1`, `OVERLAP=1`)

| Build | Zero-Miss (`z0`) | Zero-Miss CPU | Warm Wait (`~6 miss/t`) | RAM Footprint | Notes |
|---|---:|---:|---:|---:|---|
| `varloop-Q2_K_XL` (`3000 MB`, `v21`) | `69.0 ms` (`14.5 t/s`) | `138 ms` | `7.2–7.4 ms` | `4,663 MB` | `13.3 tok/s` sustained plugged-in 450 tok (`nominal → fair`) |
| `varloop-stage2-v1` + `REPACK=1` (`3000 MB`, `v25`) | `48–54 ms` (`18.5–20.8 t/s`) | `93–106 ms` | `~4–6 ms` | `4,466 MB` | `-200 MB` RAM; `1.55×` faster even when throttled (`77` vs `119 ms`) |
| `varloop-stage2-v1` + NEON `Q2_K` repack (`3250 MB`, `v26`) | **`43–44 ms` (`22.7–23.3 t/s`)** | **`86 ms`** | **`~3–5 ms`** | `4,694 MB` | `-37%` CPU heat; `-1.05 GB` flash reads; `nominal → nominal` |

---

## 3. iPhone 15 Pro Prefill / TTFT Telemetry (`v27`, `varloop-stage2-v1`, `REPACK=1`, `3250 MB`)

| Config (`-t 2`, `1p K4`) | Cold Turn 1 (`prompt → new tok`) | Cold Flash Wait / Reads | Warm Turn 2 (`prompt → new tok`) | Thermal |
|---|---:|---:|---:|---|
| `-tb 2`, `IO=2`, no state | `6.03 s` (`70 → 70`, `11.6 t/s`) | `4.01 s` (`3.75 GB`) | `3.08 s` (`40 → 40`, `13.0 t/s`) | `nominal → nominal` |
| **`-tb 4`, `IO=2`, no state** | `5.95 s` (`70 → 70`, `11.8 t/s`) | `4.04 s` (`3.75 GB`) | **`2.67 s`** (`40 → 40`, `15.0 t/s`) | `nominal → nominal` |
| `-tb 6`, `IO=2`, no state *(I/O starved)* | `6.56 s` (`70 → 70`, `10.7 t/s`) | `4.22 s` (`3.75 GB`) | `6.15 s` (`40 → 40`, `6.5 t/s`) | `nominal → nominal` |
| `-tb 6`, `IO=4`, no state | `6.79 s` (`70 → 70`, `10.3 t/s`) | `3.61 s` (`3.69 GB`) | `3.74 s` (`40 → 40`, `10.7 t/s`) | `nominal → nominal` |
| **`-tb 4`, `IO=4`, `-state-load` (12-tok sys, `0.01 s`)** — **cross-prompt** (`long1t → chat1t`) | **`4.28 s`** (`70 → 58`, `13.6 t/s`) | `3.01 s` (`3.07 GB`) | **`2.13 s`** (`40 → 28`, `13.2 t/s`) | `fair → nominal` |
| **`-tb 4`, `IO=4`, `-state-load` (12-tok sys, `0.01 s`)** — **hot-cache** (`chat1t → chat1t`) | `2.74 s` (`40 → 28`, `10.2 t/s`) | `2.10 s` (`2.05 GB`) | **`0.81 s`** (`40 → 28`, `34.5 t/s`) | `fair → fair` |

**Key findings:**
- System-prompt KV state reuse (`-state-load`, `3.10 MB` file) loads in **`0.01 s`** and saves **`-1.7 to -2.5 s`** on cold prefill.
- `-tb 4` is the optimal prefill thread count on A17 Pro (6 cores total: 4 compute + 2–4 I/O readers); `-tb 6` starves reader threads.
- **Honest cross-prompt Turn-2 TTFT is `2.13 s` (`0.81 s` on hot experts)** — down `-65%` from `6.03 s` baseline.

