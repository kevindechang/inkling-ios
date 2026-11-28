# Mac v1 vs v2 runtime cache logs

Raw `tools/gen_mac` logs (Apple M4 Pro, CPU-only, patched `llama.cpp`), summarized in [`docs/V2.md` §4](../../../docs/V2.md#4-runtime-cache-behaviour-mac-same-settings-for-v1-and-v2).

These are **Mac** measurements, not iPhone measurements. They compare v1 and v2 under identical runtime settings.

File names in the logs:
- `models/varloop-stage2-v1-run3-reap40-i8mm.gguf` is the local name of the published v1 file `inkling2-26b-flashpin-1p-hot40-q2k-i8mm.gguf`. It was re-verified as byte-identical, SHA-256 `6d856056c0ff323711a487233dc5c63425f1896547803c38aac6c5f39d2acf09`.
- `models/inkling2-26b-flashpin-v2-1p-hot40-q2k-i8mm.gguf` is the published v2 file, SHA-256 `9b6b87fdb8fa01deb3b69ffbee6e59b98ecea65b1f3305d5fbb9db3d3bebc351`.

Common flags: `-ngl 0 -t 4 -tb 6 -ub 64 -passes 1 -k 4 -lm none -lzm on -c 8192 -ctk q8_0 -ctv q8_0 -n 200`, plus these environment variables:
- `INKLING_ALLOW_ONE_PASS=1`
- `INKLING_GREEDY_SKIP_SOFTCAP=1`
- `GGML_SITU_FAST=accel`
- `GGML_MMID_OVERLAP=1`
- `GGML_MMID_REPACK=1`
- `GGML_MMID_LRU=1`
- `GGML_MMID_IO_THREADS=4`
- `GGML_MMID_CACHE_MB=2700`
- `GGML_MMID_PIN_N=36`
- `GGML_MMID_CECAR_PREFILL=1`

Routing settings by file suffix:

| Suffix | Settings |
|---|---|
| `c000` | `GGML_MMID_CECAR_BONUS=0` |
| `c010` | `GGML_MMID_CECAR_BONUS=0.10` |
| `c015` | `GGML_MMID_CECAR_BONUS=0.15` |
| `opt95` | `GGML_MMID_CECAR_BONUS=1.0` and `GGML_MMID_CECAR_LAYERS=0.18,0.29,0.29,0.33,0.33,0.44,0.12,0.21,0.21,0.21,0.25,0.15,0.18,0.12,0.15,0.21,0.44,0.00,0.38,0.29,0.21,0.25,0.50,0.18` |
| `opt50` | the `opt95` layer vector with every value halved |

Prompts: [`../iphone15pro_v35/prompts/`](../iphone15pro_v35/prompts/) (`p_math.txt`, `v34_mix.txt`).
