# `v36` — v2 GGUF on iPhone 15 Pro (`prepack=0`)

File: `inkling2-26b-flashpin-v2-1p-hot40-q2k-i8mm.gguf` (`10,080,548,288` bytes). Installed `InklingBench` build, CPU-only, no rebuild. Logs were filtered to the app's `[gen]`/`[tok]`/`[mmid]`/`[warm_ttft]`/thermal lines and generated text (model-loader and graph-dump noise removed, device identifiers not included).

Common flags: `gen -m models/inkling2-26b-flashpin-v2-1p-hot40-q2k-i8mm.gguf -ngl 0 -t 2 -tb 4 -ub 64 -passes 1 -k 4 -lm none -lzm on -c 8192 -ctk q8_0 -ctv q8_0 -n 200 -f models/<prompt>.txt -wf models/chat1t.txt` (`v36L` used `-n 1000`).

- `v36a`–`v36c`: `v35a` env (README line 159). `v36a` wired, thermal `fair`; `v36b`/`v36c` unplugged, `nominal`.
- `v36d1`–`v36d3`, `v36m1`–`v36m3`, `v36L`: `v35d` env: `{"INKLING_ALLOW_ONE_PASS":"1","INKLING_GREEDY_SKIP_SOFTCAP":"1","GGML_SITU_FAST":"accel","GGML_MMID_CACHE_MB":"2950","GGML_MMID_PIN_N":"40","GGML_MMID_OVERLAP":"1","GGML_MMID_REPACK":"1","GGML_MMID_LRU":"1","GGML_MMID_IO_THREADS":"4","GGML_MMID_STATS":"1","GGML_MMID_CECAR_PREFILL":"1","GGML_MMID_CECAR_BONUS":"0.35","LLAMA_LM_HEAD_SHORTLIST":"off"}`.
- Prompts: `p_math.txt` / `v34_mix.txt` in [`../iphone15pro_v35/prompts/`](../iphone15pro_v35/prompts/).

Per-run numbers are in the root README, section "v2 on iPhone 15 Pro (`v36`)".
