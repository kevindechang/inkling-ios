# Inkling-2-Mini phone distillation: verified facts

Checked 2026-11-23. VERIFIED means I read the code (file:line given). ASSUMED means not checked in code or not
testable here, with the reason.

Paths:
- `TF` = `train/.pydeps/transformers/` (transformers 5.24.0, the first PyPI release that has `inkling2_loop`;
  released 2026-11-21). Its `models/inkling2_loop/modeling_inkling2_loop.py` is byte-identical to the copy in the
  scratchpad (`diff` returned no difference).
- `LC` = `inkling-ios/lab/inkling-phone/llama.cpp/` (HEAD 8a97fa9 plus local uncommitted patches).
- `M` = `TF/models/inkling2_loop/modeling_inkling2_loop.py`.

Model pin: the requested revision `ed40e7da625f594fc9c4990f5f66be24873c22d7` exists on the Hub (lastModified
2026-11-20). The Hub's `main` has since moved to `b85c3cfa…`. `distill.py` pins `ed40e7da…`.

## 1. Loading the checkpoint in transformers

| Fact | Status | Evidence |
|---|---|---|
| `config.json` says `quantization_config: {quant_method: "modelopt", quant_algo: "NVFP4"}`. transformers 5.24 has no `modelopt` quantizer, so it logs "Unknown quantization type … we will skip the quantization" and loads without a quantizer. | VERIFIED | `TF/quantizers/auto.py:296-302`; no match for `modelopt` in `TF/utils/quantization_config.py` or `TF/quantizers/` |
| Experts are dequantized to the parameter dtype (bf16) **at load time** by a weight-conversion op: NVFP4 = E2M1 codes × FP8 block scale (per 16) × per-expert `amax/outer_scale_denominator`; MXFP8 = FP8 × 2^(scale−127) per 32. | VERIFIED | `TF/core_model_loading.py:772-816` (`Inkling2LoopDequantizeExperts`); wired up in `TF/conversion_mapping.py:2654-2708` |
| Which experts use which format: layers 0-19 NVFP4, layers 20-23 MXFP8. Everything else is bf16 (or fp32 for the residual queries, message queries, and `unembed`/lm_head bf16). | VERIFIED | `hf_quant_config.json` parameter groups (40 NVFP4 tensors = 20 layers × {w13, w2}; 8 MXFP8 tensors = layers 20-23) |
| No extra packages are needed to load (no modelopt, no compressed-tensors). Optional speed-ups: `causal-conv1d` and Hub kernels. | VERIFIED (code path), ASSUMED (an end-to-end load; the 19.5 GB checkpoint was not downloaded) | `TF/core_model_loading.py:772-816` is plain torch; docs page `docs/source/en/model_doc/inkling2_loop.md` (GitHub main), "Notes" |
| Backprop through frozen experts works: the expert forward is plain `F.linear` on parameters; `torch.no_grad()` covers only the routing-mask bookkeeping. Gradients reach the router through `top_k_weights` (gathered sigmoid scores). | VERIFIED | `M:787-807` (no_grad at `M:792-794`); `M:1163-1172` |
| Attention runs eager only (sdpa/flash/flex disabled because of per-token sinks and the relative position bias). This costs memory and speed: full `[B, 24, T, T]` fp32 score tensors. | VERIFIED | `M:1186-1191` |
| A faster expert path exists: `from_pretrained(..., experts_implementation="grouped_mm")` (or `batched_mm`). | VERIFIED (option exists), ASSUMED (its backward and its numbers match eager; check on the GPU before training) | `TF/integrations/moe.py:424, 545-546`; `TF/modeling_utils.py:1422-1424` |
| Memory, text model in bf16: experts 24 × 128 × (2304·2304 + 2304·1152) = 24.46 B params = 48.9 GB; embeddings 0.46 B, lm_head 0.46 B, attention + routers ≈ 0.65 B, loop gates ≈ 0.03 B → **≈ 52 GB** for one copy. Vision tower ≈ 0.4 B (dropped by `distill.py`). Teacher and student share this one copy, so the total is ≈ 52 GB of weights plus activations. | VERIFIED (arithmetic from config shapes), ASSUMED (allocator overhead) | `config.json`; `M:775-785` |

## 2. Top-k routing: transformers vs llama.cpp

transformers (`M:1163-1172`, VERIFIED):
1. `logits = x_fp32 @ W_router^T` (router kept in fp32, `M:1158`, `M:1192-1204`)
2. `scores = sigmoid(logits)` (`scoring_func="sigmoid"`, configuration `:136`)
3. select `topk(scores + e_score_correction_bias, k)`: the bias only affects **which** experts are selected
4. `weights = scores[selected] / (sum(scores[selected]) + 1e-20)`: renormalized over the **selected k only**
5. `weights *= route_scale` (8.0; `routed_scaling_factor` maps to `route_scale`, configuration `:96`)

llama.cpp (`LC/src/llama-graph.cpp`, VERIFIED): `logits = build_lora_mm(gate_inp, cur)` `:2361` → `probs =
sigmoid(logits)` `:2380` (gating SIGMOID passed at `LC/src/models/inkling2.cpp:407`) → selection probs `+
exp_probs_b` `:2399` → `argsort_top_k(selection, n_expert_used)` `:2442` → `weights = probs[selected]` (unbiased)
`:2456` → `weights / (sum + 1e-20)` `:2477,2484` (eps from `inkling2.cpp:57`, norm_w=true at `inkling2.cpp:405`) →
`× expert_weights_scale` `:2490` (8.0 in the GGUF header). Router in fp32 (`inkling2.cpp:56`; GGUF
`ffn_gate_inp` is F32). The expert activation is the same softcapped SwiGLU on both sides (`M:760-771` vs
`llama-graph.cpp:2602-2615`, beta 4.0 / linear beta 25.0).

**They agree** (VERIFIED, read side by side). The only differences are floating-point order and the experts'
quantization.

**What changes when k goes from 8 to 4** (VERIFIED from the formula): the same 4 highest-scoring experts are kept,
and their weights are renormalized among themselves, so they still sum to 8.0. Each kept expert's weight grows
by `S8/S4 = Σ_top8 s / Σ_top4 s` (between 1× and 2×). The FFN output drops the 5th-8th experts' contribution and
rescales the rest. The MoE output norm stays about the same, but its direction changes. `e_score_correction_bias`
still affects the selection, so the top-4 by `score+bias` is not always the top-4 by score. The router weights
themselves do not change.

## 3. One pass: transformers `loop_num_passes=1` vs llama.cpp `loop.count=1`

- transformers: pre-loop layers 0-5, then the first pass over layers 6-17 with `messages=None`, which is a plain
  forward through the same layer classes (`M:1381-1395`); the loop for later passes runs `range(1, 1)`, so it
  does nothing (`M:1398`); then layers 18-23 and the final mixing (`M:1427-1443`). With `messages=None` the SWA loop
  layer feeds one normed input to Q, K and V (`M:961-962`) and the full-attention loop layer skips the message
  gates (`M:1115`). That is the same computation as a non-loop layer (`M:887-910`, `M:1032-1062`). With 1 pass
  the cache gets no extra slots (`M:1354-1357`: it multiplies by `passes-1 = 0`). `loop_num_passes` is read from
  `self.config` on every forward, so it can be switched per call (`M:1398`). The config only checks `>= 1`
  (configuration `:156-158`). VERIFIED.
- llama.cpp: with `n_passes=1` the graph builds the prelude, then `pass==0` only, which calls
  `build_layer(il, ip, cur, blocks)` without messages (`LC/src/models/inkling2-loop.cpp:90-104`), then the post
  layers and the output (`:111-114`). In `build_layer`, no messages means the SWA layers norm one input for Q, K and V
  (`inkling2.cpp:317-319`) and there are no `control()` calls (`:306-311`, `:392-394`). Cache slots = `n_phys`
  (`inkling2-loop.cpp:24`). VERIFIED.
- **Same computation**: VERIFIED structurally (same layer order, same residual-block bookkeeping, no messages).
  ASSUMED numerically (no 26 B forward was run to compare logits).
- **Bug in the local patch** (VERIFIED, `LC/src/models/inkling2-loop.cpp:19`): the inserted `// local: …` comment
  swallows the rest of that source line, so the checks `!gate_rank || !core_start || core_start >= core_end ||
  core_end > n_phys` are commented out. It compiles and is harmless for this GGUF, but those validations are gone.
  The fix is to move the comment to its own line.
- Untrained, the 1-pass student keeps using the loop's **pass-1 layer outputs**, which were trained to be refined by
  passes 2-3. The post-loop layers 18-23 and the final mixing see a pass-1 state they never saw in training. This
  is the gap the LoRA has to close.

## 4. Which llama.cpp tensors a GGUF LoRA adapter can change

Rule: an adapter applies only where the graph calls `build_lora_mm` (`LC/src/llama-graph.cpp:1809-1851`) or
`build_lora_mm_id` (`:1890-1940`). VERIFIED in `LC/src/models/inkling2.cpp`:

| Through `build_lora_mm` (adapter OK) | Line | GGUF type on the phone |
|---|---|---|
| `attn_q` | 328 | Q6_K/Q8_0/… |
| `attn_k`, `attn_v` | 329, 331 | Q8_0 |
| `attn_k_conv_proj`, `attn_v_conv_proj` (full-attention layers 5, 11, 17, 23 only) | 330, 332 | BF16 |
| `attn_rel` | 333 | Q6_K/Q8_0 |
| `attn_sink_proj` | 355 | BF16 |
| `attn_gate` (output gate) | 375 | Q8_0/… |
| `attn_output` | 376 | Q8_0 |
| `ffn_gate_inp` (router) | `llama-graph.cpp:2361` | F32 |
| `output` (lm_head, padded to 201024 rows) | 433 | Q4_K |
| `loop_*_ctrl_down/gain/forget` (message gates) | 243-246 | F32 (**unused with 1 pass**) |
| experts `ffn_gate_up_exps`, `ffn_down_exps` | via `build_lora_mm_id` | would change expert outputs; excluded on purpose |

**Not adaptable** (they use `build_norm`, `ggml_mul_mat`, `ggml_ssm_conv`, `ggml_mul` directly): all RMSNorm weights
(`attn_norm`, `attn_{q,k,v}_inp_norm`, `ffn_norm`, `output_norm`, `token_embd_norm`; `:315-323, 395, 423, 268`),
the attention-residual queries `attn_res_query` / `ffn_res_query` / `output_res_query` (`mix()`, `:179-192`), the
short-conv taps `attn_k_conv`, `attn_v_conv`, `attn_post_conv`, `ffn_post_conv` (`convolve()`, `:216`), the
relative-bias table `attn_rel_bias` (`:358`), the clip scales `attn_q_clip` / `attn_rel_clip` (`clip()`, `:167`),
`exp_probs_b`, and `loop_msg_query`. Changing these means rewriting their F32 values in the GGUF. Expert bytes can
still be copied unchanged.

All 10 adapter targets exist in the phone GGUF with the expected shapes (VERIFIED by reading the
`Q2_K_XL.mainline.gguf` header only: `attn_*` in all 24 layers, conv_proj in 4 layers, `ffn_gate_inp` [2304,128]
F32 in 24 layers, `output` [2304, 201024] Q4_K).

Adapter file rules in llama.cpp (VERIFIED): `general.type=adapter`, `general.architecture` must equal the model's,
`adapter.type=lora` (`LC/src/llama-adapter.cpp:205-220`); tensors are `<base>.lora_a` / `.lora_b`
(`:276-284`); shapes `ne0(base)==ne0(a)`, `ne1(base)==ne1(b)`, `ne1(a)==ne0(b)` (`:366-375`); scale is
`user_scale × alpha/rank`, or just `user_scale` if alpha is 0 (`LC/src/llama-adapter.h:53-56`). Layers that the loop
reuses share the physical tensors (`inkling2-loop.cpp:65-69`), and adapters are looked up by tensor pointer
(`llama-graph.cpp:1833`), so one adapter covers every pass.

## 5. Does `convert_lora_to_gguf.py` know this architecture?

Partly, and not for adapters trained on transformers module names. VERIFIED:
- The base class resolves (`Inkling2LoopForConditionalGeneration` → `conversion/inkling2.py:277`), and
  `filter_tensors` rewrites `model.language_model.`/`model.llm.` to `model.` (`conversion/inkling2.py:186-195`).
- But `gguf-py/gguf/tensor_mapping.py` maps only the **raw checkpoint names** (`attn.wq_du` `:334`, `attn.wk_dv`
  `:357`, `attn.wv_dv` `:379`, `attn.wo_ud` `:424`, `attn.gate_attn` `:470`, `attn.wr_du` `:479`,
  `self_attn.sink_proj` `:498`, `…sconv.weight_proj` `:528-534`) and a few `self_attn.*` aliases. It has **no
  entry** for the transformers 5.24 names `self_attn.q_proj.linear`, `k_proj.linear`, `v_proj.linear`,
  `o_proj.linear`, `o_proj.gate_proj`, or `relative_position_bias.linear` (grep found none). A PEFT adapter saved
  from the transformers model would fail to map.
- The converter also skips `lm_head` adapters for tied heads and does not pad the vocabulary. The GGUF `output` is
  padded to 201024 rows.
- **Decision**: `distill.py` writes the adapter GGUF itself with gguf-py (`export_gguf_lora`). It uses its own
  name map (`LORA_TARGETS`), pre-scales `lora_b`, sets `alpha=0`, and zero-pads the head. The dry run checks the
  result against a base GGUF using llama.cpp's own shape rules (see §7). ASSUMED: llama.cpp accepts it at runtime
  on the real model. Only the header rules were checked; loading the 9.1 GB model was not allowed.

## 6. Can the student use experts dequantized from the phone GGUF?

Yes. VERIFIED:
- Phone GGUF expert types: gate_up = IQ1_S (6 layers), IQ2_XXS (5), IQ2_XS (2), IQ2_S (11); down = Q2_0 (12),
  NVFP4 (11), MXFP4 (1). Each expert tensor except MXFP4 has a per-expert F32 `.scale` [128], which llama.cpp
  multiplies into the matmul output (`LC/src/llama-model.cpp:2060-2077`, `llama-graph.cpp:1828-1830, 1912-1919`).
- `gguf-py` (local tree) has `dequantize_blocks` for every one of these types (`LC/gguf-py/gguf/quants.py`: Q2_0
  `:680`, MXFP4 `:711`, NVFP4 `:762`, IQ2_XXS `:838`, IQ2_XS `:903`, IQ2_S `:973/1046`, IQ1_S `:1200/1339`).
  **Ran it**: I dequantized expert 0 of one tensor of each type from the real GGUF (memory-mapped, about 1 GB peak
  RSS). Each took 0.01-0.08 s, and after the `.scale` the weight std was 0.022-0.029 for every type, consistent
  across formats. Dequantizing all 6,144 expert matrices on CPU takes about 5 min.
- Layout: GGUF `ffn_gate_up_exps` numpy shape `(128, 2·1152, 2304)` matches HF `gate_up_proj` `(E, 2I, H)` with gate
  first, then up. That holds on both sides: HF `Interleave` (`TF/core_model_loading.py`, `Interleave`) +
  `chunk(2)` (`M:810`); GGUF converter de-interleave (`LC/conversion/inkling2.py:171-175, 255-257`); llama.cpp
  views gate first (`llama-graph.cpp:2543-2545`). `ffn_down_exps` `(128, 2304, 1152)` = HF `down_proj`. So
  `W_hf[e] = dequant(gguf[e]) × scale[e]`. That is `load_experts_from_gguf` in `distill.py`, round-trip tested
  (§7).
- **Cost**: the GPU memory is the same as the checkpoint's (bf16, about 49 GB for experts). Phone experts and NVFP4
  experts **cannot both** fit on one 80 GB GPU (about 98 GB). With `--experts-from-gguf`, the teacher therefore also
  becomes "phone experts at 3 passes / top-8". See PLAN.md for the trade-off. Host RAM: one expert at a time
  (about 21 MB fp32). The 9.1 GB GGUF has to be on the VM disk.
- The phone's **dense** matmuls are quantized too (`attn_q` Q6_K/Q8_0/IQ3_XXS, `output` Q4_K, `token_embd`
  IQ4_XS). `--dense-from-gguf` loads the dequantized LoRA-target matmuls, routers and head, so the LoRA learns
  against what the phone actually multiplies. VERIFIED: gguf-py dequantizes Q4_K/Q5_K/Q6_K/Q8_0/IQ3_XXS/IQ4_XS
  (`quants.py` classes present).

## 7. Dry run (`python distill.py --dry-run-tiny`, CPU, this Mac)

Environment: Python venv with PyTorch and `transformers==5.24.0` (first release with `inkling2_loop`). The tiny
model has 6 layers, hidden 64, 16 experts, core layers 2-3, 3 passes, a full-attention layer inside the core (3)
and one after it (5). I used 16 experts instead of 8 so that top-8 is not "all experts".

```
INFO LoRA on 46 linears; 106 trainable tensors, 0.02M params
INFO step 0 eval {'eval_kl': 0.0558, 'eval_kl_untrained_student': 0.0558}
INFO step 1 loss 0.0497 {'kl': 0.0496, 'exit_rel_mse': 0.0128, 'grad_norm': 4.0016} lr 1.00e-03
INFO step 20 loss 0.1143 {'kl': 0.114, 'exit_rel_mse': 0.0262, 'grad_norm': 6.4141} lr 1.00e-04
INFO eval step 20 {'eval_kl': 0.1203, 'eval_kl_untrained_student': 0.0558}
INFO expert digest unchanged: 4b59a134107db79a
INFO CHECK teacher independent of LoRA: max|diff| = 0
INFO CHECK experts from GGUF (Q8_0 round trip): max relative error 0.0041
INFO CHECK dense from GGUF (Q8_0 round trip, layer 3 attn_q): max relative error 0.0039
INFO wrote GGUF LoRA adapter runs/dry-run-tiny/tiny_lora.gguf (53 target tensors)
INFO CHECK adapter vs base GGUF shapes: OK
INFO CHECK recover a known low-rank perturbation: held-out KL 3.7910 -> 1.8655
INFO DRY RUN PASSED
```

`--dry-run-tiny --resume` reloads the checkpoint (`resumed at step 20 (40 sequences)`) and passes again. Greedy
teacher and student samples go to `runs/dry-run-tiny/eval.jsonl`.

How to read this honestly:
- On the **random** tiny model, the 1-pass/top-4 vs 3-pass/top-8 gap behaves like noise. Twenty steps of LoRA do not
  shrink the held-out KL; it grows from 0.056 to 0.120. With the same code, **one fixed batch overfits** (KL 0.061
  → 0.012 in 40 steps), with identical numbers with and without activation checkpointing. A **learnable target**
  (student = teacher shape plus a random low-rank perturbation) halves the **held-out** KL (3.79 → 1.87 in 30
  steps). So the loss, gradients, checkpoint recompute (the student shape is the resting shape, so recompute sees
  top-4) and optimizer are correct. The dry run does **not** show that the real gap is learnable by LoRA. Only the
  GPU smoke test can show that (PLAN.md step 3).
- The dry run tests the GGUF loaders and the adapter writer against GGUFs it writes itself (Q8_0). It does not test
  them against the real phone file, which I only read headers from.

## 8. Other facts and assumptions

- Chat template (VERIFIED by rendering it with the pinned tokenizer files): every conversation gets a system
  message `Thinking effort level: <x>` first. The default effort is 0.9. Assistant turns are
  `<|message_model|><|content_text|>…<|end_message|>`, and thinking goes in `<|content_thinking|>`. The SFT
  datasets have no thinking traces, so `distill.py` renders them with `reasoning_effort="none"` by default.
- Datasets (VERIFIED license tags on the Hub, both ungated): `allenai/tulu-3-sft-mixture` (ODC-BY; its
  `*hardcoded*` identity subset is dropped via `source`), `HuggingFaceH4/ultrachat_200k` (MIT; `train_sft` for
  training, `test_sft` held out). A regex also drops any row whose assistant text claims an identity ("as an AI",
  "I am ChatGPT", "my name is", …). I avoided `HuggingFaceTB/smoltalk` because its Hub card has no license field.
- There is a research checkpoint `thinkingmachines/TML-d24-varloop-research` with the same architecture, trained
  for a variable number of passes (VERIFIED from the transformers docs page `inkling2_loop.md`, "AutoModel Variable
  Loops"). Its experts differ, so the phone GGUF would need re-quantizing. It is a strong alternative to healing;
  see PLAN.md risk 1.
- ASSUMED: the phone selects top-4 through `--override-kv inkling2_loop.expert_used_count=int:4` (the GGUF header
  says 8). The training is independent of how it is set.
