// End-to-end 100.000% BIT-EXACT LOSSLESS EAGLE Speculative Decoding benchmark for Inkling 2!
// Runs:
//   1. Standard B=1 greedy baseline (reference tokens + ms/tok + tok/s)
//   2. Speculative EAGLE K=1 (B=2) and/or K=2 (B=3) using TinyEagleDrafter (eagle_draft.hpp)
//   3. Verifies 100.000% TOKEN-FOR-TOKEN IDENTITY against B=1 baseline!
//
// Usage:
//   gen_mac_eagle -m model.gguf -eagle eagle_v2.bin [-draft 2] [-n 120] [-t 2] [-c 8192] [-p prompt]
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "eagle_draft.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

extern "C" void ggml_cpu_mmid_stats_reset(void);
extern "C" void ggml_cpu_mmid_stats_dump(void);

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct CaptureH {
    int n_embd = 2304;
    std::vector<float> h_buf;
    int n_cap = 0;
};

static bool on_eval_h(ggml_tensor * t, bool ask, void * ud) {
    if (strcmp(t->name, "result_norm") != 0) return false;
    if (ask) return true;
    CaptureH * cap = (CaptureH *) ud;
    // If inp_out_ids selected a subset, t->src[0] holds ALL tokens in the ubatch!
    ggml_tensor * src = (t->op == GGML_OP_GET_ROWS && t->src[0] != nullptr) ? t->src[0] : t;
    if (src->ne[0] != cap->n_embd || src->type != GGML_TYPE_F32) return true;
    const int n_tok = (int) src->ne[1];
    cap->n_cap = n_tok;
    cap->h_buf.resize((size_t) n_tok * cap->n_embd);
    ggml_backend_tensor_get(src, cap->h_buf.data(), 0, cap->h_buf.size() * sizeof(float));
    return true;
}

static llama_token argmax_logits(const float * logits, int n_vocab) {
    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    return best;
}

static int decode_batch_all_logits(llama_context * ctx, const llama_token * toks, int n, int pos0) {
    llama_batch b = llama_batch_init(n, 0, 1);
    b.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = pos0 + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = 1;
    }
    int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

int main(int argc, char ** argv) {
    std::string model_path, eagle_path;
    std::string prompt = "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>"
                         "<|message_user|><|content_text|>Explain clearly how a bicycle stays upright when moving, how gears change pedaling effort, and how brakes work.<|end_message|>"
                         "<|message_model|>";
    int n_gen = 120, n_ctx = 8192, n_ub = 64, threads = 2, threads_batch = 0, n_draft = 2;
    int passes = 1, n_used = 4;
    bool verify_baseline = true;

    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if      (a == "-m") model_path = v;
        else if (a == "-eagle") eagle_path = v;
        else if (a == "-n") n_gen = std::stoi(v);
        else if (a == "-c") n_ctx = std::stoi(v);
        else if (a == "-ub") n_ub = std::stoi(v);
        else if (a == "-t") threads = std::stoi(v);
        else if (a == "-tb") threads_batch = std::stoi(v);
        else if (a == "-draft") n_draft = std::stoi(v);
        else if (a == "-passes") passes = std::stoi(v);
        else if (a == "-k") n_used = std::stoi(v);
        else if (a == "-p") prompt = v;
        else if (a == "-f") { std::ifstream f(v); std::stringstream ss; ss << f.rdbuf(); prompt = ss.str(); }
        else if (a == "-no-base") verify_baseline = (v == "0");
    }
    if (threads_batch <= 0) threads_batch = threads;
    if (model_path.empty() || eagle_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -eagle eagle.bin [-draft 1|2] [-n 120] [-t 2] [-c 8192]\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.lazy_mode = LLAMA_LAZY_MODE_ON;
    ggml_backend_dev_t no_dev[1] = {nullptr};
    mp.devices = no_dev;

    llama_model_kv_override kvo[3] = {};
    int nk = 0;
    auto set_kv = [&](const char * key, int val) {
        snprintf(kvo[nk].key, sizeof kvo[nk].key, "%s", key);
        kvo[nk].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        kvo[nk].val_i64 = val;
        nk++;
    };
    if (passes > 0) set_kv("inkling2_loop.loop.count", passes);
    if (n_used > 0) set_kv("inkling2_loop.expert_used_count", n_used);
    if (nk > 0) { kvo[nk].key[0] = 0; mp.kv_overrides = kvo; }

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "[eagle_run] model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd  = llama_model_n_embd(model);

    TinyEagleDrafter drafter;
    if (!drafter.load(eagle_path, model_path, n_ctx, threads)) return 1;

    CaptureH cap;
    cap.n_embd = n_embd;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = std::max(n_ub, 16); cp.n_ubatch = std::max(n_ub, 16);
    cp.op_offload = false; cp.offload_kqv = false;
    cp.n_threads = threads; cp.n_threads_batch = threads_batch;
    cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0;
    cp.n_rs_seq = (uint32_t) std::max(4, n_draft + 1); // enable built-in SConv per-token rollback snapshots!
    cp.cb_eval = on_eval_h; cp.cb_eval_user_data = &cap;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "[eagle_run] ctx failed\n"); return 1; }
    llama_set_greedy_head(ctx, true);
    fprintf(stderr, "[eagle_run] n_rs_seq active=%u (requested %u) t=%d tb=%d\n", llama_n_rs_seq(ctx), cp.n_rs_seq, threads, threads_batch);

    std::vector<llama_token> ptoks(prompt.size() + 16);
    ptoks.resize(llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), ptoks.data(), (int) ptoks.size(), true, true));
    const int P = (int) ptoks.size();

    // ------------------------------------------------------------------ 1. Baseline B=1 greedy run
    std::vector<llama_token> base_toks;
    double base_tps = 0.0;
    if (verify_baseline) {
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int i = 0; i < P; i += n_ub) {
            int n = std::min(n_ub, P - i);
            if (llama_decode(ctx, llama_batch_get_one(ptoks.data() + i, n)) != 0) return 2;
        }
        ggml_cpu_mmid_stats_reset();
        double t0 = now_s();
        llama_token cur = llama_get_greedy_token_ith(ctx, -1);
        base_toks.push_back(cur);
        for (int g = 1; g < n_gen; ++g) {
            if (llama_vocab_is_eog(vocab, cur)) break;
            if (llama_decode(ctx, llama_batch_get_one(&cur, 1)) != 0) return 3;
            cur = llama_get_greedy_token_ith(ctx, -1);
            base_toks.push_back(cur);
        }
        double dt = now_s() - t0;
        base_tps = (base_toks.size() - 1) / std::max(dt, 1e-6);
        fprintf(stderr, "[BASE B=1] gen=%zu tok in %.3fs = %.2f tok/s (%.1f ms/tok)\n",
                base_toks.size(), dt, base_tps, 1000.0 * dt / std::max<size_t>(1, base_toks.size() - 1));
        ggml_cpu_mmid_stats_dump();
    }

    // ------------------------------------------------------------------ 2. Speculative EAGLE K=n_draft run
    llama_memory_clear(llama_get_memory(ctx), true);
    drafter.reset();

    // Prefill prompt and capture ALL prompt result_norm states h_0 .. h_{P-1}
    std::vector<float> prompt_h;
    prompt_h.reserve((size_t) P * n_embd);
    for (int i = 0; i < P; i += n_ub) {
        int n = std::min(n_ub, P - i);
        if (llama_decode(ctx, llama_batch_get_one(ptoks.data() + i, n)) != 0) return 2;
        if (cap.n_cap != n) {
            fprintf(stderr, "[eagle_run] cap mismatch %d != %d\n", cap.n_cap, n);
            return 4;
        }
        prompt_h.insert(prompt_h.end(), cap.h_buf.begin(), cap.h_buf.end());
    }
    // Feed prompt context (h_0, tok_1) .. (h_{P-2}, tok_{P-1}) into drafter KV cache
    std::vector<float> tmp_h(n_embd);
    double td_pre0 = now_s();
    for (int i = 0; i < P - 1; ++i) {
        drafter.step(prompt_h.data() + (size_t) i * n_embd, ptoks[i + 1], tmp_h.data());
    }
    fprintf(stderr, "[eagle_run] drafter prompt prime (%d steps) in %.1f ms\n", P - 1, (now_s() - td_pre0) * 1000.0);

    std::vector<float> h_last(prompt_h.end() - n_embd, prompt_h.end());
    llama_token tok_cur = llama_get_greedy_token_ith(ctx, -1);

    std::vector<llama_token> spec_toks;
    spec_toks.push_back(tok_cur);
    int cur_pos = P; // next position in target KV to write tok_cur

    ggml_cpu_mmid_stats_reset();
    double t_spec0 = now_s();
    double sum_draft_ms = 0.0, sum_verify_ms = 0.0;
    int total_steps = 0, total_drafted = 0, total_accepted = 0;
    int acc1_hits = 0, acc1_tot = 0, acc2_hits = 0, acc2_tot = 0;

    std::vector<float> h_d1(n_embd), h_d2(n_embd);
    while ((int) spec_toks.size() < n_gen && !llama_vocab_is_eog(vocab, tok_cur)) {
        const int K = std::min(n_draft, n_gen - (int) spec_toks.size());
        const int saved_kv = drafter.kv_len;

        // Draft K tokens autoregressively
        double td0 = now_s();
        llama_token drafts[4];
        drafter.step(h_last.data(), tok_cur, h_d1.data());
        drafts[0] = drafter.sample_shortlist(h_d1.data());
        if (K >= 2) {
            drafter.step(h_d1.data(), drafts[0], h_d2.data());
            drafts[1] = drafter.sample_shortlist(h_d2.data());
        }
        sum_draft_ms += (now_s() - td0) * 1000.0;

        // Verify [tok_cur, drafts[0], ..., drafts[K-1]] in ONE batch of size B = K + 1!
        llama_token batch_toks[5];
        batch_toks[0] = tok_cur;
        for (int k = 0; k < K; ++k) batch_toks[k + 1] = drafts[k];
        const int B = K + 1;

        double tv0 = now_s();
        if (decode_batch_all_logits(ctx, batch_toks, B, cur_pos) != 0) {
            fprintf(stderr, "[eagle_run] verify batch failed at pos=%d B=%d\n", cur_pos, B);
            return 5;
        }
        sum_verify_ms += (now_s() - tv0) * 1000.0;

        // Exact greedy verification against full 200K target logits (or greedy shortlist head)!
        int n_acc = 0;
        llama_token next_true = -1;
        for (int k = 0; k < B; ++k) {
            llama_token g = llama_get_greedy_token_ith(ctx, k);
            if (k < K && g == drafts[k] && !llama_vocab_is_eog(vocab, g)) {
                spec_toks.push_back(g);
                n_acc++;
            } else {
                next_true = g;
                spec_toks.push_back(g);
                break;
            }
        }
        // If ALL K drafts accepted (n_acc == K), the loop already emitted bonus token g at k = K!

        total_steps++;
        total_drafted += K;
        total_accepted += n_acc;
        acc1_tot++;
        if (n_acc >= 1) acc1_hits++;
        if (K >= 2 && n_acc >= 1) {
            acc2_tot++;
            if (n_acc >= 2) acc2_hits++;
        }

        // Rollback target KV + SConv recurrent state if any draft was rejected!
        const int n_keep = n_acc + 1; // kept positions: cur_pos .. cur_pos + n_keep - 1
        if (n_keep < B) {
            bool ok_rm = llama_memory_seq_rm(llama_get_memory(ctx), 0, cur_pos + n_keep, -1);
            if (!ok_rm) {
                fprintf(stderr, "[eagle_run] ERROR: seq_rm rollback failed (cur_pos=%d keep=%d B=%d)!\n", cur_pos, n_keep, B);
                return 6;
            }
        }
        cur_pos += n_keep;

        // Sync drafter KV with EXACT TRUE target hidden states!
        // Note: saved_kv already holds (h_last, tok_cur) from Step 1!
        drafter.truncate(saved_kv + 1);
        for (int k = 0; k < n_acc; ++k) {
            drafter.step(cap.h_buf.data() + (size_t) k * n_embd, drafts[k], tmp_h.data());
        }
        // Set new h_last = cap.h_buf[n_acc] (true target hidden state after last kept token!)
        memcpy(h_last.data(), cap.h_buf.data() + (size_t) n_acc * n_embd, (size_t) n_embd * sizeof(float));
        tok_cur = next_true;
    }

    double t_spec = now_s() - t_spec0;
    const size_t n_emitted = spec_toks.size() > 1 ? spec_toks.size() - 1 : 1;
    double spec_tps = n_emitted / std::max(t_spec, 1e-6);

    // Check 100.000% token-for-token exactness!
    int mismatches = 0;
    size_t cmp_n = std::min(base_toks.size(), spec_toks.size());
    for (size_t i = 0; i < cmp_n; ++i) {
        if (base_toks[i] != spec_toks[i]) {
            if (mismatches < 5) {
                fprintf(stderr, "[MISMATCH] tok %zu: base=%d spec=%d\n", i, base_toks[i], spec_toks[i]);
            }
            mismatches++;
        }
    }

    fprintf(stderr, "\n=== EAGLE SPECULATIVE RESULT (K=%d, B=%d, t=%d) ===\n", n_draft, n_draft + 1, threads);
    fprintf(stderr, "  gen=%zu tok in %.3fs = %.2f tok/s (%.1f ms/tok) | speedup vs B=1: %.2fx\n",
            spec_toks.size(), t_spec, spec_tps, 1000.0 * t_spec / n_emitted,
            base_tps > 0 ? spec_tps / base_tps : 0.0);
    fprintf(stderr, "  steps=%d | tok/step=%.3f | alpha1=%.1f%% (%d/%d) | alpha2|1=%.1f%% (%d/%d)\n",
            total_steps, (double) n_emitted / std::max(1, total_steps),
            100.0 * acc1_hits / std::max(1, acc1_tot), acc1_hits, acc1_tot,
            100.0 * acc2_hits / std::max(1, acc2_tot), acc2_hits, acc2_tot);
    fprintf(stderr, "  avg_draft=%.2f ms/step | avg_verify=%.2f ms/step\n",
            sum_draft_ms / std::max(1, total_steps), sum_verify_ms / std::max(1, total_steps));
    if (verify_baseline) {
        fprintf(stderr, "  BIT-EXACT TOKEN MATCH vs B=1 BASELINE: %zu/%zu (%s)\n",
                cmp_n - mismatches, cmp_n, mismatches == 0 ? "100.000% EXACT LOSSLESS!" : "MISMATCH!");
    }
    ggml_cpu_mmid_stats_dump();

    llama_free(ctx);
    llama_model_free(model);
    return mismatches == 0 ? 0 : 7;
}
