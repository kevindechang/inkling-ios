#include <algorithm>
// Phone/Mac generate command with visibility: load progress, footprint, per-token times.
// Usage: gen -m model.gguf [-ngl 99] [-n 32] [-c 2048] [-ub 64] [-t 4] [-lm mmap|none] [-lzm on|auto|off] [-cpu <tensor regex>] [-f prompt.txt | -p prompt]
#include "llama.h"
#include "ggml-backend.h"
#include "../tools/eagle_draft.hpp"
#include <mach/mach.h>
#include <sys/resource.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double footprint_mb() {
    task_vm_info_data_t info; mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &count) != KERN_SUCCESS) return -1;
    return info.phys_footprint / 1048576.0;
}
// process totals: page-ins from disk (pages) and CPU seconds (user + system, all threads)
static void usage_now(double & pageins, double & cpu_s) {
    task_events_info_data_t ev; mach_msg_type_number_t n = TASK_EVENTS_INFO_COUNT;
    pageins = task_info(mach_task_self(), TASK_EVENTS_INFO, (task_info_t) &ev, &n) == KERN_SUCCESS ? ev.pageins : -1;
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    cpu_s = ru.ru_utime.tv_sec + ru.ru_stime.tv_sec + (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1e6;
}
extern "C" void ggml_cpu_mmid_cache_stats(uint64_t * hits, uint64_t * misses, uint64_t * miss_bytes, uint64_t * bytes);

// Prompts are separated by a line that is exactly "---". Each prompt is prefilled, then n_gen greedy tokens.
static int run_prompt_sweep(llama_context * ctx, const llama_vocab * vocab, const std::string & path, int n_gen, int n_ub) {
    std::ifstream in(path);
    if (!in) { fprintf(stderr, "[prompts] open failed %s\n", path.c_str()); return 2; }
    std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<std::string> prompts;
    std::string cur;
    std::istringstream ls(all);
    std::string line;
    while (std::getline(ls, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "---") {
            if (!cur.empty() && cur.back() == '\n') cur.pop_back();
            if (!cur.empty()) prompts.push_back(cur);
            cur.clear();
        } else {
            cur += line;
            cur.push_back('\n');
        }
    }
    if (!cur.empty() && cur.back() == '\n') cur.pop_back();
    if (!cur.empty()) prompts.push_back(cur);
    int total = 0;
    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        llama_memory_clear(llama_get_memory(ctx), true);
        const std::string & prompt = prompts[pi];
        std::vector<llama_token> toks(prompt.size() + 16);
        const int ntok = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), true, true);
        if (ntok < 0) { fprintf(stderr, "[prompts] tokenize failed %zu\n", pi); return 2; }
        toks.resize((size_t) ntok);
        for (size_t i = 0; i < toks.size(); i += (size_t) n_ub) {
            const int n = (int) std::min<size_t>((size_t) n_ub, toks.size() - i);
            if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, n)) != 0) {
                fprintf(stderr, "[prompts] prefill failed %zu\n", pi); return 2;
            }
        }
        int got = 0;
        for (int g = 0; g < n_gen; g++) {
            llama_token best = llama_get_greedy_token_ith(ctx, -1);
            if (llama_vocab_is_eog(vocab, best)) break;
            if (llama_decode(ctx, llama_batch_get_one(&best, 1)) != 0) {
                fprintf(stderr, "[prompts] decode failed %zu\n", pi); return 3;
            }
            got++;
        }
        total += got;
        fprintf(stderr, "[prompt] %zu prompt_tokens=%zu gen=%d\n", pi, toks.size(), got);
    }
    fprintf(stderr, "[prompts] n=%zu gen_tokens=%d\n", prompts.size(), total);
    return 0;
}
extern "C" void ggml_cpu_mmid_stats_reset(void);
extern "C" void ggml_cpu_mmid_stats_dump(void);
extern "C" void ggml_cpu_node_prof_dump(void);
extern "C" void ggml_cpu_node_prof_reset(void);
static double g_t0, g_last;
static bool on_progress(float p, void *) {
    const double t = now_s();
    if (t - g_last > 2.0 || p >= 1.0f) { g_last = t; fprintf(stderr, "[load] %.0f%% t=%.1fs footprint=%.0f MB\n", p * 100, t - g_t0, footprint_mb()); fflush(stderr); }
    return true;
}

struct InklingEagleCap {
    int n_embd = 2304;
    std::vector<float> h_buf;
    int n_cap = 0;
};

static bool inkling_eagle_on_eval_h(ggml_tensor * t, bool ask, void * ud) {
    if (strcmp(t->name, "result_norm") != 0) return false;
    if (ask) return true;
    InklingEagleCap * cap = (InklingEagleCap *) ud;
    ggml_tensor * src = (t->op == GGML_OP_GET_ROWS && t->src[0] != nullptr) ? t->src[0] : t;
    if (src->ne[0] != cap->n_embd || src->type != GGML_TYPE_F32) return true;
    const int n_tok = (int) src->ne[1];
    cap->n_cap = n_tok;
    cap->h_buf.resize((size_t) n_tok * cap->n_embd);
    ggml_backend_tensor_get(src, cap->h_buf.data(), 0, cap->h_buf.size() * sizeof(float));
    return true;
}

static int inkling_decode_batch_all_logits(llama_context * ctx, const llama_token * toks, int n, int pos0) {
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

int inkling_gen(int argc, char ** argv) {
    std::string lora, state_save, state_load, warm_prompt, vbench_file, vbench_sizes, eagle_path, prompts_file, ctk = "f16", ctv = "f16";   // -lora <file>: GGUF LoRA adapter applied at scale 1
    std::string model, prompt = "<|message_system|><|content_text|>Thinking effort level: 0.0<|end_message|><|message_user|><|content_text|>Give me three short tips for sleeping better.<|end_message|><|message_model|>";
    int ngl = 99, n_gen = 32, n_ctx = 2048, n_ub = 64, threads = 4, threads_batch = 0, n_draft = 1; bool use_mmap = true; std::string cpu_pat; int passes = 0, n_used = 0; llama_lazy_mode lazy = LLAMA_LAZY_MODE_AUTO;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "-m") model = v; else if (a == "-lora") lora = v; else if (a == "-ngl") ngl = std::stoi(v); else if (a == "-n") n_gen = std::stoi(v);
        else if (a == "-c") n_ctx = std::stoi(v); else if (a == "-ub") n_ub = std::stoi(v); else if (a == "-t") threads = std::stoi(v);
        else if (a == "-tb") threads_batch = std::stoi(v);
        else if (a == "-eagle") eagle_path = v; else if (a == "-draft") n_draft = std::stoi(v);
        else if (a == "-ctk") ctk = v; else if (a == "-ctv") ctv = v;
        else if (a == "-vf") { std::ifstream f(v); std::stringstream ss; ss << f.rdbuf(); vbench_file = ss.str(); }
        else if (a == "-vbench") vbench_sizes = v;
        else if (a == "-lm") use_mmap = v != "none"; else if (a == "-cpu") cpu_pat = v; else if (a == "-passes") passes = std::stoi(v); else if (a == "-k") n_used = std::stoi(v); else if (a == "-p") prompt = v;
        else if (a == "-lzm") lazy = v == "on" ? LLAMA_LAZY_MODE_ON : v == "off" ? LLAMA_LAZY_MODE_OFF : LLAMA_LAZY_MODE_AUTO;
        else if (a == "-f") { std::ifstream f(v); std::stringstream ss; ss << f.rdbuf(); prompt = ss.str(); }
        else if (a == "-prompts") prompts_file = v;
        else if (a == "-wf") { std::ifstream f(v); std::stringstream ss; ss << f.rdbuf(); warm_prompt = ss.str(); }
        else if (a == "-wp") warm_prompt = v;
        else if (a == "-state-save") state_save = v; else if (a == "-state-load") state_load = v;
    }
    if (threads_batch <= 0) threads_batch = threads;
    setvbuf(stdout, nullptr, _IOLBF, 0);
    g_t0 = g_last = now_s();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl; mp.load_mode = use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE; mp.lazy_mode = lazy;
    // -cpu <regex>: keep matching tensors on the CPU (mmapped, paged by the OS) while the rest is pinned on the GPU
    llama_model_tensor_buft_override ov[2] = {{cpu_pat.c_str(), ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
    if (!cpu_pat.empty()) mp.tensor_buft_overrides = ov;
    // -passes N: override the loop pass count (lossy; the model was trained with a fixed count)
    llama_model_kv_override kvo[3] = {}; int nk = 0;
    auto set_kv = [&](const char * key, int v) { snprintf(kvo[nk].key, sizeof kvo[nk].key, "%s", key); kvo[nk].tag = LLAMA_KV_OVERRIDE_TYPE_INT; kvo[nk].val_i64 = v; nk++; };
    if (passes > 0) set_kv("inkling2_loop.loop.count", passes);
    if (n_used > 0) set_kv("inkling2_loop.expert_used_count", n_used);   // -k N: experts per layer visit (lossy)
    if (nk > 0) { kvo[nk].key[0] = 0; mp.kv_overrides = kvo; }
    // ngl 0 means CPU only: with a GPU device listed, llama.cpp still offloads ops and reserves a GPU compute buffer
    ggml_backend_dev_t no_dev[1] = {nullptr};
    if (ngl == 0) mp.devices = no_dev;
    mp.progress_callback = on_progress;
    fprintf(stderr, "[gen] loading %s ngl=%d mmap=%d footprint=%.0f MB\n", model.c_str(), ngl, use_mmap, footprint_mb());
    llama_model * m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) { fprintf(stderr, "[gen] load failed\n"); return 1; }
    fprintf(stderr, "[gen] loaded t=%.1fs footprint=%.0f MB\n", now_s() - g_t0, footprint_mb());
    auto parse_kv_type = [](const std::string & s) -> ggml_type {
        if (s == "q8_0") return GGML_TYPE_Q8_0;
        if (s == "q4_0") return GGML_TYPE_Q4_0;
        return GGML_TYPE_F16;
    };
    TinyEagleDrafter eagle_drafter;
    InklingEagleCap eagle_cap;
    const bool use_eagle = !eagle_path.empty();
    if (use_eagle) {
        eagle_cap.n_embd = llama_model_n_embd(m);
        if (!eagle_drafter.load(eagle_path, model, std::min(n_ctx, 2048), threads)) {
            fprintf(stderr, "[gen] eagle load failed: %s\n", eagle_path.c_str());
            return 1;
        }
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ub; cp.n_ubatch = n_ub; if (ngl == 0) { cp.op_offload = false; cp.offload_kqv = false; } cp.n_threads = threads; cp.n_threads_batch = threads_batch;
    cp.type_k = parse_kv_type(ctk); cp.type_v = parse_kv_type(ctv);
    if (use_eagle) {
        cp.n_rs_seq = (uint32_t) std::max(4, n_draft + 1);
        cp.cb_eval = inkling_eagle_on_eval_h;
        cp.cb_eval_user_data = &eagle_cap;
    }
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) { fprintf(stderr, "[gen] context failed\n"); return 1; }
    llama_set_greedy_head(ctx, true);
    if (!lora.empty()) {
        llama_adapter_lora * adapter = llama_adapter_lora_init(m, lora.c_str());
        float scale = 1.0f;
        if (!adapter || llama_set_adapters_lora(ctx, &adapter, 1, &scale) != 0) { fprintf(stderr, "[gen] lora failed: %s\n", lora.c_str()); return 1; }
        fprintf(stderr, "[gen] lora %s applied footprint=%.0f MB\n", lora.c_str(), footprint_mb());
    }
    fprintf(stderr, "[gen] context t=%.1fs footprint=%.0f MB\n", now_s() - g_t0, footprint_mb());
    const llama_vocab * vocab = llama_model_get_vocab(m);
    if (!prompts_file.empty()) {
        const int rc = run_prompt_sweep(ctx, vocab, prompts_file, n_gen, n_ub);
        llama_free(ctx); llama_model_free(m);
        return rc;
    }
    std::vector<llama_token> toks(prompt.size() + 16);
    toks.resize(llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), true, true));
    double t = now_s();
    size_t start = 0, reused = 0;
    if (!state_load.empty()) {
        std::vector<llama_token> saved(std::max<size_t>(llama_n_ctx(ctx), toks.size()));
        size_t n_saved = 0;
        bool ok = llama_state_load_file(ctx, state_load.c_str(), saved.data(), saved.size(), &n_saved)
            && n_saved > 0 && n_saved <= toks.size() && std::equal(saved.begin(), saved.begin() + n_saved, toks.begin());
        if (ok && n_saved == toks.size()) {
            if (llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) (n_saved - 1), -1)) n_saved--;
            else ok = false;
        }
        if (ok) {
            start = reused = n_saved;
            fprintf(stderr, "[state] loaded %zu tokens t=%.2fs, prefilling %zu new tokens\n", reused, now_s() - t, toks.size() - start);
        } else {
            llama_memory_clear(llama_get_memory(ctx), true);
            fprintf(stderr, "[state] mismatch, full prefill\n");
        }
    }
    std::vector<float> eagle_prompt_h;
    if (use_eagle) eagle_prompt_h.reserve(toks.size() * (size_t) eagle_cap.n_embd);
    for (size_t i = start; i < toks.size(); i += n_ub) {
        const int n = (int) std::min<size_t>(n_ub, toks.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, n)) != 0) { fprintf(stderr, "[gen] prefill failed\n"); return 2; }
        if (use_eagle && eagle_cap.n_cap == n) {
            eagle_prompt_h.insert(eagle_prompt_h.end(), eagle_cap.h_buf.begin(), eagle_cap.h_buf.end());
        }
        fprintf(stderr, "[prefill] %zu/%zu t=%.2fs footprint=%.0f MB\n", i + n, toks.size(), now_s() - t, footprint_mb());
    }
    const double prefill_s = now_s() - t;
    fprintf(stderr, "[prefill_done] prompt_tokens=%zu new_tokens=%zu prefill_s=%.3f (%.1f tok/s) tb=%d reused=%zu\n",
            toks.size(), toks.size() - start, prefill_s, (toks.size() - start) / std::max(prefill_s, 1e-6), threads_batch, reused);
    ggml_cpu_mmid_stats_dump();
    if (!state_save.empty()) {
        const double ts = now_s();
        if (!llama_state_save_file(ctx, state_save.c_str(), toks.data(), toks.size())) { fprintf(stderr, "[state] save failed\n"); return 4; }
        const auto bytes = std::ifstream(state_save, std::ios::binary | std::ios::ate).tellg();
        fprintf(stderr, "[state] saved %zu tokens %lld bytes t=%.2fs\n", toks.size(), (long long) bytes, now_s() - ts);
    }
    ggml_cpu_node_prof_reset();   // profile decode only
    ggml_cpu_mmid_stats_reset();  // measure decode-only wait_ms / read_mb / repack_ms
    std::string text; std::vector<double> ms;
    if (use_eagle && eagle_prompt_h.size() >= (size_t) eagle_cap.n_embd) {
        const int H = eagle_cap.n_embd;
        const int n_ph = (int) (eagle_prompt_h.size() / (size_t) H);
        eagle_drafter.reset();
        std::vector<float> tmp_h(H), h_d1(H), h_d2(H);
        const size_t p_off = toks.size() - (size_t) n_ph;
        for (int i = 0; i + 1 < n_ph; ++i) {
            eagle_drafter.step(eagle_prompt_h.data() + (size_t) i * H, toks[p_off + (size_t) i + 1], tmp_h.data());
        }
        std::vector<float> h_last(eagle_prompt_h.end() - H, eagle_prompt_h.end());
        llama_token tok_cur = llama_get_greedy_token_ith(ctx, -1);
        int cur_pos = (int) toks.size();
        int n_emitted = 0, total_steps = 0, acc1_hits = 0, acc1_tot = 0, acc2_hits = 0, acc2_tot = 0;
        double sum_draft_ms = 0.0, sum_verify_ms = 0.0;
        const double te0 = now_s();
        auto append_tok = [&](llama_token tk) {
            char buf[256];
            const int k = llama_token_to_piece(vocab, tk, buf, sizeof buf, 0, true);
            if (k > 0) text.append(buf, k);
        };
        if (!llama_vocab_is_eog(vocab, tok_cur)) {
            append_tok(tok_cur);
            n_emitted = 1;
        }
        while (n_emitted < n_gen && !llama_vocab_is_eog(vocab, tok_cur)) {
            const int K = std::min(n_draft, n_gen - n_emitted);
            const int saved_kv = eagle_drafter.kv_len;
            const double td0 = now_s();
            llama_token drafts[4];
            eagle_drafter.step(h_last.data(), tok_cur, h_d1.data());
            drafts[0] = eagle_drafter.sample_shortlist(h_d1.data());
            if (K >= 2) {
                eagle_drafter.step(h_d1.data(), drafts[0], h_d2.data());
                drafts[1] = eagle_drafter.sample_shortlist(h_d2.data());
            }
            const double d_ms = (now_s() - td0) * 1000.0;
            sum_draft_ms += d_ms;

            llama_token batch_toks[5];
            batch_toks[0] = tok_cur;
            for (int k = 0; k < K; ++k) batch_toks[k + 1] = drafts[k];
            const int B = K + 1;

            const double tv0 = now_s();
            if (inkling_decode_batch_all_logits(ctx, batch_toks, B, cur_pos) != 0) {
                fprintf(stderr, "[eagle] verify failed at pos=%d B=%d\n", cur_pos, B);
                return 3;
            }
            const double v_ms = (now_s() - tv0) * 1000.0;
            sum_verify_ms += v_ms;

            int n_acc = 0;
            llama_token next_true = LLAMA_TOKEN_NULL;
            for (int k = 0; k < B; ++k) {
                llama_token g = llama_get_greedy_token_ith(ctx, k);
                if (k < K && g == drafts[k] && !llama_vocab_is_eog(vocab, g)) {
                    append_tok(g);
                    n_acc++;
                    n_emitted++;
                } else {
                    next_true = g;
                    if (!llama_vocab_is_eog(vocab, g)) {
                        append_tok(g);
                        n_emitted++;
                    }
                    break;
                }
            }
            const int step_emitted = n_acc + 1;
            const double per_tok_ms = (d_ms + v_ms) / (double) step_emitted;
            for (int k = 0; k < step_emitted; ++k) ms.push_back(per_tok_ms);

            total_steps++;
            acc1_tot++;
            if (n_acc >= 1) acc1_hits++;
            if (K >= 2 && n_acc >= 1) { acc2_tot++; if (n_acc >= 2) acc2_hits++; }

            const int n_keep = n_acc + 1;
            if (n_keep < B) {
                if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, cur_pos + n_keep, -1)) {
                    fprintf(stderr, "[eagle] seq_rm rollback failed!\n");
                    return 3;
                }
            }
            cur_pos += n_keep;
            eagle_drafter.truncate(saved_kv + 1);
            for (int k = 0; k < n_acc; ++k) {
                eagle_drafter.step(eagle_cap.h_buf.data() + (size_t) k * H, drafts[k], tmp_h.data());
            }
            memcpy(h_last.data(), eagle_cap.h_buf.data() + (size_t) n_acc * H, (size_t) H * sizeof(float));
            tok_cur = next_true;
        }
        const double te_s = now_s() - te0;
        const int gen_out = std::max(1, n_emitted - 1);
        fprintf(stderr, "[eagle] K=%d gen=%d tok in %.3fs = %.2f tok/s (%.1f ms/tok) | steps=%d tok/step=%.3f alpha1=%.1f%% alpha2|1=%.1f%% | draft=%.2fms verify=%.2fms footprint=%.0f MB\n",
                n_draft, n_emitted, te_s, gen_out / std::max(te_s, 1e-6), 1000.0 * te_s / gen_out,
                total_steps, (double) gen_out / std::max(1, total_steps),
                100.0 * acc1_hits / std::max(1, acc1_tot), 100.0 * acc2_hits / std::max(1, acc2_tot),
                sum_draft_ms / std::max(1, total_steps), sum_verify_ms / std::max(1, total_steps), footprint_mb());
    } else for (int g = 0; g < n_gen; g++) {
        llama_token best = llama_get_greedy_token_ith(ctx, -1);
        if (llama_vocab_is_eog(vocab, best)) break;
        char buf[256]; const int k = llama_token_to_piece(vocab, best, buf, sizeof buf, 0, true);
        if (k > 0) text.append(buf, k);
        double pi0, cpu0, pi1, cpu1; usage_now(pi0, cpu0);
        t = now_s();
        if (llama_decode(ctx, llama_batch_get_one(&best, 1)) != 0) { fprintf(stderr, "[gen] decode failed\n"); return 3; }
        ms.push_back((now_s() - t) * 1000);
        usage_now(pi1, cpu1);
        uint64_t ch, cm, cmb, cb; ggml_cpu_mmid_cache_stats(&ch, &cm, &cmb, &cb);
        static uint64_t ph = 0, pm = 0, pmb = 0;
        fprintf(stderr, "[tok] %d %.0f ms footprint=%.0f MB cpu_ms=%.0f hit=%llu miss=%llu miss_mb=%.0f cache_mb=%.0f\n", g, ms.back(), footprint_mb(),
                (cpu1 - cpu0) * 1000, (unsigned long long) (ch - ph), (unsigned long long) (cm - pm), (cmb - pmb) / 1048576.0, cb / 1048576.0);
        ph = ch; pm = cm; pmb = cmb;
    }
    // Decode window, before -wf clears the counters. Prefill was dumped above; warm_ttft dumps again.
    ggml_cpu_mmid_stats_dump();
    if (!warm_prompt.empty()) {
        std::vector<llama_token> wtoks(warm_prompt.size() + 16);
        wtoks.resize(llama_tokenize(vocab, warm_prompt.c_str(), (int) warm_prompt.size(), wtoks.data(), (int) wtoks.size(), true, true));
        llama_memory_clear(llama_get_memory(ctx), true);
        ggml_cpu_mmid_stats_reset();
        size_t wstart = 0, wreused = 0;
        const double wt0 = now_s();
        if (!state_load.empty()) {
            std::vector<llama_token> saved(std::max<size_t>(llama_n_ctx(ctx), wtoks.size()));
            size_t n_saved = 0;
            bool ok = llama_state_load_file(ctx, state_load.c_str(), saved.data(), saved.size(), &n_saved)
                && n_saved > 0 && n_saved <= wtoks.size() && std::equal(saved.begin(), saved.begin() + n_saved, wtoks.begin());
            if (ok && n_saved == wtoks.size()) {
                if (llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) (n_saved - 1), -1)) n_saved--;
                else ok = false;
            }
            if (ok) { wstart = wreused = n_saved; } else { llama_memory_clear(llama_get_memory(ctx), true); }
        }
        for (size_t i = wstart; i < wtoks.size(); i += n_ub) {
            const int n = (int) std::min<size_t>(n_ub, wtoks.size() - i);
            if (llama_decode(ctx, llama_batch_get_one(wtoks.data() + i, n)) != 0) { fprintf(stderr, "[gen] warm prefill failed\n"); return 5; }
        }
        const double wt_s = now_s() - wt0;
        fprintf(stderr, "[warm_ttft] prompt_tokens=%zu new_tokens=%zu ttft_s=%.3f (%.1f tok/s) tb=%d reused=%zu\n",
                wtoks.size(), wtoks.size() - wstart, wt_s, (wtoks.size() - wstart) / std::max(wt_s, 1e-6), threads_batch, wreused);
        ggml_cpu_mmid_stats_dump();
    }
    if (!vbench_sizes.empty() && !vbench_file.empty()) {
        std::vector<llama_token> vtoks(vbench_file.size() + 16);
        vtoks.resize(llama_tokenize(vocab, vbench_file.c_str(), (int) vbench_file.size(), vtoks.data(), (int) vtoks.size(), false, true));
        std::stringstream ss(vbench_sizes);
        std::string item;
        while (std::getline(ss, item, ',')) {
            int B = std::stoi(item);
            if (B <= 0) continue;
            llama_memory_clear(llama_get_memory(ctx), true);
            // Prefill short prefix so KV is non-empty
            if (vtoks.size() < (size_t) (32 + 20 * B)) continue;
            if (llama_decode(ctx, llama_batch_get_one(vtoks.data(), 32)) != 0) break;
            ggml_cpu_mmid_stats_reset();
            std::vector<double> vms, vcpu;
            uint64_t ch0, cm0, cmb0, cb0; ggml_cpu_mmid_cache_stats(&ch0, &cm0, &cmb0, &cb0);
            int n_steps = std::min<int>(20, ((int) vtoks.size() - 32) / B);
            for (int s_idx = 0; s_idx < n_steps; s_idx++) {
                double pi0, c0, pi1, c1; usage_now(pi0, c0);
                double t0 = now_s();
                if (llama_decode(ctx, llama_batch_get_one(vtoks.data() + 32 + s_idx * B, B)) != 0) { fprintf(stderr, "[vbench] decode failed B=%d\n", B); break; }
                vms.push_back((now_s() - t0) * 1000.0);
                usage_now(pi1, c1);
                vcpu.push_back((c1 - c0) * 1000.0);
            }
            uint64_t ch1, cm1, cmb1, cb1; ggml_cpu_mmid_cache_stats(&ch1, &cm1, &cmb1, &cb1);
            if (!vms.empty()) {
                auto vs = vms; std::sort(vs.begin(), vs.end());
                double sum_ms = 0, sum_cpu = 0;
                for (double x : vms) sum_ms += x;
                for (double x : vcpu) sum_cpu += x;
                double mean_ms = sum_ms / vms.size();
                double med_ms = vs[vs.size() / 2];
                double mean_cpu = sum_cpu / vcpu.size();
                double miss_step = (double) (cm1 - cm0) / vms.size();
                double miss_mb_step = (double) (cmb1 - cmb0) / 1048576.0 / vms.size();
                fprintf(stderr, "[vbench] B=%d steps=%zu tb=%d med=%.1fms mean=%.1fms cpu=%.1fms miss=%.1f/step (%.1fMB) foot=%.0fMB | eff@3.5=%.1f eff@4.0=%.1f tok/s\n",
                        B, vms.size(), threads_batch, med_ms, mean_ms, mean_cpu, miss_step, miss_mb_step, footprint_mb(),
                        3500.0 / (mean_ms + 18.0), 4000.0 / (mean_ms + 18.0));
                ggml_cpu_mmid_stats_dump();
            }
        }
    }
    std::vector<double> s = ms; std::sort(s.begin(), s.end());
    double sum = 0; for (double x : ms) sum += x;
    fprintf(stderr, "[gen] prompt_tokens=%zu prefill_s=%.2f (%.1f tok/s) decode_tokens=%zu median_ms=%.0f mean_ms=%.0f (%.2f tok/s) footprint=%.0f MB reused=%zu\n",
           toks.size(), prefill_s, toks.size() / prefill_s, ms.size(), s.empty() ? 0 : s[s.size() / 2], ms.empty() ? 0 : sum / ms.size(),
           ms.empty() ? 0 : 1000.0 * ms.size() / sum, footprint_mb(), reused);
    fprintf(stderr, "[text] %s\n", text.c_str());
    ggml_cpu_node_prof_dump();
    ggml_cpu_mmid_stats_dump();
    llama_free(ctx); llama_model_free(m);
    return 0;
}
