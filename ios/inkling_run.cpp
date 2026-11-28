#include <algorithm>
// Phone/Mac generate command with visibility: load progress, footprint, per-token times.
// Usage: gen -m model.gguf [-ngl 99] [-n 32] [-c 2048] [-ub 64] [-t 4] [-lm mmap|none] [-lzm on|auto|off] [-cpu <tensor regex>] [-f prompt.txt | -p prompt]
//        [-temp F] [-top-k N] [-top-p F] [-min-p F] [-repeat-penalty F] [-repeat-last-n N] [-seed N]   (sampling; env INKLING_TEMP/TOP_K/TOP_P/MIN_P/REPEAT_PENALTY/REPEAT_LAST_N/SEED)
//        [-demo]   (demo mode: on-screen UI hooks + demo sampling defaults; also INKLING_DEMO=1)
//        [-script turns.txt]   (demo only, also INKLING_DEMO_SCRIPT: multi-turn chat; user turns separated by "---" lines, KV cache kept, -n = per-turn cap)
// Default temp=0 and repeat_penalty=1 => greedy path (fused argmax head), logs unchanged.
#include "llama.h"
#include "ggml-backend.h"
#include "../tools/eagle_draft.hpp"
#include <mach/mach.h>
#include <sys/resource.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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

// ---- Demo-mode UI hook (see ios_main.mm). Null callback => no extra work beyond one null check. ----
// kinds: STATUS text="loading|prefill|generating|..." progress 0..1 | PROMPT text=full prompt file |
//        TEXT answer chunk | THINK thinking chunk | FINAL text="done", n_tok, tok_s, ram_mb
// Script mode (-script) adds: TURN_BEGIN text=user turn (no template), n_tok=turn index (1-based), progress=turn count |
//        TURN_END n_tok=decoded tokens of the turn, tok_s, ram_mb.
//        FINAL then carries total decode tokens, average tok/s and peak RAM; text is "done" or a "stopped: ..." note.
// Demo numbers: every tok_s is the steady decode rate = (decoded tokens after the first 3 of the turn) / (their summed
// per-token llama_decode ms), 0 until more than 3 tokens were decoded (UI shows "—"); script FINAL carries total steady
// tokens / total steady decode time. Time to first token (turn prefill start -> first generated token available, s) rides
// in `progress` of: STATUS "generating" (sent once, when known), TEXT/THINK (0 = not yet known), TURN_END and the
// single-turn FINAL.
enum { INKLING_UI_STATUS = 0, INKLING_UI_PROMPT = 1, INKLING_UI_TEXT = 2, INKLING_UI_FINAL = 3, INKLING_UI_THINK = 4,
       INKLING_UI_TURN_BEGIN = 5, INKLING_UI_TURN_END = 6 };
typedef void (*inkling_ui_cb)(int kind, const char * text, int n_tok, double tok_s, double ram_mb, float progress);
static inkling_ui_cb g_ui_cb = nullptr;
extern "C" void inkling_set_ui_callback(inkling_ui_cb cb) { g_ui_cb = cb; }

static void ui_status(const char * s, float progress, int n_tok = 0, double tok_s = 0.0) {
    if (g_ui_cb) g_ui_cb(INKLING_UI_STATUS, s, n_tok, tok_s, footprint_mb(), progress);
}

// Accumulates generated pieces for the UI; flushes every 4 tokens or ~100 ms, and on thinking/answer switches.
static const size_t kDemoSkipTok = 3;   // demo "steady" decode rate excludes the first 3 decoded tokens of each turn
struct SteadyStats {
    size_t n = 0; double ms = 0.0;
    double tok_s() const { return n > 0 && ms > 0.0 ? 1000.0 * (double) n / ms : 0.0; }
};
static SteadyStats steady_of(const std::vector<double> & ms) {
    SteadyStats s;
    for (size_t i = kDemoSkipTok; i < ms.size(); ++i) { s.n++; s.ms += ms[i]; }
    return s;
}

struct InklingUiStream {
    std::string buf;
    bool thinking = false;
    int n = 0, pend = 0;
    double last = 0.0;
    const std::vector<double> * dec_ms = nullptr;   // per-token llama_decode ms of this turn (same values as the [gen]/[turn] lines)
    size_t acc_i = kDemoSkipTok;                    // next index of dec_ms to add to acc_ms (steady part only)
    double acc_ms = 0.0, ttft = 0.0;

    void start(const std::vector<double> * ms) {
        n = pend = 0; buf.clear(); thinking = false; last = now_s();
        dec_ms = ms; acc_i = kDemoSkipTok; acc_ms = 0.0; ttft = 0.0;
    }
    void first_token(double ttft_s) { ttft = ttft_s; ui_status("generating", (float) ttft_s); }
    double steady_tok_s() {
        if (!dec_ms) return 0.0;
        while (acc_i < dec_ms->size()) acc_ms += (*dec_ms)[acc_i++];
        const size_t cnt = acc_i - kDemoSkipTok;
        return cnt > 0 && acc_ms > 0.0 ? 1000.0 * (double) cnt / acc_ms : 0.0;
    }
    static size_t complete_len(const std::string & s) {   // do not split a UTF-8 sequence across flushes
        const size_t sz = s.size();
        for (size_t back = 1; back <= 3 && back <= sz; ++back) {
            const unsigned char c = (unsigned char) s[sz - back];
            if ((c & 0xC0) == 0x80) continue;
            const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
            return need > back ? sz - back : sz;
        }
        return sz;
    }
    void flush(bool force) {
        const size_t len = force ? buf.size() : complete_len(buf);
        if (len == 0) return;
        const double now = now_s();
        const std::string chunk = buf.substr(0, len);
        buf.erase(0, len);
        g_ui_cb(thinking ? INKLING_UI_THINK : INKLING_UI_TEXT, chunk.c_str(), n, steady_tok_s(), footprint_mb(), (float) ttft);
        pend = 0; last = now;
    }
    void token(const llama_vocab * vocab, llama_token tk) {
        n++;
        char sp[256];
        const int k = llama_token_to_piece(vocab, tk, sp, sizeof sp, 0, true);
        if (k > 2 && sp[0] == '<' && sp[1] == '|') {
            const std::string s(sp, (size_t) k);
            if (s == "<|content_thinking|>") { flush(true); thinking = true; }
            else if (s == "<|content_text|>" || s == "<|end_message|>") { flush(true); thinking = false; }
        }
        char pc[256];
        const int j = llama_token_to_piece(vocab, tk, pc, sizeof pc, 0, false);
        if (j > 0) buf.append(pc, (size_t) j);
        if (++pend >= 4 || now_s() - last >= 0.1) flush(false);
    }
};

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

// ---- Demo script mode: multi-turn chat with the KV cache kept between turns (demo + -script only). ----
// Script file: plain-text user turns (no chat template) separated by lines that are exactly "---".
static bool read_script_turns(const std::string & path, std::vector<std::string> & turns) {
    std::ifstream in(path);
    if (!in) return false;
    std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto trim = [](const std::string & s) -> std::string {
        const char * ws = " \t\r\n";
        const size_t b = s.find_first_not_of(ws);
        if (b == std::string::npos) return std::string();
        return s.substr(b, s.find_last_not_of(ws) - b + 1);
    };
    std::string cur, line;
    auto push = [&]() { const std::string t = trim(cur); if (!t.empty()) turns.push_back(t); cur.clear(); };
    std::istringstream ls(all);
    while (std::getline(ls, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "---") push();
        else { cur += line; cur.push_back('\n'); }
    }
    push();
    return true;
}

static bool tokenize_text(const llama_vocab * vocab, const std::string & s, bool add_special, std::vector<llama_token> & out) {
    out.assign(s.size() + 16, 0);
    int n = llama_tokenize(vocab, s.c_str(), (int) s.size(), out.data(), (int) out.size(), add_special, true);
    if (n < 0) {
        out.assign((size_t) -n, 0);
        n = llama_tokenize(vocab, s.c_str(), (int) s.size(), out.data(), (int) out.size(), add_special, true);
        if (n < 0) return false;
    }
    out.resize((size_t) n);
    return true;
}

static int run_demo_script(llama_context * ctx, const llama_vocab * vocab, llama_sampler * smpl,
                           const std::vector<std::string> & turns, int n_gen, int n_ub) {
    const int n_ctx = (int) llama_n_ctx(ctx);
    const int N = (int) turns.size();
    int n_past = 0, turns_done = 0;
    llama_token pending = LLAMA_TOKEN_NULL;   // EOG token sampled at the end of the last reply, not yet decoded into the cache
    size_t total_tok = 0, total_steady_n = 0;
    double total_ms = 0.0, total_steady_ms = 0.0, sum_turn_tok_s = 0.0, peak = footprint_mb();
    int rc = 0;
    std::string stop_note;
    fprintf(stderr, "[demo] script turns=%d n_ctx=%d per_turn_cap=%d\n", N, n_ctx, n_gen);
    ggml_cpu_node_prof_reset();
    for (int ti = 0; ti < N; ++ti) {
        std::string add;
        if (ti == 0) {
            add = "<|message_system|><|content_text|>Thinking effort level: 0.0<|end_message|>";
        } else if (pending == LLAMA_TOKEN_NULL) {   // previous reply hit the cap: close it by hand
            add = "<|end_message|><|content_model_end_sampling|>";
        }
        add += "<|message_user|><|content_text|>" + turns[ti] + "<|end_message|><|message_model|>";
        std::vector<llama_token> in;
        if (pending != LLAMA_TOKEN_NULL) in.push_back(pending);
        std::vector<llama_token> tail;
        if (!tokenize_text(vocab, add, ti == 0, tail)) { fprintf(stderr, "[demo] tokenize failed turn %d\n", ti + 1); return 2; }
        in.insert(in.end(), tail.begin(), tail.end());
        const int n_in = (int) in.size();
        if (n_past + n_in + 1 > n_ctx) {
            char msg[160];
            snprintf(msg, sizeof msg, "stopped: context full before turn %d/%d", ti + 1, N);
            stop_note = msg;
            fprintf(stderr, "[demo] %s (needs %d tokens, %d/%d used)\n", msg, n_in, n_past, n_ctx);
            break;
        }
        if (smpl) llama_sampler_reset(smpl);
        if (g_ui_cb) g_ui_cb(INKLING_UI_TURN_BEGIN, turns[ti].c_str(), ti + 1, 0.0, footprint_mb(), (float) N);
        if (ti > 0) ggml_cpu_mmid_stats_reset();
        if (g_ui_cb) ui_status("prefill", 0.0f);
        double t = now_s();
        const double t_pf0 = t;   // turn TTFT starts here (start of this turn's prefill)
        double ttft_s = 0.0;
        for (int i = 0; i < n_in; i += n_ub) {
            const int n = std::min(n_ub, n_in - i);
            if (llama_decode(ctx, llama_batch_get_one(in.data() + i, n)) != 0) {
                fprintf(stderr, "[demo] prefill failed turn %d\n", ti + 1); rc = 2; break;
            }
            if (g_ui_cb) ui_status("prefill", (float) (i + n) / (float) n_in);
        }
        if (rc != 0) break;
        const double prefill_s = now_s() - t;
        n_past += n_in;
        pending = LLAMA_TOKEN_NULL;
        peak = std::max(peak, footprint_mb());
        ggml_cpu_mmid_stats_dump();
        ggml_cpu_mmid_stats_reset();

        std::vector<double> ms;
        std::string text;
        InklingUiStream ui;
        if (g_ui_cb) { ui.start(&ms); ui_status("generating", 0.0f); }
        const int gen_max = std::min(n_gen, n_ctx - n_past);
        for (int g = 0; g < gen_max; ++g) {
            llama_token best = smpl ? llama_sampler_sample(smpl, ctx, -1) : llama_get_greedy_token_ith(ctx, -1);
            if (g == 0) { ttft_s = now_s() - t_pf0; if (g_ui_cb) ui.first_token(ttft_s); }
            if (llama_vocab_is_eog(vocab, best)) { pending = best; break; }
            char buf[256]; const int k = llama_token_to_piece(vocab, best, buf, sizeof buf, 0, true);
            if (k > 0) text.append(buf, (size_t) k);
            if (g_ui_cb) ui.token(vocab, best);
            t = now_s();
            if (llama_decode(ctx, llama_batch_get_one(&best, 1)) != 0) { fprintf(stderr, "[demo] decode failed turn %d\n", ti + 1); rc = 3; break; }
            ms.push_back((now_s() - t) * 1000.0);
            n_past++;
            peak = std::max(peak, footprint_mb());
        }
        if (g_ui_cb) ui.flush(true);
        if (rc != 0) break;
        std::vector<double> s = ms; std::sort(s.begin(), s.end());
        double sum = 0; for (double x : ms) sum += x;
        const double turn_tok_s = ms.empty() ? 0.0 : 1000.0 * ms.size() / sum;
        const SteadyStats st = steady_of(ms);
        fprintf(stderr, "[turn] i=%d/%d prompt_tokens=%d prefill_s=%.2f decode_tokens=%zu median_ms=%.0f mean_ms=%.0f (%.2f tok/s) ctx_used=%d footprint=%.0f MB steady_tok_s=%.2f ttft_s=%.3f\n",
                ti + 1, N, n_in, prefill_s, ms.size(), s.empty() ? 0.0 : s[s.size() / 2], ms.empty() ? 0.0 : sum / ms.size(),
                turn_tok_s, n_past, footprint_mb(), st.tok_s(), ttft_s);
        fprintf(stderr, "[turn_text] i=%d %s\n", ti + 1, text.c_str());
        ggml_cpu_mmid_stats_dump();
        total_tok += ms.size(); total_ms += sum;
        total_steady_n += st.n; total_steady_ms += st.ms;
        if (!ms.empty()) sum_turn_tok_s += turn_tok_s;
        turns_done++;
        if (g_ui_cb) g_ui_cb(INKLING_UI_TURN_END, "", (int) ms.size(), st.tok_s(), footprint_mb(), (float) ttft_s);
    }
    const double avg_tok_s = total_ms > 0.0 ? 1000.0 * total_tok / total_ms : 0.0;
    const double avg_steady_tok_s = total_steady_ms > 0.0 ? 1000.0 * (double) total_steady_n / total_steady_ms : 0.0;
    fprintf(stderr, "[demo] turns=%d total_decode_tokens=%zu avg_tok_s=%.2f mean_of_turn_tok_s=%.2f peak_footprint=%.0f MB avg_steady_tok_s=%.2f\n",
            turns_done, total_tok, avg_tok_s, turns_done > 0 ? sum_turn_tok_s / turns_done : 0.0, peak, avg_steady_tok_s);
    if (g_ui_cb && rc == 0) g_ui_cb(INKLING_UI_FINAL, stop_note.empty() ? "done" : stop_note.c_str(), (int) total_tok, avg_steady_tok_s, peak, 1.0f);
    ggml_cpu_node_prof_dump();
    return rc;
}

static double g_t0, g_last;
static double g_ui_last_load = 0;
static bool on_progress(float p, void *) {
    const double t = now_s();
    if (g_ui_cb && (t - g_ui_last_load > 0.1 || p >= 1.0f)) { g_ui_last_load = t; ui_status("loading", p); }
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
    std::string script_path, lora, state_save, state_load, warm_prompt, vbench_file, vbench_sizes, eagle_path, prompts_file, ctk = "f16", ctv = "f16";   // -lora <file>: GGUF LoRA adapter applied at scale 1
    std::string model, prompt = "<|message_system|><|content_text|>Thinking effort level: 0.0<|end_message|><|message_user|><|content_text|>Give me three short tips for sleeping better.<|end_message|><|message_model|>";
    int ngl = 99, n_gen = 32, n_ctx = 2048, n_ub = 64, threads = 4, threads_batch = 0, n_draft = 1; bool use_mmap = true; std::string cpu_pat; int passes = 0, n_used = 0; llama_lazy_mode lazy = LLAMA_LAZY_MODE_AUTO;
    bool demo = false, prompt_file_failed = false;
    float temp = 0.0f, top_p = 1.0f, min_p = 0.0f, repeat_penalty = 1.0f; int top_k = 0, repeat_last_n = 64; uint32_t seed = LLAMA_DEFAULT_SEED;
    bool have_temp = false, have_top_k = false, have_top_p = false, have_min_p = false, have_rp = false, have_rln = false, have_seed = false;
    for (int i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "-demo") == 0) { demo = true; i--; continue; }   // value-less flag
        if (i + 1 >= argc) break;
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
        else if (a == "-f") { std::ifstream f(v); if (!f) prompt_file_failed = true; std::stringstream ss; ss << f.rdbuf(); prompt = ss.str(); }
        else if (a == "-prompts") prompts_file = v;
        else if (a == "-script") script_path = v;
        else if (a == "-wf") { std::ifstream f(v); std::stringstream ss; ss << f.rdbuf(); warm_prompt = ss.str(); }
        else if (a == "-wp") warm_prompt = v;
        else if (a == "-state-save") state_save = v; else if (a == "-state-load") state_load = v;
        else if (a == "-temp") { temp = std::stof(v); have_temp = true; }
        else if (a == "-top-k") { top_k = std::stoi(v); have_top_k = true; }
        else if (a == "-top-p") { top_p = std::stof(v); have_top_p = true; }
        else if (a == "-min-p") { min_p = std::stof(v); have_min_p = true; }
        else if (a == "-repeat-penalty") { repeat_penalty = std::stof(v); have_rp = true; }
        else if (a == "-repeat-last-n") { repeat_last_n = std::stoi(v); have_rln = true; }
        else if (a == "-seed") { seed = (uint32_t) std::stoull(v); have_seed = true; }
    }
    {   // sampling: args win over INKLING_* env, env wins over demo defaults, then the greedy defaults
        const char * ed = getenv("INKLING_DEMO");
        if (ed && strcmp(ed, "1") == 0) demo = true;
        auto env_s = [](const char * n) -> const char * { const char * e = getenv(n); return e && e[0] ? e : nullptr; };
        const char * e;
        if (!have_temp)   { if ((e = env_s("INKLING_TEMP")))           temp = strtof(e, nullptr);           else if (demo) temp = 0.7f; }
        if (!have_top_k)  { if ((e = env_s("INKLING_TOP_K")))          top_k = atoi(e);                     else if (demo) top_k = 40; }
        if (!have_top_p)  { if ((e = env_s("INKLING_TOP_P")))          top_p = strtof(e, nullptr);          else if (demo) top_p = 0.9f; }
        if (!have_min_p)  { if ((e = env_s("INKLING_MIN_P")))          min_p = strtof(e, nullptr);          else if (demo) min_p = 0.05f; }
        if (!have_rp)     { if ((e = env_s("INKLING_REPEAT_PENALTY"))) repeat_penalty = strtof(e, nullptr); else if (demo) repeat_penalty = 1.1f; }
        if (!have_rln)    { if ((e = env_s("INKLING_REPEAT_LAST_N")))  repeat_last_n = atoi(e);             else if (demo) repeat_last_n = 128; }
        if (!have_seed)   { if ((e = env_s("INKLING_SEED")))           seed = (uint32_t) strtoull(e, nullptr, 10); }
    }
    bool sampling = temp > 0.0f || repeat_penalty != 1.0f;
    if (sampling && !prompts_file.empty()) {
        fprintf(stderr, "[sampling] ignored with -prompts (greedy sweep)\n");
        sampling = false;
    }
    if (sampling) {
        if (!eagle_path.empty()) { fprintf(stderr, "[sampling] eagle disabled (greedy only)\n"); eagle_path.clear(); }
        // the skip-softcap shortcut is only valid for argmax; sampling needs the true (soft-capped) logits
        if (getenv("INKLING_GREEDY_SKIP_SOFTCAP")) { fprintf(stderr, "[sampling] INKLING_GREEDY_SKIP_SOFTCAP ignored\n"); unsetenv("INKLING_GREEDY_SKIP_SOFTCAP"); }
    }
    std::vector<std::string> script_turns;
    bool script_mode = false;   // demo + script: multi-turn chat, KV cache kept between turns
    if (demo) {
        if (script_path.empty()) { const char * es = getenv("INKLING_DEMO_SCRIPT"); if (es && es[0]) script_path = es; }
        if (!script_path.empty()) {
            if (!read_script_turns(script_path, script_turns)) { fprintf(stderr, "[demo] script open failed %s\n", script_path.c_str()); return 2; }
            if (script_turns.empty()) { fprintf(stderr, "[demo] script has no turns %s\n", script_path.c_str()); return 2; }
            script_mode = true;
            if (!eagle_path.empty()) { fprintf(stderr, "[demo] eagle disabled in script mode\n"); eagle_path.clear(); }
        }
    }
    if (demo && prompt_file_failed && !script_mode) {
        prompt = "<|message_system|><|content_text|>Thinking effort level: 0.0<|end_message|><|message_user|><|content_text|>Explain in simple terms how a phone can run a large AI model without the internet.<|end_message|><|message_model|>";
        fprintf(stderr, "[demo] prompt file missing, using built-in prompt\n");
    }
    if (g_ui_cb) { if (!script_mode) g_ui_cb(INKLING_UI_PROMPT, prompt.c_str(), 0, 0.0, 0.0, 0.0f); ui_status("loading", 0.0f); }
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
    if (!sampling) llama_set_greedy_head(ctx, true);   // the fused argmax head may skip full logits; sampling needs them
    if (!lora.empty()) {
        llama_adapter_lora * adapter = llama_adapter_lora_init(m, lora.c_str());
        float scale = 1.0f;
        if (!adapter || llama_set_adapters_lora(ctx, &adapter, 1, &scale) != 0) { fprintf(stderr, "[gen] lora failed: %s\n", lora.c_str()); return 1; }
        fprintf(stderr, "[gen] lora %s applied footprint=%.0f MB\n", lora.c_str(), footprint_mb());
    }
    fprintf(stderr, "[gen] context t=%.1fs footprint=%.0f MB\n", now_s() - g_t0, footprint_mb());
    const llama_vocab * vocab = llama_model_get_vocab(m);
    llama_sampler * smpl = nullptr;
    if (sampling) {
        smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), repeat_last_n, repeat_penalty, 0.0f, 0.0f));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
        fprintf(stderr, "[sampling] temp=%g top_k=%d top_p=%g min_p=%g repeat_penalty=%g last_n=%d seed=%u\n",
                temp, top_k, top_p, min_p, repeat_penalty, repeat_last_n, (unsigned) seed);
    }
    if (!prompts_file.empty()) {
        const int rc = run_prompt_sweep(ctx, vocab, prompts_file, n_gen, n_ub);
        llama_free(ctx); llama_model_free(m);
        return rc;
    }
    if (script_mode) {
        const int rc = run_demo_script(ctx, vocab, smpl, script_turns, n_gen, n_ub);
        if (smpl) llama_sampler_free(smpl);
        llama_free(ctx); llama_model_free(m);
        return rc;
    }
    std::vector<llama_token> toks(prompt.size() + 16);
    toks.resize(llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), true, true));
    double t = now_s();
    const double ttft_t0 = t;   // demo TTFT starts with the prompt prefill (model load excluded)
    double ttft_s = 0.0;
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
    if (g_ui_cb) ui_status("prefill", 0.0f);
    for (size_t i = start; i < toks.size(); i += n_ub) {
        const int n = (int) std::min<size_t>(n_ub, toks.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, n)) != 0) { fprintf(stderr, "[gen] prefill failed\n"); return 2; }
        if (g_ui_cb) ui_status("prefill", (float) (i + n) / (float) toks.size());
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
    InklingUiStream ui;
    if (g_ui_cb) { ui.start(&ms); ui_status("generating", 0.0f); }
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
        if (demo) { ttft_s = now_s() - ttft_t0; if (g_ui_cb) ui.first_token(ttft_s); }
        int cur_pos = (int) toks.size();
        int n_emitted = 0, total_steps = 0, acc1_hits = 0, acc1_tot = 0, acc2_hits = 0, acc2_tot = 0;
        double sum_draft_ms = 0.0, sum_verify_ms = 0.0;
        const double te0 = now_s();
        auto append_tok = [&](llama_token tk) {
            char buf[256];
            const int k = llama_token_to_piece(vocab, tk, buf, sizeof buf, 0, true);
            if (k > 0) text.append(buf, k);
            if (g_ui_cb) ui.token(vocab, tk);
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
        llama_token best = smpl ? llama_sampler_sample(smpl, ctx, -1) : llama_get_greedy_token_ith(ctx, -1);
        if (demo && g == 0) { ttft_s = now_s() - ttft_t0; if (g_ui_cb) ui.first_token(ttft_s); }
        if (llama_vocab_is_eog(vocab, best)) break;
        char buf[256]; const int k = llama_token_to_piece(vocab, best, buf, sizeof buf, 0, true);
        if (k > 0) text.append(buf, k);
        if (g_ui_cb) ui.token(vocab, best);
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
    const SteadyStats demo_st = steady_of(ms);
    if (demo) fprintf(stderr, "[demo_turn] steady_tok_s=%.2f ttft_s=%.3f\n", demo_st.tok_s(), ttft_s);
    fprintf(stderr, "[text] %s\n", text.c_str());
    if (g_ui_cb) {
        ui.flush(true);
        if (demo) g_ui_cb(INKLING_UI_FINAL, "done", (int) ms.size(), demo_st.tok_s(), footprint_mb(), (float) ttft_s);
        else g_ui_cb(INKLING_UI_FINAL, "done", (int) ms.size(), ms.empty() ? 0.0 : 1000.0 * ms.size() / sum, footprint_mb(), 1.0f);
    }
    ggml_cpu_node_prof_dump();
    ggml_cpu_mmid_stats_dump();
    if (smpl) llama_sampler_free(smpl);
    llama_free(ctx); llama_model_free(m);
    return 0;
}
