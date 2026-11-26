// Compute exact REAP (Router-weighted Expert Activation Pruning) salience scores across all layers
// using llama.cpp cb_eval on real quantized weights (1p K4 + optional LoRA).
//
// Usage:
//   reap_trace -m model.gguf -f calib.jsonl [-lora adapter.gguf] [-passes 1] [-k 4] [-t 6] [-ub 128] [-max-tok 80000] -o out.json
//
// Input calib.jsonl: one JSON object per line with {"text": "<rendered inkling chat text>"} (or raw lines if not JSON).
// For each layer il in 0..n_layer-1 and expert e in 0..n_expert-1, accumulates:
//   freq[il][e]     = selection count across all evaluated tokens
//   reap_sum[il][e] = sum_t || ffn_moe_weighted[il, t, e] ||_2  (= sum_t |w_e| * ||down_e(x_t)||_2)
//   w_sum[il][e]    = sum_t w_e  (from ffn_moe_weights_scaled / ffn_moe_weights_norm / ffn_moe_weights)
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct ReapState {
    int n_layer = 24;
    int n_expert = 128;
    int n_used = 4;
    uint64_t total_tokens = 0;

    std::vector<std::vector<uint64_t>> freq;
    std::vector<std::vector<double>> reap_sum;
    std::vector<std::vector<double>> w_sum;

    // Per-layer staging during one batch eval (topk arrives before weights and weighted)
    std::vector<std::vector<int32_t>> cur_topk;
    std::vector<int> cur_k;
    std::vector<int> cur_n;

    void init(int nl, int ne, int k) {
        n_layer = nl; n_expert = ne; n_used = k;
        freq.assign(nl, std::vector<uint64_t>(ne, 0));
        reap_sum.assign(nl, std::vector<double>(ne, 0.0));
        w_sum.assign(nl, std::vector<double>(ne, 0.0));
        cur_topk.assign(nl, {});
        cur_k.assign(nl, 0);
        cur_n.assign(nl, 0);
    }
};

static bool on_eval(ggml_tensor * t, bool ask, void * ud) {
    const bool is_topk     = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    const bool is_wscaled  = strncmp(t->name, "ffn_moe_weights_scaled-", 23) == 0;
    const bool is_weighted = strncmp(t->name, "ffn_moe_weighted-", 17) == 0;
    const bool want = is_topk || is_wscaled || is_weighted;
    if (ask || !want) return want;

    ReapState * st = (ReapState *) ud;
    if (is_topk) {
        int il = atoi(t->name + 13);
        if (il < 0 || il >= st->n_layer) return true;
        int k = (int) t->ne[0], n = (int) t->ne[1];
        st->cur_k[il] = k;
        st->cur_n[il] = n;
        st->cur_topk[il].resize((size_t) k * n);
        ggml_backend_tensor_get(t, st->cur_topk[il].data(), 0, st->cur_topk[il].size() * sizeof(int32_t));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                int32_t e = st->cur_topk[il][(size_t) i * k + j];
                if (e >= 0 && e < st->n_expert) st->freq[il][e]++;
            }
        }
        return true;
    }
    if (is_wscaled) {
        int il = atoi(t->name + 23);
        if (il < 0 || il >= st->n_layer) return true;
        int k = st->cur_k[il], n = st->cur_n[il];
        if (k <= 0 || n <= 0) return true;
        std::vector<float> buf((size_t) k * n);
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                int32_t e = st->cur_topk[il][(size_t) i * k + j];
                if (e >= 0 && e < st->n_expert) st->w_sum[il][e] += (double) std::fabs(buf[(size_t) i * k + j]);
            }
        }
        return true;
    }
    if (is_weighted) {
        int il = atoi(t->name + 17);
        if (il < 0 || il >= st->n_layer) return true;
        int n_embd = (int) t->ne[0], k = (int) t->ne[1], n = (int) t->ne[2];
        if (k != st->cur_k[il] || n != st->cur_n[il]) return true;
        if (t->type != GGML_TYPE_F32) return true;
        const size_t total = (size_t) n_embd * k * n;
        std::vector<float> buf(total);
        ggml_backend_tensor_get(t, buf.data(), 0, total * sizeof(float));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                int32_t e = st->cur_topk[il][(size_t) i * k + j];
                if (e < 0 || e >= st->n_expert) continue;
                const float * v = buf.data() + ((size_t) i * k + j) * n_embd;
                double s2 = 0.0;
                for (int d = 0; d < n_embd; d++) s2 += (double) v[d] * (double) v[d];
                st->reap_sum[il][e] += std::sqrt(s2);
            }
        }
        return true;
    }
    return true;
}

static std::string extract_text(const std::string & line) {
    // Minimal JSON {"text": "..."} unescaper; falls back to raw line if not starting with '{'
    size_t p = line.find("\"text\"");
    if (line.empty() || line[0] != '{' || p == std::string::npos) return line;
    p = line.find(':', p + 6);
    if (p == std::string::npos) return "";
    p = line.find('"', p + 1);
    if (p == std::string::npos) return "";
    std::string out;
    out.reserve(line.size());
    for (size_t i = p + 1; i < line.size(); i++) {
        char c = line[i];
        if (c == '\\' && i + 1 < line.size()) {
            char nx = line[++i];
            if      (nx == 'n') out.push_back('\n');
            else if (nx == 'r') out.push_back('\r');
            else if (nx == 't') out.push_back('\t');
            else if (nx == '"' || nx == '\\' || nx == '/') out.push_back(nx);
            else if (nx == 'u' && i + 4 < line.size()) {
                unsigned code = (unsigned) std::strtoul(line.substr(i + 1, 4).c_str(), nullptr, 16);
                i += 4;
                if (code < 0x80) out.push_back((char) code);
                else if (code < 0x800) { out.push_back((char)(0xC0 | (code >> 6))); out.push_back((char)(0x80 | (code & 0x3F))); }
                else { out.push_back((char)(0xE0 | (code >> 12))); out.push_back((char)(0x80 | ((code >> 6) & 0x3F))); out.push_back((char)(0x80 | (code & 0x3F))); }
            }
        } else if (c == '"') {
            break;
        } else {
            out.push_back(c);
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    std::string model_path, calib_path, lora_path, out_path = "reap_scores.json";
    int passes = 1, n_used = 4, threads = 6, n_ub = 128, n_ctx = 2048, max_tok = 80000;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if      (a == "-m") model_path = v;
        else if (a == "-f") calib_path = v;
        else if (a == "-lora") lora_path = v;
        else if (a == "-o") out_path = v;
        else if (a == "-passes") passes = std::stoi(v);
        else if (a == "-k") n_used = std::stoi(v);
        else if (a == "-t") threads = std::stoi(v);
        else if (a == "-ub") n_ub = std::stoi(v);
        else if (a == "-c") n_ctx = std::stoi(v);
        else if (a == "-max-tok") max_tok = std::stoi(v);
    }
    if (model_path.empty() || calib_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -f calib.jsonl [-lora adapter.gguf] [-passes 1] [-k 4] [-t 6] [-ub 128] [-max-tok 80000] [-o out.json]\n", argv[0]);
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
    if (!model) { fprintf(stderr, "[reap] model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_layer = llama_model_n_layer(model);
    const int n_expert = 128; // verified via KV

    ReapState st;
    st.init(n_layer, n_expert, n_used);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ub; cp.n_ubatch = n_ub;
    cp.op_offload = false; cp.offload_kqv = false;
    cp.n_threads = threads; cp.n_threads_batch = threads;
    cp.cb_eval = on_eval; cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "[reap] ctx failed\n"); return 1; }

    if (!lora_path.empty()) {
        llama_adapter_lora * ad = llama_adapter_lora_init(model, lora_path.c_str());
        float scale = 1.0f;
        if (!ad || llama_set_adapters_lora(ctx, &ad, 1, &scale) != 0) {
            fprintf(stderr, "[reap] lora failed: %s\n", lora_path.c_str());
            return 1;
        }
        fprintf(stderr, "[reap] lora loaded: %s\n", lora_path.c_str());
    }

    auto t0 = std::chrono::steady_clock::now();
    std::ifstream in(calib_path);
    std::string line;
    int docs = 0;
    while (std::getline(in, line) && (int) st.total_tokens < max_tok) {
        std::string text = extract_text(line);
        if (text.size() < 16) continue;
        std::vector<llama_token> toks(text.size() + 16);
        int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), false, true);
        if (nt <= 0) continue;
        if (nt > n_ctx - 8) nt = n_ctx - 8;
        toks.resize(nt);

        llama_memory_clear(llama_get_memory(ctx), true);
        for (int i = 0; i < nt; i += n_ub) {
            int n = std::min(n_ub, nt - i);
            if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, n)) != 0) {
                fprintf(stderr, "[reap] decode failed doc %d\n", docs);
                return 2;
            }
        }
        st.total_tokens += nt;
        docs++;
        if (docs % 10 == 0 || (int) st.total_tokens >= max_tok) {
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "[reap] docs=%d tokens=%llu (%.1f tok/s, %.1fs)\n",
                    docs, (unsigned long long) st.total_tokens, st.total_tokens / std::max(dt, 1e-6), dt);
        }
    }

    // Write JSON summary
    FILE * out = fopen(out_path.c_str(), "w");
    fprintf(out, "{\n  \"model\": \"%s\",\n  \"lora\": \"%s\",\n  \"passes\": %d,\n  \"top_k\": %d,\n  \"docs\": %d,\n  \"total_tokens\": %llu,\n  \"n_layer\": %d,\n  \"n_expert\": %d,\n",
            model_path.c_str(), lora_path.c_str(), passes, n_used, docs, (unsigned long long) st.total_tokens, n_layer, n_expert);
    auto dump_mat_u64 = [&](const char * key, const std::vector<std::vector<uint64_t>> & m, bool comma) {
        fprintf(out, "  \"%s\": [\n", key);
        for (int l = 0; l < n_layer; l++) {
            fprintf(out, "    [");
            for (int e = 0; e < n_expert; e++) fprintf(out, "%s%llu", e ? "," : "", (unsigned long long) m[l][e]);
            fprintf(out, "]%s\n", l + 1 < n_layer ? "," : "");
        }
        fprintf(out, "  ]%s\n", comma ? "," : "");
    };
    auto dump_mat_f64 = [&](const char * key, const std::vector<std::vector<double>> & m, bool comma) {
        fprintf(out, "  \"%s\": [\n", key);
        for (int l = 0; l < n_layer; l++) {
            fprintf(out, "    [");
            for (int e = 0; e < n_expert; e++) fprintf(out, "%s%.6g", e ? "," : "", m[l][e]);
            fprintf(out, "]%s\n", l + 1 < n_layer ? "," : "");
        }
        fprintf(out, "  ]%s\n", comma ? "," : "");
    };
    dump_mat_u64("freq", st.freq, true);
    dump_mat_f64("w_sum", st.w_sum, true);
    dump_mat_f64("reap_sum", st.reap_sum, false);
    fprintf(out, "}\n");
    fclose(out);

    fprintf(stderr, "[reap] wrote %s (docs=%d tokens=%llu)\n", out_path.c_str(), docs, (unsigned long long) st.total_tokens);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
