// Dump exact quantized router/MoE inputs x_L [N_FMRD, 2304] (FP16) for all 24 layers
// AND final-normed hidden states h_t [N_EAGLE, 2304] (FP16) + token IDs for EAGLE drafter training
// in a single prefill-speed pass over a leak-free calibration JSONL corpus.
//
// Usage:
//   dump_fmrd_data -m model.gguf -f calib.jsonl -o /tmp/fmrd_data \
//                  [-passes 1] [-k 4] [-t 6] [-ub 128] [-c 2048] \
//                  [-n-fmrd 8192] [-n-eagle 65536] [-stride-fmrd 8]
//
// Outputs in <out_dir>:
//   meta.json             : shapes, layer count, n_embd, n_fmrd, n_eagle, docs, total_tokens
//   x_L00.f16 .. x_L23.f16: [n_fmrd, 2304] uint16 (IEEE FP16) exact input to blk.L.ffn_gate_inp & MoE
//   eagle_h.f16           : [n_eagle, 2304] uint16 (IEEE FP16) exact result_norm before lm_head
//   eagle_toks.i32        : [n_eagle, 4] int32 = (tok_t, tok_{t+1}, tok_{t+2}, doc_id)
//                           tok_{t+1} = -1 if last token in doc; tok_{t+2} = -1 if last 2 tokens in doc
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

static inline uint16_t f32_to_f16(float f) {
    return (uint16_t) ggml_fp32_to_fp16(f);
}

struct DumpState {
    int n_layer = 24;
    int n_embd = 2304;
    int n_fmrd_target = 8192;
    int n_eagle_target = 65536;
    int stride_fmrd = 8;

    uint64_t total_tokens = 0;
    int n_fmrd_written = 0;
    int n_eagle_written = 0;
    int cur_doc_id = 0;

    // Current batch token metadata (filled before llama_decode)
    std::vector<int32_t> batch_tok0;
    std::vector<int32_t> batch_tok1;
    std::vector<int32_t> batch_tok2;
    // Which local token indices within the current batch are selected for FMRD (same across all 24 layers!)
    std::vector<int> batch_fmrd_idx;

    std::vector<FILE *> f_x;
    FILE * f_eh = nullptr;
    FILE * f_et = nullptr;

    std::vector<float> tmp_f32;
    std::vector<uint16_t> tmp_f16;

    bool open_files(const std::string & dir, int nl, int ne) {
        n_layer = nl;
        n_embd = ne;
        mkdir(dir.c_str(), 0755);
        f_x.resize(nl, nullptr);
        for (int l = 0; l < nl; ++l) {
            char buf[512];
            snprintf(buf, sizeof(buf), "%s/x_L%02d.f16", dir.c_str(), l);
            f_x[l] = fopen(buf, "wb");
            if (!f_x[l]) { fprintf(stderr, "[dump] failed to open %s\n", buf); return false; }
        }
        std::string peh = dir + "/eagle_h.f16";
        std::string pet = dir + "/eagle_toks.i32";
        f_eh = fopen(peh.c_str(), "wb");
        f_et = fopen(pet.c_str(), "wb");
        if (!f_eh || !f_et) { fprintf(stderr, "[dump] failed to open eagle files\n"); return false; }
        return true;
    }

    void close_files() {
        for (FILE * f : f_x) if (f) fclose(f);
        if (f_eh) fclose(f_eh);
        if (f_et) fclose(f_et);
    }
};

static bool on_eval(ggml_tensor * t, bool ask, void * ud) {
    const bool is_rtr = strncmp(t->name, "ffn_moe_logits-", 15) == 0;
    const bool is_rn  = strcmp(t->name, "result_norm") == 0;
    const bool want = is_rtr || is_rn;
    if (ask || !want) return want;

    DumpState * st = (DumpState *) ud;

    if (is_rtr) {
        if (st->batch_fmrd_idx.empty()) return true;
        int il = atoi(t->name + 15);
        if (il < 0 || il >= st->n_layer) return true;
        // t = MUL_MAT(gate_inp, cur) -> t->src[1] is cur [n_embd, n_tokens] F32!
        ggml_tensor * inp = t->src[1];
        if (!inp || inp->ne[0] != st->n_embd || inp->type != GGML_TYPE_F32) {
            fprintf(stderr, "[dump] unexpected router input tensor at L%d (ne0=%lld type=%d)\n",
                    il, inp ? (long long) inp->ne[0] : -1LL, inp ? (int) inp->type : -1);
            return true;
        }
        const int n_tok = (int) inp->ne[1];
        const size_t n_el = (size_t) st->n_embd * n_tok;
        st->tmp_f32.resize(n_el);
        ggml_backend_tensor_get(inp, st->tmp_f32.data(), 0, n_el * sizeof(float));

        const int n_sel = (int) st->batch_fmrd_idx.size();
        st->tmp_f16.resize((size_t) n_sel * st->n_embd);
        for (int s = 0; s < n_sel; ++s) {
            const int ti = st->batch_fmrd_idx[s];
            const float * src = st->tmp_f32.data() + (size_t) ti * st->n_embd;
            uint16_t * dst = st->tmp_f16.data() + (size_t) s * st->n_embd;
            for (int d = 0; d < st->n_embd; ++d) dst[d] = f32_to_f16(src[d]);
        }
        fwrite(st->tmp_f16.data(), sizeof(uint16_t), st->tmp_f16.size(), st->f_x[il]);
        return true;
    }

    if (is_rn) {
        if (st->n_eagle_written >= st->n_eagle_target) return true;
        // When inp_out_ids selects only the last token, result_norm is GET_ROWS(norm_all, inp_out_ids),
        // so t->src[0] holds the FULL [n_embd, n_tokens] output of build_norm(cur, output_norm)!
        ggml_tensor * src_t = (t->op == GGML_OP_GET_ROWS && t->src[0] != nullptr) ? t->src[0] : t;
        if (src_t->ne[0] != st->n_embd || src_t->type != GGML_TYPE_F32) {
            fprintf(stderr, "[dump] unexpected result_norm tensor (op=%d ne0=%lld ne1=%lld type=%d)\n",
                    (int) t->op, (long long) src_t->ne[0], (long long) src_t->ne[1], (int) src_t->type);
            return true;
        }
        const int n_tok = (int) src_t->ne[1];
        if (n_tok != (int) st->batch_tok0.size()) {
            fprintf(stderr, "[dump] result_norm n_tok=%d != batch=%zu\n", n_tok, st->batch_tok0.size());
            return true;
        }
        const int take = std::min(n_tok, st->n_eagle_target - st->n_eagle_written);
        if (take <= 0) return true;

        const size_t n_el = (size_t) st->n_embd * n_tok;
        st->tmp_f32.resize(n_el);
        ggml_backend_tensor_get(src_t, st->tmp_f32.data(), 0, n_el * sizeof(float));

        st->tmp_f16.resize((size_t) take * st->n_embd);
        for (int i = 0; i < take; ++i) {
            const float * src = st->tmp_f32.data() + (size_t) i * st->n_embd;
            uint16_t * dst = st->tmp_f16.data() + (size_t) i * st->n_embd;
            for (int d = 0; d < st->n_embd; ++d) dst[d] = f32_to_f16(src[d]);
        }
        fwrite(st->tmp_f16.data(), sizeof(uint16_t), st->tmp_f16.size(), st->f_eh);

        std::vector<int32_t> meta((size_t) take * 4);
        for (int i = 0; i < take; ++i) {
            meta[i * 4 + 0] = st->batch_tok0[i];
            meta[i * 4 + 1] = st->batch_tok1[i];
            meta[i * 4 + 2] = st->batch_tok2[i];
            meta[i * 4 + 3] = st->cur_doc_id;
        }
        fwrite(meta.data(), sizeof(int32_t), meta.size(), st->f_et);
        st->n_eagle_written += take;
        return true;
    }

    return true;
}

static std::string extract_text(const std::string & line) {
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
    std::string model_path, calib_path, out_dir = "/tmp/fmrd_data";
    int passes = 1, n_used = 4, threads = 6, n_ub = 128, n_ctx = 2048;
    int n_fmrd = 8192, n_eagle = 65536, stride_fmrd = 8;

    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if      (a == "-m") model_path = v;
        else if (a == "-f") calib_path = v;
        else if (a == "-o") out_dir = v;
        else if (a == "-passes") passes = std::stoi(v);
        else if (a == "-k") n_used = std::stoi(v);
        else if (a == "-t") threads = std::stoi(v);
        else if (a == "-ub") n_ub = std::stoi(v);
        else if (a == "-c") n_ctx = std::stoi(v);
        else if (a == "-n-fmrd") n_fmrd = std::stoi(v);
        else if (a == "-n-eagle") n_eagle = std::stoi(v);
        else if (a == "-stride-fmrd") stride_fmrd = std::stoi(v);
    }
    if (model_path.empty() || calib_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -f calib.jsonl -o out_dir [-n-fmrd 8192] [-n-eagle 65536] [-stride-fmrd 8]\n", argv[0]);
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
    if (!model) { fprintf(stderr, "[dump] model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_layer = llama_model_n_layer(model);
    const int n_embd  = llama_model_n_embd(model);

    DumpState st;
    st.n_fmrd_target  = n_fmrd;
    st.n_eagle_target = n_eagle;
    st.stride_fmrd    = std::max(1, stride_fmrd);
    if (!st.open_files(out_dir, n_layer, n_embd)) return 1;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ub; cp.n_ubatch = n_ub;
    cp.op_offload = false; cp.offload_kqv = false;
    cp.n_threads = threads; cp.n_threads_batch = threads;
    cp.cb_eval = on_eval; cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "[dump] ctx failed\n"); return 1; }

    auto t0 = std::chrono::steady_clock::now();
    std::ifstream in(calib_path);
    std::string line;
    int docs = 0;
    uint64_t global_tok_idx = 0;

    while (std::getline(in, line) &&
           (st.n_fmrd_written < st.n_fmrd_target || st.n_eagle_written < st.n_eagle_target)) {
        std::string text = extract_text(line);
        if (text.size() < 16) continue;
        std::vector<llama_token> toks(text.size() + 16);
        int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), false, true);
        if (nt <= 4) continue;
        if (nt > n_ctx - 8) nt = n_ctx - 8;
        toks.resize(nt);

        llama_memory_clear(llama_get_memory(ctx), true);
        st.cur_doc_id = docs;

        for (int i = 0; i < nt; i += n_ub) {
            if (st.n_fmrd_written >= st.n_fmrd_target && st.n_eagle_written >= st.n_eagle_target) break;
            int n = std::min(n_ub, nt - i);

            st.batch_tok0.resize(n);
            st.batch_tok1.resize(n);
            st.batch_tok2.resize(n);
            st.batch_fmrd_idx.clear();
            for (int j = 0; j < n; ++j) {
                int pos = i + j;
                st.batch_tok0[j] = toks[pos];
                st.batch_tok1[j] = (pos + 1 < nt) ? toks[pos + 1] : -1;
                st.batch_tok2[j] = (pos + 2 < nt) ? toks[pos + 2] : -1;
                if (st.n_fmrd_written + (int) st.batch_fmrd_idx.size() < st.n_fmrd_target) {
                    if ((global_tok_idx % (uint64_t) st.stride_fmrd) == 0) {
                        st.batch_fmrd_idx.push_back(j);
                    }
                }
                global_tok_idx++;
            }

            if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, n)) != 0) {
                fprintf(stderr, "[dump] decode failed doc %d\n", docs);
                st.close_files();
                return 2;
            }
            st.n_fmrd_written += (int) st.batch_fmrd_idx.size();
            st.total_tokens += (uint64_t) n;
        }
        docs++;
        if (docs % 10 == 0 || (st.n_fmrd_written >= st.n_fmrd_target && st.n_eagle_written >= st.n_eagle_target)) {
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "[dump] docs=%d total_tok=%llu fmrd=%d/%d eagle=%d/%d (%.1f tok/s, %.1fs)\n",
                    docs, (unsigned long long) st.total_tokens,
                    st.n_fmrd_written, st.n_fmrd_target,
                    st.n_eagle_written, st.n_eagle_target,
                    st.total_tokens / std::max(dt, 1e-6), dt);
        }
    }

    st.close_files();

    std::string meta_path = out_dir + "/meta.json";
    FILE * fm = fopen(meta_path.c_str(), "w");
    fprintf(fm, "{\n  \"model\": \"%s\",\n  \"calib\": \"%s\",\n  \"passes\": %d,\n  \"top_k\": %d,\n"
                "  \"n_layer\": %d,\n  \"n_embd\": %d,\n  \"docs\": %d,\n  \"total_tokens\": %llu,\n"
                "  \"n_fmrd\": %d,\n  \"n_eagle\": %d,\n  \"stride_fmrd\": %d\n}\n",
            model_path.c_str(), calib_path.c_str(), passes, n_used,
            n_layer, n_embd, docs, (unsigned long long) st.total_tokens,
            st.n_fmrd_written, st.n_eagle_written, st.stride_fmrd);
    fclose(fm);
    fprintf(stderr, "[dump] DONE -> %s (fmrd=%d, eagle=%d)\n", out_dir.c_str(), st.n_fmrd_written, st.n_eagle_written);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
