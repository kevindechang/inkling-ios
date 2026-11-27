#pragma once
// Self-contained ARM64 NEON Q4_0 Tiny 1/2-Layer EAGLE-3 Speculative Drafter for Inkling 2.
// Supports both EAGL v1 (1-layer) and EAGL v2 (N-layer + optional residual h_skip + FP16 KV cache).
// Loads binary exported by train/export_eagle_bin.py (~25-37 MB total including K=8K/16K Q4_0 head)
// and reads target token_embd.weight rows lazily via pread + libggml-base to_float (0 MB RAM!).
//
// Zero modifications to target model weights; verified by exact target B=2/3 decode!
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#endif

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

struct EagleBlockQ4_0 {
    uint16_t d;
    uint8_t  qs[16];
};
static_assert(sizeof(EagleBlockQ4_0) == 18, "block_q4_0 size mismatch");

struct EagleBlockQ8_0 {
    float  d;
    int8_t qs[32];
};

static inline float eagle_fp16_to_f32(uint16_t h) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    __fp16 f;
    memcpy(&f, &h, 2);
    return (float) f;
#else
    return ggml_fp16_to_fp32(h);
#endif
}

static inline uint16_t eagle_f32_to_fp16(float x) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    __fp16 f = (__fp16) x;
    uint16_t h;
    memcpy(&h, &f, 2);
    return h;
#else
    return (uint16_t) ggml_fp32_to_fp16(x);
#endif
}

static inline void eagle_quantize_q8_0(const float * x, EagleBlockQ8_0 * y, int n) {
    const int nb = n / 32;
    for (int b = 0; b < nb; ++b) {
        const float * xp = x + b * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) {
            float a = std::fabs(xp[j]);
            if (a > amax) amax = a;
        }
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 127.0f / amax : 0.0f;
        y[b].d = d;
        for (int j = 0; j < 32; ++j) {
            int v = (int) std::lroundf(xp[j] * id);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            y[b].qs[j] = (int8_t) v;
        }
    }
}

static inline void eagle_gemv_q4_0_range(
        const EagleBlockQ4_0 * W, const EagleBlockQ8_0 * xq, float * out,
        int r0, int r1, int ncols) {
    const int nb = ncols / 32;
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    const int8x16_t  s8  = vdupq_n_s8(-8);
    int r = r0;
    for (; r + 3 < r1; r += 4) {
        const EagleBlockQ4_0 * w0 = W + (size_t) (r + 0) * nb;
        const EagleBlockQ4_0 * w1 = W + (size_t) (r + 1) * nb;
        const EagleBlockQ4_0 * w2 = W + (size_t) (r + 2) * nb;
        const EagleBlockQ4_0 * w3 = W + (size_t) (r + 3) * nb;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; ++b) {
            const int8x16_t a0 = vld1q_s8(xq[b].qs +  0);
            const int8x16_t a1 = vld1q_s8(xq[b].qs + 16);
            const float ad = xq[b].d;

            auto dot_vec = [&](const EagleBlockQ4_0 & wb) -> int32x4_t {
                const uint8x16_t q = vld1q_u8(wb.qs);
                const int8x16_t lo = vaddq_s8(vreinterpretq_s8_u8(vandq_u8(q, m4b)), s8);
                const int8x16_t hi = vaddq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), s8);
                int32x4_t s = vdotq_s32(vdupq_n_s32(0), lo, a0);
                return vdotq_s32(s, hi, a1);
            };
            const int32x4_t d0 = dot_vec(w0[b]);
            const int32x4_t d1 = dot_vec(w1[b]);
            const int32x4_t d2 = dot_vec(w2[b]);
            const int32x4_t d3 = dot_vec(w3[b]);
            const int32x4_t p01 = vpaddq_s32(d0, d1);
            const int32x4_t p23 = vpaddq_s32(d2, d3);
            const int32x4_t sv  = vpaddq_s32(p01, p23);
            const uint16_t dh[4] = { w0[b].d, w1[b].d, w2[b].d, w3[b].d };
            const float32x4_t wd = vcvt_f32_f16(vld1_f16((const __fp16 *) dh));
            acc = vfmaq_f32(acc, vcvtq_f32_s32(sv), vmulq_n_f32(wd, ad));
        }
        vst1q_f32(out + r, acc);
    }
    for (; r < r1; ++r) {
        const EagleBlockQ4_0 * wr = W + (size_t) r * nb;
        float sum = 0.0f;
        for (int b = 0; b < nb; ++b) {
            const uint8x16_t q = vld1q_u8(wr[b].qs);
            const int8x16_t lo = vaddq_s8(vreinterpretq_s8_u8(vandq_u8(q, m4b)), s8);
            const int8x16_t hi = vaddq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), s8);
            int32x4_t s = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(xq[b].qs + 0));
            s = vdotq_s32(s, hi, vld1q_s8(xq[b].qs + 16));
            sum += (float) vaddvq_s32(s) * (eagle_fp16_to_f32(wr[b].d) * xq[b].d);
        }
        out[r] = sum;
    }
#else
    for (int r = r0; r < r1; ++r) {
        const EagleBlockQ4_0 * wr = W + (size_t) r * nb;
        float sum = 0.0f;
        for (int b = 0; b < nb; ++b) {
            int32_t is = 0;
            for (int j = 0; j < 16; ++j) {
                is += ((int)(wr[b].qs[j] & 0xf) - 8) * (int) xq[b].qs[j];
                is += ((int)(wr[b].qs[j] >> 4)  - 8) * (int) xq[b].qs[j + 16];
            }
            sum += (float) is * (eagle_fp16_to_f32(wr[b].d) * xq[b].d);
        }
        out[r] = sum;
    }
#endif
}

struct EagleLayer {
    std::vector<float> attn_norm_w, ffn_norm_w;
    std::vector<EagleBlockQ4_0> wqkv, wo, w_gate_up, w_down;
    std::vector<uint16_t> k_cache, v_cache; // FP16 [max_ctx, d_model]
};

struct TinyEagleDrafter {
    int h_dim = 2304;
    int d_model = 1024;
    int n_heads = 8;
    int head_dim = 128;
    int ff_dim = 2560;
    int n_layers = 1;
    bool has_h_skip = false;
    int k_sl = 8192;
    int n_vocab = 201024;
    int max_ctx = 4096;
    int n_threads = 2;

    std::vector<int32_t> shortlist_ids;
    std::vector<float> tok_norm_w;
    std::vector<float> norm_h_w, norm_e_w, out_norm_w, h_skip_w;

    std::vector<EagleBlockQ4_0> fc_in, fc_out, head_sl;
    std::vector<EagleLayer> layers;

    // Lazy token_embd lookup directly from target GGUF (0 MB RAM!)
    int emb_fd = -1;
    ggml_type emb_type = GGML_TYPE_F32;
    size_t emb_base_off = 0;
    size_t emb_row_bytes = 0;
    ggml_to_float_t emb_to_float = nullptr;
    std::vector<uint8_t> emb_raw_buf;

    int kv_len = 0;

    // Scratch buffers
    std::vector<float> inv_freq;
    std::vector<float> buf_he, buf_x, buf_xn, buf_qkv, buf_attn, buf_gu, buf_ff, buf_logits;
    std::vector<EagleBlockQ8_0> buf_xq;

    ~TinyEagleDrafter() {
        if (emb_fd >= 0) close(emb_fd);
    }

    bool load(const std::string & bin_path, const std::string & target_gguf, int ctx_len = 4096, int threads = 2) {
        max_ctx = ctx_len;
        n_threads = std::max(1, threads);

        FILE * f = fopen(bin_path.c_str(), "rb");
        if (!f) { fprintf(stderr, "[eagle] failed to open %s\n", bin_path.c_str()); return false; }
        struct { char magic[4]; uint32_t ver, h, d, nh, ff, k, nv, rsv[8]; } hdr;
        if (fread(&hdr, 1, sizeof(hdr), f) != sizeof(hdr) || memcmp(hdr.magic, "EAGL", 4) != 0 || (hdr.ver != 1 && hdr.ver != 2)) {
            fprintf(stderr, "[eagle] invalid header in %s\n", bin_path.c_str());
            fclose(f);
            return false;
        }
        h_dim = (int) hdr.h; d_model = (int) hdr.d; n_heads = (int) hdr.nh;
        head_dim = d_model / n_heads; ff_dim = (int) hdr.ff; k_sl = (int) hdr.k; n_vocab = (int) hdr.nv;
        n_layers = (hdr.ver == 1 || hdr.rsv[0] == 0) ? 1 : (int) hdr.rsv[0];
        has_h_skip = (hdr.ver >= 2 && hdr.rsv[1] != 0);

        auto rd = [&](void * dst, size_t nbytes) -> bool { return fread(dst, 1, nbytes, f) == nbytes; };
        auto rd_q4 = [&](std::vector<EagleBlockQ4_0> & dst, int nrows, int ncols) -> bool {
            dst.resize((size_t) nrows * (ncols / 32));
            return rd(dst.data(), dst.size() * sizeof(EagleBlockQ4_0));
        };

        shortlist_ids.resize(k_sl);
        tok_norm_w.resize(h_dim);
        norm_h_w.resize(h_dim); norm_e_w.resize(h_dim);
        out_norm_w.resize(d_model);
        layers.resize(n_layers);
        for (int l = 0; l < n_layers; ++l) {
            layers[l].attn_norm_w.resize(d_model);
            layers[l].ffn_norm_w.resize(d_model);
        }

        bool ok = true;
        ok &= rd(shortlist_ids.data(), shortlist_ids.size() * sizeof(int32_t));
        ok &= rd(tok_norm_w.data(), tok_norm_w.size() * sizeof(float));
        ok &= rd(norm_h_w.data(), norm_h_w.size() * sizeof(float));
        ok &= rd(norm_e_w.data(), norm_e_w.size() * sizeof(float));
        if (hdr.ver == 1) {
            ok &= rd(layers[0].attn_norm_w.data(), d_model * sizeof(float));
            ok &= rd(layers[0].ffn_norm_w.data(), d_model * sizeof(float));
            ok &= rd(out_norm_w.data(), d_model * sizeof(float));
            ok &= rd_q4(fc_in,               d_model,     2 * h_dim);
            ok &= rd_q4(layers[0].wqkv,      3 * d_model, d_model);
            ok &= rd_q4(layers[0].wo,        d_model,     d_model);
            ok &= rd_q4(layers[0].w_gate_up, 2 * ff_dim,  d_model);
            ok &= rd_q4(layers[0].w_down,    d_model,     ff_dim);
            ok &= rd_q4(fc_out,              h_dim,       d_model);
            ok &= rd_q4(head_sl,             k_sl,        h_dim);
        } else {
            ok &= rd(out_norm_w.data(), d_model * sizeof(float));
            if (has_h_skip) {
                h_skip_w.resize(h_dim);
                ok &= rd(h_skip_w.data(), h_dim * sizeof(float));
            }
            for (int l = 0; l < n_layers; ++l) {
                ok &= rd(layers[l].attn_norm_w.data(), d_model * sizeof(float));
                ok &= rd(layers[l].ffn_norm_w.data(), d_model * sizeof(float));
            }
            ok &= rd_q4(fc_in, d_model, 2 * h_dim);
            for (int l = 0; l < n_layers; ++l) {
                ok &= rd_q4(layers[l].wqkv,      3 * d_model, d_model);
                ok &= rd_q4(layers[l].wo,        d_model,     d_model);
                ok &= rd_q4(layers[l].w_gate_up, 2 * ff_dim,  d_model);
                ok &= rd_q4(layers[l].w_down,    d_model,     ff_dim);
            }
            ok &= rd_q4(fc_out,  h_dim, d_model);
            ok &= rd_q4(head_sl, k_sl,  h_dim);
        }
        fclose(f);
        if (!ok) { fprintf(stderr, "[eagle] truncated file %s\n", bin_path.c_str()); return false; }

        // Open target GGUF for lazy token_embd.weight row reads
        gguf_init_params gp = { true, nullptr };
        gguf_context * gctx = gguf_init_from_file(target_gguf.c_str(), gp);
        if (!gctx) { fprintf(stderr, "[eagle] failed to parse GGUF %s\n", target_gguf.c_str()); return false; }
        int64_t tid = gguf_find_tensor(gctx, "token_embd.weight");
        if (tid < 0) { gguf_free(gctx); fprintf(stderr, "[eagle] token_embd.weight not found\n"); return false; }
        emb_type      = gguf_get_tensor_type(gctx, tid);
        emb_base_off  = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, tid);
        emb_row_bytes = ggml_row_size(emb_type, h_dim);
        emb_to_float  = ggml_get_type_traits(emb_type)->to_float;
        gguf_free(gctx);

        emb_fd = open(target_gguf.c_str(), O_RDONLY);
        if (emb_fd < 0) { fprintf(stderr, "[eagle] failed to open %s for pread\n", target_gguf.c_str()); return false; }
        emb_raw_buf.resize(emb_row_bytes);

        for (int l = 0; l < n_layers; ++l) {
            layers[l].k_cache.assign((size_t) max_ctx * d_model, 0);
            layers[l].v_cache.assign((size_t) max_ctx * d_model, 0);
        }
        inv_freq.resize(head_dim / 2);
        for (int i = 0; i < head_dim / 2; ++i) {
            inv_freq[i] = 1.0f / std::pow(10000.0f, (float) i / (float) (head_dim / 2));
        }
        buf_he.resize(2 * h_dim);
        buf_x.resize(d_model);
        buf_xn.resize(d_model);
        buf_qkv.resize(3 * d_model);
        buf_attn.resize(d_model);
        buf_gu.resize(2 * ff_dim);
        buf_ff.resize(ff_dim);
        buf_logits.resize(k_sl);
        buf_xq.resize(std::max({2 * h_dim, ff_dim, d_model}) / 32);
        kv_len = 0;

        double kv_mb = (double) n_layers * max_ctx * d_model * 4.0 / (1024.0 * 1024.0);
        fprintf(stderr, "[eagle] loaded v%u %s (L=%d d=%d ff=%d skip=%d K=%d ctx=%d kv=%.1fMB t=%d emb=%s)\n",
                hdr.ver, bin_path.c_str(), n_layers, d_model, ff_dim, (int) has_h_skip, k_sl, max_ctx, kv_mb, n_threads, ggml_type_name(emb_type));
        return true;
    }

    void reset() { kv_len = 0; }
    void truncate(int new_len) { kv_len = std::max(0, std::min(new_len, kv_len)); }

    void rms_norm(const float * x, const float * w, float * y, int n) const {
        float s2 = 0.0f;
        for (int i = 0; i < n; ++i) s2 += x[i] * x[i];
        const float inv = 1.0f / std::sqrt(s2 / (float) n + 1e-6f);
        for (int i = 0; i < n; ++i) y[i] = x[i] * inv * w[i];
    }

    void get_normed_emb(int32_t tok, float * out) {
        if (tok < 0 || tok >= n_vocab) { memset(out, 0, (size_t) h_dim * sizeof(float)); return; }
        ssize_t got = pread(emb_fd, emb_raw_buf.data(), emb_row_bytes, (off_t) (emb_base_off + (size_t) tok * emb_row_bytes));
        if (got != (ssize_t) emb_row_bytes) { memset(out, 0, (size_t) h_dim * sizeof(float)); return; }
        if (emb_type == GGML_TYPE_F32) {
            memcpy(out, emb_raw_buf.data(), (size_t) h_dim * sizeof(float));
        } else {
            emb_to_float(emb_raw_buf.data(), out, h_dim);
        }
        rms_norm(out, tok_norm_w.data(), out, h_dim);
    }

    void matvec(const std::vector<EagleBlockQ4_0> & W, const float * x, float * out, int nrows, int ncols) {
        eagle_quantize_q8_0(x, buf_xq.data(), ncols);
        const EagleBlockQ4_0 * wp = W.data();
        const EagleBlockQ8_0 * xp = buf_xq.data();
#if defined(__APPLE__)
        if (n_threads > 1 && nrows >= 1024) {
            const int nt = n_threads;
            const int chunk = ((nrows / nt) + 3) & ~3;
            dispatch_apply((size_t) nt, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t) {
                int r0 = (int) t * chunk;
                int r1 = (t + 1 == (size_t) nt) ? nrows : std::min(nrows, r0 + chunk);
                if (r0 < r1) eagle_gemv_q4_0_range(wp, xp, out, r0, r1, ncols);
            });
            return;
        }
#endif
        eagle_gemv_q4_0_range(wp, xp, out, 0, nrows, ncols);
    }

    void apply_rope_single(float * v, int pos) const {
        const int half = head_dim / 2;
        float cs[128], sn[128];
        for (int i = 0; i < half; ++i) {
            const float ang = (float) pos * inv_freq[i];
            cs[i] = std::cos(ang);
            sn[i] = std::sin(ang);
        }
        for (int h = 0; h < n_heads; ++h) {
            float * vh = v + h * head_dim;
            for (int i = 0; i < half; ++i) {
                const float x1 = vh[i], x2 = vh[i + half];
                vh[i]        = x1 * cs[i] - x2 * sn[i];
                vh[i + half] = x2 * cs[i] + x1 * sn[i];
            }
        }
    }

    void run_layer_attn(EagleLayer & L, int pos) {
        rms_norm(buf_x.data(), L.attn_norm_w.data(), buf_xn.data(), d_model);
        matvec(L.wqkv, buf_xn.data(), buf_qkv.data(), 3 * d_model, d_model);
        float * q = buf_qkv.data();
        float * k = buf_qkv.data() + d_model;
        float * v = buf_qkv.data() + 2 * d_model;
        apply_rope_single(q, pos);
        apply_rope_single(k, pos);

        uint16_t * kc_row = L.k_cache.data() + (size_t) pos * d_model;
        uint16_t * vc_row = L.v_cache.data() + (size_t) pos * d_model;
        for (int i = 0; i < d_model; ++i) {
            kc_row[i] = eagle_f32_to_fp16(k[i]);
            vc_row[i] = eagle_f32_to_fp16(v[i]);
        }

        const float scale = 1.0f / std::sqrt((float) head_dim);
        for (int h = 0; h < n_heads; ++h) {
            const float * qh = q + h * head_dim;
            float * oh = buf_attn.data() + h * head_dim;
            memset(oh, 0, (size_t) head_dim * sizeof(float));
            float M = -1e30f, S = 0.0f;
            for (int t = 0; t <= pos; ++t) {
                const uint16_t * kt = L.k_cache.data() + (size_t) t * d_model + h * head_dim;
                const uint16_t * vt = L.v_cache.data() + (size_t) t * d_model + h * head_dim;
#if defined(__aarch64__) && defined(__ARM_NEON)
                float32x4_t vd0 = vdupq_n_f32(0.0f), vd1 = vdupq_n_f32(0.0f);
                for (int d = 0; d < head_dim; d += 8) {
                    vd0 = vfmaq_f32(vd0, vld1q_f32(qh + d + 0), vcvt_f32_f16(vld1_f16((const __fp16 *)(kt + d + 0))));
                    vd1 = vfmaq_f32(vd1, vld1q_f32(qh + d + 4), vcvt_f32_f16(vld1_f16((const __fp16 *)(kt + d + 4))));
                }
                float s = vaddvq_f32(vaddq_f32(vd0, vd1)) * scale;
                if (s > M) {
                    const float ms = (M < -1e20f) ? 0.0f : std::exp(M - s);
                    M = s;
                    if (ms > 0.0f) {
                        const float32x4_t vms = vdupq_n_f32(ms);
                        for (int d = 0; d < head_dim; d += 4) {
                            vst1q_f32(oh + d, vaddq_f32(vmulq_f32(vld1q_f32(oh + d), vms), vcvt_f32_f16(vld1_f16((const __fp16 *)(vt + d)))));
                        }
                    } else {
                        for (int d = 0; d < head_dim; d += 4) {
                            vst1q_f32(oh + d, vcvt_f32_f16(vld1_f16((const __fp16 *)(vt + d))));
                        }
                    }
                    S = S * ms + 1.0f;
                } else {
                    const float w = std::exp(s - M);
                    if (w > 1e-7f) {
                        const float32x4_t vw = vdupq_n_f32(w);
                        for (int d = 0; d < head_dim; d += 4) {
                            vst1q_f32(oh + d, vfmaq_f32(vld1q_f32(oh + d), vcvt_f32_f16(vld1_f16((const __fp16 *)(vt + d))), vw));
                        }
                    }
                    S += w;
                }
#else
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += qh[d] * eagle_fp16_to_f32(kt[d]);
                float s = dot * scale;
                if (s > M) {
                    const float ms = (M < -1e20f) ? 0.0f : std::exp(M - s);
                    M = s;
                    for (int d = 0; d < head_dim; ++d) oh[d] = oh[d] * ms + eagle_fp16_to_f32(vt[d]);
                    S = S * ms + 1.0f;
                } else {
                    const float w = std::exp(s - M);
                    for (int d = 0; d < head_dim; ++d) oh[d] += w * eagle_fp16_to_f32(vt[d]);
                    S += w;
                }
#endif
            }
            const float invS = S > 0.0f ? 1.0f / S : 0.0f;
            for (int d = 0; d < head_dim; ++d) oh[d] *= invS;
        }
        matvec(L.wo, buf_attn.data(), buf_xn.data(), d_model, d_model);
        for (int i = 0; i < d_model; ++i) buf_x[i] += buf_xn[i];

        // SwiGLU FFN
        rms_norm(buf_x.data(), L.ffn_norm_w.data(), buf_xn.data(), d_model);
        matvec(L.w_gate_up, buf_xn.data(), buf_gu.data(), 2 * ff_dim, d_model);
        for (int i = 0; i < ff_dim; ++i) {
            float g = buf_gu[i];
            buf_ff[i] = (g / (1.0f + std::exp(-g))) * buf_gu[ff_dim + i];
        }
        matvec(L.w_down, buf_ff.data(), buf_xn.data(), d_model, ff_dim);
        for (int i = 0; i < d_model; ++i) buf_x[i] += buf_xn[i];
    }

    // Run 1 autoregressive drafter step at position `pos = kv_len`, advancing kv_len by 1!
    // Writes predicted h_hat_{t+1} [h_dim] into `h_out`.
    void step(const float * h_in, int32_t next_tok, float * h_out) {
        if (kv_len >= max_ctx) {
            // Sliding window fallback: drop oldest half cleanly if ctx exceeds max_ctx
            const int keep = max_ctx / 2;
            const int drop = kv_len - keep;
            for (int l = 0; l < n_layers; ++l) {
                memmove(layers[l].k_cache.data(), layers[l].k_cache.data() + (size_t) drop * d_model, (size_t) keep * d_model * sizeof(uint16_t));
                memmove(layers[l].v_cache.data(), layers[l].v_cache.data() + (size_t) drop * d_model, (size_t) keep * d_model * sizeof(uint16_t));
            }
            kv_len = keep;
        }
        const int pos = kv_len;

        // 1. Concat [norm_h(h_in), norm_e(emb(next_tok))] -> fc_in -> buf_x [d_model]
        rms_norm(h_in, norm_h_w.data(), buf_he.data(), h_dim);
        get_normed_emb(next_tok, buf_he.data() + h_dim);
        rms_norm(buf_he.data() + h_dim, norm_e_w.data(), buf_he.data() + h_dim, h_dim);
        matvec(fc_in, buf_he.data(), buf_x.data(), d_model, 2 * h_dim);

        // 2. N-layer causal transformer
        for (int l = 0; l < n_layers; ++l) {
            run_layer_attn(layers[l], pos);
        }
        kv_len = pos + 1;

        // 3. Project back to target result_norm space [h_dim] (+ optional learned residual skip!)
        rms_norm(buf_x.data(), out_norm_w.data(), buf_xn.data(), d_model);
        matvec(fc_out, buf_xn.data(), h_out, h_dim, d_model);
        if (has_h_skip) {
            for (int i = 0; i < h_dim; ++i) {
                h_out[i] += h_in[i] * h_skip_w[i];
            }
        }
    }

    int32_t sample_shortlist(const float * h_hat) {
        matvec(head_sl, h_hat, buf_logits.data(), k_sl, h_dim);
        int best_i = 0;
        float best_v = buf_logits[0];
        for (int i = 1; i < k_sl; ++i) {
            if (buf_logits[i] > best_v) { best_v = buf_logits[i]; best_i = i; }
        }
        return shortlist_ids[best_i];
    }
};
