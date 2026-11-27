// Reversible expert-slice repack. Slice size does not change, so --pack/--unpack
// with one path rewrites that file in place and records each finished slice in
// <file>.prerepack-journal. --unpack reverses only journaled slices and deletes
// the journal. A crash mid-pack can be finished by --unpack. --verify-self is
// read-only on the real model and only writes tiny files under /tmp, which it deletes.
// Copying to a new file is still available and refuses to overwrite.
//
//   prerepack_exps --verify-self gguf
//   prerepack_exps --pack   in.gguf [-o out.gguf]
//   prerepack_exps --unpack in.gguf [-o out.gguf]

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <limits.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

static const char * k_real_default = "";

enum { GGUF_MAGIC = 0x46554747u };

enum gguf_vt : uint32_t {
    VT_UINT8 = 0, VT_INT8, VT_UINT16, VT_INT16, VT_UINT32, VT_INT32, VT_FLOAT32, VT_BOOL,
    VT_STRING, VT_ARRAY, VT_UINT64, VT_INT64, VT_FLOAT64
};

struct Ten {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0;
    uint64_t offset = 0;
    size_t nbytes = 0;
};

struct Gguf {
    std::vector<char> header; // bytes [0, data_start)
    uint64_t data_start = 0;
    uint64_t file_size = 0;
    std::vector<Ten> tensors;
};

static void die(const char * msg) {
    fprintf(stderr, "prerepack: %s\n", msg);
    exit(1);
}

struct Rd {
    FILE * f = nullptr;
    explicit Rd(const char * path) {
        f = fopen(path, "rb");
        if (!f) die("open for read failed");
    }
    ~Rd() { if (f) fclose(f); }
    bool raw(void * p, size_t n) { return fread(p, 1, n, f) == n; }
    template <class T> T num() {
        T v{};
        if (!raw(&v, sizeof v)) die("truncated gguf");
        return v;
    }
    std::string str() {
        const uint64_t n = num<uint64_t>();
        if (n > (1u << 20)) die("gguf string too long");
        std::string s(n, '\0');
        if (n && !raw(s.data(), (size_t) n)) die("truncated string");
        return s;
    }
    void skip_value(uint32_t t);
};

void Rd::skip_value(uint32_t t) {
    switch (t) {
        case VT_UINT8: case VT_INT8: case VT_BOOL: num<uint8_t>(); break;
        case VT_UINT16: case VT_INT16: num<uint16_t>(); break;
        case VT_UINT32: case VT_INT32: case VT_FLOAT32: num<uint32_t>(); break;
        case VT_UINT64: case VT_INT64: case VT_FLOAT64: num<uint64_t>(); break;
        case VT_STRING: (void) str(); break;
        case VT_ARRAY: {
            const uint32_t et = num<uint32_t>();
            const uint64_t n = num<uint64_t>();
            for (uint64_t i = 0; i < n; i++) skip_value(et);
            break;
        }
        default: die("unknown gguf value type");
    }
}

static bool ends_with(const std::string & s, const char * suf) {
    const size_t n = strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

static size_t tensor_bytes(uint32_t type, const std::vector<uint64_t> & dims) {
    if (dims.empty()) return 0;
    size_t n = ggml_row_size((enum ggml_type) type, (int64_t) dims[0]);
    for (size_t i = 1; i < dims.size(); i++) n *= (size_t) dims[i];
    return n;
}

static Gguf parse_gguf(const char * path) {
    Rd rd(path);
    const uint32_t magic = rd.num<uint32_t>();
    const uint32_t version = rd.num<uint32_t>();
    if (magic != GGUF_MAGIC || version != 3) die("not a GGUF V3 file");
    const uint64_t n_tensors = rd.num<uint64_t>();
    const uint64_t n_kv = rd.num<uint64_t>();
    uint32_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        const std::string key = rd.str();
        const uint32_t vt = rd.num<uint32_t>();
        if (key == "general.alignment" && vt == VT_UINT32) {
            alignment = rd.num<uint32_t>();
        } else {
            rd.skip_value(vt);
        }
    }
    if (alignment == 0 || (alignment & (alignment - 1))) die("bad alignment");
    Gguf g;
    g.tensors.resize((size_t) n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        Ten & t = g.tensors[(size_t) i];
        t.name = rd.str();
        const uint32_t nd = rd.num<uint32_t>();
        if (nd > 8) die("too many dims");
        t.dims.resize(nd);
        for (uint32_t d = 0; d < nd; d++) t.dims[d] = rd.num<uint64_t>();
        t.type = rd.num<uint32_t>();
        t.offset = rd.num<uint64_t>();
        t.nbytes = tensor_bytes(t.type, t.dims);
    }
    const long pos = ftell(rd.f);
    if (pos < 0) die("ftell");
    const uint64_t data = ((uint64_t) pos + alignment - 1) & ~((uint64_t) alignment - 1);
    if (fseeko(rd.f, 0, SEEK_END) != 0) die("seek end");
    const off_t sz = ftello(rd.f);
    if (sz < 0) die("ftello");
    g.file_size = (uint64_t) sz;
    g.data_start = data;
    if (fseeko(rd.f, 0, SEEK_SET) != 0) die("rewind");
    g.header.resize((size_t) data);
    if (!rd.raw(g.header.data(), g.header.size())) die("read header");
    return g;
}

static bool same_file(const char * a, const char * b) {
    struct stat sa, sb;
    if (stat(a, &sa) != 0 || stat(b, &sb) != 0) return false;
    return sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino;
}

static void refuse_real_output(const char * out_path, const char * real_path) {
    if (same_file(out_path, real_path)) die("refusing to write the real model");
    char ra[PATH_MAX], rb[PATH_MAX];
    if (realpath(out_path, ra) && realpath(real_path, rb) && strcmp(ra, rb) == 0) {
        die("refusing to write the real model");
    }
}

struct ExpertView {
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    int64_t n_exp = 0;
    size_t slice = 0;
};

static ExpertView expert_view(const Ten & t) {
    ExpertView v;
    if (t.dims.size() < 3) die("expert tensor is not 3D+");
    v.ne0 = (int64_t) t.dims[0];
    v.ne1 = 1;
    for (size_t i = 1; i + 1 < t.dims.size(); i++) v.ne1 *= (int64_t) t.dims[i];
    v.n_exp = (int64_t) t.dims.back();
    v.slice = ggml_row_size((enum ggml_type) t.type, v.ne0) * (size_t) v.ne1;
    if (v.slice * (size_t) v.n_exp != t.nbytes) die("slice*experts != tensor bytes");
    return v;
}

static void transform_slice(enum ggml_type type, int64_t ne0, int64_t ne1, void * dst, const void * src, size_t n, bool unpack) {
    if (unpack) ggml_mmid_unrepack_slice(type, ne0, ne1, dst, src, n);
    else        ggml_mmid_repack_slice(type, ne0, ne1, dst, src, n);
}

static int transform_file(const char * in_path, const char * out_path, bool unpack, const char * real_path) {
    if (strcmp(in_path, out_path) == 0) die("input and output are the same path");
    if (same_file(in_path, out_path)) die("input and output are the same file");
    refuse_real_output(out_path, real_path);
    if (access(out_path, F_OK) == 0) die("output already exists; refusing to overwrite");
    Gguf g = parse_gguf(in_path);
    const int fd = open(out_path, O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd < 0) die("create output failed");
    FILE * out = fdopen(fd, "wb");
    if (!out) { close(fd); die("fdopen output"); }
    if (fwrite(g.header.data(), 1, g.header.size(), out) != g.header.size()) die("write header");
    FILE * in = fopen(in_path, "rb");
    if (!in) die("reopen input");
    std::vector<size_t> order(g.tensors.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return g.tensors[a].offset < g.tensors[b].offset;
    });
    uint64_t cursor = 0;
    auto copy_range = [&](uint64_t from, uint64_t to) {
        if (to < from) die("tensor offsets go backwards");
        if (fseeko(in, (off_t) (g.data_start + from), SEEK_SET) != 0) die("seek data");
        std::vector<char> buf(1 << 20);
        uint64_t left = to - from;
        while (left) {
            const size_t n = (size_t) std::min<uint64_t>(left, buf.size());
            if (fread(buf.data(), 1, n, in) != n) die("read data");
            if (fwrite(buf.data(), 1, n, out) != n) die("write data");
            left -= n;
        }
    };
    int n_exp_tensors = 0;
    for (size_t oi = 0; oi < order.size(); oi++) {
        const Ten & t = g.tensors[order[oi]];
        copy_range(cursor, t.offset);
        const bool exps = ends_with(t.name, "_exps.weight");
        if (!exps) {
            copy_range(t.offset, t.offset + t.nbytes);
        } else {
            const ExpertView ev = expert_view(t);
            int cols = 0;
            const size_t info = ggml_mmid_repack_info((enum ggml_type) t.type, ev.ne0, ev.ne1, 1, &cols);
            if (info != ev.slice) die("repack_info != slice size; refusing to rewrite");
            if (fseeko(in, (off_t) (g.data_start + t.offset), SEEK_SET) != 0) die("seek tensor");
            std::vector<uint8_t> src(ev.slice), dst(ev.slice);
            for (int64_t e = 0; e < ev.n_exp; e++) {
                if (fread(src.data(), 1, ev.slice, in) != ev.slice) die("read slice");
                transform_slice((enum ggml_type) t.type, ev.ne0, ev.ne1, dst.data(), src.data(), ev.slice, unpack);
                if (fwrite(dst.data(), 1, ev.slice, out) != ev.slice) die("write slice");
            }
            n_exp_tensors++;
        }
        cursor = t.offset + t.nbytes;
    }
    if (g.file_size < g.data_start + cursor) die("file shorter than tensors");
    copy_range(cursor, g.file_size - g.data_start);
    fclose(in);
    if (fclose(out) != 0) die("close output");
    fprintf(stderr, "prerepack: %s %s -> %s  expert_tensors=%d\n",
            unpack ? "unpack" : "pack", in_path, out_path, n_exp_tensors);
    return n_exp_tensors;
}

static uint32_t xorshift(uint32_t & s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}

static int roundtrip_buf(enum ggml_type type, int64_t ne0, int64_t ne1, uint32_t seed, const char * tag) {
    int cols = 0;
    const size_t info = ggml_mmid_repack_info(type, ne0, ne1, 1, &cols);
    const size_t raw = ggml_row_size(type, ne0) * (size_t) ne1;
    if (info == 0 || info != raw) {
        fprintf(stderr, "FAIL %s info=%zu raw=%zu\n", tag, info, raw);
        return 1;
    }
    std::vector<uint8_t> src(raw), packed(raw), back(raw, 0x5a);
    uint32_t st = seed;
    for (size_t i = 0; i < raw; i++) src[i] = (uint8_t) xorshift(st);
    ggml_mmid_repack_slice(type, ne0, ne1, packed.data(), src.data(), raw);
    ggml_mmid_unrepack_slice(type, ne0, ne1, back.data(), packed.data(), raw);
    const bool same = memcmp(src.data(), back.data(), raw) == 0;
    const bool moved = memcmp(src.data(), packed.data(), raw) != 0;
    fprintf(stderr, "%s %s ne0=%lld ne1=%lld bytes=%zu cols=%d bit_exact=%d layout_changed=%d\n",
            same && moved ? "PASS" : "FAIL", tag, (long long) ne0, (long long) ne1, raw, cols, same, moved);
    return (same && moved) ? 0 : 1;
}

static void wr_u32(std::vector<uint8_t> & b, uint32_t v) {
    b.insert(b.end(), (uint8_t *) &v, (uint8_t *) &v + 4);
}
static void wr_u64(std::vector<uint8_t> & b, uint64_t v) {
    b.insert(b.end(), (uint8_t *) &v, (uint8_t *) &v + 8);
}
static void wr_str(std::vector<uint8_t> & b, const char * s) {
    wr_u64(b, strlen(s));
    b.insert(b.end(), s, s + strlen(s));
}

static std::vector<uint8_t> make_synth() {
    struct Item { const char * name; enum ggml_type type; uint64_t ne0, ne1, nexp; };
    const Item items[] = {
        {"blk.0.ffn_gate_exps.weight", GGML_TYPE_Q2_K, 256, 8, 4},
        {"blk.0.ffn_up_exps.weight",   GGML_TYPE_Q2_0, 256, 8, 4},
        {"blk.0.ffn_down_exps.weight", GGML_TYPE_Q4_0, 256, 8, 4},
        {"blk.1.ffn_gate_exps.weight", GGML_TYPE_Q4_K, 256, 8, 4},
        {"blk.1.ffn_up_exps.weight",   GGML_TYPE_IQ4_NL, 256, 8, 4},
    };
    std::vector<uint8_t> h;
    wr_u32(h, GGUF_MAGIC);
    wr_u32(h, 3);
    wr_u64(h, 5);
    wr_u64(h, 1);
    wr_str(h, "general.alignment");
    wr_u32(h, VT_UINT32);
    wr_u32(h, 32);
    uint64_t off = 0;
    std::vector<std::vector<uint8_t>> data(5);
    uint32_t seed = 0x12345678u;
    for (int i = 0; i < 5; i++) {
        const size_t slice = ggml_row_size(items[i].type, (int64_t) items[i].ne0) * (size_t) items[i].ne1;
        data[i].resize(slice * (size_t) items[i].nexp);
        for (size_t k = 0; k < data[i].size(); k++) data[i][k] = (uint8_t) xorshift(seed);
        wr_str(h, items[i].name);
        wr_u32(h, 3);
        wr_u64(h, items[i].ne0);
        wr_u64(h, items[i].ne1);
        wr_u64(h, items[i].nexp);
        wr_u32(h, (uint32_t) items[i].type);
        wr_u64(h, off);
        off += data[i].size();
    }
    while (h.size() % 32) h.push_back(0);
    for (auto & d : data) h.insert(h.end(), d.begin(), d.end());
    return h;
}

static int read_file(const char * path, std::vector<uint8_t> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) return 1;
    if (fseeko(f, 0, SEEK_END) != 0) { fclose(f); return 1; }
    const off_t n = ftello(f);
    if (n < 0 || n > 64 * 1024 * 1024) { fclose(f); return 1; }
    out.resize((size_t) n);
    rewind(f);
    const bool ok = fread(out.data(), 1, out.size(), f) == out.size();
    fclose(f);
    return ok ? 0 : 1;
}

static int verify_synth_files(const char * real_path) {
    const char * a = "/tmp/prerepack-synth-in.gguf";
    const char * b = "/tmp/prerepack-synth-packed.gguf";
    const char * c = "/tmp/prerepack-synth-unpacked.gguf";
    auto cleanup = [&]() { unlink(a); unlink(b); unlink(c); };
    cleanup();
    const std::vector<uint8_t> src = make_synth();
    FILE * f = fopen(a, "wb");
    if (!f || fwrite(src.data(), 1, src.size(), f) != src.size()) die("write synth");
    fclose(f);
    transform_file(a, b, false, real_path);
    transform_file(b, c, true, real_path);
    std::vector<uint8_t> back;
    if (read_file(c, back) || back != src) {
        fprintf(stderr, "FAIL synth file unpack(pack(x)) size %zu vs %zu\n", back.size(), src.size());
        cleanup();
        return 1;
    }
    fprintf(stderr, "PASS synth file unpack(pack(file)) bytes=%zu\n", src.size());
    cleanup();
    return 0;
}

static int verify_real(const char * path) {
    struct stat before;
    if (stat(path, &before) != 0) die("stat real model");
    Gguf g = parse_gguf(path);
    int n_exp = 0;
    int seen[5] = {0, 0, 0, 0, 0};
    const enum ggml_type want[] = { GGML_TYPE_Q2_K, GGML_TYPE_Q2_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_NL };
    int bad = 0;
    FILE * f = fopen(path, "rb");
    if (!f) die("open real model");
    for (const Ten & t : g.tensors) {
        if (!ends_with(t.name, "_exps.weight")) continue;
        n_exp++;
        const ExpertView ev = expert_view(t);
        int cols = 0;
        const size_t info = ggml_mmid_repack_info((enum ggml_type) t.type, ev.ne0, ev.ne1, 1, &cols);
        const bool ok = info == ev.slice && info != 0;
        if (!ok) {
            fprintf(stderr, "FAIL size %s type=%s ne0=%lld ne1=%lld nexp=%lld info=%zu slice=%zu\n",
                    t.name.c_str(), ggml_type_name((enum ggml_type) t.type),
                    (long long) ev.ne0, (long long) ev.ne1, (long long) ev.n_exp, info, ev.slice);
            bad++;
            continue;
        }
        int slot = -1;
        for (int i = 0; i < 5; i++) if ((enum ggml_type) t.type == want[i] && !seen[i]) slot = i;
        if (slot < 0) continue;
        seen[slot] = 1;
        std::vector<uint8_t> src(ev.slice), packed(ev.slice), back(ev.slice);
        if (fseeko(f, (off_t) (g.data_start + t.offset), SEEK_SET) != 0 ||
            fread(src.data(), 1, ev.slice, f) != ev.slice) {
            die("read real slice");
        }
        ggml_mmid_repack_slice((enum ggml_type) t.type, ev.ne0, ev.ne1, packed.data(), src.data(), ev.slice);
        ggml_mmid_unrepack_slice((enum ggml_type) t.type, ev.ne0, ev.ne1, back.data(), packed.data(), ev.slice);
        const bool same = memcmp(src.data(), back.data(), ev.slice) == 0;
        const bool moved = memcmp(src.data(), packed.data(), ev.slice) != 0;
        fprintf(stderr, "%s real %s type=%s slice=%zu bit_exact=%d layout_changed=%d\n",
                same && moved ? "PASS" : "FAIL", t.name.c_str(), ggml_type_name((enum ggml_type) t.type),
                ev.slice, same, moved);
        if (!(same && moved)) bad++;
    }
    fclose(f);
    struct stat after;
    if (stat(path, &after) != 0) die("restat real model");
    const bool untouched = before.st_size == after.st_size && before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec
        && before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec && before.st_ino == after.st_ino;
    fprintf(stderr, "%s real file untouched size=%lld expert_tensors=%d i8mm=%d\n",
            untouched ? "PASS" : "FAIL", (long long) after.st_size, n_exp, ggml_cpu_has_matmul_int8());
    if (!untouched || n_exp != 48) bad++;
    for (int i = 0; i < 5; i++) if (!seen[i]) {
        fprintf(stderr, "FAIL missing a real slice of type %s\n", ggml_type_name(want[i]));
        bad++;
    }
    return bad ? 1 : 0;
}

static int verify_self(const char * real_path) {
    int bad = 0;
    const struct { enum ggml_type type; const char * name; int64_t ne0; int64_t ne1; } cases[] = {
        {GGML_TYPE_Q2_K, "Q2_K", 256, 8},
        {GGML_TYPE_Q2_K, "Q2_K", 512, 16},
        {GGML_TYPE_Q2_0, "Q2_0", 256, 8},
        {GGML_TYPE_Q2_0, "Q2_0", 512, 12},
        {GGML_TYPE_Q4_0, "Q4_0", 256, 8},
        {GGML_TYPE_Q4_0, "Q4_0", 512, 12},
        {GGML_TYPE_Q4_K, "Q4_K", 256, 8},
        {GGML_TYPE_Q4_K, "Q4_K", 512, 16},
        {GGML_TYPE_IQ4_NL, "IQ4_NL", 256, 8},
        {GGML_TYPE_IQ4_NL, "IQ4_NL", 512, 12},
    };
    for (auto c : cases) bad |= roundtrip_buf(c.type, c.ne0, c.ne1, 0xA5A5u ^ (uint32_t) c.type ^ (uint32_t) c.ne1, c.name);
    bad |= verify_synth_files(real_path);
    bad |= verify_real(real_path);
    fprintf(stderr, "%s verify-self\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

static const int k_jline = 13;

static std::string journal_for(const char * path) {
    return std::string(path) + ".prerepack-journal";
}

static void journal_write_line(FILE * jf, size_t ti, int e) {
    char line[16];
    const int n = snprintf(line, sizeof line, "%06u %05d\n", (unsigned) ti, e);
    if (n != k_jline) die("journal line width");
    if (fwrite(line, 1, (size_t) k_jline, jf) != (size_t) k_jline) die("journal write");
    fflush(jf);
}

static void apply_one_slice(int fd, const Gguf & g, size_t ti, int e, bool unpack) {
    if (ti >= g.tensors.size()) die("journal tensor index");
    const Ten & t = g.tensors[ti];
    const ExpertView ev = expert_view(t);
    if (e < 0 || (int64_t) e >= ev.n_exp) die("journal slice index");
    const off_t off = (off_t) (g.data_start + t.offset + (uint64_t) e * ev.slice);
    std::vector<uint8_t> src(ev.slice), dst(ev.slice);
    if (pread(fd, src.data(), ev.slice, off) != (ssize_t) ev.slice) die("pread slice");
    transform_slice((enum ggml_type) t.type, ev.ne0, ev.ne1, dst.data(), src.data(), ev.slice, unpack);
    if (pwrite(fd, dst.data(), ev.slice, off) != (ssize_t) ev.slice) die("pwrite slice");
}

// Reverse every slice still listed in the journal, dropping each line after it is restored.
static int unpack_journal(const char * path) {
    const std::string jp = journal_for(path);
    const int jfd = open(jp.c_str(), O_RDWR);
    if (jfd < 0) {
        fprintf(stderr, "prerepack: no journal, leaving file unchanged\n");
        return 0;
    }
    struct stat st;
    if (fstat(jfd, &st) != 0) die("stat journal");
    if (st.st_size % k_jline != 0) die("corrupt journal");
    Gguf g = parse_gguf(path);
    const int fd = open(path, O_RDWR);
    if (fd < 0) die("open model rdwr");
    long n = (long) (st.st_size / k_jline);
    fprintf(stderr, "prerepack: unpacking %ld slices in place\n", n);
    while (n > 0) {
        char line[16] = {};
        if (pread(jfd, line, k_jline, (off_t) (n - 1) * k_jline) != k_jline) die("read journal");
        unsigned ti = 0;
        int e = 0;
        if (sscanf(line, "%u %d", &ti, &e) != 2) die("bad journal line");
        apply_one_slice(fd, g, (size_t) ti, e, true);
        n--;
        if (ftruncate(jfd, (off_t) n * k_jline) != 0) die("truncate journal");
    }
    if (fsync(fd) != 0) die("fsync model");
    close(fd);
    close(jfd);
    unlink(jp.c_str());
    fprintf(stderr, "prerepack: unpacked in place\n");
    return 0;
}

static int pack_inplace(const char * path) {
    const std::string jp = journal_for(path);
    if (access(jp.c_str(), F_OK) == 0) die("journal exists; run --unpack before packing again");
    struct stat before;
    if (stat(path, &before) != 0) die("stat model");
    Gguf g = parse_gguf(path);
    const int fd = open(path, O_RDWR);
    if (fd < 0) die("open model rdwr");
    FILE * jf = fopen(jp.c_str(), "wb");
    if (!jf) die("create journal");
    int n_exp_tensors = 0, n_slices = 0;
    for (size_t ti = 0; ti < g.tensors.size(); ti++) {
        const Ten & t = g.tensors[ti];
        if (!ends_with(t.name, "_exps.weight")) continue;
        const ExpertView ev = expert_view(t);
        int cols = 0;
        const size_t info = ggml_mmid_repack_info((enum ggml_type) t.type, ev.ne0, ev.ne1, 1, &cols);
        if (info != ev.slice) die("repack_info != slice size; refusing to rewrite");
        for (int e = 0; e < (int) ev.n_exp; e++) {
            apply_one_slice(fd, g, ti, e, false);
            journal_write_line(jf, ti, e);
            n_slices++;
        }
        if (fsync(fd) != 0) die("fsync model");
        if (fsync(fileno(jf)) != 0) die("fsync journal");
        n_exp_tensors++;
        fprintf(stderr, "prerepack: packed %s slices=%d\n", t.name.c_str(), (int) ev.n_exp);
    }
    if (fclose(jf) != 0) die("close journal");
    close(fd);
    struct stat after;
    if (stat(path, &after) != 0) die("restat");
    if (after.st_size != before.st_size) die("in-place pack changed the file size");
    fprintf(stderr, "prerepack: packed in place expert_tensors=%d slices=%d size=%lld\n",
            n_exp_tensors, n_slices, (long long) after.st_size);
    return n_exp_tensors > 0 ? 0 : 1;
}

static int verify_inplace_tmp() {
    const char * a = "/tmp/prerepack-inplace.gguf";
    const std::string jp = journal_for(a);
    unlink(a);
    unlink(jp.c_str());
    const std::vector<uint8_t> src = make_synth();
    FILE * f = fopen(a, "wb");
    if (!f || fwrite(src.data(), 1, src.size(), f) != src.size()) die("write inplace synth");
    fclose(f);
    if (pack_inplace(a) != 0) return 1;
    std::vector<uint8_t> packed;
    if (read_file(a, packed) || packed == src) {
        fprintf(stderr, "FAIL inplace pack did not change bytes\n");
        unlink(a); unlink(jp.c_str());
        return 1;
    }
    if (unpack_journal(a) != 0) return 1;
    std::vector<uint8_t> back;
    if (read_file(a, back) || back != src) {
        fprintf(stderr, "FAIL inplace unpack bytes %zu vs %zu\n", back.size(), src.size());
        unlink(a); unlink(jp.c_str());
        return 1;
    }
    if (access(jp.c_str(), F_OK) == 0) {
        fprintf(stderr, "FAIL journal left behind\n");
        unlink(a); unlink(jp.c_str());
        return 1;
    }
    unlink(a);
    fprintf(stderr, "PASS inplace pack/unpack bytes=%zu\n", src.size());
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: prerepack_exps --verify-self gguf | --pack in [-o out] | --unpack in [-o out]\n");
        return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "--verify-inplace") {
        return verify_inplace_tmp();
    }
    if (cmd == "--verify-self") {
        if (argc < 3) {
            fprintf(stderr, "usage: prerepack_exps --verify-self gguf\n");
            return 2;
        }
        return verify_self(argv[2]);
    }
    if ((cmd == "--pack" || cmd == "--unpack") && argc == 3) {
        return cmd == "--unpack" ? unpack_journal(argv[2]) : pack_inplace(argv[2]);
    }
    if ((cmd == "--pack" || cmd == "--unpack") && argc == 5 && std::string(argv[3]) == "-o") {
        transform_file(argv[2], argv[4], cmd == "--unpack", k_real_default);
        return 0;
    }
    fprintf(stderr, "usage: prerepack_exps --verify-self gguf | --pack in [-o out] | --unpack in [-o out]\n");
    return 2;
}
