// cirrus_lm.cpp
//
// CIRRUS: Compressed Interleaved Recurrent Routing with Quantized and
// Uncertainty-gated Sparse memory.  This is a small, dependency-free,
// from-scratch causal neural language model designed for the local CPU.
//
// Unlike the earlier AURORA-only experiment, CIRRUS changes the learned core
// and the inference representation as well:
//   * low-rank recurrent transition W = A B;
//   * factorized output O = D C;
//   * full truncated BPTT training;
//   * post-training per-tensor int8 weight quantization;
//   * sparse product-key next-context memory;
//   * held-out calibration for the single memory gate scalar.
//
// It is intentionally not advertised as a frontier Transformer.  It is a
// controlled end-to-end experiment for measuring which costs can be removed
// without hiding a quality loss.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Config {
    std::string path;
    int steps = 3000;
    int seq = 64;
    int batch = 2;
    int hidden = 64;
    int rank = 16;
    int output_rank = 16;
    int memory_order = 5;
    int min_seen = 2;
    double min_confidence = 0.80;
    double memory_boost = 2.0;
    uint64_t seed = 1337;
};

void usage() {
    std::cerr << "usage: cirrus_lm DATA [--steps N] [--seq N] [--batch N] "
                 "[--hidden N] [--rank N] [--output-rank N] "
                 "[--memory-order N] [--min-seen N] [--confidence X] "
                 "[--boost X] [--seed N]\n";
}

bool take_arg(int &i, int argc, char **argv, const std::string &name,
              std::string &value) {
    const std::string arg = argv[i];
    if (arg == name && i + 1 < argc) {
        value = argv[++i];
        return true;
    }
    const std::string prefix = name + "=";
    if (arg.rfind(prefix, 0) == 0) {
        value = arg.substr(prefix.size());
        return true;
    }
    return false;
}

Config parse_config(int argc, char **argv) {
    if (argc < 2) {
        usage();
        std::exit(2);
    }
    Config cfg;
    cfg.path = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string v;
        if (take_arg(i, argc, argv, "--steps", v)) cfg.steps = std::stoi(v);
        else if (take_arg(i, argc, argv, "--seq", v)) cfg.seq = std::stoi(v);
        else if (take_arg(i, argc, argv, "--batch", v)) cfg.batch = std::stoi(v);
        else if (take_arg(i, argc, argv, "--hidden", v)) cfg.hidden = std::stoi(v);
        else if (take_arg(i, argc, argv, "--rank", v)) cfg.rank = std::stoi(v);
        else if (take_arg(i, argc, argv, "--output-rank", v)) cfg.output_rank = std::stoi(v);
        else if (take_arg(i, argc, argv, "--memory-order", v)) cfg.memory_order = std::stoi(v);
        else if (take_arg(i, argc, argv, "--min-seen", v)) cfg.min_seen = std::stoi(v);
        else if (take_arg(i, argc, argv, "--confidence", v)) cfg.min_confidence = std::stod(v);
        else if (take_arg(i, argc, argv, "--boost", v)) cfg.memory_boost = std::stod(v);
        else if (take_arg(i, argc, argv, "--seed", v)) cfg.seed = std::stoull(v);
        else {
            usage();
            std::exit(2);
        }
    }
    if (cfg.steps <= 0 || cfg.seq <= 1 || cfg.batch <= 0 || cfg.hidden <= 0 ||
        cfg.rank <= 0 || cfg.rank > cfg.hidden || cfg.output_rank <= 0 ||
        cfg.output_rank > cfg.hidden || cfg.memory_order <= 0 ||
        cfg.memory_order > 8 || cfg.min_seen <= 0 ||
        cfg.min_confidence < 0.0 || cfg.min_confidence > 1.0) {
        throw std::runtime_error("invalid configuration");
    }
    return cfg;
}

std::string read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open dataset: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

struct Vocab {
    std::vector<unsigned char> symbols;
    std::array<int, 256> to_id{};

    explicit Vocab(const std::string &data) {
        to_id.fill(-1);
        std::array<bool, 256> seen{};
        for (unsigned char c : data) seen[c] = true;
        for (int i = 0; i < 256; ++i) {
            if (seen[i]) {
                to_id[i] = static_cast<int>(symbols.size());
                symbols.push_back(static_cast<unsigned char>(i));
            }
        }
    }

    std::vector<int> encode(const std::string &data) const {
        std::vector<int> out;
        out.reserve(data.size());
        for (unsigned char c : data) out.push_back(to_id[c]);
        return out;
    }

    std::string decode(const std::vector<int> &ids) const {
        std::string out;
        out.reserve(ids.size());
        for (int id : ids) {
            if (id >= 0 && id < static_cast<int>(symbols.size()))
                out.push_back(static_cast<char>(symbols[id]));
        }
        return out;
    }
};

inline double act(double x) { return std::tanh(x); }

class CIRRUS {
public:
    int V, H, R, RO;
    // E: V x H; A: H x R; B: R x H; C: RO x H; D: V x RO.
    std::vector<double> E, A, B, C, D, bh, bo;

    explicit CIRRUS(int vocab, int hidden, int rank, int output_rank,
                    uint64_t seed)
        : V(vocab), H(hidden), R(rank), RO(output_rank) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> nd(0.0, 0.07);
        E.resize(V * H);
        A.resize(H * R);
        B.resize(R * H);
        C.resize(RO * H);
        D.resize(V * RO);
        bh.assign(H, 0.0);
        bo.assign(V, 0.0);
        for (double &x : E) x = nd(rng);
        for (double &x : A) x = nd(rng) / std::sqrt(static_cast<double>(R));
        for (double &x : B) x = nd(rng) / std::sqrt(static_cast<double>(H));
        for (double &x : C) x = nd(rng) / std::sqrt(static_cast<double>(H));
        for (double &x : D) x = nd(rng) / std::sqrt(static_cast<double>(RO));
    }

    size_t parameter_count() const {
        return E.size() + A.size() + B.size() + C.size() + D.size() +
               bh.size() + bo.size();
    }

    size_t float32_bytes() const { return parameter_count() * sizeof(float); }

    struct Grad {
        std::vector<double> E, A, B, C, D, bh, bo;
        void zero() {
            std::fill(E.begin(), E.end(), 0.0);
            std::fill(A.begin(), A.end(), 0.0);
            std::fill(B.begin(), B.end(), 0.0);
            std::fill(C.begin(), C.end(), 0.0);
            std::fill(D.begin(), D.end(), 0.0);
            std::fill(bh.begin(), bh.end(), 0.0);
            std::fill(bo.begin(), bo.end(), 0.0);
        }
    };

    Grad make_grad() const {
        Grad g;
        g.E.resize(E.size());
        g.A.resize(A.size());
        g.B.resize(B.size());
        g.C.resize(C.size());
        g.D.resize(D.size());
        g.bh.resize(bh.size());
        g.bo.resize(bo.size());
        g.zero();
        return g;
    }

    static void softmax(const std::vector<double> &logits,
                        std::vector<double> &prob) {
        double mx = -std::numeric_limits<double>::infinity();
        for (double x : logits) mx = std::max(mx, x);
        double sum = 0.0;
        for (size_t i = 0; i < logits.size(); ++i) {
            prob[i] = std::exp(std::max(-80.0, logits[i] - mx));
            sum += prob[i];
        }
        const double inv = 1.0 / std::max(sum, 1e-300);
        for (double &x : prob) x *= inv;
    }

    // Forward plus reverse-mode differentiation through the complete low-rank
    // recurrent core and factorized output.  No autodiff package is involved.
    double train_sequence(const std::vector<int> &inputs,
                          const std::vector<int> &targets, Grad &g) {
        const int T = static_cast<int>(inputs.size());
        std::vector<double> h((T + 1) * H, 0.0);
        std::vector<double> r(T * R, 0.0);
        std::vector<double> u(T * RO, 0.0);
        std::vector<double> all_prob(T * V, 0.0);
        std::vector<double> logits(V), prob(V), dh(H), da(H), dr(R), du(RO),
            dh_next(H, 0.0);
        double loss = 0.0;

        for (int t = 0; t < T; ++t) {
            const double *prev = &h[t * H];
            double *cur = &h[(t + 1) * H];
            double *rt = &r[t * R];
            double *ut = &u[t * RO];
            for (int k = 0; k < R; ++k) {
                double x = 0.0;
                for (int j = 0; j < H; ++j) x += B[k * H + j] * prev[j];
                rt[k] = x;
            }
            for (int i = 0; i < H; ++i) {
                double x = bh[i] + E[inputs[t] * H + i];
                for (int k = 0; k < R; ++k) x += A[i * R + k] * rt[k];
                cur[i] = act(x);
            }
            for (int q = 0; q < RO; ++q) {
                double x = 0.0;
                for (int i = 0; i < H; ++i) x += C[q * H + i] * cur[i];
                ut[q] = x;
            }
            for (int v = 0; v < V; ++v) {
                double x = bo[v];
                for (int q = 0; q < RO; ++q) x += D[v * RO + q] * ut[q];
                logits[v] = x;
            }
            softmax(logits, prob);
            for (int v = 0; v < V; ++v) all_prob[t * V + v] = prob[v];
            loss -= std::log(std::max(prob[targets[t]], 1e-300));
        }

        for (int t = T - 1; t >= 0; --t) {
            const double *prev = &h[t * H];
            const double *cur = &h[(t + 1) * H];
            const double *rt = &r[t * R];
            const double *ut = &u[t * RO];
            for (int i = 0; i < H; ++i) dh[i] = dh_next[i];
            std::fill(du.begin(), du.end(), 0.0);
            for (int v = 0; v < V; ++v) {
                const double dz = all_prob[t * V + v] -
                                  (v == targets[t] ? 1.0 : 0.0);
                g.bo[v] += dz;
                for (int q = 0; q < RO; ++q) {
                    g.D[v * RO + q] += dz * ut[q];
                    du[q] += D[v * RO + q] * dz;
                }
            }
            for (int q = 0; q < RO; ++q) {
                g.C[q * H] += 0.0; // keeps the loop shape obvious for readers
                for (int i = 0; i < H; ++i) {
                    g.C[q * H + i] += du[q] * cur[i];
                    dh[i] += C[q * H + i] * du[q];
                }
            }
            for (int i = 0; i < H; ++i) {
                da[i] = dh[i] * (1.0 - cur[i] * cur[i]);
                g.bh[i] += da[i];
                g.E[inputs[t] * H + i] += da[i];
            }
            std::fill(dr.begin(), dr.end(), 0.0);
            std::fill(dh_next.begin(), dh_next.end(), 0.0);
            for (int i = 0; i < H; ++i) {
                for (int k = 0; k < R; ++k) {
                    g.A[i * R + k] += da[i] * rt[k];
                    dr[k] += A[i * R + k] * da[i];
                }
            }
            for (int k = 0; k < R; ++k) {
                for (int j = 0; j < H; ++j) {
                    g.B[k * H + j] += dr[k] * prev[j];
                    dh_next[j] += B[k * H + j] * dr[k];
                }
            }
        }
        return loss / std::max(1, T);
    }

    void step(int token, std::vector<double> &h) const {
        std::vector<double> r(R), next(H);
        for (int k = 0; k < R; ++k) {
            r[k] = 0.0;
            for (int j = 0; j < H; ++j) r[k] += B[k * H + j] * h[j];
        }
        for (int i = 0; i < H; ++i) {
            double x = bh[i] + E[token * H + i];
            for (int k = 0; k < R; ++k) x += A[i * R + k] * r[k];
            next[i] = act(x);
        }
        h.swap(next);
    }

    void output(const std::vector<double> &h, std::vector<double> &logits) const {
        std::vector<double> u(RO, 0.0);
        for (int q = 0; q < RO; ++q)
            for (int i = 0; i < H; ++i) u[q] += C[q * H + i] * h[i];
        logits.assign(V, 0.0);
        for (int v = 0; v < V; ++v) {
            logits[v] = bo[v];
            for (int q = 0; q < RO; ++q) logits[v] += D[v * RO + q] * u[q];
        }
    }

    void logits_for(const std::vector<int> &context,
                    std::vector<double> &logits) const {
        std::vector<double> h(H, 0.0);
        for (int token : context) step(token, h);
        output(h, logits);
    }
};

class Adam {
public:
    double lr = 0.002, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    std::vector<double> m, v;
    int step = 0;
    explicit Adam(size_t n) : m(n, 0.0), v(n, 0.0) {}
    void update(std::vector<double> &p, const std::vector<double> &g) {
        ++step;
        const double c1 = 1.0 - std::pow(b1, step);
        const double c2 = 1.0 - std::pow(b2, step);
        for (size_t i = 0; i < p.size(); ++i) {
            m[i] = b1 * m[i] + (1.0 - b1) * g[i];
            v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
            p[i] -= lr * (m[i] / c1) / (std::sqrt(v[i] / c2) + eps);
        }
    }
};

void update(CIRRUS &m, CIRRUS::Grad &g, Adam &e, Adam &a, Adam &b,
            Adam &c, Adam &d, Adam &bh, Adam &bo) {
    e.update(m.E, g.E);
    a.update(m.A, g.A);
    b.update(m.B, g.B);
    c.update(m.C, g.C);
    d.update(m.D, g.D);
    bh.update(m.bh, g.bh);
    bo.update(m.bo, g.bo);
}

struct Eval {
    double loss = 0.0;
    double ppl = 0.0;
    double acc = 0.0;
    size_t tokens = 0;
    size_t correct = 0;
    size_t seconds_hits = 0;
};

template <typename Model>
Eval evaluate(const Model &model, const std::vector<int> &data,
              size_t begin, size_t end, int max_tokens) {
    Eval result;
    if (end <= begin + 1) return result;
    const size_t stop = std::min(end - 1, begin + static_cast<size_t>(max_tokens));
    std::vector<double> h(model.H, 0.0), logits(model.V), prob(model.V);
    for (size_t p = begin; p < stop; ++p) {
        model.step(data[p], h);
        model.output(h, logits);
        CIRRUS::softmax(logits, prob);
        result.loss -= std::log(std::max(prob[data[p + 1]], 1e-300));
        int best = 0;
        for (int v = 1; v < model.V; ++v) if (prob[v] > prob[best]) best = v;
        result.correct += static_cast<size_t>(best == data[p + 1]);
        ++result.tokens;
    }
    result.loss /= std::max<size_t>(1, result.tokens);
    result.ppl = std::exp(std::min(50.0, result.loss));
    result.acc = static_cast<double>(result.correct) / std::max<size_t>(1, result.tokens);
    return result;
}

// Per-tensor symmetric int8 weights.  This is a real quantized forward path,
// not merely the statement that 8-bit storage would be smaller.
struct QTensor {
    std::vector<int8_t> q;
    double scale = 1.0;
    void quantize(const std::vector<double> &src) {
        double mx = 0.0;
        for (double x : src) mx = std::max(mx, std::abs(x));
        scale = mx > 0.0 ? mx / 127.0 : 1.0;
        q.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            const double z = src[i] / scale;
            q[i] = static_cast<int8_t>(std::max(-127.0, std::min(127.0, std::round(z))));
        }
    }
    double at(size_t i) const { return static_cast<double>(q[i]) * scale; }
    size_t bytes() const { return q.size() + sizeof(float); }
};

struct QuantizedCIRRUS {
    int V, H, R, RO;
    QTensor E, A, B, C, D, bh, bo;
    explicit QuantizedCIRRUS(const CIRRUS &m) : V(m.V), H(m.H), R(m.R), RO(m.RO) {
        E.quantize(m.E); A.quantize(m.A); B.quantize(m.B); C.quantize(m.C);
        D.quantize(m.D); bh.quantize(m.bh); bo.quantize(m.bo);
    }
    size_t bytes() const {
        return E.bytes() + A.bytes() + B.bytes() + C.bytes() + D.bytes() +
               bh.bytes() + bo.bytes();
    }
    void step(int token, std::vector<double> &h) const {
        std::vector<double> r(R), next(H);
        for (int k = 0; k < R; ++k)
            for (int j = 0; j < H; ++j) r[k] += B.at(k * H + j) * h[j];
        for (int i = 0; i < H; ++i) {
            double x = bh.at(i) + E.at(token * H + i);
            for (int k = 0; k < R; ++k) x += A.at(i * R + k) * r[k];
            next[i] = std::tanh(x);
        }
        h.swap(next);
    }

    void output(const std::vector<double> &h, std::vector<double> &logits) const {
        std::vector<double> u(RO, 0.0);
        for (int q = 0; q < RO; ++q)
            for (int i = 0; i < H; ++i) u[q] += C.at(q * H + i) * h[i];
        logits.assign(V, 0.0);
        for (int v = 0; v < V; ++v) {
            logits[v] = bo.at(v);
            for (int q = 0; q < RO; ++q) logits[v] += D.at(v * RO + q) * u[q];
        }
    }

    void logits_for(const std::vector<int> &context, std::vector<double> &logits) const {
        std::vector<double> h(H, 0.0);
        for (int token : context) step(token, h);
        output(h, logits);
    }
};

class SparseMemory {
public:
    struct Entry { uint64_t key; int target; uint32_t count; uint32_t total; };
    int V, order, min_seen; double confidence;
    std::unordered_map<uint64_t, std::array<uint16_t, 128>> counts;
    std::unordered_map<uint64_t, std::vector<Entry>> slots;
    size_t observations = 0, observed_contexts = 0;
    uint32_t ka = 257, kb = 263;

    SparseMemory(int vocab, int o, int min_count, double conf)
        : V(vocab), order(o), min_seen(min_count), confidence(conf) {
        if (V > 128) throw std::runtime_error("character demo expects V <= 128");
    }
    uint64_t pack(const std::vector<int> &ctx) const {
        uint64_t x = 0;
        const int start = std::max(0, static_cast<int>(ctx.size()) - order);
        for (int i = start; i < static_cast<int>(ctx.size()); ++i)
            x = (x << 7) | static_cast<uint64_t>(ctx[i] & 127);
        return x ^ (static_cast<uint64_t>(std::min(order, static_cast<int>(ctx.size()))) << 56);
    }
    uint64_t mix(uint64_t x, uint64_t seed) const {
        x += seed + 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }
    uint64_t slot(uint64_t key) const {
        return (mix(key, 0xA17) % ka << 32) | (mix(key, 0xB29) % kb);
    }
    void build(const std::vector<int> &data) {
        std::vector<int> ctx;
        for (size_t i = 0; i < data.size(); ++i) {
            ctx.push_back(data[i]);
            if (static_cast<int>(ctx.size()) > order) ctx.erase(ctx.begin());
            if (static_cast<int>(ctx.size()) != order || i + 1 >= data.size()) continue;
            const uint64_t key = pack(ctx);
            auto it = counts.find(key);
            if (it == counts.end()) it = counts.emplace(key, std::array<uint16_t,128>{}).first;
            if (it->second[data[i + 1]] != 65535) ++it->second[data[i + 1]];
            ++observations;
        }
        observed_contexts = counts.size();
        for (const auto &kv : counts) {
            uint32_t total = 0; int best = 0; uint16_t best_count = 0;
            for (int v = 0; v < V; ++v) {
                total += kv.second[v];
                if (kv.second[v] > best_count) { best_count = kv.second[v]; best = v; }
            }
            if (total < static_cast<uint32_t>(min_seen) || total == 0) continue;
            if (static_cast<double>(best_count) / total < confidence) continue;
            slots[slot(kv.first)].push_back({kv.first, best, best_count, total});
        }
        counts.clear(); counts.rehash(0);
    }
    bool lookup(const std::vector<int> &ctx, int &target, double &conf) const {
        const uint64_t key = pack(ctx);
        auto it = slots.find(slot(key));
        if (it == slots.end()) return false;
        for (const Entry &e : it->second) if (e.key == key) {
            target = e.target; conf = static_cast<double>(e.count) / e.total; return true;
        }
        return false;
    }
    size_t packed_bytes() const {
        size_t entries = 0;
        for (const auto &kv : slots) entries += kv.second.size();
        return entries * 25 + slots.size() * 8 + 16;
    }
    size_t dense_bytes() const {
        return observed_contexts * (8 + static_cast<size_t>(2 * V)) + 2 * V;
    }
};

Eval evaluate_hybrid(const QuantizedCIRRUS &model, const SparseMemory &memory,
                     const std::vector<int> &data, double boost, size_t max_tokens) {
    Eval result;
    if (data.size() < 2) return result;
    const size_t stop = std::min(data.size() - 1, max_tokens);
    std::vector<int> context;
    std::vector<double> h(model.H, 0.0), logits(model.V), prob(model.V);
    context.reserve(memory.order);
    for (size_t p = 0; p < stop; ++p) {
        model.step(data[p], h);
        model.output(h, logits);
        context.push_back(data[p]);
        if (static_cast<int>(context.size()) > memory.order) context.erase(context.begin());
        int target = 0; double conf = 0.0;
        if (static_cast<int>(context.size()) == memory.order && memory.lookup(context, target, conf)) {
            logits[target] += boost * conf;
            ++result.seconds_hits;
        }
        CIRRUS::softmax(logits, prob);
        result.loss -= std::log(std::max(prob[data[p + 1]], 1e-300));
        int best = 0;
        for (int v = 1; v < model.V; ++v) if (prob[v] > prob[best]) best = v;
        result.correct += static_cast<size_t>(best == data[p + 1]);
        ++result.tokens;
    }
    result.loss /= std::max<size_t>(1, result.tokens);
    result.ppl = std::exp(std::min(50.0, result.loss));
    result.acc = static_cast<double>(result.correct) / std::max<size_t>(1, result.tokens);
    return result;
}

std::string escape_json(const std::string &s) {
    std::string out;
    for (unsigned char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c >= 32 && c < 127) out.push_back(static_cast<char>(c));
        else out += "?";
    }
    return out;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Config cfg = parse_config(argc, argv);
        const std::string raw = read_file(cfg.path);
        const Vocab vocab(raw);
        const std::vector<int> ids = vocab.encode(raw);
        if (ids.size() < static_cast<size_t>(cfg.seq + 10))
            throw std::runtime_error("dataset is too short");

        const size_t train_end = static_cast<size_t>(ids.size() * 0.80);
        const size_t calibration_end = static_cast<size_t>(ids.size() * 0.90);
        std::vector<int> train(ids.begin(), ids.begin() + train_end);
        std::vector<int> calibration(ids.begin() + train_end,
                                     ids.begin() + calibration_end);
        std::vector<int> test(ids.begin() + calibration_end, ids.end());

        CIRRUS model(static_cast<int>(vocab.symbols.size()), cfg.hidden, cfg.rank,
                     cfg.output_rank, cfg.seed);
        CIRRUS::Grad grad = model.make_grad();
        Adam optE(model.E.size()), optA(model.A.size()), optB(model.B.size()),
            optC(model.C.size()), optD(model.D.size()), optbh(model.bh.size()),
            optbo(model.bo.size());
        std::mt19937_64 rng(cfg.seed + 1);
        std::uniform_int_distribution<size_t> start_dist(
            0, train.size() - static_cast<size_t>(cfg.seq + 1));

        const auto t0 = std::chrono::steady_clock::now();
        double last_loss = 0.0;
        for (int step = 1; step <= cfg.steps; ++step) {
            grad.zero(); last_loss = 0.0;
            for (int b = 0; b < cfg.batch; ++b) {
                const size_t start = start_dist(rng);
                std::vector<int> x(cfg.seq), y(cfg.seq);
                for (int t = 0; t < cfg.seq; ++t) { x[t] = train[start + t]; y[t] = train[start + t + 1]; }
                last_loss += model.train_sequence(x, y, grad);
            }
            const double inv = 1.0 / cfg.batch;
            for (double &x : grad.E) x *= inv;
            for (double &x : grad.A) x *= inv;
            for (double &x : grad.B) x *= inv;
            for (double &x : grad.C) x *= inv;
            for (double &x : grad.D) x *= inv;
            for (double &x : grad.bh) x *= inv;
            for (double &x : grad.bo) x *= inv;
            update(model, grad, optE, optA, optB, optC, optD, optbh, optbo);
            last_loss /= cfg.batch;
            if (step == 1 || step % std::max(1, cfg.steps / 6) == 0 || step == cfg.steps)
                std::cerr << "step " << step << "/" << cfg.steps << " train_loss "
                          << std::fixed << std::setprecision(4) << last_loss << "\n";
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double train_seconds = std::chrono::duration<double>(t1 - t0).count();

        const auto f0 = std::chrono::steady_clock::now();
        const Eval base_float = evaluate(model, test, 0, test.size(), 50000);
        const auto f1 = std::chrono::steady_clock::now();
        const double float_eval_seconds = std::chrono::duration<double>(f1 - f0).count();

        QuantizedCIRRUS qmodel(model);
        const auto q0 = std::chrono::steady_clock::now();
        const Eval base_q8 = evaluate(qmodel, test, 0, test.size(), 50000);
        const auto q1 = std::chrono::steady_clock::now();
        const double q8_eval_seconds = std::chrono::duration<double>(q1 - q0).count();

        SparseMemory memory(static_cast<int>(vocab.symbols.size()), cfg.memory_order,
                            cfg.min_seen, cfg.min_confidence);
        memory.build(train);
        double selected_boost = cfg.memory_boost;
        Eval calibration_hybrid = evaluate_hybrid(qmodel, memory, calibration, selected_boost, 50000);
        for (double candidate : {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0}) {
            const Eval e = evaluate_hybrid(qmodel, memory, calibration, candidate, 50000);
            if (e.loss < calibration_hybrid.loss) { calibration_hybrid = e; selected_boost = candidate; }
        }
        const Eval test_hybrid = evaluate_hybrid(qmodel, memory, test, selected_boost, 50000);

        std::vector<int> seed(test.begin(), test.begin() + std::min<size_t>(cfg.memory_order, test.size()));
        std::vector<double> logits;
        model.logits_for(seed, logits);
        int next = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        seed.push_back(next);
        std::cout << "{\n"
                  << "  \"dataset_bytes\": " << ids.size() << ",\n"
                  << "  \"train_tokens\": " << train.size() << ",\n"
                  << "  \"calibration_tokens\": " << calibration.size() << ",\n"
                  << "  \"test_tokens\": " << test.size() << ",\n"
                  << "  \"vocab\": " << vocab.symbols.size() << ",\n"
                  << "  \"hidden\": " << cfg.hidden << ",\n"
                  << "  \"recurrent_rank\": " << cfg.rank << ",\n"
                  << "  \"output_rank\": " << cfg.output_rank << ",\n"
                  << "  \"cirrus_parameters\": " << model.parameter_count() << ",\n"
                  << "  \"cirrus_float32_bytes\": " << model.float32_bytes() << ",\n"
                  << "  \"cirrus_q8_bytes\": " << qmodel.bytes() << ",\n"
                  << "  \"steps\": " << cfg.steps << ",\n"
                  << "  \"seq\": " << cfg.seq << ",\n"
                  << "  \"batch\": " << cfg.batch << ",\n"
                  << "  \"train_seconds\": " << std::setprecision(6) << train_seconds << ",\n"
                  << "  \"last_train_loss\": " << last_loss << ",\n"
                  << "  \"float_test_loss\": " << base_float.loss << ",\n"
                  << "  \"float_test_perplexity\": " << base_float.ppl << ",\n"
                  << "  \"float_test_top1\": " << base_float.acc << ",\n"
                  << "  \"q8_test_loss\": " << base_q8.loss << ",\n"
                  << "  \"q8_test_perplexity\": " << base_q8.ppl << ",\n"
                  << "  \"q8_test_top1\": " << base_q8.acc << ",\n"
                  << "  \"q8_eval_seconds\": " << q8_eval_seconds << ",\n"
                  << "  \"float_eval_seconds\": " << float_eval_seconds << ",\n"
                  << "  \"calibration_hybrid_loss\": " << calibration_hybrid.loss << ",\n"
                  << "  \"selected_boost\": " << selected_boost << ",\n"
                  << "  \"test_hybrid_loss\": " << test_hybrid.loss << ",\n"
                  << "  \"test_hybrid_perplexity\": " << test_hybrid.ppl << ",\n"
                  << "  \"test_hybrid_top1\": " << test_hybrid.acc << ",\n"
                  << "  \"test_hybrid_memory_hits\": " << test_hybrid.seconds_hits << ",\n"
                  << "  \"memory_observed_contexts\": " << memory.observed_contexts << ",\n"
                  << "  \"memory_entries\": " << (memory.packed_bytes() - memory.slots.size() * 8 - 16) / 25 << ",\n"
                  << "  \"dense_memory_bytes_estimate\": " << memory.dense_bytes() << ",\n"
                  << "  \"sparse_memory_bytes_estimate\": " << memory.packed_bytes() << ",\n"
                  << "  \"context_memory_reduction\": " << (static_cast<double>(memory.dense_bytes()) / std::max<size_t>(1, memory.packed_bytes())) << ",\n"
                  << "  \"sample_seed_plus_prediction\": \"" << escape_json(vocab.decode(seed)) << "\"\n"
                  << "}\n";
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
