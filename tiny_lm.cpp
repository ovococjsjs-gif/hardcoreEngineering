// tiny_lm.cpp
// A self-contained byte/character-level causal neural LM for the local
// experiment.  It uses an Elman recurrent core (not a claim of a frontier
// Transformer) and an integrated AURORA product-key residual memory.
//
// Build:
//   c++ -O3 -std=c++17 -Wall -Wextra -pedantic tiny_lm.cpp -o tiny_lm
// Run:
//   ./tiny_lm data/tinyshakespeare_200k.txt --steps 1200 --seq 64 --batch 2

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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Config {
    std::string path;
    int steps = 1200;
    int seq = 64;
    int batch = 2;
    int hidden = 64;
    int memory_order = 5;
    int min_seen = 2;
    double min_confidence = 0.80;
    double memory_boost = 2.0;
    uint64_t seed = 1337;
};

void usage() {
    std::cerr << "usage: tiny_lm DATA [--steps N] [--seq N] [--batch N] "
                 "[--hidden N] [--memory-order N] [--min-seen N] "
                 "[--confidence X] [--boost X] [--seed N]\n";
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
    Config cfg;
    if (argc < 2) {
        usage();
        std::exit(2);
    }
    cfg.path = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string v;
        if (take_arg(i, argc, argv, "--steps", v)) cfg.steps = std::stoi(v);
        else if (take_arg(i, argc, argv, "--seq", v)) cfg.seq = std::stoi(v);
        else if (take_arg(i, argc, argv, "--batch", v)) cfg.batch = std::stoi(v);
        else if (take_arg(i, argc, argv, "--hidden", v)) cfg.hidden = std::stoi(v);
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
        cfg.memory_order <= 0 || cfg.memory_order > 8 || cfg.min_seen <= 0 ||
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
        for (unsigned char c : data) {
            if (to_id[c] < 0) throw std::runtime_error("vocabulary encoding error");
            out.push_back(to_id[c]);
        }
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

inline double sigmoidish(double x) {
    return std::tanh(x);
}

// A small causal recurrent LM.  It is intentionally simple enough that every
// gradient operation can be inspected and reproduced without a ML framework.
class TinyRNN {
public:
    int V;
    int H;
    std::vector<double> emb;  // V x H
    std::vector<double> W;    // H x H
    std::vector<double> out;  // V x H
    std::vector<double> bh;   // H
    std::vector<double> bo;   // V

    explicit TinyRNN(int vocab, int hidden, uint64_t seed) : V(vocab), H(hidden) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> nd(0.0, 0.08);
        emb.resize(V * H);
        W.resize(H * H);
        out.resize(V * H);
        bh.assign(H, 0.0);
        bo.assign(V, 0.0);
        for (double &x : emb) x = nd(rng);
        for (double &x : W) x = nd(rng) / std::sqrt(static_cast<double>(H));
        for (double &x : out) x = nd(rng) / std::sqrt(static_cast<double>(H));
    }

    size_t parameter_count() const {
        return emb.size() + W.size() + out.size() + bh.size() + bo.size();
    }

    static void softmax(const std::vector<double> &logits,
                        std::vector<double> &probs) {
        double mx = -std::numeric_limits<double>::infinity();
        for (double x : logits) mx = std::max(mx, x);
        double sum = 0.0;
        for (size_t i = 0; i < logits.size(); ++i) {
            probs[i] = std::exp(std::max(-80.0, logits[i] - mx));
            sum += probs[i];
        }
        const double inv = 1.0 / std::max(sum, 1e-300);
        for (double &x : probs) x *= inv;
    }

    struct Grad {
        std::vector<double> emb, W, out, bh, bo;
        void zero() {
            std::fill(emb.begin(), emb.end(), 0.0);
            std::fill(W.begin(), W.end(), 0.0);
            std::fill(out.begin(), out.end(), 0.0);
            std::fill(bh.begin(), bh.end(), 0.0);
            std::fill(bo.begin(), bo.end(), 0.0);
        }
    };

    Grad make_grad() const {
        Grad g;
        g.emb.resize(emb.size());
        g.W.resize(W.size());
        g.out.resize(out.size());
        g.bh.resize(bh.size());
        g.bo.resize(bo.size());
        g.zero();
        return g;
    }

    // One truncated-BPTT update.  inputs[t] predicts targets[t].
    double train_sequence(const std::vector<int> &inputs,
                          const std::vector<int> &targets, Grad &g) {
        const int T = static_cast<int>(inputs.size());
        std::vector<double> hs((T + 1) * H, 0.0);
        std::vector<double> all_probs(T * V, 0.0);
        std::vector<double> logits(V), probs(V), dh(H), da(H), dh_next(H, 0.0);
        double loss = 0.0;

        // Forward pass first.  Keeping probabilities is cheap at character
        // vocabulary sizes and makes the reverse pass genuine BPTT rather than
        // the common (and incorrect) one-step approximation.
        for (int t = 0; t < T; ++t) {
            const double *prev = &hs[t * H];
            double *cur = &hs[(t + 1) * H];
            for (int i = 0; i < H; ++i) {
                double x = bh[i] + emb[inputs[t] * H + i];
                for (int j = 0; j < H; ++j) x += W[i * H + j] * prev[j];
                cur[i] = sigmoidish(x);
            }
            for (int v = 0; v < V; ++v) {
                double x = bo[v];
                for (int i = 0; i < H; ++i) x += out[v * H + i] * cur[i];
                logits[v] = x;
            }
            softmax(logits, probs);
            for (int v = 0; v < V; ++v) all_probs[t * V + v] = probs[v];
            loss -= std::log(std::max(probs[targets[t]], 1e-300));
        }

        for (int t = T - 1; t >= 0; --t) {
            const double *prev = &hs[t * H];
            const double *cur = &hs[(t + 1) * H];
            for (int i = 0; i < H; ++i) dh[i] = dh_next[i];
            for (int v = 0; v < V; ++v) {
                const double dz = all_probs[t * V + v] -
                                  (v == targets[t] ? 1.0 : 0.0);
                g.bo[v] += dz;
                for (int i = 0; i < H; ++i) {
                    g.out[v * H + i] += dz * cur[i];
                    dh[i] += out[v * H + i] * dz;
                }
            }
            for (int i = 0; i < H; ++i) {
                da[i] = dh[i] * (1.0 - cur[i] * cur[i]);
                g.bh[i] += da[i];
                g.emb[inputs[t] * H + i] += da[i];
            }
            std::fill(dh_next.begin(), dh_next.end(), 0.0);
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < H; ++j) {
                    g.W[i * H + j] += da[i] * prev[j];
                    dh_next[j] += W[i * H + j] * da[i];
                }
            }
        }
        return loss / std::max(1, T);
    }

    double loss_on(const std::vector<int> &data, size_t begin, size_t end,
                   int max_tokens) const {
        if (end <= begin + 1) return 0.0;
        const size_t stop = std::min(end - 1, begin + static_cast<size_t>(max_tokens));
        std::vector<double> h(H, 0.0), next(H), logits(V), probs(V);
        double loss = 0.0;
        size_t n = 0;
        for (size_t p = begin; p < stop; ++p) {
            for (int i = 0; i < H; ++i) {
                double x = bh[i] + emb[data[p] * H + i];
                for (int j = 0; j < H; ++j) x += W[i * H + j] * h[j];
                next[i] = sigmoidish(x);
            }
            for (int v = 0; v < V; ++v) {
                double x = bo[v];
                for (int i = 0; i < H; ++i) x += out[v * H + i] * next[i];
                logits[v] = x;
            }
            softmax(logits, probs);
            loss -= std::log(std::max(probs[data[p + 1]], 1e-300));
            h.swap(next);
            ++n;
        }
        return loss / std::max<size_t>(1, n);
    }

    void logits_for(const std::vector<int> &context, std::vector<double> &logits) const {
        std::vector<double> h(H, 0.0), next(H);
        for (int token : context) {
            for (int i = 0; i < H; ++i) {
                double x = bh[i] + emb[token * H + i];
                for (int j = 0; j < H; ++j) x += W[i * H + j] * h[j];
                next[i] = sigmoidish(x);
            }
            h.swap(next);
        }
        logits.assign(V, 0.0);
        for (int v = 0; v < V; ++v) {
            logits[v] = bo[v];
            for (int i = 0; i < H; ++i) logits[v] += out[v * H + i] * h[i];
        }
    }

    std::vector<int> generate(const std::vector<int> &seed_context,
                              int count, std::mt19937_64 &rng,
                              const class AuroraMemory *memory,
                              double boost) const;
};

struct Adam {
    double lr = 0.002;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    std::vector<double> m, v;
    int step = 0;

    explicit Adam(size_t n) : m(n, 0.0), v(n, 0.0) {}

    void update(std::vector<double> &p, const std::vector<double> &g) {
        ++step;
        const double b1corr = 1.0 - std::pow(beta1, step);
        const double b2corr = 1.0 - std::pow(beta2, step);
        for (size_t i = 0; i < p.size(); ++i) {
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
            p[i] -= lr * (m[i] / b1corr) /
                    (std::sqrt(v[i] / b2corr) + eps);
        }
    }
};

// AURORA product-key memory.  The full packed context key is retained as the
// fingerprint.  Thus two contexts mapping to one product-key slot cannot
// create a false retrieval.
class AuroraMemory {
public:
    struct Entry {
        uint64_t fingerprint = 0;
        int target = 0;
        uint32_t count = 0;
        uint32_t total = 0;
    };

    int V;
    int order;
    int min_seen;
    double min_confidence;
    uint32_t key_a = 257;
    uint32_t key_b = 263;
    std::unordered_map<uint64_t, std::array<uint16_t, 128>> counts;
    std::unordered_map<uint64_t, std::vector<Entry>> slots;
    size_t observations = 0;
    size_t hits = 0;
    size_t observed_contexts = 0;

    AuroraMemory(int vocab, int context_order, int min_count,
                 double confidence)
        : V(vocab), order(context_order), min_seen(min_count),
          min_confidence(confidence) {
        if (V > 128) throw std::runtime_error("this demo expects V <= 128");
    }

    uint64_t pack_context(const std::vector<int> &context) const {
        uint64_t key = 0;
        const int start = std::max(0, static_cast<int>(context.size()) - order);
        for (int i = start; i < static_cast<int>(context.size()); ++i) {
            key = (key << 7) | static_cast<uint64_t>(context[i] & 127);
        }
        // Include length for short generation prefixes.
        key ^= static_cast<uint64_t>(std::min(order, static_cast<int>(context.size()))) << 56;
        return key;
    }

    uint64_t mix(uint64_t x, uint64_t seed) const {
        x += seed + 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    std::pair<uint32_t, uint32_t> address(uint64_t key) const {
        return {static_cast<uint32_t>(mix(key, 0xA17) % key_a),
                static_cast<uint32_t>(mix(key, 0xB29) % key_b)};
    }

    uint64_t address_key(std::pair<uint32_t, uint32_t> a) const {
        return (static_cast<uint64_t>(a.first) << 32) | a.second;
    }

    void observe(const std::vector<int> &context, int target) {
        const uint64_t key = pack_context(context);
        auto it = counts.find(key);
        if (it == counts.end()) {
            std::array<uint16_t, 128> zero{};
            it = counts.emplace(key, zero).first;
        }
        auto &row = it->second;
        if (row[target] != std::numeric_limits<uint16_t>::max()) ++row[target];
        ++observations;
    }

    void build(const std::vector<int> &train) {
        std::vector<int> ctx;
        ctx.reserve(order);
        for (size_t i = 0; i < train.size(); ++i) {
            ctx.push_back(train[i]);
            if (static_cast<int>(ctx.size()) > order) ctx.erase(ctx.begin());
            if (static_cast<int>(ctx.size()) == order && i + 1 < train.size()) {
                observe(ctx, train[i + 1]);
            }
        }
        observed_contexts = counts.size();
        for (const auto &kv : counts) {
            uint32_t total = 0;
            int best = 0;
            uint16_t best_count = 0;
            for (int v = 0; v < V; ++v) {
                total += kv.second[v];
                if (kv.second[v] > best_count) {
                    best_count = kv.second[v];
                    best = v;
                }
            }
            if (total < static_cast<uint32_t>(min_seen) || total == 0) continue;
            const double confidence = static_cast<double>(best_count) / total;
            if (confidence < min_confidence) continue;
            const uint64_t slot = address_key(address(kv.first));
            slots[slot].push_back(Entry{kv.first, best, best_count, total});
        }
        // The trainer's full count vectors are a build-time structure.  A
        // deployed AURORA instance keeps only the compiled sparse entries.
        counts.clear();
        counts.rehash(0);
    }

    bool lookup(const std::vector<int> &context, int &target,
                double &confidence) const {
        const uint64_t key = pack_context(context);
        const uint64_t slot = address_key(address(key));
        auto it = slots.find(slot);
        if (it == slots.end()) return false;
        for (const Entry &entry : it->second) {
            if (entry.fingerprint == key) {
                target = entry.target;
                confidence = static_cast<double>(entry.count) / entry.total;
                return true;
            }
        }
        return false;
    }

    size_t stored_entries() const {
        size_t n = 0;
        for (const auto &kv : slots) n += kv.second.size();
        return n;
    }

    size_t packed_bytes_estimate() const {
        // address (8) + fingerprint (8) + target (1) + count/total (8),
        // plus a small table header.  Counts are not retained after compile.
        return stored_entries() * 25 + slots.size() * 8 + 16;
    }

    size_t dense_bytes_estimate() const {
        // What a dense uint16 next-token vector per observed context costs.
        return observed_contexts * (8 + static_cast<size_t>(2 * V)) + 2 * V;
    }
};

std::vector<int> TinyRNN::generate(const std::vector<int> &seed_context,
                                   int count, std::mt19937_64 &rng,
                                   const AuroraMemory *memory,
                                   double boost) const {
    std::vector<int> result = seed_context;
    std::vector<double> logits;
    std::vector<double> probs(V);
    for (int step = 0; step < count; ++step) {
        logits_for(result, logits);
        if (memory != nullptr) {
            int target = 0;
            double confidence = 0.0;
            if (memory->lookup(result, target, confidence)) {
                logits[target] += boost * confidence;
            }
        }
        softmax(logits, probs);
        // Temperature-free categorical sampling makes the experiment easy to
        // reproduce.  For deterministic previews, use the argmax in caller
        // code; this generation path intentionally samples.
        std::uniform_real_distribution<double> ud(0.0, 1.0);
        double r = ud(rng);
        int selected = V - 1;
        for (int v = 0; v < V; ++v) {
            r -= probs[v];
            if (r <= 0.0) {
                selected = v;
                break;
            }
        }
        result.push_back(selected);
    }
    return result;
}

struct EvalResult {
    double loss = 0.0;
    double perplexity = 0.0;
    double top1_accuracy = 0.0;
    size_t tokens = 0;
    size_t correct = 0;
    size_t memory_hits = 0;
};

EvalResult evaluate_hybrid(const TinyRNN &model, const AuroraMemory *memory,
                           const std::vector<int> &data, size_t begin,
                           size_t end, double boost, int max_tokens) {
    if (end <= begin + 1) return {};
    const size_t stop = std::min(end - 1, begin + static_cast<size_t>(max_tokens));
    std::vector<double> h(model.H, 0.0), next(model.H), logits(model.V), probs(model.V);
    std::vector<int> context;
    if (memory != nullptr) context.reserve(memory->order);
    EvalResult result;
    for (size_t p = begin; p < stop; ++p) {
        for (int i = 0; i < model.H; ++i) {
            double x = model.bh[i] + model.emb[data[p] * model.H + i];
            for (int j = 0; j < model.H; ++j) x += model.W[i * model.H + j] * h[j];
            next[i] = sigmoidish(x);
        }
        for (int v = 0; v < model.V; ++v) {
            logits[v] = model.bo[v];
            for (int i = 0; i < model.H; ++i) logits[v] += model.out[v * model.H + i] * next[i];
        }
        if (memory != nullptr) {
            context.push_back(data[p]);
            if (static_cast<int>(context.size()) > memory->order) context.erase(context.begin());
            if (static_cast<int>(context.size()) == memory->order) {
                int target = 0;
                double confidence = 0.0;
                if (memory->lookup(context, target, confidence)) {
                    logits[target] += boost * confidence;
                    ++result.memory_hits;
                }
            }
        }
        TinyRNN::softmax(logits, probs);
        result.loss -= std::log(std::max(probs[data[p + 1]], 1e-300));
        int best = 0;
        for (int v = 1; v < model.V; ++v) {
            if (probs[v] > probs[best]) best = v;
        }
        result.correct += static_cast<size_t>(best == data[p + 1]);
        ++result.tokens;
        h.swap(next);
    }
    result.loss /= std::max<size_t>(1, result.tokens);
    result.perplexity = std::exp(std::min(50.0, result.loss));
    result.top1_accuracy = static_cast<double>(result.correct) /
                           std::max<size_t>(1, result.tokens);
    return result;
}

void adam_update(TinyRNN &model, TinyRNN::Grad &g,
                 Adam &opt_emb, Adam &opt_W, Adam &opt_out,
                 Adam &opt_bh, Adam &opt_bo) {
    opt_emb.update(model.emb, g.emb);
    opt_W.update(model.W, g.W);
    opt_out.update(model.out, g.out);
    opt_bh.update(model.bh, g.bh);
    opt_bo.update(model.bo, g.bo);
}

std::string json_escape(const std::string &s) {
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
        if (raw.size() < static_cast<size_t>(cfg.seq + 2))
            throw std::runtime_error("dataset is too short");
        const Vocab vocab(raw);
        const std::vector<int> ids = vocab.encode(raw);
        const size_t train_end = static_cast<size_t>(ids.size() * 0.80);
        const size_t calibration_end = static_cast<size_t>(ids.size() * 0.90);
        std::vector<int> train(ids.begin(), ids.begin() + train_end);
        std::vector<int> calibration(ids.begin() + train_end,
                                     ids.begin() + calibration_end);
        std::vector<int> test(ids.begin() + calibration_end, ids.end());
        if (train.size() <= static_cast<size_t>(cfg.seq + 1) ||
            calibration.size() <= 2 || test.size() <= 2)
            throw std::runtime_error("train/calibration/test split is too small");

        TinyRNN model(static_cast<int>(vocab.symbols.size()), cfg.hidden, cfg.seed);
        TinyRNN::Grad grad = model.make_grad();
        Adam opt_emb(model.emb.size());
        Adam opt_W(model.W.size());
        Adam opt_out(model.out.size());
        Adam opt_bh(model.bh.size());
        Adam opt_bo(model.bo.size());
        std::mt19937_64 rng(cfg.seed + 1);
        std::uniform_int_distribution<size_t> start_dist(
            0, train.size() - static_cast<size_t>(cfg.seq + 1));

        const auto t0 = std::chrono::steady_clock::now();
        double last_loss = 0.0;
        for (int step = 1; step <= cfg.steps; ++step) {
            grad.zero();
            last_loss = 0.0;
            for (int b = 0; b < cfg.batch; ++b) {
                const size_t start = start_dist(rng);
                std::vector<int> x(cfg.seq), y(cfg.seq);
                for (int t = 0; t < cfg.seq; ++t) {
                    x[t] = train[start + t];
                    y[t] = train[start + t + 1];
                }
                last_loss += model.train_sequence(x, y, grad);
            }
            const double inv_batch = 1.0 / cfg.batch;
            for (double &x : grad.emb) x *= inv_batch;
            for (double &x : grad.W) x *= inv_batch;
            for (double &x : grad.out) x *= inv_batch;
            for (double &x : grad.bh) x *= inv_batch;
            for (double &x : grad.bo) x *= inv_batch;
            adam_update(model, grad, opt_emb, opt_W, opt_out, opt_bh, opt_bo);
            last_loss /= cfg.batch;
            if (step == 1 || step % std::max(1, cfg.steps / 6) == 0 || step == cfg.steps)
                std::cerr << "step " << step << "/" << cfg.steps
                          << " train_loss " << std::fixed << std::setprecision(4)
                          << last_loss << "\n";
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double train_seconds = std::chrono::duration<double>(t1 - t0).count();

        const EvalResult calibration_base = evaluate_hybrid(
            model, nullptr, calibration, 0, calibration.size(), 0.0, 50000);
        const EvalResult test_base = evaluate_hybrid(
            model, nullptr, test, 0, test.size(), 0.0, 50000);

        AuroraMemory memory(static_cast<int>(vocab.symbols.size()), cfg.memory_order,
                            cfg.min_seen, cfg.min_confidence);
        memory.build(train);
        // Select the gate strength on a calibration split and report final
        // numbers only on the untouched test split.  This prevents the scalar
        // gate from being tuned on the same tokens used for the headline score.
        const std::array<double, 7> boost_grid{{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0}};
        double selected_boost = cfg.memory_boost;
        EvalResult calibration_hybrid = evaluate_hybrid(
            model, &memory, calibration, 0, calibration.size(), selected_boost, 50000);
        for (double candidate_boost : boost_grid) {
            const EvalResult candidate = evaluate_hybrid(
                model, &memory, calibration, 0, calibration.size(), candidate_boost, 50000);
            if (candidate.loss < calibration_hybrid.loss) {
                calibration_hybrid = candidate;
                selected_boost = candidate_boost;
            }
        }
        const EvalResult test_hybrid = evaluate_hybrid(
            model, &memory, test, 0, test.size(), selected_boost, 50000);

        std::vector<int> seed;
        for (int i = 0; i < std::min<int>(cfg.memory_order, static_cast<int>(test.size())); ++i)
            seed.push_back(test[i]);
        const std::vector<int> sample_base = model.generate(seed, 320, rng, nullptr, 0.0);
        const std::vector<int> sample_hybrid = model.generate(seed, 320, rng, &memory, selected_boost);

        const double dense_memory = static_cast<double>(memory.dense_bytes_estimate());
        const double sparse_memory = static_cast<double>(memory.packed_bytes_estimate());
        std::cout << "{\n"
                  << "  \"dataset_bytes\": " << ids.size() << ",\n"
                  << "  \"train_tokens\": " << train.size() << ",\n"
                  << "  \"calibration_tokens\": " << calibration.size() << ",\n"
                  << "  \"test_tokens\": " << test.size() << ",\n"
                  << "  \"vocab\": " << vocab.symbols.size() << ",\n"
                  << "  \"hidden\": " << cfg.hidden << ",\n"
                  << "  \"rnn_parameters\": " << model.parameter_count() << ",\n"
                  << "  \"steps\": " << cfg.steps << ",\n"
                  << "  \"seq\": " << cfg.seq << ",\n"
                  << "  \"batch\": " << cfg.batch << ",\n"
                  << "  \"train_seconds\": " << std::setprecision(6) << train_seconds << ",\n"
                  << "  \"last_train_loss\": " << last_loss << ",\n"
                  << "  \"calibration_base_loss\": " << calibration_base.loss << ",\n"
                  << "  \"calibration_base_perplexity\": " << calibration_base.perplexity << ",\n"
                  << "  \"calibration_base_top1_accuracy\": " << calibration_base.top1_accuracy << ",\n"
                  << "  \"calibration_hybrid_loss\": " << calibration_hybrid.loss << ",\n"
                  << "  \"calibration_hybrid_perplexity\": " << calibration_hybrid.perplexity << ",\n"
                  << "  \"selected_boost\": " << selected_boost << ",\n"
                  << "  \"test_base_loss\": " << test_base.loss << ",\n"
                  << "  \"test_base_perplexity\": " << test_base.perplexity << ",\n"
                  << "  \"test_base_top1_accuracy\": " << test_base.top1_accuracy << ",\n"
                  << "  \"test_hybrid_loss\": " << test_hybrid.loss << ",\n"
                  << "  \"test_hybrid_perplexity\": " << test_hybrid.perplexity << ",\n"
                  << "  \"test_hybrid_top1_accuracy\": " << test_hybrid.top1_accuracy << ",\n"
                  << "  \"test_hybrid_memory_hits\": " << test_hybrid.memory_hits << ",\n"
                  << "  \"memory_entries\": " << memory.stored_entries() << ",\n"
                  << "  \"observed_contexts_during_compile\": " << memory.observed_contexts << ",\n"
                  << "  \"memory_observations\": " << memory.observations << ",\n"
                  << "  \"dense_memory_bytes_estimate\": " << memory.dense_bytes_estimate() << ",\n"
                  << "  \"sparse_memory_bytes_estimate\": " << memory.packed_bytes_estimate() << ",\n"
                  << "  \"memory_reduction_factor\": " << (dense_memory / std::max(1.0, sparse_memory)) << ",\n"
                  << "  \"sample_base\": \"" << json_escape(vocab.decode(sample_base)) << "\",\n"
                  << "  \"sample_hybrid\": \"" << json_escape(vocab.decode(sample_hybrid)) << "\"\n"
                  << "}\n";
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
