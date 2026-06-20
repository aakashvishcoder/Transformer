#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct CharTokenizer {
    std::vector<char> id_to_char;
    std::unordered_map<char, int> char_to_id;

    explicit CharTokenizer(const std::string& corpus) {
        for (char c : corpus) {
            if (char_to_id.find(c) == char_to_id.end()) {
                int id = static_cast<int>(id_to_char.size());
                char_to_id[c] = id;
                id_to_char.push_back(c);
            }
        }
        if (id_to_char.empty()) {
            throw std::runtime_error("Tokenizer corpus must not be empty");
        }
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (char c : text) {
            auto it = char_to_id.find(c);
            if (it == char_to_id.end()) {
                ids.push_back(0);
            } else {
                ids.push_back(it->second);
            }
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string out;
        out.reserve(ids.size());
        for (int id : ids) {
            if (id < 0 || id >= static_cast<int>(id_to_char.size())) {
                out.push_back('?');
            } else {
                out.push_back(id_to_char[static_cast<size_t>(id)]);
            }
        }
        return out;
    }

    int vocab_size() const {
        return static_cast<int>(id_to_char.size());
    }
};

struct TinyTransformer {
    int vocab;
    int d_model;
    int d_ff;
    int max_seq;

    // Embeddings.
    std::vector<float> token_emb; // [vocab, d_model]
    std::vector<float> pos_emb;   // [max_seq, d_model]

    // Single-head attention projections.
    std::vector<float> wq; // [d_model, d_model]
    std::vector<float> wk;
    std::vector<float> wv;
    std::vector<float> wo;

    // Feed-forward.
    std::vector<float> w1; // [d_model, d_ff]
    std::vector<float> w2; // [d_ff, d_model]

    explicit TinyTransformer(int vocab_size,
                             int d_model_ = 24,
                             int d_ff_ = 48,
                             int max_seq_ = 64,
                             uint32_t seed = 1337)
        : vocab(vocab_size),
          d_model(d_model_),
          d_ff(d_ff_),
          max_seq(max_seq_),
          token_emb(static_cast<size_t>(vocab_size * d_model_)),
          pos_emb(static_cast<size_t>(max_seq_ * d_model_)),
          wq(static_cast<size_t>(d_model_ * d_model_)),
          wk(static_cast<size_t>(d_model_ * d_model_)),
          wv(static_cast<size_t>(d_model_ * d_model_)),
          wo(static_cast<size_t>(d_model_ * d_model_)),
          w1(static_cast<size_t>(d_model_ * d_ff_)),
          w2(static_cast<size_t>(d_ff_ * d_model_)) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.0f, 0.08f);

        auto init = [&](std::vector<float>& vec) {
            for (float& x : vec) x = nd(rng);
        };

        init(token_emb);
        init(pos_emb);
        init(wq);
        init(wk);
        init(wv);
        init(wo);
        init(w1);
        init(w2);
    }

    static float dot_row(const std::vector<float>& a, int a_row, int a_cols,
                         const std::vector<float>& b, int b_row, int b_cols) {
        float s = 0.0f;
        for (int i = 0; i < a_cols; ++i) {
            s += a[static_cast<size_t>(a_row * a_cols + i)] * b[static_cast<size_t>(b_row * b_cols + i)];
        }
        return s;
    }

    static std::vector<float> matmul(const std::vector<float>& a, int a_rows, int a_cols,
                                     const std::vector<float>& b, int b_cols) {
        // b is [a_cols, b_cols]
        std::vector<float> out(static_cast<size_t>(a_rows * b_cols), 0.0f);
        for (int r = 0; r < a_rows; ++r) {
            for (int c = 0; c < b_cols; ++c) {
                float s = 0.0f;
                for (int k = 0; k < a_cols; ++k) {
                    s += a[static_cast<size_t>(r * a_cols + k)] * b[static_cast<size_t>(k * b_cols + c)];
                }
                out[static_cast<size_t>(r * b_cols + c)] = s;
            }
        }
        return out;
    }

    std::vector<float> forward(const std::vector<int>& ids) const {
        const int seq = static_cast<int>(ids.size());
        if (seq <= 0 || seq > max_seq) {
            throw std::runtime_error("Input sequence length is invalid for TinyTransformer");
        }

        // 1) Embedding + positional encoding. X: [seq, d_model]
        std::vector<float> x(static_cast<size_t>(seq * d_model), 0.0f);
        for (int t = 0; t < seq; ++t) {
            int tok = ids[static_cast<size_t>(t)];
            for (int i = 0; i < d_model; ++i) {
                x[static_cast<size_t>(t * d_model + i)] =
                    token_emb[static_cast<size_t>(tok * d_model + i)] +
                    pos_emb[static_cast<size_t>(t * d_model + i)];
            }
        }

        // 2) Single-head causal self-attention.
        std::vector<float> q = matmul(x, seq, d_model, wq, d_model); // [seq, d]
        std::vector<float> k = matmul(x, seq, d_model, wk, d_model); // [seq, d]
        std::vector<float> v = matmul(x, seq, d_model, wv, d_model); // [seq, d]

        std::vector<float> attn_out(static_cast<size_t>(seq * d_model), 0.0f);
        const float scale = 1.0f / std::sqrt(static_cast<float>(d_model));

        for (int t = 0; t < seq; ++t) {
            std::vector<float> scores(static_cast<size_t>(t + 1), 0.0f);
            float max_score = -1e30f;
            for (int j = 0; j <= t; ++j) {
                float s = dot_row(q, t, d_model, k, j, d_model) * scale;
                scores[static_cast<size_t>(j)] = s;
                max_score = std::max(max_score, s);
            }

            float exp_sum = 0.0f;
            for (int j = 0; j <= t; ++j) {
                float e = std::exp(scores[static_cast<size_t>(j)] - max_score);
                scores[static_cast<size_t>(j)] = e;
                exp_sum += e;
            }

            for (int j = 0; j <= t; ++j) {
                float a = scores[static_cast<size_t>(j)] / exp_sum;
                for (int d = 0; d < d_model; ++d) {
                    attn_out[static_cast<size_t>(t * d_model + d)] +=
                        a * v[static_cast<size_t>(j * d_model + d)];
                }
            }
        }

        std::vector<float> attn_proj = matmul(attn_out, seq, d_model, wo, d_model);

        // Residual 1: x + attention.
        std::vector<float> h1(static_cast<size_t>(seq * d_model), 0.0f);
        for (size_t i = 0; i < h1.size(); ++i) {
            h1[i] = x[i] + attn_proj[i];
        }

        // 3) Feed-forward block.
        std::vector<float> ff1 = matmul(h1, seq, d_model, w1, d_ff);
        for (float& z : ff1) {
            z = std::max(0.0f, z);
        }
        std::vector<float> ff2 = matmul(ff1, seq, d_ff, w2, d_model);

        // Residual 2: h1 + ff2.
        std::vector<float> h2(static_cast<size_t>(seq * d_model), 0.0f);
        for (size_t i = 0; i < h2.size(); ++i) {
            h2[i] = h1[i] + ff2[i];
        }

        // 4) Output logits for next token from final position.
        // Weight tying: logits = last_hidden * token_emb^T
        std::vector<float> logits(static_cast<size_t>(vocab), 0.0f);
        const int last = seq - 1;
        for (int tok = 0; tok < vocab; ++tok) {
            float s = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                s += h2[static_cast<size_t>(last * d_model + d)] *
                     token_emb[static_cast<size_t>(tok * d_model + d)];
            }
            logits[static_cast<size_t>(tok)] = s;
        }

        return logits;
    }

    std::vector<int> generate(const std::vector<int>& prompt,
                              int steps,
                              int context_window = 32) const {
        if (prompt.empty()) {
            throw std::runtime_error("Prompt must not be empty");
        }
        std::vector<int> ids = prompt;
        ids.reserve(prompt.size() + static_cast<size_t>(steps));

        for (int s = 0; s < steps; ++s) {
            int start = 0;
            if (static_cast<int>(ids.size()) > context_window) {
                start = static_cast<int>(ids.size()) - context_window;
            }
            std::vector<int> ctx(ids.begin() + start, ids.end());
            std::vector<float> logits = forward(ctx);

            int next_id = 0;
            float best = logits[0];
            for (int i = 1; i < static_cast<int>(logits.size()); ++i) {
                if (logits[static_cast<size_t>(i)] > best) {
                    best = logits[static_cast<size_t>(i)];
                    next_id = i;
                }
            }
            ids.push_back(next_id);
        }

        return ids;
    }
};

int main() {
    const std::string corpus =
        "hello world. this is a tiny transformer demo for character generation. ";

    CharTokenizer tok(corpus);
    TinyTransformer model(tok.vocab_size(), 24, 48, 64, 1234);

    std::string prompt = "hello ";
    std::vector<int> prompt_ids = tok.encode(prompt);
    std::vector<int> out_ids = model.generate(prompt_ids, 80, 32);

    std::cout << "Tiny Transformer milestone (inference-only)\n";
    std::cout << "vocab_size=" << tok.vocab_size() << " d_model=24 heads=1 blocks=1\n";
    std::cout << "prompt:    " << prompt << "\n";
    std::cout << "generated: " << tok.decode(out_ids) << "\n";

    return 0;
}
