#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
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

    // Output projection.
    std::vector<float> out_proj; // [d_model, vocab]
    std::vector<float> out_bias; // [vocab]

    struct ForwardCache {
        int seq = 0;
        std::vector<float> x;            // [seq, d_model]
        std::vector<float> q;            // [seq, d_model]
        std::vector<float> k;            // [seq, d_model]
        std::vector<float> v;            // [seq, d_model]
        std::vector<float> attn_out;     // [seq, d_model]
        std::vector<float> h1;           // [seq, d_model]
        std::vector<float> ff1_lin;      // [seq, d_ff]
        std::vector<float> ff1_act;      // [seq, d_ff]
        std::vector<float> h2;           // [seq, d_model]
        std::vector<float> logits_all;   // [seq, vocab]
        std::vector<float> attn_weights; // [seq, seq]
    };

    struct OutputHeadBackward {
        float loss = 0.0f;
        std::vector<float> grad_out_proj; // [d_model, vocab]
        std::vector<float> grad_out_bias; // [vocab]
        std::vector<float> grad_h2;       // [seq, d_model]
    };

    struct FFNBackward {
        std::vector<float> grad_w2; // [d_ff, d_model]
        std::vector<float> grad_w1; // [d_model, d_ff]
        std::vector<float> grad_h1; // [seq, d_model]
    };

    struct AttentionProjBackward {
        std::vector<float> grad_wo;       // [d_model, d_model]
        std::vector<float> grad_attn_out; // [seq, d_model]
    };

    struct AttentionCoreBackward {
        std::vector<float> grad_wq; // [d_model, d_model]
        std::vector<float> grad_wk; // [d_model, d_model]
        std::vector<float> grad_wv; // [d_model, d_model]
        std::vector<float> grad_x;  // [seq, d_model]
    };

    struct EmbeddingBackward {
        std::vector<float> grad_token_emb; // [vocab, d_model]
        std::vector<float> grad_pos_emb;   // [max_seq, d_model]
    };

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
          w2(static_cast<size_t>(d_ff_ * d_model_)),
          out_proj(static_cast<size_t>(d_model_ * vocab_size)),
          out_bias(static_cast<size_t>(vocab_size), 0.0f) {
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
        init(out_proj);
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

    ForwardCache forward_cache(const std::vector<int>& ids) const {
        const int seq = static_cast<int>(ids.size());
        if (seq <= 0 || seq > max_seq) {
            throw std::runtime_error("Input sequence length is invalid for TinyTransformer");
        }

        ForwardCache cache;
        cache.seq = seq;

        // 1) Embedding + positional encoding. X: [seq, d_model]
        cache.x.assign(static_cast<size_t>(seq * d_model), 0.0f);
        for (int t = 0; t < seq; ++t) {
            int tok = ids[static_cast<size_t>(t)];
            for (int i = 0; i < d_model; ++i) {
                cache.x[static_cast<size_t>(t * d_model + i)] =
                    token_emb[static_cast<size_t>(tok * d_model + i)] +
                    pos_emb[static_cast<size_t>(t * d_model + i)];
            }
        }

        // 2) Single-head causal self-attention.
        cache.q = matmul(cache.x, seq, d_model, wq, d_model); // [seq, d]
        cache.k = matmul(cache.x, seq, d_model, wk, d_model); // [seq, d]
        cache.v = matmul(cache.x, seq, d_model, wv, d_model); // [seq, d]

        cache.attn_out.assign(static_cast<size_t>(seq * d_model), 0.0f);
        cache.attn_weights.assign(static_cast<size_t>(seq * seq), 0.0f);
        const float scale = 1.0f / std::sqrt(static_cast<float>(d_model));

        for (int t = 0; t < seq; ++t) {
            std::vector<float> scores(static_cast<size_t>(t + 1), 0.0f);
            float max_score = -1e30f;
            for (int j = 0; j <= t; ++j) {
                float s = dot_row(cache.q, t, d_model, cache.k, j, d_model) * scale;
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
                cache.attn_weights[static_cast<size_t>(t * seq + j)] = a;
                for (int d = 0; d < d_model; ++d) {
                    cache.attn_out[static_cast<size_t>(t * d_model + d)] +=
                        a * cache.v[static_cast<size_t>(j * d_model + d)];
                }
            }
        }

        std::vector<float> attn_proj = matmul(cache.attn_out, seq, d_model, wo, d_model);

        // Residual 1: x + attention.
        cache.h1.assign(static_cast<size_t>(seq * d_model), 0.0f);
        for (size_t i = 0; i < cache.h1.size(); ++i) {
            cache.h1[i] = cache.x[i] + attn_proj[i];
        }

        // 3) Feed-forward block.
        cache.ff1_lin = matmul(cache.h1, seq, d_model, w1, d_ff);
        cache.ff1_act = cache.ff1_lin;
        for (float& z : cache.ff1_act) {
            z = std::max(0.0f, z);
        }
        std::vector<float> ff2 = matmul(cache.ff1_act, seq, d_ff, w2, d_model);

        // Residual 2: h1 + ff2.
        cache.h2.assign(static_cast<size_t>(seq * d_model), 0.0f);
        for (size_t i = 0; i < cache.h2.size(); ++i) {
            cache.h2[i] = cache.h1[i] + ff2[i];
        }

        // 4) Output logits for all positions.
        cache.logits_all.assign(static_cast<size_t>(seq * vocab), 0.0f);
        for (int t = 0; t < seq; ++t) {
            for (int tok = 0; tok < vocab; ++tok) {
                float s = out_bias[static_cast<size_t>(tok)];
                for (int d = 0; d < d_model; ++d) {
                    s += cache.h2[static_cast<size_t>(t * d_model + d)] *
                         out_proj[static_cast<size_t>(d * vocab + tok)];
                }
                cache.logits_all[static_cast<size_t>(t * vocab + tok)] = s;
            }
        }

        return cache;
    }

    std::vector<float> forward(const std::vector<int>& ids) const {
        ForwardCache cache = forward_cache(ids);
        std::vector<float> logits(static_cast<size_t>(vocab), 0.0f);
        const int last = cache.seq - 1;
        for (int tok = 0; tok < vocab; ++tok) {
            logits[static_cast<size_t>(tok)] = cache.logits_all[static_cast<size_t>(last * vocab + tok)];
        }
        return logits;
    }

    static void stable_softmax(const std::vector<float>& logits, std::vector<float>& probs) {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (float x : logits) max_logit = std::max(max_logit, x);
        probs.resize(logits.size());
        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            float e = std::exp(logits[i] - max_logit);
            probs[i] = e;
            sum += e;
        }
        if (sum <= 0.0f) {
            float u = 1.0f / static_cast<float>(probs.size());
            for (float& p : probs) p = u;
            return;
        }
        for (float& p : probs) p /= sum;
    }

    OutputHeadBackward backprop_output_head(const ForwardCache& cache,
                                            const std::vector<int>& targets) const {
        if (static_cast<int>(targets.size()) != cache.seq) {
            throw std::runtime_error("Target length must match cache sequence length");
        }

        OutputHeadBackward grads;
        grads.grad_out_proj.assign(static_cast<size_t>(d_model * vocab), 0.0f);
        grads.grad_out_bias.assign(static_cast<size_t>(vocab), 0.0f);
        grads.grad_h2.assign(static_cast<size_t>(cache.seq * d_model), 0.0f);

        for (int p = 0; p < cache.seq; ++p) {
            std::vector<float> logits(static_cast<size_t>(vocab), 0.0f);
            for (int tok = 0; tok < vocab; ++tok) {
                logits[static_cast<size_t>(tok)] = cache.logits_all[static_cast<size_t>(p * vocab + tok)];
            }

            std::vector<float> probs;
            stable_softmax(logits, probs);

            int y = targets[static_cast<size_t>(p)];
            grads.loss += -std::log(std::max(probs[static_cast<size_t>(y)], 1e-8f));

            probs[static_cast<size_t>(y)] -= 1.0f;

            for (int tok = 0; tok < vocab; ++tok) {
                float g = probs[static_cast<size_t>(tok)];
                grads.grad_out_bias[static_cast<size_t>(tok)] += g;
                for (int d = 0; d < d_model; ++d) {
                    grads.grad_out_proj[static_cast<size_t>(d * vocab + tok)] +=
                        cache.h2[static_cast<size_t>(p * d_model + d)] * g;
                    grads.grad_h2[static_cast<size_t>(p * d_model + d)] +=
                        out_proj[static_cast<size_t>(d * vocab + tok)] * g;
                }
            }
        }

        grads.loss /= static_cast<float>(std::max(1, cache.seq));
        return grads;
    }

    FFNBackward backprop_ffn(const ForwardCache& cache,
                             const std::vector<float>& grad_h2) const {
        if (grad_h2.size() != cache.h2.size()) {
            throw std::runtime_error("grad_h2 shape mismatch in backprop_ffn");
        }

        FFNBackward grads;
        grads.grad_w2.assign(static_cast<size_t>(d_ff * d_model), 0.0f);
        grads.grad_w1.assign(static_cast<size_t>(d_model * d_ff), 0.0f);
        grads.grad_h1.assign(static_cast<size_t>(cache.seq * d_model), 0.0f);

        // Residual path: h2 = h1 + ff2.
        // grad_h1 gets direct skip contribution and FFN contribution.
        std::vector<float> grad_ff2 = grad_h2;
        for (size_t i = 0; i < grads.grad_h1.size(); ++i) {
            grads.grad_h1[i] = grad_h2[i];
        }

        // ff2 = ff1_act @ w2.
        for (int f = 0; f < d_ff; ++f) {
            for (int d = 0; d < d_model; ++d) {
                float acc = 0.0f;
                for (int t = 0; t < cache.seq; ++t) {
                    acc += cache.ff1_act[static_cast<size_t>(t * d_ff + f)] *
                           grad_ff2[static_cast<size_t>(t * d_model + d)];
                }
                grads.grad_w2[static_cast<size_t>(f * d_model + d)] = acc;
            }
        }

        std::vector<float> grad_ff1_act(static_cast<size_t>(cache.seq * d_ff), 0.0f);
        for (int t = 0; t < cache.seq; ++t) {
            for (int f = 0; f < d_ff; ++f) {
                float acc = 0.0f;
                for (int d = 0; d < d_model; ++d) {
                    acc += grad_ff2[static_cast<size_t>(t * d_model + d)] *
                           w2[static_cast<size_t>(f * d_model + d)];
                }
                grad_ff1_act[static_cast<size_t>(t * d_ff + f)] = acc;
            }
        }

        std::vector<float> grad_ff1_lin(static_cast<size_t>(cache.seq * d_ff), 0.0f);
        for (size_t i = 0; i < grad_ff1_lin.size(); ++i) {
            grad_ff1_lin[i] = cache.ff1_lin[i] > 0.0f ? grad_ff1_act[i] : 0.0f;
        }

        // ff1_lin = h1 @ w1.
        for (int d = 0; d < d_model; ++d) {
            for (int f = 0; f < d_ff; ++f) {
                float acc = 0.0f;
                for (int t = 0; t < cache.seq; ++t) {
                    acc += cache.h1[static_cast<size_t>(t * d_model + d)] *
                           grad_ff1_lin[static_cast<size_t>(t * d_ff + f)];
                }
                grads.grad_w1[static_cast<size_t>(d * d_ff + f)] = acc;
            }
        }

        std::vector<float> grad_h1_from_ffn(static_cast<size_t>(cache.seq * d_model), 0.0f);
        for (int t = 0; t < cache.seq; ++t) {
            for (int d = 0; d < d_model; ++d) {
                float acc = 0.0f;
                for (int f = 0; f < d_ff; ++f) {
                    acc += grad_ff1_lin[static_cast<size_t>(t * d_ff + f)] *
                           w1[static_cast<size_t>(d * d_ff + f)];
                }
                grad_h1_from_ffn[static_cast<size_t>(t * d_model + d)] = acc;
            }
        }

        for (size_t i = 0; i < grads.grad_h1.size(); ++i) {
            grads.grad_h1[i] += grad_h1_from_ffn[i];
        }

        return grads;
    }

    AttentionProjBackward backprop_attention_proj(const ForwardCache& cache,
                                                  const std::vector<float>& grad_h1) const {
        if (grad_h1.size() != cache.h1.size()) {
            throw std::runtime_error("grad_h1 shape mismatch in backprop_attention_proj");
        }

        AttentionProjBackward grads;
        grads.grad_wo.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_attn_out.assign(static_cast<size_t>(cache.seq * d_model), 0.0f);

        // h1 = x + attn_proj and attn_proj = attn_out @ wo.
        // grad_attn_proj is grad_h1 (residual passthrough).
        for (int i = 0; i < d_model; ++i) {
            for (int j = 0; j < d_model; ++j) {
                float acc = 0.0f;
                for (int t = 0; t < cache.seq; ++t) {
                    acc += cache.attn_out[static_cast<size_t>(t * d_model + i)] *
                           grad_h1[static_cast<size_t>(t * d_model + j)];
                }
                grads.grad_wo[static_cast<size_t>(i * d_model + j)] = acc;
            }
        }

        for (int t = 0; t < cache.seq; ++t) {
            for (int i = 0; i < d_model; ++i) {
                float acc = 0.0f;
                for (int j = 0; j < d_model; ++j) {
                    acc += grad_h1[static_cast<size_t>(t * d_model + j)] *
                           wo[static_cast<size_t>(i * d_model + j)];
                }
                grads.grad_attn_out[static_cast<size_t>(t * d_model + i)] = acc;
            }
        }

        return grads;
    }

    AttentionCoreBackward backprop_attention_core(const ForwardCache& cache,
                                                  const std::vector<float>& grad_attn_out) const {
        if (grad_attn_out.size() != cache.attn_out.size()) {
            throw std::runtime_error("grad_attn_out shape mismatch in backprop_attention_core");
        }

        AttentionCoreBackward grads;
        grads.grad_wq.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_wk.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_wv.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_x.assign(static_cast<size_t>(cache.seq * d_model), 0.0f);

        std::vector<float> grad_q(static_cast<size_t>(cache.seq * d_model), 0.0f);
        std::vector<float> grad_k(static_cast<size_t>(cache.seq * d_model), 0.0f);
        std::vector<float> grad_v(static_cast<size_t>(cache.seq * d_model), 0.0f);

        const float scale = 1.0f / std::sqrt(static_cast<float>(d_model));

        // Backprop through attn_out[t] = sum_j a[t,j] * v[j].
        for (int t = 0; t < cache.seq; ++t) {
            std::vector<float> grad_a(static_cast<size_t>(t + 1), 0.0f);

            for (int j = 0; j <= t; ++j) {
                float acc = 0.0f;
                for (int d = 0; d < d_model; ++d) {
                    acc += grad_attn_out[static_cast<size_t>(t * d_model + d)] *
                           cache.v[static_cast<size_t>(j * d_model + d)];
                    grad_v[static_cast<size_t>(j * d_model + d)] +=
                        cache.attn_weights[static_cast<size_t>(t * cache.seq + j)] *
                        grad_attn_out[static_cast<size_t>(t * d_model + d)];
                }
                grad_a[static_cast<size_t>(j)] = acc;
            }

            // Backprop through masked softmax: grad_s = a * (grad_a - dot(grad_a, a)).
            float dot_ga_a = 0.0f;
            for (int j = 0; j <= t; ++j) {
                dot_ga_a += grad_a[static_cast<size_t>(j)] *
                            cache.attn_weights[static_cast<size_t>(t * cache.seq + j)];
            }

            for (int j = 0; j <= t; ++j) {
                float a = cache.attn_weights[static_cast<size_t>(t * cache.seq + j)];
                float grad_s = a * (grad_a[static_cast<size_t>(j)] - dot_ga_a);

                // s = scale * dot(q_t, k_j)
                for (int d = 0; d < d_model; ++d) {
                    grad_q[static_cast<size_t>(t * d_model + d)] +=
                        grad_s * scale * cache.k[static_cast<size_t>(j * d_model + d)];
                    grad_k[static_cast<size_t>(j * d_model + d)] +=
                        grad_s * scale * cache.q[static_cast<size_t>(t * d_model + d)];
                }
            }
        }

        // q = x @ wq, k = x @ wk, v = x @ wv.
        for (int i = 0; i < d_model; ++i) {
            for (int o = 0; o < d_model; ++o) {
                float gq = 0.0f;
                float gk = 0.0f;
                float gv = 0.0f;
                for (int t = 0; t < cache.seq; ++t) {
                    gq += cache.x[static_cast<size_t>(t * d_model + i)] *
                          grad_q[static_cast<size_t>(t * d_model + o)];
                    gk += cache.x[static_cast<size_t>(t * d_model + i)] *
                          grad_k[static_cast<size_t>(t * d_model + o)];
                    gv += cache.x[static_cast<size_t>(t * d_model + i)] *
                          grad_v[static_cast<size_t>(t * d_model + o)];
                }
                grads.grad_wq[static_cast<size_t>(i * d_model + o)] = gq;
                grads.grad_wk[static_cast<size_t>(i * d_model + o)] = gk;
                grads.grad_wv[static_cast<size_t>(i * d_model + o)] = gv;
            }
        }

        for (int t = 0; t < cache.seq; ++t) {
            for (int i = 0; i < d_model; ++i) {
                float acc = 0.0f;
                for (int o = 0; o < d_model; ++o) {
                    acc += grad_q[static_cast<size_t>(t * d_model + o)] * wq[static_cast<size_t>(i * d_model + o)];
                    acc += grad_k[static_cast<size_t>(t * d_model + o)] * wk[static_cast<size_t>(i * d_model + o)];
                    acc += grad_v[static_cast<size_t>(t * d_model + o)] * wv[static_cast<size_t>(i * d_model + o)];
                }
                grads.grad_x[static_cast<size_t>(t * d_model + i)] = acc;
            }
        }

        return grads;
    }

    EmbeddingBackward backprop_embeddings(const std::vector<int>& ids,
                                          const std::vector<float>& grad_x) const {
        if (grad_x.size() != ids.size() * static_cast<size_t>(d_model)) {
            throw std::runtime_error("grad_x shape mismatch in backprop_embeddings");
        }

        EmbeddingBackward grads;
        grads.grad_token_emb.assign(static_cast<size_t>(vocab * d_model), 0.0f);
        grads.grad_pos_emb.assign(static_cast<size_t>(max_seq * d_model), 0.0f);

        for (size_t t = 0; t < ids.size(); ++t) {
            int tok = ids[t];
            if (tok < 0 || tok >= vocab) {
                throw std::runtime_error("Token id out of range in backprop_embeddings");
            }
            for (int d = 0; d < d_model; ++d) {
                float g = grad_x[t * static_cast<size_t>(d_model) + static_cast<size_t>(d)];
                grads.grad_token_emb[static_cast<size_t>(tok * d_model + d)] += g;
                grads.grad_pos_emb[static_cast<size_t>(t * d_model + d)] += g;
            }
        }

        return grads;
    }

    float evaluate_next_token_loss(const std::vector<int>& corpus_ids,
                                   int context_window = 24) const {
        if (corpus_ids.size() < 2) {
            throw std::runtime_error("Corpus must contain at least 2 tokens");
        }
        const int cw = std::max(2, std::min(context_window, max_seq));

        float total_loss = 0.0f;
        size_t total_targets = 0;

        for (size_t start = 0; start + 1 < corpus_ids.size(); ++start) {
            size_t end = std::min(start + static_cast<size_t>(cw), corpus_ids.size() - 1);
            if (end <= start) continue;

            std::vector<int> ctx(corpus_ids.begin() + static_cast<std::ptrdiff_t>(start),
                                 corpus_ids.begin() + static_cast<std::ptrdiff_t>(end));
            std::vector<int> tgt(corpus_ids.begin() + static_cast<std::ptrdiff_t>(start + 1),
                                 corpus_ids.begin() + static_cast<std::ptrdiff_t>(end + 1));

            ForwardCache cache = forward_cache(ctx);
            for (size_t p = 0; p < ctx.size(); ++p) {
                std::vector<float> row(static_cast<size_t>(vocab), 0.0f);
                for (int tok = 0; tok < vocab; ++tok) {
                    row[static_cast<size_t>(tok)] = cache.logits_all[static_cast<size_t>(p * vocab + tok)];
                }
                std::vector<float> probs;
                stable_softmax(row, probs);
                int y = tgt[p];
                total_loss += -std::log(std::max(probs[static_cast<size_t>(y)], 1e-8f));
                ++total_targets;
            }
        }

        return total_loss / static_cast<float>(std::max<size_t>(1, total_targets));
    }

    float train_next_token(const std::vector<int>& corpus_ids,
                           int epochs = 40,
                           int context_window = 24,
                           float lr = 0.04f,
                           int report_every = 10) {
        if (corpus_ids.size() < 2) {
            throw std::runtime_error("Corpus must contain at least 2 tokens");
        }
        const int cw = std::max(2, std::min(context_window, max_seq));

        float final_loss = 0.0f;
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            float total_loss = 0.0f;
            size_t total_targets = 0;

            for (size_t start = 0; start + 1 < corpus_ids.size(); ++start) {
                size_t end = std::min(start + static_cast<size_t>(cw), corpus_ids.size() - 1);
                if (end <= start) continue;

                std::vector<int> ctx(corpus_ids.begin() + static_cast<std::ptrdiff_t>(start),
                                     corpus_ids.begin() + static_cast<std::ptrdiff_t>(end));
                std::vector<int> tgt(corpus_ids.begin() + static_cast<std::ptrdiff_t>(start + 1),
                                     corpus_ids.begin() + static_cast<std::ptrdiff_t>(end + 1));

                ForwardCache cache = forward_cache(ctx);
                OutputHeadBackward grads = backprop_output_head(cache, tgt);
                total_loss += grads.loss * static_cast<float>(ctx.size());
                total_targets += ctx.size();

                for (size_t i = 0; i < out_bias.size(); ++i) {
                    out_bias[i] -= lr * grads.grad_out_bias[i];
                }
                for (size_t i = 0; i < out_proj.size(); ++i) {
                    out_proj[i] -= lr * grads.grad_out_proj[i];
                }

                // Conservative FFN update: this validates deeper gradient flow without destabilizing training.
                FFNBackward ffn_grads = backprop_ffn(cache, grads.grad_h2);
                const float ffn_lr = lr * 0.02f;
                for (size_t i = 0; i < w2.size(); ++i) {
                    w2[i] -= ffn_lr * ffn_grads.grad_w2[i];
                }
                for (size_t i = 0; i < w1.size(); ++i) {
                    w1[i] -= ffn_lr * ffn_grads.grad_w1[i];
                }

                // Attention output projection update (toward full attention backprop).
                AttentionProjBackward attn_grads = backprop_attention_proj(cache, ffn_grads.grad_h1);
                const float attn_lr = lr * 0.01f;
                for (size_t i = 0; i < wo.size(); ++i) {
                    wo[i] -= attn_lr * attn_grads.grad_wo[i];
                }

                AttentionCoreBackward attn_core_grads = backprop_attention_core(cache, attn_grads.grad_attn_out);
                const float qkv_lr = lr * 0.005f;
                for (size_t i = 0; i < wq.size(); ++i) {
                    wq[i] -= qkv_lr * attn_core_grads.grad_wq[i];
                }
                for (size_t i = 0; i < wk.size(); ++i) {
                    wk[i] -= qkv_lr * attn_core_grads.grad_wk[i];
                }
                for (size_t i = 0; i < wv.size(); ++i) {
                    wv[i] -= qkv_lr * attn_core_grads.grad_wv[i];
                }

                EmbeddingBackward emb_grads = backprop_embeddings(ctx, attn_core_grads.grad_x);
                const float emb_lr = lr * 0.002f;
                for (size_t i = 0; i < token_emb.size(); ++i) {
                    token_emb[i] -= emb_lr * emb_grads.grad_token_emb[i];
                }
                for (size_t i = 0; i < pos_emb.size(); ++i) {
                    pos_emb[i] -= emb_lr * emb_grads.grad_pos_emb[i];
                }
            }

            final_loss = total_loss / static_cast<float>(std::max<size_t>(1, total_targets));
            if (report_every > 0 && (epoch % report_every == 0 || epoch == 1 || epoch == epochs)) {
                std::cout << "epoch=" << epoch << " loss=" << final_loss << "\n";
            }
        }

        return final_loss;
    }

    std::vector<int> generate(const std::vector<int>& prompt,
                              int steps,
                              int context_window = 32,
                              float temperature = 1.0f,
                              int top_k = 0,
                              uint32_t seed = 2026) const {
        if (prompt.empty()) {
            throw std::runtime_error("Prompt must not be empty");
        }
        std::vector<int> ids = prompt;
        ids.reserve(prompt.size() + static_cast<size_t>(steps));
        std::mt19937 rng(seed);

        for (int s = 0; s < steps; ++s) {
            int start = 0;
            if (static_cast<int>(ids.size()) > context_window) {
                start = static_cast<int>(ids.size()) - context_window;
            }
            std::vector<int> ctx(ids.begin() + start, ids.end());
            std::vector<float> logits = forward(ctx);

            if (temperature > 0.0f) {
                for (float& x : logits) {
                    x /= temperature;
                }
            }

            if (top_k > 0 && top_k < vocab) {
                std::vector<int> idx(static_cast<size_t>(vocab), 0);
                for (int i = 0; i < vocab; ++i) idx[static_cast<size_t>(i)] = i;
                std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(), [&](int a, int b) {
                    return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)];
                });
                std::vector<char> keep(static_cast<size_t>(vocab), 0);
                for (int i = 0; i < top_k; ++i) keep[static_cast<size_t>(idx[static_cast<size_t>(i)])] = 1;
                for (int i = 0; i < vocab; ++i) {
                    if (!keep[static_cast<size_t>(i)]) {
                        logits[static_cast<size_t>(i)] = -std::numeric_limits<float>::infinity();
                    }
                }
            }

            int next_id = 0;
            if (temperature <= 0.0f || top_k == 1) {
                float best = logits[0];
                for (int i = 1; i < static_cast<int>(logits.size()); ++i) {
                    if (logits[static_cast<size_t>(i)] > best) {
                        best = logits[static_cast<size_t>(i)];
                        next_id = i;
                    }
                }
            } else {
                std::vector<float> probs;
                stable_softmax(logits, probs);
                std::discrete_distribution<int> dist(probs.begin(), probs.end());
                next_id = dist(rng);
            }
            ids.push_back(next_id);
        }

        return ids;
    }
};

#ifndef TOY_LLM_NO_MAIN
int main() {
    const std::string corpus =
        "hello world. this is a tiny transformer demo for character generation. "
        "hello world. this is a tiny transformer demo for character generation. ";

    CharTokenizer tok(corpus);
    TinyTransformer model(tok.vocab_size(), 24, 48, 64, 1234);

    std::vector<int> train_ids = tok.encode(corpus);
    float start_loss = model.evaluate_next_token_loss(train_ids, 24);
    float end_loss = model.train_next_token(train_ids, 40, 24, 0.04f, 10);

    std::string prompt = "hello ";
    std::vector<int> prompt_ids = tok.encode(prompt);
    std::vector<int> greedy_ids = model.generate(prompt_ids, 80, 32, 0.0f, 1, 7);
    std::vector<int> sampled_ids = model.generate(prompt_ids, 80, 32, 0.9f, 5, 7);

    std::cout << "Tiny Transformer milestone (single-head, one-block)\n";
    std::cout << "vocab_size=" << tok.vocab_size() << " d_model=24 heads=1 blocks=1\n";
    std::cout << "train_loss: start=" << start_loss << " end=" << end_loss << "\n";
    std::cout << "prompt:    " << prompt << "\n";
    std::cout << "greedy:    " << tok.decode(greedy_ids) << "\n";
    std::cout << "sampled:   " << tok.decode(sampled_ids) << "\n";

    return 0;
}
#endif
