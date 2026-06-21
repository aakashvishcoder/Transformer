#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
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
    int num_layers;
    bool use_layernorm;
    int num_heads;

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

    struct LayerCache {
        int seq = 0;                        // sequence length
        std::vector<float> x_in;         // [seq, d_model] input to this layer
        std::vector<float> q;            // [seq, d_model]
        std::vector<float> k;            // [seq, d_model]
        std::vector<float> v;            // [seq, d_model]
        std::vector<float> attn_out;     // [seq, d_model]
        std::vector<float> h1;           // [seq, d_model] (after attention + residual)
        std::vector<float> ff1_lin;      // [seq, d_ff]
        std::vector<float> ff1_act;      // [seq, d_ff]
        std::vector<float> h2;           // [seq, d_model] (after FFN + residual)
        std::vector<float> attn_weights; // [seq, seq]
        std::vector<float> attn_input;   // [seq, d_model] (pre-norm attention input)
        std::vector<float> ffn_input;    // [seq, d_model] (pre-norm FFN input)
    };

    struct ForwardCache {
        int seq = 0;
        std::vector<LayerCache> layers;  // [num_layers]
        std::vector<float> logits_all;   // [seq, vocab] final output logits
        
        // Convenience accessors for backward compat (single-layer access)
        const std::vector<float>& x() const { return layers.back().x_in; }
        const std::vector<float>& q() const { return layers.back().q; }
        const std::vector<float>& k() const { return layers.back().k; }
        const std::vector<float>& v() const { return layers.back().v; }
        const std::vector<float>& attn_out() const { return layers.back().attn_out; }
        const std::vector<float>& h1() const { return layers.back().h1; }
        const std::vector<float>& h2() const { return layers.back().h2; }
        const std::vector<float>& ff1_lin() const { return layers.back().ff1_lin; }
        const std::vector<float>& ff1_act() const { return layers.back().ff1_act; }
        const std::vector<float>& attn_weights() const { return layers.back().attn_weights; }
        const std::vector<float>& attn_input() const { return layers.back().attn_input; }
        const std::vector<float>& ffn_input() const { return layers.back().ffn_input; }
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
                                                         uint32_t seed = 1337,
                                                         int num_layers_ = 1,
                                                         bool use_layernorm_ = false,
                                                         int num_heads_ = 1)
        : vocab(vocab_size),
          d_model(d_model_),
          d_ff(d_ff_),
          max_seq(max_seq_),
                    num_layers(std::max(1, num_layers_)),
                    use_layernorm(use_layernorm_),
                    num_heads(std::max(1, num_heads_)),
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

    static void layernorm_rows(const std::vector<float>& in,
                               int rows,
                               int cols,
                               std::vector<float>& out,
                               float eps = 1e-5f) {
        out.assign(in.size(), 0.0f);
        for (int r = 0; r < rows; ++r) {
            float mean = 0.0f;
            for (int c = 0; c < cols; ++c) {
                mean += in[static_cast<size_t>(r * cols + c)];
            }
            mean /= static_cast<float>(cols);

            float var = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float d = in[static_cast<size_t>(r * cols + c)] - mean;
                var += d * d;
            }
            var /= static_cast<float>(cols);
            float inv_std = 1.0f / std::sqrt(var + eps);

            for (int c = 0; c < cols; ++c) {
                out[static_cast<size_t>(r * cols + c)] =
                    (in[static_cast<size_t>(r * cols + c)] - mean) * inv_std;
            }
        }
    }

    static void layernorm_rows_backward(const std::vector<float>& in,
                                        const std::vector<float>& grad_out,
                                        int rows,
                                        int cols,
                                        std::vector<float>& grad_in,
                                        float eps = 1e-5f) {
        grad_in.assign(in.size(), 0.0f);
        for (int r = 0; r < rows; ++r) {
            float mean = 0.0f;
            for (int c = 0; c < cols; ++c) {
                mean += in[static_cast<size_t>(r * cols + c)];
            }
            mean /= static_cast<float>(cols);

            float var = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float d = in[static_cast<size_t>(r * cols + c)] - mean;
                var += d * d;
            }
            var /= static_cast<float>(cols);
            float std_val = std::sqrt(var + eps);
            float inv_std = 1.0f / std_val;

            float sum_grad = 0.0f;
            float sum_grad_x = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float x_norm = (in[static_cast<size_t>(r * cols + c)] - mean) * inv_std;
                sum_grad += grad_out[static_cast<size_t>(r * cols + c)];
                sum_grad_x += grad_out[static_cast<size_t>(r * cols + c)] * x_norm;
            }

            float c_cols = static_cast<float>(cols);
            for (int c = 0; c < cols; ++c) {
                float x_norm = (in[static_cast<size_t>(r * cols + c)] - mean) * inv_std;
                float grad_x_norm = grad_out[static_cast<size_t>(r * cols + c)] -
                                    (sum_grad / c_cols) -
                                    x_norm * (sum_grad_x / c_cols);
                grad_in[static_cast<size_t>(r * cols + c)] = grad_x_norm * inv_std;
            }
        }
    }

    template <typename T>
    static bool write_pod(std::ofstream& out, const T& value) {
        out.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
        return out.good();
    }

    template <typename T>
    static bool read_pod(std::ifstream& in, T& value) {
        in.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
        return in.good();
    }

    static bool write_vector(std::ofstream& out, const std::vector<float>& v) {
        uint64_t sz = static_cast<uint64_t>(v.size());
        if (!write_pod(out, sz)) return false;
        if (sz == 0) return true;
        out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(sizeof(float) * v.size()));
        return out.good();
    }

    static bool read_vector(std::ifstream& in, std::vector<float>& v, size_t expected_size) {
        uint64_t sz = 0;
        if (!read_pod(in, sz)) return false;
        if (sz != static_cast<uint64_t>(expected_size)) return false;
        v.resize(expected_size);
        if (expected_size == 0) return true;
        in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(sizeof(float) * v.size()));
        return in.good();
    }

    bool save_checkpoint(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out.is_open()) return false;

        const char magic[8] = {'T', 'L', 'L', 'M', 'C', 'K', '1', '\0'};
        out.write(magic, 8);
        if (!out.good()) return false;

        int32_t v_vocab = vocab;
        int32_t v_d_model = d_model;
        int32_t v_d_ff = d_ff;
        int32_t v_max_seq = max_seq;
        int32_t v_num_layers = num_layers;
        int32_t v_use_layernorm = use_layernorm ? 1 : 0;
        int32_t v_num_heads = num_heads;
        if (!write_pod(out, v_vocab) || !write_pod(out, v_d_model) ||
            !write_pod(out, v_d_ff) || !write_pod(out, v_max_seq) ||
            !write_pod(out, v_num_layers) || !write_pod(out, v_use_layernorm) ||
            !write_pod(out, v_num_heads)) {
            return false;
        }

        return write_vector(out, token_emb) &&
               write_vector(out, pos_emb) &&
               write_vector(out, wq) &&
               write_vector(out, wk) &&
               write_vector(out, wv) &&
               write_vector(out, wo) &&
               write_vector(out, w1) &&
               write_vector(out, w2) &&
               write_vector(out, out_proj) &&
               write_vector(out, out_bias);
    }

    bool load_checkpoint(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) return false;

        char magic[8] = {0};
        in.read(magic, 8);
        if (!in.good()) return false;
        const char expected[8] = {'T', 'L', 'L', 'M', 'C', 'K', '1', '\0'};
        for (int i = 0; i < 8; ++i) {
            if (magic[i] != expected[i]) return false;
        }

        int32_t v_vocab = 0;
        int32_t v_d_model = 0;
        int32_t v_d_ff = 0;
        int32_t v_max_seq = 0;
        int32_t v_num_layers = 0;
        int32_t v_use_layernorm = 0;
        int32_t v_num_heads = 0;
        if (!read_pod(in, v_vocab) || !read_pod(in, v_d_model) ||
            !read_pod(in, v_d_ff) || !read_pod(in, v_max_seq) ||
            !read_pod(in, v_num_layers) || !read_pod(in, v_use_layernorm) ||
            !read_pod(in, v_num_heads)) {
            return false;
        }

        if (v_vocab != vocab || v_d_model != d_model || v_d_ff != d_ff ||
            v_max_seq != max_seq || v_num_layers != num_layers ||
            v_use_layernorm != (use_layernorm ? 1 : 0) ||
            v_num_heads != num_heads) {
            return false;
        }

        return read_vector(in, token_emb, token_emb.size()) &&
               read_vector(in, pos_emb, pos_emb.size()) &&
               read_vector(in, wq, wq.size()) &&
               read_vector(in, wk, wk.size()) &&
               read_vector(in, wv, wv.size()) &&
               read_vector(in, wo, wo.size()) &&
               read_vector(in, w1, w1.size()) &&
               read_vector(in, w2, w2.size()) &&
               read_vector(in, out_proj, out_proj.size()) &&
               read_vector(in, out_bias, out_bias.size());
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
        cache.layers.resize(static_cast<size_t>(num_layers));

        // 1) Embedding + positional encoding. X: [seq, d_model]
        std::vector<float> x_cur(static_cast<size_t>(seq * d_model), 0.0f);
        for (int t = 0; t < seq; ++t) {
            int tok = ids[static_cast<size_t>(t)];
            for (int i = 0; i < d_model; ++i) {
                x_cur[static_cast<size_t>(t * d_model + i)] =
                    token_emb[static_cast<size_t>(tok * d_model + i)] +
                    pos_emb[static_cast<size_t>(t * d_model + i)];
            }
        }

        if (d_model % num_heads != 0) {
            throw std::runtime_error("d_model must be divisible by num_heads");
        }
        const int head_dim = d_model / num_heads;

        for (int layer = 0; layer < num_layers; ++layer) {
            LayerCache& layer_cache = cache.layers[static_cast<size_t>(layer)];
            layer_cache.seq = seq;
            layer_cache.x_in = x_cur;

            std::vector<float> attn_input;
            if (use_layernorm) {
                layernorm_rows(x_cur, seq, d_model, attn_input);
            } else {
                attn_input = x_cur;
            }
            layer_cache.attn_input = attn_input;

            layer_cache.q = matmul(attn_input, seq, d_model, wq, d_model); // [seq, d]
            layer_cache.k = matmul(attn_input, seq, d_model, wk, d_model); // [seq, d]
            layer_cache.v = matmul(attn_input, seq, d_model, wv, d_model); // [seq, d]

            layer_cache.attn_out.assign(static_cast<size_t>(seq * d_model), 0.0f);
            layer_cache.attn_weights.assign(static_cast<size_t>(seq * seq), 0.0f);

            for (int h = 0; h < num_heads; ++h) {
                const int d0 = h * head_dim;
                for (int t = 0; t < seq; ++t) {
                    std::vector<float> scores(static_cast<size_t>(t + 1), 0.0f);
                    float max_score = -1e30f;
                    for (int j = 0; j <= t; ++j) {
                        float s = 0.0f;
                        for (int d = 0; d < head_dim; ++d) {
                            s += layer_cache.q[static_cast<size_t>(t * d_model + d0 + d)] *
                                 layer_cache.k[static_cast<size_t>(j * d_model + d0 + d)];
                        }
                        s *= (1.0f / std::sqrt(static_cast<float>(head_dim)));
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
                        layer_cache.attn_weights[static_cast<size_t>(t * seq + j)] += a / static_cast<float>(num_heads);
                        for (int d = 0; d < head_dim; ++d) {
                            layer_cache.attn_out[static_cast<size_t>(t * d_model + d0 + d)] +=
                                a * layer_cache.v[static_cast<size_t>(j * d_model + d0 + d)];
                        }
                    }
                }
            }

            std::vector<float> attn_proj = matmul(layer_cache.attn_out, seq, d_model, wo, d_model);

            // Residual 1.
            layer_cache.h1.assign(static_cast<size_t>(seq * d_model), 0.0f);
            for (size_t i = 0; i < layer_cache.h1.size(); ++i) {
                layer_cache.h1[i] = x_cur[i] + attn_proj[i];
            }

            std::vector<float> ffn_input;
            if (use_layernorm) {
                layernorm_rows(layer_cache.h1, seq, d_model, ffn_input);
            } else {
                ffn_input = layer_cache.h1;
            }
            layer_cache.ffn_input = ffn_input;

            layer_cache.ff1_lin = matmul(ffn_input, seq, d_model, w1, d_ff);
            layer_cache.ff1_act = layer_cache.ff1_lin;
            for (float& z : layer_cache.ff1_act) {
                z = std::max(0.0f, z);
            }
            std::vector<float> ff2 = matmul(layer_cache.ff1_act, seq, d_ff, w2, d_model);

            layer_cache.h2.assign(static_cast<size_t>(seq * d_model), 0.0f);
            for (size_t i = 0; i < layer_cache.h2.size(); ++i) {
                layer_cache.h2[i] = layer_cache.h1[i] + ff2[i];
            }

            x_cur = layer_cache.h2;
        }

        // 4) Output logits for all positions.
        cache.logits_all.assign(static_cast<size_t>(seq * vocab), 0.0f);
        std::vector<float>& final_h2 = cache.layers.back().h2;
        for (int t = 0; t < seq; ++t) {
            for (int tok = 0; tok < vocab; ++tok) {
                float s = out_bias[static_cast<size_t>(tok)];
                for (int d = 0; d < d_model; ++d) {
                    s += final_h2[static_cast<size_t>(t * d_model + d)] *
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
                        cache.h2()[static_cast<size_t>(p * d_model + d)] * g;
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
        if (grad_h2.size() != cache.h2().size()) {
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
                    acc += cache.ff1_act()[static_cast<size_t>(t * d_ff + f)] *
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
            grad_ff1_lin[i] = cache.ff1_lin()[i] > 0.0f ? grad_ff1_act[i] : 0.0f;
        }

        // ff1_lin = h1 @ w1 (or ffn_input @ w1 if LayerNorm is used).
        std::vector<float> grad_ffn_input(static_cast<size_t>(cache.seq * d_model), 0.0f);
        for (int d = 0; d < d_model; ++d) {
            for (int f = 0; f < d_ff; ++f) {
                float acc = 0.0f;
                for (int t = 0; t < cache.seq; ++t) {
                    acc += cache.ffn_input()[static_cast<size_t>(t * d_model + d)] *
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
                grad_ffn_input[static_cast<size_t>(t * d_model + d)] = acc;
            }
        }

        // Apply LayerNorm backward if needed.
        if (use_layernorm) {
            std::vector<float> grad_h1_ln;
            layernorm_rows_backward(cache.h1(), grad_ffn_input, cache.seq, d_model, grad_h1_ln);
            for (size_t i = 0; i < grad_h1_from_ffn.size(); ++i) {
                grad_h1_from_ffn[i] = grad_h1_ln[i];
            }
        } else {
            grad_h1_from_ffn = grad_ffn_input;
        }

        for (size_t i = 0; i < grads.grad_h1.size(); ++i) {
            grads.grad_h1[i] += grad_h1_from_ffn[i];
        }

        return grads;
    }

    AttentionProjBackward backprop_attention_proj(const ForwardCache& cache,
                                                  const std::vector<float>& grad_h1) const {
        if (grad_h1.size() != cache.h1().size()) {
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
                    acc += cache.attn_out()[static_cast<size_t>(t * d_model + i)] *
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
        if (grad_attn_out.size() != cache.attn_out().size()) {
            throw std::runtime_error("grad_attn_out shape mismatch in backprop_attention_core");
        }
        if (d_model % num_heads != 0) {
            throw std::runtime_error("d_model must be divisible by num_heads");
        }

        AttentionCoreBackward grads;
        grads.grad_wq.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_wk.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_wv.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_x.assign(static_cast<size_t>(cache.seq * d_model), 0.0f);

        std::vector<float> grad_q(static_cast<size_t>(cache.seq * d_model), 0.0f);
        std::vector<float> grad_k(static_cast<size_t>(cache.seq * d_model), 0.0f);
        std::vector<float> grad_v(static_cast<size_t>(cache.seq * d_model), 0.0f);
        const int head_dim = d_model / num_heads;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Backprop attention independently per head.
        for (int h = 0; h < num_heads; ++h) {
            const int d0 = h * head_dim;
            for (int t = 0; t < cache.seq; ++t) {
                std::vector<float> scores(static_cast<size_t>(t + 1), 0.0f);
                float max_score = -1e30f;
                for (int j = 0; j <= t; ++j) {
                    float s = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        s += cache.q()[static_cast<size_t>(t * d_model + d0 + d)] *
                             cache.k()[static_cast<size_t>(j * d_model + d0 + d)];
                    }
                    s *= scale;
                    scores[static_cast<size_t>(j)] = s;
                    max_score = std::max(max_score, s);
                }

                float exp_sum = 0.0f;
                std::vector<float> a(static_cast<size_t>(t + 1), 0.0f);
                for (int j = 0; j <= t; ++j) {
                    float e = std::exp(scores[static_cast<size_t>(j)] - max_score);
                    a[static_cast<size_t>(j)] = e;
                    exp_sum += e;
                }
                for (int j = 0; j <= t; ++j) {
                    a[static_cast<size_t>(j)] /= exp_sum;
                }

                std::vector<float> grad_a(static_cast<size_t>(t + 1), 0.0f);
                for (int j = 0; j <= t; ++j) {
                    float acc = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        acc += grad_attn_out[static_cast<size_t>(t * d_model + d0 + d)] *
                               cache.v()[static_cast<size_t>(j * d_model + d0 + d)];
                        grad_v[static_cast<size_t>(j * d_model + d0 + d)] +=
                            a[static_cast<size_t>(j)] *
                            grad_attn_out[static_cast<size_t>(t * d_model + d0 + d)];
                    }
                    grad_a[static_cast<size_t>(j)] = acc;
                }

                float dot_ga_a = 0.0f;
                for (int j = 0; j <= t; ++j) {
                    dot_ga_a += grad_a[static_cast<size_t>(j)] * a[static_cast<size_t>(j)];
                }

                for (int j = 0; j <= t; ++j) {
                    float grad_s = a[static_cast<size_t>(j)] * (grad_a[static_cast<size_t>(j)] - dot_ga_a);
                    for (int d = 0; d < head_dim; ++d) {
                        grad_q[static_cast<size_t>(t * d_model + d0 + d)] +=
                            grad_s * scale * cache.k()[static_cast<size_t>(j * d_model + d0 + d)];
                        grad_k[static_cast<size_t>(j * d_model + d0 + d)] +=
                            grad_s * scale * cache.q()[static_cast<size_t>(t * d_model + d0 + d)];
                    }
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
                    gq += cache.attn_input()[static_cast<size_t>(t * d_model + i)] *
                          grad_q[static_cast<size_t>(t * d_model + o)];
                    gk += cache.attn_input()[static_cast<size_t>(t * d_model + i)] *
                          grad_k[static_cast<size_t>(t * d_model + o)];
                    gv += cache.attn_input()[static_cast<size_t>(t * d_model + i)] *
                          grad_v[static_cast<size_t>(t * d_model + o)];
                }
                grads.grad_wq[static_cast<size_t>(i * d_model + o)] = gq;
                grads.grad_wk[static_cast<size_t>(i * d_model + o)] = gk;
                grads.grad_wv[static_cast<size_t>(i * d_model + o)] = gv;
            }
        }

        std::vector<float> grad_attn_input(static_cast<size_t>(cache.seq * d_model), 0.0f);
        for (int t = 0; t < cache.seq; ++t) {
            for (int i = 0; i < d_model; ++i) {
                float acc = 0.0f;
                for (int o = 0; o < d_model; ++o) {
                    acc += grad_q[static_cast<size_t>(t * d_model + o)] * wq[static_cast<size_t>(i * d_model + o)];
                    acc += grad_k[static_cast<size_t>(t * d_model + o)] * wk[static_cast<size_t>(i * d_model + o)];
                    acc += grad_v[static_cast<size_t>(t * d_model + o)] * wv[static_cast<size_t>(i * d_model + o)];
                }
                grad_attn_input[static_cast<size_t>(t * d_model + i)] = acc;
            }
        }

        // Apply LayerNorm backward if needed.
        if (use_layernorm) {
            layernorm_rows_backward(cache.x(), grad_attn_input, cache.seq, d_model, grads.grad_x);
        } else {
            grads.grad_x = grad_attn_input;
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

    // Internal backprop functions that work with LayerCache directly (for multi-layer support)
    FFNBackward backprop_ffn_internal(const LayerCache& layer_cache,
                                      const std::vector<float>& grad_h2) const {
        if (grad_h2.size() != layer_cache.h2.size()) {
            throw std::runtime_error("grad_h2 shape mismatch in backprop_ffn_internal");
        }

        FFNBackward grads;
        grads.grad_w2.assign(static_cast<size_t>(d_ff * d_model), 0.0f);
        grads.grad_w1.assign(static_cast<size_t>(d_model * d_ff), 0.0f);
        grads.grad_h1.assign(static_cast<size_t>(layer_cache.seq * d_model), 0.0f);

        std::vector<float> grad_ff2 = grad_h2;
        for (size_t i = 0; i < grads.grad_h1.size(); ++i) {
            grads.grad_h1[i] = grad_h2[i];
        }

        // ff2 = ff1_act @ w2
        for (int f = 0; f < d_ff; ++f) {
            for (int d = 0; d < d_model; ++d) {
                float acc = 0.0f;
                for (int t = 0; t < layer_cache.seq; ++t) {
                    acc += layer_cache.ff1_act[static_cast<size_t>(t * d_ff + f)] *
                           grad_ff2[static_cast<size_t>(t * d_model + d)];
                }
                grads.grad_w2[static_cast<size_t>(f * d_model + d)] = acc;
            }
        }

        std::vector<float> grad_ff1_act(static_cast<size_t>(layer_cache.seq * d_ff), 0.0f);
        for (int t = 0; t < layer_cache.seq; ++t) {
            for (int f = 0; f < d_ff; ++f) {
                float acc = 0.0f;
                for (int d = 0; d < d_model; ++d) {
                    acc += grad_ff2[static_cast<size_t>(t * d_model + d)] *
                           w2[static_cast<size_t>(f * d_model + d)];
                }
                grad_ff1_act[static_cast<size_t>(t * d_ff + f)] = acc;
            }
        }

        std::vector<float> grad_ff1_lin(static_cast<size_t>(layer_cache.seq * d_ff), 0.0f);
        for (size_t i = 0; i < grad_ff1_lin.size(); ++i) {
            grad_ff1_lin[i] = layer_cache.ff1_lin[i] > 0.0f ? grad_ff1_act[i] : 0.0f;
        }

        // ff1_lin = ffn_input @ w1 (ffn_input already has LayerNorm applied if needed)
        for (int d = 0; d < d_model; ++d) {
            for (int f = 0; f < d_ff; ++f) {
                float acc = 0.0f;
                for (int t = 0; t < layer_cache.seq; ++t) {
                    acc += layer_cache.ffn_input[static_cast<size_t>(t * d_model + d)] *
                           grad_ff1_lin[static_cast<size_t>(t * d_ff + f)];
                }
                grads.grad_w1[static_cast<size_t>(d * d_ff + f)] = acc;
            }
        }

        std::vector<float> grad_h1_from_ffn(static_cast<size_t>(layer_cache.seq * d_model), 0.0f);
        for (int t = 0; t < layer_cache.seq; ++t) {
            for (int d = 0; d < d_model; ++d) {
                float acc = 0.0f;
                for (int f = 0; f < d_ff; ++f) {
                    acc += grad_ff1_lin[static_cast<size_t>(t * d_ff + f)] *
                           w1[static_cast<size_t>(d * d_ff + f)];
                }
                grad_h1_from_ffn[static_cast<size_t>(t * d_model + d)] = acc;
            }
        }

        // Apply LayerNorm backward if needed
        if (use_layernorm) {
            std::vector<float> grad_h1_ln;
            layernorm_rows_backward(layer_cache.h1, grad_h1_from_ffn, layer_cache.seq, d_model, grad_h1_ln);
            for (size_t i = 0; i < grad_h1_from_ffn.size(); ++i) {
                grad_h1_from_ffn[i] = grad_h1_ln[i];
            }
        }

        for (size_t i = 0; i < grads.grad_h1.size(); ++i) {
            grads.grad_h1[i] += grad_h1_from_ffn[i];
        }

        return grads;
    }

    AttentionProjBackward backprop_attention_proj_internal(const LayerCache& layer_cache,
                                                           const std::vector<float>& grad_h1) const {
        if (grad_h1.size() != layer_cache.h1.size()) {
            throw std::runtime_error("grad_h1 shape mismatch in backprop_attention_proj_internal");
        }

        AttentionProjBackward grads;
        grads.grad_wo.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_attn_out.assign(static_cast<size_t>(layer_cache.seq * d_model), 0.0f);

        // h1 = x + attn_proj and attn_proj = attn_out @ wo
        for (int i = 0; i < d_model; ++i) {
            for (int j = 0; j < d_model; ++j) {
                float acc = 0.0f;
                for (int t = 0; t < layer_cache.seq; ++t) {
                    acc += layer_cache.attn_out[static_cast<size_t>(t * d_model + i)] *
                           grad_h1[static_cast<size_t>(t * d_model + j)];
                }
                grads.grad_wo[static_cast<size_t>(i * d_model + j)] = acc;
            }
        }

        for (int t = 0; t < layer_cache.seq; ++t) {
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

    AttentionCoreBackward backprop_attention_core_internal(const LayerCache& layer_cache,
                                                           const std::vector<float>& grad_attn_out) const {
        if (grad_attn_out.size() != layer_cache.attn_out.size()) {
            throw std::runtime_error("grad_attn_out shape mismatch in backprop_attention_core_internal");
        }
        if (d_model % num_heads != 0) {
            throw std::runtime_error("d_model must be divisible by num_heads");
        }

        AttentionCoreBackward grads;
        grads.grad_wq.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_wk.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_wv.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_x.assign(static_cast<size_t>(layer_cache.seq * d_model), 0.0f);

        std::vector<float> grad_q(static_cast<size_t>(layer_cache.seq * d_model), 0.0f);
        std::vector<float> grad_k(static_cast<size_t>(layer_cache.seq * d_model), 0.0f);
        std::vector<float> grad_v(static_cast<size_t>(layer_cache.seq * d_model), 0.0f);
        const int head_dim = d_model / num_heads;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Backprop attention per head
        for (int h = 0; h < num_heads; ++h) {
            const int d0 = h * head_dim;
            for (int t = 0; t < layer_cache.seq; ++t) {
                std::vector<float> scores(static_cast<size_t>(t + 1), 0.0f);
                float max_score = -1e30f;
                for (int j = 0; j <= t; ++j) {
                    float s = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        s += layer_cache.q[static_cast<size_t>(t * d_model + d0 + d)] *
                             layer_cache.k[static_cast<size_t>(j * d_model + d0 + d)];
                    }
                    s *= scale;
                    scores[static_cast<size_t>(j)] = s;
                    max_score = std::max(max_score, s);
                }

                float exp_sum = 0.0f;
                std::vector<float> a(static_cast<size_t>(t + 1), 0.0f);
                for (int j = 0; j <= t; ++j) {
                    float e = std::exp(scores[static_cast<size_t>(j)] - max_score);
                    a[static_cast<size_t>(j)] = e;
                    exp_sum += e;
                }
                for (int j = 0; j <= t; ++j) {
                    a[static_cast<size_t>(j)] /= exp_sum;
                }

                std::vector<float> grad_a(static_cast<size_t>(t + 1), 0.0f);
                for (int j = 0; j <= t; ++j) {
                    float acc = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        acc += grad_attn_out[static_cast<size_t>(t * d_model + d0 + d)] *
                               layer_cache.v[static_cast<size_t>(j * d_model + d0 + d)];
                        grad_v[static_cast<size_t>(j * d_model + d0 + d)] +=
                            a[static_cast<size_t>(j)] *
                            grad_attn_out[static_cast<size_t>(t * d_model + d0 + d)];
                    }
                    grad_a[static_cast<size_t>(j)] = acc;
                }

                float dot_ga_a = 0.0f;
                for (int j = 0; j <= t; ++j) {
                    dot_ga_a += grad_a[static_cast<size_t>(j)] * a[static_cast<size_t>(j)];
                }

                for (int j = 0; j <= t; ++j) {
                    float grad_s = a[static_cast<size_t>(j)] * (grad_a[static_cast<size_t>(j)] - dot_ga_a);
                    for (int d = 0; d < head_dim; ++d) {
                        grad_q[static_cast<size_t>(t * d_model + d0 + d)] +=
                            grad_s * scale * layer_cache.k[static_cast<size_t>(j * d_model + d0 + d)];
                        grad_k[static_cast<size_t>(j * d_model + d0 + d)] +=
                            grad_s * scale * layer_cache.q[static_cast<size_t>(t * d_model + d0 + d)];
                    }
                }
            }
        }

        // q = x @ wq, k = x @ wk, v = x @ wv
        for (int i = 0; i < d_model; ++i) {
            for (int o = 0; o < d_model; ++o) {
                float gq = 0.0f;
                float gk = 0.0f;
                float gv = 0.0f;
                for (int t = 0; t < layer_cache.seq; ++t) {
                    gq += layer_cache.attn_input[static_cast<size_t>(t * d_model + i)] *
                          grad_q[static_cast<size_t>(t * d_model + o)];
                    gk += layer_cache.attn_input[static_cast<size_t>(t * d_model + i)] *
                          grad_k[static_cast<size_t>(t * d_model + o)];
                    gv += layer_cache.attn_input[static_cast<size_t>(t * d_model + i)] *
                          grad_v[static_cast<size_t>(t * d_model + o)];
                }
                grads.grad_wq[static_cast<size_t>(i * d_model + o)] = gq;
                grads.grad_wk[static_cast<size_t>(i * d_model + o)] = gk;
                grads.grad_wv[static_cast<size_t>(i * d_model + o)] = gv;
            }
        }

        std::vector<float> grad_attn_input(static_cast<size_t>(layer_cache.seq * d_model), 0.0f);
        for (int t = 0; t < layer_cache.seq; ++t) {
            for (int i = 0; i < d_model; ++i) {
                float acc = 0.0f;
                for (int o = 0; o < d_model; ++o) {
                    acc += grad_q[static_cast<size_t>(t * d_model + o)] * wq[static_cast<size_t>(i * d_model + o)];
                    acc += grad_k[static_cast<size_t>(t * d_model + o)] * wk[static_cast<size_t>(i * d_model + o)];
                    acc += grad_v[static_cast<size_t>(t * d_model + o)] * wv[static_cast<size_t>(i * d_model + o)];
                }
                grad_attn_input[static_cast<size_t>(t * d_model + i)] = acc;
            }
        }

        // Apply LayerNorm backward if needed (attn_input = LN(x_in))
        if (use_layernorm) {
            layernorm_rows_backward(layer_cache.x_in, grad_attn_input, layer_cache.seq, d_model, grads.grad_x);
        } else {
            grads.grad_x = grad_attn_input;
        }

        return grads;
    }

    EmbeddingBackward backprop_embeddings_internal(const std::vector<int>& ids,
                                                   const std::vector<float>& grad_x) const {
        if (grad_x.size() != ids.size() * static_cast<size_t>(d_model)) {
            throw std::runtime_error("grad_x shape mismatch in backprop_embeddings_internal");
        }

        EmbeddingBackward grads;
        grads.grad_token_emb.assign(static_cast<size_t>(vocab * d_model), 0.0f);
        grads.grad_pos_emb.assign(static_cast<size_t>(max_seq * d_model), 0.0f);

        for (size_t t = 0; t < ids.size(); ++t) {
            int tok = ids[t];
            if (tok < 0 || tok >= vocab) {
                throw std::runtime_error("Token id out of range in backprop_embeddings_internal");
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
                           int report_every = 10,
                           const std::string& optimizer = "sgd",
                           int warmup_steps = 0,
                           float min_lr_ratio = 0.1f,
                           float weight_decay = 0.0f,
                           int batch_size = 1,
                           float grad_clip_norm = 0.0f,
                           bool report_lr = false,
                           bool report_grad_norm = false) {
        if (batch_size <= 0) {
            throw std::runtime_error("batch_size must be >= 1");
        }
        if (corpus_ids.size() < 2) {
            throw std::runtime_error("Corpus must contain at least 2 tokens");
        }
        const int cw = std::max(2, std::min(context_window, max_seq));

        struct AdamState {
            std::vector<float> m;
            std::vector<float> v;
        };

        AdamState st_out_bias, st_out_proj, st_w1, st_w2, st_wo, st_wq, st_wk, st_wv, st_tok, st_pos;

        auto ensure_state = [](AdamState& st, size_t n) {
            if (st.m.size() != n) {
                st.m.assign(n, 0.0f);
                st.v.assign(n, 0.0f);
            }
        };

        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float eps = 1e-8f;
        int global_step = 0;

        int updates_per_epoch = 0;
        for (size_t start = 0; start + 1 < corpus_ids.size(); ++start) {
            size_t end = std::min(start + static_cast<size_t>(cw), corpus_ids.size() - 1);
            if (end > start) ++updates_per_epoch;
        }
        updates_per_epoch = std::max(1, updates_per_epoch);
        const int steps_per_epoch = std::max(1, (updates_per_epoch + batch_size - 1) / batch_size);
        const int total_steps = std::max(1, epochs * steps_per_epoch);

        auto scheduled_lr = [&](int step) {
            if (warmup_steps > 0 && step <= warmup_steps) {
                return lr * (static_cast<float>(step) / static_cast<float>(warmup_steps));
            }
            int denom = std::max(1, total_steps - warmup_steps);
            float p = static_cast<float>(step - warmup_steps) / static_cast<float>(denom);
            p = std::min(1.0f, std::max(0.0f, p));
            float c = 0.5f * (1.0f + std::cos(3.1415926535f * p));
            return lr * (min_lr_ratio + (1.0f - min_lr_ratio) * c);
        };

        auto apply_update = [&](std::vector<float>& param,
                                const std::vector<float>& grad,
                                AdamState& st,
                                float lr_scale,
                                int step) {
            if (optimizer == "adamw") {
                ensure_state(st, param.size());
                float lr_eff = scheduled_lr(step) * lr_scale;
                float b1_corr = 1.0f - std::pow(beta1, static_cast<float>(step));
                float b2_corr = 1.0f - std::pow(beta2, static_cast<float>(step));
                for (size_t i = 0; i < param.size(); ++i) {
                    float g = grad[i] + weight_decay * param[i];
                    st.m[i] = beta1 * st.m[i] + (1.0f - beta1) * g;
                    st.v[i] = beta2 * st.v[i] + (1.0f - beta2) * g * g;
                    float m_hat = st.m[i] / b1_corr;
                    float v_hat = st.v[i] / b2_corr;
                    param[i] -= lr_eff * (m_hat / (std::sqrt(v_hat) + eps));
                }
            } else {
                float lr_eff = scheduled_lr(step) * lr_scale;
                for (size_t i = 0; i < param.size(); ++i) {
                    float g = grad[i] + weight_decay * param[i];
                    param[i] -= lr_eff * g;
                }
            }
        };

        std::vector<float> acc_out_bias(out_bias.size(), 0.0f);
        std::vector<float> acc_out_proj(out_proj.size(), 0.0f);
        std::vector<float> acc_w2(w2.size(), 0.0f);
        std::vector<float> acc_w1(w1.size(), 0.0f);
        std::vector<float> acc_wo(wo.size(), 0.0f);
        std::vector<float> acc_wq(wq.size(), 0.0f);
        std::vector<float> acc_wk(wk.size(), 0.0f);
        std::vector<float> acc_wv(wv.size(), 0.0f);
        std::vector<float> acc_tok(token_emb.size(), 0.0f);
        std::vector<float> acc_pos(pos_emb.size(), 0.0f);

        auto zero_vec = [](std::vector<float>& v) {
            std::fill(v.begin(), v.end(), 0.0f);
        };
        auto add_inplace = [](std::vector<float>& dst, const std::vector<float>& src) {
            for (size_t i = 0; i < dst.size(); ++i) {
                dst[i] += src[i];
            }
        };
        auto scale_inplace = [](std::vector<float>& v, float s) {
            for (float& x : v) {
                x *= s;
            }
        };

        auto clear_accumulators = [&]() {
            zero_vec(acc_out_bias);
            zero_vec(acc_out_proj);
            zero_vec(acc_w2);
            zero_vec(acc_w1);
            zero_vec(acc_wo);
            zero_vec(acc_wq);
            zero_vec(acc_wk);
            zero_vec(acc_wv);
            zero_vec(acc_tok);
            zero_vec(acc_pos);
        };

        auto global_grad_norm = [&]() {
            double sum_sq = 0.0;
            auto add_sq = [&](const std::vector<float>& v) {
                for (float x : v) {
                    sum_sq += static_cast<double>(x) * static_cast<double>(x);
                }
            };
            add_sq(acc_out_bias);
            add_sq(acc_out_proj);
            add_sq(acc_w2);
            add_sq(acc_w1);
            add_sq(acc_wo);
            add_sq(acc_wq);
            add_sq(acc_wk);
            add_sq(acc_wv);
            add_sq(acc_tok);
            add_sq(acc_pos);
            return static_cast<float>(std::sqrt(sum_sq));
        };

        auto apply_clipping_if_needed = [&]() {
            if (grad_clip_norm <= 0.0f) return;
            float norm = global_grad_norm();
            if (norm <= grad_clip_norm || norm <= 0.0f) return;

            float s = grad_clip_norm / norm;
            scale_inplace(acc_out_bias, s);
            scale_inplace(acc_out_proj, s);
            scale_inplace(acc_w2, s);
            scale_inplace(acc_w1, s);
            scale_inplace(acc_wo, s);
            scale_inplace(acc_wq, s);
            scale_inplace(acc_wk, s);
            scale_inplace(acc_wv, s);
            scale_inplace(acc_tok, s);
            scale_inplace(acc_pos, s);
        };

        float final_loss = 0.0f;
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            float total_loss = 0.0f;
            size_t total_targets = 0;
            int batch_count = 0;
            float epoch_lr_sum = 0.0f;
            float epoch_grad_norm_sum = 0.0f;
            int epoch_update_steps = 0;

            clear_accumulators();

            auto flush_batch = [&]() {
                if (batch_count <= 0) return;

                float inv_bs = 1.0f / static_cast<float>(batch_count);
                scale_inplace(acc_out_bias, inv_bs);
                scale_inplace(acc_out_proj, inv_bs);
                scale_inplace(acc_w2, inv_bs);
                scale_inplace(acc_w1, inv_bs);
                scale_inplace(acc_wo, inv_bs);
                scale_inplace(acc_wq, inv_bs);
                scale_inplace(acc_wk, inv_bs);
                scale_inplace(acc_wv, inv_bs);
                scale_inplace(acc_tok, inv_bs);
                scale_inplace(acc_pos, inv_bs);

                float grad_norm_before_clip = global_grad_norm();

                apply_clipping_if_needed();

                ++global_step;
                float lr_step = scheduled_lr(global_step);
                epoch_lr_sum += lr_step;
                epoch_grad_norm_sum += grad_norm_before_clip;
                ++epoch_update_steps;

                apply_update(out_bias, acc_out_bias, st_out_bias, 1.0f, global_step);
                apply_update(out_proj, acc_out_proj, st_out_proj, 1.0f, global_step);
                apply_update(w2, acc_w2, st_w2, 0.02f, global_step);
                apply_update(w1, acc_w1, st_w1, 0.02f, global_step);
                apply_update(wo, acc_wo, st_wo, 0.01f, global_step);
                apply_update(wq, acc_wq, st_wq, 0.005f, global_step);
                apply_update(wk, acc_wk, st_wk, 0.005f, global_step);
                apply_update(wv, acc_wv, st_wv, 0.005f, global_step);
                apply_update(token_emb, acc_tok, st_tok, 0.002f, global_step);
                apply_update(pos_emb, acc_pos, st_pos, 0.002f, global_step);

                batch_count = 0;
                clear_accumulators();
            };

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
                add_inplace(acc_out_bias, grads.grad_out_bias);
                add_inplace(acc_out_proj, grads.grad_out_proj);

                // Multi-layer backprop: process layers in reverse
                std::vector<float> grad_h2_cur = grads.grad_h2;
                
                for (int layer = num_layers - 1; layer >= 0; --layer) {
                    LayerCache& layer_cache = cache.layers[static_cast<size_t>(layer)];
                    
                    // FFN backward for this layer
                    FFNBackward ffn_grads = backprop_ffn_internal(layer_cache, grad_h2_cur);
                    add_inplace(acc_w2, ffn_grads.grad_w2);
                    add_inplace(acc_w1, ffn_grads.grad_w1);

                    // Attention projection backward for this layer
                    AttentionProjBackward attn_grads = backprop_attention_proj_internal(layer_cache, ffn_grads.grad_h1);
                    add_inplace(acc_wo, attn_grads.grad_wo);

                    // Attention core backward for this layer
                    AttentionCoreBackward attn_core_grads = backprop_attention_core_internal(layer_cache, attn_grads.grad_attn_out);
                    add_inplace(acc_wq, attn_core_grads.grad_wq);
                    add_inplace(acc_wk, attn_core_grads.grad_wk);
                    add_inplace(acc_wv, attn_core_grads.grad_wv);

                    // If this is the first layer, backprop through embeddings
                    if (layer == 0) {
                        EmbeddingBackward emb_grads = backprop_embeddings_internal(ctx, attn_core_grads.grad_x);
                        add_inplace(acc_tok, emb_grads.grad_token_emb);
                        add_inplace(acc_pos, emb_grads.grad_pos_emb);
                    } else {
                        // For non-first layers, grad_x becomes grad_h2 for the previous layer
                        grad_h2_cur = attn_core_grads.grad_x;
                    }
                }

                ++batch_count;
                if (batch_count >= batch_size) {
                    flush_batch();
                }
            }

            flush_batch();

            final_loss = total_loss / static_cast<float>(std::max<size_t>(1, total_targets));
            if (report_every > 0 && (epoch % report_every == 0 || epoch == 1 || epoch == epochs)) {
                std::cout << "epoch=" << epoch << " loss=" << final_loss;
                if (report_lr && epoch_update_steps > 0) {
                    std::cout << " avg_lr=" << (epoch_lr_sum / static_cast<float>(epoch_update_steps));
                }
                if (report_grad_norm && epoch_update_steps > 0) {
                    std::cout << " avg_grad_norm=" << (epoch_grad_norm_sum / static_cast<float>(epoch_update_steps));
                }
                std::cout << "\n";
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
int main(int argc, char* argv[]) {
    const std::string corpus =
        "hello world. this is a tiny transformer demo for character generation. "
        "hello world. this is a tiny transformer demo for character generation. ";

    CharTokenizer tok(corpus);
    int num_layers = 1;
    if (argc > 1) {
        num_layers = std::atoi(argv[1]);
        if (num_layers < 1) num_layers = 1;
    }
    TinyTransformer model(tok.vocab_size(), 24, 48, 64, 1234, num_layers);

    std::vector<int> train_ids = tok.encode(corpus);
    float start_loss = model.evaluate_next_token_loss(train_ids, 24);
    float end_loss = model.train_next_token(train_ids, 40, 24, 0.04f, 10);

    std::string prompt = "hello ";
    std::vector<int> prompt_ids = tok.encode(prompt);
    std::vector<int> greedy_ids = model.generate(prompt_ids, 80, 32, 0.0f, 1, 7);
    std::vector<int> sampled_ids = model.generate(prompt_ids, 80, 32, 0.9f, 5, 7);

    std::cout << "Tiny Transformer milestone (single-head, one-block)\n";
    std::cout << "vocab_size=" << tok.vocab_size() << " d_model=24 heads=1 blocks=1\n";
        std::cout << "Model has " << model.num_layers << " layer(s)\n";
    std::cout << "train_loss: start=" << start_loss << " end=" << end_loss << "\n";
    std::cout << "prompt:    " << prompt << "\n";
    std::cout << "greedy:    " << tok.decode(greedy_ids) << "\n";
    std::cout << "sampled:   " << tok.decode(sampled_ids) << "\n";

    return 0;
}
#endif
