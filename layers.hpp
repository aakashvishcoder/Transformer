#pragma once
#include "tensor.hpp"
#include "activations.hpp"
#include <optional>

using namespace std;


template<typename T>
class Dense {
public:
    Tensor<T, 2> weight_; // [in_features, out_features]
    Tensor<T, 1> bias_;   // [out_features]

    Dense(size_t in_features, size_t out_features)
        : weight_({in_features, out_features}), bias_({out_features}) {
        weight_.fill_random(-0.1, 0.1);
        bias_.zeros();
    }

    // Forward for N-dim tensor: apply matmul along last axis
    template<size_t N>
    Tensor<T, N> forward(const Tensor<T, N>& input) const {
        const auto& in_shape = input.get_shape_ref();
        if (in_shape[N-1] != weight_.get_shape_ref()[0])
            throw runtime_error("Input last dim does not match weight rows");

        // Output shape: same as input, last dim = weight cols
        auto out_shape = in_shape;
        out_shape[N-1] = weight_.get_shape_ref()[1];
        Tensor<T, N> output(out_shape);

        const auto& in_data = input.get_data_ref();
        const auto& w_data = weight_.get_data_ref();
        const auto& w_shape = weight_.get_shape_ref();
        auto& out_data = output.get_data_ref();
        const size_t batch_size = input.size() / in_shape[N-1];
        const size_t in_features = w_shape[0];
        const size_t out_features = w_shape[1];

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t o = 0; o < out_features; ++o) {
                T sum = 0;
                for (size_t i = 0; i < in_features; ++i) {
                    sum += in_data[b * in_features + i] * w_data[i * out_features + o];
                }
                out_data[b * out_features + o] = sum + bias_.get_data_ref()[o];
            }
        }

        return output;
    }
};

template<typename T>
class FeedForward {
public:
    Dense<T> fc1;
    Dense<T> fc2;

    FeedForward(size_t embed_dim, size_t hidden_dim)
        : fc1(embed_dim, hidden_dim), fc2(hidden_dim, embed_dim) {}

    template<size_t N>
    Tensor<T,N> forward(const Tensor<T,N>& x) {
        auto out = fc1.forward(x);

        // ReLU activation
        Tensor<T,N> relu_out(out.get_shape_ref());
        const auto& in_data = out.get_data_ref();
        auto& relu_data = relu_out.get_data_ref();
        for (size_t i = 0; i < in_data.size(); ++i)
            relu_data[i] = max(in_data[i], T(0));

        out = fc2.forward(relu_out);
        return out;
    }
};

template<typename T, size_t N>
class LayerNormalization {
public:
    using Shape = array<size_t, N>;

    Tensor<T, N> gamma;
    Tensor<T, N> beta;
    T epsilon;

    LayerNormalization() : epsilon(1e-5) {} // default constructor

    // proper constructor with input shape
    LayerNormalization(size_t param_dim, const Shape& input_shape)
        : epsilon(1e-5), gamma(), beta()
    {
        Shape param_shape = input_shape;
        // Make gamma and beta the same shape as input
        gamma = Tensor<T, N>(input_shape);
        gamma.fill_value(T(1));
        beta = Tensor<T, N>(input_shape);
        beta.fill_value(T(0));
    }

    template<size_t NA>
    Tensor<T, NA> forward(const Tensor<T, NA>& input) {
        auto mean = input.mean_axis(N-1);       // shape reduced along last axis
        auto stddev = input.std_axis(N-1);      // same

        // broadcast mean and stddev to input shape
        auto mean_b = mean.broadcast_to(input.get_shape_ref());
        auto std_b  = stddev.broadcast_to(input.get_shape_ref());

        auto normalized = (input - mean_b) / (std_b + epsilon);

        // broadcast gamma and beta if needed
        auto gamma_b = gamma.broadcast_to(input.get_shape_ref());
        auto beta_b  = beta.broadcast_to(input.get_shape_ref());

        return normalized * gamma_b + beta_b;
    }
};


template <typename T,size_t M>
class ScaledDotProductAttention {
public:
    pair<Tensor<T,M>, Tensor<T,M>> forward(
        const Tensor<T,M>& queries,
        const Tensor<T,M>& keys,
        const Tensor<T,M>& values,
        const optional<Tensor<T,M>>& mask = nullopt
    ) {
        // transpose last two dims of keys
        array<size_t, M> perm;
        for (size_t i = 0; i < M; i++) perm[i] = i;
        perm[M-1] = M-2;
        perm[M-2] = M-1;

        auto K_T = keys.transpose(perm);

        auto matmul = dot(queries, K_T);

        // scale
        float scale = 1.0f / sqrt((float)queries.get_shape_ref().back());
        matmul = matmul * scale;

        // softmax along last axis
        auto attn_weights = Activations::Softmax(matmul, matmul.get_shape_ref().size() - 1);

        // weighted sum
        auto context = dot(attn_weights, values);

        return {context, attn_weights};
    }
};

template <typename T, size_t M>
class MultiHeadAttentionLayer {
public:
    size_t num_heads;
    size_t head_dim;
    size_t embed_dim;
    size_t out_features;
    
    vector<Dense<T>> wq, wk, wv; // per-head projections
    Dense<T> wo;                       // final output projection

    MultiHeadAttentionLayer(size_t embed_dim_, size_t num_heads_, size_t out_features_)
        : num_heads(num_heads_),
          embed_dim(embed_dim_),
          out_features(out_features_),
          head_dim((embed_dim_ + num_heads_ - 1) / num_heads_), // ceil division
          wo(head_dim * num_heads_, out_features_)             // input = concat(heads)
    {
        for (size_t i = 0; i < num_heads; i++) {
            wq.emplace_back(Dense<T>(embed_dim, head_dim));
            wk.emplace_back(Dense<T>(embed_dim, head_dim));
            wv.emplace_back(Dense<T>(embed_dim, head_dim));
        }
    }

    Tensor<T, M> forward(
        const Tensor<T, M>& queries,
        const Tensor<T, M>& keys,
        const Tensor<T, M>& values,
        const optional<Tensor<T,M>>& mask = nullopt
    ) {
        vector<Tensor<T, M>> head_outputs;

        for (size_t i = 0; i < num_heads; i++) {
            auto q = wq[i].forward(queries);
            auto k = wk[i].forward(keys);
            auto v = wv[i].forward(values);

            ScaledDotProductAttention<T, M> sdpa;
            auto [context, attn_weights] = sdpa.forward(q, k, v, mask);
            head_outputs.push_back(context);
        }

        // Concatenate along last axis
        auto concat = Tensor<T, M>::concat(head_outputs, M-1);

        // Final linear projection to out_features
        return wo.forward(concat);
    }
};

template <typename T, size_t M>
class PositionalEncodingLayer {
public:
    Tensor<T, 2> positional_encoding(const size_t seq_len, const size_t d_model) {
        using Shape = vector<size_t>;

        vector<T> flat_data;
        flat_data.reserve(seq_len * d_model); 

        for (size_t pos = 0; pos < seq_len; ++pos) {
            for (size_t dim = 0; dim < d_model; ++dim) {
                T angle = pos / pow(T(10000.0), T(2.0 * (dim / 2)) / d_model);
                flat_data.push_back((dim % 2 == 0) ? sin(angle) : cos(angle));
            }
        }

        return Tensor<T, 2>({seq_len, d_model}, flat_data);
    }


    void add_positional_encoding(Tensor<T,M>& batch_input) {
        const auto& shape = batch_input.get_shape();
        size_t batch_size = shape[0];
        size_t seq_len = shape[1];
        size_t d_model = shape[2];

        auto pe = positional_encoding(seq_len, d_model);

        for(size_t b = 0; b < batch_size; b++) {
            for(size_t pos = 0; pos < seq_len; pos++) {
                for(size_t i = 0; i < d_model; i++) {
                    batch_input({b,pos,i}) += pe({pos,i});
                }
            }
        }
    }
};