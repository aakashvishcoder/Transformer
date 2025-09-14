#pragma once
#include "tensor.hpp"
#include "activations.hpp"
#include <optional>

using namespace std;

template<typename T>
class Dense {
public:
    Tensor<T,2> W;
    Tensor<T,1> b;

    Dense(size_t in_features, size_t out_features)
        : W({in_features, out_features}, true),
          b({out_features}, true)
    {
        // Xavier initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        T limit = std::sqrt(6.0 / (in_features + out_features));
        std::uniform_real_distribution<T> dist(-limit, limit);

        for(size_t i = 0; i < in_features; ++i)
            for(size_t j = 0; j < out_features; ++j)
                W({i,j}) = dist(gen);

        for(size_t j = 0; j < out_features; ++j)
            b({j}) = 0;
    }

    Tensor<T,2> forward(Tensor<T,2>& X) {
        const size_t batch = X.get_shape()[0];
        const size_t in_features = W.get_shape()[0];
        const size_t out_features = W.get_shape()[1];

        Tensor<T,2> Y({batch, out_features}, true);

        // Forward pass
        for(size_t i = 0; i < batch; ++i)
            for(size_t j = 0; j < out_features; ++j) {
                T sum = 0;
                for(size_t k = 0; k < in_features; ++k)
                    sum += X({i,k}) * W({k,j});
                Y({i,j}) = sum + b({j});
            }

        // Backward
        Y.set_backward_fn([this, &X, batch, in_features, out_features](Tensor<T,2>* Y_ptr){
            auto& dY = Y_ptr->get_grad_ref(); // upstream gradient

            // Grad w.r.t X
            for(size_t i = 0; i < batch; ++i)
                for(size_t k = 0; k < in_features; ++k) {
                    T grad = 0;
                    for(size_t j = 0; j < out_features; ++j)
                        grad += dY[i*out_features + j] * W({k,j}); // use this->W implicitly
                    X.get_grad_ref()[i*in_features + k] += grad;
                }

            // Grad w.r.t W
            for(size_t k = 0; k < in_features; ++k)
                for(size_t j = 0; j < out_features; ++j) {
                    T grad = 0;
                    for(size_t i = 0; i < batch; ++i)
                        grad += X({i,k}) * dY[i*out_features + j];
                    W.get_grad_ref()[k*out_features + j] += grad;
                }

            // Grad w.r.t b
            for(size_t j = 0; j < out_features; ++j) {
                T grad_sum = 0;
                for(size_t i = 0; i < batch; ++i)
                    grad_sum += dY[i*out_features + j];
                b.get_grad_ref()[j] += grad_sum;
            }
        });
        return Y;
    }
};

template<typename T>
class FeedForward {
public:
    Dense<T> fc1;
    Dense<T> fc2;

    FeedForward(size_t embed_dim, size_t hidden_dim)
        : fc1(embed_dim, hidden_dim), fc2(hidden_dim, embed_dim) {}

    Tensor<T,2> forward(Tensor<T,2>& x) {
        // First dense layer
        auto out1 = fc1.forward(x);

        // ReLU with backward support
        auto relu_out = Activations::ReLU<T,2>(out1);

        // Second dense layer
        auto out2 = fc2.forward(relu_out);

        return out2; // backward flows through fc2 → ReLU → fc1 automatically
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