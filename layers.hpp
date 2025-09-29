#pragma once
#include "tensor.hpp"
#include "activations.hpp"
#include <memory>
#include <random>
#include <cmath>
#include <array>
#include <optional>   // add at the top if not already
using namespace std;

template<typename T>
class Dense {
public:
    std::shared_ptr<Tensor<T,2>> W;
    std::shared_ptr<Tensor<T,1>> b;

    Dense(size_t in_features, size_t out_features) {
        W = std::make_shared<Tensor<T,2>>(std::array<size_t,2>{in_features, out_features}, true);
        b = std::make_shared<Tensor<T,1>>(std::array<size_t,1>{out_features}, true);

        std::random_device rd;
        std::mt19937 gen(rd());
        T limit = std::sqrt(6.0 / (in_features + out_features));
        std::uniform_real_distribution<T> dist(-limit, limit);

        for(size_t i = 0; i < in_features; ++i)
            for(size_t j = 0; j < out_features; ++j)
                (*W)({i,j}) = dist(gen);

        for(size_t j = 0; j < out_features; ++j)
            (*b)({j}) = 0;
    }

    // Forward pass
    std::shared_ptr<Tensor<T,2>> forward(std::shared_ptr<Tensor<T,2>> X) {
        // Matrix multiply
        auto Z = matmul(X, W); // <-- pass shared_ptr directly

        // Broadcast bias
        std::array<size_t,2> target_shape = { X->get_shape()[0], b->get_shape()[0] };
        auto b_broadcast = b->unsqueeze(0).broadcast_to(target_shape);

        // Add bias
        Tensor<T,2> Y = *Z + b_broadcast;

        return std::make_shared<Tensor<T,2>>(Y);
    }
};


template<typename T>
class FeedForward {
public:
    Dense<T> fc1;
    Dense<T> fc2;

    FeedForward(size_t embed_dim, size_t hidden_dim)
        : fc1(embed_dim, hidden_dim), fc2(hidden_dim, embed_dim) {}

    std::shared_ptr<Tensor<T,2>> forward(std::shared_ptr<Tensor<T,2>> x) {
        auto out1 = fc1.forward(x);
        auto relu_out = Activations::ReLU<T,2>(out1); // ReLU expects shared_ptr
        auto out2 = fc2.forward(relu_out);
        return out2;
    }
};

template<typename T, size_t N>
class LayerNormalization {
public:
    using Shape = array<size_t, N>;

    Tensor<T, N> gamma;
    Tensor<T, N> beta;
    T epsilon;

    LayerNormalization() : epsilon(1e-5) {}

    LayerNormalization(size_t param_dim, const Shape& input_shape)
        : epsilon(1e-5), gamma(input_shape), beta(input_shape)
    {
        gamma.fill_value(T(1));
        beta.fill_value(T(0));
    }

    template<size_t NA>
    Tensor<T, NA> forward(const Tensor<T, NA>& input) {
        auto mean = input.mean_axis(N-1);
        auto stddev = input.std_axis(N-1);

        auto mean_b = mean.broadcast_to(input.get_shape_ref());
        auto std_b  = stddev.broadcast_to(input.get_shape_ref());

        auto normalized = (input - mean_b) / (std_b + epsilon);

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
        array<size_t, M> perm;
        for (size_t i = 0; i < M; i++) perm[i] = i;
        perm[M-1] = M-2; perm[M-2] = M-1;

        auto K_T = keys.transpose(perm);

        auto matmul_res = dot(queries, K_T);

        float scale = 1.0f / sqrt((float)queries.get_shape_ref().back());
        matmul_res = matmul_res * scale;

        auto attn_weights = Activations::Softmax(matmul_res, M-1);

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

    vector<Dense<T>> wq, wk, wv;
    Dense<T> wo;

    MultiHeadAttentionLayer(size_t embed_dim_, size_t num_heads_, size_t out_features_)
        : num_heads(num_heads_),
          embed_dim(embed_dim_),
          out_features(out_features_),
          head_dim((embed_dim_ + num_heads_ - 1)/num_heads_),
          wo(head_dim*num_heads_, out_features_)
    {
        for(size_t i=0;i<num_heads;i++){
            wq.emplace_back(Dense<T>(embed_dim, head_dim));
            wk.emplace_back(Dense<T>(embed_dim, head_dim));
            wv.emplace_back(Dense<T>(embed_dim, head_dim));
        }
    }

    shared_ptr<Tensor<T,M>> forward(shared_ptr<Tensor<T,M>> queries,
                                    shared_ptr<Tensor<T,M>> keys,
                                    shared_ptr<Tensor<T,M>> values,
                                    const optional<shared_ptr<Tensor<T,M>>>& mask = nullopt)
    {
        vector<Tensor<T,M>> head_outputs;

        for(size_t i=0;i<num_heads;i++){
            auto q = *(wq[i].forward(queries));
            auto k = *(wk[i].forward(keys));
            auto v = *(wv[i].forward(values));

            ScaledDotProductAttention<T,M> sdpa;
            auto [context, attn_weights] = sdpa.forward(q,k,v);
            head_outputs.push_back(context);
        }

        auto concat = Tensor<T,M>::concat(head_outputs, M-1);
        return wo.forward(make_shared<Tensor<T,M>>(concat));
    }
};

template <typename T, size_t M>
class PositionalEncodingLayer {
public:
    Tensor<T, 2> positional_encoding(const size_t seq_len, const size_t d_model) {
        vector<T> flat_data;
        flat_data.reserve(seq_len * d_model);

        for(size_t pos=0;pos<seq_len;pos++){
            for(size_t dim=0;dim<d_model;dim++){
                T angle = pos / pow(T(10000.0), T(2*(dim/2))/d_model);
                flat_data.push_back(dim%2==0 ? sin(angle) : cos(angle));
            }
        }

        return Tensor<T,2>({seq_len,d_model}, flat_data);
    }

    void add_positional_encoding(Tensor<T,M>& batch_input){
        const auto& shape = batch_input.get_shape();
        size_t batch_size = shape[0];
        size_t seq_len = shape[1];
        size_t d_model = shape[2];

        auto pe = positional_encoding(seq_len,d_model);

        for(size_t b=0;b<batch_size;b++)
            for(size_t pos=0;pos<seq_len;pos++)
                for(size_t i=0;i<d_model;i++)
                    batch_input({b,pos,i}) += pe({pos,i});
    }
};
