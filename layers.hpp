#pragma once
#include "tensor.hpp"

template <typename T>
class Dense {
public:
    Dense(size_t in_features, size_t out_features) 
        : in_features_(in_features), out_features_(out_features),
        weights_({in_features, out_features}),
        bias_({1, out_features})
    {
        weights_.fill_random(-0.1,0.1);
        bias_.fill_random(-0.1,0.1);
    }

    template <size_t N>
    Tensor<T, N> forward(const Tensor<T, N>& input) {
        // Compute matmul along the last axis
        Tensor<T, N> output = dot(input, weights_); // broadcast batch dims

        // Broadcast bias to match output shape
        array<size_t, N> target_shape = output.get_shape_ref();
        target_shape[N - 2] = 1; // keep batch dims, last axis = output_features
        target_shape[N - 1] = out_features_;
        Tensor<T, N> bias_bcast = bias_.broadcast_to(target_shape);

        output += bias_bcast;

        return output;
    }

    Tensor<T,2>& get_weights() { return weights_; }
    Tensor<T,2>& get_bias() { return bias_; }
private:
    size_t in_features_;
    size_t out_features_;
    Tensor<T,2> weights_;
    Tensor<T,2> bias_;
};