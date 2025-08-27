#pragma once
#include "tensor.hpp"
#include <vector>
#include <stdexcept>

struct Activations {
    // In-place ReLU
    template<typename T, size_t N>
    static void ReLU(Tensor<T,N>& X) {
        auto& data = X.get_data_ref();
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = (data[i] > T(0)) ? data[i] : T(0);
    }

    // In-place LeakyReLU
    template<typename T, size_t N>
    static void LeakyReLU(Tensor<T,N>& X, T alpha = T(0.01)) {
        auto& data = X.get_data_ref();
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = (data[i] > T(0)) ? data[i] : alpha * data[i];
    }

    // Softmax along a given axis, returns new tensor
    template<typename T, size_t N>
    static Tensor<T,N> Softmax(const Tensor<T,N>& X, size_t axis) {
        // Step 1: find max along the axis (rank N-1)
        Tensor<T,N-1> max_val = X.max_axis(axis);

        // Step 2: unsqueeze back to original rank so broadcasting works
        Tensor<T,N> max_broadcast = max_val.unsqueeze(axis).broadcast_to(X.get_shape());

        // Step 3: subtract max and exponentiate
        Tensor<T,N> exp_shifted = (X - max_broadcast).exp();

        // Step 4: sum along the axis
        Tensor<T,N-1> sum_exp = exp_shifted.sum_axis(axis);

        // Step 5: unsqueeze sum to original rank and broadcast
        Tensor<T,N> sum_broadcast = sum_exp.unsqueeze(axis).broadcast_to(X.get_shape());

        // Step 6: divide to get softmax
        return exp_shifted / sum_broadcast;
    }


    // Sigmoid, returns new tensor
    template<typename T, size_t N>
    static Tensor<T,N> Sigmoid(const Tensor<T,N>& X) {
        Tensor<T,N> result(X.get_shape());
        const auto& in = X.get_data_ref();
        auto& out = result.get_data_ref();
        for (size_t i = 0; i < in.size(); i++) {
            out[i] = T(1) / (T(1) + std::exp(-in[i]));
        }
        return result;
    }

    template<typename T, size_t N>
    static Tensor<T,N> Tanh(const Tensor<T,N>& X) {
        Tensor<T,N> result(X.get_shape());
        const auto& in = X.get_data_ref();
        auto& out = result.get_data_ref();
        for (size_t i = 0; i < in.size(); i++) {
            out[i] = std::tanh(in[i]);  // Or: 2*sigmoid(2x) - 1
        }
        return result;
    }
};
