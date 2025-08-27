#pragma once
#include "tensor.hpp"
#include <bits/stdc++.h>

template <typename T, size_t N>
class Activations {
public:
    void ReLU(Tensor<T,N>& input) {
        auto& data = input.get_data_ref();
        for(size_t i = 0; i < data.size(); i++) {
            data[i] = (data[i]>T(0))?data[i]:T(0);
        }
    }

    void LeakyReLU(Tensor<T,N>& input, const T alpha = 0.01) {
        auto& data = input.get_data_ref();
        for(size_t i = 0; i < data.size(); i++) {
            T other = (learning_rate*data[i]);
            data[i] = (data[i]>other)?data[i]:other;
        }
    }

};