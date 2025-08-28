#pragma once
#include "tensor.hpp"
#include <algorithm>
#include <vector>
#include <stdexcept>
using namespace std;

class LossFunctions {
public:
    template <typename T, size_t N>
    static T MSE(const Tensor<T,N>& y_pred, const Tensor<T,N>& y_true) {
        return (T(1)/y_pred.get_data().size())*((y_pred-y_true)^2).sum();
    }

    template <typename T, size_t N>
    static T RMSE(const Tensor<T,N>& y_pred, const Tensor<T,N>& y_true) {
        return std::sqrt(MSE(y_pred,y_true));
    }

    template <typename T, size_t N>
    static T BCE(Tensor<T,N>& output, const Tensor<T,N>& X) {
        output.clip(T(1e-10), T(1)-T(1e-10));
        return -(X * output.log() + (T(1)-X) * (T(1)-output).log()).sum() / static_cast<T>(X.get_data().size());
    }

    template <typename T, size_t N>
    static T CE(Tensor<T,N>& output, const Tensor<T,N>& X) {
        output.clip(T(1e-10), T(1)-T(1e-10));
        return -(X*output.log()).sum() / static_cast<T>(X.get_data().size());
    }
};