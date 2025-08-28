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
        return (T(1)/y_pred.get_data().size())*((y_pred-y_true)^2).sqrt().sum();
    }

    template <typename T, size_t N>
    static T BCE(const Tensor<T,N>& y_pred, const Tensor<T,N>& y_true) {
        
    }
};