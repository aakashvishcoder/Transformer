#pragma once
#include "tensor.hpp"
#include <algorithm>
#include <vector>
#include <stdexcept>
using namespace std;

class LossFunctions {
public:
    // Mean Squared Error
    template <typename T, size_t N>
    static T MSE(const Tensor<T,N>& y_pred, const Tensor<T,N>& y_true) {
        return (T(1)/y_pred.get_data().size()) * ((y_pred - y_true) ^ 2).sum();
    }

    template <typename T, size_t N>
    static Tensor<T,N> MSE_Backward(const Tensor<T,N>& y_pred, const Tensor<T,N>& y_true) {
        T inv_n = T(2) / y_pred.get_data().size();
        return (y_pred - y_true) * inv_n;  // dL/dy_pred
    }

    // Root Mean Squared Error
    template <typename T, size_t N>
    static T RMSE(const Tensor<T,N>& y_pred, const Tensor<T,N>& y_true) {
        return std::sqrt(MSE(y_pred,y_true));
    }

    template <typename T, size_t N>
    static Tensor<T,N> RMSE_Backward(const Tensor<T,N>& y_pred, const Tensor<T,N>& y_true) {
        T mse = MSE(y_pred, y_true);
        T rmse = std::sqrt(mse);
        if (rmse == 0) return Tensor<T,N>(y_pred.get_shape()); // zero grad if perfect match
        return MSE_Backward(y_pred, y_true) * (T(1) / (2 * rmse));
    }

    // Binary Cross Entropy
    template <typename T, size_t N>
    static T BCE(Tensor<T,N>& output, const Tensor<T,N>& X) {
        output.clip(T(1e-10), T(1)-T(1e-10));
        return -(X * output.log() + (T(1)-X) * (T(1)-output).log()).sum()
               / static_cast<T>(X.get_data().size());
    }

    template <typename T, size_t N>
    static Tensor<T,N> BCE_Backward(Tensor<T,N>& output, const Tensor<T,N>& X) {
        output.clip(T(1e-10), T(1)-T(1e-10));
        T inv_n = T(1) / output.get_data().size();
        return ((output - X) / (output * (T(1) - output))) * inv_n;  // dL/dy_pred
    }

    // ===== Cross Entropy with one-hot labels (old version) =====
    template <typename T, size_t N>
    static T CE_OneHot(Tensor<T,N>& output, const Tensor<T,N>& X) {
        output.clip(T(1e-10), T(1)-T(1e-10));
        return -(X * output.log()).sum()
               / static_cast<T>(X.get_data().size());
    }

    template <typename T, size_t N>
    static Tensor<T,N> CE_Backward_OneHot(Tensor<T,N>& output, const Tensor<T,N>& X) {
        output.clip(T(1e-10), T(1)-T(1e-10));
        T inv_n = T(1) / output.get_data().size();
        return -(X / output) * inv_n;
    }

    // ===== Cross Entropy with class indices (new version) =====
    template <typename T>
    static T CE_Indices(Tensor<T,2>& output, const vector<size_t>& y_true) {
        // output shape: [batch, num_classes]
        auto& data = output.get_data_ref();
        auto shape = output.get_shape_ref();
        size_t batch = shape[0];
        size_t num_classes = shape[1];
        if (y_true.size() != batch)
            throw runtime_error("y_true size must equal batch size");

        output.clip(T(1e-10), T(1)-T(1e-10));

        T loss = 0;
        for (size_t i = 0; i < batch; i++) {
            size_t cls = y_true[i];
            if (cls >= num_classes) throw runtime_error("class index out of range");
            loss -= log(data[i * num_classes + cls]);
        }
        return loss / batch;
    }

    template <typename T>
    static Tensor<T,2> CE_Backward_Indices(Tensor<T,2>& output, const vector<size_t>& y_true) {
        // gradient shape = same as output
        auto shape = output.get_shape_ref();
        size_t batch = shape[0];
        size_t num_classes = shape[1];
        Tensor<T,2> grad(shape);

        auto& gdata = grad.get_data_ref();
        auto& odata = output.get_data_ref();

        output.clip(T(1e-10), T(1)-T(1e-10));
        T inv_batch = T(1) / batch;

        for (size_t i = 0; i < batch; i++) {
            for (size_t j = 0; j < num_classes; j++) {
                size_t cls = y_true[i];
                if (j == cls)
                    gdata[i * num_classes + j] = -inv_batch / odata[i * num_classes + j];
                else
                    gdata[i * num_classes + j] = 0;
            }
        }
        return grad;
    }
};
