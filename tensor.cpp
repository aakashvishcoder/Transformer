
#include "tensor_minimal.hpp"
#include <iostream>

static float dataset_mse(const Tensor<float>& w,
                         const Tensor<float>& b,
                         const std::vector<float>& xs,
                         const std::vector<float>& ys) {
    float total = 0.0f;
    for (size_t i = 0; i < xs.size(); ++i) {
        float pred = xs[i] * w.data[0] + b.data[0];
        float diff = pred - ys[i];
        total += diff * diff;
    }
    return total / static_cast<float>(xs.size());
}

int main() {
    std::vector<float> xs = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> ys;
    ys.reserve(xs.size());
    for (float x : xs) {
        ys.push_back(2.0f * x + 1.0f);
    }

    Tensor<float> w({1});
    Tensor<float> b({1});
    w.data[0] = -0.5f;
    b.data[0] = 0.0f;
    const float lr = 0.05f;
    const int epochs = 300;

    float start_loss = dataset_mse(w, b, xs, ys);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float grad_w = 0.0f;
        float grad_b = 0.0f;
        for (size_t i = 0; i < xs.size(); ++i) {
            float pred = xs[i] * w.data[0] + b.data[0];
            float diff = pred - ys[i];
            grad_w += 2.0f * diff * xs[i];
            grad_b += 2.0f * diff;
        }
        grad_w /= static_cast<float>(xs.size());
        grad_b /= static_cast<float>(xs.size());
        w.data[0] -= lr * grad_w;
        b.data[0] -= lr * grad_b;
    }

    float end_loss = dataset_mse(w, b, xs, ys);

    std::cout << "Tiny linear model trained with SGD\n";
    std::cout << "start_loss=" << start_loss << " end_loss=" << end_loss << "\n";
    std::cout << "learned w=" << w.data[0] << " b=" << b.data[0] << "\n";


    return 0;
}


