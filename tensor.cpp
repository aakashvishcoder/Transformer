#include "tensor.hpp"
#include <iostream>
#include <memory>

int main() {
    // Test 1: Basic autograd
    Tensor<float> a({2,3}, true);
    auto b = a.exp().sum();
    b.backward(); // Should not crash

    // Test 2: Matmul
    Tensor<float> x({1, 10}, true), w({5, 10}, true);
    auto y = matmul(x, w.transpose_last_two()); // {1,5}
    y.sum().backward();
}