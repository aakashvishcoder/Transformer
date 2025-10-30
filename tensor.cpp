#include "tensor.hpp"
#include <iostream>
#include <memory>

int main() {
    Tensor<float> a({2, 3}, true);
    a.data = {1,2,3,4,5,6};

    auto b = a.exp();        // element-wise exp
    auto c = b.sum();        // scalar tensor
    c.backward();            // gradients in a.grad

    a.print();               // original data
    std::cout << "Grad: ";
    for (auto g : a.grad) std::cout << g << " ";
    std::cout << std::endl;
}