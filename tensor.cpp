#include "tensor.hpp"
#include <iostream>
#include <memory>

int main() {
    Tensor<float> a({2, 3}, true);
    a.data = {1, 2, 3, 4, 5, 6};
    
    // Store the intermediate tensor explicitly
    auto exp_result = a.exp();    // This keeps the tensor alive
    auto sum_result = exp_result.sum();
    sum_result.backward();
    
    a.print();
    std::cout << "Grad: ";
    for (auto g : a.grad) std::cout << g << " ";
    std::cout << std::endl;
}