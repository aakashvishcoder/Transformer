#include "tensor.hpp"
#include <iostream>
#include <memory>

int main() {
    // Test 2D input
    Tensor<float> x({2, 3}, true);  // batch=2, in_features=3
    x.fill_random(-1, 1);
    
    Linear<float> linear(3, 4);  // in=3, out=4
    auto y = linear.forward(x);  // should be (2, 4)
    
    std::cout << "Input shape: ";
    for (auto s : x.shape) std::cout << s << " ";
    std::cout << "\nOutput shape: ";
    for (auto s : y.shape) std::cout << s << " ";
    std::cout << std::endl;
    
    // Test backward
    auto loss = y.sum();
    loss.backward();
    
    std::cout << "Gradients computed successfully!" << std::endl;
}