#include <iostream>
#include "tensor.hpp"
#include "layers.hpp" // Your Dense class

int main() {
    // --- Create input tensor X (batch_size=2, in_features=3) ---
    Tensor<float,2> X({2,3}, true);
    X({0,0}) = 1; X({0,1}) = 2; X({0,2}) = 3;
    X({1,0}) = 4; X({1,1}) = 5; X({1,2}) = 6;

    // --- Create Dense layer (3 in_features -> 2 out_features) ---
    Dense<float> layer(3,2);

    // --- Forward pass ---
    Tensor<float,2> Y = layer.forward(X);

    std::cout << "Forward Y:\n";
    for (size_t i=0;i<Y.get_shape()[0];i++){
        for (size_t j=0;j<Y.get_shape()[1];j++)
            std::cout << Y({i,j}) << " ";
        std::cout << "\n";
    }

    // --- Set upstream gradient (dL/dY) to ones for testing ---
    for (size_t i=0;i<Y.get_grad_ref().size();i++)
        Y.get_grad_ref()[i] = 1.0f;

    // --- Backward pass ---
    if(Y.get_backward_fn())
        Y.get_backward_fn()(&Y);

    // --- Print gradients ---
    std::cout << "\nGrad w.r.t X:\n";
    for (size_t i=0;i<X.get_shape()[0];i++){
        for (size_t j=0;j<X.get_shape()[1];j++)
            std::cout << X.get_grad_ref()[i*X.get_shape()[1]+j] << " ";
        std::cout << "\n";
    }

    std::cout << "\nGrad w.r.t W:\n";
    for (size_t i=0;i<layer.W.get_shape()[0];i++){
        for (size_t j=0;j<layer.W.get_shape()[1];j++)
            std::cout << layer.W.get_grad_ref()[i*layer.W.get_shape()[1]+j] << " ";
        std::cout << "\n";
    }

    std::cout << "\nGrad w.r.t b:\n";
    for (size_t j=0;j<layer.b.get_shape()[0];j++)
        std::cout << layer.b.get_grad_ref()[j] << " ";
    std::cout << "\n";

    return 0;
}
