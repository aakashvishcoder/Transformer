#include "tensor.hpp"
#include "layers.hpp"
#include "activations.hpp"
#include <iostream>
#include <memory>

int main() {
    using T = float;

    auto x = std::make_shared<Tensor<T,2>>(std::array<size_t,2>{2,3}, true);
    (*x)({0,0}) = 1.0f; (*x)({0,1}) = -2.0f; (*x)({0,2}) = 3.0f;
    (*x)({1,0}) = -1.0f; (*x)({1,1}) = 0.0f; (*x)({1,2}) = 2.0f;

    FeedForward<T> ff(3,4);
    auto out = ff.forward(x);

    std::cout << "Forward output:\n";
    for(size_t i=0;i<out->size();++i)
        std::cout << out->get_data()[i] << " ";
    std::cout << "\n";

    // Backward using sum(out) as loss
    out->zero_grad();
    x->zero_grad();
    for(size_t i=0;i<out->size();++i)
        out->get_grad_ref()[i] = 1.0f;

    out->backward();

    std::cout << "Gradients w.r.t input x:\n";
    for(size_t i=0;i<x->size();++i)
        std::cout << x->get_grad_ref()[i] << " ";
    std::cout << "\n";
}
