#include <iostream>
#include "tensor.hpp"

int main() {
    // --- Define A: shape [2,2] ---
    Tensor<float, 2> A({2,2}, true);
    A.get_data_ref() = {1,2,3,4}; // Fill values

    // --- Define B: shape [2,2] ---
    Tensor<float, 2> B({2,2}, true);
    B.get_data_ref() = {5,6,7,8}; // Fill values

    // --- Compute C = dot_autograd_nd(A,B) ---
    auto C = dot_autograd_nd(A, B); // should be shape [2,2]

    // --- Set upstream gradient (like from loss) ---
    for (size_t i = 0; i < C.size(); ++i) C.get_grad_ref()[i] = 1.0f;

    // --- Backward pass ---
    C.backward();

    // --- Print results ---
    std::cout << "Forward result C:\n";
    for (size_t i = 0; i < C.size(); ++i) std::cout << C.get_data_ref()[i] << " ";
    std::cout << "\n\nGradient w.r.t A:\n";
    for (size_t i = 0; i < A.size(); ++i) std::cout << A.get_grad_ref()[i] << " ";
    std::cout << "\n\nGradient w.r.t B:\n";
    for (size_t i = 0; i < B.size(); ++i) std::cout << B.get_grad_ref()[i] << " ";
    std::cout << "\n";

    return 0;
}
