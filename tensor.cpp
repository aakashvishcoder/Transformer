#include <iostream>
#include "tensor.hpp"
#include "layers.hpp"

// Include your FeedForward class

int main() {
    using T = float;

    // --- Input tensor (batch=2, embed_dim=3) ---
    Tensor<T,2> x({2,3}, true); // requires grad
    auto& x_data = x.get_data_ref();
    x_data = {1.0, -2.0, 0.0,
              -0.5, 2.0, -3.0};

    // --- Create FeedForward network ---
    FeedForward<T> ff(3, 4); // embed_dim=3, hidden_dim=4
    ff.fc1.W.set_requires_grad(true);
    ff.fc1.b.set_requires_grad(true);
    ff.fc2.W.set_requires_grad(true);
    ff.fc2.b.set_requires_grad(true);

    // --- Forward pass ---
    std::cout << "x shape: " << x.get_shape()[0] << " x " << x.get_shape()[1] << std::endl;
    std::cout << "fc1 W shape: " << ff.fc1.W.get_shape()[0] << " x " << ff.fc1.W.get_shape()[1] << std::endl;
    auto y = ff.forward(x);

    std::cout << "Forward output:\n";
    for (auto v : y.get_data_ref()) std::cout << v << " ";
    std::cout << "\n";

    // --- Set upstream gradient (dL/dy = 1) ---
    auto& grad = y.get_grad_ref();
    grad.assign(grad.size(), 1.0f);

    // --- Backward pass ---
    y.backward();

    // --- Gradients ---
    std::cout << "Gradient wrt input x:\n";
    for (auto g : x.get_grad_ref()) std::cout << g << " ";
    std::cout << "\n";

    std::cout << "Gradient wrt fc1 weights W:\n";
    for (auto g : ff.fc1.W.get_grad_ref()) std::cout << g << " ";
    std::cout << "\n";

    std::cout << "Gradient wrt fc1 biases b:\n";
    for (auto g : ff.fc1.b.get_grad_ref()) std::cout << g << " ";
    std::cout << "\n";

    std::cout << "Gradient wrt fc2 weights W:\n";
    for (auto g : ff.fc2.W.get_grad_ref()) std::cout << g << " ";
    std::cout << "\n";

    std::cout << "Gradient wrt fc2 biases b:\n";
    for (auto g : ff.fc2.b.get_grad_ref()) std::cout << g << " ";
    std::cout << "\n";

    return 0;
}
