#include "tensor.hpp"
#include "activations.hpp"
#include "loss_functions.hpp"
#include "dataframe.hpp"
#include "layers.hpp"
#include "architecture.hpp"
#include <iostream>
using namespace std;

int main() {
    // Parameters
    size_t batch_size = 2;
    size_t seq_len = 3;
    size_t embed_dim = 4;
    size_t num_heads = 2;
    size_t ffn_hidden_dim = 8;
    size_t num_layers = 2;

    // Create input tensor [batch_size, seq_len, embed_dim]
    Tensor<float, 3> input({batch_size, seq_len, embed_dim});
    input.fill_random(-0.1f, 0.1f);

    // Print input
    std::cout << "--- Input Tensor ---\n";
    input.print();

    // Create Transformer Encoder
    TransformerEncoder<float, 3> encoder(num_layers, embed_dim, num_heads, ffn_hidden_dim);

    // Forward pass
    Tensor<float, 3> output = encoder.forward(input);

    // Print output
    std::cout << "\n--- Transformer Encoder Output ---\n";
    output.print();

    return 0;
}
