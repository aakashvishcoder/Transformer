#pragma once
#include "tensor.hpp"
#include "layers.hpp" 
#include <vector>

template<typename T, size_t M>
class TransformerEncoderBlock {
public:
    MultiHeadAttentionLayer<T, M> mha;
    LayerNormalization<T, M> ln1;
    FeedForward<T> ffn;
    LayerNormalization<T, M> ln2;

    TransformerEncoderBlock(size_t embed_dim, size_t num_heads, size_t ffn_hidden_dim)
        : mha(embed_dim, num_heads, embed_dim),
          ln1(embed_dim, {1, 1, embed_dim}),
          ffn(embed_dim, ffn_hidden_dim),
          ln2(embed_dim, {1, 1, embed_dim}) {}

    Tensor<T, M> forward(const Tensor<T, M>& x) {
        auto attn_out = mha.forward(x, x, x);
        auto res1 = ln1.forward(x + attn_out);
        auto ffn_out = ffn.forward(res1);
        auto res2 = ln2.forward(res1 + ffn_out);
        return res2;
    }
};

template<typename T, size_t M>
class TransformerEncoder {
public:
    std::vector<TransformerEncoderBlock<T, M>> blocks;

    // Constructor: num_layers = number of encoder blocks
    TransformerEncoder(size_t num_layers, size_t embed_dim, size_t num_heads, size_t ffn_hidden_dim) {
        for (size_t i = 0; i < num_layers; i++) {
            blocks.emplace_back(embed_dim, num_heads, ffn_hidden_dim);
        }
    }

    Tensor<T, M> forward(const Tensor<T, M>& x) {
        Tensor<T, M> out = x;
        for (auto& block : blocks) {
            out = block.forward(out);
        }
        return out;
    }
};
