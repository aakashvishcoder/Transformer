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

template <typename T, size_t M>
class TransformerDecoderBlock {
public:
    MultiHeadAttentionLayer<T,M> mh1;
    LayerNormalization<T,M> ln1;
    MultiHeadAttentionLayer<T,M> mh2;
    LayerNormalization<T,M> ln2;
    FeedForward<T> f1;
    LayerNormalization<T,M> ln3;

    TransformerDecoderBlock(size_t embed_dim, size_t num_heads, size_t ffn_hidden_dim)
        : mh1(embed_dim, num_heads, embed_dim),
          ln1(embed_dim, {1, 1, embed_dim}),
          mh2(embed_dim, num_heads, embed_dim),
          f1(embed_dim, ffn_hidden_dim),
          ln2(embed_dim, {1, 1, embed_dim}),
          ln3(embed_dim, {1, 1, embed_dim}) {}
    
    Tensor<T,M> forward(const Tensor<T,M>& queries, const Tensor<T,M>& keys, const Tensor<T,M>& x) {
        auto attn_out_1 = mh1.forward(x, x, x);
        auto res1 = ln1.forward(x + attn_out_1);
        auto attn_out_2 = mh2.forward(queries, keys, res1);
        auto res2 = ln2.forward(res1 + attn_out_2);
        auto ffn_out = f1.forward(res2);
        auto res3 = ln3.forward(ffn_out + res2);
        return res3;
    }
};

template <typename T, size_t M>
class TransformerDecoder {
public:
    std::vector<TransformerDecoderBlock<T,M>> blocks;
    // Constructor
    TransformerDecoder(size_t num_layers, size_t embed_dim, size_t num_heads, size_t ffn_hidden_dim) {
        for(size_t i = 0; i < num_layers; i++) {
            blocks.emplace_back(embed_dim, num_heads, ffn_hidden_dim);
        }
    }

    Tensor<T,M> forward(const Tensor<T,M>& qr, const Tensor<T,M>& x,size_t embed_size) {
        Tensor<T, M> out = x;
        for (auto& block : blocks) {
            out = block.forward(qr,qr,out);
        }
        Dense<T> l1(embed_size,embed_size);
        out = l1.forward(out);
        auto probabilities = Activations::Softmax(out,M-1);
        return probabilities;
    }
};