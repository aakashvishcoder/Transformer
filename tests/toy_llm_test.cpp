#define TOY_LLM_NO_MAIN
#include "../toy_llm.cpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

static float single_window_loss(TinyTransformer& model,
                                const std::vector<int>& ids,
                                const std::vector<int>& targets) {
    TinyTransformer::ForwardCache cache = model.forward_cache(ids);
    TinyTransformer::OutputHeadBackward head = model.backprop_output_head(cache, targets);
    return head.loss;
}

static size_t argmax_abs_index(const std::vector<float>& v) {
    size_t best = 0;
    float best_abs = std::fabs(v[0]);
    for (size_t i = 1; i < v.size(); ++i) {
        float a = std::fabs(v[i]);
        if (a > best_abs) {
            best_abs = a;
            best = i;
        }
    }
    return best;
}

static float finite_difference_param(std::vector<float>& param,
                                     size_t idx,
                                     const std::function<float()>& loss_fn,
                                     float eps = 1e-3f) {
    float orig = param[idx];
    param[idx] = orig + eps;
    float lp = loss_fn();
    param[idx] = orig - eps;
    float lm = loss_fn();
    param[idx] = orig;
    return (lp - lm) / (2.0f * eps);
}

struct ToyTestRunner {
    int passed = 0;
    int failed = 0;

    void run(const std::string& name, const std::function<void()>& fn) {
        try {
            fn();
            ++passed;
            std::cout << "[PASS] " << name << "\n";
        } catch (const std::exception& e) {
            ++failed;
            std::cout << "[FAIL] " << name << ": " << e.what() << "\n";
        }
    }
};

int main() {
    ToyTestRunner runner;

    runner.run("Causal mask is strictly lower-triangular", [] {
        CharTokenizer tok("abc abc");
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 42);
        std::vector<int> ids = tok.encode("abca");

        TinyTransformer::ForwardCache cache = model.forward_cache(ids);
        const int seq = static_cast<int>(ids.size());

        for (int t = 0; t < seq; ++t) {
            float row_sum = 0.0f;
            for (int j = 0; j < seq; ++j) {
                float w = cache.attn_weights[static_cast<size_t>(t * seq + j)];
                row_sum += w;
                if (j > t) {
                    assert(std::fabs(w) < 1e-7f);
                }
            }
            assert(std::fabs(row_sum - 1.0f) < 1e-4f);
        }
    });

    runner.run("Output head backward populates hidden gradients", [] {
        std::string corpus = "abcdabcdabcd";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 19);

        std::vector<int> ids = tok.encode("abcd");
        TinyTransformer::ForwardCache cache = model.forward_cache(ids);
        TinyTransformer::OutputHeadBackward grads = model.backprop_output_head(cache, ids);

        assert(grads.grad_h2.size() == cache.h2.size());
        assert(grads.grad_out_proj.size() == static_cast<size_t>(model.d_model * model.vocab));
        assert(grads.grad_out_bias.size() == static_cast<size_t>(model.vocab));

        float hidden_norm = 0.0f;
        for (float g : grads.grad_h2) {
            hidden_norm += std::fabs(g);
        }
        assert(hidden_norm > 0.0f);
    });

    runner.run("FFN backward gradients are non-zero and shaped", [] {
        std::string corpus = "abcdabcdabcd";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 31);

        std::vector<int> ids = tok.encode("abcd");
        TinyTransformer::ForwardCache cache = model.forward_cache(ids);
        TinyTransformer::OutputHeadBackward head_grads = model.backprop_output_head(cache, ids);
        TinyTransformer::FFNBackward ffn_grads = model.backprop_ffn(cache, head_grads.grad_h2);

        assert(ffn_grads.grad_w1.size() == model.w1.size());
        assert(ffn_grads.grad_w2.size() == model.w2.size());
        assert(ffn_grads.grad_h1.size() == cache.h1.size());

        float w1_norm = 0.0f;
        float w2_norm = 0.0f;
        for (float g : ffn_grads.grad_w1) w1_norm += std::fabs(g);
        for (float g : ffn_grads.grad_w2) w2_norm += std::fabs(g);

        assert(w1_norm > 0.0f);
        assert(w2_norm > 0.0f);
    });

    runner.run("Attention projection backward gradients are non-zero and shaped", [] {
        std::string corpus = "abcdabcdabcd";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 55);

        std::vector<int> ids = tok.encode("abcd");
        TinyTransformer::ForwardCache cache = model.forward_cache(ids);
        TinyTransformer::OutputHeadBackward head_grads = model.backprop_output_head(cache, ids);
        TinyTransformer::FFNBackward ffn_grads = model.backprop_ffn(cache, head_grads.grad_h2);
        TinyTransformer::AttentionProjBackward attn_grads = model.backprop_attention_proj(cache, ffn_grads.grad_h1);

        assert(attn_grads.grad_wo.size() == model.wo.size());
        assert(attn_grads.grad_attn_out.size() == cache.attn_out.size());

        float wo_norm = 0.0f;
        float attn_out_norm = 0.0f;
        for (float g : attn_grads.grad_wo) wo_norm += std::fabs(g);
        for (float g : attn_grads.grad_attn_out) attn_out_norm += std::fabs(g);

        assert(wo_norm > 0.0f);
        assert(attn_out_norm > 0.0f);
    });

    runner.run("Attention core backward gradients are non-zero and shaped", [] {
        std::string corpus = "abcdabcdabcd";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 77);

        std::vector<int> ids = tok.encode("abcd");
        TinyTransformer::ForwardCache cache = model.forward_cache(ids);
        TinyTransformer::OutputHeadBackward head_grads = model.backprop_output_head(cache, ids);
        TinyTransformer::FFNBackward ffn_grads = model.backprop_ffn(cache, head_grads.grad_h2);
        TinyTransformer::AttentionProjBackward attn_grads = model.backprop_attention_proj(cache, ffn_grads.grad_h1);
        TinyTransformer::AttentionCoreBackward core_grads = model.backprop_attention_core(cache, attn_grads.grad_attn_out);

        assert(core_grads.grad_wq.size() == model.wq.size());
        assert(core_grads.grad_wk.size() == model.wk.size());
        assert(core_grads.grad_wv.size() == model.wv.size());
        assert(core_grads.grad_x.size() == cache.x.size());

        float q_norm = 0.0f;
        float k_norm = 0.0f;
        float v_norm = 0.0f;
        float x_norm = 0.0f;
        for (float g : core_grads.grad_wq) q_norm += std::fabs(g);
        for (float g : core_grads.grad_wk) k_norm += std::fabs(g);
        for (float g : core_grads.grad_wv) v_norm += std::fabs(g);
        for (float g : core_grads.grad_x) x_norm += std::fabs(g);

        assert(q_norm > 0.0f);
        assert(k_norm > 0.0f);
        assert(v_norm > 0.0f);
        assert(x_norm > 0.0f);
    });

    runner.run("Embedding backward gradients are non-zero and shaped", [] {
        std::string corpus = "abcdabcdabcd";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 99);

        std::vector<int> ids = tok.encode("abcd");
        TinyTransformer::ForwardCache cache = model.forward_cache(ids);
        TinyTransformer::OutputHeadBackward head_grads = model.backprop_output_head(cache, ids);
        TinyTransformer::FFNBackward ffn_grads = model.backprop_ffn(cache, head_grads.grad_h2);
        TinyTransformer::AttentionProjBackward attn_grads = model.backprop_attention_proj(cache, ffn_grads.grad_h1);
        TinyTransformer::AttentionCoreBackward core_grads = model.backprop_attention_core(cache, attn_grads.grad_attn_out);
        TinyTransformer::EmbeddingBackward emb_grads = model.backprop_embeddings(ids, core_grads.grad_x);

        assert(emb_grads.grad_token_emb.size() == model.token_emb.size());
        assert(emb_grads.grad_pos_emb.size() == model.pos_emb.size());

        float tok_norm = 0.0f;
        float pos_norm = 0.0f;
        for (float g : emb_grads.grad_token_emb) tok_norm += std::fabs(g);
        for (float g : emb_grads.grad_pos_emb) pos_norm += std::fabs(g);

        assert(tok_norm > 0.0f);
        assert(pos_norm > 0.0f);
    });

    runner.run("Finite difference gradients for wq wk wv and token_emb", [] {
        std::string corpus = "abcdabcdabcd";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 6, 10, 16, 123);

        std::vector<int> ids = tok.encode("abcd");
        std::vector<int> targets = tok.encode("bcda");

        TinyTransformer::ForwardCache cache = model.forward_cache(ids);
        TinyTransformer::OutputHeadBackward head_grads = model.backprop_output_head(cache, targets);
        TinyTransformer::FFNBackward ffn_grads = model.backprop_ffn(cache, head_grads.grad_h2);
        TinyTransformer::AttentionProjBackward attn_grads = model.backprop_attention_proj(cache, ffn_grads.grad_h1);
        TinyTransformer::AttentionCoreBackward core_grads = model.backprop_attention_core(cache, attn_grads.grad_attn_out);
        TinyTransformer::EmbeddingBackward emb_grads = model.backprop_embeddings(ids, core_grads.grad_x);

        auto loss_fn = [&]() {
            return single_window_loss(model, ids, targets);
        };

        size_t idx_wq = argmax_abs_index(core_grads.grad_wq);
        size_t idx_wk = argmax_abs_index(core_grads.grad_wk);
        size_t idx_wv = argmax_abs_index(core_grads.grad_wv);
        size_t idx_te = argmax_abs_index(emb_grads.grad_token_emb);

        float g_wq = core_grads.grad_wq[idx_wq];
        float g_wk = core_grads.grad_wk[idx_wk];
        float g_wv = core_grads.grad_wv[idx_wv];
        float g_te = emb_grads.grad_token_emb[idx_te];

        const float inv_seq = 1.0f / static_cast<float>(ids.size());
        g_wq *= inv_seq;
        g_wk *= inv_seq;
        g_wv *= inv_seq;
        g_te *= inv_seq;

        assert(std::fabs(g_wq) > 1e-7f);
        assert(std::fabs(g_wk) > 1e-7f);
        assert(std::fabs(g_wv) > 1e-7f);
        assert(std::fabs(g_te) > 1e-7f);

        float n_wq = finite_difference_param(model.wq, idx_wq, loss_fn);
        float n_wk = finite_difference_param(model.wk, idx_wk, loss_fn);
        float n_wv = finite_difference_param(model.wv, idx_wv, loss_fn);
        float n_te = finite_difference_param(model.token_emb, idx_te, loss_fn);

        const float tol = 6e-2f;
        assert(std::fabs(g_wq - n_wq) < tol);
        assert(std::fabs(g_wk - n_wk) < tol);
        assert(std::fabs(g_wv - n_wv) < tol);
        assert(std::fabs(g_te - n_te) < tol);
    });

    runner.run("Finite difference gradients for w1 w2 and out_proj", [] {
        std::string corpus = "abcdabcdabcd";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 6, 10, 16, 321);

        std::vector<int> ids = tok.encode("abcd");
        std::vector<int> targets = tok.encode("bcda");

        TinyTransformer::ForwardCache cache = model.forward_cache(ids);
        TinyTransformer::OutputHeadBackward head_grads = model.backprop_output_head(cache, targets);
        TinyTransformer::FFNBackward ffn_grads = model.backprop_ffn(cache, head_grads.grad_h2);

        auto loss_fn = [&]() {
            return single_window_loss(model, ids, targets);
        };

        size_t idx_w1 = argmax_abs_index(ffn_grads.grad_w1);
        size_t idx_w2 = argmax_abs_index(ffn_grads.grad_w2);
        size_t idx_op = argmax_abs_index(head_grads.grad_out_proj);

        float g_w1 = ffn_grads.grad_w1[idx_w1];
        float g_w2 = ffn_grads.grad_w2[idx_w2];
        float g_op = head_grads.grad_out_proj[idx_op];

        const float inv_seq = 1.0f / static_cast<float>(ids.size());
        g_w1 *= inv_seq;
        g_w2 *= inv_seq;
        g_op *= inv_seq;

        assert(std::fabs(g_w1) > 1e-7f);
        assert(std::fabs(g_w2) > 1e-7f);
        assert(std::fabs(g_op) > 1e-7f);

        float n_w1 = finite_difference_param(model.w1, idx_w1, loss_fn);
        float n_w2 = finite_difference_param(model.w2, idx_w2, loss_fn);
        float n_op = finite_difference_param(model.out_proj, idx_op, loss_fn);

        const float tol_ffn = 8e-2f;
        const float tol_out = 6e-2f;
        assert(std::fabs(g_w1 - n_w1) < tol_ffn);
        assert(std::fabs(g_w2 - n_w2) < tol_ffn);
        assert(std::fabs(g_op - n_op) < tol_out);
    });

    runner.run("Next-token training reduces loss", [] {
        std::string corpus = "abababababababababababab";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 123);

        std::vector<int> ids = tok.encode(corpus);
        float before = model.evaluate_next_token_loss(ids, 8);
        float after = model.train_next_token(ids, 25, 8, 0.05f, 0);

        assert(after < before);
    });

    runner.run("Generation is deterministic with fixed seed", [] {
        std::string corpus = "hello hello hello world world ";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 7);
        std::vector<int> ids = tok.encode(corpus);
        (void)model.train_next_token(ids, 10, 8, 0.04f, 0);

        std::vector<int> prompt = tok.encode("he");
        std::vector<int> s1 = model.generate(prompt, 20, 8, 0.9f, 3, 11);
        std::vector<int> s2 = model.generate(prompt, 20, 8, 0.9f, 3, 11);
        assert(s1 == s2);

        std::vector<int> g1 = model.generate(prompt, 20, 8, 0.0f, 1, 1);
        std::vector<int> g2 = model.generate(prompt, 20, 8, 1.0f, 1, 999);
        assert(g1 == g2);
    });

    std::cout << "Toy LLM tests: passed=" << runner.passed << " failed=" << runner.failed << "\n";
    return runner.failed == 0 ? 0 : 1;
}
