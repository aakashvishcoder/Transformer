#define TOY_LLM_NO_MAIN
#include "../toy_llm.cpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
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

    runner.run("LayerNorm rows produce near-zero mean", [] {
        std::vector<float> in = {
            1.0f, 2.0f, 3.0f,
            2.0f, 2.0f, 2.0f
        };
        std::vector<float> out;
        TinyTransformer::layernorm_rows(in, 2, 3, out);
        assert(out.size() == in.size());

        for (int r = 0; r < 2; ++r) {
            float mean = 0.0f;
            for (int c = 0; c < 3; ++c) {
                mean += out[static_cast<size_t>(r * 3 + c)];
            }
            mean /= 3.0f;
            assert(std::fabs(mean) < 1e-4f);
        }
    });

    runner.run("Forward supports multi-layer pre-norm mode", [] {
        std::string corpus = "abcdabcdabcd";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 404, 2, true);

        std::vector<int> ids = tok.encode("abcd");
        std::vector<float> logits = model.forward(ids);
        assert(static_cast<int>(logits.size()) == tok.vocab_size());

        // Deterministic for fixed weights and same input.
        std::vector<float> logits2 = model.forward(ids);
        assert(logits == logits2);
    });

    runner.run("Forward supports multi-head mode", [] {
        std::string corpus = "abcdabcdabcd";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 505, 1, false, 2);

        std::vector<int> ids = tok.encode("abcd");
        std::vector<float> logits = model.forward(ids);
        assert(static_cast<int>(logits.size()) == tok.vocab_size());

        // Ensure cached attention rows still normalize after head-averaging.
        TinyTransformer::ForwardCache cache = model.forward_cache(ids);
        const int seq = static_cast<int>(ids.size());
        for (int t = 0; t < seq; ++t) {
            float sum = 0.0f;
            for (int j = 0; j < seq; ++j) sum += cache.attn_weights()[static_cast<size_t>(t * seq + j)];
            assert(std::fabs(sum - 1.0f) < 1e-4f);
        }
    });

    runner.run("Causal mask is strictly lower-triangular", [] {
        CharTokenizer tok("abc abc");
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 42);
        std::vector<int> ids = tok.encode("abca");

        TinyTransformer::ForwardCache cache = model.forward_cache(ids);
        const int seq = static_cast<int>(ids.size());

        for (int t = 0; t < seq; ++t) {
            float row_sum = 0.0f;
            for (int j = 0; j < seq; ++j) {
                float w = cache.attn_weights()[static_cast<size_t>(t * seq + j)];
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

        assert(grads.grad_h2.size() == cache.h2().size());
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
        assert(ffn_grads.grad_h1.size() == cache.h1().size());

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
        assert(attn_grads.grad_attn_out.size() == cache.attn_out().size());

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
        assert(core_grads.grad_x.size() == cache.x().size());

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

    runner.run("Finite difference gradients for multi-head attention core", [] {
        std::string corpus = "abcdabcdabcd";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 8, 10, 16, 444, 1, false, 2);

        std::vector<int> ids = tok.encode("abcd");
        std::vector<int> targets = tok.encode("bcda");

        TinyTransformer::ForwardCache cache = model.forward_cache(ids);
        TinyTransformer::OutputHeadBackward head_grads = model.backprop_output_head(cache, targets);
        TinyTransformer::FFNBackward ffn_grads = model.backprop_ffn(cache, head_grads.grad_h2);
        TinyTransformer::AttentionProjBackward attn_grads = model.backprop_attention_proj(cache, ffn_grads.grad_h1);
        TinyTransformer::AttentionCoreBackward core_grads = model.backprop_attention_core(cache, attn_grads.grad_attn_out);

        auto loss_fn = [&]() {
            return single_window_loss(model, ids, targets);
        };

        size_t idx_wq = argmax_abs_index(core_grads.grad_wq);
        size_t idx_wk = argmax_abs_index(core_grads.grad_wk);
        size_t idx_wv = argmax_abs_index(core_grads.grad_wv);

        float g_wq = core_grads.grad_wq[idx_wq];
        float g_wk = core_grads.grad_wk[idx_wk];
        float g_wv = core_grads.grad_wv[idx_wv];

        const float inv_seq = 1.0f / static_cast<float>(ids.size());
        g_wq *= inv_seq;
        g_wk *= inv_seq;
        g_wv *= inv_seq;

        assert(std::fabs(g_wq) > 1e-7f);
        assert(std::fabs(g_wk) > 1e-7f);
        assert(std::fabs(g_wv) > 1e-7f);

        float n_wq = finite_difference_param(model.wq, idx_wq, loss_fn);
        float n_wk = finite_difference_param(model.wk, idx_wk, loss_fn);
        float n_wv = finite_difference_param(model.wv, idx_wv, loss_fn);

        const float tol = 8e-2f;
        assert(std::fabs(g_wq - n_wq) < tol);
        assert(std::fabs(g_wk - n_wk) < tol);
        assert(std::fabs(g_wv - n_wv) < tol);
    });

    runner.run("Checkpoint save and load round-trip", [] {
        std::string corpus = "hello world hello world";
        CharTokenizer tok(corpus);
        TinyTransformer m1(tok.vocab_size(), 8, 16, 16, 2026);
        TinyTransformer m2(tok.vocab_size(), 8, 16, 16, 7);

        std::vector<int> ids = tok.encode("hello ");
        std::vector<int> train_ids = tok.encode(corpus);
        (void)m1.train_next_token(train_ids, 2, 8, 0.03f, 0);

        const std::string ckpt_path = "toy_llm_test_checkpoint.bin";
        bool saved = m1.save_checkpoint(ckpt_path);
        assert(saved);

        bool loaded = m2.load_checkpoint(ckpt_path);
        assert(loaded);

        std::vector<float> l1 = m1.forward(ids);
        std::vector<float> l2 = m2.forward(ids);
        assert(l1.size() == l2.size());
        for (size_t i = 0; i < l1.size(); ++i) {
            assert(std::fabs(l1[i] - l2[i]) < 1e-7f);
        }

        std::remove(ckpt_path.c_str());
    });

    runner.run("Checkpoint includes model mode metadata", [] {
        std::string corpus = "hello world hello world";
        CharTokenizer tok(corpus);
        TinyTransformer m1(tok.vocab_size(), 8, 16, 16, 2027, 2, true, 2);
        TinyTransformer wrong(tok.vocab_size(), 8, 16, 16, 7, 1, false, 1);
        TinyTransformer right(tok.vocab_size(), 8, 16, 16, 7, 2, true, 2);

        const std::string ckpt_path = "toy_llm_test_checkpoint_mode.bin";
        assert(m1.save_checkpoint(ckpt_path));
        assert(!wrong.load_checkpoint(ckpt_path));
        assert(right.load_checkpoint(ckpt_path));
        std::remove(ckpt_path.c_str());
    });

    runner.run("Resume training keeps optimizer state continuity", [] {
        std::string corpus = "abcabcabcabcabcabcabcabcabcabc";
        CharTokenizer tok(corpus);

        TinyTransformer full(tok.vocab_size(), 8, 16, 16, 3030);
        TinyTransformer split_a(tok.vocab_size(), 8, 16, 16, 3030);
        TinyTransformer split_b(tok.vocab_size(), 8, 16, 16, 99);

        std::vector<int> ids = tok.encode(corpus);
        (void)full.train_next_token(ids, 12, 10, 0.03f, 0, "adamw", 0, 1.0f, 1e-4f, 2, 0.5f);

        (void)split_a.train_next_token(ids, 6, 10, 0.03f, 0, "adamw", 0, 1.0f, 1e-4f, 2, 0.5f);
        const std::string ckpt_path = "toy_llm_resume_state.bin";
        assert(split_a.save_checkpoint(ckpt_path));

        assert(split_b.load_checkpoint(ckpt_path));
        (void)split_b.train_next_token(ids, 6, 10, 0.03f, 0, "adamw", 0, 1.0f, 1e-4f, 2, 0.5f, false, false, 0.0f, false, true);

        std::vector<int> prompt = tok.encode("abc");
        std::vector<float> logits_full = full.forward(prompt);
        std::vector<float> logits_resumed = split_b.forward(prompt);
        assert(logits_full.size() == logits_resumed.size());
        for (size_t i = 0; i < logits_full.size(); ++i) {
            assert(std::fabs(logits_full[i] - logits_resumed[i]) < 1e-5f);
        }

        std::remove(ckpt_path.c_str());
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

    runner.run("Next-token training supports AdamW with scheduler", [] {
        std::string corpus = "abababababababababababab";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 124);

        std::vector<int> ids = tok.encode(corpus);
        float before = model.evaluate_next_token_loss(ids, 8);
        float after = model.train_next_token(ids, 20, 8, 0.03f, 0, "adamw", 10, 0.2f, 1e-4f);

        assert(after < before);
    });

    runner.run("Next-token training supports minibatch with grad clipping", [] {
        std::string corpus = "abcabcabcabcabcabcabcabcabcabc";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 125, 1, false, 2);

        std::vector<int> ids = tok.encode(corpus);
        float before = model.evaluate_next_token_loss(ids, 10);
        float after = model.train_next_token(ids, 20, 10, 0.03f, 0, "adamw", 8, 0.2f, 1e-4f, 4, 1.0f);

        assert(after < before);
    });

    runner.run("AdamW converges at least as well as SGD on same seed", [] {
        std::string corpus = "abcabcabcabcabcabcabcabcabcabc";
        CharTokenizer tok(corpus);
        TinyTransformer sgd_model(tok.vocab_size(), 8, 16, 16, 127, 1, false, 2);
        TinyTransformer adam_model(tok.vocab_size(), 8, 16, 16, 127, 1, false, 2);

        std::vector<int> ids = tok.encode(corpus);
        float sgd_after = sgd_model.train_next_token(ids, 18, 10, 0.03f, 0, "sgd", 0, 0.1f, 1e-4f, 4, 1.0f);
        float adam_after = adam_model.train_next_token(ids, 18, 10, 0.03f, 0, "adamw", 8, 0.2f, 1e-4f, 4, 1.0f);

        assert(adam_after <= sgd_after + 0.08f);
    });

    runner.run("Training diagnostics flags execute without error", [] {
        std::string corpus = "abababababababab";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 128);

        std::vector<int> ids = tok.encode(corpus);
        float after = model.train_next_token(ids, 4, 8, 0.03f, 1, "adamw", 2, 0.2f, 1e-4f, 2, 0.5f, true, true);
        assert(after > 0.0f);
    });

    runner.run("Training supports validation split and val perplexity reporting", [] {
        std::string corpus = "abcabcabcabcabcabcabcabc";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 129);

        std::vector<int> ids = tok.encode(corpus);
        auto split = split_train_val_ids(ids, 0.25f);
        assert(split.first.size() >= 2);
        assert(split.second.size() >= 2);

        float before = model.evaluate_next_token_loss(split.first, 8);
        float after = model.train_next_token(ids,
                                             8,
                                             8,
                                             0.03f,
                                             1,
                                             "adamw",
                                             2,
                                             0.2f,
                                             1e-4f,
                                             2,
                                             0.5f,
                                             true,
                                             true,
                                             0.25f,
                                             true);
        assert(after < before);

        float val_loss = model.evaluate_next_token_loss(split.second, 8);
        assert(std::isfinite(val_loss));
        assert(std::exp(val_loss) > 0.0f);
    });

    runner.run("Next-token training rejects invalid batch size", [] {
        std::string corpus = "abababababab";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 126);

        std::vector<int> ids = tok.encode(corpus);
        bool threw = false;
        try {
            (void)model.train_next_token(ids, 2, 8, 0.03f, 0, "sgd", 0, 0.1f, 0.0f, 0, 0.0f);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        assert(threw);
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

    runner.run("Top-p and repetition-penalty generation path is stable", [] {
        std::string corpus = "hello hello hello world world ";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 707);
        std::vector<int> ids = tok.encode(corpus);
        (void)model.train_next_token(ids, 10, 8, 0.04f, 0);

        std::vector<int> prompt = tok.encode("he");
        std::vector<int> s1 = model.generate(prompt, 24, 8, 0.95f, 0, 99, 0.75f, 1.15f);
        std::vector<int> s2 = model.generate(prompt, 24, 8, 0.95f, 0, 99, 0.75f, 1.15f);
        assert(s1 == s2);
        assert(s1.size() == prompt.size() + 24u);
    });

    runner.run("Vocab builder round-trip loads into subword tokenizer", [] {
        std::string corpus = "hello world hello tiny world";
        std::vector<std::string> vocab = build_vocab_from_corpus_frequency(corpus, 32);
        assert(!vocab.empty());
        assert(vocab[0] == "<unk>");

        const std::string vocab_path = "toy_llm_vocab_builder_test.txt";
        write_vocab_file(vocab_path, vocab);

        std::vector<std::string> loaded = SubwordTokenizer::load_vocab_file(vocab_path);
        assert(!loaded.empty());
        SubwordTokenizer sub_tok(loaded);

        std::vector<int> encoded = sub_tok.encode("hello world");
        assert(!encoded.empty());
        std::string decoded = sub_tok.decode(encoded);
        assert(!decoded.empty());

        std::remove(vocab_path.c_str());
    });

    runner.run("Run config JSON parser reads scalars and arrays", [] {
        const std::string cfg_path = "toy_llm_run_config_test.json";
        {
            std::ofstream out(cfg_path);
            out << "{\n";
            out << "  \"num_layers\": 2,\n";
            out << "  \"optimizer\": \"adamw\",\n";
            out << "  \"report_lr\": true,\n";
            out << "  \"lr\": 0.003,\n";
            out << "  \"data\": [\"data.csv\", \"data.csv\"]\n";
            out << "}\n";
        }

        RunConfigJson cfg = load_run_config_json(cfg_path);
        assert(cfg.scalars.at("num_layers") == "2");
        assert(cfg.scalars.at("optimizer") == "adamw");
        assert(cfg.scalars.at("report_lr") == "true");
        assert(cfg.scalars.at("lr") == "0.003");
        assert(cfg.string_arrays.at("data").size() == 2u);
        assert(cfg.string_arrays.at("data")[0] == "data.csv");

        std::remove(cfg_path.c_str());
    });

    runner.run("Strict config validation rejects unknown keys", [] {
        RunConfigJson ok_cfg;
        ok_cfg.scalars["epochs"] = "2";
        ok_cfg.string_arrays["data"] = {"data.csv"};
        validate_run_config_keys(ok_cfg, true);

        RunConfigJson bad_cfg;
        bad_cfg.scalars["epochs"] = "2";
        bad_cfg.scalars["optimzer"] = "adamw";
        bool threw = false;
        bool suggested = false;
        try {
            validate_run_config_keys(bad_cfg, true);
        } catch (const std::runtime_error& e) {
            threw = true;
            std::string msg = e.what();
            suggested = msg.find("optimizer") != std::string::npos;
        }
        assert(threw);
        assert(suggested);
    });

    runner.run("Config key suggestion helper finds close key", [] {
        std::vector<std::string> s1 = suggest_config_keys("temprature");
        assert(!s1.empty());
        assert(s1[0] == "temperature");
        std::vector<std::string> s2 = suggest_config_keys("batc_size");
        assert(!s2.empty());
        assert(s2[0] == "batch_size");
    });

    runner.run("Config key suggestion helper can return multiple candidates", [] {
        std::vector<std::string> s = suggest_config_keys("top", 3);
        bool has_top_k = false;
        bool has_top_p = false;
        for (const std::string& x : s) {
            if (x == "top_k") has_top_k = true;
            if (x == "top_p") has_top_p = true;
        }
        assert(has_top_k);
        assert(has_top_p);
    });

    runner.run("Non-strict config warning collector reports unknown keys", [] {
        RunConfigJson cfg;
        cfg.scalars["optimzer"] = "adamw";
        cfg.scalars["epochs"] = "2";
        cfg.string_arrays["data"] = {"data.csv"};
        cfg.string_arrays["warnngs"] = {"x"};

        std::vector<std::string> warnings = collect_non_strict_config_warnings(cfg);
        assert(warnings.size() == 2u);
        bool has_optimizer_hint = false;
        bool has_warnngs_key_warning = false;
        for (const std::string& w : warnings) {
            if (w.find("optimizer") != std::string::npos) has_optimizer_hint = true;
            if (w.find("warnngs") != std::string::npos) has_warnngs_key_warning = true;
        }
        assert(has_optimizer_hint);
        assert(has_warnngs_key_warning);
    });

    runner.run("Effective config JSON writer round-trips key fields", [] {
        EffectiveRunConfig cfg;
        cfg.num_layers = 2;
        cfg.data = {"data.csv", "data.csv"};
        cfg.warnings = {"Unknown config key (ignored): optimzer. Did you mean 'optimizer'?"};
        cfg.vocab = "vocab.txt";
        cfg.prompt = "hello";
        cfg.optimizer = "adamw";
        cfg.epochs = 12;
        cfg.lr = 0.003f;
        cfg.report_lr = true;
        cfg.report_grad = true;
        cfg.save_config = "roundtrip.json";
        cfg.dry_run = true;
        cfg.strict_config = true;

        const std::string out_path = "toy_llm_effective_config_test.json";
        write_effective_run_config_json(out_path, cfg);
        RunConfigJson loaded = load_run_config_json(out_path);

        assert(loaded.scalars.at("num_layers") == "2");
        assert(loaded.scalars.at("optimizer") == "adamw");
        assert(loaded.scalars.at("epochs") == "12");
        assert(loaded.scalars.at("report_lr") == "true");
        assert(loaded.scalars.at("report_grad") == "true");
        assert(loaded.scalars.at("dry_run") == "true");
        assert(loaded.scalars.at("strict_config") == "true");
        assert(loaded.scalars.at("save_config") == "roundtrip.json");
        assert(loaded.string_arrays.at("data").size() == 2u);
        assert(loaded.string_arrays.at("warnings").size() == 1u);
        assert(loaded.string_arrays.at("warnings")[0].find("optimzer") != std::string::npos);
        assert(std::fabs(std::stof(loaded.scalars.at("lr")) - 0.003f) < 1e-7f);

        std::remove(out_path.c_str());
    });

    runner.run("Training with LayerNorm reduces loss", [] {
        std::string corpus = "abababababababababababab";
        CharTokenizer tok(corpus);
        TinyTransformer model(tok.vocab_size(), 8, 16, 16, 1234, 1, true);

        std::vector<int> ids = tok.encode(corpus);
        float before = model.evaluate_next_token_loss(ids, 8);
        (void)model.train_next_token(ids, 30, 8, 0.04f, 0);
        float after = model.evaluate_next_token_loss(ids, 8);

        assert(after < before);
        assert(std::fabs(after - before) > 0.1f);
    });

    std::cout << "Toy LLM tests: passed=" << runner.passed << " failed=" << runner.failed << "\n";
    return runner.failed == 0 ? 0 : 1;
}
