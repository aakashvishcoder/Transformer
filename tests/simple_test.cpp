#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include "tensor_minimal.hpp"

static float finite_difference_gradient(
    const std::function<float(const std::vector<float>&)>& fn,
    const std::vector<float>& x,
    size_t idx,
    float eps = 1e-3f) {
    std::vector<float> xp = x;
    std::vector<float> xm = x;
    xp[idx] += eps;
    xm[idx] -= eps;
    return (fn(xp) - fn(xm)) / (2.0f * eps);
}

// Simple test framework
struct TestRunner {
    std::vector<std::pair<std::string, std::function<void()>>> tests;
    int passed = 0;
    int failed = 0;
    
    void add_test(const std::string& name, std::function<void()> test_fn) {
        tests.push_back({name, test_fn});
    }
    
    void run_all() {
        std::cout << "Running " << tests.size() << " tests...\n\n";
        for (const auto& [name, test_fn] : tests) {
            try {
                test_fn();
                passed++;
                std::cout << "[PASS] " << name << "\n";
            } catch (const std::exception& e) {
                failed++;
                std::cout << "[FAIL] " << name << ": " << e.what() << "\n";
            }
        }
        print_summary();
    }
    
    void print_summary() {
        std::cout << "\n" << std::string(50, '=') << "\n";
        std::cout << "Tests passed: " << passed << "\n";
        std::cout << "Tests failed: " << failed << "\n";
        std::cout << "Total tests:  " << tests.size() << "\n";
        std::cout << std::string(50, '=') << "\n";
    }
};

static float linear_dataset_mse(const Tensor<float>& w,
                                const Tensor<float>& b,
                                const std::vector<float>& xs,
                                const std::vector<float>& ys) {
    float total = 0.0f;
    for (size_t i = 0; i < xs.size(); ++i) {
        float pred = xs[i] * w.data[0] + b.data[0];
        float diff = pred - ys[i];
        total += diff * diff;
    }
    return total / static_cast<float>(xs.size());
}



int main() {
    TestRunner runner;
    
    // Test 1: Tensor creation and shape
    runner.add_test("Tensor creation and shape", [] {
        Tensor<float> t({2, 3, 4});
        assert(t.shape.size() == 3);
        assert(t.shape[0] == 2);
        assert(t.shape[1] == 3);
        assert(t.shape[2] == 4);
        assert(t.data.size() == 24);  // 2 * 3 * 4
    });
    
    // Test 2: Tensor numel calculation
    runner.add_test("Tensor numel calculation", [] {
        std::vector<size_t> shape1 = {2, 3};
        assert(Tensor<float>::numel_from_shape(shape1) == 6);
        
        std::vector<size_t> shape2 = {5, 4, 3};
        assert(Tensor<float>::numel_from_shape(shape2) == 60);
        
        std::vector<size_t> shape3 = {1};
        assert(Tensor<float>::numel_from_shape(shape3) == 1);
    });
    
    // Test 3: Tensor fill_random
    runner.add_test("Tensor fill_random", [] {
        Tensor<float> t({10});
        t.fill_random(-1.0f, 1.0f);
        
        assert(t.data.size() == 10);
        for (auto val : t.data) {
            assert(val >= -1.0f);
            assert(val <= 1.0f);
        }
    });
    
    // Test 4: Tensor default construction
    runner.add_test("Tensor default construction", [] {
        Tensor<float> t;
        assert(t.shape.empty());
        assert(t.data.empty());
    });
    
    // Test 5: Different data types
    runner.add_test("Tensor with different types", [] {
        Tensor<int> ti({2, 2});
        assert(ti.data.size() == 4);
        
        Tensor<double> td({3});
        assert(td.data.size() == 3);
    });
    
    // Test 6: Large tensor
    runner.add_test("Large tensor creation", [] {
        Tensor<float> t({100, 100, 10});
        assert(t.data.size() == 100 * 100 * 10);
        assert(t.shape.size() == 3);
    });

    // Test 7: Bounds-checked indexing
    runner.add_test("Bounds-checked indexing", [] {
        Tensor<float> t({2, 3});
        t.at({1, 2}) = 42.0f;
        assert(t.at({1, 2}) == 42.0f);

        bool threw = false;
        try {
            (void)t.at({2, 0});
        } catch (const std::out_of_range&) {
            threw = true;
        }
        assert(threw);
    });

    // Test 8: Stable softmax over last axis
    runner.add_test("Softmax normalizes rows", [] {
        Tensor<float> t({2, 3});
        t.data = {1.0f, 2.0f, 3.0f, 10.0f, 11.0f, 12.0f};

        Tensor<float> s = t.softmax(-1);
        for (size_t row = 0; row < 2; ++row) {
            float sum = 0.0f;
            for (size_t col = 0; col < 3; ++col) {
                float v = s.data[row * 3 + col];
                assert(v >= 0.0f);
                assert(v <= 1.0f);
                sum += v;
            }
            assert(std::fabs(sum - 1.0f) < 1e-4f);
        }
    });

    // Test 9: Generic 3D transpose path
    runner.add_test("3D transpose permutation", [] {
        Tensor<int> t({2, 3, 4});
        for (size_t i = 0; i < t.numel(); ++i) {
            t.data[i] = static_cast<int>(i);
        }

        Tensor<int> out = t.transpose({1, 0, 2});
        assert(out.shape[0] == 3 && out.shape[1] == 2 && out.shape[2] == 4);
        assert(out.at({2, 1, 3}) == t.at({1, 2, 3}));
    });

    // Test 10: Batched matmul (3D)
    runner.add_test("Batched matmul 3D", [] {
        Tensor<float> a({2, 2, 3});
        Tensor<float> b({2, 3, 2});

        a.data = {
            1, 2, 3,
            4, 5, 6,
            1, 0, 2,
            0, 1, 2
        };

        b.data = {
            1, 2,
            3, 4,
            5, 6,
            1, 1,
            2, 2,
            3, 3
        };

        Tensor<float> c = matmul(a, b);
        assert(c.shape[0] == 2 && c.shape[1] == 2 && c.shape[2] == 2);

        assert(std::fabs(c.at({0, 0, 0}) - 22.0f) < 1e-5f);
        assert(std::fabs(c.at({0, 0, 1}) - 28.0f) < 1e-5f);
        assert(std::fabs(c.at({0, 1, 0}) - 49.0f) < 1e-5f);
        assert(std::fabs(c.at({0, 1, 1}) - 64.0f) < 1e-5f);

        assert(std::fabs(c.at({1, 0, 0}) - 7.0f) < 1e-5f);
        assert(std::fabs(c.at({1, 0, 1}) - 7.0f) < 1e-5f);
        assert(std::fabs(c.at({1, 1, 0}) - 8.0f) < 1e-5f);
        assert(std::fabs(c.at({1, 1, 1}) - 8.0f) < 1e-5f);
    });

    // Test 11: TensorView shares storage with tensor
    runner.add_test("TensorView aliasing", [] {
        Tensor<float> t({2, 3});
        t.zeros();
        auto v = t.view_ref({3, 2});
        v.at({2, 1}) = 9.0f;
        assert(std::fabs(t.data[5] - 9.0f) < 1e-6f);
    });

    // Test 12: Float16 and BFloat16 roundtrip
    runner.add_test("Low-precision dtype conversion", [] {
        Float16 h(1.5f);
        BFloat16 b(1.5f);
        assert(std::fabs(static_cast<float>(h) - 1.5f) < 0.02f);
        assert(std::fabs(static_cast<float>(b) - 1.5f) < 0.02f);
    });

    // Test 13: Mixed precision matmul accumulates in float
    runner.add_test("Mixed precision matmul", [] {
        Tensor<Float16> a({2, 2});
        Tensor<BFloat16> b({2, 2});
        a.data = {Float16(1.0f), Float16(2.0f), Float16(3.0f), Float16(4.0f)};
        b.data = {BFloat16(5.0f), BFloat16(6.0f), BFloat16(7.0f), BFloat16(8.0f)};

        Tensor<float> c = matmul_mixed<float>(a, b);
        assert(c.shape[0] == 2 && c.shape[1] == 2);
        assert(std::fabs(c.at({0, 0}) - 19.0f) < 0.1f);
        assert(std::fabs(c.at({0, 1}) - 22.0f) < 0.1f);
        assert(std::fabs(c.at({1, 0}) - 43.0f) < 0.2f);
        assert(std::fabs(c.at({1, 1}) - 50.0f) < 0.2f);
    });

    // Test 14: Gradient buffer preparation helpers
    runner.add_test("Gradient preparation helpers", [] {
        Tensor<float> t({2, 2});
        t.set_requires_grad(true);
        assert(t.requires_grad);
        assert(t.grad.size() == 4);

        t.add_grad({1.0f, 2.0f, 3.0f, 4.0f});
        assert(std::fabs(t.grad[0] - 1.0f) < 1e-6f);
        assert(std::fabs(t.grad[3] - 4.0f) < 1e-6f);

        t.zero_grad();
        for (float g : t.grad) {
            assert(std::fabs(g) < 1e-6f);
        }
    });

    // Test 15: Autograd for add and mul
    runner.add_test("Autograd add and mul chain", [] {
        Tensor<float> x({1});
        Tensor<float> y({1});
        x.data = {3.0f};
        y.data = {2.0f};
        x.set_requires_grad(true);
        y.set_requires_grad(true);

        Tensor<float> z = x * y + y;
        z.backward();

        assert(std::fabs(x.grad[0] - 2.0f) < 1e-5f);
        assert(std::fabs(y.grad[0] - 4.0f) < 1e-5f);
    });

    // Test 16: Autograd ReLU gate
    runner.add_test("Autograd relu", [] {
        Tensor<float> x({3});
        x.data = {-1.0f, 0.5f, 2.0f};
        x.set_requires_grad(true);

        Tensor<float> y = x.relu();
        Tensor<float> s({1});
        s.data = {y.data[0] + y.data[1] + y.data[2]};
        s.set_requires_grad(true);

        // Manual scalar wrapper to drive backward over relu output gradient = 1.
        y.grad = {1.0f, 1.0f, 1.0f};
        if (y.grad_fn && y.grad_fn->backward_fn) {
            y.grad_fn->backward_fn(y.grad);
        }

        assert(std::fabs(x.grad[0] - 0.0f) < 1e-5f);
        assert(std::fabs(x.grad[1] - 1.0f) < 1e-5f);
        assert(std::fabs(x.grad[2] - 1.0f) < 1e-5f);
    });

    // Test 17: Finite difference check for multiply gradient
    runner.add_test("Finite difference mul grad", [] {
        float x = 1.75f;
        float y = -0.8f;

        Tensor<float> tx({1});
        Tensor<float> ty({1});
        tx.data = {x};
        ty.data = {y};
        tx.set_requires_grad(true);
        ty.set_requires_grad(true);

        Tensor<float> out = tx * ty;
        out.backward();

        auto fn = [y](const std::vector<float>& v) { return v[0] * y; };
        float num_dx = finite_difference_gradient(fn, {x}, 0);
        assert(std::fabs(tx.grad[0] - num_dx) < 1e-3f);
    });

    // Test 18: Finite difference check for matmul gradient (w.r.t a00)
    runner.add_test("Finite difference matmul grad", [] {
        Tensor<float> a({2, 2});
        Tensor<float> b({2, 2});
        a.data = {1.0f, 2.0f, 3.0f, 4.0f};
        b.data = {0.5f, -1.0f, 1.5f, 2.0f};
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        Tensor<float> c = matmul(a, b);
        Tensor<float> loss({1});
        loss.data = {c.data[0] + c.data[1] + c.data[2] + c.data[3]};
        loss.set_requires_grad(true);

        c.grad = {1.0f, 1.0f, 1.0f, 1.0f};
        if (c.grad_fn && c.grad_fn->backward_fn) {
            c.grad_fn->backward_fn(c.grad);
        }

        auto fn = [](const std::vector<float>& v) {
            Tensor<float> aa({2, 2});
            Tensor<float> bb({2, 2});
            aa.data = {v[0], 2.0f, 3.0f, 4.0f};
            bb.data = {0.5f, -1.0f, 1.5f, 2.0f};
            Tensor<float> cc = matmul(aa, bb);
            return cc.sum().data[0];
        };
        float num_da00 = finite_difference_gradient(fn, {1.0f}, 0);
        assert(std::fabs(a.grad[0] - num_da00) < 2e-3f);
    });

    // Test 19: Native sum backward
    runner.add_test("Autograd sum backward", [] {
        Tensor<float> x({3});
        x.data = {2.0f, -1.0f, 4.0f};
        x.set_requires_grad(true);

        Tensor<float> loss = x.sum();
        loss.backward();

        assert(std::fabs(x.grad[0] - 1.0f) < 1e-6f);
        assert(std::fabs(x.grad[1] - 1.0f) < 1e-6f);
        assert(std::fabs(x.grad[2] - 1.0f) < 1e-6f);
    });

    // Test 20: Broadcast add backward reduction
    runner.add_test("Broadcast add backward", [] {
        Tensor<float> a({2, 1});
        Tensor<float> b({1, 3});
        a.data = {1.0f, 2.0f};
        b.data = {3.0f, 4.0f, 5.0f};
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        Tensor<float> out = a + b;
        Tensor<float> loss = out.sum();
        loss.backward();

        // Each a element is used across 3 columns.
        assert(std::fabs(a.grad[0] - 3.0f) < 1e-6f);
        assert(std::fabs(a.grad[1] - 3.0f) < 1e-6f);
        // Each b element is used across 2 rows.
        assert(std::fabs(b.grad[0] - 2.0f) < 1e-6f);
        assert(std::fabs(b.grad[1] - 2.0f) < 1e-6f);
        assert(std::fabs(b.grad[2] - 2.0f) < 1e-6f);
    });

    // Test 21: Subtraction/division backward
    runner.add_test("Autograd subtraction and division", [] {
        Tensor<float> x({1});
        Tensor<float> y({1});
        x.data = {6.0f};
        y.data = {2.0f};
        x.set_requires_grad(true);
        y.set_requires_grad(true);

        Tensor<float> z = x / y - y;
        z.backward();

        assert(std::fabs(x.grad[0] - 0.5f) < 1e-6f);
        assert(std::fabs(y.grad[0] - (-2.5f)) < 1e-6f);
    });

    // Test 22: Batched matmul backward
    runner.add_test("Batched matmul backward", [] {
        Tensor<float> a({2, 2, 3});
        Tensor<float> b({2, 3, 2});
        a.data = {
            1, 2, 3,
            4, 5, 6,
            1, 0, 2,
            0, 1, 2
        };
        b.data = {
            1, 2,
            3, 4,
            5, 6,
            1, 1,
            2, 2,
            3, 3
        };
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        Tensor<float> out = matmul(a, b);
        Tensor<float> loss = out.sum();
        loss.backward();

        // Compare one gradient element against finite-difference.
        auto fn = [&](const std::vector<float>& v) {
            Tensor<float> aa({2, 2, 3});
            Tensor<float> bb({2, 3, 2});
            aa.data = a.data;
            bb.data = b.data;
            aa.data[0] = v[0];
            return matmul(aa, bb).sum().data[0];
        };
        float num_da0 = finite_difference_gradient(fn, {a.data[0]}, 0, 1e-3f);
        assert(std::fabs(a.grad[0] - num_da0) < 2e-2f);
    });

    // Test 23: Tiny SGD linear training decreases loss
    runner.add_test("SGD linear training", [] {
        std::vector<float> xs = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        std::vector<float> ys;
        ys.reserve(xs.size());
        for (float x : xs) {
            ys.push_back(2.0f * x + 1.0f);
        }

        Tensor<float> w({1});
        Tensor<float> b({1});
        w.data[0] = -0.5f;
        b.data[0] = 0.0f;

        float start_loss = linear_dataset_mse(w, b, xs, ys);
        const float lr = 0.05f;

        for (int epoch = 0; epoch < 250; ++epoch) {
            float grad_w = 0.0f;
            float grad_b = 0.0f;
            for (size_t i = 0; i < xs.size(); ++i) {
                float pred = xs[i] * w.data[0] + b.data[0];
                float diff = pred - ys[i];
                grad_w += 2.0f * diff * xs[i];
                grad_b += 2.0f * diff;
            }
            grad_w /= static_cast<float>(xs.size());
            grad_b /= static_cast<float>(xs.size());
            w.data[0] -= lr * grad_w;
            b.data[0] -= lr * grad_b;
        }

        float end_loss = linear_dataset_mse(w, b, xs, ys);
        assert(end_loss < start_loss);
        assert(end_loss < 1e-2f);
    });

    // Test 24: 2D linear layer training with SGD optimizer
    runner.add_test("2D linear layer SGD optimizer", [] {
        Tensor<float> x({4, 2});
        x.data = {
            1.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 1.0f,
            2.0f, -1.0f
        };

        Tensor<float> y({4, 1});
        // Ground truth: y = x @ [2, -3]^T + 0.5
        y.data = {
            2.5f,
            -2.5f,
            -0.5f,
            7.5f
        };

        Tensor<float> w({2, 1});
        Tensor<float> b({1});
        w.data = {-0.2f, 0.3f};
        b.data = {0.0f};
        w.set_requires_grad(true);
        b.set_requires_grad(true);

        SGD<float> opt({&w, &b}, 0.05f, 0.0f, 5.0f);

        float start_loss = 0.0f;
        float end_loss = 0.0f;
        for (int epoch = 0; epoch < 160; ++epoch) {
            Tensor<float> pred = matmul(x, w) + b;

            std::vector<float> diff(4, 0.0f);
            float loss = 0.0f;
            for (size_t r = 0; r < 4; ++r) {
                diff[r] = pred.data[r] - y.data[r];
                loss += diff[r] * diff[r];
            }
            if (epoch == 0) start_loss = loss;
            end_loss = loss;

            float gw0 = 0.0f;
            float gw1 = 0.0f;
            float gb0 = 0.0f;
            for (size_t r = 0; r < 4; ++r) {
                float d = 2.0f * diff[r];
                gw0 += d * x.data[r * 2 + 0];
                gw1 += d * x.data[r * 2 + 1];
                gb0 += d;
            }

            w.grad = {gw0, gw1};
            b.grad = {gb0};
            opt.step();
        }

        assert(end_loss < start_loss);
        assert(end_loss < 1e-3f);
        assert(std::fabs(w.data[0] - 2.0f) < 0.1f);
        assert(std::fabs(w.data[1] + 3.0f) < 0.1f);
        assert(std::fabs(b.data[0] - 0.5f) < 0.1f);
    });
    
    runner.run_all();
    return runner.failed > 0 ? 1 : 0;
}
