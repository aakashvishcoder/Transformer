#pragma once
#include <vector>
#include <functional>
#include <numeric>
#include <cassert>
#include <iostream>
#include <unordered_set>
#include <cmath>
#include <string> 
#include <algorithm>

template<typename T>
class Tensor {
public:
    // --- Core data ---
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::vector<T> data;
    bool requires_grad = false;
    std::vector<T> grad;

    // --- Autograd graph ---
    std::vector<Tensor*> parents;
    std::function<void(const std::vector<T>&)> backward_fn;

    // --- Constructors ---
    Tensor() = default;

    explicit Tensor(const std::vector<size_t>& shape, bool requires_grad = false)
    : shape(shape), data(compute_numel(shape)), requires_grad(requires_grad),
      grad(requires_grad ? data.size() : 0, T(0))
    {
        std::cout << "Creating tensor, shape = {";
        for (auto s : shape) std::cout << s << " ";
        std::cout << "}, numel = " << data.size() << std::endl;
        compute_strides();
    }


    // Copy is shallow (data shared, grad copied if needed)
    Tensor(const Tensor& other) = default;
    Tensor& operator=(const Tensor& other) = default;

    // --- Utilities ---
    size_t ndim() const { return shape.size(); }
    size_t numel() const { return data.size(); }

    static size_t compute_numel(const std::vector<size_t>& shape) {
        return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<>());
    }

    void compute_strides() {
        if (shape.empty()) {
            strides.clear();
            return;
        }

        strides.assign(shape.size(), 1);
        for (size_t i = shape.size() - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * shape[i];
        }
    }


    // --- Indexing ---
    T& operator()(const std::vector<size_t>& indices) {
        assert(indices.size() == shape.size());
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            assert(indices[i] < shape[i]);
            offset += indices[i] * strides[i];
        }
        return data[offset];
    }

    const T& operator()(const std::vector<size_t>& indices) const {
        assert(indices.size() == shape.size());
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            assert(indices[i] < shape[i]);
            offset += indices[i] * strides[i];
        }
        return data[offset];
    }

    T& operator()(std::initializer_list<size_t> indices) {
        return (*this)(std::vector<size_t>(indices));
    }
    const T& operator()(std::initializer_list<size_t> indices) const {
        return (*this)(std::vector<size_t>(indices));
    }

    // --- Gradient ---
    void zero_grad() {
        if (requires_grad) std::fill(grad.begin(), grad.end(), T(0));
    }

    // --- Autograd ---
    void backward() {
        // Seed gradient for scalar tensors
        if (shape.empty()) {
            if (grad.empty()) grad = {T(1)};
            else if (grad[0] == T(0)) grad[0] = T(1);
        }
        std::unordered_set<void*> visited;
        backward_recursive(visited);
    }

private:
    void backward_recursive(std::unordered_set<void*>& visited) {
        if (visited.count(this)) return;
        visited.insert(this);

        // Call backward function with current grad as upstream
        if (backward_fn) {
            backward_fn(grad);
        }

        // Propagate to parents (they accumulate in their own grad)
        for (Tensor* parent : parents) {
            parent->backward_recursive(visited);
        }
    }

public:
    // --- Element-wise addition (same shape only for simplicity) ---
    Tensor operator+(const Tensor& other) {
        assert(shape == other.shape && "Broadcasting not fully supported in backward yet");
        Tensor out(shape, requires_grad || other.requires_grad);
        for (size_t i = 0; i < data.size(); ++i) {
            out.data[i] = data[i] + other.data[i];
        }
        if (out.requires_grad) {
            Tensor* self = this;
            Tensor* rhs = const_cast<Tensor*>(&other);
            out.parents = {self, rhs};
            out.backward_fn = [self, rhs](const std::vector<T>& upstream_grad) {
                if (self->requires_grad) {
                    self->accumulate_grad(upstream_grad);
                }
                if (rhs->requires_grad) {
                    rhs->accumulate_grad(upstream_grad);
                }
            };
        }
        return out;
    }

    // --- Broadcasting helper (forward only) ---
    static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
        size_t ndim = std::max(a.size(), b.size());
        std::vector<size_t> a_pad(ndim - a.size(), 1);
        a_pad.insert(a_pad.end(), a.begin(), a.end());
        std::vector<size_t> b_pad(ndim - b.size(), 1);
        b_pad.insert(b_pad.end(), b.begin(), b.end());

        std::vector<size_t> out(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            if (a_pad[i] == b_pad[i]) out[i] = a_pad[i];
            else if (a_pad[i] == 1) out[i] = b_pad[i];
            else if (b_pad[i] == 1) out[i] = a_pad[i];
            else throw std::invalid_argument("Broadcast failed");
        }
        return out;
    }

    Tensor broadcast_to(const std::vector<size_t>& target_shape) const {
    if (shape == target_shape) return *this;

    // Broadcasting rule: align trailing dimensions
    if (target_shape.size() < shape.size()) {
        throw std::invalid_argument("Cannot broadcast to smaller rank");
    }

    size_t offset = target_shape.size() - shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
        size_t target_dim = i + offset;
        if (shape[i] != target_shape[target_dim] && shape[i] != 1) {
            throw std::invalid_argument("Broadcast incompatible at dimension " + std::to_string(i));
        }
    }

    Tensor out(target_shape, requires_grad);
        std::vector<size_t> out_idx(target_shape.size());

        auto recurse = [&](auto&& self, size_t dim) -> void {
            if (dim == target_shape.size()) {
                // Build source index by aligning to trailing dims
                std::vector<size_t> src_idx(shape.size());
                for (size_t i = 0; i < shape.size(); ++i) {
                    size_t out_dim = i + offset;
                    // If original dim was 1, always use index 0; else use out_idx
                    src_idx[i] = (shape[i] == 1) ? 0 : out_idx[out_dim];
                }
                out(out_idx) = (*this)(src_idx);
                return;
            }
            for (out_idx[dim] = 0; out_idx[dim] < target_shape[dim]; ++out_idx[dim]) {
                self(self, dim + 1);
            }
        };
        recurse(recurse, 0);
        return out;
    }

    // --- Nonlinearity: exp ---
    Tensor exp() {
        std::cout << "exp" << std::endl;
        Tensor out(shape, requires_grad);
        for (size_t i = 0; i < data.size(); ++i) {
            out.data[i] = std::exp(data[i]);
        }
        if (requires_grad) {
            Tensor* self = this;
            out.parents = {self};
            out.backward_fn = [self, out_data = out.data](const std::vector<T>& upstream_grad) {
                std::vector<T> self_grad(upstream_grad.size());
                for (size_t i = 0; i < upstream_grad.size(); ++i) {
                    self_grad[i] = upstream_grad[i] * out_data[i]; // d(exp(x))/dx = exp(x)
                }
                self->accumulate_grad(self_grad);
            };
        }
        return out;
    }

    // --- Reduction: sum all elements ---
    Tensor sum() {
        T s = std::accumulate(data.begin(), data.end(), T(0));
        Tensor out({}, requires_grad); // scalar
        out.data = {s};
        std::cout << "sum" << std::endl;
        if (requires_grad) {
            Tensor* self = this;
            out.parents = {self};
            out.backward_fn = [self](const std::vector<T>& upstream_grad) {
                // upstream_grad is size 1 (scalar)
                T scalar_grad = upstream_grad[0];
                std::vector<T> self_grad(self->data.size(), scalar_grad);
                self->accumulate_grad(self_grad);
            };
        }
        return out;
    }

    // --- Accumulate gradient safely ---
    void accumulate_grad(const std::vector<T>& upstream_grad) {
        if (!requires_grad) return;
        if (grad.empty()) {
            grad.resize(data.size(), T(0));
        }
        assert(upstream_grad.size() == grad.size());
        for (size_t i = 0; i < grad.size(); ++i) {
            grad[i] += upstream_grad[i];
        }
    }

    // --- Print ---
    void print() const {
        if (shape.empty()) {
            std::cout << data[0] << std::endl;
            return;
        }
        print_recursive({}, 0);
        std::cout << std::endl;
    }

    void print_recursive(std::vector<size_t> idx, size_t dim) const {
        if (dim == shape.size()) {
            std::cout << (*this)(idx);
            return;
        }
        std::cout << "[";
        for (size_t i = 0; i < shape[dim]; ++i) {
            idx.push_back(i);
            print_recursive(idx, dim + 1);
            idx.pop_back();
            if (i != shape[dim] - 1) std::cout << ", ";
        }
        std::cout << "]";
    }
};