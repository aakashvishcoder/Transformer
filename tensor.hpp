#pragma once
#include <random>
#include <limits>
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
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::vector<T> data;
    bool requires_grad = false;
    std::vector<T> grad;

    std::vector<Tensor*> parents;
    std::function<void(const std::vector<T>&)> backward_fn;

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

    Tensor(const Tensor& other) = default;
    Tensor& operator=(const Tensor& other) = default;

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

    void zero_grad() {
        if (requires_grad) std::fill(grad.begin(), grad.end(), T(0));
    }

    void backward() {
        if (shape.empty()) {
            if (grad.empty()) grad = {T(1)};
            else if (grad[0] == T(0)) grad[0] = T(1);
        }
        std::unordered_set<void*> visited;
        backward_recursive(visited);
    }

    void zeros() {
        std::fill(data.begin(), data.end(), T(0));
    }

    void ones() {
        std::fill(data.begin(), data.end(), T(1));
    }

    void fill_random(T low = T(0), T high = T(1)) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(low, high);
        for (auto& val: data) {
            val = dis(gen);
        } 
    }

private:
    void backward_recursive(std::unordered_set<void*>& visited) {
        if (visited.count(this)) return;
        visited.insert(this);

        if (backward_fn) {
            backward_fn(grad);
        }

        for (Tensor* parent : parents) {
            parent->backward_recursive(visited);
        }
    }

public:
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

    Tensor operator-(const Tensor& other) {
        assert(shape == other.shape);
        Tensor out(shape, requires_grad || other.requires_grad);
        for (size_t i = 0; i < data.size(); ++i) {
            out.data[i] = data[i] - other.data[i];
        }
        if (out.requires_grad) {
            Tensor* self = this;
            Tensor* rhs = const_cast<Tensor*>(&other);
            out.parents = {self, rhs};
            out.backward_fn = [self, rhs](const std::vector<T>& upstream_grad) {
                if (self->requires_grad) self->accumulate_grad(upstream_grad);
                if (rhs->requires_grad) {
                    std::vector<T> neg_grad(upstream_grad.size());
                    for (size_t i = 0; i < upstream_grad.size(); ++i) {
                        neg_grad[i] = -upstream_grad[i];
                    }
                    rhs->accumulate_grad(neg_grad);
                }
            };
        }
        return out;
    }

    Tensor operator/(const Tensor& other) {
        assert(shape == other.shape);
        Tensor out(shape, requires_grad || other.requires_grad);
        for (size_t i = 0; i < data.size(); ++i) {
            out.data[i] = data[i] / other.data[i];
        }
        if (out.requires_grad) {
            Tensor* self = this;
            Tensor* rhs = const_cast<Tensor*>(&other);
            out.parents = {self, rhs};
            out.backward_fn = [self, rhs, out_data = out.data](const std::vector<T>& upstream_grad) {
                if (self->requires_grad) {
                    std::vector<T> grad_self(upstream_grad.size());
                    for (size_t i = 0; i < upstream_grad.size(); ++i) {
                        grad_self[i] = upstream_grad[i] / rhs->data[i];
                    }
                    self->accumulate_grad(grad_self);
                }
                if (rhs->requires_grad) {
                    std::vector<T> grad_rhs(upstream_grad.size());
                    for (size_t i = 0; i < upstream_grad.size(); ++i) {
                        grad_rhs[i] = -upstream_grad[i] * out_data[i] / rhs->data[i];
                    }
                    rhs->accumulate_grad(grad_rhs);
                }
            };
        }
        return out;
    }

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
                std::vector<size_t> src_idx(shape.size());
                for (size_t i = 0; i < shape.size(); ++i) {
                    size_t out_dim = i + offset;
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
                    self_grad[i] = upstream_grad[i] * out_data[i]; 
                }
                self->accumulate_grad(self_grad);
            };
        }
        return out;
    }

    Tensor sum() {
        T s = std::accumulate(data.begin(), data.end(), T(0));
        Tensor out({}, requires_grad); 
        out.data = {s};
        std::cout << "sum" << std::endl;
        if (requires_grad) {
            Tensor* self = this;
            out.parents = {self};
            out.backward_fn = [self](const std::vector<T>& upstream_grad) {
                T scalar_grad = upstream_grad[0];
                std::vector<T> self_grad(self->data.size(), scalar_grad);
                self->accumulate_grad(self_grad);
            };
        }
        return out;
    }

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

    Tensor pow(T exponent) {
        Tensor out(shape, requires_grad);
        for (size_t i = 0; i < data.size(); ++i) {
            out.data[i] = std::pow(data[i], exponent);
        }
        if (requires_grad) {
            Tensor* self = this;
            out.parents = {self};
            out.backward_fn = [self, exponent, out_data = out.data](const std::vector<T>& upstream_grad) {
                std::vector<T> self_grad(upstream_grad.size());
                for (size_t i = 0; i < upstream_grad.size(); ++i) {
                    self_grad[i] = upstream_grad[i] * exponent * std::pow(self->data[i], exponent - 1);
                }
                self->accumulate_grad(self_grad);
            };
        }
        return out;
    }

    Tensor sqrt() {
        Tensor out(shape, requires_grad);
        for (size_t i = 0; i < data.size(); ++i) {
            out.data[i] = std::sqrt(data[i]);
        }
        if (requires_grad) {
            Tensor* self = this;
            out.parents = {self};
            out.backward_fn = [self, out_data = out.data](const std::vector<T>& upstream_grad) {
                std::vector<T> self_grad(upstream_grad.size());
                for (size_t i = 0; i < upstream_grad.size(); ++i) {
                    self_grad[i] = upstream_grad[i] / (2 * out_data[i]);
                }
                self->accumulate_grad(self_grad);
            };
        }
        return out;
    }

    Tensor relu() {
        Tensor out(shape, requires_grad);
        for (size_t i = 0; i < data.size(); ++i) {
            out.data[i] = std::max(T(0), data[i]);
        }
        if (requires_grad) {
            Tensor* self = this;
            out.parents = {self};
            out.backward_fn = [self](const std::vector<T>& upstream_grad) {
                std::vector<T> self_grad(upstream_grad.size());
                for (size_t i = 0; i < upstream_grad.size(); ++i) {
                    self_grad[i] = upstream_grad[i] * (self->data[i] > 0 ? T(1) : T(0));
                }
                self->accumulate_grad(self_grad);
            };
        }
        return out;
    }

    Tensor mean_axis(size_t dim, bool keepdim = false) {
        if (dim >= shape.size()) {
            throw std::invalid_argument("Invalid dimension!");
        }
        size_t reduce_size = shape[dim];
        std::vector<size_t> new_shape = shape;
        new_shape.erase(new_shape.begin() + dim);
        if (keepdim) {
            new_shape.insert(new_shape.begin() + dim, 1);
        }

        Tensor out(new_shape, requires_grad);
        out.zeros();

        std::vector<size_t> idx(shape.size());
        auto recurse = [&](auto&& self, size_t d) -> void {
            if (d == shape.size()) {
                std::vector<size_t> out_idx = idx;
                if (!keepdim) {
                    out_idx.erase(out_idx.begin() + dim);
                } else {
                    out_idx[dim] = 0;
                }
                out(out_idx) += (*this)(idx);
                return;
            }
            for (idx[d] = 0; idx[d] < shape[d]; ++idx[d]) {
                self(self, d + 1);
            }
        };
        recurse(recurse, 0);

        T inv_size = T(1) / T(reduce_size);
        for (auto& x : out.data) x *= inv_size;

        if (requires_grad) {
            Tensor* self = this;
            out.parents = {self};
            out.backward_fn = [self, dim, reduce_size, keepdim, out_shape = out.shape, &out](const std::vector<T>& upstream_grad) {
                std::vector<T> self_grad(self->data.size(), T(0));
                std::vector<size_t> idx(self->shape.size());
                auto recurse_grad = [&](auto&& self_g, size_t d) -> void {
                    if (d == self->shape.size()) {
                        std::vector<size_t> out_idx = idx;
                        if (!keepdim) {
                            out_idx.erase(out_idx.begin() + dim);
                        } else {
                            out_idx[dim] = 0;
                        }
                        auto out_strides = compute_strides_from_shape(out_shape);
                        size_t out_offset = multi_index_to_offset(out_idx, out_strides);
                        self_grad[multi_index_to_offset(idx, self->strides)] = upstream_grad[out_offset] / T(reduce_size);
                        return;
                    }
                    for (idx[d] = 0; idx[d] < self->shape[d]; ++idx[d]) {
                        self_g(self_g, d + 1);
                    }
                };
                recurse_grad(recurse_grad, 0);
                self->accumulate_grad(self_grad);
            };
        }
        return out;
    }

private:
    static size_t multi_index_to_offset(const std::vector<size_t>& idx, const std::vector<size_t>& strides) {
        size_t offset = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            offset += idx[i] * strides[i];
        }
        return offset;
    }

    static std::vector<size_t> compute_strides_from_shape(const std::vector<size_t>& shape) {
        if (shape.empty()) return {};
        std::vector<size_t> strides(shape.size(), 1);
        for (size_t i = shape.size() - 1; i > 0; --i) {
            strides[i-1] = strides[i] * shape[i];
        }
        return strides;
    }
public:
    Tensor transpose(const std::vector<size_t>& axes) {
        if (axes.size() != shape.size()) {
            throw std::invalid_argument("Axes must match tensor rank");
        }
        std::vector<bool> seen(shape.size(), false);
        for (size_t a : axes) {
            if (a >= shape.size() || seen[a]) {
                throw std::invalid_argument("Invalid or duplicate axes");
            }
            seen[a] = true;
        }

        std::vector<size_t> new_shape(shape.size());
        for (size_t i = 0; i < axes.size(); ++i) {
            new_shape[i] = shape[axes[i]];
        }

        Tensor out(new_shape, requires_grad);
        std::vector<size_t> idx(new_shape.size());
        auto recurse = [&](auto&& self, size_t d) -> void {
            if (d == new_shape.size()) {
                std::vector<size_t> src_idx(shape.size());
                for (size_t i = 0; i < axes.size(); ++i) {
                    src_idx[axes[i]] = idx[i];
                }
                out(idx) = (*this)(src_idx);
                return;
            }
            for (idx[d] = 0; idx[d] < new_shape[d]; ++idx[d]) {
                self(self, d + 1);
            }
        };
        recurse(recurse, 0);

        if (requires_grad) {
            Tensor* self = this;
            out.parents = {self};
            out.backward_fn = [self, axes](const std::vector<T>& upstream_grad) {
                std::vector<T> self_grad(self->data.size(), T(0));
                std::vector<size_t> idx(self->shape.size());
                auto recurse_grad = [&](auto&& self_g, size_t d) -> void {
                    if (d == self->shape.size()) {
                        std::vector<size_t> out_idx(axes.size());
                        for (size_t i = 0; i < axes.size(); ++i) {
                            out_idx[i] = idx[axes[i]];
                        }
                        auto out_strides = compute_strides_from_shape(self->shape);
                        size_t out_offset = multi_index_to_offset(out_idx, out_strides);
                        self_grad[multi_index_to_offset(idx, self->strides)] = upstream_grad[out_offset];
                        return;
                    }
                    for (idx[d] = 0; idx[d] < self->shape[d]; ++idx[d]) {
                        self_g(self_g, d + 1);
                    }
                };
                recurse_grad(recurse_grad, 0);
                self->accumulate_grad(self_grad);
            };
        }
        return out;
    }

    Tensor transpose_last_two() {
        if (shape.size() < 2) {
            throw std::invalid_argument("Need at least 2 dimensions");
        }
        std::vector<size_t> axes(shape.size());
        std::iota(axes.begin(), axes.end(), 0);
        std::swap(axes[axes.size() - 1], axes[axes.size() - 2]);
        return transpose(axes);
    }

    Tensor view(const std::vector<size_t>& new_shape) {
        size_t new_numel = compute_numel(new_shape);
        if (new_numel != numel()) {
            throw std::invalid_argument("View size mismatch");
        }
        Tensor out(new_shape, requires_grad);
        out.data = data;
        if (requires_grad) {
            out.grad = grad;
            out.parents = {this};
            out.backward_fn = [self = this](const std::vector<T>& upstream_grad) {
                self->accumulate_grad(upstream_grad);
            };
        }
        return out;
    }

    Tensor sum_axis(size_t dim, bool keepdim = false) {
        if (dim >= shape.size()) {
            throw std::invalid_argument("Invalid dimensions");
        }
        std::vector<size_t> new_shape = shape;
        new_shape.erase(new_shape.begin() + dim);

        Tensor out(new_shape, requires_grad);
        out.zeros();

        std::vector<size_t> idx(shape.size());
        auto recurse = [&](auto&& self, size_t d) -> void {
            if (d == shape.size()) {
                std::vector<size_t> out_idx = idx;
                if (!keepdim) {
                    out_idx.erase(out_idx.begin() + dim);
                } else {
                    out_idx[dim] = 0;
                }
                out(out_idx) += (*this)(idx);
                return;
            }
            for (idx[d] = 0; idx[d] < shape[d]; ++idx[d]) {
                self(self, d + 1);
            }
        };
        recurse(recurse, 0);

        if (requires_grad) {
            Tensor* self = this;
            out.parents = {self};
            out.backward_fn = [self, dim, keepdim, out_shape = out.shape](const std::vector<T>& upstream_grad) {
                std::vector<T> self_grad(self->data.size(), T(0));
                std::vector<size_t> idx(self->shape.size());
                auto recurse_grad = [&](auto&& self_g, size_t d) -> void {
                    if (d == self->shape.size()) {
                        std::vector<size_t> out_idx = idx;
                        if (!keepdim) {
                            out_idx.erase(out_idx.begin() + dim);
                        } else {
                            out_idx[dim] = 0;
                        }
                        size_t out_offset = 0;
                        auto out_strides = compute_strides_from_shape(out_shape);
                        for (size_t i = 0; i < out_shape.size(); ++i) {
                            out_offset += out_idx[i] * out_strides[i];
                        }
                        self_grad[multi_index_to_offset(idx, self->strides)] = upstream_grad[out_offset];
                        return;
                    }
                    for (idx[d] = 0; idx[d] < self->shape[d]; ++idx[d]) {
                        self_g(self_g, d + 1);
                    }
                };
                recurse_grad(recurse_grad, 0);
                self->accumulate_grad(self_grad);
            };
        }
        return out;
    }

    Tensor max_axis(size_t dim, bool keepdim = false) {
        if (dim >= shape.size()) {
            throw std::invalid_argument("Invalid dimension");
        }
        std::vector<size_t> new_shape = shape;
        new_shape.erase(new_shape.begin() + dim);
        if (keepdim) {
            new_shape.insert(new_shape.begin() + dim, 1);
        }

        Tensor out(new_shape, requires_grad);
        for (auto& x : out.data) x = -std::numeric_limits<T>::infinity();

        std::vector<size_t> idx(shape.size());
        auto recurse = [&](auto&& self, size_t d) -> void {
            if (d == shape.size()) {
                std::vector<size_t> out_idx = idx;
                if (!keepdim) {
                    out_idx.erase(out_idx.begin() + dim);
                } else {
                    out_idx[dim] = 0;
                }
                T& current_max = out(out_idx);
                T val = (*this)(idx);
                if (val > current_max) current_max = val;
                return;
            }
            for (idx[d] = 0; idx[d] < shape[d]; ++idx[d]) {
                self(self, d + 1);
            }
        };
        recurse(recurse, 0);
        
        if (requires_grad) {
            Tensor* self = this;
            out.parents = {self};
            out.backward_fn = [self, dim, keepdim, out_shape = out.shape, &out](const std::vector<T>& upstream_grad) {
                std::vector<T> self_grad(self->data.size(), T(0));
                std::vector<size_t> idx(self->shape.size());
                auto recurse_grad = [&](auto&& self_g, size_t d) -> void {
                    if (d == self->shape.size()) {
                        std::vector<size_t> out_idx = idx;
                        if (!keepdim) {
                            out_idx.erase(out_idx.begin() + dim);
                        } else {
                            out_idx[dim] = 0;
                        }
                        auto out_strides = compute_strides_from_shape(out_shape);
                        size_t out_offset = multi_index_to_offset(out_idx, out_strides);

                        T max_val = out(out_idx);
                        if (std::abs(self->operator()(idx) - max_val) < 1e-6) {
                            self_grad[multi_index_to_offset(idx, self->strides)] = upstream_grad[out_offset];
                        }
                        return;
                    }
                    for (idx[d] = 0; idx[d] < self->shape[d]; ++idx[d]) {
                        self_g(self_g, d + 1);
                    }
                };
                recurse_grad(recurse_grad, 0);
                self->accumulate_grad(self_grad);
            };
        }
        return out;
    }

    Tensor softmax(size_t dim = -1) {
        if (dim < 0) dim += shape.size();
        auto max_vals = max_axis(dim, true);
        auto exp_self = exp();
        auto sum_exp = exp_self.sum_axis(dim, true);
        return exp_self / sum_exp;
    }
};

template<typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.ndim() == 2 && b.ndim() == 2) {
        assert(a.shape[1] == b.shape[0]);
        size_t M = a.shape[0], K = a.shape[1], N = b.shape[1];
        Tensor<T> out({M, N}, a.requires_grad || b.requires_grad);
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                T sum = T(0);
                for (size_t k = 0; k < K; ++k) {
                    sum += a.data[i * K + k] * b.data[k * N + j];
                }
                out.data[i * N + j] = sum;
            }
        }
        if (out.requires_grad) {
            out.parents = {const_cast<Tensor<T>*>(&a), const_cast<Tensor<T>*>(&b)};
            out.backward_fn = [a_ptr = const_cast<Tensor<T>*>(&a), b_ptr = const_cast<Tensor<T>*>(&b), M, K, N](const std::vector<T>& upstream) {
                if (a_ptr->requires_grad) {
                    std::vector<T> grad_a(a_ptr->data.size(), T(0));
                    for (size_t i = 0; i < M; ++i)
                        for (size_t k = 0; k < K; ++k)
                            for (size_t j = 0; j < N; ++j)
                                grad_a[i * K + k] += upstream[i * N + j] * b_ptr->data[k * N + j];
                    a_ptr->accumulate_grad(grad_a);
                }
                if (b_ptr->requires_grad) {
                    std::vector<T> grad_b(b_ptr->data.size(), T(0));
                    for (size_t k = 0; k < K; ++k)
                        for (size_t j = 0; j < N; ++j) 
                            for (size_t i = 0; i < M; ++i)
                                grad_b[k * N + j] += a_ptr->data[i * K + k] * upstream[i * N + j];
                    b_ptr->accumulate_grad(grad_b);
                }
            };
        }
        return out;
    } else if (a.ndim() == 3 && b.ndim() == 3) {
        assert(a.shape[0] == b.shape[0] && a.shape[2] == b.shape[1]);
        size_t B = a.shape[0], M = a.shape[1], K = a.shape[2], N = b.shape[2];
        Tensor<T> out({B, M, N}, a.requires_grad || b.requires_grad);
        for (size_t b_idx = 0; b_idx < B; ++b_idx) {
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    T sum = T(0);
                    for (size_t k = 0; k < K; ++k) {
                        sum += a.data[(b_idx * M + i) * K + k] * b.data[(b_idx * K + k) * N + j];
                    }
                    out.data[(b_idx * M + i) * N + j] = sum;
                }
            }
        }
        if (out.requires_grad) {
            out.parents = {const_cast<Tensor<T>*>(&a), const_cast<Tensor<T>*>(&b)};
            out.backward_fn = [a_ptr = const_cast<Tensor<T>*>(&a), b_ptr = const_cast<Tensor<T>*>(&b), B, M, K, N](const std::vector<T>& upstream) {
                if (a_ptr->requires_grad) {
                    std::vector<T> grad_a(a_ptr->data.size(), T(0));
                    for (size_t b_idx = 0; b_idx < B; ++b_idx) 
                        for (size_t i = 0; i < M; ++i)
                            for (size_t k = 0; k < K; ++k)
                                for (size_t j = 0; j < N; ++j)
                                    grad_a[(b_idx * M + i) * K + k] += upstream[(b_idx * M + i) * N + j] * b_ptr->data[(b_idx * K + k) * N + j];
                    a_ptr->accumulate_grad(grad_a);
                }
                if (b_ptr->requires_grad) {
                    std::vector<T> grad_b(b_ptr->data.size(), T(0));
                    for (size_t b_idx = 0; b_idx < B; ++b_idx)
                        for (size_t k = 0; k < K; ++k)
                            for (size_t j = 0; j < N; ++j)
                                for (size_t i = 0; i < M; ++i)
                                    grad_b[(b_idx * K + k) * N + j] += a_ptr->data[(b_idx * M + i) * K + k] * upstream[(b_idx * M + i) * N + j];
                    b_ptr->accumulate_grad(grad_b);
                }
            };
        }
        return out;
    } else {
        throw std::invalid_argument("matmul only supports 2D or 3D tensors");
    }
};

template<typename T>
Tensor<T> operator*(T scalar, const Tensor<T>& t) {
    return t * scalar;
}

template<typename T>
class Linear {
public: 
    Tensor<T> weight;
    Tensor<T> bias;

    Linear(size_t in_features, size_t out_features, bool bias = true) 
        : weight({out_features, in_features}, true), bias(bias ? Tensor<T>({out_features}, true) : Tensor<T>()) {
        weight.fill_random(-0.1, 0.1);
        if (bias) this->bias.zeros();
    }

    Tensor<T> forward(const Tensor<T>& x) {
        auto out = matmul(x, weight.transpose_last_two());
        if (bias.data.size() > 0) out = out + bias;
        return out;
    }
};

