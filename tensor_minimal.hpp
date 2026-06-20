#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <vector>

struct Float16 {
    uint16_t bits = 0;

    Float16() = default;
    Float16(float v) : bits(float_to_half_bits(v)) {}
    Float16(double v) : Float16(static_cast<float>(v)) {}
    Float16(int v) : Float16(static_cast<float>(v)) {}

    operator float() const { return half_bits_to_float(bits); }

private:
    static uint16_t float_to_half_bits(float value) {
        uint32_t f = 0;
        std::memcpy(&f, &value, sizeof(float));

        uint32_t sign = (f >> 16) & 0x8000u;
        int32_t exp = static_cast<int32_t>((f >> 23) & 0xffu) - 127 + 15;
        uint32_t mant = f & 0x7fffffu;

        if (exp <= 0) {
            if (exp < -10) return static_cast<uint16_t>(sign);
            mant = (mant | 0x800000u) >> static_cast<uint32_t>(1 - exp);
            return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
        }
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);

        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000u) >> 13));
    }

    static float half_bits_to_float(uint16_t h) {
        uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
        uint32_t exp = (h >> 10) & 0x1fu;
        uint32_t mant = h & 0x3ffu;
        uint32_t out = 0;

        if (exp == 0) {
            if (mant == 0) {
                out = sign;
            } else {
                exp = 127 - 15 + 1;
                while ((mant & 0x400u) == 0) {
                    mant <<= 1;
                    --exp;
                }
                mant &= 0x3ffu;
                out = sign | (exp << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            out = sign | 0x7f800000u | (mant << 13);
        } else {
            uint32_t exp32 = exp + (127 - 15);
            out = sign | (exp32 << 23) | (mant << 13);
        }

        float result = 0.0f;
        std::memcpy(&result, &out, sizeof(float));
        return result;
    }
};

struct BFloat16 {
    uint16_t bits = 0;

    BFloat16() = default;
    BFloat16(float v) : bits(float_to_bfloat16_bits(v)) {}
    BFloat16(double v) : BFloat16(static_cast<float>(v)) {}
    BFloat16(int v) : BFloat16(static_cast<float>(v)) {}

    operator float() const { return bfloat16_bits_to_float(bits); }

private:
    static uint16_t float_to_bfloat16_bits(float value) {
        uint32_t u = 0;
        std::memcpy(&u, &value, sizeof(float));
        uint32_t lsb = (u >> 16) & 1u;
        u += 0x7fffu + lsb;
        return static_cast<uint16_t>(u >> 16);
    }

    static float bfloat16_bits_to_float(uint16_t b) {
        uint32_t u = static_cast<uint32_t>(b) << 16;
        float out = 0.0f;
        std::memcpy(&out, &u, sizeof(float));
        return out;
    }
};

template<typename U>
struct is_low_precision_float : std::false_type {};

template<>
struct is_low_precision_float<Float16> : std::true_type {};

template<>
struct is_low_precision_float<BFloat16> : std::true_type {};

template<typename U>
struct tensor_compute_type {
    using type = std::conditional_t<is_low_precision_float<U>::value, float, U>;
};

template<typename U>
using tensor_compute_t = typename tensor_compute_type<U>::type;

template<typename U>
inline tensor_compute_t<U> to_compute(U value) {
    if constexpr (is_low_precision_float<U>::value) {
        return static_cast<float>(value);
    } else {
        return value;
    }
}

template<typename U, typename V>
inline U from_compute(V value) {
    if constexpr (is_low_precision_float<U>::value) {
        return U(static_cast<float>(value));
    } else {
        return static_cast<U>(value);
    }
}

template<typename T>
class Tensor;

template<typename T>
struct AutogradNode {
    std::vector<Tensor<T>*> parents;
    std::vector<std::shared_ptr<Tensor<T>>> owned_parents;
    std::function<void(const std::vector<float>&)> backward_fn;
};

template<typename T>
class TensorView {
public:
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    T* data_ptr = nullptr;

    TensorView() = default;
    TensorView(const std::vector<size_t>& shape_, const std::vector<size_t>& strides_, T* data_)
        : shape(shape_), strides(strides_), data_ptr(data_) {}

    size_t ndim() const { return shape.size(); }

    size_t numel() const {
        if (shape.empty()) return 0;
        size_t total = 1;
        for (size_t d : shape) total *= d;
        return total;
    }

    T& at(const std::vector<size_t>& indices) {
        return data_ptr[compute_offset(indices)];
    }

    const T& at(const std::vector<size_t>& indices) const {
        return data_ptr[compute_offset(indices)];
    }

private:
    size_t compute_offset(const std::vector<size_t>& indices) const {
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("indices rank must match tensor rank");
        }
        size_t offset = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("tensor view index out of bounds");
            }
            offset += indices[i] * strides[i];
        }
        return offset;
    }
};

template<typename T>
class Tensor {
public:
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::vector<T> data;

    bool requires_grad = false;
    std::vector<float> grad;
    std::shared_ptr<AutogradNode<T>> grad_fn;

    Tensor() = default;

    explicit Tensor(const std::vector<size_t>& shape_)
        : shape(shape_), strides(compute_strides(shape_)), data(numel_from_shape(shape_)) {}

    static size_t numel_from_shape(const std::vector<size_t>& shape_) {
        if (shape_.empty()) return 0;
        size_t total = 1;
        for (size_t dim : shape_) {
            if (dim == 0) return 0;
            if (total > std::numeric_limits<size_t>::max() / dim) {
                throw std::overflow_error("Tensor shape product overflow");
            }
            total *= dim;
        }
        return total;
    }

    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape_) {
        std::vector<size_t> out(shape_.size(), 1);
        size_t stride = 1;
        for (size_t i = shape_.size(); i > 0; --i) {
            out[i - 1] = stride;
            stride *= shape_[i - 1];
        }
        return out;
    }

    size_t numel() const { return data.size(); }
    size_t ndim() const { return shape.size(); }
    size_t bytes() const { return data.size() * sizeof(T); }

    bool is_contiguous() const {
        return strides == compute_strides(shape);
    }

    void reshape(const std::vector<size_t>& new_shape) {
        if (numel_from_shape(new_shape) != numel()) {
            throw std::invalid_argument("Reshape shape must have same number of elements");
        }
        shape = new_shape;
        strides = compute_strides(shape);
        if (requires_grad && grad.size() != numel()) {
            grad.assign(numel(), 0.0f);
        }
    }

    void set_requires_grad(bool flag) {
        requires_grad = flag;
        if (requires_grad) {
            if (grad.size() != numel()) grad.assign(numel(), 0.0f);
        } else {
            grad.clear();
            grad_fn.reset();
        }
    }

    void zero_grad() {
        if (!requires_grad) return;
        if (grad.size() != numel()) {
            grad.assign(numel(), 0.0f);
        } else {
            std::fill(grad.begin(), grad.end(), 0.0f);
        }
    }

    void add_grad(const std::vector<float>& incoming) {
        if (!requires_grad) {
            throw std::runtime_error("add_grad called on tensor with requires_grad=false");
        }
        if (incoming.size() != numel()) {
            throw std::invalid_argument("incoming gradient size mismatch");
        }
        if (grad.size() != numel()) {
            grad.assign(numel(), 0.0f);
        }
        for (size_t i = 0; i < grad.size(); ++i) {
            grad[i] += incoming[i];
        }
    }

    void backward() {
        if (!requires_grad) {
            throw std::runtime_error("backward called on tensor with requires_grad=false");
        }
        if (numel() != 1) {
            throw std::runtime_error("backward currently requires scalar tensor (numel==1)");
        }

        grad.assign(1, 1.0f);
        std::vector<Tensor<T>*> topo;
        std::unordered_set<Tensor<T>*> visited;
        build_topo(this, topo, visited);

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            Tensor<T>* node = *it;
            if (node->grad_fn && node->grad_fn->backward_fn) {
                node->grad_fn->backward_fn(node->grad);
            }
        }
    }

    void fill_random(T low = T(0), T high = T(1)) {
        static std::random_device rd;
        static std::mt19937 gen(rd());

        if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dis(low, high);
            for (auto& val : data) val = dis(gen);
        } else {
            float lo = static_cast<float>(to_compute(low));
            float hi = static_cast<float>(to_compute(high));
            std::uniform_real_distribution<float> dis(lo, hi);
            for (auto& val : data) val = from_compute<T>(dis(gen));
        }
    }

    void fill(T value) {
        std::fill(data.begin(), data.end(), value);
    }

    void zeros() { fill(T(0)); }
    void ones() { fill(T(1)); }

    T& at(const std::vector<size_t>& indices) {
        return data[compute_offset(indices)];
    }

    const T& at(const std::vector<size_t>& indices) const {
        return data[compute_offset(indices)];
    }

    TensorView<T> view_ref() { return TensorView<T>(shape, strides, data.data()); }

    TensorView<const T> view_ref() const { return TensorView<const T>(shape, strides, data.data()); }

    TensorView<T> view_ref(const std::vector<size_t>& new_shape) {
        if (numel_from_shape(new_shape) != numel()) {
            throw std::invalid_argument("View shape must have same number of elements");
        }
        if (!is_contiguous()) {
            throw std::runtime_error("Reshaped view_ref currently requires contiguous tensor");
        }
        return TensorView<T>(new_shape, compute_strides(new_shape), data.data());
    }

    TensorView<const T> view_ref(const std::vector<size_t>& new_shape) const {
        if (numel_from_shape(new_shape) != numel()) {
            throw std::invalid_argument("View shape must have same number of elements");
        }
        if (!is_contiguous()) {
            throw std::runtime_error("Reshaped view_ref currently requires contiguous tensor");
        }
        return TensorView<const T>(new_shape, compute_strides(new_shape), data.data());
    }

    Tensor view(const std::vector<size_t>& new_shape) const {
        if (numel_from_shape(new_shape) != numel()) {
            throw std::invalid_argument("View shape must have same number of elements");
        }
        Tensor result(new_shape);
        result.data = data;
        return result;
    }

    Tensor transpose(const std::vector<size_t>& axes) const {
        if (axes.size() != shape.size()) {
            throw std::invalid_argument("axes size must match tensor dimensions");
        }

        std::vector<int> seen(shape.size(), 0);
        for (size_t ax : axes) {
            if (ax >= shape.size() || seen[ax]) {
                throw std::invalid_argument("axes must be a valid permutation");
            }
            seen[ax] = 1;
        }

        std::vector<size_t> new_shape(shape.size());
        for (size_t i = 0; i < axes.size(); ++i) {
            new_shape[i] = shape[axes[i]];
        }

        Tensor result(new_shape);
        std::vector<size_t> out_idx(shape.size(), 0);
        std::vector<size_t> src_idx(shape.size(), 0);
        for (size_t linear = 0; linear < result.numel(); ++linear) {
            unravel_row_major(linear, result.shape, result.strides, out_idx);
            for (size_t d = 0; d < shape.size(); ++d) {
                src_idx[axes[d]] = out_idx[d];
            }
            result.data[linear] = data[compute_offset(src_idx)];
        }

        return result;
    }

    Tensor transpose_last_two() const {
        if (shape.size() < 2) {
            throw std::invalid_argument("transpose_last_two requires at least 2D tensor");
        }
        std::vector<size_t> axes(shape.size());
        std::iota(axes.begin(), axes.end(), 0);
        std::swap(axes[shape.size() - 2], axes[shape.size() - 1]);
        return transpose(axes);
    }

    Tensor relu() const {
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            auto v = to_compute(data[i]);
            result.data[i] = from_compute<T>(v > 0 ? v : 0);
        }

        if (requires_grad) {
            result.set_requires_grad(true);
            auto node = std::make_shared<AutogradNode<T>>();
            Tensor<T>* a = const_cast<Tensor<T>*>(this);
            const std::vector<T> x_data = data;
            node->parents = {a};
            node->backward_fn = [a, x_data](const std::vector<float>& grad_out) {
                if (!a->requires_grad) return;
                if (a->grad.size() != a->numel()) a->grad.assign(a->numel(), 0.0f);
                for (size_t i = 0; i < a->numel(); ++i) {
                    float mask = to_compute(x_data[i]) > 0 ? 1.0f : 0.0f;
                    a->grad[i] += grad_out[i] * mask;
                }
            };
            result.grad_fn = node;
        }

        return result;
    }

    Tensor softmax(int axis = -1) const {
        Tensor result(shape);
        if (data.empty()) return result;

        size_t ax = normalize_axis(axis);
        size_t outer = 1;
        for (size_t i = 0; i < ax; ++i) outer *= shape[i];
        size_t axis_size = shape[ax];
        size_t inner = 1;
        for (size_t i = ax + 1; i < shape.size(); ++i) inner *= shape[i];

        for (size_t o = 0; o < outer; ++o) {
            for (size_t in = 0; in < inner; ++in) {
                size_t base = o * axis_size * inner + in;
                float max_val = static_cast<float>(to_compute(data[base]));
                for (size_t a = 1; a < axis_size; ++a) {
                    size_t idx = base + a * inner;
                    max_val = std::max(max_val, static_cast<float>(to_compute(data[idx])));
                }

                double exp_sum = 0.0;
                for (size_t a = 0; a < axis_size; ++a) {
                    size_t idx = base + a * inner;
                    double e = std::exp(static_cast<double>(to_compute(data[idx])) - static_cast<double>(max_val));
                    exp_sum += e;
                    result.data[idx] = from_compute<T>(e);
                }

                if (exp_sum > 0.0) {
                    for (size_t a = 0; a < axis_size; ++a) {
                        size_t idx = base + a * inner;
                        double prob = static_cast<double>(to_compute(result.data[idx])) / exp_sum;
                        result.data[idx] = from_compute<T>(prob);
                    }
                }
            }
        }
        return result;
    }

    Tensor mean_axis(int axis, bool keepdims = false) const {
        if (data.empty()) return Tensor({});

        size_t ax = normalize_axis(axis);
        std::vector<size_t> out_shape;
        out_shape.reserve(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i == ax) {
                if (keepdims) out_shape.push_back(1);
            } else {
                out_shape.push_back(shape[i]);
            }
        }

        Tensor result(out_shape);
        size_t outer = 1;
        for (size_t i = 0; i < ax; ++i) outer *= shape[i];
        size_t axis_size = shape[ax];
        size_t inner = 1;
        for (size_t i = ax + 1; i < shape.size(); ++i) inner *= shape[i];

        for (size_t o = 0; o < outer; ++o) {
            for (size_t in = 0; in < inner; ++in) {
                double sum = 0.0;
                size_t base = o * axis_size * inner + in;
                for (size_t a = 0; a < axis_size; ++a) {
                    sum += static_cast<double>(to_compute(data[base + a * inner]));
                }
                result.data[o * inner + in] = from_compute<T>(sum / static_cast<double>(axis_size));
            }
        }

        if (requires_grad) {
            result.set_requires_grad(true);
            auto node = std::make_shared<AutogradNode<T>>();
            Tensor<T>* a = const_cast<Tensor<T>*>(this);
            node->parents = {a};
            node->backward_fn = [a, outer, inner, axis_size](const std::vector<float>& grad_out) {
                if (!a->requires_grad) return;
                if (a->grad.size() != a->numel()) a->grad.assign(a->numel(), 0.0f);
                for (size_t o = 0; o < outer; ++o) {
                    for (size_t in = 0; in < inner; ++in) {
                        float g = grad_out[o * inner + in] / static_cast<float>(axis_size);
                        size_t base = o * axis_size * inner + in;
                        for (size_t ax = 0; ax < axis_size; ++ax) {
                            a->grad[base + ax * inner] += g;
                        }
                    }
                }
            };
            result.grad_fn = node;
        }

        return result;
    }

    Tensor pow(T exponent) const {
        Tensor result(shape);
        double exp_v = static_cast<double>(to_compute(exponent));
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = from_compute<T>(std::pow(static_cast<double>(to_compute(data[i])), exp_v));
        }
        return result;
    }

    Tensor sqrt() const {
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = from_compute<T>(std::sqrt(static_cast<double>(to_compute(data[i]))));
        }
        return result;
    }

    Tensor sum() const {
        Tensor result({1});
        double acc = 0.0;
        for (const auto& v : data) {
            acc += static_cast<double>(to_compute(v));
        }
        result.data[0] = from_compute<T>(acc);

        if (requires_grad) {
            result.set_requires_grad(true);
            auto node = std::make_shared<AutogradNode<T>>();
            Tensor<T>* a = const_cast<Tensor<T>*>(this);
            node->parents = {a};
            node->backward_fn = [a](const std::vector<float>& grad_out) {
                if (!a->requires_grad) return;
                if (a->grad.size() != a->numel()) a->grad.assign(a->numel(), 0.0f);
                float g = grad_out.empty() ? 0.0f : grad_out[0];
                for (size_t i = 0; i < a->numel(); ++i) {
                    a->grad[i] += g;
                }
            };
            result.grad_fn = node;
        }

        return result;
    }

    static std::vector<size_t> broadcast_shape(const std::vector<size_t>& shape1,
                                               const std::vector<size_t>& shape2) {
        std::vector<size_t> result;
        int i = static_cast<int>(shape1.size()) - 1;
        int j = static_cast<int>(shape2.size()) - 1;

        while (i >= 0 || j >= 0) {
            size_t dim1 = (i >= 0) ? shape1[static_cast<size_t>(i)] : 1;
            size_t dim2 = (j >= 0) ? shape2[static_cast<size_t>(j)] : 1;

            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                throw std::invalid_argument("Incompatible shapes for broadcasting");
            }

            result.insert(result.begin(), std::max(dim1, dim2));
            --i;
            --j;
        }

        return result;
    }

    Tensor broadcast_to(const std::vector<size_t>& new_shape) const {
        auto target_shape = broadcast_shape(shape, new_shape);
        if (target_shape != new_shape) {
            throw std::invalid_argument("new_shape must be a valid broadcast target");
        }

        Tensor result(new_shape);
        if (data.empty()) return result;

        std::vector<size_t> out_idx(new_shape.size(), 0);
        std::vector<size_t> src_idx(shape.size(), 0);
        for (size_t linear = 0; linear < result.numel(); ++linear) {
            unravel_row_major(linear, result.shape, result.strides, out_idx);
            for (size_t i = 0; i < shape.size(); ++i) {
                size_t out_axis = new_shape.size() - shape.size() + i;
                src_idx[i] = (shape[i] == 1) ? 0 : out_idx[out_axis];
            }
            result.data[linear] = data[compute_offset(src_idx)];
        }
        return result;
    }

    Tensor operator+(const Tensor& other) const {
        std::vector<size_t> out_shape = broadcast_shape(shape, other.shape);
        Tensor lhs = (shape == out_shape) ? *this : this->broadcast_to(out_shape);
        Tensor rhs = (other.shape == out_shape) ? other : other.broadcast_to(out_shape);

        Tensor result(out_shape);
        for (size_t i = 0; i < result.data.size(); ++i) {
            result.data[i] = from_compute<T>(to_compute(lhs.data[i]) + to_compute(rhs.data[i]));
        }

        if (requires_grad || other.requires_grad) {
            result.set_requires_grad(true);
            auto node = std::make_shared<AutogradNode<T>>();
            Tensor<T>* a = const_cast<Tensor<T>*>(this);
            Tensor<T>* b = const_cast<Tensor<T>*>(&other);

            Tensor<T>* parent_a = a;
            if (a->grad_fn) {
                node->owned_parents.push_back(std::make_shared<Tensor<T>>(*a));
                parent_a = node->owned_parents.back().get();
            }

            Tensor<T>* parent_b = b;
            if (b->grad_fn) {
                node->owned_parents.push_back(std::make_shared<Tensor<T>>(*b));
                parent_b = node->owned_parents.back().get();
            }

            node->parents = {parent_a, parent_b};
            std::vector<size_t> left_shape = shape;
            std::vector<size_t> right_shape = other.shape;
            std::vector<size_t> out_shape_copy = out_shape;
            node->backward_fn = [parent_a, parent_b, left_shape, right_shape, out_shape_copy](const std::vector<float>& grad_out) {
                std::vector<float> g_left = reduce_grad_to_shape(grad_out, out_shape_copy, left_shape);
                std::vector<float> g_right = reduce_grad_to_shape(grad_out, out_shape_copy, right_shape);
                if (parent_a->requires_grad) {
                    if (parent_a->grad.size() != parent_a->numel()) parent_a->grad.assign(parent_a->numel(), 0.0f);
                    for (size_t i = 0; i < parent_a->numel(); ++i) parent_a->grad[i] += g_left[i];
                }
                if (parent_b->requires_grad) {
                    if (parent_b->grad.size() != parent_b->numel()) parent_b->grad.assign(parent_b->numel(), 0.0f);
                    for (size_t i = 0; i < parent_b->numel(); ++i) parent_b->grad[i] += g_right[i];
                }
            };
            result.grad_fn = node;
        }

        return result;
    }

    Tensor operator-(const Tensor& other) const {
        std::vector<size_t> out_shape = broadcast_shape(shape, other.shape);
        Tensor lhs = (shape == out_shape) ? *this : this->broadcast_to(out_shape);
        Tensor rhs = (other.shape == out_shape) ? other : other.broadcast_to(out_shape);

        Tensor result(out_shape);
        for (size_t i = 0; i < result.data.size(); ++i) {
            result.data[i] = from_compute<T>(to_compute(lhs.data[i]) - to_compute(rhs.data[i]));
        }

        if (requires_grad || other.requires_grad) {
            result.set_requires_grad(true);
            auto node = std::make_shared<AutogradNode<T>>();
            Tensor<T>* a = const_cast<Tensor<T>*>(this);
            Tensor<T>* b = const_cast<Tensor<T>*>(&other);

            Tensor<T>* parent_a = a;
            if (a->grad_fn) {
                node->owned_parents.push_back(std::make_shared<Tensor<T>>(*a));
                parent_a = node->owned_parents.back().get();
            }

            Tensor<T>* parent_b = b;
            if (b->grad_fn) {
                node->owned_parents.push_back(std::make_shared<Tensor<T>>(*b));
                parent_b = node->owned_parents.back().get();
            }

            std::vector<size_t> left_shape = shape;
            std::vector<size_t> right_shape = other.shape;
            std::vector<size_t> out_shape_copy = out_shape;
            node->parents = {parent_a, parent_b};
            node->backward_fn = [parent_a, parent_b, left_shape, right_shape, out_shape_copy](const std::vector<float>& grad_out) {
                std::vector<float> g_left = reduce_grad_to_shape(grad_out, out_shape_copy, left_shape);
                std::vector<float> g_right = reduce_grad_to_shape(grad_out, out_shape_copy, right_shape);
                if (parent_a->requires_grad) {
                    if (parent_a->grad.size() != parent_a->numel()) parent_a->grad.assign(parent_a->numel(), 0.0f);
                    for (size_t i = 0; i < parent_a->numel(); ++i) parent_a->grad[i] += g_left[i];
                }
                if (parent_b->requires_grad) {
                    if (parent_b->grad.size() != parent_b->numel()) parent_b->grad.assign(parent_b->numel(), 0.0f);
                    for (size_t i = 0; i < parent_b->numel(); ++i) parent_b->grad[i] -= g_right[i];
                }
            };
            result.grad_fn = node;
        }

        return result;
    }

    Tensor operator*(const Tensor& other) const {
        std::vector<size_t> out_shape = broadcast_shape(shape, other.shape);
        Tensor lhs = (shape == out_shape) ? *this : this->broadcast_to(out_shape);
        Tensor rhs = (other.shape == out_shape) ? other : other.broadcast_to(out_shape);

        Tensor result(out_shape);
        for (size_t i = 0; i < result.data.size(); ++i) {
            result.data[i] = from_compute<T>(to_compute(lhs.data[i]) * to_compute(rhs.data[i]));
        }

        if (requires_grad || other.requires_grad) {
            result.set_requires_grad(true);
            auto node = std::make_shared<AutogradNode<T>>();
            Tensor<T>* a = const_cast<Tensor<T>*>(this);
            Tensor<T>* b = const_cast<Tensor<T>*>(&other);
            std::vector<T> a_data = lhs.data;
            std::vector<T> b_data = rhs.data;

            Tensor<T>* parent_a = a;
            if (a->grad_fn) {
                node->owned_parents.push_back(std::make_shared<Tensor<T>>(*a));
                parent_a = node->owned_parents.back().get();
            }

            Tensor<T>* parent_b = b;
            if (b->grad_fn) {
                node->owned_parents.push_back(std::make_shared<Tensor<T>>(*b));
                parent_b = node->owned_parents.back().get();
            }

            node->parents = {parent_a, parent_b};
            std::vector<size_t> left_shape = shape;
            std::vector<size_t> right_shape = other.shape;
            std::vector<size_t> out_shape_copy = out_shape;
            node->backward_fn = [parent_a, parent_b, a_data, b_data, left_shape, right_shape, out_shape_copy](const std::vector<float>& grad_out) {
                std::vector<float> g_left(out_shape_copy.empty() ? 0 : Tensor<T>::numel_from_shape(out_shape_copy), 0.0f);
                std::vector<float> g_right(out_shape_copy.empty() ? 0 : Tensor<T>::numel_from_shape(out_shape_copy), 0.0f);
                for (size_t i = 0; i < g_left.size(); ++i) {
                    g_left[i] = grad_out[i] * static_cast<float>(to_compute(b_data[i]));
                    g_right[i] = grad_out[i] * static_cast<float>(to_compute(a_data[i]));
                }
                std::vector<float> red_left = reduce_grad_to_shape(g_left, out_shape_copy, left_shape);
                std::vector<float> red_right = reduce_grad_to_shape(g_right, out_shape_copy, right_shape);
                if (parent_a->requires_grad) {
                    if (parent_a->grad.size() != parent_a->numel()) parent_a->grad.assign(parent_a->numel(), 0.0f);
                    for (size_t i = 0; i < parent_a->numel(); ++i) {
                        parent_a->grad[i] += red_left[i];
                    }
                }
                if (parent_b->requires_grad) {
                    if (parent_b->grad.size() != parent_b->numel()) parent_b->grad.assign(parent_b->numel(), 0.0f);
                    for (size_t i = 0; i < parent_b->numel(); ++i) {
                        parent_b->grad[i] += red_right[i];
                    }
                }
            };
            result.grad_fn = node;
        }

        return result;
    }

    Tensor operator/(const Tensor& other) const {
        std::vector<size_t> out_shape = broadcast_shape(shape, other.shape);
        Tensor lhs = (shape == out_shape) ? *this : this->broadcast_to(out_shape);
        Tensor rhs = (other.shape == out_shape) ? other : other.broadcast_to(out_shape);

        Tensor result(out_shape);
        for (size_t i = 0; i < result.data.size(); ++i) {
            auto denom = to_compute(rhs.data[i]);
            if (denom == 0) throw std::runtime_error("Division by zero");
            result.data[i] = from_compute<T>(to_compute(lhs.data[i]) / denom);
        }

        if (requires_grad || other.requires_grad) {
            result.set_requires_grad(true);
            auto node = std::make_shared<AutogradNode<T>>();
            Tensor<T>* a = const_cast<Tensor<T>*>(this);
            Tensor<T>* b = const_cast<Tensor<T>*>(&other);
            std::vector<T> a_data = lhs.data;
            std::vector<T> b_data = rhs.data;

            Tensor<T>* parent_a = a;
            if (a->grad_fn) {
                node->owned_parents.push_back(std::make_shared<Tensor<T>>(*a));
                parent_a = node->owned_parents.back().get();
            }

            Tensor<T>* parent_b = b;
            if (b->grad_fn) {
                node->owned_parents.push_back(std::make_shared<Tensor<T>>(*b));
                parent_b = node->owned_parents.back().get();
            }

            std::vector<size_t> left_shape = shape;
            std::vector<size_t> right_shape = other.shape;
            std::vector<size_t> out_shape_copy = out_shape;
            node->parents = {parent_a, parent_b};
            node->backward_fn = [parent_a, parent_b, a_data, b_data, left_shape, right_shape, out_shape_copy](const std::vector<float>& grad_out) {
                std::vector<float> g_left(out_shape_copy.empty() ? 0 : Tensor<T>::numel_from_shape(out_shape_copy), 0.0f);
                std::vector<float> g_right(out_shape_copy.empty() ? 0 : Tensor<T>::numel_from_shape(out_shape_copy), 0.0f);
                for (size_t i = 0; i < g_left.size(); ++i) {
                    float a_v = static_cast<float>(to_compute(a_data[i]));
                    float b_v = static_cast<float>(to_compute(b_data[i]));
                    g_left[i] = grad_out[i] / b_v;
                    g_right[i] = -grad_out[i] * a_v / (b_v * b_v);
                }
                std::vector<float> red_left = reduce_grad_to_shape(g_left, out_shape_copy, left_shape);
                std::vector<float> red_right = reduce_grad_to_shape(g_right, out_shape_copy, right_shape);

                if (parent_a->requires_grad) {
                    if (parent_a->grad.size() != parent_a->numel()) parent_a->grad.assign(parent_a->numel(), 0.0f);
                    for (size_t i = 0; i < parent_a->numel(); ++i) parent_a->grad[i] += red_left[i];
                }
                if (parent_b->requires_grad) {
                    if (parent_b->grad.size() != parent_b->numel()) parent_b->grad.assign(parent_b->numel(), 0.0f);
                    for (size_t i = 0; i < parent_b->numel(); ++i) parent_b->grad[i] += red_right[i];
                }
            };
            result.grad_fn = node;
        }
        return result;
    }

    Tensor add_scalar(T scalar) const {
        Tensor result(shape);
        auto s = to_compute(scalar);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = from_compute<T>(to_compute(data[i]) + s);
        }
        return result;
    }

private:
    static std::vector<float> reduce_grad_to_shape(const std::vector<float>& grad_out,
                                                   const std::vector<size_t>& out_shape,
                                                   const std::vector<size_t>& target_shape) {
        if (target_shape == out_shape) {
            return grad_out;
        }

        size_t target_numel = numel_from_shape(target_shape);
        std::vector<float> reduced(target_numel, 0.0f);
        std::vector<size_t> out_strides = compute_strides(out_shape);
        std::vector<size_t> target_strides = compute_strides(target_shape);
        std::vector<size_t> out_idx(out_shape.size(), 0);
        std::vector<size_t> tgt_idx(target_shape.size(), 0);

        for (size_t linear = 0; linear < grad_out.size(); ++linear) {
            unravel_row_major(linear, out_shape, out_strides, out_idx);

            for (size_t i = 0; i < target_shape.size(); ++i) {
                size_t out_axis = out_shape.size() - target_shape.size() + i;
                tgt_idx[i] = (target_shape[i] == 1) ? 0 : out_idx[out_axis];
            }

            size_t target_linear = 0;
            for (size_t i = 0; i < target_shape.size(); ++i) {
                target_linear += tgt_idx[i] * target_strides[i];
            }
            reduced[target_linear] += grad_out[linear];
        }
        return reduced;
    }

    static void build_topo(Tensor<T>* node, std::vector<Tensor<T>*>& topo, std::unordered_set<Tensor<T>*>& visited) {
        if (!node || visited.count(node)) return;
        visited.insert(node);
        if (node->grad_fn) {
            for (Tensor<T>* p : node->grad_fn->parents) build_topo(p, topo, visited);
        }
        topo.push_back(node);
    }

    size_t normalize_axis(int axis) const {
        if (shape.empty()) {
            throw std::invalid_argument("Axis operation requires non-empty shape");
        }
        int ndim_i = static_cast<int>(shape.size());
        int ax = axis;
        if (ax < 0) ax += ndim_i;
        if (ax < 0 || ax >= ndim_i) {
            throw std::out_of_range("Axis out of range");
        }
        return static_cast<size_t>(ax);
    }

    size_t compute_offset(const std::vector<size_t>& indices) const {
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("indices rank must match tensor rank");
        }
        size_t offset = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("tensor index out of bounds");
            }
            offset += indices[i] * strides[i];
        }
        return offset;
    }

    static void unravel_row_major(size_t linear,
                                  const std::vector<size_t>& shape_,
                                  const std::vector<size_t>& strides_,
                                  std::vector<size_t>& out_idx) {
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (shape_[i] == 0) {
                out_idx[i] = 0;
                continue;
            }
            out_idx[i] = (linear / strides_[i]) % shape_[i];
        }
    }
};

template<typename Scalar, typename T>
Tensor<T> operator*(Scalar scalar, const Tensor<T>& t) {
    Tensor<T> result(t.shape);
    auto s = static_cast<float>(scalar);
    for (size_t i = 0; i < t.data.size(); ++i) {
        result.data[i] = from_compute<T>(s * to_compute(t.data[i]));
    }
    return result;
}

template<typename OutT, typename AT, typename BT>
Tensor<OutT> matmul_mixed(const Tensor<AT>& a, const Tensor<BT>& b) {
    if (a.ndim() < 2 || b.ndim() < 2) {
        throw std::invalid_argument("matmul requires at least 2D tensors");
    }

    using Acc = float;

    if (a.ndim() == 2 && b.ndim() == 2) {
        size_t m = a.shape[0];
        size_t k = a.shape[1];
        size_t n = b.shape[1];

        if (k != b.shape[0]) {
            throw std::invalid_argument("Incompatible shapes for matmul");
        }

        Tensor<OutT> result({m, n});
#if defined(_OPENMP)
#pragma omp parallel for collapse(2)
#endif
        for (long long i = 0; i < static_cast<long long>(m); ++i) {
            for (long long j = 0; j < static_cast<long long>(n); ++j) {
                Acc sum = 0;
                for (size_t p = 0; p < k; ++p) {
                    sum += static_cast<Acc>(to_compute(a.data[static_cast<size_t>(i) * k + p])) *
                           static_cast<Acc>(to_compute(b.data[p * n + static_cast<size_t>(j)]));
                }
                result.data[static_cast<size_t>(i) * n + static_cast<size_t>(j)] = from_compute<OutT>(sum);
            }
        }
        return result;
    }

    if (a.ndim() == 3 && b.ndim() == 3) {
        size_t batch = a.shape[0];
        size_t m = a.shape[1];
        size_t k = a.shape[2];
        if (batch != b.shape[0] || k != b.shape[1]) {
            throw std::invalid_argument("Incompatible batched shapes for matmul");
        }
        size_t n = b.shape[2];

        Tensor<OutT> result({batch, m, n});
#if defined(_OPENMP)
#pragma omp parallel for collapse(2)
#endif
        for (long long bt = 0; bt < static_cast<long long>(batch); ++bt) {
            for (long long i = 0; i < static_cast<long long>(m); ++i) {
                for (size_t j = 0; j < n; ++j) {
                    Acc sum = 0;
                    for (size_t p = 0; p < k; ++p) {
                        size_t a_idx = static_cast<size_t>(bt) * (m * k) + static_cast<size_t>(i) * k + p;
                        size_t b_idx = static_cast<size_t>(bt) * (k * n) + p * n + j;
                        sum += static_cast<Acc>(to_compute(a.data[a_idx])) *
                               static_cast<Acc>(to_compute(b.data[b_idx]));
                    }
                    result.data[static_cast<size_t>(bt) * (m * n) + static_cast<size_t>(i) * n + j] = from_compute<OutT>(sum);
                }
            }
        }
        return result;
    }

    throw std::invalid_argument("matmul supports 2D or batched 3D tensors");
}

template<typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.ndim() < 2 || b.ndim() < 2) {
        throw std::invalid_argument("matmul requires at least 2D tensors");
    }

    if (a.ndim() == 2 && b.ndim() == 2) {
        size_t m = a.shape[0];
        size_t k = a.shape[1];
        size_t n = b.shape[1];

        if (k != b.shape[0]) {
            throw std::invalid_argument("Incompatible shapes for matmul");
        }

        Tensor<T> result({m, n});
#if defined(_OPENMP)
#pragma omp parallel for collapse(2)
#endif
        for (long long i = 0; i < static_cast<long long>(m); ++i) {
            for (long long j = 0; j < static_cast<long long>(n); ++j) {
                float sum = 0.0f;
                for (size_t p = 0; p < k; ++p) {
                    sum += static_cast<float>(to_compute(a.data[static_cast<size_t>(i) * k + p])) *
                           static_cast<float>(to_compute(b.data[p * n + static_cast<size_t>(j)]));
                }
                result.data[static_cast<size_t>(i) * n + static_cast<size_t>(j)] = from_compute<T>(sum);
            }
        }

        if (a.requires_grad || b.requires_grad) {
            result.set_requires_grad(true);
            auto node = std::make_shared<AutogradNode<T>>();
            Tensor<T>* pa = const_cast<Tensor<T>*>(&a);
            Tensor<T>* pb = const_cast<Tensor<T>*>(&b);
            std::vector<T> a_data = a.data;
            std::vector<T> b_data = b.data;

            Tensor<T>* parent_a = pa;
            if (pa->grad_fn) {
                node->owned_parents.push_back(std::make_shared<Tensor<T>>(*pa));
                parent_a = node->owned_parents.back().get();
            }

            Tensor<T>* parent_b = pb;
            if (pb->grad_fn) {
                node->owned_parents.push_back(std::make_shared<Tensor<T>>(*pb));
                parent_b = node->owned_parents.back().get();
            }

            node->parents = {parent_a, parent_b};
            node->backward_fn = [parent_a, parent_b, a_data, b_data, m, k, n](const std::vector<float>& grad_out) {
                if (parent_a->requires_grad) {
                    if (parent_a->grad.size() != parent_a->numel()) parent_a->grad.assign(parent_a->numel(), 0.0f);
                    for (size_t i = 0; i < m; ++i) {
                        for (size_t p = 0; p < k; ++p) {
                            float acc = 0.0f;
                            for (size_t j = 0; j < n; ++j) {
                                acc += grad_out[i * n + j] * static_cast<float>(to_compute(b_data[p * n + j]));
                            }
                            parent_a->grad[i * k + p] += acc;
                        }
                    }
                }
                if (parent_b->requires_grad) {
                    if (parent_b->grad.size() != parent_b->numel()) parent_b->grad.assign(parent_b->numel(), 0.0f);
                    for (size_t p = 0; p < k; ++p) {
                        for (size_t j = 0; j < n; ++j) {
                            float acc = 0.0f;
                            for (size_t i = 0; i < m; ++i) {
                                acc += static_cast<float>(to_compute(a_data[i * k + p])) * grad_out[i * n + j];
                            }
                            parent_b->grad[p * n + j] += acc;
                        }
                    }
                }
            };
            result.grad_fn = node;
        }

        return result;
    }

    if (a.ndim() == 3 && b.ndim() == 3) {
        size_t batch = a.shape[0];
        size_t m = a.shape[1];
        size_t k = a.shape[2];
        if (batch != b.shape[0] || k != b.shape[1]) {
            throw std::invalid_argument("Incompatible batched shapes for matmul");
        }
        size_t n = b.shape[2];

        Tensor<T> result({batch, m, n});
#if defined(_OPENMP)
#pragma omp parallel for collapse(2)
#endif
        for (long long bt = 0; bt < static_cast<long long>(batch); ++bt) {
            for (long long i = 0; i < static_cast<long long>(m); ++i) {
                for (size_t j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (size_t p = 0; p < k; ++p) {
                        size_t a_idx = static_cast<size_t>(bt) * (m * k) + static_cast<size_t>(i) * k + p;
                        size_t b_idx = static_cast<size_t>(bt) * (k * n) + p * n + j;
                        sum += static_cast<float>(to_compute(a.data[a_idx])) *
                               static_cast<float>(to_compute(b.data[b_idx]));
                    }
                    result.data[static_cast<size_t>(bt) * (m * n) + static_cast<size_t>(i) * n + j] = from_compute<T>(sum);
                }
            }
        }

        if (a.requires_grad || b.requires_grad) {
            result.set_requires_grad(true);
            auto node = std::make_shared<AutogradNode<T>>();
            Tensor<T>* pa = const_cast<Tensor<T>*>(&a);
            Tensor<T>* pb = const_cast<Tensor<T>*>(&b);
            std::vector<T> a_data = a.data;
            std::vector<T> b_data = b.data;

            Tensor<T>* parent_a = pa;
            if (pa->grad_fn) {
                node->owned_parents.push_back(std::make_shared<Tensor<T>>(*pa));
                parent_a = node->owned_parents.back().get();
            }

            Tensor<T>* parent_b = pb;
            if (pb->grad_fn) {
                node->owned_parents.push_back(std::make_shared<Tensor<T>>(*pb));
                parent_b = node->owned_parents.back().get();
            }

            node->parents = {parent_a, parent_b};
            node->backward_fn = [parent_a, parent_b, a_data, b_data, batch, m, k, n](const std::vector<float>& grad_out) {
                if (parent_a->requires_grad) {
                    if (parent_a->grad.size() != parent_a->numel()) parent_a->grad.assign(parent_a->numel(), 0.0f);
                    for (size_t bt = 0; bt < batch; ++bt) {
                        for (size_t i = 0; i < m; ++i) {
                            for (size_t p = 0; p < k; ++p) {
                                float acc = 0.0f;
                                for (size_t j = 0; j < n; ++j) {
                                    size_t go_idx = bt * (m * n) + i * n + j;
                                    size_t b_idx = bt * (k * n) + p * n + j;
                                    acc += grad_out[go_idx] * static_cast<float>(to_compute(b_data[b_idx]));
                                }
                                size_t a_idx = bt * (m * k) + i * k + p;
                                parent_a->grad[a_idx] += acc;
                            }
                        }
                    }
                }

                if (parent_b->requires_grad) {
                    if (parent_b->grad.size() != parent_b->numel()) parent_b->grad.assign(parent_b->numel(), 0.0f);
                    for (size_t bt = 0; bt < batch; ++bt) {
                        for (size_t p = 0; p < k; ++p) {
                            for (size_t j = 0; j < n; ++j) {
                                float acc = 0.0f;
                                for (size_t i = 0; i < m; ++i) {
                                    size_t a_idx = bt * (m * k) + i * k + p;
                                    size_t go_idx = bt * (m * n) + i * n + j;
                                    acc += static_cast<float>(to_compute(a_data[a_idx])) * grad_out[go_idx];
                                }
                                size_t b_idx = bt * (k * n) + p * n + j;
                                parent_b->grad[b_idx] += acc;
                            }
                        }
                    }
                }
            };
            result.grad_fn = node;
        }

        return result;
    }

    return matmul_mixed<T, T, T>(a, b);
}

template<typename T>
class SGD {
public:
    SGD(std::vector<Tensor<T>*> params_,
        float lr_,
        float weight_decay_ = 0.0f,
        float max_grad_norm_ = 0.0f)
        : params(std::move(params_)),
          lr(lr_),
          weight_decay(weight_decay_),
          max_grad_norm(max_grad_norm_) {}

    void zero_grad() {
        for (Tensor<T>* p : params) {
            if (p) p->zero_grad();
        }
    }

    void step() {
        float grad_scale = 1.0f;
        if (max_grad_norm > 0.0f) {
            double norm_sq = 0.0;
            for (Tensor<T>* p : params) {
                if (!p || !p->requires_grad || p->grad.size() != p->numel()) continue;
                for (float g : p->grad) {
                    norm_sq += static_cast<double>(g) * static_cast<double>(g);
                }
            }
            float norm = static_cast<float>(std::sqrt(norm_sq));
            if (norm > max_grad_norm) {
                grad_scale = max_grad_norm / (norm + 1e-12f);
            }
        }

        for (Tensor<T>* p : params) {
            if (!p || !p->requires_grad || p->grad.size() != p->numel()) continue;
            for (size_t i = 0; i < p->numel(); ++i) {
                float g = p->grad[i] * grad_scale;
                float wd = weight_decay * static_cast<float>(to_compute(p->data[i]));
                float update = lr * (g + wd);
                p->data[i] = from_compute<T>(static_cast<float>(to_compute(p->data[i])) - update);
            }
        }
    }

private:
    std::vector<Tensor<T>*> params;
    float lr;
    float weight_decay;
    float max_grad_norm;
};
