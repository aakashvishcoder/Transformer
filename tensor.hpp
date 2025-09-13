#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <memory>   
#include <functional> 
#include <unordered_set>

using namespace std;

template<typename T, size_t N>
class Tensor {
public:
    using Shape = array<size_t, N>;
    vector<Tensor<T,N>*> parents_;
    // Constructors
    Tensor() = default;

    Tensor(const Shape& shape_, bool requires_grad = false)
        : shape__(shape_), requires_grad_(requires_grad) {
        compute_strides();
        size_t total_size = 1;
        for (auto s : shape__) total_size *= s;
        data_.resize(total_size);
        if (requires_grad_) grad_.resize(total_size, T(0));
    }

    Tensor(const Shape& shape_, const std::vector<T>& data, bool requires_grad = false)
        : shape__(shape_), data_(data), requires_grad_(requires_grad) {
        compute_strides();
        size_t expected_size = 1;
        for (auto s : shape__) expected_size *= s;
        if (expected_size != data_.size())
            throw std::runtime_error("Data size does not match shape");
        if (requires_grad_) grad_.resize(data_.size(), T(0));
    }

    // Getters
    const Shape& get_shape() const { return shape__; }
    const vector<T>& get_data() const { return data_; }
    const Shape& get_strides() const { return strides_; }
    
    void set_requires_grad(bool flag) {
        requires_grad_ = flag;
        if (flag && grad_.size() != data_.size())
            grad_.resize(data_.size(), T(0));
    }

    bool requires_grad() const { return requires_grad_; }

    vector<T>& get_data_ref() { return data_; }
    Shape& get_shape_ref() { return shape__; }
    Shape& get_strides_ref() { return strides_; }
    const std::vector<T>& grad() const { return grad_; }
    std::vector<T>& grad() { return grad_; }


    const vector<T>& get_data_ref() const { return data_; }
    const Shape& get_shape_ref() const { return shape__; }
    const Shape& get_strides_ref() const { return strides_; }
    const std::function<void(Tensor<T,2>*)>& get_backward_fn() const { return backward_fn_; }

    // Setter

    void initialize(const Shape& shape_) {
        shape__ = shape_;
        compute_strides();
        size_t total_size = 1;
        for (auto s : shape__) total_size *= s;
        data_.resize(total_size);
    }

    void set_backward_fn(function<void(Tensor<T,N>*)> fn) { backward_fn_ = std::move(fn); }

    // Print
    void print() const {
        array<size_t, N> idx{};
        print_recursive(0, idx);
        cout << "\n";
    }

    void print_shape() const {
        cout << "[";
        for (size_t i = 0; i < N; i++) {
            cout << shape__[i];
            if (i != N - 1) cout << ", ";
        }
        cout << "]\n";
    }

    // Sum
    Tensor<T,1> sum() const {
        T total = std::accumulate(data_.begin(), data_.end(), T(0));
        Tensor<T,1> result({1}, {total}, requires_grad_);

        if (requires_grad_) {
            auto* lhs = const_cast<Tensor<T,N>*>(this);
            result.parents_ = {lhs};

            result.set_backward_fn([lhs](Tensor<T,1>* self) {
                const auto grad_out = self->grad_[0];
                if (lhs->requires_grad_) {
                    if (lhs->grad_.empty()) lhs->grad_.resize(lhs->data_.size(), T(0));
                    for (size_t i=0; i<lhs->data_.size(); i++)
                        lhs->grad_[i] += grad_out;
                }
            });
        }
        return result;
    }

    // Fill
    void fill_value(T val = T(0)) { for (auto& v : data_) v = val; }
    void ones() { fill_value(T(1)); }
    void zeros() { fill_value(T(0)); }

    void fill_random(T min_val = T(0), T max_val = T(1)) {
        random_device rd;
        mt19937 gen(rd());

        if constexpr (is_floating_point_v<T>) {
            // Floating point: uniform_real_distribution
            uniform_real_distribution<T> dist(min_val, max_val);
            for (auto& x : data_) x = dist(gen);
        } else if constexpr (is_integral_v<T>) {
            // Integer: use signed type to handle negative ranges
            using signed_type = make_signed_t<T>;
            uniform_int_distribution<signed_type> dist(
                static_cast<signed_type>(min_val),
                static_cast<signed_type>(max_val)
            );
            for (auto& x : data_) x = dist(gen);
        } else {
            static_assert(is_arithmetic_v<T>, "fill_random requires numeric type");
        }
    }

    void clip(T min_val = T(1e-10), T max_val = T(1-1e-10)) {
        for(auto& v : data_) {
            if (v < min_val) v = min_val;
            else if (v > max_val) v = max_val;
        }
    }

    // Access operators
    T& operator()(const array<size_t, N>& idx) {
        size_t flat = 0;
        for (size_t i = 0; i < N; i++) flat += idx[i] * strides_[i];
        return data_[flat];
    }

    const T& operator()(const array<size_t, N>& idx) const {
        size_t flat = 0;
        for (size_t i = 0; i < N; i++) flat += idx[i] * strides_[i];
        return data_[flat];
    }

    // Arithmetic with another tensor
    Tensor<T,N>& operator+=(const Tensor<T,N>& other) {
        if (shape__ != other.shape__)
            throw runtime_error("Dimension mismatch for addition!");
        for (size_t i = 0; i < data_.size(); i++)
            data_[i] += other.data_[i];
        return *this;
    }

    Tensor<T,N> operator+(const Tensor<T,N>& other) const {
        Tensor<T,N> result(shape__);
        for (size_t i=0; i<data_.size(); i++)
            result.data_[i] = data_[i] + other.data_[i];

        if (requires_grad_ || other.requires_grad_) {
            result.requires_grad_ = true;
            auto* lhs = const_cast<Tensor<T,N>*>(this);
            auto* rhs = const_cast<Tensor<T,N>*>(&other);
            result.parents_ = {lhs, rhs};

            result.set_backward_fn([lhs, rhs](Tensor<T,N>* self) {
                const auto& grad_out = self->get_grad_ref();
                if (lhs->requires_grad_) {
                    if (lhs->grad_.empty()) lhs->grad_.resize(lhs->data_.size(), T(0));
                    for (size_t i=0; i<lhs->data_.size(); i++)
                        lhs->grad_[i] += grad_out[i];
                }
                if (rhs->requires_grad_) {
                    if (rhs->grad_.empty()) rhs->grad_.resize(rhs->data_.size(), T(0));
                    for (size_t i=0; i<rhs->data_.size(); i++)
                        rhs->grad_[i] += grad_out[i];
                }
            });
        }
        return result;
    }


    Tensor<T,N>& operator-=(const Tensor<T,N>& other) {
        if (shape__ != other.shape__)
            throw runtime_error("Dimension mismatch for subtraction!");
        for (size_t i = 0; i < data_.size(); i++)
            data_[i] -= other.data_[i];
        return *this;
    }

    Tensor<T,N> operator-(const Tensor<T,N>& other) const {
        Tensor<T,N> result(shape__);
        for (size_t i=0; i<data_.size(); i++)
            result.data_[i] = data_[i] - other.data_[i];

        if (requires_grad_ || other.requires_grad_) {
            result.requires_grad_ = true;
            auto* lhs = const_cast<Tensor<T,N>*>(this);
            auto* rhs = const_cast<Tensor<T,N>*>(&other);
            result.parents_ = {lhs, rhs};

            result.set_backward_fn([lhs, rhs](Tensor<T,N>* self) {
                const auto& grad_out = self->get_grad_ref();
                if (lhs->requires_grad_) {
                    if (lhs->grad_.empty()) lhs->grad_.resize(lhs->data_.size(), T(0));
                    for (size_t i=0; i<lhs->data_.size(); i++)
                        lhs->grad_[i] += grad_out[i];
                }
                if (rhs->requires_grad_) {
                    if (rhs->grad_.empty()) rhs->grad_.resize(rhs->data_.size(), T(0));
                    for (size_t i=0; i<rhs->data_.size(); i++)
                        rhs->grad_[i] -= grad_out[i];   // 🔥 FIXED SIGN
                }
            });
        }
        return result;
    }

    Tensor<T,N>& operator*=(const Tensor<T,N>& other) {
        for(size_t flat = 0; flat < data_.size(); ++flat) {
            // convert flat to multi-index
            array<size_t,N> idx{};
            size_t rem = flat;
            for(int d=N-1; d>=0; --d){
                idx[d] = rem % shape__[d];
                rem /= shape__[d];
            }
            size_t other_flat = other.flat_index_broadcast(idx, other);
            data_[flat] *= other.get_data_ref()[other_flat];
        }
        return *this;
    }

    Tensor<T,N> operator*(const Tensor<T,N>& other) const {
        Tensor<T,N> result(shape__);
        for (size_t i=0; i<data_.size(); i++)
            result.data_[i] = data_[i] * other.data_[i];

        if (requires_grad_ || other.requires_grad_) {
            result.requires_grad_ = true;
            auto* lhs = const_cast<Tensor<T,N>*>(this);
            auto* rhs = const_cast<Tensor<T,N>*>(&other);
            result.parents_ = {lhs, rhs};

            result.set_backward_fn([lhs, rhs](Tensor<T,N>* self) {
                const auto& grad_out = self->get_grad_ref();
                if (lhs->requires_grad_) {
                    if (lhs->grad_.empty()) lhs->grad_.resize(lhs->data_.size(), T(0));
                    for (size_t i=0; i<lhs->data_.size(); i++)
                        lhs->grad_[i] += rhs->data_[i] * grad_out[i];
                }
                if (rhs->requires_grad_) {
                    if (rhs->grad_.empty()) rhs->grad_.resize(rhs->data_.size(), T(0));
                    for (size_t i=0; i<rhs->data_.size(); i++)
                        rhs->grad_[i] += lhs->data_[i] * grad_out[i];
                }
            });
        }
        return result;
    }

    Tensor<T,N>& operator/=(const Tensor<T,N>& other) {
        if (shape__ != other.shape__)
            throw runtime_error("Dimension mismatch for division!");
        for (size_t i = 0; i < data_.size(); i++)
            data_[i] /= other.data_[i];
        return *this;
    }

    Tensor<T,N> operator/(const Tensor<T,N>& other) const {
        Tensor<T,N> result(shape__);
        for (size_t i=0; i<data_.size(); i++)
            result.data_[i] = data_[i] / other.data_[i];

        if (requires_grad_ || other.requires_grad_) {
            result.requires_grad_ = true;
            auto* lhs = const_cast<Tensor<T,N>*>(this);
            auto* rhs = const_cast<Tensor<T,N>*>(&other);
            result.parents_ = {lhs, rhs};

            result.set_backward_fn([lhs, rhs](Tensor<T,N>* self) {
                const auto& grad_out = self->get_grad_ref();
                if (lhs->requires_grad_) {
                    if (lhs->grad_.empty()) lhs->grad_.resize(lhs->data_.size(), T(0));
                    for (size_t i=0; i<lhs->data_.size(); i++)
                        lhs->grad_[i] += grad_out[i] / rhs->data_[i];
                }
                if (rhs->requires_grad_) {
                    if (rhs->grad_.empty()) rhs->grad_.resize(rhs->data_.size(), T(0));
                    for (size_t i=0; i<rhs->data_.size(); i++)
                        rhs->grad_[i] += -lhs->data_[i] * grad_out[i] / (rhs->data_[i] * rhs->data_[i]);
                }
            });
        }
        return result;
    }

    // Exponential
    Tensor<T,N> exp() const {
        Tensor<T,N> result(shape__);
        for (size_t i=0; i<data_.size(); i++)
            result.data_[i] = std::exp(data_[i]);

        if (requires_grad_) {
            result.requires_grad_ = true;
            auto* lhs = const_cast<Tensor<T,N>*>(this);
            result.parents_ = {lhs};

            result.set_backward_fn([lhs](Tensor<T,N>* self) {
                const auto& grad_out = self->get_grad_ref();
                for (size_t i=0; i<lhs->data_.size(); i++) {
                    if (lhs->grad_.empty()) lhs->grad_.resize(lhs->data_.size(), T(0));
                    lhs->grad_[i] += std::exp(lhs->data_[i]) * grad_out[i];
                }
            });
        }
        return result;
    }

    // Natural logarithm
    Tensor<T,N> log() const {
        Tensor<T,N> result(shape__);
        for (size_t i=0; i<data_.size(); i++)
            result.data_[i] = std::log(data_[i]);

        if (requires_grad_) {
            result.requires_grad_ = true;
            auto* lhs = const_cast<Tensor<T,N>*>(this);
            result.parents_ = {lhs};

            result.set_backward_fn([lhs](Tensor<T,N>* self) {
                const auto& grad_out = self->get_grad_ref();
                for (size_t i=0; i<lhs->data_.size(); i++) {
                    if (lhs->grad_.empty()) lhs->grad_.resize(lhs->data_.size(), T(0));
                    lhs->grad_[i] += (1.0 / lhs->data_[i]) * grad_out[i];
                }
            });
        }
        return result;
    }

    // Square root
    Tensor<T,N> sqrt() const {
        Tensor<T,N> result(shape__);
        for (size_t i=0; i<data_.size(); i++)
            result.data_[i] = std::sqrt(data_[i]);

        if (requires_grad_) {
            result.requires_grad_ = true;
            auto* lhs = const_cast<Tensor<T,N>*>(this);
            result.parents_ = {lhs};

            result.set_backward_fn([lhs](Tensor<T,N>* self) {
                const auto& grad_out = self->get_grad_ref();
                for (size_t i=0; i<lhs->data_.size(); i++) {
                    if (lhs->grad_.empty()) lhs->grad_.resize(lhs->data_.size(), T(0));
                    lhs->grad_[i] += (0.5 / std::sqrt(lhs->data_[i])) * grad_out[i];
                }
            });
        }
        return result;
    }

    // Negation
    Tensor<T,N> operator-() const {
        Tensor<T,N> result(shape__);
        for (size_t i=0; i<data_.size(); i++)
            result.data_[i] = -data_[i];

        if (requires_grad_) {
            result.requires_grad_ = true;
            auto* lhs = const_cast<Tensor<T,N>*>(this);
            result.parents_ = {lhs};

            result.set_backward_fn([lhs](Tensor<T,N>* self) {
                const auto& grad_out = self->get_grad_ref();
                for (size_t i=0; i<lhs->data_.size(); i++) {
                    if (lhs->grad_.empty()) lhs->grad_.resize(lhs->data_.size(), T(0));
                    lhs->grad_[i] -= grad_out[i];  // 🔥 derivative of -x is -1
                }
            });
        }
        return result;
    }

    Tensor<T,N> exp_() {
        for(auto &v : data_) v = std::exp(v);
        return *this;
    }

    Tensor<T,N> sqrt_() {
        for(auto &v : data_) v = std::sqrt(v);
        return *this;
    }

    Tensor<T,N> log_() {
        for(auto& v: data_) v = std::log(v);
        return *this;
    }

    // Addition with scalar: x + scalar
    Tensor<T,N> operator+(T scalar) {
        Tensor<T,N> result(shape__, requires_grad_);
        if (requires_grad_) result.grad_.resize(result.data_.size(), T(0));

        auto& out_data = result.get_data_ref();
        for(size_t i=0;i<data_.size();++i)
            out_data[i] = data_[i] + scalar;

        if(requires_grad_) {
            result.parents_ = {this};
            result.set_backward_fn([this, &result]() {
                auto& grad_out = result.get_grad_ref();
                for(size_t i=0;i<data_.size();++i)
                    this->grad_[i] += grad_out[i];  // dy/dx = 1
            });
        }

        return result;
    }

    // Subtraction with scalar: x - scalar
    Tensor<T,N> operator-(T scalar) {
        Tensor<T,N> result(shape__, requires_grad_);
        if (requires_grad_) result.grad_.resize(result.data_.size(), T(0));

        auto& out_data = result.get_data_ref();
        for(size_t i=0;i<data_.size();++i)
            out_data[i] = data_[i] - scalar;

        if(requires_grad_) {
            result.parents_ = {this};
            result.set_backward_fn([this, &result]() {
                auto& grad_out = result.get_grad_ref();
                for(size_t i=0;i<data_.size();++i)
                    this->grad_[i] += grad_out[i];  // dy/dx = 1
            });
        }

        return result;
    }

    // Multiplication with scalar: x * scalar
    Tensor<T,N> operator*(T scalar) const {
        Tensor<T,N> result(shape__);
        for (size_t i=0; i<data_.size(); i++)
            result.data_[i] = data_[i] * scalar;

        if (requires_grad_) {
            result.requires_grad_ = true;
            auto* lhs = const_cast<Tensor<T,N>*>(this);
            result.parents_ = {lhs};

            result.set_backward_fn([lhs, scalar](Tensor<T,N>* self) {
                const auto& grad_out = self->get_grad_ref();
                if (lhs->requires_grad_) {
                    if (lhs->grad_.empty()) lhs->grad_.resize(lhs->data_.size(), T(0));
                    for (size_t i=0; i<lhs->data_.size(); i++)
                        lhs->grad_[i] += scalar * grad_out[i];
                }
            });
        }
        return result;
    }

    // Division by scalar: x / scalar
    Tensor<T,N> operator/(T scalar) {
        Tensor<T,N> result(shape__, requires_grad_);
        if (requires_grad_) result.grad_.resize(result.data_.size(), T(0));

        auto& out_data = result.get_data_ref();
        for(size_t i=0;i<data_.size();++i)
            out_data[i] = data_[i] / scalar;

        if(requires_grad_) {
            result.parents_ = {this};
            result.set_backward_fn([this, &result, scalar]() {
                auto& grad_out = result.get_grad_ref();
                for(size_t i=0;i<data_.size();++i)
                    this->grad_[i] += grad_out[i] / scalar;  // dy/dx = 1/scalar
            });
        }

        return result;
    }

    // Scalar - x (global operator)
    template<typename U=size_t>
    friend Tensor<T,N> operator-(T scalar, const Tensor<T,N>& tensor) {
        Tensor<T,N> result(tensor.get_shape(), tensor.requires_grad());
        if(result.requires_grad_) result.grad_.resize(result.data_.size(), T(0));

        auto& out_data = result.get_data_ref();
        const auto& data = tensor.get_data_ref();
        for(size_t i=0;i<data.size();++i)
            out_data[i] = scalar - data[i];

        if(tensor.requires_grad_) {
            result.parents_ = {const_cast<Tensor<T,N>*>(&tensor)};
            result.set_backward_fn([&tensor, &result]() {
                auto& grad_out = result.get_grad_ref();
                auto& grad_in  = const_cast<Tensor<T,N>&>(tensor).grad_;
                for(size_t i=0;i<grad_out.size();++i)
                    grad_in[i] += -grad_out[i];  // dy/dx = -1
            });
        }

        return result;
    }


    // Power with scalar
    Tensor<T,N> pow(T exponent) const {
        Tensor<T,N> result(shape__);
        for (size_t i=0; i<data_.size(); i++)
            result.data_[i] = std::pow(data_[i], exponent);

        if (requires_grad_) {
            result.requires_grad_ = true;
            auto* lhs = const_cast<Tensor<T,N>*>(this);
            result.parents_ = {lhs};

            result.set_backward_fn([lhs, exponent](Tensor<T,N>* self) {
                const auto& grad_out = self->get_grad_ref();
                if (lhs->requires_grad_) {
                    if (lhs->grad_.empty()) lhs->grad_.resize(lhs->data_.size(), T(0));
                    for (size_t i=0; i<lhs->data_.size(); i++)
                        lhs->grad_[i] += exponent * std::pow(lhs->data_[i], exponent-1) * grad_out[i];
                }
            });
        }
        return result;
    }

    Tensor<T,N>& operator^=(T scalar) {
        for (auto &v : data_) v = std::pow(v, scalar);
        return *this;
    }

    // Sum along axis (N -> N-1 tensor)
    Tensor<T, N-1> sum_axis(size_t axis)  {
        if(axis >= N) throw runtime_error("Axis out of bounds");

        // Compute output shape
        array<size_t, N-1> new_shape{};
        for(size_t i=0,j=0;i<N;i++) if(i!=axis) new_shape[j++] = shape__[i];

        Tensor<T, N-1> out(new_shape);
        std::fill(out.get_data_ref().begin(), out.get_data_ref().end(), T(0));

        // Forward pass: sum along axis
        array<size_t,N> idx{};
        for(size_t flat=0; flat<data_.size(); ++flat){
            size_t rem = flat;
            for(int d=N-1; d>=0; --d){
                idx[d] = rem % shape__[d];
                rem /= shape__[d];
            }

            array<size_t,N-1> out_idx{};
            for(size_t i=0,j=0;i<N;i++) if(i!=axis) out_idx[j++] = idx[i];

            size_t flat_out = 0;
            const auto& out_strides = out.get_strides();
            for(size_t d=0; d<N-1; ++d) flat_out += out_idx[d] * out_strides[d];

            out.get_data_ref()[flat_out] += data_[flat];
        }

        // --- Autograd ---
        if (requires_grad_) {
            out.set_requires_grad(true);
            auto A_ptr = this;  // capture 'this' tensor

            // Capture by value to extend lifetime
            out.set_backward_fn([A_ptr, out]() mutable {
                if (A_ptr->requires_grad()) {
                    // Broadcast N-1 grad to N shape
                    auto grad_broadcasted = out.template broadcast_to<N>(A_ptr->get_shape_ref());
                    A_ptr->add_grad(grad_broadcasted);
                }
            });
        }

        return out;
    }

    void add_grad(const Tensor<T,N>& other) {
        if (!requires_grad_) throw std::runtime_error("Tensor does not require grad");
        if (other.data_.size() != data_.size())
            throw std::runtime_error("Shape mismatch in add_grad");
        for (size_t i = 0; i < data_.size(); i++)
            grad_[i] += other.data_[i];
    }

    Tensor<T,N> sum_grad_axis(size_t axis) const {
        if (!requires_grad_) throw runtime_error("Tensor does not require grad");
        auto summed = sum_axis(axis);  // N-1 tensor
        auto grad_broadcasted = summed.broadcast_to(shape__);  // back to original shape
        return grad_broadcasted;
    }

    // Max in an axis
    Tensor<T, N-1> max_axis(size_t axis) const {
        if (axis >= N) throw runtime_error("Axis out of bounds");

        // Compute new shape for output tensor
        array<size_t, N-1> new_shape{};
        size_t j = 0;
        for (size_t i = 0; i < N; i++) {
            if (i != axis) new_shape[j++] = shape__[i];
        }

        Tensor<T, N-1> out(new_shape);

        // Initialize with the smallest possible value for type T
        fill(out.get_data_ref().begin(), out.get_data_ref().end(),
                numeric_limits<T>::lowest());

        array<size_t, N> idx{};
        array<size_t, N-1> out_idx{};

        // Loop through all elements in the original tensor
        for (size_t flat = 0; flat < data_.size(); flat++) {
            // Convert flat index -> N-dimensional index
            size_t rem = flat;
            for (int d = N - 1; d >= 0; d--) {
                idx[d] = rem % shape__[d];
                rem /= shape__[d];
            }

            // Build reduced index by skipping 'axis'
            size_t k = 0;
            for (size_t i = 0; i < N; i++) {
                if (i != axis) out_idx[k++] = idx[i];
            }

            // Compute flat index in reduced tensor
            size_t flat_out = 0;
            for (size_t d = 0; d < N - 1; d++) {
                flat_out += out_idx[d] * out.get_strides()[d];
            }

            // Update max
            T value = data_[flat];
            T &current_max = out.get_data_ref()[flat_out];
            if (value > current_max) {
                current_max = value;
            }
        }

        return out;
    }

    template<size_t M>
    vector<T> broadcast_from(
        const Tensor<T, M>& src,
        size_t reduced_axis) const
    {
        static_assert(M == N - 1, "Source tensor must have one less dimension");
        vector<T> result(data_.size()); // Same size as this tensor
        const auto& src_data = src.get_data_ref();
        const auto& src_strides = src.get_strides();

        for (size_t flat = 0; flat < result.size(); ++flat) {
            size_t rem = flat;
            array<size_t, N> idx{};
            for (int d = N - 1; d >= 0; --d) {
                idx[d] = rem % shape__[d];
                rem /= shape__[d];
            }

            size_t flat_src = 0;
            for (size_t d = 0, j = 0; d < N; ++d) {
                if (d == reduced_axis) continue;
                flat_src += idx[d] * src_strides[j++];
            }

            result[flat] = src_data[flat_src];
        }

        return result;
    }

    // Compute mean along a given axis, but broadcast to original shape
    Tensor<T, N> mean_axis(size_t axis) const {
        if (axis >= N) throw std::runtime_error("Axis out of bounds");

        std::array<size_t, N> out_shape = shape__;
        out_shape[axis] = 1;  // reduced axis has size 1
        Tensor<T, N> out(out_shape);
        out.zeros();

        for (size_t flat = 0; flat < data_.size(); ++flat) {
            // unravel flat index
            size_t rem = flat;
            std::array<size_t, N> idx{};
            for (int d = N - 1; d >= 0; --d) {
                idx[d] = rem % shape__[d];
                rem /= shape__[d];
            }

            // compute flat index in output
            size_t flat_out = 0;
            for (size_t d = 0; d < N; ++d) {
                size_t ix = (d == axis ? 0 : idx[d]);
                flat_out += ix * out.get_strides()[d];
            }

            out.get_data_ref()[flat_out] += data_[flat];
        }

        // divide by number of elements along axis
        T divisor = T(shape__[axis]);
        for (auto& x : out.get_data_ref()) x /= divisor;

        return out;
    }

    // Standard deviation along axis, result has size 1 along that axis
    Tensor<T, N> std_axis(size_t axis) const {
        Tensor<T, N> mean = mean_axis(axis);

        std::array<size_t, N> out_shape = shape__;
        out_shape[axis] = 1;
        Tensor<T, N> out(out_shape);
        out.zeros();

        for (size_t flat = 0; flat < data_.size(); ++flat) {
            size_t rem = flat;
            std::array<size_t, N> idx{};
            for (int d = N - 1; d >= 0; --d) {
                idx[d] = rem % shape__[d];
                rem /= shape__[d];
            }

            // index in output
            size_t flat_out = 0;
            for (size_t d = 0; d < N; ++d) {
                size_t ix = (d == axis ? 0 : idx[d]);
                flat_out += ix * out.get_strides()[d];
            }

            T diff = data_[flat] - mean.get_data_ref()[flat_out];
            out.get_data_ref()[flat_out] += diff * diff;
        }

        // divide by axis length and sqrt
        T divisor = T(shape__[axis]);
        for (auto& x : out.get_data_ref()) x = std::sqrt(x / divisor);

        return out;
    }

    // Squeeze / Unsqueeze
    Tensor<T,N-1> squeeze(size_t axis) const {
        if(axis >= N) throw runtime_error("Axis out of bounds");
        if(shape__[axis]!=1) throw runtime_error("Cannot squeeze non-1 dimension");

        array<size_t,N-1> new_shape{};
        for(size_t i=0,j=0;i<N;i++) if(i!=axis) new_shape[j++]=shape__[i];
        Tensor<T,N-1> out(new_shape);
        out.get_data_ref() = data_;
        return out;
    }

    Tensor<T, N+1> unsqueeze(size_t axis) const {
        static_assert(N >= 1, "Tensor rank must be >= 1 for unsqueeze");

        if(axis > N) throw runtime_error("Axis out of bounds");

        array<size_t, N+1> new_shape{};
        for(size_t i = 0, j = 0; i <= N; i++){
            if(i == axis) new_shape[i] = 1;
            else new_shape[i] = this->shape__[j++];
        }

        Tensor<T, N+1> out(new_shape);
        out.get_data_ref() = data_; // broadcasted flat copy works because 1-size dims
        return out;
    }


    // Transpose
    Tensor<T,N> transpose(const array<size_t,N>& axes) const {
        array<size_t,N> new_shape;
        for(size_t i=0;i<N;i++) new_shape[i]=shape__[axes[i]];
        Tensor<T,N> out(new_shape);

        array<size_t,N> idx_in, idx_out;
        for(size_t flat=0; flat<data_.size(); flat++){
            size_t rem = flat;
            for(int i=N-1;i>=0;i--){
                idx_in[i] = rem % shape__[i];
                rem /= shape__[i];
            }
            for(size_t i=0;i<N;i++) idx_out[i] = idx_in[axes[i]];
            out(idx_out) = data_[flat];
        }
        return out;
    }

    // Transpose only the last two dimensions
    Tensor<T, N> transpose_last_two() const {
        if constexpr (N < 2) throw std::runtime_error("Need at least 2 dimensions to transpose last two");
        
        array<size_t, N> axes{};
        for (size_t i = 0; i < N - 2; ++i) axes[i] = i; // keep first N-2 dims
        axes[N - 2] = N - 1;  // swap last two
        axes[N - 1] = N - 2;

        return this->transpose(axes);
    }

    // In Tensor<T,N>
    template<size_t OutRank>
    Tensor<T, OutRank> broadcast_to(const array<size_t, OutRank>& target_shape) const {
        static_assert(OutRank >= N, "Output rank must be >= tensor rank");

        // Build broadcast shape with prepended 1s for new dimensions
        array<size_t, OutRank> b_shape{};
        for (size_t i = 0; i < OutRank - N; ++i) b_shape[i] = 1;
        for (size_t i = 0; i < N; ++i) b_shape[OutRank - N + i] = shape__[i];

        Tensor<T, OutRank> out(target_shape);
        const auto& src_data = data_;
        const auto& src_strides = strides_;
        auto& out_data = out.get_data_ref();

        for (size_t flat = 0; flat < out_data.size(); ++flat) {
            size_t rem = flat;
            array<size_t, OutRank> idx{};
            for (int d = OutRank - 1; d >= 0; --d) {
                idx[d] = rem % target_shape[d];
                rem /= target_shape[d];
            }

            size_t flat_src = 0;
            for (size_t d = 0; d < N; ++d) {
                size_t src_idx = (shape__[d] == 1) ? 0 : idx[OutRank - N + d];
                flat_src += src_idx * src_strides[d];
            }

            out_data[flat] = src_data[flat_src];
        }

        return out;
    }

    size_t size() const { return data_.size(); }

    const T* data() const { return data_.data(); }
    T* data() { return data_.data(); }

    // --- Concatenate tensors along a given axis ---
    static Tensor<T, N> concat(const std::vector<Tensor<T, N>>& tensors, size_t axis) {
        if (tensors.empty()) throw std::runtime_error("No tensors to concat");

        auto shape0 = tensors[0].get_shape_ref();
        size_t concat_dim = 0;
        for (const auto& t : tensors) {
            auto s = t.get_shape_ref();
            for (size_t i = 0; i < N; i++) {
                if (i == axis) continue;
                if (s[i] != shape0[i])
                    throw std::runtime_error("Shapes must match except on concat axis");
            }
            concat_dim += s[axis];
        }

        Shape new_shape = shape0;
        new_shape[axis] = concat_dim;
        Tensor<T, N> result(new_shape);

        // Copy each tensor into the result
        size_t offset = 0;
        for (const auto& t : tensors) {
            result.copy_along_axis(t, axis, offset);
            offset += t.get_shape_ref()[axis];
        }

        return result;
    }

    // --- Copy source tensor into this tensor along a given axis with offset ---
    void copy_along_axis(const Tensor<T, N>& source, size_t axis, size_t offset) {
        std::array<size_t, N> idx_src{};
        std::array<size_t, N> idx_dest{};
        recursive_copy(source, idx_src, idx_dest, 0, axis, offset);
    }

    // --- Recursive helper for copy_along_axis ---
    void recursive_copy(const Tensor<T, N>& source,
                        std::array<size_t, N>& idx_src,
                        std::array<size_t, N>& idx_dest,
                        size_t dim,
                        size_t axis,
                        size_t offset)
    {
        if (dim == N) {
            size_t dest_flat = get_flat_index(idx_dest);
            size_t src_flat  = source.get_flat_index(idx_src);

            if (dest_flat >= data_.size())
                throw std::runtime_error("concat: destination flat index out of bounds");
            if (src_flat >= source.size())
                throw std::runtime_error("concat: source flat index out of bounds");

            data_[dest_flat] = source.get_data_ref()[src_flat];
            return;
        }

        for (size_t i = 0; i < source.get_shape_ref()[dim]; ++i) {
            idx_src[dim] = i;
            idx_dest[dim] = (dim == axis) ? i + offset : i;
            recursive_copy(source, idx_src, idx_dest, dim + 1, axis, offset);
        }
    }

    size_t get_flat_index(const std::array<size_t, N>& idx) const {
        size_t flat = 0;
        for (size_t i = 0; i < N; i++) {
            if (idx[i] >= shape__[i])
                throw std::runtime_error("Index out of bounds in get_flat_index");
            flat += idx[i] * strides_[i];
        }
        return flat;
    }

    // Ensure grad is allocated
    void zero_grad() { 
        if (requires_grad_) {
            if (grad_.size() != data_.size()) 
                grad_.resize(data_.size(), T(0));
            else
                std::fill(grad_.begin(), grad_.end(), T(0));
        }
    }

    // Non-const accessor for grad
    vector<T>& get_grad_ref() { 
        if (!requires_grad_) throw runtime_error("Tensor does not require grad");
        if (grad_.size() != data_.size()) grad_.resize(data_.size(), T(0));
        return grad_;
    }

    // Const accessor (read-only)
    const vector<T>& get_grad_ref() const { 
        return grad_;
    }

    // Utility: collect all tensors in topological order
    void build_topo(std::vector<Tensor<T, N>*>& topo,
                    std::unordered_set<Tensor<T, N>*>& visited) {
        if (visited.count(this)) return;
        visited.insert(this);
        for (auto* p : parents_) {
            if (p) p->build_topo(topo, visited);
        }
        topo.push_back(this);
    }

    // Proper backward pass with topological sorting
    void backward(T init_grad = T(1)) {
        if (!requires_grad_) return;

        // Collect nodes in topological order
        std::vector<Tensor<T, N>*> topo;
        std::unordered_set<Tensor<T, N>*> visited;
        build_topo(topo, visited);

        // Initialize all grads to zero
        for (auto* t : topo) {
            if (t->requires_grad_) {
                if (t->grad_.empty()) {
                    t->grad_.resize(t->data_.size(), T(0));
                } else {
                    std::fill(t->grad_.begin(), t->grad_.end(), T(0));
                }
            }
        }

        // Seed gradient at this output (the node on which backward() was called)
        // find `this` in topo and set its grad to init_grad (or for non-scalar,
        // distribute as ones; here we set all entries to init_grad)
        if (grad_.empty()) grad_.resize(data_.size(), T(0));
        for (auto &g : grad_) g = init_grad;

        // traverse in reverse topo order and call backward_fn_ with node ptr
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            Tensor<T, N>* t = *it;
            if (t->backward_fn_) {
                t->backward_fn_(t);
            }
        }
    }


    Tensor<T,N> sum_axis_keepdim(size_t axis) const {
        return this->sum_axis(axis).unsqueeze(axis);
    }

    array<size_t, N> unravel_index(size_t flat_index) const {
        array<size_t, N> idx{};
        size_t rem = flat_index;
        for (int i = N - 1; i >= 0; --i) {
            idx[i] = rem % shape__[i];
            rem /= shape__[i];
        }
        return idx;
    }

private:
    vector<T> data_;
    vector<T> grad_;
    Shape shape__;
    Shape strides_;
    bool requires_grad_;

    function<void(Tensor<T,N>*)> backward_fn_;

    void compute_strides() {
        size_t acc = 1;
        for(int i=N-1;i>=0;i--){
            strides_[i] = acc;
            acc *= shape__[i];
        }
    }

    void print_recursive(size_t dim, array<size_t,N>& idx) const {
        cout << "[";
        for(size_t i=0;i<shape__[dim];i++){
            idx[dim]=i;
            if(dim==N-1){
                size_t flat=0;
                for(size_t d=0;d<N;d++) flat+=idx[d]*strides_[d];
                cout << data_[flat];
            } else print_recursive(dim+1, idx);
            if(i!=shape__[dim]-1) cout << ", ";
        }
        cout << "]";
    }

    // Utility to multiply elements in a vector slice
    static size_t product(const vector<size_t>& v, size_t start, size_t end) {
        size_t p = 1;
        for (size_t i = start; i < end; i++) p *= v[i];
        return p;
    }

    size_t flat_index_broadcast(const array<size_t, N>& idx, const Tensor<T,N>& other) const {
        size_t flat = 0;
        const auto& s = other.get_shape_ref();
        const auto& str = other.get_strides_ref();
        for (size_t i = 0; i < N; ++i) {
            size_t dim_idx = (s[i] == 1) ? 0 : idx[i];
            flat += dim_idx * str[i];
        }
        return flat;
    }
};

// Global scalar operators
template<typename T, size_t N>
Tensor<T,N> operator+(T scalar, const Tensor<T,N>& tensor) {
    return tensor + scalar;
}

template<typename T, size_t N>
Tensor<T,N> operator-(T scalar, const Tensor<T,N>& tensor) {
    Tensor<T,N> result(tensor.get_shape());
    const auto& data = tensor.get_data_ref();
    auto& out = result.get_data_ref();
    for (size_t i = 0; i < data.size(); ++i)
        out[i] = scalar - data[i];
    return result;
}

template<typename T, size_t N>
Tensor<T,N> operator*(T scalar, const Tensor<T,N>& tensor) {
    return tensor * scalar;
}

template<typename T, size_t N>
Tensor<T,N> operator/(T scalar, const Tensor<T,N>& tensor) {
    Tensor<T,N> result(tensor.get_shape());
    const auto& data = tensor.get_data_ref();
    auto& out = result.get_data_ref();
    for (size_t i = 0; i < data.size(); ++i)
        out[i] = scalar / data[i];
    return result;
}

template<typename T, size_t NA, size_t NB>
Tensor<T, NA + NB - 2> dot(Tensor<T, NA>& A, Tensor<T, NB>& B) {
    // --- Compute output shape ---
    std::array<size_t, NA + NB - 2> out_shape;
    for (size_t i = 0; i < NA - 1; ++i) out_shape[i] = A.get_shape_ref()[i];
    for (size_t i = 1; i < NB; ++i) out_shape[NA - 1 + i - 1] = B.get_shape_ref()[i];

    Tensor<T, NA + NB - 2> out(out_shape, A.requires_grad() || B.requires_grad());
    out.zeros();

    size_t K = A.get_shape_ref()[NA - 1]; // inner dimension

    // --- Forward pass ---
    for (size_t flat_out = 0; flat_out < out.size(); ++flat_out) {
        auto out_idx = out.unravel_index(flat_out);
        T sum = 0;

        for (size_t k = 0; k < K; ++k) {
            std::array<size_t, NA> a_idx{};
            for (size_t i = 0; i < NA - 1; ++i) a_idx[i] = out_idx[i];
            a_idx[NA - 1] = k;

            std::array<size_t, NB> b_idx{};
            b_idx[0] = k;
            for (size_t i = 1; i < NB; ++i) b_idx[i] = out_idx[NA - 1 + i - 1];

            sum += A(a_idx) * B(b_idx);
        }
        out(out_idx) = sum;
    }

    // --- Backward pass ---
    if (out.requires_grad()) {
        out.set_backward_fn([&A, &B, &out, K](Tensor<T, NA + NB - 2>* self) {
            const auto& grad_out = self->get_grad_ref();

            for (size_t flat_out = 0; flat_out < self->size(); ++flat_out) {
                auto out_idx = self->unravel_index(flat_out);

                for (size_t k = 0; k < K; ++k) {
                    std::array<size_t, NA> a_idx{};
                    for (size_t i = 0; i < NA - 1; ++i) a_idx[i] = out_idx[i];
                    a_idx[NA - 1] = k;

                    std::array<size_t, NB> b_idx{};
                    b_idx[0] = k;
                    for (size_t i = 1; i < NB; ++i) b_idx[i] = out_idx[NA - 1 + i - 1];

                    A.get_grad_ref()[A.get_flat_index(a_idx)] += B(b_idx) * grad_out[flat_out];
                    B.get_grad_ref()[B.get_flat_index(b_idx)] += A(a_idx) * grad_out[flat_out];
                }
            }
        });
    }

    return out;
}
