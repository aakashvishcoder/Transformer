#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <stdexcept>
#include <algorithm>

using namespace std;

template<typename T, size_t N>
class Tensor {
public:
    using Shape = array<size_t, N>;

    // Constructors
    Tensor(const Shape& shape_)
        : shape__(shape_) {
        compute_strides();
        size_t total_size = 1;
        for (auto s : shape__) total_size *= s;
        data_.resize(total_size);
    }

    Tensor(const Shape& shape_, const vector<T>& data)
        : shape__(shape_), data_(data) {
        compute_strides();
        size_t expected_size = 1;
        for (auto s : shape__) expected_size *= s;
        if (expected_size != data_.size())
            throw runtime_error("Data size does not match shape");
    }

    // Getters
    const Shape& get_shape() const { return shape__; }
    const vector<T>& get_data() const { return data_; }
    const Shape& get_strides() const { return strides_; }

    vector<T>& get_data_ref() { return data_; }
    Shape& get_shape_ref() { return shape__; }
    Shape& get_strides_ref() { return strides_; }

    const vector<T>& get_data_ref() const { return data_; }
    const Shape& get_shape_ref() const { return shape__; }
    const Shape& get_strides_ref() const { return strides_; }

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
    auto sum() const {
        return accumulate(data_.begin(), data_.end(), 0);
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
        Tensor<T,N> result = *this;
        result += other;
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
        Tensor<T,N> result = *this;
        result -= other;
        return result;
    }

    Tensor<T,N>& operator*=(const Tensor<T,N>& other) {
        if (shape__ != other.shape__)
            throw runtime_error("Dimension mismatch for multiplication!");
        for (size_t i = 0; i < data_.size(); i++)
            data_[i] *= other.data_[i];
        return *this;
    }

    Tensor<T,N> operator*(const Tensor<T,N>& other) const {
        Tensor<T,N> result = *this;
        result *= other;
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
        Tensor<T,N> result = *this;
        result /= other;
        return result;
    }

    Tensor<T,N> operator-() const {
        Tensor<T,N> result(shape__);           
        const auto& in = data_;
        auto& out = result.get_data_ref();
        for (size_t i = 0; i < in.size(); i++)
            out[i] = -in[i];                  
        return result;
    }

    Tensor<T,N> sqrt() const {
        Tensor<T,N> result(shape__);
        const auto& in_data = get_data_ref();
        auto& out_data = result.get_data_ref();
        for(size_t i = 0; i < in_data.size(); i++) {
            out_data[i] = std::sqrt(in_data[i]);
        }
        return result;
    }

    Tensor<T,N> exp() const {
        Tensor<T,N> result(shape__);
        const auto& in_data = get_data_ref(); // calls const version
        auto& out_data = result.get_data_ref(); // non-const for writing
        for (size_t i = 0; i < in_data.size(); i++) {
            out_data[i] = std::exp(in_data[i]);
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

    // Arithmetic with scalar
    Tensor<T,N>& operator+=(T scalar) { 
        for(auto &v : data_) v += scalar; 
        return *this; 
    }

    Tensor<T,N>& operator-=(T scalar) { 
        for(auto &v : data_) v -= scalar; 
        return *this; 
    }

    Tensor<T,N>& operator*=(T scalar) { 
        for(auto &v : data_) v *= scalar; 
        return *this; 
    }

    Tensor<T,N>& operator/=(T scalar) { 
        for(auto &v : data_) v /= scalar; 
        return *this; 
    }

    Tensor<T,N>& operator^=(T scalar) {
        for (auto& v : data_) {
            v = std::pow(v, scalar);  // element-wise power
        }
        return *this;
    }

    Tensor<T,N> operator+(T scalar) const { Tensor<T,N> r = *this; r += scalar; return r; }
    Tensor<T,N> operator-(T scalar) const { Tensor<T,N> r = *this; r -= scalar; return r; }
    Tensor<T,N> operator*(T scalar) const { Tensor<T,N> r = *this; r *= scalar; return r; }
    Tensor<T,N> operator/(T scalar) const { Tensor<T,N> r = *this; r /= scalar; return r; }
    Tensor<T,N> operator^(T scalar) const { Tensor<T,N> r = *this; r ^= scalar; return r; }

    // Sum along axis
    Tensor<T,N-1> sum_axis(size_t axis) const {
        if(axis >= N) throw runtime_error("Axis out of bounds");
        array<size_t, N-1> new_shape{};
        for(size_t i=0,j=0;i<N;i++) if(i!=axis) new_shape[j++]=shape__[i];

        Tensor<T,N-1> out(new_shape);
        fill(out.get_data_ref().begin(), out.get_data_ref().end(), T(0));

        array<size_t,N> idx{};
        for(size_t flat=0; flat<data_.size(); ++flat){
            size_t rem = flat;
            for(int d=N-1; d>=0; --d){
                idx[d]=rem%shape__[d];
                rem/=shape__[d];
            }
            array<size_t,N-1> out_idx{};
            for(size_t i=0,j=0;i<N;i++) if(i!=axis) out_idx[j++]=idx[i];
            size_t flat_out=0;
            for(size_t d=0; d<N-1; ++d) flat_out+=out_idx[d]*out.get_strides()[d];
            out.get_data_ref()[flat_out] += data_[flat];
        }
        return out;
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

    Tensor<T,N-1> mean_axis(size_t axis) const {
        Tensor<T,N-1> s = sum_axis(axis);
        T div = T(shape__[axis]);
        for(auto &v : s.get_data_ref()) v /= div;
        return s;
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

    Tensor<T,N+1> unsqueeze(size_t axis) const {
        if(axis > N) throw runtime_error("Axis out of bounds");
        array<size_t,N+1> new_shape{};
        for(size_t i=0,j=0;i<N+1;i++){
            if(i==axis) new_shape[i]=1;
            else new_shape[i]=shape__[j++];
        }
        Tensor<T,N+1> out(new_shape);
        out.get_data_ref() = data_;
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

    Tensor<T, N> broadcast_to(const array<size_t, N>& target_shape) const {
        // Validate shape compatibility
        for (size_t i = 0; i < N; ++i) {
            if (shape__[i] != target_shape[i] && shape__[i] != 1) {
                throw runtime_error("Shapes are not broadcast compatible");
            }
        }

        Tensor<T, N> out(target_shape);
        auto& out_data = out.get_data_ref();

        const auto& src_data = data_;
        const auto& src_strides = get_strides();

        // Fill data by repeating values as per broadcasting
        for (size_t flat = 0; flat < out_data.size(); ++flat) {
            size_t rem = flat;
            array<size_t, N> idx{};
            for (int d = N - 1; d >= 0; --d) {
                idx[d] = rem % target_shape[d];
                rem /= target_shape[d];
            }

            // Map target index to source index (respect broadcasted dims)
            size_t flat_src = 0;
            for (size_t d = 0; d < N; ++d) {
                size_t src_idx = (shape__[d] == 1) ? 0 : idx[d];
                flat_src += src_idx * src_strides[d];
            }

            out_data[flat] = src_data[flat_src];
        }

        return out;
    }

private:
    vector<T> data_;
    Shape shape__;
    Shape strides_;

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
};

// Matmul
template<typename T, size_t N>
Tensor<T, N> matmul(const Tensor<T, N>& A, const Tensor<T, N>& B) {
    static_assert(N >= 2, "Tensor must have at least 2 dimensions");

    size_t M  = A.get_shape()[N - 2];
    size_t K  = A.get_shape()[N - 1];
    size_t K2 = B.get_shape()[N - 2];
    size_t N2 = B.get_shape()[N - 1];
    if (K != K2) throw runtime_error("Inner dimensions must match");

    // Compute batch shape
    array<size_t, N - 2> batch_shape;
    for (size_t i = 0; i < N - 2; i++) {
        size_t a_dim = A.get_shape()[i], b_dim = B.get_shape()[i];
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw runtime_error("Cannot broadcast batch dims");
        }
        batch_shape[i] = max(a_dim, b_dim);
    }

    // Full target shapes for A and B
    array<size_t, N> a_target_shape;
    array<size_t, N> b_target_shape;
    for (size_t i = 0; i < N - 2; i++) {
        a_target_shape[i] = batch_shape[i];
        b_target_shape[i] = batch_shape[i];
    }
    a_target_shape[N - 2] = M;  a_target_shape[N - 1] = K;
    b_target_shape[N - 2] = K;  b_target_shape[N - 1] = N2;

    // Broadcast A and B
    Tensor<T, N> A_bcast = A.broadcast_to(a_target_shape);
    Tensor<T, N> B_bcast = B.broadcast_to(b_target_shape);

    // Result shape
    array<size_t, N> result_shape;
    for (size_t i = 0; i < N - 2; i++) result_shape[i] = batch_shape[i];
    result_shape[N - 2] = M; result_shape[N - 1] = N2;
    Tensor<T, N> result(result_shape);

    // Compute sizes
    size_t total_batches = 1;
    for (auto b : batch_shape) total_batches *= b;

    size_t batch_size_A = M * K;
    size_t batch_size_B = K * N2;
    size_t batch_size_R = M * N2;

    auto& a_data = A_bcast.get_data_ref();
    auto& b_data = B_bcast.get_data_ref();
    auto& r_data = result.get_data_ref();

    for (size_t batch = 0; batch < total_batches; ++batch) {
        size_t a_offset = batch * batch_size_A;
        size_t b_offset = batch * batch_size_B;
        size_t r_offset = batch * batch_size_R;

        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N2; ++j) {
                T sum = 0;
                for (size_t k = 0; k < K; ++k) {
                    sum += a_data[a_offset + i * K + k] *
                           b_data[b_offset + k * N2 + j];
                }
                r_data[r_offset + i * N2 + j] = sum;
            }
        }
    }

    return result;
}

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
