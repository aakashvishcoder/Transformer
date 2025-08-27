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

    // Fill
    void fill_value(T val = T(0)) { for (auto& v : data_) v = val; }
    void ones() { fill_value(T(1)); }
    void zeros() { fill_value(T(0)); }

    void fill_random(T min_val = T(0), T max_val = T(1)) {
        random_device rd;
        mt19937 gen(rd());
        if constexpr (is_integral<T>::value) {
            uniform_int_distribution<T> dist(min_val, max_val);
            for (auto& v : data_) v = dist(gen);
        } else {
            uniform_real_distribution<T> dist(min_val, max_val);
            for (auto& v : data_) v = dist(gen);
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

    // Arithmetic with scalar
    Tensor<T,N>& operator+=(T num) { for (auto &v : data_) v += num; return *this; }
    Tensor<T,N>& operator-=(T num) { for (auto &v : data_) v -= num; return *this; }
    Tensor<T,N>& operator*=(T num) { for (auto &v : data_) v *= num; return *this; }
    Tensor<T,N>& operator/=(T num) { for (auto &v : data_) v /= num; return *this; }

    Tensor<T,N> operator+(T num) const { Tensor<T,N> r = *this; r += num; return r; }
    Tensor<T,N> operator-(T num) const { Tensor<T,N> r = *this; r -= num; return r; }
    Tensor<T,N> operator*(T num) const { Tensor<T,N> r = *this; r *= num; return r; }
    Tensor<T,N> operator/(T num) const { Tensor<T,N> r = *this; r /= num; return r; }

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
Tensor<T,N> matmul(const Tensor<T,N>& A, const Tensor<T,N>& B) {
    static_assert(N >= 2, "Tensor must have at least 2 dims for matmul");
    size_t M  = A.get_shape()[N-2];
    size_t K  = A.get_shape()[N-1];
    size_t K2 = B.get_shape()[N-2];
    size_t N2 = B.get_shape()[N-1];
    if(K!=K2) throw runtime_error("Inner dimensions must match");

    array<size_t,N-2> batch_shape;
    for(size_t i=0;i<N-2;i++){
        size_t a_dim=A.get_shape()[i], b_dim=B.get_shape()[i];
        if(a_dim!=b_dim && a_dim!=1 && b_dim!=1) throw runtime_error("Cannot broadcast batch dims");
        batch_shape[i]=max(a_dim,b_dim);
    }

    array<size_t,N> result_shape;
    for(size_t i=0;i<N-2;i++) result_shape[i]=batch_shape[i];
    result_shape[N-2]=M; result_shape[N-1]=N2;
    Tensor<T,N> result(result_shape);

    size_t total_batches = 1;
    for(auto b: batch_shape) total_batches *= b;

    array<size_t,N> a_idx,b_idx,r_idx;
    for(size_t batch=0; batch<total_batches; batch++){
        size_t tmp=batch;
        array<size_t,N-2> batch_index;
        for(int i=N-3;i>=0;i--){ batch_index[i]=tmp%batch_shape[i]; tmp/=batch_shape[i]; }

        for(size_t i=0;i<N-2;i++){
            a_idx[i]=(A.get_shape()[i]==1)?0:batch_index[i];
            b_idx[i]=(B.get_shape()[i]==1)?0:batch_index[i];
            r_idx[i]=batch_index[i];
        }

        for(size_t i=0;i<M;i++){
            for(size_t j=0;j<N2;j++){
                T sum=0;
                for(size_t k=0;k<K;k++){
                    a_idx[N-2]=i; a_idx[N-1]=k;
                    b_idx[N-2]=k; b_idx[N-1]=j;
                    sum+=A(a_idx)*B(b_idx);
                }
                r_idx[N-2]=i; r_idx[N-1]=j;
                result(r_idx)=sum;
            }
        }
    }
    return result;
}
