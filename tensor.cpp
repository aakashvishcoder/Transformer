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
    const Shape& get_strides() const { return strides_; }
    const vector<T>& get_data() const { return data_; }
    const Shape& get_shape() const { return shape__; };

    // Data References
    vector<T>& get_data_ref() { return data_; }
    Shape& get_shape_ref() { return shape__; }
    Shape& get_strides_ref() { return strides_; }

    // Print tensor
    void print() const {
        array<size_t, N> indices{};
        print_recursive(0, indices);
        cout << "\n";
    }

    // Fill tensor
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

    void fill_value(T val = T(0)) { for(auto& v: data_) v = val; }
    void ones() { fill_value(T(1)); }
    void zeros() { fill_value(T(0)); }

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
    Tensor<T,N>& operator+=(T num) { for(auto &v: data_) v += num; return *this; }
    Tensor<T,N> operator+(T num) const { Tensor<T,N> r = *this; r += num; return r; }

    Tensor<T,N>& operator-=(T num) { for(auto &v: data_) v -= num; return *this; }
    Tensor<T,N> operator-(T num) const { Tensor<T,N> r = *this; r -= num; return r; }

    Tensor<T,N>& operator*=(T num) { for(auto &v: data_) v *= num; return *this; }
    Tensor<T,N> operator*(T num) const { Tensor<T,N> r = *this; r *= num; return r; }

    Tensor<T,N>& operator/=(T num) { for(auto &v: data_) v /= num; return *this; }
    Tensor<T,N> operator/(T num) const { Tensor<T,N> r = *this; r /= num; return r; }

    // Sum, mean, min, max
    T sum() const { T total = T(0); for(auto &v: data_) total += v; return total; }
    T mean() const { return sum() / T(data_.size()); }
    T max() const { return *max_element(data_.begin(), data_.end()); }
    T min() const { return *min_element(data_.begin(), data_.end()); }

    // Sum along runtime axis
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
        for(auto &v : s.data_) v /= div;
        return s;
    }

    // Squeeze a runtime axis
    Tensor<T,N-1> squeeze(size_t axis) const {
        if(axis >= N) throw runtime_error("Axis out of bounds");
        if(shape__[axis]!=1) throw runtime_error("Cannot squeeze non-1 dimension");

        array<size_t,N-1> new_shape{};
        for(size_t i=0,j=0;i<N;i++) if(i!=axis) new_shape[j++]=shape__[i];
        Tensor<T,N-1> out(new_shape);
        out.get_data_ref() = data_;
        return out;
    }

    // Unsqueeze a runtime axis
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

    // Shape access
    void print_shape() const {
        cout << "[";
        for(size_t i=0;i<N;i++){
            cout << shape__[i];
            if(i!=N-1) cout << ", ";
        }
        cout << "]\n";
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

template<typename T, size_t N>
auto matmul(const Tensor<T,N>& A, const Tensor<T,N>& B) {
    static_assert(N >= 2, "Tensor must have at least 2 dims for matmul");

    // Last two dims are the matrix dims
    size_t M  = A.get_shape()[N-2];
    size_t K  = A.get_shape()[N-1];
    size_t K2 = B.get_shape()[N-2];
    size_t N2 = B.get_shape()[N-1];

    if (K != K2) throw runtime_error("Inner dimensions must match");

    // Compute broadcasted batch shape (all dims except last two)
    array<size_t, N-2> batch_shape;
    for (size_t i = 0; i < N-2; ++i) {
        size_t a_dim = A.get_shape()[i];
        size_t b_dim = B.get_shape()[i];
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1)
            throw runtime_error("Cannot broadcast batch dimensions");
        batch_shape[i] = max(a_dim, b_dim);
    }

    // Result shape = broadcasted batch dims + matrix dims
    array<size_t, N> result_shape;
    for (size_t i = 0; i < N-2; ++i) result_shape[i] = batch_shape[i];
    result_shape[N-2] = M;
    result_shape[N-1] = N2;

    Tensor<T,N> result(result_shape);

    // Total number of batches
    size_t total_batches = 1;
    for (auto b : batch_shape) total_batches *= b;

    array<size_t,N> a_idx;
    array<size_t,N> b_idx;
    array<size_t,N> r_idx;

    // Loop over all batches
    for (size_t batch = 0; batch < total_batches; ++batch) {
        size_t tmp = batch;
        array<size_t, N-2> batch_index;
        for (int i = N-3; i >= 0; --i) {
            batch_index[i] = tmp % batch_shape[i];
            tmp /= batch_shape[i];
        }

        // Map batch indices to A/B indices (broadcast if 1)
        for (size_t i = 0; i < N-2; ++i) {
            a_idx[i] = (A.get_shape()[i] == 1) ? 0 : batch_index[i];
            b_idx[i] = (B.get_shape()[i] == 1) ? 0 : batch_index[i];
            r_idx[i] = batch_index[i];
        }

        // Standard 2D matmul on last two dims
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N2; ++j) {
                T sum = 0;
                for (size_t k = 0; k < K; ++k) {
                    a_idx[N-2] = i; a_idx[N-1] = k;
                    b_idx[N-2] = k; b_idx[N-1] = j;
                    sum += A(a_idx) * B(b_idx);
                }
                r_idx[N-2] = i; r_idx[N-1] = j;
                result(r_idx) = sum;
            }
        }
    }

    return result;
}

int main() {
    Tensor<float, 2> A({2, 3});
    A.fill_random(1, 5);
    cout << "Original A: ";
    A.print();

    // Unsqueeze using runtime axis
    auto B = A.unsqueeze(0);
    cout << "Unsqueezed at axis 0: ";
    B.print();

    auto C = A.unsqueeze(2);
    cout << "Unsqueezed at axis 2: ";
    C.print();

    // Squeeze (only works if shape has 1)
    Tensor<float, 3> D({2, 1, 3}, {1,2,3,4,5,6});
    cout << "Original D: ";
    D.print();
    auto D_squeezed = D.squeeze(1);
    cout << "After squeeze axis 1: ";
    D_squeezed.print();

    // Sum along axis (runtime version)
    Tensor<float, 3> E({2, 2, 3});
    E.fill_random(1, 5);
    cout << "Original E: ";
    E.print();

    auto sum0 = E.sum_axis(0);
    cout << "Sum along axis 0: ";
    sum0.print();

    auto sum1 = E.sum_axis(1);
    cout << "Sum along axis 1: ";
    sum1.print();

    auto sum2 = E.sum_axis(2);
    cout << "Sum along axis 2: ";
    sum2.print();

    // Matmul test (still using 3D tensors)
    Tensor<float,3> a({2,4,5});
    Tensor<float,3> b({2,5,6});

    a.fill_random();
    b.fill_random();

    Tensor<float,3> A3({2, 2, 3}); // 2 batches of 2x3 matrices
    Tensor<float,3> B3({1, 3, 4}); // 1 batch of 3x4 matrices -> should broadcast to 2 batches

    A3.fill_random(1,2);
    B3.fill_random(1,2);

    cout << "\nMatmul broadcasting test:" << endl << flush;
    auto d = matmul(A3, B3);
    cout << "Shape: ";
    d.print_shape();  // Should print [2,2,4]
    d.print();

    cout << "\nOperations : " << endl;
    Tensor<float,3> test({2,3,4});
    Tensor<float,3> test1({2,3,4});
    test.fill_value(2);
    test1.fill_value(4);
    test += test1;
    test.print();

    test += 3;
    test.print();

    return 0;
}
