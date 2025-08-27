#include <bits/stdc++.h>
using namespace std;

template<typename T, size_t N>
class Tensor {
public:
    using Shape = array<size_t, N>;

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
        if (expected_size != data_.size()) {
            throw runtime_error("Data size does not match shape_");
        }
    }

    // Print tensor
    void print() const {
        array<size_t, N> indices{};
        print_recursive(0, indices);
        cout << "\n";
    }

    // Fill with random values
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

    void ones() {
        for(auto& v : data_) v = T(1);
    }

    void zeros() {
        for(auto& v : data_) v = T(0);
    }

    void fill_value(T val = T(0)) {
        if(val == 0) { zeros(); return; }
        for(auto& v : data_) v = val;
    }

    // Access
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

    // Addition operator - inplace
    Tensor<T, N>& operator+=(const Tensor<T, N>& other) {
        if (shape__ != other.shape__)
            throw runtime_error("Dimension mismatch for addition!");

        for (size_t i = 0; i < data_.size(); ++i)
            data_[i] += other.data_[i]; 

        return *this;
    }

    // Addition operator 
    Tensor<T,N>& operator+(const Tensor<T,N>& other) {
        if (shape__ != other.shape__)
            throw runtime_error("Dimension mismatch for addition!");
        Tensor<T,N> result = *this;
        result += other;
        return result;
    }

    // Subtraction operator - inplace
    Tensor<T, N>& operator-=(const Tensor<T, N>& other) {
        if (shape__ != other.shape__)
            throw runtime_error("Dimension mismatch for addition!");

        for (size_t i = 0; i < data_.size(); ++i)
            data_[i] -= other.data_[i]; 

        return *this;
    }

    // Subtraction operator 
    Tensor<T,N>& operator-(const Tensor<T,N>& other) {
        if (shape__ != other.shape__)
            throw runtime_error("Dimension mismatch for addition!");
        Tensor<T,N> result = *this;
        result -= other;
        return result;
    }

    // Multiplication operator - inplace
    Tensor<T, N>& operator*=(const Tensor<T, N>& other) {
        if (shape__ != other.shape__)
            throw runtime_error("Dimension mismatch for addition!");

        for (size_t i = 0; i < data_.size(); ++i)
            data_[i] *= other.data_[i]; 

        return *this;
    }

    // Multiplication operator 
    Tensor<T,N>& operator*(const Tensor<T,N>& other) {
        if (shape__ != other.shape__)
            throw runtime_error("Dimension mismatch for addition!");
        Tensor<T,N> result = *this;
        result *= other;
        return result;
    }

    // Division operator - inplace
    Tensor<T, N>& operator/=(const Tensor<T, N>& other) {
        if (shape__ != other.shape__)
            throw runtime_error("Dimension mismatch for addition!");

        for (size_t i = 0; i < data_.size(); ++i)
            data_[i] /= other.data_[i]; 

        return *this;
    }

    // Division operator 
    Tensor<T,N>& operator/(const Tensor<T,N>& other) {
        if (shape__ != other.shape__)
            throw runtime_error("Dimension mismatch for addition!");
        Tensor<T,N> result = *this;
        result /= other;
        return result;
    }

    // Integer operators

    // Addition 
    Tensor<T,N> operator+(const T num) const {
        Tensor<T,N> result = *this;  
        for(size_t i = 0; i < result.data_.size(); i++) {
            result.data_[i] += num;
        }
        return result;
    }

    Tensor<T,N>& operator+=(const T num) {
        for(size_t i = 0; i < data_.size(); i++) {
            data_[i] += num;
        }
        return *this;
    }

    // Subtraction
    Tensor<T,N> operator-(const T num) const {
        Tensor<T,N> result = *this;  
        for(size_t i = 0; i < result.data_.size(); i++) {
            result.data_[i] -= num;
        }
        return result;
    }

    Tensor<T,N>& operator-=(const T num) {
        for(size_t i = 0; i < data_.size(); i++) {
            data_[i] -= num;
        }
        return *this;
    }

    // Multiplication
    Tensor<T,N> operator*(const T num) const {
        Tensor<T,N> result = *this;  
        for(size_t i = 0; i < result.data_.size(); i++) {
            result.data_[i] *= num;
        }
        return result;
    }

    Tensor<T,N>& operator*=(const T num) {
        for(size_t i = 0; i < data_.size(); i++) {
            data_[i] *= num;
        }
        return *this;
    }

    // Division
    Tensor<T,N> operator/(const T num) const {
        Tensor<T,N> result = *this;  
        for(size_t i = 0; i < result.data_.size(); i++) {
            result.data_[i] /= num;
        }
        return result;
    }

    Tensor<T,N>& operator/=(const T num) {
        for(size_t i = 0; i < data_.size(); i++) {
            data_[i] /= num;
        }
        return *this;
    }

    // Squeeze (remove size-1 axis)
    template<size_t Axis>
    auto squeeze() const {
        static_assert(Axis < N, "Axis out of bounds");
        if (shape__[Axis] != 1) throw runtime_error("Cannot squeeze non-1 dimension");

        array<size_t, N - 1> new_shape_{};
        for (size_t i = 0, j = 0; i < N; i++) {
            if (i != Axis) new_shape_[j++] = shape__[i];
        }
        Tensor<T, N - 1> out(new_shape_);
        out.data_ = data_;
        return out;
    }

    // Squeeze all size-1 axes
    auto squeeze_all() const {
        vector<size_t> new_shape__vec;
        for (auto s : shape__) if (s != 1) new_shape__vec.push_back(s);
        if (new_shape__vec.empty()) new_shape__vec.push_back(1);

        constexpr size_t M = 1; // default
        size_t M_actual = new_shape__vec.size();

        // Dynamic squeeze result
        if (M_actual == N) return *this;
        else if (M_actual == N - 1) {
            array<size_t, N - 1> new_shape_{};
            for (size_t i = 0; i < N - 1; i++) new_shape_[i] = new_shape__vec[i];
            Tensor<T, N - 1> out(new_shape_);
            out.data_ = data_;
            return out;
        } else {
            throw runtime_error("Full dynamic squeeze not implemented for >1 reduction");
        }
    }

    // Unsqueeze (add dimension)
    template<size_t Axis>
    auto unsqueeze() const {
        static_assert(Axis <= N, "Axis out of bounds");
        array<size_t, N + 1> new_shape_{};
        for (size_t i = 0, j = 0; i < N + 1; i++) {
            if (i == Axis) new_shape_[i] = 1;
            else new_shape_[i] = shape__[j++];
        }
        Tensor<T, N + 1> out(new_shape_);
        out.data_ = data_;
        return out;
    }

    // Sum
    auto sum() const {
        T Total = T(0);
        for(const auto &x : data_) Total+=x;
        return Total;
    }

    // Mean 
    auto mean() const {
        T sum = sum();
        return sum/T(data_.size());
    }

    // Max 
    auto max() const {
        return T(*max_element(data_.begin(),data_.end()));
    }

    // Min 
    auto min() const {
        return T(*min_element(data_.begin(),data_.end()));
    }

    // Sum along axis
    template<size_t Axis>
    auto sum_axis() const {
        static_assert(Axis < N, "Axis out of bounds");
        array<size_t, N - 1> new_shape_{};
        for (size_t i = 0, j = 0; i < N; i++) {
            if (i != Axis) new_shape_[j++] = shape__[i];
        }
        Tensor<T, N - 1> out(new_shape_);
        fill(out.data_.begin(), out.data_.end(), T(0));

        array<size_t, N> indices{};
        recurse_sum_axis<Axis, 0>(out, indices);
        return out;
    }

    // Shape of the tensor
    auto get_shape_() const {
        return shape__;
    }

    // Print tensor shape_
    void print_shape_() {
        cout << "[";
        for(size_t i = 0; i < N; i++) {
            cout << shape__[i];
            if(i != N-1) cout << ", ";
        }
        cout << ']';
    }

private:
    vector<T> data_;
    Shape shape__;
    Shape strides_;

    template<typename U, size_t M>
    friend class Tensor; // Allow cross-template access

    void compute_strides() {
        size_t acc = 1;
        for (int i = N - 1; i >= 0; i--) {
            strides_[i] = acc;
            acc *= shape__[i];
        }
    }

    void print_recursive(size_t dim, array<size_t, N>& indices) const {
        cout << "[";
        for (size_t i = 0; i < shape__[dim]; i++) {
            indices[dim] = i;
            if (dim == N - 1) {
                size_t flat = 0;
                for (size_t d = 0; d < N; d++) flat += indices[d] * strides_[d];
                cout << data_[flat];
            } else {
                print_recursive(dim + 1, indices);
            }
            if (i != shape__[dim] - 1) cout << ", ";
        }
        cout << "]";
    }

    template<size_t Axis, size_t D, typename OutTensor>
    void recurse_sum_axis(OutTensor& out, array<size_t, N>& idx) const {
        if constexpr (D == N) {
            array<size_t, N - 1> out_idx{};
            for (size_t i = 0, j = 0; i < N; i++) {
                if (i != Axis) out_idx[j++] = idx[i];
            }
            size_t flat_out = 0;
            for (size_t i = 0; i < N - 1; i++) flat_out += out_idx[i] * out.strides_[i];

            size_t flat_in = 0;
            for (size_t i = 0; i < N; i++) flat_in += idx[i] * strides_[i];
            out.data_[flat_out] += data_[flat_in];
        } else {
            for (size_t i = 0; i < shape__[D]; i++) {
                idx[D] = i;
                recurse_sum_axis<Axis, D + 1>(out, idx);
            }
        }
    }
};

template<typename T, size_t N>
Tensor<T,N> matmul(const Tensor<T,N>& A, const Tensor<T,N>& B){
    static_assert(N>=2,"Tensor must have at least 2 dims for matmul");

    size_t M = A.get_shape_()[N-2];
    size_t K = A.get_shape_()[N-1];
    size_t K2 = B.get_shape_()[N-2];
    size_t N2 = B.get_shape_()[N-1];
    if(K != K2) throw runtime_error("Inner dimensions must match");

    // Compute batch shape_
    array<size_t,N-2> batch_shape_;
    for(size_t i=0;i<N-2;++i){
        size_t a_dim = A.get_shape_()[i];
        size_t b_dim = B.get_shape_()[i];
        if(a_dim != b_dim && a_dim != 1 && b_dim != 1)
            throw runtime_error("Cannot broadcast batch dimensions");
        batch_shape_[i] = max(a_dim,b_dim);
    }

    // Result shape_
    array<size_t,N> result_shape_;
    for(size_t i=0;i<N-2;++i) result_shape_[i] = batch_shape_[i];
    result_shape_[N-2] = M;
    result_shape_[N-1] = N2;

    Tensor<T,N> result(result_shape_);

    // Total number of batches
    size_t total_batches = 1;
    for(auto b: batch_shape_) total_batches *= b;

    array<size_t,N> a_idx;
    array<size_t,N> b_idx;
    array<size_t,N> r_idx;

    // Loop over all batches
    for(size_t batch=0; batch<total_batches; ++batch){
        size_t tmp = batch;
        array<size_t,N-2> batch_index;
        for(int i=N-3;i>=0;--i){
            batch_index[i] = tmp % batch_shape_[i];
            tmp /= batch_shape_[i];
        }

        // Map batch indices to a_idx and b_idx
        for(size_t i=0;i<N-2;++i){
            a_idx[i] = (A.get_shape_()[i]==1)?0:batch_index[i];
            b_idx[i] = (B.get_shape_()[i]==1)?0:batch_index[i];
            r_idx[i] = batch_index[i];
        }

        // Standard 2D matmul on last two dims
        for(size_t i=0;i<M;++i){
            for(size_t j=0;j<N2;++j){
                T sum=0;
                for(size_t k=0;k<K;++k){
                    a_idx[N-2]=i; a_idx[N-1]=k;
                    b_idx[N-2]=k; b_idx[N-1]=j;
                    sum += A(a_idx)*B(b_idx);
                }
                r_idx[N-2]=i; r_idx[N-1]=j;
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

    // Unsqueeze
    auto B = A.unsqueeze<0>();
    cout << "Unsqueezed at axis 0: ";
    B.print();

    auto C = A.unsqueeze<2>();
    cout << "Unsqueezed at axis 2: ";
    C.print();

    // Squeeze (only works if shape_ has 1)
    Tensor<float, 3> D({2, 1, 3}, {1,2,3,4,5,6});
    cout << "Original D: ";
    D.print();
    auto D_squeezed = D.squeeze<1>();
    cout << "After squeeze axis 1: ";
    D_squeezed.print();

    // Sum along axis
    Tensor<float, 3> E({2, 2, 3});
    E.fill_random(1, 5);
    cout << "Original E: ";
    E.print();

    auto sum0 = E.sum_axis<0>();
    cout << "Sum along axis 0: ";
    sum0.print();

    auto sum1 = E.sum_axis<1>();
    cout << "Sum along axis 1: ";
    sum1.print();

    auto sum2 = E.sum_axis<2>();
    cout << "Sum along axis 2: ";
    sum2.print();

    Tensor<float,3> a({2,4,5});
    Tensor<float,3> b({2,5,6});

    a.fill_random();
    b.fill_random();

    auto mult = matmul(a,b);
    mult.print();
    mult.print_shape_();
    cout << "\nStats :"  << endl;
    cout << mult.max() << endl;

    cout << "\nOperations : " << endl;
    Tensor<float,3> test({2,3,4});
    Tensor<float,3> test1({2,3,4});
    test.fill_value(2);
    test1.fill_value(4);
    test+=test1;
    test.print();

    test+=3;
    test.print();

    return 0;
}