#include <vector>
#include <array>
#include <iostream>
#include <numeric>
#include <random>
using namespace std;

template <typename T, size_t N>
class Tensor {
public:
    Tensor(const array<size_t, N>& shape) 
        : shape(shape) 
    {
        strides[N-1] = 1;
        for (int i = N-2; i >= 0; --i) {
            strides[i] = strides[i+1] * shape[i+1];
        }

        size_t total_size = 1;
        for (auto s : shape) total_size *= s;
        data.resize(total_size);
    }

    template <typename... Args>
    T& operator()(Args... args) {
        static_assert(sizeof...(args) == N, "Invalid number of indices");
        array<size_t, N> indices{static_cast<size_t>(args)...};
        size_t idx = 0;
        for (size_t i = 0; i < N; ++i) {
            idx += indices[i] * strides[i];
        }
        return data[idx];
    }

    Tensor<T,N> operator+(const Tensor<T,N>& other) const {
        check_same_shape(other);
        Tensor<T,N> result(shape);
        for(size_t i = 0; i < data.size(); i++)
            result.data[i] = data[i] + other.data[i];
        return result;
    }

    Tensor<T,N> operator-(const Tensor<T,N>& other) const {
        check_same_shape(other);
        Tensor<T,N> result(shape);
        for(size_t i = 0; i < data.size(); i++) 
            result.data[i] = data[i] - other.data[i];
        return result;
    }

    Tensor<T,N> operator*(const Tensor<T,N>& other) const {
        check_same_shape(other);
        Tensor<T,N> result(shape);
        for(size_t i = 0; i < data.size(); i++) 
            result.data[i] = data[i] * other.data[i];
        return result;
    }

    Tensor<T,N> operator/(const Tensor<T,N>& other) const {
        check_same_shape(other);
        Tensor<T,N> result(shape);
        for(size_t i = 0; i < data.size(); i++) 
            result.data[i] = data[i] / other.data[i];
        return result;
    }

    void print_flat() const {
        for (auto x : data) cout << x << " ";
        cout << "\n";
    }

    const array<size_t, N>& get_shape() const { return shape; }

    void print_shape() {
        cout << "Shape: [";
        bool first = true;
        for (auto s : get_shape()) {
            if (!first) std::cout << ", ";
            std::cout << s;
            first = false;
        }
        cout << "]\n";
    }

    void print() const {
        array<size_t, N> indices{};
        print_recursive(0, indices);
        cout << "\n";
    }

    void fill_random(T min = T(0), T max = T(1)) {
        random_device rd;
        mt19937 gen(rd());

        if constexpr (is_integral<T>::value) {
            uniform_int_distribution<T> dist(min,max);
            for (auto&x : data) x = dist(gen);
        } else {
            uniform_real_distribution<T> dist(min,max);
            for(auto& x: data) x = dist(gen);
        }
    }
private:

    void check_same_shape(const Tensor<T,N>& other) const {
        if (shape != other.shape)
            throw runtime_error("Shapes do not match for element-wise operation");
    }

    private:
    void print_recursive(size_t dim, array<size_t, N>& indices) const {
        cout << "[";
        for (size_t i = 0; i < shape[dim]; ++i) {
            indices[dim] = i;
            if (dim == N - 1) {
                // Last dimension: print actual value
                size_t flat_index = 0;
                for (size_t d = 0; d < N; ++d)
                    flat_index += indices[d] * strides[d];
                cout << data[flat_index];
            } else {
                // Recurse into next dimension
                print_recursive(dim + 1, indices);
            }
            if (i != shape[dim] - 1) cout << ", ";
        }
        cout << "]";
    }

    array<size_t, N> shape;
    array<size_t, N> strides;
    vector<T> data;
};

template<typename T, size_t N>
Tensor<T,N> matmul(const Tensor<T,N>& A, const Tensor<T,N>& B){
    static_assert(N>=2,"Tensor must have at least 2 dims for matmul");

    size_t M = A.get_shape()[N-2];
    size_t K = A.get_shape()[N-1];
    size_t K2 = B.get_shape()[N-2];
    size_t N2 = B.get_shape()[N-1];
    if(K != K2) throw runtime_error("Inner dimensions must match");

    // Compute batch shape
    array<size_t,N-2> batch_shape;
    for(size_t i=0;i<N-2;++i){
        size_t a_dim = A.get_shape()[i];
        size_t b_dim = B.get_shape()[i];
        if(a_dim != b_dim && a_dim != 1 && b_dim != 1)
            throw runtime_error("Cannot broadcast batch dimensions");
        batch_shape[i] = max(a_dim,b_dim);
    }

    // Result shape
    array<size_t,N> result_shape;
    for(size_t i=0;i<N-2;++i) result_shape[i] = batch_shape[i];
    result_shape[N-2] = M;
    result_shape[N-1] = N2;

    Tensor<T,N> result(result_shape);

    // Total number of batches
    size_t total_batches = 1;
    for(auto b: batch_shape) total_batches *= b;

    array<size_t,N> a_idx;
    array<size_t,N> b_idx;
    array<size_t,N> r_idx;

    // Loop over all batches
    for(size_t batch=0; batch<total_batches; ++batch){
        size_t tmp = batch;
        array<size_t,N-2> batch_index;
        for(int i=N-3;i>=0;--i){
            batch_index[i] = tmp % batch_shape[i];
            tmp /= batch_shape[i];
        }

        // Map batch indices to a_idx and b_idx
        for(size_t i=0;i<N-2;++i){
            a_idx[i] = (A.get_shape()[i]==1)?0:batch_index[i];
            b_idx[i] = (B.get_shape()[i]==1)?0:batch_index[i];
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

// ----------------- Test -----------------
int main(){
    Tensor<float,4> A({2,3,4,5});
    Tensor<float,4> B({2,3,5,6});

    A.fill_random(0.1f,1.0f);
    B.fill_random(0.1f,1.0f);

    auto C = matmul(A,B);

    A.print_shape();
    B.print_shape();
    C.print_shape();

    cout<<"\nC[0,0,:,:] =\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<6;j++){
            array<size_t,4> idx = {0,0,(size_t)i,(size_t)j};
            cout<<C(idx)<<" ";
        }
        cout<<"\n";
    }
}
