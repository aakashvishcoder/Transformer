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

int main() { 
    Tensor<float,4> t({2,3,4,5});
    t.fill_random(0.15f, 0.69f);
    t.print();
}
