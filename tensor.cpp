#include <vector>
#include <array>
#include <iostream>
#include <numeric>

template <typename T, size_t N>
class Tensor {
public:
    Tensor(const std::array<size_t, N>& shape) 
        : shape(shape) 
    {
        // compute strides
        strides[N-1] = 1;
        for (int i = N-2; i >= 0; --i) {
            strides[i] = strides[i+1] * shape[i+1];
        }

        // allocate storage
        size_t total_size = 1;
        for (auto s : shape) total_size *= s;
        data.resize(total_size);
    }

    template <typename... Args>
    T& operator()(Args... args) {
        static_assert(sizeof...(args) == N, "Invalid number of indices");
        std::array<size_t, N> indices{static_cast<size_t>(args)...};
        size_t idx = 0;
        for (size_t i = 0; i < N; ++i) {
            idx += indices[i] * strides[i];
        }
        return data[idx];
    }

    const std::array<size_t, N>& get_shape() const { return shape; }

private:
    std::array<size_t, N> shape;
    std::array<size_t, N> strides;
    std::vector<T> data;
};

// Example usage
int main() {
    Tensor<int, 3> tensor({2, 3, 4});  // 3D tensor of ints

    tensor(1, 2, 3) = 42;  // assign value

    std::cout << "Value at (1,2,3): " << tensor(1, 2, 3) << "\n";

    Tensor<double, 2> tensor2({3, 3}); // 2D tensor of doubles
    tensor2(1,1) = 3.14;
    std::cout << "Value at (1,1): " << tensor2(1,1) << "\n";
}
