#include "tensor.hpp"
#include <iostream>
using namespace std;

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

    Tensor<float, 3> T({2, 3, 4});
    T.fill_random();
    array<size_t, 3> axes = {2, 0, 1}; 
    auto T2 = T.transpose(axes);
    cout << "\nTranspose : " << endl;
    T2.print_shape();
    T2.print();
    return 0;
}
