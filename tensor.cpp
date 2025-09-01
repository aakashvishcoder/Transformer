#include "tensor.hpp"
#include "activations.hpp"
#include "loss_functions.hpp"
#include "dataframe.hpp"
#include "layers.hpp"
#include "architecture.hpp"
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
    auto d = dot(A3, B3);
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

    Tensor<float, 3> dan({2,3,4});
    dan.fill_random();
    auto sum = dan.sum_axis(2);
    sum.print();
    
    auto max = dan.max_axis(2);
    max.print();
    cout << "test2 : " << endl;
    Tensor<float, 3> test2({2,3,4});
    test2.fill_random(-1.0f, 1.0f);
    test2.print();
    cout << "\nActivations : " << endl;
    cout << "ReLU: " << endl;
    Activations::ReLU(test2);
    test2.print();
    
    cout << "\ntest2 : " << endl;
    test2.fill_random(-1.0f, 1.0f);
    test2.print();
    cout << "Leaky ReLU: " << endl;
    Activations::LeakyReLU(test2,1e-2f);
    test2.print();

    cout << "\ntest2 : " << endl;
    test2.fill_random(-1.0f, 1.0f);
    test2.print();
    cout << "Leaky ReLU: " << endl;
    Activations::LeakyReLU(test2,1e-2f);
    test2.print();

    cout << "\ntest2 : " << endl;
    test2.fill_random(-1.0f, 1.0f);
    test2.print();
    cout << "Softmax: " << endl;
    auto t3 = Activations::Softmax(test2,2);
    t3.print();

    cout << "\ntest2 : " << endl;
    test2.fill_random(-1.0f, 1.0f);
    test2.print();
    cout << "Sigmoid: " << endl;
    t3 = Activations::Sigmoid(test2);
    t3.print();

    cout << "\ntest2 : " << endl;
    test2.fill_random(-1.0f, 1.0f);
    test2.print();
    cout << "TanH: " << endl;
    t3 = Activations::Tanh(test2);
    t3.print();
    Tensor<float,3> t7({2,3,4});
    t7.fill_random(1,100);
    t7.print();

    t7=-t7;
    t7.exp_();
    t7.print();

    Tensor<float,3> dummy({2,3,4});
    Tensor<float,3> dummy1({2,3,4});
    dummy.fill_random(-1.0f, 1.0f);
    dummy1.fill_random(-1.0f, 1.0f);
    cout <<"\nDummy :"<<endl;
    dummy.print();
    cout <<"\nDummy1 :"<<endl;
    dummy1.print();
    float loss = LossFunctions::MSE(dummy,dummy1);
    cout << "\nMSE : " << loss << endl;

    loss = LossFunctions::RMSE(dummy,dummy1);
    cout << "RMSE : " << loss << endl;

    dummy.fill_random(0.0f, 1.0f);
    dummy1.fill_random(0.0f, 1.0f);
    loss = LossFunctions::BCE(dummy,dummy1);
    cout << "BCE : " << loss << endl;

    dummy.fill_random(0,3);
    dummy1.fill_random(0,3);
    loss = LossFunctions::CE(dummy,dummy1);
    cout << "CE : " << loss << endl;

    Tensor<float, 3> input({2, 3, 4}); // batch=2, seq=3, features=4
    input.fill_random();

    Dense<float> layer(4, 5);     // input features=4, output features=5
    auto output = layer.forward(input);
    cout << endl;
    output.print();              // should print: [2,3,5]

    // output shape = {2, 3, 5}

    Tensor<float, 2> l({3, 4});   // (3,4)
    Tensor<float, 2> m({4, 5});   // (4,5)
    l.fill_random(-1.0f,1.0f);
    m.fill_random(-1.0f,1.0f);
    auto p = dot(l, m);           // (3,5)
    cout << endl;
    p.print();
    Tensor<float, 3> X({2, 3, 4}); // (2,3,4)
    Tensor<float, 2> W({4, 6});    // (4,6)
    X.fill_random(-1.0f,1.0f);
    W.fill_random(-1.0f,1.0f);
    auto Y = dot(X, W);            // (2,3,6)
    cout << endl;
    Y.print_shape();

    auto Z = Y.std_axis(2);
    cout << endl;
    Z.print();

    Z = Y.mean_axis(2);
    cout << endl;
    Z.print();

    Tensor<float,3> TST({2,3,4});
    TST.fill_random(-1.0f,1.0f);
    cout << endl;
    TST.print();
    cout<<endl;
    LayerNormalization<float,3> ln(TST.get_shape_ref()[2], TST.get_shape_ref());
    auto out = ln.forward(TST);
    out.print();

    // Parameters
    size_t batch_size = 2;
    size_t seq_len = 3;
    size_t embed_dim = 4;
    size_t num_heads = 2;
    size_t ffn_hidden_dim = 8;
    size_t num_layers = 2;

    // Create input tensor [batch_size, seq_len, embed_dim]
    Tensor<float, 3> input1({batch_size, seq_len, embed_dim});
    input1.fill_random(-0.1f, 0.1f);

    // Print input
    std::cout << "--- Input Tensor ---\n";
    input1.print();

    // Create Transformer Encoder
    TransformerEncoder<float, 3> encoder(num_layers, embed_dim, num_heads, ffn_hidden_dim);

    // Forward pass
    Tensor<float, 3> output1 = encoder.forward(input);

    // Print output
    std::cout << "\n--- Transformer Encoder Output ---\n";
    output1.print();

    Tensor<float, 3> input2({batch_size, seq_len, embed_dim});
    input2.fill_random(-0.1f, 0.1f);

    cout << "\n--- Input Tensor ---\n";
    input2.print();

    TransformerDecoder<float,3> decoder(num_layers, embed_dim, num_heads, ffn_hidden_dim);
    Tensor<float, 3> output2 = decoder.forward(output1, input, embed_dim);

    cout << "\n--- Transformer Decoder Output ---\n";
    output2.print();

    return 0;
}
