template<typename T>
class Dense {
public:
    Tensor<T, 2> weight_;
    Tensor<T, 1> bias_;

    Dense(size_t in_features, size_t out_features)
        : weight_({in_features, out_features}), bias_({out_features})
    {
        weight_.fill_random(-0.1, 0.1);
        bias_.fill_value(T(0));
    }

    template<size_t N>
    Tensor<T, N> forward(const Tensor<T, N>& input) const {
        // input: rank N (e.g., 2 or 3)
        auto output = dot(input, weight_);  // output has same rank as input
        output += bias_.broadcast_to(output.get_shape()); // broadcast bias
        return output;  // rank is dynamic, matches dot result
    }
};

template<typename T, size_t N>
class LayerNormalization {
public:
    using Shape = std::array<size_t, N>;

    Tensor<T, N> gamma;
    Tensor<T, N> beta;
    T epsilon;

    LayerNormalization() : epsilon(1e-5) {} // default constructor

    // proper constructor with input shape
    LayerNormalization(size_t param_dim, const Shape& input_shape)
        : epsilon(1e-5), gamma(), beta()
    {
        Shape param_shape = input_shape;
        param_shape[N-1] = param_dim;  // last dimension = param_dim
        gamma = Tensor<T, N>(param_shape);
        gamma.fill_value(T(1));        // default ones
        beta = Tensor<T, N>(param_shape);
        beta.fill_value(T(0));         // default zeros
    }

    template<size_t NA>
    Tensor<T, NA> forward(const Tensor<T, NA>& input) {
        // mean and std along last axis
        auto mean = input.mean_axis(N-1);
        auto stddev = input.std_axis(N-1);

        auto normalized = (input - mean) / (stddev + epsilon);
        return normalized * gamma + beta;
    }
};

