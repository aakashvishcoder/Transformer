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
