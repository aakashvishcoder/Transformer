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
    Tensor() = default;

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

    const vector<T>& get_data_ref() const { return data_; }
    const Shape& get_shape_ref() const { return shape__; }
    const Shape& get_strides_ref() const { return strides_; }
    
    // Setter

    void initialize(const Shape& shape_) {
        shape__ = shape_;
        compute_strides();
        size_t total_size = 1;
        for (auto s : shape__) total_size *= s;
        data_.resize(total_size);
    }

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

    // Sum
    T sum() const {
        T total = 0;
        for (const auto& val : data_) {
            total += val;
        }
        return total;
    } 

    // Fill
    void fill_value(T val = T(0)) { for (auto& v : data_) v = val; }
    void ones() { fill_value(T(1)); }
    void zeros() { fill_value(T(0)); }

    void fill_random(T min_val = T(0), T max_val = T(1)) {
        random_device rd;
        mt19937 gen(rd());

        if constexpr (is_floating_point_v<T>) {
            // Floating point: uniform_real_distribution
            uniform_real_distribution<T> dist(min_val, max_val);
            for (auto& x : data_) x = dist(gen);
        } else if constexpr (is_integral_v<T>) {
            // Integer: use signed type to handle negative ranges
            using signed_type = make_signed_t<T>;
            uniform_int_distribution<signed_type> dist(
                static_cast<signed_type>(min_val),
                static_cast<signed_type>(max_val)
            );
            for (auto& x : data_) x = dist(gen);
        } else {
            static_assert(is_arithmetic_v<T>, "fill_random requires numeric type");
        }
    }

    void clip(T min_val = T(1e-10), T max_val = T(1-1e-10)) {
        for(auto& v : data_) {
            if (v < min_val) v = min_val;
            else if (v > max_val) v = max_val;
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
        for(size_t flat = 0; flat < data_.size(); ++flat) {
            // convert flat to multi-index
            array<size_t,N> idx{};
            size_t rem = flat;
            for(int d=N-1; d>=0; --d){
                idx[d] = rem % shape__[d];
                rem /= shape__[d];
            }
            size_t other_flat = other.flat_index_broadcast(idx, other);
            data_[flat] *= other.get_data_ref()[other_flat];
        }
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

    Tensor<T,N> operator-() const {
        Tensor<T,N> result(shape__);           
        const auto& in = data_;
        auto& out = result.get_data_ref();
        for (size_t i = 0; i < in.size(); i++)
            out[i] = -in[i];                  
        return result;
    }

    Tensor<T,N> sqrt() const {
        Tensor<T,N> result(shape__);
        const auto& in_data = get_data_ref();
        auto& out_data = result.get_data_ref();
        for(size_t i = 0; i < in_data.size(); i++) {
            out_data[i] = std::sqrt(in_data[i]);
        }
        return result;
    }

    Tensor<T,N> exp() const {
        Tensor<T,N> result(shape__);
        const auto& in_data = get_data_ref(); // calls const version
        auto& out_data = result.get_data_ref(); // non-const for writing
        for (size_t i = 0; i < in_data.size(); i++) {
            out_data[i] = std::exp(in_data[i]);
        }
        return result;
    }

    Tensor<T,N> log() const {
        Tensor<T,N> result(shape__);
        const auto& in_data = get_data_ref(); // calls const version
        auto& out_data = result.get_data_ref(); // non-const for writing
        for (size_t i = 0; i < in_data.size(); i++) {
            out_data[i] = std::log(in_data[i]);
        }
        return result;
    }

    Tensor<T,N> exp_() {
        for(auto &v : data_) v = std::exp(v);
        return *this;
    }

    Tensor<T,N> sqrt_() {
        for(auto &v : data_) v = std::sqrt(v);
        return *this;
    }

    Tensor<T,N> log_() {
        for(auto& v: data_) v = std::log(v);
        return *this;
    }

    // Arithmetic with scalar
    Tensor<T,N>& operator+=(T scalar) { 
        for(auto &v : data_) v += scalar; 
        return *this; 
    }

    Tensor<T,N>& operator-=(T scalar) { 
        for(auto &v : data_) v -= scalar; 
        return *this; 
    }

    Tensor<T,N>& operator*=(T scalar) { 
        for(auto &v : data_) v *= scalar; 
        return *this; 
    }

    Tensor<T,N>& operator/=(T scalar) { 
        for(auto &v : data_) v /= scalar; 
        return *this; 
    }

    Tensor<T,N>& operator^=(T scalar) {
        for (auto& v : data_) {
            v = std::pow(v, scalar);  // element-wise power
        }
        return *this;
    }

    Tensor<T,N> operator+(T scalar) const { Tensor<T,N> r = *this; r += scalar; return r; }
    Tensor<T,N> operator-(T scalar) const { Tensor<T,N> r = *this; r -= scalar; return r; }
    Tensor<T,N> operator*(T scalar) const { Tensor<T,N> r = *this; r *= scalar; return r; }
    Tensor<T,N> operator/(T scalar) const { Tensor<T,N> r = *this; r /= scalar; return r; }
    Tensor<T,N> operator^(T scalar) const { Tensor<T,N> r = *this; r ^= scalar; return r; }

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

    // Max in an axis
    Tensor<T, N-1> max_axis(size_t axis) const {
        if (axis >= N) throw runtime_error("Axis out of bounds");

        // Compute new shape for output tensor
        array<size_t, N-1> new_shape{};
        size_t j = 0;
        for (size_t i = 0; i < N; i++) {
            if (i != axis) new_shape[j++] = shape__[i];
        }

        Tensor<T, N-1> out(new_shape);

        // Initialize with the smallest possible value for type T
        fill(out.get_data_ref().begin(), out.get_data_ref().end(),
                numeric_limits<T>::lowest());

        array<size_t, N> idx{};
        array<size_t, N-1> out_idx{};

        // Loop through all elements in the original tensor
        for (size_t flat = 0; flat < data_.size(); flat++) {
            // Convert flat index -> N-dimensional index
            size_t rem = flat;
            for (int d = N - 1; d >= 0; d--) {
                idx[d] = rem % shape__[d];
                rem /= shape__[d];
            }

            // Build reduced index by skipping 'axis'
            size_t k = 0;
            for (size_t i = 0; i < N; i++) {
                if (i != axis) out_idx[k++] = idx[i];
            }

            // Compute flat index in reduced tensor
            size_t flat_out = 0;
            for (size_t d = 0; d < N - 1; d++) {
                flat_out += out_idx[d] * out.get_strides()[d];
            }

            // Update max
            T value = data_[flat];
            T &current_max = out.get_data_ref()[flat_out];
            if (value > current_max) {
                current_max = value;
            }
        }

        return out;
    }

    template<size_t M>
    vector<T> broadcast_from(
        const Tensor<T, M>& src,
        size_t reduced_axis) const
    {
        static_assert(M == N - 1, "Source tensor must have one less dimension");
        vector<T> result(data_.size()); // Same size as this tensor
        const auto& src_data = src.get_data_ref();
        const auto& src_strides = src.get_strides();

        for (size_t flat = 0; flat < result.size(); ++flat) {
            size_t rem = flat;
            array<size_t, N> idx{};
            for (int d = N - 1; d >= 0; --d) {
                idx[d] = rem % shape__[d];
                rem /= shape__[d];
            }

            size_t flat_src = 0;
            for (size_t d = 0, j = 0; d < N; ++d) {
                if (d == reduced_axis) continue;
                flat_src += idx[d] * src_strides[j++];
            }

            result[flat] = src_data[flat_src];
        }

        return result;
    }

    // Compute mean along a given axis, but broadcast to original shape
    Tensor<T, N> mean_axis(size_t axis) const {
        if (axis >= N) throw std::runtime_error("Axis out of bounds");

        std::array<size_t, N> out_shape = shape__;
        out_shape[axis] = 1;  // reduced axis has size 1
        Tensor<T, N> out(out_shape);
        out.zeros();

        for (size_t flat = 0; flat < data_.size(); ++flat) {
            // unravel flat index
            size_t rem = flat;
            std::array<size_t, N> idx{};
            for (int d = N - 1; d >= 0; --d) {
                idx[d] = rem % shape__[d];
                rem /= shape__[d];
            }

            // compute flat index in output
            size_t flat_out = 0;
            for (size_t d = 0; d < N; ++d) {
                size_t ix = (d == axis ? 0 : idx[d]);
                flat_out += ix * out.get_strides()[d];
            }

            out.get_data_ref()[flat_out] += data_[flat];
        }

        // divide by number of elements along axis
        T divisor = T(shape__[axis]);
        for (auto& x : out.get_data_ref()) x /= divisor;

        return out;
    }

    // Standard deviation along axis, result has size 1 along that axis
    Tensor<T, N> std_axis(size_t axis) const {
        Tensor<T, N> mean = mean_axis(axis);

        std::array<size_t, N> out_shape = shape__;
        out_shape[axis] = 1;
        Tensor<T, N> out(out_shape);
        out.zeros();

        for (size_t flat = 0; flat < data_.size(); ++flat) {
            size_t rem = flat;
            std::array<size_t, N> idx{};
            for (int d = N - 1; d >= 0; --d) {
                idx[d] = rem % shape__[d];
                rem /= shape__[d];
            }

            // index in output
            size_t flat_out = 0;
            for (size_t d = 0; d < N; ++d) {
                size_t ix = (d == axis ? 0 : idx[d]);
                flat_out += ix * out.get_strides()[d];
            }

            T diff = data_[flat] - mean.get_data_ref()[flat_out];
            out.get_data_ref()[flat_out] += diff * diff;
        }

        // divide by axis length and sqrt
        T divisor = T(shape__[axis]);
        for (auto& x : out.get_data_ref()) x = std::sqrt(x / divisor);

        return out;
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

    // In Tensor<T,N>
    template<size_t OutRank>
    Tensor<T, OutRank> broadcast_to(const array<size_t, OutRank>& target_shape) const {
        static_assert(OutRank >= N, "Output rank must be >= tensor rank");

        // Build broadcast shape with prepended 1s for new dimensions
        array<size_t, OutRank> b_shape{};
        for (size_t i = 0; i < OutRank - N; ++i) b_shape[i] = 1;
        for (size_t i = 0; i < N; ++i) b_shape[OutRank - N + i] = shape__[i];

        Tensor<T, OutRank> out(target_shape);
        const auto& src_data = data_;
        const auto& src_strides = strides_;
        auto& out_data = out.get_data_ref();

        for (size_t flat = 0; flat < out_data.size(); ++flat) {
            size_t rem = flat;
            array<size_t, OutRank> idx{};
            for (int d = OutRank - 1; d >= 0; --d) {
                idx[d] = rem % target_shape[d];
                rem /= target_shape[d];
            }

            size_t flat_src = 0;
            for (size_t d = 0; d < N; ++d) {
                size_t src_idx = (shape__[d] == 1) ? 0 : idx[OutRank - N + d];
                flat_src += src_idx * src_strides[d];
            }

            out_data[flat] = src_data[flat_src];
        }

        return out;
    }

    size_t size() const { return data_.size(); }

    const T* data() const { return data_.data(); }
    T* data() { return data_.data(); }

    // --- Concatenate tensors along a given axis ---
    static Tensor<T, N> concat(const std::vector<Tensor<T, N>>& tensors, size_t axis) {
        if (tensors.empty()) throw std::runtime_error("No tensors to concat");

        auto shape0 = tensors[0].get_shape_ref();
        size_t concat_dim = 0;
        for (const auto& t : tensors) {
            auto s = t.get_shape_ref();
            for (size_t i = 0; i < N; i++) {
                if (i == axis) continue;
                if (s[i] != shape0[i])
                    throw std::runtime_error("Shapes must match except on concat axis");
            }
            concat_dim += s[axis];
        }

        Shape new_shape = shape0;
        new_shape[axis] = concat_dim;
        Tensor<T, N> result(new_shape);

        // Copy each tensor into the result
        size_t offset = 0;
        for (const auto& t : tensors) {
            result.copy_along_axis(t, axis, offset);
            offset += t.get_shape_ref()[axis];
        }

        return result;
    }

    // --- Copy source tensor into this tensor along a given axis with offset ---
    void copy_along_axis(const Tensor<T, N>& source, size_t axis, size_t offset) {
        std::array<size_t, N> idx_src{};
        std::array<size_t, N> idx_dest{};
        recursive_copy(source, idx_src, idx_dest, 0, axis, offset);
    }

    // --- Recursive helper for copy_along_axis ---
    void recursive_copy(const Tensor<T, N>& source,
                        std::array<size_t, N>& idx_src,
                        std::array<size_t, N>& idx_dest,
                        size_t dim,
                        size_t axis,
                        size_t offset)
    {
        if (dim == N) {
            size_t dest_flat = get_flat_index(idx_dest);
            size_t src_flat  = source.get_flat_index(idx_src);

            if (dest_flat >= data_.size())
                throw std::runtime_error("concat: destination flat index out of bounds");
            if (src_flat >= source.size())
                throw std::runtime_error("concat: source flat index out of bounds");

            data_[dest_flat] = source.get_data_ref()[src_flat];
            return;
        }

        for (size_t i = 0; i < source.get_shape_ref()[dim]; ++i) {
            idx_src[dim] = i;
            idx_dest[dim] = (dim == axis) ? i + offset : i;
            recursive_copy(source, idx_src, idx_dest, dim + 1, axis, offset);
        }
    }

    size_t get_flat_index(const std::array<size_t, N>& idx) const {
        size_t flat = 0;
        for (size_t i = 0; i < N; i++) {
            if (idx[i] >= shape__[i])
                throw std::runtime_error("Index out of bounds in get_flat_index");
            flat += idx[i] * strides_[i];
        }
        return flat;
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

    // Utility to multiply elements in a vector slice
    static size_t product(const vector<size_t>& v, size_t start, size_t end) {
        size_t p = 1;
        for (size_t i = start; i < end; i++) p *= v[i];
        return p;
    }

    size_t flat_index_broadcast(const array<size_t, N>& idx, const Tensor<T,N>& other) const {
        size_t flat = 0;
        const auto& s = other.get_shape_ref();
        const auto& str = other.get_strides_ref();
        for (size_t i = 0; i < N; ++i) {
            size_t dim_idx = (s[i] == 1) ? 0 : idx[i];
            flat += dim_idx * str[i];
        }
        return flat;
    }
};

template<typename T, size_t NA, size_t NB>
auto dot(const Tensor<T, NA>& A, const Tensor<T, NB>& B) {
    // Determine batch rank: any dimensions before last two in A or B
    constexpr size_t batch_rank = (NA > 2) ? NA - 2 : 0;
    constexpr size_t batch_rank_B = (NB > 2) ? NB - 2 : 0;
    constexpr size_t max_batch_rank = (batch_rank > batch_rank_B) ? batch_rank : batch_rank_B;
    constexpr size_t OUT_RANK = max_batch_rank + 2;

    const auto& A_shape = A.get_shape();
    const auto& B_shape = B.get_shape();

    std::array<size_t, OUT_RANK> out_shape{};

    // Compute broadcasted batch dimensions
    for(size_t i = 0; i < max_batch_rank; i++){
        size_t a_dim = (i < batch_rank) ? A_shape[i] : 1;
        size_t b_dim = (i < batch_rank_B) ? B_shape[i] : 1;
        if(a_dim != 1 && b_dim != 1 && a_dim != b_dim)
            throw std::runtime_error("Incompatible batch dims in dot()");
        out_shape[i] = std::max(a_dim, b_dim);
    }

    // Last two dimensions: matmul
    size_t A_rows = (NA >= 2) ? A_shape[NA-2] : 1;
    size_t A_cols = (NA >= 1) ? A_shape[NA-1] : 1;
    size_t B_rows = (NB >= 2) ? B_shape[NB-2] : 1;
    size_t B_cols = (NB >= 1) ? B_shape[NB-1] : 1;

    if(A_cols != B_rows)
        throw std::runtime_error("Inner dimensions do not match for dot product");

    out_shape[OUT_RANK-2] = A_rows;
    out_shape[OUT_RANK-1] = B_cols;

    Tensor<T, OUT_RANK> out(out_shape);

    const auto& A_data = A.get_data_ref();
    const auto& B_data = B.get_data_ref();
    auto& out_data = out.get_data_ref();

    const auto& A_strides = A.get_strides();
    const auto& B_strides = B.get_strides();
    const auto& out_strides = out.get_strides();

    size_t total = out.size();
    for(size_t idx = 0; idx < total; idx++){
        std::vector<size_t> idx_multi(OUT_RANK);
        size_t tmp = idx;
        for(int i=int(OUT_RANK)-1; i>=0; i--){
            idx_multi[i] = tmp % out_shape[i];
            tmp /= out_shape[i];
        }

        T sum = 0;
        for(size_t k = 0; k < A_cols; k++){
            // Compute offset for A
            size_t a_offset = 0;
            for(size_t i = 0; i < batch_rank; i++){
                size_t ix = (i < batch_rank) ? idx_multi[i] : 0;
                if(A_shape[i]==1) ix = 0;
                a_offset += ix * A_strides[i];
            }
            if(NA >= 2) a_offset += idx_multi[max_batch_rank] * A_strides[NA-2];
            if(NA >= 1) a_offset += k * A_strides[NA-1];

            // Compute offset for B
            size_t b_offset = 0;
            for(size_t i = 0; i < batch_rank_B; i++){
                size_t ix = (i < batch_rank_B) ? idx_multi[i] : 0;
                if(B_shape[i]==1) ix = 0;
                b_offset += ix * B_strides[i];
            }
            if(NB >= 2) b_offset += k * B_strides[NB-2];
            if(NB >= 1) b_offset += idx_multi[max_batch_rank+1] * B_strides[NB-1];

            sum += A_data[a_offset] * B_data[b_offset];
        }

        // Compute output offset
        size_t out_offset = 0;
        for(size_t i = 0; i < OUT_RANK; i++)
            out_offset += idx_multi[i] * out_strides[i];

        out_data[out_offset] = sum;
    }

    return out;
}

// Global scalar operators
template<typename T, size_t N>
Tensor<T,N> operator+(T scalar, const Tensor<T,N>& tensor) {
    return tensor + scalar;
}

template<typename T, size_t N>
Tensor<T,N> operator-(T scalar, const Tensor<T,N>& tensor) {
    Tensor<T,N> result(tensor.get_shape());
    const auto& data = tensor.get_data_ref();
    auto& out = result.get_data_ref();
    for (size_t i = 0; i < data.size(); ++i)
        out[i] = scalar - data[i];
    return result;
}

template<typename T, size_t N>
Tensor<T,N> operator*(T scalar, const Tensor<T,N>& tensor) {
    return tensor * scalar;
}

template<typename T, size_t N>
Tensor<T,N> operator/(T scalar, const Tensor<T,N>& tensor) {
    Tensor<T,N> result(tensor.get_shape());
    const auto& data = tensor.get_data_ref();
    auto& out = result.get_data_ref();
    for (size_t i = 0; i < data.size(); ++i)
        out[i] = scalar / data[i];
    return result;
}
