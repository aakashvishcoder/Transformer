// ops_autograd.hpp
#pragma once
#include "autograd.hpp"
#include "tensor.hpp"
// Elementwise add
template<typename T, size_t N>
std::shared_ptr<Var<T,N>> add(const std::shared_ptr<Var<T,N>>& a,
                              const std::shared_ptr<Var<T,N>>& b)
{
    auto out = std::make_shared<Var<T,N>>(a->value + b->value,
                                          a->requires_grad || b->requires_grad);
    if(!out->requires_grad) return out;

    out->grad_fn = std::make_shared<typename Var<T,N>::Edge>();
    out->grad_fn->parents = {a, b};
    out->grad_fn->backward = [a,b](const Tensor<T,N>& upstream){
        if(a && a->requires_grad) a->grad += upstream;
        if(b && b->requires_grad) b->grad += upstream;
    };
    return out;
}

// Elementwise sub
template<typename T, size_t N>
std::shared_ptr<Var<T,N>> sub(const std::shared_ptr<Var<T,N>>& a,
                              const std::shared_ptr<Var<T,N>>& b)
{
    auto out = std::make_shared<Var<T,N>>(a->value - b->value,
                                          a->requires_grad || b->requires_grad);
    if(!out->requires_grad) return out;

    out->grad_fn = std::make_shared<typename Var<T,N>::Edge>();
    out->grad_fn->parents = {a, b};
    out->grad_fn->backward = [a,b](const Tensor<T,N>& upstream){
        if(a && a->requires_grad) a->grad += upstream;
        if(b && b->requires_grad) b->grad -= upstream;
    };
    return out;
}

// Elementwise mul
template<typename T, size_t N>
std::shared_ptr<Var<T,N>> mul(const std::shared_ptr<Var<T,N>>& a,
                              const std::shared_ptr<Var<T,N>>& b)
{
    auto out = std::make_shared<Var<T,N>>(a->value * b->value,
                                          a->requires_grad || b->requires_grad);
    if(!out->requires_grad) return out;

    out->grad_fn = std::make_shared<typename Var<T,N>::Edge>();
    out->grad_fn->parents = {a, b};
    out->grad_fn->backward = [a,b](const Tensor<T,N>& upstream){
        if(a && a->requires_grad) {
            Tensor<T,N> tmp = upstream * b->value;
            a->grad += tmp;
        }
        if(b && b->requires_grad) {
            Tensor<T,N> tmp = upstream * a->value;
            b->grad += tmp;
        }
    };
    return out;
}

// Elementwise div
template<typename T, size_t N>
std::shared_ptr<Var<T,N>> divv(const std::shared_ptr<Var<T,N>>& a,
                               const std::shared_ptr<Var<T,N>>& b)
{
    auto out = std::make_shared<Var<T,N>>(a->value / b->value,
                                          a->requires_grad || b->requires_grad);
    if(!out->requires_grad) return out;

    out->grad_fn = std::make_shared<typename Var<T,N>::Edge>();
    out->grad_fn->parents = {a, b};
    out->grad_fn->backward = [a,b](const Tensor<T,N>& upstream){
        if(a && a->requires_grad) a->grad += upstream / b->value;
        if(b && b->requires_grad) {
            // d(a/b)/db = -a/(b^2)
            Tensor<T,N> tmp = a->value;
            tmp /= (b->value * b->value);
            tmp *= T(-1);
            tmp *= upstream;
            b->grad += tmp;
        }
    };
    return out;
}

// Unary exp / log / sqrt
template<typename T, size_t N>
std::shared_ptr<Var<T,N>> vexp(const std::shared_ptr<Var<T,N>>& a){
    auto val = a->value.exp();
    auto out = std::make_shared<Var<T,N>>(val, a->requires_grad);
    if(!out->requires_grad) return out;

    out->grad_fn = std::make_shared<typename Var<T,N>::Edge>();
    out->grad_fn->parents = {a};
    out->grad_fn->backward = [a, val](const Tensor<T,N>& upstream){
        if(a && a->requires_grad){
            Tensor<T,N> tmp = upstream * val; // d exp = exp * da
            a->grad += tmp;
        }
    };
    return out;
}

template<typename T, size_t N>
std::shared_ptr<Var<T,N>> vlog(const std::shared_ptr<Var<T,N>>& a){
    auto out = std::make_shared<Var<T,N>>(a->value.log(), a->requires_grad);
    if(!out->requires_grad) return out;
    out->grad_fn = std::make_shared<typename Var<T,N>::Edge>();
    out->grad_fn->parents = {a};
    out->grad_fn->backward = [a](const Tensor<T,N>& upstream){
        if(a && a->requires_grad){
            Tensor<T,N> tmp = upstream / a->value; // d log = (1/a) da
            a->grad += tmp;
        }
    };
    return out;
}

template<typename T, size_t N>
std::shared_ptr<Var<T,N>> vsqrt(const std::shared_ptr<Var<T,N>>& a){
    auto val = a->value.sqrt();
    auto out = std::make_shared<Var<T,N>>(val, a->requires_grad);
    if(!out->requires_grad) return out;
    out->grad_fn = std::make_shared<typename Var<T,N>::Edge>();
    out->grad_fn->parents = {a};
    out->grad_fn->backward = [a, val](const Tensor<T,N>& upstream){
        if(a && a->requires_grad){
            // d sqrt(a) = (1/(2*sqrt(a))) da
            Tensor<T,N> tmp = upstream / (val * T(2));
            a->grad += tmp;
        }
    };
    return out;
}

// Power by scalar: a ^ c
template<typename T, size_t N>
std::shared_ptr<Var<T,N>> pow_scalar(const std::shared_ptr<Var<T,N>>& a, T c){
    auto val = (a->value ^ c);
    auto out = std::make_shared<Var<T,N>>(val, a->requires_grad);
    if(!out->requires_grad) return out;

    out->grad_fn = std::make_shared<typename Var<T,N>::Edge>();
    out->grad_fn->parents = {a};
    out->grad_fn->backward = [a, c, val](const Tensor<T,N>& upstream){
        if(a && a->requires_grad){
            // d(a^c) = c * a^(c-1) da = c * val / a * da (stable when a>0)
            Tensor<T,N> tmp = val / a->value;
            tmp *= c;
            tmp *= upstream;
            a->grad += tmp;
        }
    };
    return out;
}

// Sum keep-dim over an axis (rank stays N)
template<typename T, size_t N>
std::shared_ptr<Var<T,N>> sum_keepdim(const std::shared_ptr<Var<T,N>>& a, size_t axis){
    auto val = sum_axis_keepdim(a->value, axis);
    auto out = std::make_shared<Var<T,N>>(val, a->requires_grad);
    if(!out->requires_grad) return out;

    out->grad_fn = std::make_shared<typename Var<T,N>::Edge>();
    out->grad_fn->parents = {a};
    out->grad_fn->backward = [a, axis](const Tensor<T,N>& upstream){
        if(!a || !a->requires_grad) return;
        // upstream has size-1 on 'axis'. We need to tile it along that axis.
        Tensor<T,N> tiled = upstream; // shape [...,1,...]
        // simple tiling via repeated add; for speed you can implement a proper tile
        auto s = a->value.get_shape();
        size_t len = s[axis];
        // Make a temp result initialized to zeros (same shape as a)
        Tensor<T,N> g(a->value.get_shape()); g.zeros();

        // Iterate along axis, placing upstream in each slice
        std::array<size_t, N> idx{};
        // fill whole tensor by copying upstream at each position of axis
        for(size_t flat=0; flat<g.size(); ++flat){
            size_t rem = flat;
            for(int d=N-1; d>=0; --d){ idx[d] = rem % s[d]; rem/=s[d]; }
            auto idx_up = idx; idx_up[axis] = 0; // because upstream has size 1 at axis
            g(idx) += upstream(idx_up);
        }
        a->grad += g;
    };
    return out;
}
