// autograd.hpp
#pragma once
#include <memory>
#include <functional>
#include <vector>
#include <stack>
#include <unordered_set>
#include "tensor.hpp"
// Forward-declare your Tensor
template<typename T, size_t N> class Tensor;

template<typename T, size_t N>
struct Var {
    Tensor<T, N> value;                 // forward value
    Tensor<T, N> grad;                  // accumulated gradient (same shape)
    bool requires_grad = false;

    // Each Var can have a grad_fn that knows how to push gradients to parents
    struct Edge {
        // parents (leaves have empty vector)
        std::vector<std::shared_ptr<Var>> parents;
        // backward callback: given upstream grad (same shape as value),
        // compute and accumulate grads into each parent->grad
        std::function<void(const Tensor<T, N>& upstream)> backward;
    };

    std::shared_ptr<Edge> grad_fn;      // null for leaves

    Var() = default;
    Var(const Tensor<T,N>& v, bool req=false) : value(v), requires_grad(req) {
        grad = Tensor<T,N>(v.get_shape()); grad.zeros();
    }

    static std::shared_ptr<Var> leaf(const Tensor<T,N>& v, bool req=true) {
        return std::make_shared<Var>(v, req);
    }

    // Zero grads in this subgraph (simple DFS)
    void zero_grad() {
        std::unordered_set<const Var*> seen;
        std::stack<Var*> st; st.push(this);
        while(!st.empty()){
            Var* cur = st.top(); st.pop();
            if(seen.count(cur)) continue;
            seen.insert(cur);
            cur->grad.zeros();
            if(cur->grad_fn){
                for(auto& p: cur->grad_fn->parents) if(p) st.push(p.get());
            }
        }
    }

    // Start backprop from this node.
    // If you’re calling backward on a scalar-like tensor (all dims 1),
    // seed with ones. Otherwise pass an explicit upstream.
    void backward() {
        Tensor<T,N> ones(value.get_shape()); ones.ones();
        _backward(ones);
    }

    void backward(const Tensor<T,N>& upstream) {
        _backward(upstream);
    }

private:
    void _backward(const Tensor<T,N>& upstream) {
        // Non-recursive post-order traversal
        struct Frame { Var* v; int state; }; // 0 = visit, 1 = backprop
        std::unordered_set<const Var*> seen;
        std::vector<Var*> topo;

        std::stack<Frame> st; st.push({this,0});
        while(!st.empty()){
            auto [v, state] = st.top(); st.pop();
            if(state==0){
                if(seen.count(v)) continue;
                seen.insert(v);
                st.push({v,1});
                if(v->grad_fn){
                    for(auto& p: v->grad_fn->parents) if(p) st.push({p.get(),0});
                }
            } else {
                topo.push_back(v);
            }
        }

        // seed root
        this->grad += upstream;

        // backprop along topo order (excluding the root which is last)
        for(int i=int(topo.size())-1; i>=0; --i){
            Var* v = topo[i];
            if(v->grad_fn){
                v->grad_fn->backward(v->grad); // pushes into parents
            }
        }
    }
};
