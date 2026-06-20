# C++ Tensor + Autograd Mini-Core

A lightweight C++ tensor core aimed at future LLM experimentation.

This project now includes:
- N-D tensor storage with shape/stride support
- Tensor view support (`TensorView`) for non-owning access
- Broadcast-aware elementwise ops (`+`, `-`, `*`, `/`)
- 2D and batched 3D `matmul`
- Basic mixed-precision data types (`Float16`, `BFloat16`)
- Minimal autograd graph with backward support for core ops
- SGD optimizer utility with optional weight decay and grad clipping
- Tiny transformer-style inference demo (char tokenizer + causal attention + autoregressive generation)
- CMake-based test suite with numerical gradient checks

## Current Scope

Implemented and tested:
- Tensor construction, indexing, reshape/view, transpose
- `relu`, `softmax`, `mean_axis`, `sum`
- Broadcast forward and backward reduction
- Autograd for key operations:
	- elementwise add/subtract/multiply/divide
	- `relu`, `mean_axis`, `sum`
	- 2D and batched 3D `matmul` backward
- SGD optimizer (`zero_grad`, `step`, clipping, weight decay)

Limitations (intentional for this stage):
- Not yet a full deep-learning framework
- No full module stack yet (e.g., complete MLP/attention module API)
- Some long-chain autograd patterns still need hardening for production-scale training

## Build (Git Bash / MINGW64 on Windows)

If CMake is not on PATH in your shell:

```bash
export PATH="/c/Program Files/CMake/bin:$PATH"
```

Configure and build:

```bash
cmake -S . -B build-mingw -G "MinGW Makefiles" \
	-DCMAKE_C_COMPILER=/c/msys64/ucrt64/bin/gcc.exe \
	-DCMAKE_CXX_COMPILER=/c/msys64/ucrt64/bin/g++.exe

cmake --build build-mingw -j
```

Run tests:

```bash
ctest --test-dir build-mingw --output-on-failure
```

Run demo app:

```bash
./build-mingw/main.exe
```

Run transformer-style generation demo:

```bash
./build-mingw/toy_llm.exe
```

## Project Layout

- `tensor_minimal.hpp`: tensor core, autograd, mixed precision, optimizer
- `tensor.cpp`: tiny training demo executable
- `toy_llm.cpp`: tiny transformer-style inference/generation executable
- `tests/simple_test.cpp`: functional + gradient validation suite
- `CMakeLists.txt`: build + test config

## Next Suggested Milestones

1. Add a `Linear` module wrapper and parameter container API.
2. Add 2-layer MLP training example using existing optimizer.
3. Harden autograd graph ownership for deeper chained expression trees.
4. Add attention-path primitives (`qk^T`, scaling, masking, softmax, value projection).

## License

See `LICENSE`.
