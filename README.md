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
- **M2 Complete**: Multi-layer backprop with per-layer cache; training with configurable depth (1-N layers, tested with 1-4 layers)

## Current Scope

Implemented and tested:
- Tensor construction, indexing, reshape/view, transpose
- `relu`, `softmax`, `mean_axis`, `sum`
- Broadcast forward and backward reduction
- Autograd for key operations:
	- elementwise add/subtract/multiply/divide
	- `relu`, `mean_axis`, `sum`
	- 2D and batched 3D `matmul` backward
	- LayerNorm backward (pre-norm architecture)
- SGD optimizer (`zero_grad`, `step`, clipping, weight decay)
- Tiny transformer training with:
	- Multi-head causal attention (configurable heads)
	- Pre-norm LayerNorm (configurable)
 	- Configurable depth (1-N layers supported for training)
	- AdamW with bias correction
	- Learning rate scheduling (warmup + cosine)

Limitations (intentional for this stage):
 - Full backprop/training currently remains enabled for `num_layers=1` with `use_layernorm=false`.
## Build (Git Bash / MINGW64 on Windows)
Limitations (intentional for this stage):
- Not yet a full deep-learning framework
- No full module stack yet (e.g., complete MLP/attention module API)
- Some long-chain autograd patterns still need hardening for production-scale training

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

The demo now includes a tiny next-token training loop and prints loss drop before generation.

## Project Layout

- `tensor_minimal.hpp`: tensor core, autograd, mixed precision, optimizer
- `tensor.cpp`: tiny training demo executable
- `toy_llm.cpp`: tiny transformer-style inference/generation executable
- `tests/simple_test.cpp`: functional + gradient validation suite
- `tests/toy_llm_test.cpp`: causal mask + training-loss + deterministic-generation tests
- `FINAL_LLM_ROADMAP.md`: staged roadmap to a production-like LLM stack
- `CMakeLists.txt`: build + test config

## Next Suggested Milestones

Use the staged plan in `FINAL_LLM_ROADMAP.md`.

Current status:
- M1 (reliability infrastructure) is complete: checkpoint save/load + round-trip test.
- M2 is in progress: forward supports configurable depth, optional pre-norm layernorm, and multi-head attention; training/backprop supports multi-head for single-layer no-LayerNorm mode.
- M3 is in progress: optional AdamW + warmup/cosine scheduler, minibatch training, and global gradient clipping are available in training.

Current technical caveat:
- Full backprop/training currently remains enabled for `num_layers=1` with `use_layernorm=false`.

## License

See `LICENSE`.
