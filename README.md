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

Run with external dataset file(s) (char tokenizer mode):

```bash
./build-mingw/toy_llm.exe --data data.csv --prompt "hello "
```

Run in subword mode with a vocab file (one token per line):

```bash
./build-mingw/toy_llm.exe --data data.csv --vocab vocab.txt --prompt "hello "
```

Build a starter vocab file from corpus frequency stats:

```bash
./build-mingw/toy_llm.exe --data data.csv --build-vocab vocab.txt --vocab-size 256
```

Run with validation split and sampling controls:

```bash
./build-mingw/toy_llm.exe --data data.csv --vocab vocab.txt --val-split 0.1 --top-p 0.9 --rep-penalty 1.1 --temperature 0.9 --top-k 20 --steps 120 --prompt "hello "
```

Save training checkpoint (includes optimizer state and scheduler step):

```bash
./build-mingw/toy_llm.exe --data data.csv --vocab vocab.txt --save-ckpt run.ckpt
```

Resume training from checkpoint:

```bash
./build-mingw/toy_llm.exe --data data.csv --vocab vocab.txt --resume run.ckpt --save-ckpt run.ckpt
```

Run with explicit training hyperparameters:

```bash
./build-mingw/toy_llm.exe --data data.csv --vocab vocab.txt --optimizer adamw --epochs 80 --ctx-window 32 --lr 0.003 --warmup 100 --min-lr-ratio 0.2 --weight-decay 0.0001 --batch-size 8 --grad-clip 1.0 --report-every 5 --report-lr --report-grad
```

Run from a JSON config file (CLI flags override config values):

```bash
./build-mingw/toy_llm.exe --config run.json --save-ckpt run.ckpt
```

Export the fully resolved effective config (after CLI overrides):

```bash
./build-mingw/toy_llm.exe --config run.json --epochs 120 --save-config effective-run.json
```

Dry-run to validate settings and print the resolved config without training:

```bash
./build-mingw/toy_llm.exe --config run.json --dry-run
```

Enable strict config validation (reject unknown JSON keys):

```bash
./build-mingw/toy_llm.exe --config run.json --strict-config --dry-run
```

Example `run.json`:

```json
{
	"num_layers": 1,
	"data": ["data.csv"],
	"vocab": "vocab.txt",
	"prompt": "hello ",
	"epochs": 80,
	"ctx_window": 32,
	"lr": 0.003,
	"optimizer": "adamw",
	"warmup": 100,
	"min_lr_ratio": 0.2,
	"weight_decay": 0.0001,
	"batch_size": 8,
	"grad_clip": 1.0,
	"report_every": 5,
	"report_lr": true,
	"report_grad": true,
	"val_split": 0.1,
	"temperature": 0.9,
	"top_k": 20,
	"top_p": 0.9,
	"rep_penalty": 1.1,
	"steps": 120,
	"resume": "run.ckpt",
	"save_ckpt": "run.ckpt",
	"save_config": "effective-run.json",
	"dry_run": false,
	"strict_config": false
}
```

Vocab file notes:
- Put one token per line.
- Optional special token `<unk>` can be included for unknown chunks.
- Escapes `\\n`, `\\t`, `\\r`, and `\\s` are supported in vocab lines.

Training notes:
- `--val-split` reserves the tail of tokenized data for validation.
- When validation data is present, the demo prints validation loss/perplexity.
- `--resume` continues from stored optimizer moments and global step.
- `--optimizer` supports `sgd` and `adamw`.
- `--warmup` and `--min-lr-ratio` control warmup + cosine schedule.
- `--config` accepts a flat JSON object with scalar fields and `data` string array.
- `--save-config` writes the merged effective config for reproducible reruns.
- `--dry-run` prints resolved settings and exits before dataset/tokenizer/training steps.
- `--strict-config` fails fast on unknown config keys.
- In non-strict mode, unknown config keys are ignored but emitted in a `warnings` array in dry-run/effective config output.

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
