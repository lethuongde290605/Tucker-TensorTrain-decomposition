# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Activation-aware Tucker decomposition for compressing OPT attention layers. The core idea: collect Q/K/V/O activations from calibration data, Tucker-decompose them, then project the original weight matrices into the compressed subspace via the Kronecker product of Tucker factors. This reduces the head dimension stored in the KV-cache at inference time.

## Commands

```bash
# Tucker-only smoke test (OPT-125m, CPU, fast)
python test_tucker_inference.py --model facebook/opt-125m --nsamples 4 --no_baseline

# With baseline comparison
python test_tucker_inference.py --model facebook/opt-125m --nsamples 4

# Custom prompt
python test_tucker_inference.py --model facebook/opt-125m --prompt "The Eiffel Tower is located in"
```

## Key Modules (`decompose/`)

- **`tucker_utils.py`** — Tucker decomposition of activations: `factorize_dim`, `calculate_rank`, `tucker_decompose_opt_layer`. Reshapes activations `(N, seq, D)` → `(N*seq, d_1, ..., d_k)` then runs `partial_tucker` on modes 1..k.
- **`eigen_attn.py`** — Main pipeline: iterates transformer layers, collects activations, calls Tucker decomposition, computes compressed weights, swaps in `OPTTuckerDecoderLayer`.
- **`eigen_attn_utils.py`** — Helpers: `get_kqv_opt`, `get_out_proj_input_opt`, `project_bias_with_tucker_factors`, `compute_kron_product`, `compute_tucker_only_weights_qkv`, `compute_tucker_only_weights_o`.

## Key Modules (`models/`)

- **`decompose_modules.py`** — Compressed layer classes: `OPTTuckerAttention`, `OPTTuckerDecoderLayer`. `OPTTuckerAttention` accepts pre-projected weights `w_q/w_k/w_v` of shape `(Q, embed_dim)` and `w_o` of shape `(embed_dim, Q_o)`.

## Architecture Flow

1. `eigenattn()` in `eigen_attn.py` iterates decoder layers and hooks activations via `get_kqv_opt` / `get_out_proj_input_opt`.
2. `tucker_decompose_opt_layer()` reshapes activations and runs `partial_tucker`, returning factor matrices `U_i` of shape `(n_i, q_i)` for each projection (Q/K/V/O).
3. `compute_kron_product(factors)` computes `kron(U_1, ..., U_k)` → shape `(embed_dim, Q)`.
4. `compute_tucker_only_weights_qkv`: `new_weight = kron_W.T @ orig_weight` → shape `(Q, embed_dim)`.
5. `compute_tucker_only_weights_o`: `new_weight = orig_weight @ kron_W` → shape `(embed_dim, Q_o)`.
6. `OPTTuckerDecoderLayer` wraps `OPTTuckerAttention` and replaces the original layer in-place.
7. Inference: KV-cache stores `(seq_len, head_dim_compressed)` instead of `(seq_len, head_dim_orig)`.

## Rank / Dimension Notes (OPT-125m)

- `hidden_size = 768`, `num_heads = 12`, original `head_dim = 64`
- `factorize_dim(768, 5)` → `[4, 4, 4, 4, 3]` (product = 768)
- `calculate_rank` hardcodes compressed ranks `[4, 4, 4, 3, 3]` → `Q = 576`
- Compressed `head_dim = 576 // 12 = 48` — 25% KV-cache reduction per layer
