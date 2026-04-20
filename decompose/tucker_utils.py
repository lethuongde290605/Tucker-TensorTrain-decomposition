import math
import numpy as np
import tensorly as tl
import torch

tl.set_backend('pytorch')  # TensorLy must use the PyTorch backend since all tensors are torch.Tensor

from tensorly.decomposition import (
    tucker,
    partial_tucker
)

from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.tenalg import multi_mode_dot
from tensorly.random import random_tucker


# ---------------------------------------------------------------------------
# Utility: factorize a single dimension into `count` nearly-equal factors
# (moved from test_tensortrain.py)
# ---------------------------------------------------------------------------

def factorize_dim(tensor_dim: int, count: int) -> list[int]:
    """
    Split tensor_dim into `count` factors whose product equals tensor_dim,
    choosing factors as equal as possible (largest factors first).

    Args:
        tensor_dim: The dimension value to factorize (e.g. 768).
        count:      How many factors to produce (e.g. 5).

    Returns:
        A list of `count` integers whose product == tensor_dim,
        sorted in descending order so larger factors come first.

    Example:
        >>> factorize_dim(768, 5)
        [4, 4, 4, 4, 3]   # 4*4*4*4*3 = 768
    """
    if count <= 0:
        raise ValueError(f"count must be a positive integer, got {count}")
    if tensor_dim <= 0:
        raise ValueError(f"tensor_dim must be a positive integer, got {tensor_dim}")

    factors = []
    remaining = tensor_dim

    for i in range(count, 0, -1):
        f = round(remaining ** (1.0 / i))
        while f > 1 and remaining % f != 0:
            f -= 1
        if f < 1:
            f = 1
        factors.append(f)
        remaining //= f

    if remaining != 1:
        raise ValueError(
            f"Cannot factorize {tensor_dim} into exactly {count} integer factors. "
            f"Partial result: {factors}, leftover: {remaining}"
        )

    factors.sort(reverse=True)
    return factors


# ---------------------------------------------------------------------------
# Tensor preparation
# ---------------------------------------------------------------------------

def prepare_tensor(tensor, dim1_factors: list[int]):
    """
    Reshape a 2-D tensor of shape (N, D) by factorizing the second dimension D
    into sub-dimensions given by dim1_factors.

    The first dimension N is kept as-is, and the second dimension is expanded
    into len(dim1_factors) sub-dimensions.

    Args:
        tensor:       2-D numpy array of shape (N, D).
        dim1_factors: Factorization of D (product must equal D).

    Returns:
        Reshaped tensor of shape (N, *dim1_factors).
    """
    result = tensor.reshape(tensor.shape[0] * tensor.shape[1], *dim1_factors)
    return result


# ---------------------------------------------------------------------------
# Rank calculation helpers (module-level, not nested)
# ---------------------------------------------------------------------------

def _tucker_compressed_size(tensor_shape: tuple, modes: list[int], dims: list[int], ranks: list[int]) -> int:
    """
    Compute the total number of parameters after a Partial Tucker decomposition.

    Args:
        tensor_shape: Shape of the original tensor.
        modes:        Mode indices being decomposed.
        dims:         Original dimension size for each mode.
        ranks:        Current Tucker rank for each mode.

    Returns:
        Number of parameters = product(core shape) + sum(dim_i * rank_i).
    """
    core_shape = list(tensor_shape)
    for i, m in enumerate(modes):
        core_shape[m] = ranks[i]
    core_sz = math.prod(core_shape)
    factors_sz = sum(d * r for d, r in zip(dims, ranks))
    return core_sz + factors_sz


def _tucker_compression_rate(tensor_shape: tuple, modes: list[int], dims: list[int], ranks: list[int]) -> float:
    """
    Compute the compression rate for the given ranks.

    compression_rate = (size_core + size_factors) / size_original

    Args:
        tensor_shape: Shape of the original tensor.
        modes:        Mode indices being decomposed.
        dims:         Original dimension size for each mode.
        ranks:        Current Tucker rank for each mode.

    Returns:
        Compression rate in (0, 1].
    """
    original_size = math.prod(tensor_shape)
    return _tucker_compressed_size(tensor_shape, modes, dims, ranks) / original_size


def _rank_imbalance(dims: list[int], ranks: list[int]) -> float:
    """
    Measure how unbalanced the rank reductions are across modes.

    Defined as max(rank_i / dim_i) - min(rank_i / dim_i).
    A value of 0 means all modes are reduced proportionally.

    Args:
        dims:  Original dimension sizes.
        ranks: Current Tucker ranks.

    Returns:
        Imbalance score (lower is more uniform).
    """
    ratios = [r / d for r, d in zip(ranks, dims)]
    return max(ratios) - min(ratios)


# ---------------------------------------------------------------------------
# Rank calculation
# ---------------------------------------------------------------------------

def calculate_rank(tensor_shape: tuple, modes: list[int], target_ratio: float, num_head: int) -> list[int]:
    """
    Find Tucker ranks for the given modes such that the compression rate
    (size_core + size_factors) / size_original is as close as possible to
    target_ratio without exceeding it.

    Algorithm (greedy rank reduction):
        1. Start with full ranks (rank[i] = dim[i] for each mode).
        2. At each step, enumerate all single-rank-decrement candidates
           (i.e. reduce exactly one mode's rank by 1).
        3. Among candidates whose resulting rate is still <= target_ratio,
           prefer the one closest to the target, breaking ties by rank balance.
        4. If no candidate is within the target yet, pick the most balanced
           decrement to approach the target uniformly.
        5. Stop when the compression rate reaches or falls below target_ratio,
           or when no further reduction improves the rate.

    Args:
        tensor_shape: Shape of the original tensor (e.g. (1024, 4, 4, 4, 4, 3)).
        modes:        List of mode indices to decompose (e.g. [1, 2, 3, 4, 5]).
        target_ratio: Target compression rate in (0, 1].
                      compression_rate = (size_core + size_factors) / size_tensor
        num_head:     The product of the resulting ranks must be divisible by this value.

    Returns:
        A list of ranks, one per entry in `modes`, whose compression rate is
        closest to (but does not exceed) target_ratio.
    """
    dims = [tensor_shape[m] for m in modes]
    ranks = dims.copy()  # start at full rank (lossless)

    best_ranks = ranks.copy()
    best_rate = _tucker_compression_rate(tensor_shape, modes, dims, ranks)

    while True:
        # Build all single-decrement candidates
        candidates = []
        for i in range(len(ranks)):
            if ranks[i] > 1:
                new_ranks = ranks.copy()
                new_ranks[i] -= 1
                rate = _tucker_compression_rate(tensor_shape, modes, dims, new_ranks)
                imb  = _rank_imbalance(dims, new_ranks)
                candidates.append((rate, imb, i, new_ranks))

        if not candidates:
            break

        # Prefer candidates that already satisfy the target ratio
        valid = [c for c in candidates if c[0] <= target_ratio]

        if valid:
            # Among valid candidates: closest to target first, then most balanced
            valid.sort(key=lambda x: (target_ratio - x[0], x[1]))
            best = valid[0]
        else:
            # Not yet within target: reduce further, prioritise balance
            candidates.sort(key=lambda x: (x[1], x[0]))
            best = candidates[0]

        new_rate, _, idx, new_ranks = best

        # Tính tích các phần tử trong new_ranks để kiểm tra ràng buộc
        prod_ranks = 1
        for r in new_ranks:
            prod_ranks *= r

        # Stop if this step does not improve the rate AND constraints are met
        if new_rate >= best_rate and prod_ranks % num_head == 0:
            break

        ranks = new_ranks
        best_rate = new_rate
        best_ranks = ranks.copy()

        # Stop if target ratio is reached AND the product is divisible by num_head
        if best_rate <= target_ratio and prod_ranks % num_head == 0:
            break

    print(f"  Target compression ratio : {target_ratio:.4f}")
    print(f"  Achieved compression rate: {best_rate:.4f}")
    print(f"  Calculated ranks         : {best_ranks}")

    return best_ranks


# ---------------------------------------------------------------------------
# Decompose & Reconstruct (split from original test_partial_tucker)
# ---------------------------------------------------------------------------

def decompose(tensor, rank, modes):
    """
    Decompose a tensor using Partial Tucker decomposition.

    Args:
        tensor: Input numpy array.
        rank:   Tucker ranks for the specified modes.
        modes:  List of mode indices to decompose.

    Returns:
        core:    The Tucker core tensor.
        factors: List of factor matrices.
    """

    tensor = tensor.to(torch.float32)

    # Guard: replace NaN/Inf that would crash CUDA SVD (cusolver)
    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()
    if n_nan > 0 or n_inf > 0:
        print(f"  [WARN] tensor contains {n_nan} NaN and {n_inf} Inf values — replacing with 0")
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        with torch.amp.autocast('cuda', enabled=False):
            (core, factors), rec_errors = partial_tucker(
                tensor, rank=rank, modes=modes, n_iter_max=200, verbose=True
            )
    except RuntimeError as e:
        # Fallback: run decomposition on CPU if CUDA SVD fails
        print(f"  [WARN] CUDA SVD failed ({e}). Retrying on CPU ...")
        tensor_cpu = tensor.cpu()
        with torch.amp.autocast('cpu', enabled=False):
            (core, factors), rec_errors = partial_tucker(
                tensor_cpu, rank=rank, modes=modes, n_iter_max=200, verbose=True
            )
        # Move results back to original device
        dev = tensor.device if tensor.is_cuda else torch.device('cpu')
        core    = core.to(dev)
        factors = [f.to(dev) for f in factors]

    print("core shape: ", core.shape)
    print("factor shapes:")
    for f in factors:
        print(" ", f.shape)

    # Compression stats
    original_size = math.prod(tensor.shape)
    compressed_size = sum(f.numel() for f in factors) + core.numel()
    compression_ratio = compressed_size / original_size

    print()
    print("--- Compression ---")
    print(f"  Original size    : {original_size} params")
    print(f"  Compressed size  : {compressed_size} params")
    print(f"  Compression ratio: {compression_ratio:.4f}  ({compression_ratio * 100:.2f} %)")

    return core, factors, compression_ratio


def reconstruct(tensor, core, factors, modes):
    """
    Reconstruct the original tensor from a Partial Tucker decomposition and
    compute reconstruction error metrics.

    Args:
        tensor:  Original numpy array (used for error calculation).
        core:    Tucker core tensor.
        factors: List of factor matrices from decomposition.
        modes:   List of mode indices that were decomposed.

    Returns:
        relative_error: Frobenius-norm relative error.
    """
    reconstructed_tensor = multi_mode_dot(core, factors, modes=modes)

    tensor_t = tensor
    reconstructed_t = reconstructed_tensor
    diff = tensor_t - reconstructed_t

    frob_error = torch.norm(diff).item()
    frob_orig  = torch.norm(tensor_t).item()

    relative_error = frob_error / frob_orig
    mse = torch.mean(diff ** 2).item()
    mae = torch.mean(torch.abs(diff)).item()

    print()
    print("--- Reconstruction Error ---")
    print(f"  Frobenius norm (original) : {frob_orig:.6f}")
    print(f"  Frobenius norm (error)    : {frob_error:.6f}")
    print(f"  Relative error            : {relative_error:.6f}  ({relative_error * 100:.4f} %)")
    print(f"  MSE                       : {mse:.8f}")
    print(f"  MAE                       : {mae:.8f}")

    return relative_error

def _decompose_one(name: str, tensor, rank: list[int], modes: list[int]) -> dict:
    """
    Run decompose + reconstruct for a single tensor and return a result dict.

    Args:
        name:   Label used for log messages (e.g. "Q", "K", "V").
        tensor: Input numpy array to decompose.
        rank:   Tucker ranks per mode.
        modes:  Mode indices to decompose.

    Returns:
        Dict with keys "core", "factors", "compression_ratio", "relative_error".
    """
    print(f"\n--- {name} ---")
    core, factors, comp = decompose(tensor, rank=rank, modes=modes)
    rel_err = reconstruct(tensor, core, factors, modes=modes)
    return {"core": core, "factors": factors, "compression_ratio": comp, "relative_error": rel_err}


def tucker_decompose_opt_layer(layer, fp_inps, args, num_heads, layer_id, compression_ratio=0.7, num_factors=5):
    """
    Decompose the Q, K, V, and out_proj-input (O) activation tensors of an OPT
    attention layer using Partial Tucker decomposition.

    Args:
        layer:             Transformer layer whose Q/K/V/O projections are hooked.
        fp_inps:           Full-precision calibration inputs.
        args:              Argument namespace (must contain eigen_attn_params).
        num_heads:         Number of attention heads.
        layer_id:          Layer index (kept for API consistency).
        compression_ratio: Target Tucker compression rate in (0, 1].
        num_factors:       Number of sub-dimensions to split each feature dim into.

    Returns:
        Dict with keys "Q", "K", "V", "O", each mapping to a sub-dict containing
        "core", "factors", "compression_ratio", and "relative_error".
    """
    from decompose.eigen_attn_utils import get_kqv_opt, get_out_proj_input_opt

    # ---- Q / K / V activations -----------------------------------------------
    tensor_k, tensor_q, tensor_v = get_kqv_opt(layer, fp_inps, args)
    print(f"Original Shape - Q: {tensor_q.shape}, K: {tensor_k.shape}, V: {tensor_v.shape}")

    # ---- O (input to out_proj / W_o) activations ------------------------------
    tensor_o = get_out_proj_input_opt(layer, fp_inps, args)
    print(f"Original Shape - O (out_proj input): {tensor_o.shape}")

    # Factorize the feature dimension into num_factors sub-dimensions.
    # Q/K/V all share the same feature dim (head_dim * num_heads).
    # O has the same embed_dim in standard OPT, so we can reuse dim1_factors;
    # but we compute it separately in case the dim differs (e.g. GQA variants).
    dim1_factors_qkv = factorize_dim(tensor_q.shape[2], count=num_factors)
    print(f"Dim-1 factors QKV ({tensor_q.shape[2]} -> {dim1_factors_qkv})")

    dim1_factors_o = factorize_dim(tensor_o.shape[2], count=num_factors)
    print(f"Dim-1 factors O   ({tensor_o.shape[2]} -> {dim1_factors_o})")

    tensor_q = prepare_tensor(tensor_q, dim1_factors_qkv)
    tensor_k = prepare_tensor(tensor_k, dim1_factors_qkv)
    tensor_v = prepare_tensor(tensor_v, dim1_factors_qkv)
    tensor_o = prepare_tensor(tensor_o, dim1_factors_o)
    print(f"Tensor shape after prepare_tensor: Q={tensor_q.shape}, O={tensor_o.shape}")

    # Tucker ranks
    modes_qkv = list(range(1, tensor_q.ndim))
    rank_qkv  = calculate_rank(tensor_q.shape, modes_qkv, compression_ratio, num_heads)
    print(f"Target Ranks QKV: {rank_qkv}")

    modes_o = list(range(1, tensor_o.ndim))
    rank_o  = calculate_rank(tensor_o.shape, modes_o, compression_ratio, num_heads)
    print(f"Target Ranks O  : {rank_o}")

    # Decompose
    named_tensors_qkv = {"Q": tensor_q, "K": tensor_k, "V": tensor_v}
    results = {
        name: _decompose_one(name, tensor, rank_qkv, modes_qkv)
        for name, tensor in named_tensors_qkv.items()
    }
    results["O"] = _decompose_one("O", tensor_o, rank_o, modes_o)

    # Summary
    print("\n--- Summary ---")
    print(f"  Core Q shape    : {results['Q']['core'].shape}")
    print(f"  Factors Q shapes: {[f.shape for f in results['Q']['factors']]}")
    print(f"  Core O shape    : {results['O']['core'].shape}")
    print(f"  Factors O shapes: {[f.shape for f in results['O']['factors']]}")
    for name in ("Q", "K", "V", "O"):
        print(
            f"  {name} | rel_err={results[name]['relative_error']:.6f}"
            f"  comp_ratio={results[name]['compression_ratio']:.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# Standalone test (no real model needed)
# ---------------------------------------------------------------------------

def test_tucker_pipeline(
    nsamples: int = 8,
    seq_len: int = 32,
    hidden_dim: int = 768,
    num_factors: int = 5,
    compression_ratio: float = 0.7,
    seed: int = 42,
):
    """
    Test the full Tucker decomposition pipeline with random Q/K/V tensors,
    mirroring the steps inside tucker_decompose_opt_layer.

    Args:
        nsamples:          Number of samples (first dimension of each tensor).
        seq_len:           Sequence length (second dimension).
        hidden_dim:        Feature dimension to be factorized (e.g. 768).
        num_factors:       Number of sub-dimensions to split hidden_dim into.
        compression_ratio: Target Tucker compression rate in (0, 1].
        seed:              Random seed for reproducibility.

    Returns:
        Same results dict as tucker_decompose_opt_layer:
        {"Q": {...}, "K": {...}, "V": {...}}
    """
    torch.manual_seed(seed)
    print("=" * 60)
    print("  Tucker Pipeline Test  (random tensors)")
    print("=" * 60)

    # Simulate activation tensors: (nsamples, seq_len, hidden_dim)
    # Using torch.rand (uniform [0,1]) to mimic real activations which have
    # non-zero means and concentrated energy in leading singular values.
    # (torch.randn gives zero-mean Gaussian with no low-rank structure, yielding
    #  artificially high reconstruction error on any compression.)
    tensor_q = torch.rand(nsamples, seq_len, hidden_dim)
    tensor_k = torch.rand(nsamples, seq_len, hidden_dim)
    tensor_v = torch.rand(nsamples, seq_len, hidden_dim)
    print(f"Original Shape - Q: {tensor_q.shape}, K: {tensor_k.shape}, V: {tensor_v.shape}")

    # Step 1: factorize the feature dimension
    dim1_factors = factorize_dim(hidden_dim, count=num_factors)
    print(f"Dim-1 factors ({hidden_dim} -> {dim1_factors})")

    # Step 2: reshape tensors
    tensor_q = prepare_tensor(tensor_q, dim1_factors)
    tensor_k = prepare_tensor(tensor_k, dim1_factors)
    tensor_v = prepare_tensor(tensor_v, dim1_factors)
    print(f"Tensor shape after prepare_tensor: {tensor_q.shape}")

    # Step 3: calculate Tucker ranks for the target compression ratio
    modes = list(range(1, tensor_q.ndim))
    rank = calculate_rank(tensor_q.shape, modes, compression_ratio)
    print(f"Target Ranks: {rank}")

    # Step 4: decompose + reconstruct each tensor
    named_tensors = {"Q": tensor_q, "K": tensor_k, "V": tensor_v}
    results = {
        name: _decompose_one(name, tensor, rank, modes)
        for name, tensor in named_tensors.items()
    }

    # Summary
    print("\n--- Summary ---")
    print(f"  Core Q shape    : {results['Q']['core'].shape}")
    print(f"  Factors Q shapes: {[f.shape for f in results['Q']['factors']]}")
    for name in ("Q", "K", "V"):
        print(
            f"  {name} | rel_err={results[name]['relative_error']:.6f}"
            f"  comp_ratio={results[name]['compression_ratio']:.4f}"
        )

    return results


if __name__ == "__main__":
    test_tucker_pipeline(
        nsamples=8,
        seq_len=32,
        hidden_dim=768,
        num_factors=5,
        compression_ratio=0.7,
    )