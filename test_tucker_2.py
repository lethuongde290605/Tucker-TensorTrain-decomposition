import pytest
import math
import numpy as np
import tensorly as tl
import torch

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
    N = tensor.shape[0]
    result = tensor.reshape(N, *dim1_factors)
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

def calculate_rank(tensor_shape: tuple, modes: list[int], target_ratio: float) -> list[int]:
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

        # Stop if this step does not improve the rate
        if new_rate >= best_rate:
            break

        ranks = new_ranks
        best_rate = new_rate
        best_ranks = ranks.copy()

        if best_rate <= target_ratio:
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
    (core, factors), rec_errors = partial_tucker(
        tensor, rank=rank, modes=modes, n_iter_max=200, verbose=True
    )

    print("core shape: ", core.shape)
    print("factor shapes:")
    for f in factors:
        print(" ", f.shape)

    # Compression stats
    original_size = math.prod(tensor.shape)
    compressed_size = sum(f.size for f in factors) + core.size
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

    tensor_t = torch.from_numpy(tensor)
    reconstructed_t = torch.from_numpy(reconstructed_tensor)
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


if __name__ == "__main__":

    rng = tl.check_random_state(1234)
    rand_tensor = tl.tensor(rng.random_sample((1024, 768)))

    # Factorize the second dimension (768) into 5 sub-dims
    dim1_factors = factorize_dim(rand_tensor.shape[1], count=5)
    print(f"Dim-1 factors (768 -> {dim1_factors})")

    tensor = prepare_tensor(rand_tensor, dim1_factors)
    print(f"Tensor shape after prepare_tensor: {tensor.shape}")

    modes = list(range(1, tensor.ndim))

    errors = []
    compressions = []

    for i in np.arange(0, 1, 0.1):
        rank = calculate_rank(tensor.shape, modes, target_ratio=i)
        print(f"\n{'='*50}")
        print(f"Compression Ratio: {i}")
        print(f"Rank: {rank}")
        core, factors, comp = decompose(tensor, rank=rank, modes=modes)
        rel_err = reconstruct(tensor, core, factors, modes=modes)

        errors.append(rel_err)
        compressions.append(comp)

    import matplotlib.pyplot as plt

    plt.scatter(compressions, errors, marker="o")
    plt.xlabel("Compression Ratio")
    plt.ylabel("Relative Error")
    plt.title("Tucker Compression Trade-off")
    plt.grid(True)
    plt.show()