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


def prepare_feature_tensor(tensor: torch.Tensor, dim_factors: list[int]) -> torch.Tensor:
    """
    Reshape activation features into Tucker feature modes while preserving mode 0.

    Accepted inputs:
      - (batch, seq_len, hidden_dim) -> (batch * seq_len, *dim_factors)
      - (tokens, hidden_dim)         -> (tokens, *dim_factors)
    """
    hidden_dim = tensor.shape[-1]
    if math.prod(dim_factors) != hidden_dim:
        raise ValueError(
            f"Cannot reshape hidden_dim={hidden_dim} with factors={dim_factors} "
            f"(product={math.prod(dim_factors)})"
        )

    return tensor.reshape(-1, *dim_factors)


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

    # return best_ranks
    return [4, 4, 4, 3, 3]


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

import torch
import math


def unfold_tensor(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    """
    Matricize tensor theo mode chỉ định.

    Ví dụ tensor shape (8, 8, 12):
        mode 0 -> (8, 96)
        mode 1 -> (8, 96)
        mode 2 -> (12, 64)
    """
    ndim = tensor.ndim
    order = [mode] + [i for i in range(ndim) if i != mode]
    unfolded = tensor.permute(order).reshape(tensor.shape[mode], -1)
    return unfolded


def fold_matrix(matrix: torch.Tensor, original_shape: tuple, mode: int) -> torch.Tensor:
    """
    Fold matrix đã matricize theo mode về lại tensor gốc.
    """
    ndim = len(original_shape)

    mode_dim = original_shape[mode]
    other_dims = [original_shape[i] for i in range(ndim) if i != mode]

    # shape sau khi unfold theo mode:
    # (mode_dim, dim_others...)
    tensor_perm = matrix.reshape(mode_dim, *other_dims)

    # inverse permutation
    order = [mode] + [i for i in range(ndim) if i != mode]
    inv_order = [0] * ndim
    for i, j in enumerate(order):
        inv_order[j] = i

    tensor = tensor_perm.permute(inv_order)
    return tensor


def truncated_svd_reconstruct(matrix: torch.Tensor, rank: int):
    """
    Reconstruct matrix bằng truncated SVD rank-r.
    """
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    matrix_rec = (U_r * S_r.unsqueeze(0)) @ Vh_r

    return matrix_rec, S


def energy_ratio_from_singular_values(S: torch.Tensor):
    """
    Energy = tổng bình phương singular values.
    """
    energy = S ** 2
    total_energy = energy.sum()

    if total_energy.item() == 0:
        return torch.zeros_like(energy)

    cumulative_energy = torch.cumsum(energy, dim=0) / total_energy
    return cumulative_energy


def rank_for_energy(S: torch.Tensor, target_energy: float):
    """
    Tìm rank nhỏ nhất sao cho giữ được target_energy.
    """
    cumulative_energy = energy_ratio_from_singular_values(S)
    rank = int(torch.searchsorted(cumulative_energy, target_energy).item()) + 1
    return min(rank, S.numel())


def _rank_product_candidates(dims: list[int], target_product: int) -> list[tuple[int, ...]]:
    """
    Enumerate Tucker rank tuples bounded by dims whose product equals target_product.
    """
    candidates = []

    def backtrack(idx: int, remaining: int, current: list[int]):
        if idx == len(dims):
            if remaining == 1:
                candidates.append(tuple(current))
            return

        max_rank = min(dims[idx], remaining)
        for rank in range(1, max_rank + 1):
            if remaining % rank == 0:
                current.append(rank)
                backtrack(idx + 1, remaining // rank, current)
                current.pop()

    backtrack(0, int(target_product), [])
    return candidates


def _mode_cumulative_energies(tensor: torch.Tensor, modes: list[int]) -> dict[int, torch.Tensor]:
    """
    Compute cumulative SVD energy for each requested Tucker mode.
    """
    energies = {}
    tensor_f32 = tensor.to(torch.float32)

    for mode in modes:
        matrix = unfold_tensor(tensor_f32, mode)
        _, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
        energies[mode] = energy_ratio_from_singular_values(singular_values).detach()

    return energies


def select_tucker_ranks_by_energy(
    tensor: torch.Tensor,
    modes: list[int],
    target_product: int,
    logger=None,
) -> dict:
    """
    Select per-mode Tucker ranks whose product equals target_product.

    The selected candidate maximizes the weakest retained mode energy first,
    then the average retained energy. Mode 0 is excluded by passing modes=[1,2,3].
    """
    dims = [int(tensor.shape[m]) for m in modes]
    target_product = int(target_product)
    max_product = math.prod(dims)

    if target_product < 1 or target_product > max_product:
        raise ValueError(
            f"target_product={target_product} is outside valid Tucker product range "
            f"[1, {max_product}] for dims={dims}"
        )

    energies_by_mode = _mode_cumulative_energies(tensor, modes)
    candidates = _rank_product_candidates(dims, target_product)

    exact_product = True
    if not candidates:
        exact_product = False
        feasible_products = sorted(
            {
                math.prod(candidate)
                for product in range(1, max_product + 1)
                for candidate in _rank_product_candidates(dims, product)
            }
        )
        fallback_product = min(feasible_products, key=lambda p: (abs(p - target_product), p > target_product))
        candidates = _rank_product_candidates(dims, fallback_product)
        if logger:
            logger.info(
                f"Tucker QK cannot match rank_kq={target_product} exactly with dims={dims}; "
                f"using nearest feasible product={fallback_product}"
            )
        else:
            print(
                f"  [WARN] Tucker QK cannot match rank_kq={target_product} exactly with dims={dims}; "
                f"using nearest feasible product={fallback_product}"
            )
        target_product = fallback_product

    scored = []
    for candidate in candidates:
        retained = []
        for mode, rank in zip(modes, candidate):
            cumulative = energies_by_mode[mode]
            retained.append(float(cumulative[rank - 1].item()))

        min_energy = min(retained)
        mean_energy = sum(retained) / len(retained)
        balance = _rank_imbalance(dims, list(candidate))
        scored.append((min_energy, mean_energy, -balance, candidate, retained))

    scored.sort(reverse=True)
    min_energy, mean_energy, neg_balance, ranks, retained = scored[0]

    return {
        "ranks": list(ranks),
        "rank_product": math.prod(ranks),
        "exact_product": exact_product,
        "target_product": target_product,
        "modes": modes,
        "dims": dims,
        "retained_energies": retained,
        "min_retained_energy": min_energy,
        "mean_retained_energy": mean_energy,
        "rank_balance": -neg_balance,
    }


def get_tucker_attn_config(args, hidden_dim: int) -> dict:
    """
    Read Tucker-attention options using the repo's existing args style.

    Preferred location:
      args.eigen_attn_params["tucker"] = {...}

    Also accepted:
      args.tucker_attn_params = {...}
    """
    params = {}
    eigen_params = getattr(args, "eigen_attn_params", {}) or {}
    if isinstance(eigen_params.get("tucker"), dict):
        params.update(eigen_params["tucker"])
    if hasattr(args, "tucker_attn_params") and isinstance(args.tucker_attn_params, dict):
        params.update(args.tucker_attn_params)

    factor_dims = (
        params.get("factor_dims")
        or params.get("feature_factors")
        or params.get("tucker_factor_dims")
    )
    if factor_dims is None:
        num_factors = int(params.get("num_factors", 3))
        factor_dims = factorize_dim(hidden_dim, num_factors)
    factor_dims = [int(x) for x in factor_dims]

    if math.prod(factor_dims) != hidden_dim:
        raise ValueError(
            f"Tucker factor_dims={factor_dims} product={math.prod(factor_dims)} "
            f"must equal hidden_dim={hidden_dim}"
        )

    modes = params.get("modes") or params.get("decomposed_modes")
    modes_were_explicit = modes is not None
    if modes is None:
        modes = list(range(1, len(factor_dims) + 1))
    modes = [int(m) for m in modes]

    compress_sample_mode = bool(params.get("compress_sample_mode", False))
    if 0 in modes and not compress_sample_mode:
        modes = [m for m in modes if m != 0]
    if 0 in modes:
        raise NotImplementedError(
            "Tucker attention currently folds only feature-mode projections into weights; "
            "compressing calibration-token mode 0 is not supported."
        )

    return {
        "enabled": bool(params.get("enabled", params.get("use_tucker", False))),
        "factor_dims": factor_dims,
        "modes": modes,
        "initial_threshold": float(
            params.get("initial_threshold", params.get("threshold", eigen_params.get("threshold", 0.98)))
        ),
        "threshold_step": float(params.get("threshold_step", params.get("step", 0.02))),
        "min_threshold": float(params.get("min_threshold", 0.30)),
        "manual_ranks_kq": params.get("manual_ranks_kq", params.get("manual_ranks_qk")),
        "manual_ranks_v": params.get("manual_ranks_v"),
        "log_reconstruction_error": bool(params.get("log_reconstruction_error", False)),
        "materialize_projection": bool(params.get("materialize_projection", True)),
        "preserve_heads": bool(params.get("preserve_heads", True)),
        "modes_were_explicit": modes_were_explicit,
    }


def resolve_tucker_modes_for_opt(cfg: dict, num_heads: int) -> dict:
    """
    Keep OPT head mode uncompressed by default.

    For hidden_dim reshaped as (num_heads, d1, d2), the prepared tensor shape is
    (calibration_tokens, num_heads, d1, d2). Tucker feature modes are therefore
    [1, 2, 3], and preserving heads means decomposing only modes [2, 3].
    """
    cfg = cfg.copy()
    if (
        cfg.get("preserve_heads", True)
        and not cfg.get("modes_were_explicit", False)
        and len(cfg["factor_dims"]) >= 2
    ):
        if cfg["factor_dims"][0] != int(num_heads):
            hidden_dim = math.prod(cfg["factor_dims"])
            if hidden_dim % int(num_heads) != 0:
                raise ValueError(
                    f"Cannot preserve OPT heads for factor_dims={cfg['factor_dims']} "
                    f"and num_heads={num_heads}"
                )
            head_dim = hidden_dim // int(num_heads)
            cfg["factor_dims"] = [int(num_heads)] + factorize_dim(head_dim, len(cfg["factor_dims"]) - 1)
        cfg["modes"] = list(range(2, len(cfg["factor_dims"]) + 1))
    return cfg


def resolve_tucker_modes_for_llama(cfg: dict, num_key_value_heads: int) -> dict:
    """
    Keep the LLaMA KV-head mode uncompressed by default.

    Tucker LLaMA compresses K/V projections only. Their feature dimension is
    num_key_value_heads * head_dim, so preserving heads means reshaping as
    (tokens, num_key_value_heads, ...head_dim factors) and decomposing only the
    per-head feature modes.
    """
    cfg = cfg.copy()
    if not cfg.get("preserve_heads", True):
        raise ValueError("LLaMA Tucker requires preserve_heads=True because RoPE is applied per KV head")
    if cfg.get("modes_were_explicit", False) and 1 in cfg["modes"]:
        raise ValueError("LLaMA Tucker cannot decompose the KV-head mode; remove mode 1 from tucker modes")
    if len(cfg["factor_dims"]) < 2:
        raise ValueError(f"LLaMA Tucker requires at least 2 factor dims, got {cfg['factor_dims']}")

    if cfg["factor_dims"][0] != int(num_key_value_heads):
        kv_dim = math.prod(cfg["factor_dims"])
        if kv_dim % int(num_key_value_heads) != 0:
            raise ValueError(
                f"Cannot preserve LLaMA KV heads for factor_dims={cfg['factor_dims']} "
                f"and num_key_value_heads={num_key_value_heads}"
            )
        head_dim = kv_dim // int(num_key_value_heads)
        cfg["factor_dims"] = [int(num_key_value_heads)] + factorize_dim(head_dim, len(cfg["factor_dims"]) - 1)

    if not cfg.get("modes_were_explicit", False):
        cfg["modes"] = list(range(2, len(cfg["factor_dims"]) + 1))
    return cfg


def _candidate_ranks_at_least(
    dims: list[int],
    min_ranks: list[int],
    divisor: int,
    fixed_product: int = 1,
) -> list[tuple[int, ...]]:
    candidates = []

    def backtrack(idx: int, current: list[int]):
        if idx == len(dims):
            product = math.prod(current) * fixed_product
            if product % divisor == 0:
                candidates.append(tuple(current))
            return

        for rank in range(min_ranks[idx], dims[idx] + 1):
            current.append(rank)
            backtrack(idx + 1, current)
            current.pop()

    backtrack(0, [])
    return candidates


def _select_mode_ranks_from_energies(
    energies_by_mode: dict[int, torch.Tensor],
    modes: list[int],
    dims: list[int],
    threshold: float,
    num_heads: int,
    manual_ranks=None,
    fixed_product: int = 1,
) -> tuple[list[int], list[float]]:
    if manual_ranks is not None:
        ranks = [int(r) for r in manual_ranks]
        if len(ranks) != len(modes):
            raise ValueError(f"manual_ranks length {len(ranks)} must match modes length {len(modes)}")
        for rank, dim in zip(ranks, dims):
            if rank < 1 or rank > dim:
                raise ValueError(f"manual rank {rank} is outside valid range [1, {dim}]")
    else:
        ranks = []
        for mode in modes:
            cumulative = energies_by_mode[mode]
            rank = int(torch.searchsorted(cumulative, float(threshold)).item()) + 1
            ranks.append(min(rank, cumulative.numel()))

    candidates = _candidate_ranks_at_least(dims, ranks, int(num_heads), fixed_product=fixed_product)
    if not candidates:
        raise ValueError(
            f"No Tucker rank tuple >= {ranks} with dims={dims} and fixed_product={fixed_product} "
            f"has latent dimension divisible by num_heads={num_heads}"
        )

    def candidate_key(candidate):
        retained = [float(energies_by_mode[m][r - 1].item()) for m, r in zip(modes, candidate)]
        return (math.prod(candidate), -min(retained), -sum(retained) / len(retained), _rank_imbalance(dims, list(candidate)))

    selected = min(candidates, key=candidate_key)
    retained = [float(energies_by_mode[m][r - 1].item()) for m, r in zip(modes, selected)]
    return list(selected), retained


def materialize_tucker_projection(factors: list[torch.Tensor]) -> torch.Tensor:
    """
    Build the Kronecker/product projection P with shape (hidden_dim, latent_dim).
    """
    if not factors:
        raise ValueError("Cannot materialize Tucker projection from an empty factor list")

    projection = factors[0].contiguous()
    for factor in factors[1:]:
        projection = torch.kron(projection.contiguous(), factor.contiguous())
    return projection.contiguous()


def materialize_headwise_tucker_projection(head_projection_factors: list[list[torch.Tensor]]) -> torch.Tensor:
    """
    Build a block-diagonal projection with one Tucker projection per attention head.

    Each head projection maps head_dim -> per_head_latent_dim. The final matrix
    maps hidden_dim -> num_heads * per_head_latent_dim, matching OPT's head-major
    hidden layout.
    """
    if not head_projection_factors:
        raise ValueError("Cannot materialize headwise Tucker projection without head factors")

    head_projections = [
        materialize_tucker_projection(factors)
        for factors in head_projection_factors
    ]
    return torch.block_diag(*head_projections).contiguous()


def _tucker_reconstruction_error_from_factors(
    tensor: torch.Tensor,
    factors: list[torch.Tensor],
    modes: list[int],
) -> float:
    down_factors = [factor.t().contiguous() for factor in factors]
    core = multi_mode_dot(tensor.to(torch.float32), down_factors, modes=modes)
    reconstructed = multi_mode_dot(core, factors, modes=modes)
    denom = torch.linalg.norm(tensor.to(torch.float32))
    if denom.item() == 0:
        return 0.0
    return float((torch.linalg.norm(tensor.to(torch.float32) - reconstructed) / denom).item())


def _headwise_tucker_reconstruction_error_from_factors(
    prepared: torch.Tensor,
    head_factors: list[list[torch.Tensor]],
    local_modes: list[int],
) -> dict:
    errors = []
    for head_idx, factors in enumerate(head_factors):
        head_tensor = prepared[:, head_idx, ...].to(torch.float32)
        errors.append(_tucker_reconstruction_error_from_factors(head_tensor, factors, local_modes))

    return {
        "max": max(errors) if errors else 0.0,
        "mean": sum(errors) / len(errors) if errors else 0.0,
        "per_head": errors,
    }


def _partial_tucker_factors(
    tensor: torch.Tensor,
    ranks: list[int],
    modes: list[int],
    n_iter_max: int = 200,
) -> list[torch.Tensor]:
    """
    Compute Tucker factors with TensorLy after ranks have been selected.

    Rank selection still uses per-mode SVD energy. This helper then runs
    Partial Tucker/HOOI so the factors are optimized jointly across modes.
    """
    tensor_f32 = tensor.to(torch.float32)
    original_device = tensor_f32.device

    def _run_partial_tucker(input_tensor):
        result = partial_tucker(
            input_tensor,
            rank=ranks,
            modes=modes,
            n_iter_max=n_iter_max,
            verbose=False,
        )
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], (tuple, list)):
            (_, factors), _ = result
        else:
            _, factors = result
        return factors

    try:
        if tensor_f32.is_cuda:
            with torch.amp.autocast('cuda', enabled=False):
                factors = _run_partial_tucker(tensor_f32)
        else:
            factors = _run_partial_tucker(tensor_f32)
    except RuntimeError:
        factors = _run_partial_tucker(tensor_f32.cpu())
        factors = [factor.to(original_device) for factor in factors]

    return [factor.contiguous() for factor in factors]


def _headwise_mode_cumulative_energies(
    prepared: torch.Tensor,
    local_modes: list[int],
) -> list[dict[int, torch.Tensor]]:
    energies = []
    for head_idx in range(int(prepared.shape[1])):
        head_tensor = prepared[:, head_idx, ...]
        energies.append(_mode_cumulative_energies(head_tensor, local_modes))
    return energies


def _select_headwise_mode_ranks_from_energies(
    energies_by_head: list[dict[int, torch.Tensor]],
    local_modes: list[int],
    dims: list[int],
    threshold: float,
    manual_ranks=None,
    fixed_product: int = 1,
) -> tuple[list[int], list[float], list[list[float]]]:
    if manual_ranks is not None:
        min_ranks = [int(r) for r in manual_ranks]
        if len(min_ranks) != len(local_modes):
            raise ValueError(f"manual_ranks length {len(min_ranks)} must match modes length {len(local_modes)}")
        for rank, dim in zip(min_ranks, dims):
            if rank < 1 or rank > dim:
                raise ValueError(f"manual rank {rank} is outside valid range [1, {dim}]")
    else:
        per_head_min_ranks = []
        for energies in energies_by_head:
            ranks = []
            for mode in local_modes:
                cumulative = energies[mode]
                rank = int(torch.searchsorted(cumulative, float(threshold)).item()) + 1
                ranks.append(min(rank, cumulative.numel()))
            per_head_min_ranks.append(ranks)

        min_ranks = [
            max(head_ranks[mode_idx] for head_ranks in per_head_min_ranks)
            for mode_idx in range(len(local_modes))
        ]

    candidates = _candidate_ranks_at_least(dims, min_ranks, divisor=1, fixed_product=fixed_product)
    if not candidates:
        raise ValueError(
            f"No headwise Tucker rank tuple >= {min_ranks} with dims={dims} "
            f"and fixed_product={fixed_product}"
        )

    def retained_for(candidate):
        per_head_retained = []
        flat_retained = []
        for energies in energies_by_head:
            retained = [
                float(energies[mode][rank - 1].item())
                for mode, rank in zip(local_modes, candidate)
            ]
            per_head_retained.append(retained)
            flat_retained.extend(retained)
        return per_head_retained, flat_retained

    def candidate_key(candidate):
        _, flat_retained = retained_for(candidate)
        return (
            math.prod(candidate) * fixed_product,
            -min(flat_retained),
            -sum(flat_retained) / len(flat_retained),
            _rank_imbalance(dims, list(candidate)),
        )

    selected = min(candidates, key=candidate_key)
    per_head_retained, _ = retained_for(selected)
    retained_by_mode_min = [
        min(head_retained[mode_idx] for head_retained in per_head_retained)
        for mode_idx in range(len(local_modes))
    ]
    return list(selected), retained_by_mode_min, per_head_retained


def build_headwise_tucker_projection_from_tensor(
    tensor: torch.Tensor,
    factor_dims: list[int],
    modes: list[int],
    threshold: float,
    num_heads: int,
    manual_ranks=None,
    log_reconstruction_error: bool = False,
) -> dict:
    """
    Build a Tucker projection with independent factors for each attention head.

    This path assumes factor_dims[0] is the head mode and that the head mode is
    preserved. Ranks are chosen as the smallest common per-head rank tuple that
    reaches the requested energy threshold for every head, mirroring
    EigenAttention's max-rank-across-heads behavior.
    """
    if not factor_dims or int(factor_dims[0]) != int(num_heads):
        raise ValueError(
            f"Headwise Tucker requires factor_dims[0] == num_heads, got "
            f"factor_dims={factor_dims}, num_heads={num_heads}"
        )
    if 1 in modes:
        raise ValueError("Headwise Tucker expects the OPT head mode to be preserved, but mode 1 was requested")

    prepared = prepare_feature_tensor(tensor, factor_dims).to(torch.float32)
    head_feature_modes = list(range(2, len(factor_dims) + 1))
    local_feature_modes = list(range(1, len(factor_dims)))
    local_modes = [mode - 1 for mode in modes]
    if any(mode not in local_feature_modes for mode in local_modes):
        raise ValueError(
            f"Invalid headwise Tucker modes={modes} for factor_dims={factor_dims}; "
            f"expected feature modes from {head_feature_modes}"
        )

    dims = [int(prepared.shape[mode]) for mode in modes]
    undecomposed_modes = [mode for mode in head_feature_modes if mode not in modes]
    fixed_product = math.prod(int(prepared.shape[mode]) for mode in undecomposed_modes)
    energies_by_head = _headwise_mode_cumulative_energies(prepared, local_modes)
    ranks, retained, head_retained = _select_headwise_mode_ranks_from_energies(
        energies_by_head,
        local_modes,
        dims,
        threshold,
        manual_ranks=manual_ranks,
        fixed_product=fixed_product,
    )

    head_factors = []
    head_projection_factors = []
    for head_idx in range(num_heads):
        head_tensor = prepared[:, head_idx, ...]
        factors = _partial_tucker_factors(head_tensor, ranks, local_modes)
        factor_by_local_mode = {
            mode: factor
            for mode, factor in zip(local_modes, factors)
        }
        head_factors.append(factors)

        projection_factors = []
        for local_mode in local_feature_modes:
            if local_mode in factor_by_local_mode:
                projection_factors.append(factor_by_local_mode[local_mode])
            else:
                dim = int(head_tensor.shape[local_mode])
                projection_factors.append(torch.eye(dim, dtype=prepared.dtype, device=prepared.device))
        head_projection_factors.append(projection_factors)

    projection = materialize_headwise_tucker_projection(head_projection_factors)
    reconstruction_error = None
    if log_reconstruction_error:
        reconstruction_error = _headwise_tucker_reconstruction_error_from_factors(
            prepared,
            head_factors,
            local_modes,
        )

    latent_dim = int(projection.shape[1])
    per_head_latent_dim = latent_dim // int(num_heads)
    return {
        "projection": projection,
        "factors": head_factors,
        "projection_factors": head_projection_factors,
        "ranks": ranks,
        "latent_dim": latent_dim,
        "per_head_latent_dim": per_head_latent_dim,
        "retained_energies": retained,
        "head_retained_energies": head_retained,
        "min_retained_energy": min(retained),
        "mean_retained_energy": sum(sum(x) for x in head_retained) / sum(len(x) for x in head_retained),
        "reconstruction_error": reconstruction_error,
        "prepared_shape": tuple(prepared.shape),
        "modes": modes,
        "local_modes": local_modes,
        "factor_dims": factor_dims,
        "headwise": True,
        "num_heads": int(num_heads),
    }


def build_tucker_projection_from_tensor(
    tensor: torch.Tensor,
    factor_dims: list[int],
    modes: list[int],
    threshold: float,
    num_heads: int,
    manual_ranks=None,
    log_reconstruction_error: bool = False,
) -> dict:
    """
    Build Tucker mode factors and a materialized hidden_dim -> latent_dim projection.
    """
    prepared = prepare_feature_tensor(tensor, factor_dims).to(torch.float32)
    dims = [int(prepared.shape[m]) for m in modes]
    feature_modes = list(range(1, len(factor_dims) + 1))
    undecomposed_modes = [m for m in feature_modes if m not in modes]
    fixed_product = math.prod(int(prepared.shape[m]) for m in undecomposed_modes)
    energies_by_mode = _mode_cumulative_energies(prepared, modes)
    ranks, retained = _select_mode_ranks_from_energies(
        energies_by_mode,
        modes,
        dims,
        threshold,
        num_heads,
        manual_ranks=manual_ranks,
        fixed_product=fixed_product,
    )

    factors = _partial_tucker_factors(prepared, ranks, modes)
    factor_by_mode = {}
    for mode, factor in zip(modes, factors):
        factor_by_mode[mode] = factor

    projection_factors = []
    for mode in feature_modes:
        if mode in factor_by_mode:
            projection_factors.append(factor_by_mode[mode])
        else:
            dim = int(prepared.shape[mode])
            projection_factors.append(torch.eye(dim, dtype=prepared.dtype, device=prepared.device))

    projection = materialize_tucker_projection(projection_factors)
    reconstruction_error = None
    if log_reconstruction_error:
        reconstruction_error = _tucker_reconstruction_error_from_factors(prepared, factors, modes)

    latent_dim = int(projection.shape[1])
    return {
        "projection": projection,
        "factors": factors,
        "projection_factors": projection_factors,
        "ranks": ranks,
        "latent_dim": latent_dim,
        "retained_energies": retained,
        "min_retained_energy": min(retained),
        "mean_retained_energy": sum(retained) / len(retained),
        "reconstruction_error": reconstruction_error,
        "prepared_shape": tuple(prepared.shape),
        "modes": modes,
        "factor_dims": factor_dims,
        "headwise": False,
    }


@torch.no_grad()
def get_opt_tucker_activations(layer, fp_inps, args):
    from decompose.eigen_attn_utils import get_kqv_opt

    tensor_k, tensor_q, tensor_v = get_kqv_opt(layer, fp_inps, args)
    return tensor_k, tensor_q, tensor_v


def build_opt_tucker_projection_config(
    tensor_k: torch.Tensor,
    tensor_q: torch.Tensor,
    tensor_v: torch.Tensor,
    args,
    num_heads: int,
    threshold: float,
) -> dict:
    hidden_dim = int(tensor_q.shape[-1])
    cfg = get_tucker_attn_config(args, hidden_dim)
    cfg = resolve_tucker_modes_for_opt(cfg, num_heads)

    flat_k = tensor_k.reshape(-1, hidden_dim)
    flat_q = tensor_q.reshape(-1, hidden_dim)
    flat_v = tensor_v.reshape(-1, hidden_dim)
    flat_kq = torch.cat([flat_k, flat_q], dim=0)

    use_headwise = (
        cfg.get("preserve_heads", True)
        and cfg["factor_dims"][0] == int(num_heads)
        and 1 not in cfg["modes"]
    )
    projection_builder = (
        build_headwise_tucker_projection_from_tensor
        if use_headwise
        else build_tucker_projection_from_tensor
    )

    qk = projection_builder(
        flat_kq,
        cfg["factor_dims"],
        cfg["modes"],
        threshold,
        num_heads,
        manual_ranks=cfg["manual_ranks_kq"],
        log_reconstruction_error=cfg["log_reconstruction_error"],
    )
    v = projection_builder(
        flat_v,
        cfg["factor_dims"],
        cfg["modes"],
        threshold,
        num_heads,
        manual_ranks=cfg["manual_ranks_v"],
        log_reconstruction_error=cfg["log_reconstruction_error"],
    )

    return {
        "threshold": float(threshold),
        "factor_dims": cfg["factor_dims"],
        "modes": cfg["modes"],
        "headwise": use_headwise,
        "qk": qk,
        "v": v,
    }


@torch.no_grad()
def get_llama_tucker_activations(layer, fp_inps, args, attention_mask, position_ids):
    from decompose.eigen_attn_utils import get_kqv_llama

    tensor_k, tensor_q, tensor_v = get_kqv_llama(layer, fp_inps, args, attention_mask, position_ids)
    return tensor_k, tensor_q, tensor_v


def build_llama_tucker_projection_config(
    tensor_k: torch.Tensor,
    tensor_q: torch.Tensor,
    tensor_v: torch.Tensor,
    args,
    num_key_value_heads: int,
    threshold: float,
) -> dict:
    kv_dim = int(tensor_k.shape[-1])
    if int(tensor_v.shape[-1]) != kv_dim:
        raise ValueError(f"Expected LLaMA K/V dims to match, got K={tensor_k.shape[-1]} V={tensor_v.shape[-1]}")

    cfg = get_tucker_attn_config(args, kv_dim)
    cfg = resolve_tucker_modes_for_llama(cfg, num_key_value_heads)

    flat_k = tensor_k.reshape(-1, kv_dim)
    flat_v = tensor_v.reshape(-1, kv_dim)

    k = build_headwise_tucker_projection_from_tensor(
        flat_k,
        cfg["factor_dims"],
        cfg["modes"],
        threshold,
        num_key_value_heads,
        manual_ranks=cfg["manual_ranks_kq"],
        log_reconstruction_error=cfg["log_reconstruction_error"],
    )
    v = build_headwise_tucker_projection_from_tensor(
        flat_v,
        cfg["factor_dims"],
        cfg["modes"],
        threshold,
        num_key_value_heads,
        manual_ranks=cfg["manual_ranks_v"],
        log_reconstruction_error=cfg["log_reconstruction_error"],
    )

    return {
        "threshold": float(threshold),
        "factor_dims": cfg["factor_dims"],
        "modes": cfg["modes"],
        "headwise": True,
        "qk": k,
        "k": k,
        "v": v,
    }


def tucker_attention_param_ratio(embed_dim: int, qk_latent_dim: int, v_latent_dim: int) -> float:
    original = 4 * embed_dim * embed_dim
    compressed = (2 * qk_latent_dim * embed_dim) + (2 * v_latent_dim * embed_dim)
    return compressed / original


def llama_tucker_attention_param_ratio(
    hidden_size: int,
    kv_dim: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    k_latent_dim: int,
    v_latent_dim: int,
) -> float:
    original = (2 * hidden_size * hidden_size) + (2 * hidden_size * kv_dim)
    k_per_kv_head = k_latent_dim // num_key_value_heads
    v_per_kv_head = v_latent_dim // num_key_value_heads
    compressed = hidden_size * hidden_size
    compressed += hidden_size * k_latent_dim
    compressed += num_key_value_heads * kv_dim // num_key_value_heads * k_per_kv_head
    compressed += hidden_size * v_latent_dim
    compressed += hidden_size * (num_attention_heads * v_per_kv_head)
    return compressed / original


def tucker_analyze_opt_kq(
    layer,
    fp_inps,
    args,
    num_heads,
    layer_id,
    rank_kq,
    feature_factors=(8, 8, 12),
    logger=None,
):
    """
    Analyze Tucker compression on the same concatenated K/Q activation used by EigenAttention.

    Shape flow:
      K,Q: (chunks, seq_len, hidden_dim)
      QK : (2 * chunks * seq_len, hidden_dim)
      Tucker input: (2 * chunks * seq_len, *feature_factors)

    Partial Tucker is applied only to feature modes [1, 2, 3].
    """
    from decompose.eigen_attn_utils import get_kqv_opt

    rank_kq = int(rank_kq.item() if torch.is_tensor(rank_kq) else rank_kq)
    tensor_k, tensor_q, _ = get_kqv_opt(layer, fp_inps, args)

    qk = torch.cat(
        [
            tensor_k.reshape(-1, tensor_k.shape[-1]),
            tensor_q.reshape(-1, tensor_q.shape[-1]),
        ],
        dim=0,
    )
    qk = prepare_feature_tensor(qk, list(feature_factors))
    modes = list(range(1, qk.ndim))

    rank_info = select_tucker_ranks_by_energy(qk, modes, rank_kq, logger=logger)
    ranks = rank_info["ranks"]

    if logger:
        logger.info(
            f"layer {layer_id} Tucker-QK input_shape={tuple(qk.shape)} "
            f"feature_factors={tuple(feature_factors)} modes={modes}"
        )
        logger.info(
            f"layer {layer_id} Tucker-QK ranks={ranks} product={rank_info['rank_product']} "
            f"rank_kq={rank_kq} exact_product={rank_info['exact_product']} "
            f"mode_energy={rank_info['retained_energies']} "
            f"min_energy={rank_info['min_retained_energy']:.6f} "
            f"mean_energy={rank_info['mean_retained_energy']:.6f}"
        )
    else:
        print(
            f"layer {layer_id} Tucker-QK ranks={ranks} product={rank_info['rank_product']} "
            f"rank_kq={rank_kq} exact_product={rank_info['exact_product']}"
        )
        print(
            f"  mode_energy={rank_info['retained_energies']} "
            f"min={rank_info['min_retained_energy']:.6f} "
            f"mean={rank_info['mean_retained_energy']:.6f}"
        )

    core, factors, compression_ratio = decompose(qk, rank=ranks, modes=modes)
    relative_error = reconstruct(qk, core, factors, modes=modes)

    if logger:
        logger.info(
            f"layer {layer_id} Tucker-QK rel_error={relative_error:.6f} "
            f"compression_ratio={compression_ratio:.4f}"
        )

    rank_info.update(
        {
            "core": core,
            "factors": factors,
            "compression_ratio": compression_ratio,
            "relative_error": relative_error,
        }
    )
    return rank_info


def inspect_single_token_mode_rank(
    tensor_q: torch.Tensor,
    sample_idx: int = 0,
    token_idx: int = 0,
    reshape_shape: tuple = (8, 8, 12),
    target_energies=(0.90, 0.95, 0.99),
    manual_ranks=None,
    verbose: bool = True,
):
    """
    Kiểm tra low-rank structure của 1 token trong tensor_q.

    Args:
        tensor_q:
            Tensor Q shape (batch/nsamples, seq_len, hidden_dim),
            ví dụ (2, 2048, 768).

        sample_idx:
            Chọn sample nào trong batch/nsamples.

        token_idx:
            Chọn token nào trong seq_len.

        reshape_shape:
            Shape để reshape hidden_dim.
            Với OPT-125M hidden_dim = 768, có thể dùng (8, 8, 12).

        target_energies:
            Các mức energy muốn kiểm tra, ví dụ 90%, 95%, 99%.

        manual_ranks:
            Nếu muốn kiểm tra thêm rank cụ thể.
            Ví dụ manual_ranks=[1, 2, 4, 6, 8].
            Nếu None thì chỉ dùng rank theo target_energies.

    Returns:
        Dict chứa thông tin từng mode:
            - unfolded_shape
            - singular_values
            - cumulative_energy
            - energy_ranks
            - reconstructions
            - relative_errors
    """
    if tensor_q.ndim != 3:
        raise ValueError(
            f"Expected tensor_q shape (batch, seq_len, hidden_dim), "
            f"but got {tuple(tensor_q.shape)}"
        )

    batch_size, seq_len, hidden_dim = tensor_q.shape

    if sample_idx < 0 or sample_idx >= batch_size:
        raise ValueError(f"sample_idx={sample_idx} out of range [0, {batch_size - 1}]")

    if token_idx < 0 or token_idx >= seq_len:
        raise ValueError(f"token_idx={token_idx} out of range [0, {seq_len - 1}]")

    if math.prod(reshape_shape) != hidden_dim:
        raise ValueError(
            f"Cannot reshape hidden_dim={hidden_dim} into {reshape_shape}, "
            f"because product={math.prod(reshape_shape)}"
        )

    # 1. Lấy 1 token: (768,)
    token_vec = tensor_q[sample_idx, token_idx].detach()

    # Nên đưa về float32 để SVD ổn định hơn nếu tensor đang là fp16/bf16
    token_vec = token_vec.to(torch.float32)

    # 2. Reshape: (768,) -> (8, 8, 12)
    token_tensor = token_vec.reshape(*reshape_shape)

    if manual_ranks is None:
        manual_ranks = []

    results = {
        "sample_idx": sample_idx,
        "token_idx": token_idx,
        "token_vec_shape": tuple(token_vec.shape),
        "token_tensor_shape": tuple(token_tensor.shape),
        "modes": {},
    }

    if verbose:
        print("\n=== Single Token Mode-wise Rank Inspection ===")
        print(f"tensor_q shape       : {tuple(tensor_q.shape)}")
        print(f"selected token       : sample_idx={sample_idx}, token_idx={token_idx}")
        print(f"token vector shape   : {tuple(token_vec.shape)}")
        print(f"reshaped tensor shape: {tuple(token_tensor.shape)}")

    # 3. Matricize theo từng mode
    for mode in range(token_tensor.ndim):
        matrix = unfold_tensor(token_tensor, mode)

        # SVD full
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        cumulative_energy = energy_ratio_from_singular_values(S)

        # Các rank theo target energy
        energy_ranks = {
            float(e): rank_for_energy(S, float(e))
            for e in target_energies
        }

        # Union giữa rank theo energy và manual ranks
        ranks_to_test = sorted(
            set(energy_ranks.values()).union(set(manual_ranks))
        )

        # Rank không được vượt quá min(matrix.shape)
        max_rank = min(matrix.shape)
        ranks_to_test = [r for r in ranks_to_test if 1 <= r <= max_rank]

        reconstructions = {}
        relative_errors = {}
        retained_energies = {}

        for r in ranks_to_test:
            matrix_rec = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]
            tensor_rec = fold_matrix(matrix_rec, token_tensor.shape, mode)

            rel_error = torch.linalg.norm(token_tensor - tensor_rec) / torch.linalg.norm(token_tensor)

            reconstructions[r] = tensor_rec
            relative_errors[r] = float(rel_error.item())
            retained_energies[r] = float(cumulative_energy[r - 1].item())

        results["modes"][mode] = {
            "unfolded_shape": tuple(matrix.shape),
            "singular_values": S.detach().cpu(),
            "cumulative_energy": cumulative_energy.detach().cpu(),
            "energy_ranks": energy_ranks,
            "tested_ranks": ranks_to_test,
            "retained_energies": retained_energies,
            "relative_errors": relative_errors,
            "reconstructions": reconstructions,
        }

        if verbose:
            print(f"\n--- Mode {mode} unfolding ---")
            print(f"matrix shape: {tuple(matrix.shape)}")
            print(f"max possible rank: {max_rank}")

            print("\nRank needed for target energy:")
            for e, r in energy_ranks.items():
                print(f"  energy >= {e:.2f}: rank {r}")

            print("\nReconstruction by truncated SVD:")
            for r in ranks_to_test:
                print(
                    f"  rank={r:<3d} | "
                    f"energy={retained_energies[r]:.6f} | "
                    f"rel_error={relative_errors[r]:.6f}"
                )

    return results

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
    inspect_result = inspect_single_token_mode_rank(
        tensor_q=tensor_q,
        sample_idx=0,
        token_idx=0,
        reshape_shape=(8, 8, 12),
        target_energies=(0.90, 0.95, 0.99),
        manual_ranks=[1, 2, 4, 6, 8],
        verbose=True,
    )

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
