import math
import torch
import tensorly as tl
from tensorly.decomposition._tt import tensor_train_matrix
from tensorly.tt_matrix import tt_matrix_to_tensor

tl.set_backend('pytorch')  # TensorLy must use the PyTorch backend since all tensors are torch.Tensor


# ---------------------------------------------------------------------------
# Utility: factorize a single dimension into `count` nearly-equal factors
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
# Core reshape helpers
# ---------------------------------------------------------------------------

def prepare_tensor(t: torch.Tensor, row_factors: list[int], col_factors: list[int]) -> torch.Tensor:
    """
    Reshape a 2-D weight matrix into a 2k-D tensor for TT-matrix decomposition.

    tensor_train_matrix expects shape (*row_factors, *col_factors):
      - first k dims = row sub-dimensions
      - last  k dims = col sub-dimensions

    No interleaving is needed; the matrix structure is preserved in each core.

    Args:
        t:           2-D tensor of shape (R, C).
        row_factors: Factorization of R (len == k).
        col_factors: Factorization of C (len == k).

    Returns:
        Tensor of shape (*row_factors, *col_factors).
    """
    
    return t.view(*row_factors, *col_factors)


def reverse_prepare_tensor(t: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """
    Reshape the reconstructed tensor back to the original 2-D weight matrix.

    tt_matrix_to_tensor returns a tensor of shape (*row_factors, *col_factors),
    so a single reshape is sufficient.

    Args:
        t:              Tensor of shape (*row_factors, *col_factors).
        original_shape: Target 2-D shape (R, C).

    Returns:
        2-D tensor of shape `original_shape`.
    """
    return t.reshape(*original_shape)


# ---------------------------------------------------------------------------
# TT-matrix decomposition helper
# ---------------------------------------------------------------------------

def run_tensor_train_matrix(tensor: torch.Tensor, target_ranks: list[int]) -> list[torch.Tensor]:
    """
    Run TensorLy tensor_train_matrix decomposition in float32 precision.

    Input tensor must have shape (*row_factors, *col_factors).
    Each output core has shape (r_i, m_i, n_i, r_{i+1}).

    Args:
        tensor:       2k-D tensor of shape (*row_factors, *col_factors).
        target_ranks: TT-rank list (length == k + 1, first and last == 1).

    Returns:
        List of TT-matrix core tensors shaped (r_i, m_i, n_i, r_{i+1}).
    """
    tensor = tensor.to(torch.float32)
    with torch.cuda.amp.autocast(enabled=False):
        tt_mat = tensor_train_matrix(tensor, rank=target_ranks)
    return list(tt_mat.factors)


def compute_max_ranks(row_factors: list[int], col_factors: list[int]) -> list[int]:
    """
    Compute the maximum TT-matrix ranks for mode pairs (m_i, n_i).

    max_rank[i] = min(prod(m_j * n_j for j < i),
                      prod(m_j * n_j for j >= i))

    Args:
        row_factors: Row sub-dimensions per mode.
        col_factors: Col sub-dimensions per mode.

    Returns:
        List of k+1 integers (first and last are always 1).
    """
    mode_sizes = [m * n for m, n in zip(row_factors, col_factors)]
    k = len(mode_sizes)
    max_ranks = [1]
    for i in range(1, k):
        left  = math.prod(mode_sizes[:i])
        right = math.prod(mode_sizes[i:])
        max_ranks.append(min(left, right))
    max_ranks.append(1)
    return max_ranks


# ---------------------------------------------------------------------------
# Main decompose / reconstruct
# ---------------------------------------------------------------------------

def tensor_train_decompose(
    tensor: torch.Tensor,
    num_factors: int = 5,
    ranks: list[int] | None = None,
):
    """
    Perform TT-matrix decomposition on a 2-D weight matrix.

    Args:
        tensor:      2-D weight matrix of shape (R, C).
        num_factors: Number of TT-matrix modes (k). Both R and C are each split
                     into `num_factors` sub-dimensions.
        ranks:       TT-ranks, length k+1, first/last == 1.
                     None -> exact decomposition (uses max ranks).

    Returns:
        factors:  List of k TT-matrix cores shaped (r_i, m_i, n_i, r_{i+1}).
        metadata: Dict with keys row_factors, col_factors, original_shape.
    """
    assert tensor.ndim == 2, f"Expected 2-D tensor, got shape {tensor.shape}"
    R, C = tensor.shape
    print(f"Original Shape: {tensor.shape}")

    row_factors = factorize_dim(R, num_factors)
    col_factors = factorize_dim(C, num_factors)

    print(f"Row factors   : {row_factors}  (product={math.prod(row_factors)})")
    print(f"Col factors   : {col_factors}  (product={math.prod(col_factors)})")

    prepared = prepare_tensor(tensor, row_factors, col_factors)
    print(f"Reshaped for TT-matrix: {list(prepared.shape)}")

    max_ranks = compute_max_ranks(row_factors, col_factors)
    expected_rank_len = num_factors + 1

    if ranks is None:
        target_ranks = max_ranks

    elif isinstance(ranks, list) and len(ranks) == expected_rank_len:
        target_ranks = ranks

    else:
        raise ValueError(
            f"ranks must be None (exact) or a list of {expected_rank_len} elements "
            f"[1, r1, ..., r{num_factors - 1}, 1]. Max ranks: {max_ranks}"
        )

    print(f"Max Ranks              : {max_ranks}")
    print(f"Target Ranks (TT-Ranks): {target_ranks}")

    factors = run_tensor_train_matrix(prepared, target_ranks)
    # Cores are already (r_i, m_i, n_i, r_{i+1}) — no reshape needed
    print(f"Factor shapes: {[list(f.shape) for f in factors]}")

    metadata = {
        "row_factors": row_factors,
        "col_factors": col_factors,
        "original_shape": (R, C),
    }
    return factors, metadata


def tensor_train_decompose_bias(
    tensor: torch.Tensor,
    num_factors: int = 5,
    ranks: list[int] | None = None,
):
    """
    Perform TT-matrix decomposition on a 1-D bias vector.

    The bias vector of shape (R,) is treated as a TT-matrix with col_factors
    all equal to 1, so each core has shape (r_i, m_i, 1, r_{i+1}).

    Args:
        tensor:      1-D bias vector of shape (R,).
        num_factors: Number of TT-matrix modes (k). R is split into
                     `num_factors` sub-dimensions.
        ranks:       TT-ranks, length k+1, first/last == 1.
                     None -> exact decomposition (uses max ranks).

    Returns:
        factors:  List of k TT-matrix cores shaped (r_i, m_i, 1, r_{i+1}).
        metadata: Dict with keys row_factors, col_factors, original_shape.
    """
    assert tensor.ndim == 1, f"Expected 1-D tensor, got shape {tensor.shape}"
    R = tensor.shape[0]
    print(f"Original Shape: {tensor.shape}")

    # Bias has no column dimension: treat each col sub-dim as 1
    row_factors = factorize_dim(R, num_factors)
    col_factors = [1] * num_factors

    print(f"Row factors   : {row_factors}  (product={math.prod(row_factors)})")
    print(f"Col factors   : {col_factors}  (product={math.prod(col_factors)})")

    # Reshape (R,) -> (*row_factors, 1, 1, ..., 1)
    prepared = tensor.view(*row_factors, *col_factors)
    print(f"Reshaped for TT-matrix: {list(prepared.shape)}")

    max_ranks = compute_max_ranks(row_factors, col_factors)
    expected_rank_len = num_factors + 1

    if ranks is None:
        target_ranks = max_ranks

    elif isinstance(ranks, list) and len(ranks) == expected_rank_len:
        target_ranks = ranks

    else:
        raise ValueError(
            f"ranks must be None (exact) or a list of {expected_rank_len} elements "
            f"[1, r1, ..., r{num_factors - 1}, 1]. Max ranks: {max_ranks}"
        )

    print(f"Max Ranks              : {max_ranks}")
    print(f"Target Ranks (TT-Ranks): {target_ranks}")

    factors = run_tensor_train_matrix(prepared, target_ranks)
    # Cores are already (r_i, m_i, n_i, r_{i+1}) — no reshape needed
    print(f"Factor shapes: {[list(f.shape) for f in factors]}")

    metadata = {
        "row_factors": row_factors,
        "col_factors": col_factors,
        "original_shape": (R,),
    }
    return factors, metadata

def tensor_train_reconstruct(
    factors: list[torch.Tensor],
    metadata: dict,
) -> torch.Tensor:
    """
    Reconstruct the original 2-D weight matrix from TT-matrix factors.

    Args:
        factors:  TT-matrix cores shaped (r_i, m_i, n_i, r_{i+1}).
        metadata: Dict returned by tensor_train_decompose
                  (keys: row_factors, col_factors, original_shape).

    Returns:
        Reconstructed 2-D tensor of shape `metadata['original_shape']`.
    """
    original_shape = metadata["original_shape"]
    # tt_matrix_to_tensor returns (*row_factors, *col_factors)
    reconstructed = tt_matrix_to_tensor(factors)
    return reverse_prepare_tensor(reconstructed, original_shape)
