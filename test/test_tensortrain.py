import math
import tensorly as tl
from tensorly.decomposition._tt import tensor_train_matrix
from tensorly.tt_matrix import tt_matrix_to_tensor
import torch
import time

tl.set_backend('pytorch')

# Simulated weight matrix dimensions (hidden_dim x hidden_dim)
HIDDEN_DIM = 768


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
      - first k dims  = row sub-dimensions
      - last  k dims  = col sub-dimensions

    No interleaving is needed; the matrix structure is preserved in each core.

    Args:
        t:           2-D tensor of shape (R, C).
        row_factors: Factorization of R (len == k).
        col_factors: Factorization of C (len == k).

    Returns:
        Tensor of shape (*row_factors, *col_factors).
    """
    assert len(row_factors) == len(col_factors), \
        "row_factors and col_factors must have the same length"
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tensor_train_roundtrip(tensor_shape=(HIDDEN_DIM, HIDDEN_DIM), num_factors=5, ranks=None):
    """
    Create a weight matrix of given shape, run TT-matrix decompose then
    reconstruct, and report the reconstruction error.

    Args:
        tensor_shape: Shape of the test weight matrix (R, C).
        num_factors:  Number of TT-matrix modes.
        ranks:        TT-ranks (None = exact).
    """
    print("=" * 60)
    print("  Tensor-Train Matrix Decomposition Round-trip Test")
    print("=" * 60)
    print(f"Weight matrix shape : {tensor_shape}")
    print()

    torch.manual_seed(42)
    tensor = torch.randn(*tensor_shape)
    print("Original value (first row):", tensor[0, :5], "...")

    # Decompose
    t0 = time.time()
    factors, metadata = tensor_train_decompose(
        tensor, num_factors=num_factors, ranks=ranks
    )
    t1 = time.time()
    print(f"\nTT decomposition took {t1 - t0:.4f} s")

    # Reconstruct
    t2 = time.time()
    reconstructed = tensor_train_reconstruct(factors, metadata)
    t3 = time.time()
    print(f"Reconstruction took   {t3 - t2:.4f} s")

    print("Reconstructed value (first row):", reconstructed[0, :5], "...")

    # Error metrics
    diff = tensor - reconstructed
    frob_error = torch.norm(diff).item()
    frob_orig  = torch.norm(tensor).item()
    relative_error = frob_error / frob_orig
    mse = torch.mean(diff ** 2).item()
    mae = torch.mean(torch.abs(diff)).item()

    print()
    print("--- Error Metrics ---")
    print(f"  Frobenius norm (original) : {frob_orig:.6f}")
    print(f"  Frobenius norm (error)    : {frob_error:.6f}")
    print(f"  Relative error            : {relative_error:.6f}  ({relative_error * 100:.4f} %)")
    print(f"  MSE                       : {mse:.8f}")
    print(f"  MAE                       : {mae:.8f}")

    # Compression ratio
    original_size   = tensor.numel()
    compressed_size = sum(f.numel() for f in factors)
    compression_ratio = compressed_size / original_size

    print()
    print("--- Compression ---")
    print(f"  Original size    : {original_size} params")
    print(f"  Compressed size  : {compressed_size} params")
    print(f"  Compression ratio: {compression_ratio:.4f}  ({compression_ratio * 100:.2f} %)")

    return relative_error, compression_ratio


def test_tensor_train_various_ranks(tensor_shape=(HIDDEN_DIM, HIDDEN_DIM), num_factors=5):
    """
    Test TT-matrix decomposition with different rank configurations
    to see the trade-off between compression and accuracy.
    """
    print("\n" + "=" * 60)
    print("  Testing various TT-matrix rank configurations")
    print("=" * 60)

    row_factors = factorize_dim(tensor_shape[0], num_factors)
    col_factors = factorize_dim(tensor_shape[1], num_factors)
    max_ranks   = compute_max_ranks(row_factors, col_factors)

    print(f"Tensor shape  : {tensor_shape}")
    print(f"Row factors   : {row_factors}")
    print(f"Col factors   : {col_factors}")
    print(f"Max TT-ranks  : {max_ranks}")

    k = len(row_factors)

    rank_configs = [
        ("Low",      [1] + [4]  * (k - 1) + [1]),
        ("Mid",      [1] + [min(36,  max_ranks[i]) for i in range(1, k)] + [1]),
        ("High",     [1] + [min(64,  max_ranks[i]) for i in range(1, k)] + [1]),
        ("Example",  [1, 36, 64, 64, 36, 1]),   # from example code
        ("Max",      max_ranks),
    ]

    results = []
    for name, r in rank_configs:
        print(f"\n{'─' * 50}")
        print(f"Config: {name}  ranks={r}")
        print(f"{'─' * 50}")
        rel_err, comp_ratio = test_tensor_train_roundtrip(
            tensor_shape=tensor_shape, num_factors=num_factors, ranks=r
        )
        results.append((name, r, rel_err, comp_ratio))

    # Summary table
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"{'Config':<12} {'Ranks':<35} {'Rel. Error':>12} {'Compression':>12}")
    print("-" * 73)
    for name, r, rel_err, comp_ratio in results:
        print(f"{name:<12} {str(r):<35} {rel_err:>11.6f}  {comp_ratio * 100:>10.2f} %")


if __name__ == "__main__":
    # Single test with exact (max) ranks
    test_tensor_train_roundtrip()

    # Comparison across different rank configurations
    test_tensor_train_various_ranks()
