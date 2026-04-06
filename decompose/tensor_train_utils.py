import math
import torch
import tensorly as tl
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor

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
    Reshape a 2-D weight matrix into an N-D tensor suitable for TT decomposition.

    Steps:
      1. view  : (R, C) -> (*row_factors, *col_factors)
      2. permute: interleave row/col factor axes -> (r0,c0, r1,c1, ..., rk,ck)
      3. reshape: combine each (ri, ci) pair -> (ri*ci, ...)

    Args:
        t:           2-D tensor of shape (R, C).
        row_factors: Factorization of R  (len == k).
        col_factors: Factorization of C  (len == k).

    Returns:
        N-D tensor of shape (r0*c0, r1*c1, ..., rk*ck) where N == k.
    """
    assert len(row_factors) == len(col_factors), \
        "row_factors and col_factors must have the same length"

    k = len(row_factors)

    # Step 1: view into individual factors
    t = t.view(*row_factors, *col_factors)

    # Step 2: interleave – axes 0..k-1 are row dims, k..2k-1 are col dims
    perm = []
    for i in range(k):
        perm.append(i)
        perm.append(k + i)
    t = t.permute(*perm)

    # Step 3: combine each pair
    combined = [r * c for r, c in zip(row_factors, col_factors)]
    t = t.reshape(*combined)
    return t


def reverse_prepare_tensor(
    t: torch.Tensor,
    row_factors: list[int],
    col_factors: list[int],
    original_shape: tuple,
) -> torch.Tensor:
    """
    Inverse of prepare_tensor.

    Args:
        t:              N-D tensor produced by prepare_tensor.
        row_factors:    Same list used in prepare_tensor.
        col_factors:    Same list used in prepare_tensor.
        original_shape: The original 2-D shape (R, C).

    Returns:
        2-D tensor of shape `original_shape`.
    """
    k = len(row_factors)

    # Step 1: un-combine pairs -> individual factor axes
    t = t.view(*[x for pair in zip(row_factors, col_factors) for x in pair])

    # Step 2: inverse permutation
    perm = []
    for i in range(k):
        perm.append(i)
        perm.append(k + i)
    inv_perm = [0] * (2 * k)
    for dst, src in enumerate(perm):
        inv_perm[src] = dst
    t = t.permute(*inv_perm)

    # Step 3: reshape to original
    t = t.reshape(*original_shape)
    return t


# ---------------------------------------------------------------------------
# TT decomposition / reshaping helpers
# ---------------------------------------------------------------------------

def run_tensor_train(tensor: torch.Tensor, target_ranks: list[int]) -> list[torch.Tensor]:
    """
    Run TensorLy tensor_train decomposition in float32 precision.

    Args:
        tensor:       Input N-D tensor.
        target_ranks: TT-rank list (length == ndim + 1, first and last == 1).

    Returns:
        List of raw TT-core tensors with shapes (r_i, d_i, r_{i+1}).
    """
    tensor = tensor.to(torch.float32)
    with torch.cuda.amp.autocast(enabled=False):
        tt_tensor = tensor_train(tensor, rank=target_ranks, verbose=False)
    return tt_tensor.factors


def reshape_factors(
    factors: list[torch.Tensor],
    dim_factors: list[tuple[int, int]],
) -> list[torch.Tensor]:
    """
    Reshape raw TT-cores from (r1, m*n, r2) -> (m, r1, n, r2).

    Args:
        factors:     Raw TT-cores (r1, mode_size, r2).
        dim_factors: List of (m, n) pairs matching each core's mode dimension.

    Returns:
        Reshaped TT-cores (m, r1, n, r2).
    """
    reshaped = []
    for f, (m, n) in zip(factors, dim_factors):
        r1, _mn, r2 = f.shape
        f = f.view(r1, m, n, r2)
        f = f.permute(1, 0, 2, 3)   # (m, r1, n, r2)
        reshaped.append(f)
    return reshaped


def reverse_reshape_factors(factors: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Inverse of reshape_factors: (m, r1, n, r2) -> (r1, m*n, r2).

    Args:
        factors: Reshaped TT-cores (m, r1, n, r2).

    Returns:
        Raw TT-cores (r1, m*n, r2).
    """
    reversed_factors = []
    for f in factors:
        m, r1, n, r2 = f.shape
        f = f.permute(1, 0, 2, 3)      # (r1, m, n, r2)
        f = f.reshape(r1, m * n, r2)   # (r1, m*n, r2)
        reversed_factors.append(f)
    return reversed_factors


def compute_max_ranks(dims: list[int]) -> list[int]:
    """
    Compute maximum TT-ranks for a tensor with given mode dimensions.

    max_rank[i] = min(product(dims[:i]), product(dims[i:]))

    Args:
        dims: Mode dimensions of the N-D tensor.

    Returns:
        List of N+1 integers (first and last are always 1).
    """
    n = len(dims)
    max_ranks = [1]
    for i in range(1, n):
        left = math.prod(dims[:i])
        right = math.prod(dims[i:])
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
    Perform Tensor-Train decomposition on a 2-D weight matrix.

    Args:
        tensor:      2-D weight matrix of arbitrary shape (R, C).
        num_factors: How many TT-modes to use (i.e. the order of the TT decomposition).
                     Row and column dimensions are each split into `num_factors` factors.
        ranks:       TT-ranks.  None -> exact decomposition (uses max ranks).
                     Must be a list of length (num_factors + 1) with first/last == 1.

    Returns:
        reshaped_factors: List of TT-cores shaped (m, r1, n, r2).
        raw_factors:      List of TT-cores shaped (r1, m*n, r2) – for reconstruction.
        metadata:         Dict with keys row_factors, col_factors, original_shape.
    """
    assert tensor.ndim == 2, f"Expected 2-D tensor, got shape {tensor.shape}"
    R, C = tensor.shape
    print(f"Original Shape: {tensor.shape}")

    row_factors = factorize_dim(R, num_factors)
    col_factors = factorize_dim(C, num_factors)
    dim_factors = list(zip(row_factors, col_factors))

    print(f"Row factors   : {row_factors}  (product={math.prod(row_factors)})")
    print(f"Col factors   : {col_factors}  (product={math.prod(col_factors)})")

    prepared = prepare_tensor(tensor, row_factors, col_factors)
    print(f"Reshaped for TensorTrain: {list(prepared.shape)}")

    dims = list(prepared.shape)
    max_ranks = compute_max_ranks(dims)

    n_modes = len(dims)
    expected_rank_len = n_modes + 1

    if ranks is None:
        target_ranks = max_ranks

    elif isinstance(ranks, list) and len(ranks) == expected_rank_len:
        for i, (r, m) in enumerate(zip(ranks, max_ranks)):
            if r > m:
                raise ValueError(
                    f"ranks[{i}]={r} exceeds max_rank={m}. "
                    f"Max ranks for tensor {dims}: {max_ranks}"
                )
        target_ranks = ranks

    else:
        raise ValueError(
            f"ranks must be None (exact) or a list of {expected_rank_len} elements "
            f"[1, r1, …, r{n_modes - 1}, 1]. Max ranks: {max_ranks}"
        )

    print(f"Max Ranks              : {max_ranks}")
    print(f"Target Ranks (TT-Ranks): {target_ranks}")

    raw_factors = run_tensor_train(prepared, target_ranks)
    print(f"Factors shapes (raw)         : {[list(f.shape) for f in raw_factors]}")

    reshaped = reshape_factors(raw_factors, dim_factors)
    print(f"Factors shapes (after reshape): {[list(f.shape) for f in reshaped]}")

    metadata = {
        "row_factors": row_factors,
        "col_factors": col_factors,
        "original_shape": (R, C),
    }
    return reshaped, raw_factors, metadata


def tensor_train_reconstruct(
    reshaped_factors: list[torch.Tensor],
    metadata: dict,
) -> torch.Tensor:
    """
    Reverse tensor_train_decompose to reconstruct the original 2-D weight matrix.

    Args:
        reshaped_factors: TT-cores shaped (m, r1, n, r2).
        metadata:         Dict returned by tensor_train_decompose
                          (keys: row_factors, col_factors, original_shape).

    Returns:
        Reconstructed 2-D tensor of shape `metadata['original_shape']`.
    """
    row_factors = metadata["row_factors"]
    col_factors = metadata["col_factors"]
    original_shape = metadata["original_shape"]

    raw_factors = reverse_reshape_factors(reshaped_factors)
    reconstructed = tt_to_tensor(raw_factors)
    reconstructed = reverse_prepare_tensor(reconstructed, row_factors, col_factors, original_shape)
    return reconstructed



# ---------------------------------------------------------------------------
# Standalone tests
# ---------------------------------------------------------------------------

def test_tensor_train_roundtrip(tensor_shape=(768, 768), num_factors=5, ranks=None):
    """
    Create a weight matrix of given shape, run TensorTrain decompose then
    reconstruct, and report the reconstruction error.
    """
    import time

    print("=" * 60)
    print("  Tensor-Train Decomposition Round-trip Test")
    print("=" * 60)
    print(f"Weight matrix shape : {tensor_shape}")
    print()

    torch.manual_seed(42)
    tensor = torch.randn(*tensor_shape)
    print("Original value (first row):", tensor[0, :5], "...")

    t0 = time.time()
    reshaped_factors, raw_factors, metadata = tensor_train_decompose(
        tensor, num_factors=num_factors, ranks=ranks
    )
    t1 = time.time()
    print(f"\nTT decomposition took {t1 - t0:.4f} s")

    t2 = time.time()
    reconstructed = tensor_train_reconstruct(reshaped_factors, metadata)
    t3 = time.time()
    print(f"Reconstruction took   {t3 - t2:.4f} s")

    print("Reconstructed value (first row):", reconstructed[0, :5], "...")

    diff = tensor - reconstructed
    frob_error = torch.norm(diff).item()
    frob_orig = torch.norm(tensor).item()
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

    original_size = tensor.numel()
    compressed_size = sum(f.numel() for f in reshaped_factors)
    compression_ratio = compressed_size / original_size

    print()
    print("--- Compression ---")
    print(f"  Original size    : {original_size} params")
    print(f"  Compressed size  : {compressed_size} params")
    print(f"  Compression ratio: {compression_ratio:.4f}  ({compression_ratio * 100:.2f} %)")

    return relative_error, compression_ratio


if __name__ == "__main__":
    test_tensor_train_roundtrip()
