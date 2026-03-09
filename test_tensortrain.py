import tensorly as tl
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor
import torch
tl.set_backend('pytorch')

# Simulated weight matrix dimensions (hidden_dim x hidden_dim)
HIDDEN_DIM = 768


def tensor_train_decompose(tensor, ranks=None):
    """
    Perform Tensor-Train decomposition on a weight matrix.
    Follows the same logic as tensor_train_decompose_opt_layer in eigen_attn_utils.py.

    Args:
        tensor: Weight matrix of shape (HIDDEN_DIM, HIDDEN_DIM) e.g. (768, 768).
        ranks: TT-ranks. Can be None (exact), int (uniform), or list.

    Returns:
        factors: List of TT-cores after reshape to (m, r1, n, r2).
        tt_factors_raw: List of TT-cores in raw format (r1, m*n, r2) for reconstruction.
    """
    print(f"Original Shape: {tensor.shape}")

    # Reshape (768, 768) -> (4, 4, 4, 4, 3, 4, 4, 4, 4, 3)
    # Then interleave and combine to (16, 16, 16, 16, 9)
    def prepare_tensor(t):
        # (768, 768) -> (4, 4, 4, 4, 3, 4, 4, 4, 4, 3)
        t = t.view(4, 4, 4, 4, 3, 4, 4, 4, 4, 3)
        # Interleave row/col factors: (0,5), (1,6), (2,7), (3,8), (4,9)
        t = t.permute(0, 5, 1, 6, 2, 7, 3, 8, 4, 9)
        # Reshape to combine pairs: (4*4, 4*4, 4*4, 4*4, 3*3) = (16, 16, 16, 16, 9)
        t = t.reshape(16, 16, 16, 16, 9)
        return t

    prepared_tensor = prepare_tensor(tensor)
    print(f"Reshaped for TensorTrain: {prepared_tensor.shape}")  # (16, 16, 16, 16, 9)

    # Tính TT-rank tối đa: r_i = min(∏ dims[:i+1], ∏ dims[i+1:])
    dims = list(prepared_tensor.shape)  # [16, 16, 16, 16, 9]
    n = len(dims)
    max_ranks = [1]
    for i in range(1, n):
        left = 1
        for d in dims[:i]:
            left *= d
        right = 1
        for d in dims[i:]:
            right *= d
        max_ranks.append(min(left, right))
    max_ranks.append(1)
    # max_ranks = [1, 16, 256, 144, 9, 1]

    # Tensor 5D => rank list cần 6 phần tử: [1, r1, r2, r3, r4, 1]
    if ranks is None:
        # Exact decomposition: dùng max ranks
        target_ranks = max_ranks

    elif isinstance(ranks, list) and len(ranks) == 6:
        # Validate: mỗi rank không được vượt quá max rank tương ứng
        for i, (r, m) in enumerate(zip(ranks, max_ranks)):
            if r > m:
                raise ValueError(
                    f"ranks[{i}]={r} vượt quá max rank={m}. "
                    f"Max ranks cho tensor {dims}: {max_ranks}"
                )
        target_ranks = ranks

    else:
        raise ValueError(
            f"ranks phải là None (exact) hoặc list 6 phần tử [1, r1, r2, r3, r4, 1]. "
            f"Max ranks: {max_ranks}"
        )

    print(f"Max Ranks:              {max_ranks}")
    print(f"Target Ranks (TT-Ranks): {target_ranks}")

    # Run TT decomposition
    def run_tensor_train(tensor, target_ranks):
        tensor = tensor.to(torch.float32)

        with torch.cuda.amp.autocast(enabled=False):
            tt_tensor = tensor_train(tensor, rank=target_ranks, verbose=False)

        factors = tt_tensor.factors
        return factors

    raw_factors = run_tensor_train(prepared_tensor, target_ranks)

    print(f"Factors shapes (raw): {[f.shape for f in raw_factors]}")

    # Reshape factors from (r1, m*n, r2) -> (m, r1, n, r2)
    dim_factors = [(4, 4), (4, 4), (4, 4), (4, 4), (3, 3)]

    def reshape_factors(factors):
        reshaped = []
        for i, f in enumerate(factors):
            r1, mn, r2 = f.shape
            m, n = dim_factors[i]
            f = f.view(r1, m, n, r2)
            f = f.permute(1, 0, 2, 3)  # (m, r1, n, r2)
            reshaped.append(f)
        return reshaped

    reshaped_factors = reshape_factors(raw_factors)

    print(f"Factors shapes (after reshape): {[f.shape for f in reshaped_factors]}")

    return reshaped_factors, raw_factors


def tensor_train_reconstruct(reshaped_factors):
    """
    Reverse the tensor_train_decompose process to reconstruct the original weight matrix.

    Args:
        reshaped_factors: List of TT-cores with shape (m, r1, n, r2).

    Returns:
        Reconstructed weight matrix with shape (HIDDEN_DIM, HIDDEN_DIM).
    """
    dim_factors = [(4, 4), (4, 4), (4, 4), (4, 4), (3, 3)]

    # Reverse reshape_factors: (m, r1, n, r2) -> (r1, m*n, r2)
    def reverse_reshape_factors(factors):
        reversed_factors = []
        for i, f in enumerate(factors):
            m, r1, n, r2 = f.shape
            # (m, r1, n, r2) -> (r1, m, n, r2)
            f = f.permute(1, 0, 2, 3)
            # (r1, m, n, r2) -> (r1, m*n, r2)
            f = f.reshape(r1, m * n, r2)
            reversed_factors.append(f)
        return reversed_factors

    # Reverse prepare_tensor: (16,16,16,16,9) -> (768, 768)
    def reverse_prepare_tensor(t):
        # (16,16,16,16,9) -> (4,4, 4,4, 4,4, 4,4, 3,3)
        t = t.view(4, 4, 4, 4, 4, 4, 4, 4, 3, 3)
        # Reverse permute(0,5,1,6,2,7,3,8,4,9)
        # Inverse permutation: (0,2,4,6,8, 1,3,5,7,9)
        t = t.permute(0, 2, 4, 6, 8, 1, 3, 5, 7, 9)
        # (4,4,4,4,3, 4,4,4,4,3) -> (768, 768)
        t = t.reshape(HIDDEN_DIM, HIDDEN_DIM)
        return t

    raw_factors = reverse_reshape_factors(reshaped_factors)
    reconstructed = tt_to_tensor(raw_factors)
    reconstructed = reverse_prepare_tensor(reconstructed)

    return reconstructed


def test_tensor_train_roundtrip(ranks=None):
    """
    Create a weight matrix of shape (HIDDEN_DIM, HIDDEN_DIM), run TensorTrain decompose
    then reconstruct, and report the reconstruction error.
    """
    import time

    print("=" * 60)
    print("  Tensor-Train Decomposition Round-trip Test")
    print("=" * 60)
    print(f"Weight matrix shape : ({HIDDEN_DIM}, {HIDDEN_DIM})")
    print()

    # Create a random weight matrix
    torch.manual_seed(42)
    tensor = torch.randn(HIDDEN_DIM, HIDDEN_DIM)

    print("Original value (first row):", tensor[0, :5], "...")

    # Decompose
    t0 = time.time()
    reshaped_factors, raw_factors = tensor_train_decompose(tensor, ranks=ranks)
    t1 = time.time()
    print(f"\nTT decomposition took {t1 - t0:.4f} s")

    # Reconstruct
    t2 = time.time()
    reconstructed = tensor_train_reconstruct(reshaped_factors)
    t3 = time.time()
    print(f"Reconstruction took   {t3 - t2:.4f} s")

    print("Reconstructed value (first row):", reconstructed[0, :5], "...")

    # Error metrics
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

    # Compression ratio
    original_size = tensor.numel()
    compressed_size = sum([f.numel() for f in reshaped_factors])
    compression_ratio = (original_size - compressed_size) / original_size

    print()
    print("--- Compression ---")
    print(f"  Original size    : {original_size} params")
    print(f"  Compressed size  : {compressed_size} params")
    print(f"  Compression ratio: {compression_ratio:.4f}  ({compression_ratio * 100:.2f} %)")

    return relative_error, compression_ratio


def test_tensor_train_various_ranks():
    """
    Test TensorTrain decomposition with different rank configurations
    to see the trade-off between compression and accuracy.
    """
    print("\n" + "=" * 60)
    print("  Testing various TT-rank configurations")
    print("=" * 60)

    # Max TT-ranks for (16,16,16,16,9): [1, 16, 256, 144, 9, 1]
    # Original size = 768*768 = 589,824 params
    rank_configs = [
        ("Exact (max ranks)",                   None),
        ("Low  [1, 4, 4, 4, 4, 1]",            [1, 4, 4, 4, 4, 1]),
        ("Mid  [1, 8, 16, 16, 8, 1]",          [1, 8, 16, 16, 8, 1]),
        ("High [1, 16, 64, 64, 9, 1]",         [1, 16, 64, 64, 9, 1]),
        ("Near-max [1, 16, 128, 128, 9, 1]",   [1, 16, 128, 128, 9, 1]),
        ("Max-capped [1, 16, 256, 144, 9, 1]", [1, 16, 256, 144, 9, 1]),
    ]

    results = []
    for name, ranks in rank_configs:
        print(f"\n{'─' * 50}")
        print(f"Config: {name}")
        print(f"{'─' * 50}")
        rel_err, comp_ratio = test_tensor_train_roundtrip(ranks=ranks)
        results.append((name, rel_err, comp_ratio))

    # Summary table
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"{'Config':<30} {'Rel. Error':>12} {'Compression':>12}")
    print("-" * 56)
    for name, rel_err, comp_ratio in results:
        print(f"{name:<30} {rel_err:>11.6f}  {comp_ratio * 100:>10.2f} %")


if __name__ == "__main__":
    # Single test with exact (max) ranks
    test_tensor_train_roundtrip()

    # Comparison across different rank configurations
    test_tensor_train_various_ranks()
