"""
test_combine.py
===============
End-to-end test of the Tucker + TT-matrix combination pipeline:

  1.  Tucker decomposition on an activation tensor (1, 2048, 768)
      → Tucker factors [f_i], each shape (n_i, q_i)

  2.  TT-matrix decomposition on a weight tensor (768, 768)
      → TT-matrix cores [G_i], each shape (r_i, m_i, n_i, r_{i+1})

  3.  Mode multiplication (apply_tucker_factors_to_tt_cores)
      For each mode i:   new_G_i[r,m,q,R] = Σ_n  G_i[r,m,n,R] * f_i[n,q]
      → new cores [H_i], each shape (r_i, m_i, q_i, r_{i+1})

  4.  Reconstruction (reconstruct_combined_tt_cores)
      tt_matrix_to_tensor → permute → reshape
      → 2-D weight matrix of shape (M, Q)
         M = ∏ m_i = 768,   Q = ∏ q_i  (Tucker-compressed)
"""

import math
import torch
import tensorly as tl

tl.set_backend('pytorch')

from decompose.tucker_utils import (
    factorize_dim     as tucker_factorize_dim,
    prepare_tensor    as tucker_prepare_tensor,
    calculate_rank,
    decompose         as tucker_decompose,
)
from decompose.tensor_train_utils import (
    factorize_dim          as tt_factorize_dim,
    prepare_tensor         as tt_prepare_tensor,
    compute_max_ranks,
    run_tensor_train_matrix,
)
from decompose.eigen_attn_utils import (
    apply_tucker_factors_to_tt_cores,
    reconstruct_combined_tt_cores,
)


def section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_combine(
    tucker_shape    = (1, 2048, 768),   # (nsamples, seq_len, dim)
    tt_shape        = (768, 768),        # (row_dim, col_dim)
    num_factors     = 5,
    tucker_ratio    = 0.7,
    tt_ranks        = None,              # None → use [1, 36, 64, 64, 36, 1]
    seed            = 42,
):
    torch.manual_seed(seed)

    # -------------------------------------------------------------------------
    # 1. Tucker decomposition
    # -------------------------------------------------------------------------
    section("1. Tucker Decomposition  (activation tensor)")

    tucker_raw = torch.randn(*tucker_shape)
    print(f"Input shape            : {list(tucker_raw.shape)}")

    dim = tucker_shape[-1]                                    # 768
    dim1_factors = tucker_factorize_dim(dim, num_factors)     # [4,4,4,4,3]
    print(f"Dim factors (768 → {dim1_factors})")

    # (1, 2048, 768) → (2048, 4, 4, 4, 4, 3)
    tucker_tensor = tucker_prepare_tensor(tucker_raw, dim1_factors)
    print(f"Prepared shape         : {list(tucker_tensor.shape)}")

    # Decompose along modes 1..5  (leave batch/sample axis 0 untouched)
    modes = list(range(1, tucker_tensor.ndim))                 # [1,2,3,4,5]
    rank  = calculate_rank(tucker_tensor.shape, modes, tucker_ratio)
    print(f"Tucker ranks           : {rank}")

    core, tucker_factors, comp = tucker_decompose(tucker_tensor, rank=rank, modes=modes)
    print(f"Tucker core shape      : {list(core.shape)}")
    print(f"Tucker factor shapes   : {[list(f.shape) for f in tucker_factors]}")

    # -------------------------------------------------------------------------
    # 2. TT-matrix decomposition
    # -------------------------------------------------------------------------
    section("2. TT-matrix Decomposition  (weight tensor)")

    tt_raw = torch.randn(*tt_shape)
    print(f"Input shape            : {list(tt_raw.shape)}")

    row_factors = tt_factorize_dim(tt_shape[0], num_factors)   # [4,4,4,4,3]
    col_factors = tt_factorize_dim(tt_shape[1], num_factors)   # [4,4,4,4,3]
    print(f"Row factors            : {row_factors}")
    print(f"Col factors            : {col_factors}")

    # (768, 768) → (4,4,4,4,3, 4,4,4,4,3)
    tt_prepared = tt_prepare_tensor(tt_raw, row_factors, col_factors)
    print(f"Prepared shape         : {list(tt_prepared.shape)}")

    max_ranks = compute_max_ranks(row_factors, col_factors)
    target_ranks = tt_ranks if tt_ranks is not None else [1, 36, 64, 64, 36, 1]
    print(f"Max TT-ranks           : {max_ranks}")
    print(f"Target TT-ranks        : {target_ranks}")

    tt_cores = run_tensor_train_matrix(tt_prepared, target_ranks)
    print(f"TT core shapes         : {[list(c.shape) for c in tt_cores]}")

    # -------------------------------------------------------------------------
    # 3. Mode multiplication: Tucker factors × TT cores
    # -------------------------------------------------------------------------
    section("3. Combine – mode multiplication")

    print("Tucker factors (n_i, q_i):")
    for i, f in enumerate(tucker_factors):
        print(f"  factors[{i}]: {list(f.shape)}")

    print("TT cores (r_i, m_i, n_i, r_{i+1}):")
    for i, c in enumerate(tt_cores):
        print(f"  cores[{i}]:   {list(c.shape)}")

    # Verify n_i alignment
    for i, (f, c) in enumerate(zip(tucker_factors, tt_cores)):
        assert f.shape[0] == c.shape[2], (
            f"Mode {i}: Tucker factor n_i={f.shape[0]} ≠ TT core n_i={c.shape[2]}"
        )
    print("✓  n_i alignment check passed")

    new_cores = apply_tucker_factors_to_tt_cores(tucker_factors, tt_cores)
    print("New cores (r_i, m_i, q_i, r_{i+1}):")
    for i, c in enumerate(new_cores):
        print(f"  new_cores[{i}]: {list(c.shape)}")

    # -------------------------------------------------------------------------
    # 4. Reconstruct combined weight matrix
    # -------------------------------------------------------------------------
    section("4. Reconstruct combined weight matrix")

    result = reconstruct_combined_tt_cores(new_cores)

    M = math.prod(row_factors)                                    # 768
    Q = math.prod(f.shape[1] for f in tucker_factors)            # prod of Tucker ranks
    print(f"Expected shape         : ({M}, {Q})")
    print(f"Reconstructed shape    : {list(result.shape)}")

    assert result.shape == (M, Q), (
        f"Shape mismatch! got {result.shape}, expected ({M}, {Q})"
    )
    print("✓  Shape check passed!")

    # Quick stats
    print(f"\nOriginal weight size   : {tt_raw.numel():,} params")
    print(f"Combined result size   : {result.numel():,} params")
    print(f"Compression ratio      : {result.numel() / tt_raw.numel():.4f}")

    return result


if __name__ == "__main__":
    test_combine()
