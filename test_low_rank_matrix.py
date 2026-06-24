import torch
import math


# ============================================================
# Unfold / Fold helpers
# ============================================================

def unfold_tensor(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    """
    Matricize tensor theo mode chỉ định.

    Với tensor shape (8, 8, 12):
        mode 0 -> (8, 96)
        mode 1 -> (8, 96)
        mode 2 -> (12, 64)
    """
    ndim = tensor.ndim
    order = [mode] + [i for i in range(ndim) if i != mode]
    return tensor.permute(order).reshape(tensor.shape[mode], -1)


def fold_matrix(matrix: torch.Tensor, original_shape: tuple, mode: int) -> torch.Tensor:
    """
    Fold matrix đã unfold theo mode về tensor gốc.
    """
    ndim = len(original_shape)

    mode_dim = original_shape[mode]
    other_dims = [original_shape[i] for i in range(ndim) if i != mode]

    tensor_perm = matrix.reshape(mode_dim, *other_dims)

    order = [mode] + [i for i in range(ndim) if i != mode]

    inv_order = [0] * ndim
    for i, j in enumerate(order):
        inv_order[j] = i

    return tensor_perm.permute(inv_order)


# ============================================================
# Tensor n-mode product
# ============================================================

def mode_dot(tensor: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor:
    """
    N-mode product: tensor ×_mode matrix.

    tensor shape:
        (I0, I1, ..., In)

    matrix shape:
        (J, I_mode)

    output shape:
        (I0, ..., I_{mode-1}, J, I_{mode+1}, ..., In)
    """
    unfolded = unfold_tensor(tensor, mode)          # (I_mode, -1)
    product = matrix @ unfolded                    # (J, -1)

    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[0]

    return fold_matrix(product, tuple(new_shape), mode)


# ============================================================
# Energy / SVD helpers
# ============================================================

def energy_ratio_from_singular_values(S: torch.Tensor):
    """
    cumulative_energy[r-1] = phần năng lượng giữ lại khi dùng rank r.
    """
    energy = S ** 2
    total_energy = energy.sum()

    if total_energy.item() == 0:
        return torch.zeros_like(energy)

    cumulative_energy = torch.cumsum(energy, dim=0) / total_energy

    # tránh trường hợp float rounding làm energy > 1.0 một chút
    cumulative_energy = torch.clamp(cumulative_energy, max=1.0)

    return cumulative_energy


def truncated_svd_reconstruct(matrix: torch.Tensor, rank: int):
    """
    Reconstruct matrix bằng truncated SVD rank-r.
    """
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

    matrix_rec = (U[:, :rank] * S[:rank].unsqueeze(0)) @ Vh[:rank, :]

    return matrix_rec, S


# ============================================================
# Generate one Tucker low-rank tensor
# ============================================================

def random_orthonormal_matrix(rows: int, cols: int, seed: int = None, dtype=torch.float64):
    """
    Generate matrix shape (rows, cols) có các cột gần trực chuẩn.
    """
    if seed is not None:
        torch.manual_seed(seed)

    A = torch.randn(rows, cols, dtype=dtype)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q[:, :cols]


def generate_tucker_low_rank_tensor(
    tensor_shape=(8, 8, 12),
    multilinear_rank=(3, 4, 5),
    seed=1234,
    dtype=torch.float64,
):
    """
    Generate 1 tensor low-rank theo Tucker.

    X = G ×0 U0 ×1 U1 ×2 U2

    tensor_shape:
        shape tensor cuối cùng, ví dụ (8, 8, 12)

    multilinear_rank:
        rank mong muốn theo từng mode, ví dụ (3, 4, 5)

    Với tensor X:
        rank(unfold(X, mode=0)) <= 3
        rank(unfold(X, mode=1)) <= 4
        rank(unfold(X, mode=2)) <= 5
    """
    if len(tensor_shape) != len(multilinear_rank):
        raise ValueError("tensor_shape and multilinear_rank must have same length")

    for dim, rank in zip(tensor_shape, multilinear_rank):
        if rank < 1 or rank > dim:
            raise ValueError(
                f"Invalid rank={rank} for dim={dim}. Rank must be in [1, {dim}]"
            )

    torch.manual_seed(seed)

    # Core tensor G shape = multilinear_rank
    core = torch.randn(*multilinear_rank, dtype=dtype)

    # Factor matrices
    factors = []
    for i, (dim, rank) in enumerate(zip(tensor_shape, multilinear_rank)):
        U = random_orthonormal_matrix(
            rows=dim,
            cols=rank,
            seed=seed + 100 + i,
            dtype=dtype,
        )
        factors.append(U)

    # X = G ×0 U0 ×1 U1 ×2 U2
    tensor = core
    for mode, U in enumerate(factors):
        tensor = mode_dot(tensor, U, mode)

    return tensor, core, factors


# ============================================================
# Inspect SVD từng mode
# ============================================================

def inspect_tensor_mode_svd(
    tensor: torch.Tensor,
    test_ranks=(1, 2, 3, 4, 5, 6, 7, 8, 12),
    verbose=True,
):
    """
    Với 1 tensor, unfold từng mode rồi phân tích SVD.

    Với mỗi mode:
        - unfold matrix
        - actual matrix rank
        - singular values
        - reconstruct bằng truncated SVD rank r
        - tính retained energy
        - tính relative error trên unfolding matrix
        - tính relative error sau khi fold về tensor
    """
    tensor = tensor.to(torch.float64)
    original_shape = tuple(tensor.shape)

    results = {}

    if verbose:
        print("\n============================================================")
        print("Tensor Mode-wise SVD Inspection")
        print("============================================================")
        print(f"tensor shape: {original_shape}")

    for mode in range(tensor.ndim):
        matrix = unfold_tensor(tensor, mode)

        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        cumulative_energy = energy_ratio_from_singular_values(S)

        actual_rank = int(torch.linalg.matrix_rank(matrix).item())
        max_rank = min(matrix.shape)

        ranks_to_test = [r for r in test_ranks if 1 <= r <= max_rank]

        mode_result = {
            "unfold_shape": tuple(matrix.shape),
            "actual_rank": actual_rank,
            "singular_values": S.detach().cpu(),
            "retained_energies": {},
            "relative_errors_matrix": {},
            "relative_errors_tensor": {},
        }

        if verbose:
            print("\n" + "-" * 70)
            print(f"Mode {mode} unfolding")
            print("-" * 70)
            print(f"unfold shape     : {tuple(matrix.shape)}")
            print(f"actual rank      : {actual_rank}")
            print(f"max possible rank: {max_rank}")
            print(
                "singular values  :",
                [round(x, 6) for x in S.detach().cpu().tolist()],
            )
            print("\nReconstruction by truncated SVD:")

        for r in ranks_to_test:
            matrix_rec = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]

            # Error trên matrix unfolding
            rel_error_matrix = (
                torch.linalg.norm(matrix - matrix_rec) / torch.linalg.norm(matrix)
            )

            # Fold về tensor để kiểm tra error trên tensor gốc
            tensor_rec = fold_matrix(matrix_rec, original_shape, mode)
            rel_error_tensor = (
                torch.linalg.norm(tensor - tensor_rec) / torch.linalg.norm(tensor)
            )

            retained_energy = cumulative_energy[r - 1]

            mode_result["retained_energies"][r] = float(retained_energy.item())
            mode_result["relative_errors_matrix"][r] = float(rel_error_matrix.item())
            mode_result["relative_errors_tensor"][r] = float(rel_error_tensor.item())

            if verbose:
                print(
                    f"rank={r:<2d} | "
                    f"energy={retained_energy.item():.8f} | "
                    f"rel_err={rel_error_matrix.item():.8f} | "
                    # f"rel_err_tensor={rel_error_tensor.item():.8f}"
                )

        results[mode] = mode_result

    return results


# ============================================================
# Main experiment
#import math
import tensorly as tl
from tensorly.decomposition import partial_tucker
from tensorly.tenalg import multi_mode_dot


def reconstruction_metrics(tensor, core, factors):
    reconstructed = multi_mode_dot(core, factors)

    abs_error = tl.norm(tensor - reconstructed)
    rel_error = abs_error / tl.norm(tensor)

    norm_tensor = tl.norm(tensor)
    norm_rec = tl.norm(reconstructed)

    return {
        "reconstructed": reconstructed,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "norm_tensor": norm_tensor,
        "norm_rec": norm_rec,
        "norm_diff": tl.abs(norm_tensor - norm_rec),
    }


def tucker_param_count_from_result(tensor, core, factors):
    original_params = math.prod(tensor.shape)
    core_params = math.prod(core.shape)
    factor_params = sum(math.prod(f.shape) for f in factors)
    compressed_params = core_params + factor_params

    return {
        "original_params": original_params,
        "core_params": core_params,
        "factor_params": factor_params,
        "compressed_params": compressed_params,
        "compression_ratio_core_only": core_params / original_params,
        "compression_ratio_with_factors": compressed_params / original_params,
    }

def run_one_experiment(tensor, rank, title, n_iter_max=200, init="svd", random_state=None):
    print(f"\n=== {title} ===")
    print(f"rank: {rank}")

    # ------------------------------------------------------------
    # FIX: TensorLy default backend là NumPy, nên nếu input là torch.Tensor
    # thì cần convert sang numpy / tensorly tensor trước.
    # ------------------------------------------------------------
    if isinstance(tensor, torch.Tensor):
        tensor_tl = tl.tensor(tensor.detach().cpu().numpy())
    else:
        tensor_tl = tl.tensor(tensor)

    (core, factors), errors = partial_tucker(
        tensor_tl,
        rank=rank,
        n_iter_max=n_iter_max,
        init=init,
        random_state=random_state,
        verbose=False,
    )

    metrics = reconstruction_metrics(tensor_tl, core, factors)

    print("\n--- Reconstruction Metrics ---")
    print(f"norm_tensor : {metrics['norm_tensor']:.8f}")
    print(f"norm_rec    : {metrics['norm_rec']:.8f}")
    print(f"norm_diff   : {metrics['norm_diff']:.8f}")
    print(f"abs_error   : {metrics['abs_error']:.8f}")
    print(f"rel_error   : {metrics['rel_error']:.8f}")

    print("\n--- Shapes ---")
    print(f"tensor shape: {tensor_tl.shape}")
    print(f"core shape  : {core.shape}")
    for i, factor in enumerate(factors):
        print(f"factor[{i}] shape: {factor.shape}")

    if rank is not None:
        # Nên tính theo shape thực tế của core/factors,
        # tránh lỗi nếu TensorLy tự clip rank.
        params = tucker_param_count_from_result(tensor_tl, core, factors)

        print("\n--- Parameter Count ---")
        print(f"original params           : {params['original_params']}")
        print(f"core params               : {params['core_params']}")
        print(f"factor params             : {params['factor_params']}")
        print(f"compressed params         : {params['compressed_params']}")
        print(f"core-only ratio           : {params['compression_ratio_core_only']:.6f}")
        print(f"ratio with Tucker factors : {params['compression_ratio_with_factors']:.6f}")

    return core, factors, errors


def run_tucker_low_rank_tensor_experiment(
    tensor_shape=(8, 8, 12),
    multilinear_rank=(3, 4, 5),
    test_ranks=(1, 2, 3, 4, 5, 6, 7, 8, 12),
    seed=1234,
):
    """
    Generate 1 tensor sao cho unfold từng mode là low-rank,
    rồi phân tích SVD từng mode và tính relative error từng mode.
    """
    print("============================================================")
    print("Generate One Tucker Low-rank Tensor")
    print("============================================================")
    print(f"target tensor shape     : {tensor_shape}")
    print(f"target multilinear rank : {multilinear_rank}")
    print(f"test ranks              : {test_ranks}")

    tensor, core, factors = generate_tucker_low_rank_tensor(
        tensor_shape=tensor_shape,
        multilinear_rank=multilinear_rank,
        seed=seed,
        dtype=torch.float64,
    )

    print("\n--- Generated Tensor Info ---")
    print(f"tensor shape: {tuple(tensor.shape)}")
    print(f"core shape  : {tuple(core.shape)}")


    for i, U in enumerate(factors):
        print(f"factor[{i}] shape: {tuple(U.shape)}")

    results = inspect_tensor_mode_svd(
        tensor=tensor,
        test_ranks=test_ranks,
        verbose=True,
    )

    # Low-rank Tucker
    ranks = [2, 4, 5]
    run_one_experiment(
        tensor,
        rank=ranks,
        title="Partial Tucker",
        n_iter_max=200,
    )


    return {
        "tensor": tensor,
        "core": core,
        "factors": factors,
        "results": results,
    }


if __name__ == "__main__":
    output = run_tucker_low_rank_tensor_experiment(
        tensor_shape=(8, 8, 12),
        multilinear_rank=(3, 4, 5),
        test_ranks=(1, 2, 3, 4, 5, 6, 7, 8, 12),
        seed=1234,
    )