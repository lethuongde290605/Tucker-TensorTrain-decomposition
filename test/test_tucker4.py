import math
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


def tucker_param_count(tensor_shape, ranks):
    original_params = math.prod(tensor_shape)
    core_params = math.prod(ranks)
    factor_params = sum(I * R for I, R in zip(tensor_shape, ranks))
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

    (core, factors), errors = partial_tucker(
        tensor,
        rank=rank,
        n_iter_max=n_iter_max,
        init=init,
        random_state=random_state,
        verbose=False,
    )

    metrics = reconstruction_metrics(tensor, core, factors)

    print("\n--- Reconstruction Metrics ---")
    print(f"norm_tensor : {metrics['norm_tensor']}")
    print(f"norm_rec    : {metrics['norm_rec']}")
    print(f"norm_diff   : {metrics['norm_diff']}")
    print(f"abs_error   : {metrics['abs_error']}")
    print(f"rel_error   : {metrics['rel_error']}")

    print("\n--- Shapes ---")
    print(f"tensor shape: {tensor.shape}")
    print(f"core shape  : {core.shape}")
    for i, factor in enumerate(factors):
        print(f"factor[{i}] shape: {factor.shape}")

    if rank is not None:
        params = tucker_param_count(tensor.shape, rank)

        print("\n--- Parameter Count ---")
        print(f"original params          : {params['original_params']}")
        print(f"core params              : {params['core_params']}")
        print(f"factor params            : {params['factor_params']}")
        print(f"compressed params         : {params['compressed_params']}")
        print(f"core-only ratio           : {params['compression_ratio_core_only']}")
        print(f"ratio with Tucker factors : {params['compression_ratio_with_factors']}")

    return core, factors, errors


def run_partial_tucker_demo():
    rng = tl.check_random_state(1234)
    tensor = tl.tensor(rng.standard_normal((8, 8, 12)))

    # Low-rank Tucker
    ranks = [7, 7, 10]
    run_one_experiment(
        tensor,
        rank=ranks,
        title="Partial Tucker",
        n_iter_max=200,
    )

    # Random state consistency
    print("\n=== Random State Consistency ===")

    (core1, factors1), _ = partial_tucker(
        tensor,
        rank=ranks,
        random_state=0,
        init="random",
    )

    (core2, factors2), _ = partial_tucker(
        tensor,
        rank=ranks,
        random_state=0,
        init="random",
    )

    core_diff = tl.max(tl.abs(core1 - core2))
    factor_diffs = [
        tl.max(tl.abs(f1 - f2))
        for f1, f2 in zip(factors1, factors2)
    ]

    print(f"core max abs diff: {core_diff}")
    for i, diff in enumerate(factor_diffs):
        print(f"factor[{i}] max abs diff: {diff}")


if __name__ == "__main__":
    run_partial_tucker_demo()