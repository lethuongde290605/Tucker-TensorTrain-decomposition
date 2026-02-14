import tensorly as tl
import torch
tl.set_backend('pytorch')

NSAMPLES = 8
SEQ_LEN = 2048
DIM = 768


def tucker_decompose(tensor, ranks=None):
    
    # shape = (nsamples / avg_dim, sq_len = 2048, dim = 768)
    print(f"Original Shape: {tensor.shape}") 

    def prepare_tensor(t):
        # keep batch_size and seq_len: [65536, 12, 64] => [65536, 4, 4, 4, 4, 3]
        t = t.view(-1, t.shape[2])
        # reshape tensor for Tucker: [65536, 3, 4, 4, 4, 4]
        t = t.view(-1, 4, 4, 4, 4, 3)
        return t

    prepaired_tensor = prepare_tensor(tensor)
    
    print(f"Reshaped for Tucker: {prepaired_tensor.shape}") 

    if ranks is None:
        target_ranks = [prepaired_tensor.shape[0], 3, 3, 3, 3, 2]
    else:
        target_ranks = ranks

    print(f"Target Ranks: {target_ranks}")

    results = {}
    
    def run_tucker(tensor, target_ranks):
        # tensorly only support for float32
        tensor = tensor.to(torch.float32)
        
        with torch.cuda.amp.autocast(enabled=False):
            core, factors = tl.decomposition.tucker(tensor, rank=target_ranks, init='svd')
            # core, factors = tl.decomposition.tucker(
            #     tensor,
            #     rank=target_ranks,
            #     init=(core0, factors0),   
            #     fixed_factors=[0], 
            # )

        
        return core, factors

    core, factors = run_tucker(prepaired_tensor, target_ranks)

    print(f"Factors shapes: {[f.shape for f in factors]}")
    print(f"Core shape: {core.shape}")

    return core, factors


def tucker_reconstruct(core, factors, original_shape):
    """
    Reverse the tucker_decompose process to reconstruct the original tensor.

    Args:
        core: Core tensor from Tucker decomposition.
        factors: List of factor matrices from Tucker decomposition.
        original_shape: Tuple (nsamples, seq_len, dim) of the original tensor.

    Returns:
        Reconstructed tensor with the original shape.
    """
    # reconstructed shape: (batch, 4, 4, 4, 4, 3)
    reconstructed = tl.tucker_to_tensor((core, factors))

    # (batch, 4, 4, 4, 4, 3) -> (batch, 768)
    reconstructed = reconstructed.reshape(reconstructed.shape[0], -1)

    # Reshape back to (nsamples, seq_len, dim)
    nsamples, seq_len, dim = original_shape
    reconstructed = reconstructed.view(nsamples, seq_len, dim)

    return reconstructed


def test_tucker_roundtrip():
    """
    Create a tensor of shape (NSAMPLES, SEQ_LEN, DIM), run Tucker decompose
    then reconstruct, and report the reconstruction error.
    """
    import time

    original_shape = (NSAMPLES, SEQ_LEN, DIM)
    print(f"Tensor shape : {original_shape}")
    print(f"  NSAMPLES={NSAMPLES}, SEQ_LEN={SEQ_LEN}, DIM={DIM}")
    print()

    # Create a random tensor
    torch.manual_seed(42)
    tensor = torch.randn(*original_shape)

    print("orignial value: ", tensor[0]) 

    # Decompose 
    t0 = time.time()
    core, factors = tucker_decompose(tensor)
    t1 = time.time()
    print(f"\nTucker decomposition took {t1 - t0:.4f} s")

    # Reconstruct 
    t2 = time.time()
    reconstructed = tucker_reconstruct(core, factors, original_shape)
    t3 = time.time()
    print(f"Reconstruction took       {t3 - t2:.4f} s")

    print("reconstructed value: ", reconstructed[0]) 

    # Error metrics 
    diff = tensor - reconstructed
    frob_error = torch.norm(diff).item()
    frob_orig = torch.norm(tensor).item()
    relative_error = frob_error / frob_orig

    mse = torch.mean(diff ** 2).item()
    mae = torch.mean(torch.abs(diff)).item()

    print(f"  Frobenius norm (original) : {frob_orig:.6f}")
    print(f"  Frobenius norm (error)    : {frob_error:.6f}")
    print(f"  Relative error            : {relative_error:.6f}  ({relative_error * 100:.4f} %)")
    print(f"  MSE                       : {mse:.8f}")
    print(f"  MAE                       : {mae:.8f}")


if __name__ == "__main__":
    test_tucker_roundtrip()