import torch
from functools import partial
import numpy as np
import os
import scipy
import tensorly
from tensorly.tt_tensor import tt_to_tensor
from tensorly.tt_matrix import tt_matrix_to_tensor
import math
tensorly.set_backend('pytorch')


def generate_basis_vectors_per_head(feat, threshold, num_heads, args, layer_id):
    U_kq = []
    S_kq = []
    
    X=torch.transpose(feat.reshape(feat.shape[0],num_heads,-1),0,1)
    dtype = X.dtype
    device = X.device
    X = X.to(torch.float32)


    for i in range(num_heads):
        # X[i] = torch.matmul(X[i], hadamard)
        u,s,v = torch.svd(X[i].t())
        u = u.to(device)

        s_val = s.cpu().numpy()
        s_val = s_val/np.sum(s_val)
        S_kq.append(torch.tensor(s_val))
        U_kq.append(u)
        
    X = X.to(dtype)
    
    return torch.stack(U_kq), torch.stack(S_kq)

def generate_basis_vectors_per_layer(feat, threshold, num_heads, args, layer_id):
    X = feat
    X=torch.transpose(X.reshape(X.shape[0],num_heads,-1),0,1)
    X = X.reshape(-1, X.shape[-1])
    dtype = X.dtype
    device = X.device
    X = X.to(torch.float32)
    u,s,v = torch.svd(X.t())
    u = u.to(device)

    s_val = s.cpu().numpy()
    s_val = s_val/np.sum(s_val)
    
    S_v = s_val
    k=np.sum(np.cumsum(s_val)<threshold)+1
    U_v = u.to(dtype)
    X = X.to(dtype)
    
    return U_v, torch.tensor(S_v)

@torch.no_grad()
def get_kqv_mpt(layer, fp_inps, args, position_bias, attention_mask):
    
    avg_kqv = []
    kqv = {}
    
    avg_dim = args.eigen_attn_params['avg_dim_features']

    def forward_hook_kqv(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        y_size = y.shape
        # breakpoint()

        if name in kqv.keys():
            kqv[name] += [y.view(-1,y_size[-1])]
        else:
            kqv[name] = [y.view(-1,y_size[-1])]
        
        dim = len(kqv[name])
        
        if dim >= avg_dim:
            Y = torch.stack(kqv.pop(name)).mean(dim = 0)
            avg_kqv.append(Y)

    
    hooks = []
    for n, m in layer.named_modules():
        if (isinstance(m, torch.nn.Linear)) and  '.Wqkv' in n :
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_kqv, name=n)))
        

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for j in range(args.nsamples):
        layer(fp_inps[j].unsqueeze(0), attention_mask = attention_mask, position_bias = position_bias)
    
    del kqv

    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    avg_kqv = torch.stack(avg_kqv)
    # avg_kqv = torch.cat(avg_kqv)
    if args.low_rank:
        rank_kq = layer.attn.rank_kq
        avg_q = avg_kqv[:,:,0:rank_kq]
        avg_k = avg_kqv[:,:,rank_kq:2*rank_kq]
        avg_v = avg_kqv[:,:,2*rank_kq:]
    else:
        avg_q, avg_k, avg_v  = avg_kqv.chunk(3, dim=-1)
    
    

    return avg_k, avg_q, avg_v

@torch.no_grad()
def get_kqv_llama(layer, fp_inps, args, attention_mask, position_ids):
    
    avg_k = []
    k = {}

    avg_q = []
    q = {}

    avg_v = []
    v = {}
    
    avg_dim = args.eigen_attn_params['avg_dim_features']

    def forward_hook_k(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        y_size = y.shape
        # breakpoint()

        if name in k.keys():
            k[name] += [y.view(-1,y_size[-1])]
        else:
            k[name] = [y.view(-1,y_size[-1])]
        
        dim = len(k[name])
        
        if dim >= avg_dim:
            Y = torch.stack(k.pop(name)).mean(dim = 0)
            avg_k.append(Y)
    
    def forward_hook_q(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        y_size = y.shape
        # breakpoint()

        if name in q.keys():
            q[name] += [y.view(-1,y_size[-1])]
        else:
            q[name] = [y.view(-1,y_size[-1])]
        
        dim = len(q[name])
        
        if dim >= avg_dim:
            Y = torch.stack(q.pop(name)).mean(dim = 0)
            avg_q.append(Y)
    
    def forward_hook_v(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        y_size = y.shape
        # breakpoint()

        if name in v.keys():
            v[name] += [y.view(-1,y_size[-1])]
        else:
            v[name] = [y.view(-1,y_size[-1])]
        
        dim = len(v[name])
        
        if dim >= avg_dim:
            Y = torch.stack(v.pop(name)).mean(dim = 0)
            avg_v.append(Y)

    
    hooks = []
    for n, m in layer.named_modules():
        if (isinstance(m, torch.nn.Linear)) and  '.k_proj' in n and 'up' not in n :
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_k, name=n)))
            
        elif (isinstance(m, torch.nn.Linear)) and  '.q_proj' in n and 'up' not in n:
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_q, name=n)))
        
        elif (isinstance(m, torch.nn.Linear)) and  '.v_proj' in n and 'up' not in n:
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_v, name=n)))
        

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for j in range(args.nsamples):
        layer(fp_inps[j].unsqueeze(0), attention_mask = attention_mask, position_ids = position_ids)
    
    # del k, q, v

    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    avg_k = torch.stack(avg_k)
    avg_q = torch.stack(avg_q)
    avg_v = torch.stack(avg_v)

    return avg_k, avg_q, avg_v


@torch.no_grad()
def get_kqv_opt(layer, fp_inps, args):
    
    avg_k = []
    k = {}

    avg_q = []
    q = {}

    avg_v = []
    v = {}
    
    avg_dim = args.eigen_attn_params['avg_dim_features']

    def forward_hook_k(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        y_size = y.shape
        # breakpoint()

        if name in k.keys():
            k[name] += [y.view(-1,y_size[-1])]
        else:
            k[name] = [y.view(-1,y_size[-1])]
        
        dim = len(k[name])
        
        if dim >= avg_dim:
            Y = torch.stack(k.pop(name)).mean(dim = 0)
            avg_k.append(Y)
    
    def forward_hook_q(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        y_size = y.shape
        # breakpoint()

        if name in q.keys():
            q[name] += [y.view(-1,y_size[-1])]
        else:
            q[name] = [y.view(-1,y_size[-1])]
        
        dim = len(q[name])
        
        if dim >= avg_dim:
            Y = torch.stack(q.pop(name)).mean(dim = 0)
            avg_q.append(Y)
    
    def forward_hook_v(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        y_size = y.shape
        # breakpoint()

        if name in v.keys():
            v[name] += [y.view(-1,y_size[-1])]
        else:
            v[name] = [y.view(-1,y_size[-1])]
        
        dim = len(v[name])
        
        if dim >= avg_dim:
            Y = torch.stack(v.pop(name)).mean(dim = 0)
            avg_v.append(Y)

    
    hooks = []
    for n, m in layer.named_modules():
        if (isinstance(m, torch.nn.Linear)) and  '.k_proj' in n :
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_k, name=n)))
            
        elif (isinstance(m, torch.nn.Linear)) and  '.q_proj' in n :
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_q, name=n)))
        
        elif (isinstance(m, torch.nn.Linear)) and  '.v_proj' in n and 'v_proj2' not in n :
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_v, name=n)))
        

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for j in range(args.nsamples):
        layer(fp_inps[j].unsqueeze(0))
    
    del k, q, v

    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()

    avg_k = torch.stack(avg_k)
    avg_q = torch.stack(avg_q)
    avg_v = torch.stack(avg_v)

    return avg_k, avg_q, avg_v


@torch.no_grad()
def get_attn_output_opt(layer, fp_inps, args):
    
    avg_o = []
    o = {}

    
    avg_dim = args.eigen_attn_params['avg_dim_features']

    def forward_hook_o(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        y_size = y.shape

        if name in o.keys():
            o[name] += [y.view(-1,y_size[-1])]
        else:
            o[name] = [y.view(-1,y_size[-1])]
        
        dim = len(o[name])
        
        if dim >= avg_dim:
            Y = torch.stack(o.pop(name)).mean(dim = 0)
            avg_o.append(Y)
    
    
    hooks = []
    for n, m in layer.named_modules():
        if (isinstance(m, torch.nn.Linear)) and  '.out_proj' in n :
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_o, name=n)))
            
        
        

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for j in range(args.nsamples):
        layer(fp_inps[j].unsqueeze(0))
    
    del o

    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()

    avg_o = torch.stack(avg_o)

    return avg_o


@torch.no_grad()
def get_attn_output_mpt(layer, fp_inps, args, position_bias, attention_mask):
    
    avg_o = []
    o = {}

    
    avg_dim = args.eigen_attn_params['avg_dim_features']

    def forward_hook_o(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        
        if isinstance(x, tuple):
            x = x[0]
        # x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        y_size = y.shape

        if name in o.keys():
            o[name] += [y.view(-1,y_size[-1])]
        else:
            o[name] = [y.view(-1,y_size[-1])]
        
        dim = len(o[name])
        
        if dim >= avg_dim:
            Y = torch.stack(o.pop(name)).mean(dim = 0)
            avg_o.append(Y)
    
    
    hooks = []
    for n, m in layer.named_modules():
        if (isinstance(m, torch.nn.Linear)) and  '.out_proj' in n :
            hooks.append(
                m.register_forward_hook(
                    partial(forward_hook_o, name=n)))
            
        
        

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for j in range(args.nsamples):
        layer(fp_inps[j].unsqueeze(0), position_bias= position_bias, attention_mask = attention_mask)
    
    del o

    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()

    avg_o = torch.stack(avg_o)

    return avg_o

def decompose_opt_layer(layer, fp_inps, args, num_heads, layer_id):

    # get calib features
    feat_k, feat_q, feat_v = get_kqv_opt(layer, fp_inps, args)


    basis_kq, eval_kq = generate_basis_vectors_per_head(torch.cat([feat_k.view(-1, feat_k.shape[-1]), feat_q.view(-1, feat_q.shape[-1])]), args.eigen_attn_params['threshold'], num_heads, args, layer_id)
    basis_v, eval_v = generate_basis_vectors_per_head(feat_v.view(-1, feat_v.shape[-1]), args.eigen_attn_params['threshold'], num_heads, args, layer_id)

    return basis_kq, eval_kq, basis_v, eval_v

def decompose_mpt_layer(layer, fp_inps, args, num_heads, layer_id, attention_mask, position_bias):

    # get calib features
    feat_k, feat_q, feat_v = get_kqv_mpt(layer, fp_inps, args, position_bias, attention_mask)

    basis_kq, eval_kq = generate_basis_vectors_per_head(torch.cat([feat_k.view(-1, feat_k.shape[-1]), feat_q.view(-1, feat_q.shape[-1])]), args.eigen_attn_params['threshold'], num_heads, args, layer_id)
    basis_v, eval_v = generate_basis_vectors_per_head(feat_v.view(-1, feat_v.shape[-1]), args.eigen_attn_params['threshold'], num_heads, args, layer_id)

    return basis_kq, eval_kq, basis_v, eval_v

def decompose_llama_layer(layer, fp_inps, args, num_heads, layer_id, attention_mask, position_ids):

    # get calib features
    feat_k, feat_q, feat_v = get_kqv_llama(layer, fp_inps, args, attention_mask, position_ids)
    basis_kq, eval_kq = generate_basis_vectors_per_layer(feat_k.view(-1, feat_k.shape[-1]), args.eigen_attn_params['threshold'], num_heads, args, layer_id)
    basis_v, eval_v = generate_basis_vectors_per_head(feat_v.view(-1, feat_v.shape[-1]), args.eigen_attn_params['threshold'], num_heads, args, layer_id)
    return basis_kq, eval_kq, basis_v, eval_v


def tucker_decompose_opt_layer(layer, fp_inps, args, num_heads, layer_id, compression_ratio=0.7):
    """
    Decompose the Q, K, V activation tensors of an OPT attention layer
    using Partial Tucker decomposition (via tucker_utils helpers).

    Args:
        layer:             Transformer layer whose Q/K/V projections are hooked.
        fp_inps:           Full-precision calibration inputs.
        args:              Argument namespace (must contain eigen_attn_params).
        num_heads:         Number of attention heads (unused, kept for API consistency).
        layer_id:          Layer index (unused, kept for API consistency).
        compression_ratio: Target Tucker compression rate in (0, 1].

    Returns:
        Dict with keys "Q", "K", "V", each mapping to a sub-dict containing
        "core", "factors", "compression_ratio", and "relative_error".
    """
    from decompose.tucker_utils import (
        factorize_dim,
        prepare_tensor,
        calculate_rank,
        _decompose_one,
    )

    # shape returned by get_kqv_opt: (nsamples / avg_dim, seq_len, dim)
    tensor_k, tensor_q, tensor_v = get_kqv_opt(layer, fp_inps, args)
    print(f"Original Shape - Q: {tensor_q.shape}, K: {tensor_k.shape}, V: {tensor_v.shape}")

    # Factorize the feature dimension into 5 sub-dimensions
    dim1_factors = factorize_dim(tensor_q.shape[2], count=5)
    print(f"Dim-1 factors ({tensor_q.shape[2]} -> {dim1_factors})")

    tensor_q = prepare_tensor(tensor_q, dim1_factors)
    tensor_k = prepare_tensor(tensor_k, dim1_factors)
    tensor_v = prepare_tensor(tensor_v, dim1_factors)
    print(f"Tensor shape after prepare_tensor: {tensor_q.shape}")

    # Calculate Tucker ranks for all modes except the batch dimension (mode 0)
    modes = list(range(1, tensor_q.ndim))
    rank = calculate_rank(tensor_q.shape, modes, compression_ratio)
    print(f"Target Ranks: {rank}")

    # Decompose each tensor (partial Tucker + reconstruction error)
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


def tensor_train_decompose_opt_layer(layer, fp_inps, args, num_heads, layer_id, ranks=None, num_factors=5):
    """
    Decompose the Q, K, V weight matrices of an OPT attention layer
    using Tensor-Train decomposition (via tensor_train_utils helpers).

    Args:
        layer:       Transformer layer containing self_attn.{q,k,v}_proj.weight.
        fp_inps:     Full-precision calibration inputs (unused, kept for API consistency).
        args:        Argument namespace (unused, kept for API consistency).
        num_heads:   Number of attention heads (unused, kept for API consistency).
        layer_id:    Layer index (unused, kept for API consistency).
        ranks:       TT-ranks.  None -> exact decomposition (uses max ranks).
                     Must be a list of length (num_factors + 1) with first/last == 1.
        num_factors: Number of TT-modes to split each weight dimension into.

    Returns:
        Dict with keys "Q", "K", "V", each mapping to a sub-dict containing
        "factors", "metadata", and "compression_ratio".
    """
    from decompose.tensor_train_utils import tensor_train_decompose

    w_q = layer.self_attn.q_proj.weight.data
    w_k = layer.self_attn.k_proj.weight.data
    w_v = layer.self_attn.v_proj.weight.data
    print(f"Original Shape - Q: {w_q.shape}, K: {w_k.shape}, V: {w_v.shape}")

    results = {}
    for name, weight in (("Q", w_q), ("K", w_k), ("V", w_v)):
        print(f"\n--- {name} ---")
        factors, metadata = tensor_train_decompose(
            weight, num_factors=num_factors, ranks=ranks
        )

        # Compression stats
        original_size = weight.numel()
        compressed_size = sum(f.numel() for f in factors)
        compression_ratio = compressed_size / original_size
        print(f"  Compression ratio: {compression_ratio:.4f}  ({compression_ratio * 100:.2f} %)")

        results[name] = {
            "factors": factors,
            "metadata": metadata,
            "compression_ratio": compression_ratio,
        }

    # Summary
    print("\n--- Summary ---")
    for name in ("Q", "K", "V"):
        r = results[name]
        print(
            f"  {name} | factors={[list(f.shape) for f in r['factors']]}"
            f"  comp_ratio={r['compression_ratio']:.4f}"
        )

    return results


def apply_tucker_factors_to_tt_cores(
    tucker_factors: list,
    tt_cores: list,
) -> list:
    """
    Contract Tucker factors into TT-matrix cores via mode multiplication.

    For each mode i:
        tucker_factors[i] has shape  (n_i, q_i)
        tt_cores[i]       has shape  (r_i, m_i, n_i, r_{i+1})

    The n_i axis is shared and is contracted out:
        new_core[r, m, q, R] = sum_n  core[r, m, n, R] * factor[n, q]

    This is a mode-2 product of the TT-matrix core with the Tucker factor matrix,
    equivalent to:
        new_core_i = einsum('rmnR, nq -> rmqR', core_i, factor_i)

    Result shape per core: (r_i, m_i, q_i, r_{i+1})

    Args:
        tucker_factors: List of k factor matrices, each of shape (n_i, q_i).
        tt_cores:       List of k TT-matrix cores, each of shape (r_i, m_i, n_i, r_{i+1}).

    Returns:
        List of k new TT-matrix cores, each of shape (r_i, m_i, q_i, r_{i+1}).
    """
    assert len(tucker_factors) == len(tt_cores), (
        f"Number of Tucker factors ({len(tucker_factors)}) must match "
        f"number of TT cores ({len(tt_cores)})"
    )

    new_cores = []
    for i, (factor, core) in enumerate(zip(tucker_factors, tt_cores)):
        # factor: (n_i, q_i)
        # core:   (r_i, m_i, n_i, r_{i+1})
        assert factor.shape[0] == core.shape[2], (
            f"Mode {i}: factor.shape[0]={factor.shape[0]} must equal "
            f"core.shape[2]={core.shape[2]} (the shared n_i dimension)"
        )
        # Contract n_i: (r, m, n, R) x (n, q) -> (r, m, q, R)
        new_core = torch.einsum('rmnR, nq -> rmqR', core, factor)
        new_cores.append(new_core)

    return new_cores


def reconstruct_combined_tt_cores(new_cores: list) -> torch.Tensor:
    """
    Reconstruct a 2-D weight matrix from TT-matrix cores that have already been
    contracted with Tucker factors (output of apply_tucker_factors_to_tt_cores).

    Each core has shape (r_i, m_i, q_i, r_{i+1}).

    Steps:
      1. tt_matrix_to_tensor(new_cores)
             → interleaved tensor of shape (m1, q1, m2, q2, ..., mk, qk)
      2. Permute to separate row and col groups:
             → (m1, m2, ..., mk, q1, q2, ..., qk)
      3. Reshape to 2-D weight matrix:
             → (m1 * m2 * ... * mk,  q1 * q2 * ... * qk)

    Args:
        new_cores: List of k TT-matrix cores, each of shape (r_i, m_i, q_i, r_{i+1}).

    Returns:
        2-D tensor of shape (M, Q) where M = prod(m_i) and Q = prod(q_i).
    """

    k = len(new_cores)

    # Collect m and q dims from the cores
    m_dims = [core.shape[1] for core in new_cores]   # row sub-dims
    q_dims = [core.shape[2] for core in new_cores]   # col sub-dims (Tucker-compressed)

    # Step 1: reconstruct — output shape is interleaved (m1, q1, m2, q2, ..., mk, qk)
    reconstructed = tt_matrix_to_tensor(new_cores)   # shape: (m1, q1, ..., mk, qk)

    # Step 2: permute to (m1, m2, ..., mk, q1, q2, ..., qk)
    # Interleaved indices:  m at 0, 2, 4, ...   q at 1, 3, 5, ...
    m_axes = list(range(0, 2 * k, 2))   # [0, 2, 4, ..., 2k-2]
    q_axes = list(range(1, 2 * k, 2))   # [1, 3, 5, ..., 2k-1]
    perm = m_axes + q_axes              # rows first, then cols
    reconstructed = reconstructed.permute(*perm)    # (m1,...,mk, q1,...,qk)

    # Step 3: reshape to 2-D
    M = math.prod(m_dims)
    Q = math.prod(q_dims)
    reconstructed = reconstructed.reshape(M, Q)

    return reconstructed


def compress_bias(bias: torch.Tensor, tucker_factors: list) -> torch.Tensor:
    """
    Treat bias (embed_dim,) like a TT-vector, apply Tucker same as W.
    
    bias:           (embed_dim,)
    tucker_factors: [F_i: (m_i, q_i)]  — same Tucker factors used for W
    row_factors:    [m1, m2, m3, m4, m5]  — same row factorization as W
    """
    # Step 1: reshape to multi-dim TT-vector format
    b = bias.reshape(*row_factors).to(torch.float32)
    # shape: (m1, m2, m3, m4, m5)

    # Step 2: TT-vector decompose
    from tensorly.decomposition._tt import tensor_train_matrix
    tt_cores = tensor_train_matrix(b, rank=[1, None, None, None, None, 1])
    # cores[i]: (r_i, m_i, r_{i+1})

    # Step 3: Apply Tucker on m_i dimension (parallel với n_i của W)
    new_cores = []
    for i, (core, F) in enumerate(zip(tt_cores.factors, tucker_factors)):
        # core: (r_i, m_i, r_{i+1})
        # F:    (m_i, q_i)
        new_core = torch.einsum('rmR, mq -> rqR', core, F)
        # new_core: (r_i, q_i, r_{i+1})
        new_cores.append(new_core)

    # Step 4: reconstruct → TT-vector → (q1, q2, q3, q4, q5) → flatten
    from tensorly.tt_matrix import tt_matrix_to_tensor
    b_comp = tt_matrix_to_tensor(new_cores)  # (q1, q2, q3, q4, q5)
    return b_comp.reshape(-1)         # (Q,)
