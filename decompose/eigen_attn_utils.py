import torch
from functools import partial
import numpy as np
import os
import scipy
import tensorly


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



def tucker_decompose_opt_layer(layer, fp_inps, args, num_heads, layer_id, ranks=None):
    """
    Thực hiện Tucker Decomposition trên Q, K, V của layer.
    
    Args:
        ranks (list): List chứa rank mong muốn cho các chiều [Rank_Token, Rank_Head, Rank_HeadDim].
                      Ví dụ: [4096, 16, 64] (Giữ nguyên Token, nén Head và Dim).
    
    Returns:
        results (dict): Chứa core và factors cho từng loại Q, K, V.
    """
    # get calib features
    tensor_k, tensor_q, tensor_v = get_kqv_opt(layer, fp_inps, args)
    
    # 1. Lấy dữ liệu Q, K, V dạng Tensor 3D: (Tokens, Heads, Head_Dim)
    print(f"Shape of Q Tensor: {tensor_q.shape}") 
    print(f"Shape of K Tensor: {tensor_k.shape}") 
    print(f"Shape of V Tensor: {tensor_v.shape}") 
    
    results = {}
    
    # Hàm con để thực hiện Tucker trên 1 tensor
    def run_tucker(tensor, name):
        print(f"Decomposing {name} with Tucker, target ranks: {ranks}...")
        X=torch.transpose(tensor.reshape(tensor.shape[0],num_heads,-1),0,1)
        
        tensor = tensor.to(torch.float32)
        
        core, factors = tensorly.tucker(tensor, rank=ranks, init='svd', tol=10e-5)
        
        # factors[0]: Ma trận liên quan đến chiều Token (Thường rất lớn, ít dùng để nén weight)
        # factors[1]: Ma trận nén chiều Heads
        # factors[2]: Ma trận nén chiều Head_Dim
        return core, factors

    if ranks is None:
        r_token = tensor_k.shape[0] # Giữ nguyên chiều token
        r_head = tensor_k.shape[1] // 2 
        r_dim = tensor_k.shape[2] // 2
        target_ranks = [r_token, r_head, r_dim]
    else:
        target_ranks = ranks

    print(f"Using target ranks for Tucker Decomposition: {target_ranks}")
    core_q, factors_q = run_tucker(tensor_q, "Query")
    core_k, factors_k = run_tucker(tensor_k, "Key")
    core_v, factors_v = run_tucker(tensor_v, "Value")

    print(f"core_q shape: {core_q.shape}, core_k shape: {core_k.shape}, core_v shape: {core_v.shape}")
    print(f"factors_q shapes: {[f.shape for f in factors_q]}")
    print(f"factors_k shapes: {[f.shape for f in factors_k]}")
    print(f"factors_v shapes: {[f.shape for f in factors_v]}")


    results = {
        "Q": {"core": core_q, "factors": factors_q},
        "K": {"core": core_k, "factors": factors_k},
        "V": {"core": core_v, "factors": factors_v}
    }
    
    return results
