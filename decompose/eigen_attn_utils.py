import torch
from functools import partial
import numpy as np
import os
import scipy
import tensorly
from tensorly.decomposition import tensor_train
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


def tucker_decompose_opt_layer(layer, fp_inps, args, num_heads, layer_id, ranks=None):
    
    # shape = (nsamples / avg_dim, sq_len = 2048, dim = 768)
    tensor_k, tensor_q, tensor_v = get_kqv_opt(layer, fp_inps, args)
    print(f"Original Shape - Q: {tensor_q.shape}, K: {tensor_k.shape}, V: {tensor_v.shape}") 

    head_dim = tensor_q.shape[-1] // num_heads # 768 / 12 = 64
    
    
    def prepare_tensor(t):
        # [32, 2048, 768] -> [32, 2048, 12, 64]
        t = t.view(t.shape[0], t.shape[1], num_heads, head_dim)
        # keep batch_size and seq_len: [65536, 12, 64]
        t = t.view(-1, num_heads, head_dim)
        # reshape tensor for Tucker: [65536, 3, 4, 4, 4, 4]
        t = t.view(-1, 3, 4, 4, 4, 4)
        return t

    tensor_q = prepare_tensor(tensor_q)
    tensor_k = prepare_tensor(tensor_k)
    tensor_v = prepare_tensor(tensor_v)
    
    print(f"Reshaped for Tucker - Q: {tensor_q.shape}") 

    if ranks is None:
        target_ranks = [tensor_q.shape[0], 2, 2, 2, 2, 2]
    else:
        target_ranks = ranks

    print(f"Target Ranks: {target_ranks}")

    results = {}
    
    def run_tucker(tensor, name, target_ranks):
        print(f"Decomposing {name}...")
        
        # tensorly only support for float32
        tensor = tensor.to(torch.float32)
        
        with torch.cuda.amp.autocast(enabled=False):
            core, factors = tensorly.decomposition.tucker(tensor, rank=target_ranks, init='svd')
            # core, factors = tensorly.decomposition.tucker(
            #     tensor,
            #     rank=target_ranks,
            #     init=(core0, factors0),   
            #     fixed_factors=[0], 
            # )

        
        return core, factors

    core_q, factors_q = run_tucker(tensor_q, "Query", target_ranks)
    core_k, factors_k = run_tucker(tensor_k, "Key", target_ranks)
    core_v, factors_v = run_tucker(tensor_v, "Value", target_ranks)

    print(f"Factors Q shapes: {[f.shape for f in factors_q]}")

    results = {
        "Q": {"core": core_q, "factors": factors_q},
        "K": {"core": core_k, "factors": factors_k},
        "V": {"core": core_v, "factors": factors_v}
    }
    
    return results


def tensor_train_decompose_opt_layer(layer, fp_inps, args, num_heads, layer_id, ranks=None):
    # Input: W_q, W_k, W_v 
    w_q = layer.self_attn.q_proj.weight.data  # shape: (hidden_dim, hidden_dim) = (768, 768)
    w_k = layer.self_attn.k_proj.weight.data 
    w_v = layer.self_attn.v_proj.weight.data
    print(f"Original Shape - Q: {w_q.shape}, K: {w_k.shape}, V: {w_v.shape}") 
    print(f"Q: {w_q}")
    
    # Ta tách hidden_dim 768 = 4x4x4x4x3
    # Reshape (768, 768) -> (16, 16, 16, 16, 9) theo TT-matrix format
    def prepare_tensor(t):
        # (768, 768) -> (4, 4, 4, 4, 3, 4, 4, 4, 4, 3)
        t = t.view(4, 4, 4, 4, 3, 4, 4, 4, 4, 3)
        # Interleave row/col factors: (0,5), (1,6), (2,7), (3,8), (4,9)
        t = t.permute(0, 5, 1, 6, 2, 7, 3, 8, 4, 9)
        # Reshape to combine pairs: (4*4, 4*4, 4*4, 4*4, 3*3) = (16, 16, 16, 16, 9)
        t = t.reshape(16, 16, 16, 16, 9)
        return t

    w_q = prepare_tensor(w_q)
    w_k = prepare_tensor(w_k)
    w_v = prepare_tensor(w_v)
    print(f"Reshaped for TensorTrain - Q: {w_q.shape}")  # Expected: (16, 16, 16, 16, 9)
    print(f"Q after reshape: {w_q}")
    # Tensor 5 chiều => List rank cần 6 phần tử: [1, r1, r2, r3, r4, 1]
    if ranks is None:
        # Đặt rank mặc định dựa trên kích thước dimensions
        r1 = min(w_q.shape[0], w_q.shape[1])  # min(16, 16) = 16
        r2 = min(w_q.shape[1], w_q.shape[2])  # min(16, 16) = 16
        r3 = min(w_q.shape[2], w_q.shape[3])  # min(16, 16) = 16
        r4 = min(w_q.shape[3], w_q.shape[4])  # min(16, 9) = 9
        target_ranks = [1, r1, r2, r3, r4, 1]
        
    elif isinstance(ranks, int):
        # Nếu truyền 1 số nguyên => dùng chung cho các rank ở giữa
        target_ranks = [1, ranks, ranks, ranks, ranks, 1]
        
    elif isinstance(ranks, list):
        if len(ranks) == 4:
            # Nếu truyền list [r1, r2, r3, r4] => thêm 1 vào đầu và cuối
            target_ranks = [1] + ranks + [1]
        elif len(ranks) == 6:
            # Nếu đã truyền đủ
            target_ranks = ranks
        else:
            raise ValueError("Với Tensor 5 chiều, ranks phải là list 4 hoặc 6 phần tử.")

    print(f"Target Ranks (TT-Ranks): {target_ranks}")

    def run_tensor_train(tensor, name, target_ranks):
        print(f"Decomposing {name} with shape {tensor.shape}...")
        
        tensor = tensor.to(torch.float32)
        
        with torch.cuda.amp.autocast(enabled=False):
            tt_tensor = tensor_train(tensor, rank=target_ranks, verbose=False)
        
        # Lấy danh sách các factors
        # Factors sẽ gồm 3 tensor con:
        # F1: (1, rows, r1)
        # F2: (r1, heads, r2)
        # F3: (r2, dim, 1)
        factors = tt_tensor.factors 
        
        return factors

    # Thực hiện phân rã
    factors_q = run_tensor_train(w_q, "Query", target_ranks)
    factors_k = run_tensor_train(w_k, "Key", target_ranks)
    factors_v = run_tensor_train(w_v, "Value", target_ranks)

    print(f"Factors Q shapes (before reshape): {[f.shape for f in factors_q]}")

    # Reshape factors từ (r1, m*n, r2) -> (m, r1, n, r2)
    # Với tensor 5D (16, 16, 16, 16, 9), mỗi chiều = m*n trong đó m=n=4 hoặc m=n=3
    # Các chiều: 16=4*4, 16=4*4, 16=4*4, 16=4*4, 9=3*3
    dim_factors = [(4, 4), (4, 4), (4, 4), (4, 4), (3, 3)]  # (m, n) cho mỗi chiều
    
    def reshape_factors(factors):
        reshaped = []
        for i, f in enumerate(factors):
            # f có shape (r1, m*n, r2)
            r1, mn, r2 = f.shape
            m, n = dim_factors[i]
            # Reshape (r1, m*n, r2) -> (r1, m, n, r2) -> (m, r1, n, r2)
            f = f.view(r1, m, n, r2)
            f = f.permute(1, 0, 2, 3)  # (m, r1, n, r2)
            reshaped.append(f)
        return reshaped
    
    factors_q = reshape_factors(factors_q)
    factors_k = reshape_factors(factors_k)
    factors_v = reshape_factors(factors_v)

    print(f"Factors Q shapes (after reshape): {[f.shape for f in factors_q]}")

    results = {
        "Q": {"factors": factors_q},
        "K": {"factors": factors_k},
        "V": {"factors": factors_v}
    }
    
    return results