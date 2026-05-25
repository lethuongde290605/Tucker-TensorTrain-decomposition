import torch
import math

def build_u_full(factors):
    # factors: list of (orig_i, comp_i)
    # returns U_full of shape (prod(orig_i), prod(comp_i))
    res = factors[0]
    for f in factors[1:]:
        # kronecker product
        # res: (O_prev, C_prev)
        # f: (O_new, C_new)
        O_prev, C_prev = res.shape
        O_new, C_new = f.shape
        # res_exp = res.view(O_prev, 1, C_prev, 1)
        # f_exp = f.view(1, O_new, 1, C_new)
        # return (res_exp * f_exp).view(O_prev * O_new, C_prev * C_new)
        res = torch.kron(res, f)
    return res

f1 = torch.randn(4, 2)
f2 = torch.randn(3, 2)
res = build_u_full([f1, f2])
print(res.shape)
