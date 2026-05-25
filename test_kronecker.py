import torch
import tensorly as tl
tl.set_backend('pytorch')

N = 2
d1, r1 = 4, 3
d2, r2 = 5, 2

U1 = torch.randn(d1, r1)
U2 = torch.randn(d2, r2)
C = torch.randn(N, r1, r2)

# Tensorly construct
T1 = tl.tenalg.multi_mode_dot(C, [U1, U2], modes=[1, 2])
T1_flat = T1.reshape(N, -1)

# Kronecker construct
U_kron = torch.kron(U1, U2) # shape (d1*d2, r1*r2)
print("U_kron shape:", U_kron.shape)
C_flat = C.reshape(N, -1)
T2_flat = torch.matmul(C_flat, U_kron.T)

print("Diff kron(U1, U2):", torch.norm(T1_flat - T2_flat).item())

U_kron_rev = torch.kron(U2, U1) 
print("Diff kron(U2, U1):", torch.norm(T1_flat - torch.matmul(C_flat, U_kron_rev.T)).item())

