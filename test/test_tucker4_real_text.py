"""
test_tucker4_real_text.py
=========================
Dùng hidden-state embedding thật của một câu qua OPT-125m.

Flow:
    hidden:    (1, seq_len, 768)
    squeeze:   (seq_len, 768)
    reshape:   (seq_len, 8, 8, 12)

Sau đó Partial Tucker chỉ decompose các mode embedding (1, 2, 3),
giữ nguyên mode 0 là seq_len.

Chạy:
    python test_tucker4_real_text.py
"""

import math
import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import partial_tucker
from tensorly.tenalg import multi_mode_dot
from transformers import AutoTokenizer, OPTForCausalLM


# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
MODEL_NAME = "facebook/opt-125m"
SENTENCE   = "The quick brown fox jumps over the lazy dog."
DEVICE     = "cpu"

# Reshape hidden_size = 768 thành 8 x 8 x 12
EMBED_RESHAPE = (8, 8, 12)

# Giữ nguyên mode 0 = seq_len, chỉ Tucker trên mode 1, 2, 3
TUCKER_MODES = [1, 2, 3]

# Rank tương ứng với các mode 1, 2, 3
# Ví dụ: (8, 8, 12) -> (4, 4, 6)
RANKS = [7, 7, 11]


# ---------------------------------------------------------------------------
# Hàm thu thập hidden-state từ layer đầu tiên
# ---------------------------------------------------------------------------
def get_first_layer_input(model_name: str, sentence: str, device: str):
    """
    Tokenize câu -> forward qua embedding + positional encoding của OPT
    -> thu hidden-state đầu vào của decoder layer 0.

    Trả về tensor shape (1, seq_len, hidden_size).
    """
    print(f"[*] Loading tokenizer & model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Nếu transformers bản cũ báo lỗi dtype, đổi lại thành torch_dtype=torch.float32
    model = OPTForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
    )

    model.eval()
    model.to(device)

    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    print(f"[*] Tokenized sentence → {input_ids.shape[1]} tokens")

    captured = {}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            captured["inp"] = inp.detach().cpu()
            raise ValueError("stop here")

    layers = model.model.decoder.layers
    layers[0] = Catcher(layers[0])

    with torch.no_grad():
        try:
            model(**inputs)
        except ValueError:
            pass

    layers[0] = layers[0].module

    hidden = captured["inp"]
    print(f"[*] Captured hidden state shape: {hidden.shape}")

    return hidden


# ---------------------------------------------------------------------------
# Metrics / param count
# ---------------------------------------------------------------------------
def reconstruction_metrics(tensor, core, factors, modes):
    """
    Tính reconstruction error trên tensor đã reshape,
    tức là trước khi reshape ngược về (seq_len, 768).
    """
    reconstructed = multi_mode_dot(core, factors, modes=modes)

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
    """
    Đếm param theo kết quả thực tế TensorLy trả ra,
    không dùng RANKS gốc để tránh sai khi rank bị clip.
    """
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


# ---------------------------------------------------------------------------
# Hàm chạy một thí nghiệm
# ---------------------------------------------------------------------------
def run_one_experiment(
    tensor,
    rank,
    modes,
    title,
    n_iter_max=200,
    init="svd",
    random_state=None,
):
    print(f"\n=== {title} ===")
    print(f"tensor shape: {tensor.shape}")
    print(f"modes      : {modes}")
    print(f"rank       : {rank}")

    (core, factors), errors = partial_tucker(
        tensor,
        rank=rank,
        modes=modes,
        n_iter_max=n_iter_max,
        init=init,
        random_state=random_state,
        verbose=False,
    )

    metrics = reconstruction_metrics(tensor, core, factors, modes)

    print("\n--- Reconstruction Metrics on Reshaped Tensor ---")
    print(f"norm_tensor : {metrics['norm_tensor']:.8f}")
    print(f"norm_rec    : {metrics['norm_rec']:.8f}")
    print(f"norm_diff   : {metrics['norm_diff']:.8f}")
    print(f"abs_error   : {metrics['abs_error']:.8f}")
    print(f"rel_error   : {metrics['rel_error']:.8f}")

    print("\n--- Shapes ---")
    print(f"tensor shape: {tensor.shape}")
    print(f"core shape  : {core.shape}")

    for i, factor in enumerate(factors):
        mode = modes[i]
        print(f"factor for mode {mode} shape: {factor.shape}")

    params = tucker_param_count_from_result(tensor, core, factors)

    print("\n--- Parameter Count ---")
    print(f"original params           : {params['original_params']}")
    print(f"core params               : {params['core_params']}")
    print(f"factor params             : {params['factor_params']}")
    print(f"compressed params         : {params['compressed_params']}")
    print(f"core-only ratio           : {params['compression_ratio_core_only']:.4f}")
    print(f"ratio with Tucker factors : {params['compression_ratio_with_factors']:.4f}")

    return core, factors, errors, metrics


# ---------------------------------------------------------------------------
# Demo chính
# ---------------------------------------------------------------------------
def run_real_text_tucker_demo():
    # 1. Thu thập embedding thật
    hidden = get_first_layer_input(MODEL_NAME, SENTENCE, DEVICE)

    # hidden: (1, seq_len, hidden_size)
    hidden_2d = hidden.squeeze(0)  # (seq_len, 768)

    seq_len, hidden_size = hidden_2d.shape
    expected_hidden_size = math.prod(EMBED_RESHAPE)

    if hidden_size != expected_hidden_size:
        raise ValueError(
            f"Cannot reshape hidden_size={hidden_size} into {EMBED_RESHAPE}, "
            f"because product={expected_hidden_size}"
        )

    # 2. Reshape embedding dimension: 768 -> (8, 8, 12)
    hidden_4d = hidden_2d.reshape(seq_len, *EMBED_RESHAPE)

    # Chuyển sang tensorly tensor
    tensor = tl.tensor(hidden_4d.numpy())

    print(f"\n[*] Original hidden_2d shape : {tuple(hidden_2d.shape)}")
    print(f"[*] Reshaped tensor shape    : {tensor.shape}")
    print(f"[*] Tucker modes             : {TUCKER_MODES}")
    print(f"[*] Tucker ranks             : {RANKS}")

    # 3. Partial Tucker trên mode 1, 2, 3
    core, factors, errors, metrics = run_one_experiment(
        tensor,
        rank=RANKS,
        modes=TUCKER_MODES,
        title="Partial Tucker – Real Text Embedding Reshaped",
        n_iter_max=200,
    )

    # 4. Nếu muốn kiểm tra reshape ngược sau reconstruct
    reconstructed_4d = metrics["reconstructed"]
    reconstructed_2d = tl.reshape(reconstructed_4d, (seq_len, hidden_size))

    print("\n--- Reshape Back Check ---")
    print(f"reconstructed_4d shape: {reconstructed_4d.shape}")
    print(f"reconstructed_2d shape: {reconstructed_2d.shape}")

    # 5. Random state consistency
    print("\n=== Random State Consistency ===")

    (core1, factors1), _ = partial_tucker(
        tensor,
        rank=RANKS,
        modes=TUCKER_MODES,
        random_state=0,
        init="random",
    )

    (core2, factors2), _ = partial_tucker(
        tensor,
        rank=RANKS,
        modes=TUCKER_MODES,
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
        mode = TUCKER_MODES[i]
        print(f"factor for mode {mode} max abs diff: {diff}")


if __name__ == "__main__":
    run_real_text_tucker_demo()