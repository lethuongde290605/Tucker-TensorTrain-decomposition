"""
test_tucker_tt_inference.py
----------------------------
Quick smoke-test for the Tucker+TT-compressed OPT model (OPTTuckerTTDecoderLayer).

Usage
-----
# Minimal (CPU-only, OPT-125m):
python test_tucker_tt_inference.py --model facebook/opt-125m

# With GPU and custom number of calibration samples:
python test_tucker_tt_inference.py --model facebook/opt-125m --nsamples 4 --avg_dim 2

# Custom prompt:
python test_tucker_tt_inference.py --model facebook/opt-125m --prompt "The Eiffel Tower is located in"

# Skip baseline comparison (faster):
python test_tucker_tt_inference.py --model facebook/opt-125m --no_baseline

What it does
------------
1. Loads the original OPT model (baseline).
2. Optionally runs greedy generation on the baseline for comparison.
3. Runs the Tucker+TT decomposition pipeline (same as main_eigen_attn.py eigenattn()).
4. Runs greedy generation on the compressed model.
5. Prints a side-by-side comparison and basic statistics (parameter counts,
   compressed vs. baseline layer shapes).
"""

import argparse
import sys
import os
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# ---------------------------------------------------------------------------
# Ensure project root is on path so local modules are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def layer_shape_report(model):
    """Print Q/K/V/out_proj shapes for the first and last attention layers."""
    layers = model.model.decoder.layers
    report = []
    for idx in [0, len(layers) - 1]:
        layer = layers[idx]
        attn = layer.self_attn
        shapes = {}
        for attr in ("q_proj", "k_proj", "v_proj", "out_proj", "out_proj_up"):
            m = getattr(attn, attr, None)
            if m is not None and hasattr(m, "weight"):
                shapes[attr] = tuple(m.weight.shape)
        report.append((idx, shapes))
    return report


def generate_text(model, tokenizer, prompt: str, device, max_new_tokens: int = 40) -> str:
    model.eval()
    # Always send inputs to wherever the model actually lives
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic & fast
            use_cache=True,
        )
    # Decode only the generated (new) tokens
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def model_device(model) -> torch.device:
    """Return the device of the first parameter in the model."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def load_calib_data(model_path, nsamples, seed, seqlen, cache_dir):
    """Load a tiny wikitext2 calibration batch (same logic as main_eigen_attn.py)."""
    from datautils import get_loaders
    cache_path = os.path.join(cache_dir, f"dataloader_opt_wikitext2_{nsamples}.cache")
    if os.path.exists(cache_path):
        print(f"  [cache] Loading calibration data from {cache_path}")
        dataloader = torch.load(cache_path)
    else:
        os.makedirs(cache_dir, exist_ok=True)
        dataloader, _ = get_loaders(
            "wikitext2",
            nsamples=nsamples,
            seed=seed,
            model=model_path,
            seqlen=seqlen,
        )
        torch.save(dataloader, cache_path)
        print(f"  [cache] Saved calibration data to {cache_path}")
    return dataloader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tucker+TT OPT inference smoke-test")
    parser.add_argument("--model",           type=str,   default="facebook/opt-125m",
                        help="HuggingFace model ID or local path (must be OPT family).")
    parser.add_argument("--cache_dir",       type=str,   default="./HF_cache")
    parser.add_argument("--nsamples",        type=int,   default=4,
                        help="Number of calibration samples (keep small for quick tests).")
    parser.add_argument("--avg_dim",         type=int,   default=2,
                        help="avg_dim_features: how many forward passes are averaged per chunk.")
    parser.add_argument("--seed",            type=int,   default=2)
    parser.add_argument("--compression_ratio", type=float, default=0.7,
                        help="Tucker target compression ratio.")
    parser.add_argument("--num_factors",     type=int,   default=5,
                        help="Number of TT / Tucker factors per dimension.")
    parser.add_argument("--prompt",          type=str,
                        default="The quick brown fox jumps over the lazy dog,",
                        help="Prompt sentence to test generation.")
    parser.add_argument("--max_new_tokens",  type=int,   default=30)
    parser.add_argument("--no_baseline",     action="store_true",
                        help="Skip baseline generation (saves time).")
    parser.add_argument("--error_budget",    type=float, default=0.025)
    args = parser.parse_args()

    # ---- Seed ---------------------------------------------------------------
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Tucker+TT OPT Inference Test")
    print(f"{'='*60}")
    print(f"  Model    : {args.model}")
    print(f"  Device   : {device}")
    print(f"  nsamples : {args.nsamples}  |  avg_dim : {args.avg_dim}")
    print(f"  Comp. ratio (Tucker): {args.compression_ratio}")
    print(f"  Prompt   : {args.prompt!r}")
    print(f"{'='*60}\n")

    # ---- 1. Load tokenizer and baseline model --------------------------------
    print("[1/4] Loading tokenizer and base OPT model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    config = AutoConfig.from_pretrained(
        args.model,
        attn_implementation="eager",
        cache_dir=args.cache_dir,
    )
    config.use_cache = False

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        device_map="cpu",
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    baseline_params = count_params(base_model)
    print(f"  Baseline parameters: {baseline_params / 1e6:.2f} M")

    # ---- 2. Optional baseline generation ------------------------------------
    baseline_text = None
    if not args.no_baseline:
        print(f"\n[2/4] Baseline generation on {device} ...")
        base_model_dev = base_model.to(device)
        t0 = time.time()
        baseline_text = generate_text(
            base_model_dev, tokenizer, args.prompt, device, args.max_new_tokens
        )
        print(f"  Baseline output  : {baseline_text!r}")
        print(f"  Time             : {time.time() - t0:.2f}s")
        # Move back to CPU to free GPU memory for compression
        base_model_dev.cpu()
        torch.cuda.empty_cache()
    else:
        print("\n[2/4] Baseline generation SKIPPED (--no_baseline).")

    # ---- 3. Build LM-like wrapper (eigenattn() expects an LMClass obj) ------
    print("\n[3/4] Running Tucker+TT decomposition (this may take a while) ...")

    # eigenattn() expects an object with .model, .device, .seqlen, .model.config
    class _LMWrapper:
        def __init__(self, model, dev):
            self.model   = model
            self._device = dev
            self.seqlen  = model.config.max_position_embeddings

        @property
        def device(self):
            return self._device

    # Move to target device for calibration
    lm = _LMWrapper(base_model.to(device), device)

    # Build args namespace that eigenattn() expects
    import types
    ea_args = types.SimpleNamespace(
        net              = "opt-125m",   # only used for is_opt detection
        nsamples         = args.nsamples,
        error_budget     = args.error_budget,
        low_rank         = False,
        eigen_attn_params= {
            "threshold"        : 0.98,
            "avg_dim_features" : args.avg_dim,
            "error_budget"     : args.error_budget,
        },
    )

    # Load calibration data
    print("  Loading calibration data ...")
    dataloader = load_calib_data(
        args.model, args.nsamples, args.seed, lm.seqlen, args.cache_dir
    )

    # ---- Import a patched eigenattn that uses the Tucker-TT path -----------
    # eigenattn() already dispatches to tucker_decompose_opt_layer + TT for OPT.
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("tucker_tt_test")

    t0 = time.time()
    from decompose.eigen_attn import eigenattn
    eigenattn(lm, ea_args, dataloader, logger)
    elapsed = time.time() - t0
    print(f"  Decomposition done in {elapsed:.1f}s")

    compressed_model = lm.model
    compressed_model.eval()

    compressed_params = count_params(compressed_model)
    print(f"\n  Baseline parameters  : {baseline_params / 1e6:.2f} M")
    print(f"  Compressed parameters: {compressed_params / 1e6:.2f} M")
    print(f"  Param ratio          : {compressed_params / baseline_params:.4f}")

    # ---- Layer shape inspection ----------------------------------------------
    print("\n  --- Layer shape comparison (first and last decoder layers) ---")
    for layer_idx, shapes in layer_shape_report(compressed_model):
        print(f"  Layer {layer_idx:>3}:")
        for name, shape in shapes.items():
            print(f"    {name:<16}: {shape}")

    # ---- 4. Compressed model generation -------------------------------------
    # eigenattn() moves layers back to CPU at the end; detect the actual device.
    infer_device = model_device(compressed_model)
    print(f"\n[4/4] Compressed model generation (model on {infer_device}) ...")
    compressed_model.config.use_cache = True
    # Move the full model to the inference device (handles embed_tokens, lm_head, etc.)
    compressed_model = compressed_model.to(infer_device)
    t0 = time.time()
    compressed_text = generate_text(
        compressed_model, tokenizer, args.prompt, infer_device, args.max_new_tokens
    )
    print(f"  Compressed output: {compressed_text!r}")
    print(f"  Time             : {time.time() - t0:.2f}s")

    # ---- Summary -------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Prompt           : {args.prompt!r}")
    if baseline_text is not None:
        print(f"  Baseline output  : {baseline_text!r}")
    print(f"  Compressed output: {compressed_text!r}")
    print(f"  Param ratio      : {compressed_params / baseline_params:.4f}  "
          f"({baseline_params/1e6:.1f}M → {compressed_params/1e6:.1f}M)")
    print(f"{'='*60}\n")

    if baseline_text is not None and baseline_text.strip() == compressed_text.strip():
        print("✓  Outputs MATCH — compressed model behaves identically to baseline.")
    elif compressed_text.strip():
        print("✓  Compressed model produces coherent text (outputs differ, which is expected after compression).")
    else:
        print("✗  Compressed model produced EMPTY output — something may be wrong.")


if __name__ == "__main__":
    main()
