"""
test_tucker_tt_inference.py
----------------------------
Quick smoke-test for the Tucker+TT-compressed OPT model with Deep Debugging support.
"""

import argparse
import sys
import os
import time
import json
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.decompose_modules import OPTTuckerTTAttention

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())

def layer_shape_report(model):
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

def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def load_calib_data(model_path, nsamples, seed, seqlen, cache_dir):
    from datautils import get_loaders
    cache_path = os.path.join(cache_dir, f"dataloader_opt_wikitext2_{nsamples}.cache")
    if os.path.exists(cache_path):
        print(f"  [cache] Loading calibration data from {cache_path}")
        dataloader = torch.load(cache_path)
    else:
        os.makedirs(cache_dir, exist_ok=True)
        dataloader, _ = get_loaders(
            "wikitext2", nsamples=nsamples, seed=seed, model=model_path, seqlen=seqlen
        )
        torch.save(dataloader, cache_path)
        print(f"  [cache] Saved calibration data to {cache_path}")
    return dataloader

# ---------------------------------------------------------------------------
# Monkey-Patching for Debug Deep
# ---------------------------------------------------------------------------

def apply_tucker_tt_attention_patch():
    original_forward = OPTTuckerTTAttention.forward

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if not hasattr(self, 'debug_cache'):
            self.debug_cache = {}

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling

        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        # Capture raw matmul outputs before masking & softmax
        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2))

        # --- DEBUG CACHE ---
        self.debug_cache["Q_mean"] = float(query_states.mean().item())
        self.debug_cache["Q_std"] = float(query_states.std().item())
        self.debug_cache["K_mean"] = float(key_states.mean().item())
        self.debug_cache["K_std"] = float(key_states.std().item())
        self.debug_cache["V_mean"] = float(value_states.mean().item())
        self.debug_cache["V_std"] = float(value_states.std().item())
        self.debug_cache["pre_softmax_attn_weights_mean"] = float(attn_weights.mean().item())
        self.debug_cache["pre_softmax_attn_weights_max"] = float(attn_weights.max().item())

        if attention_mask is not None:
            attn_weights = (attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask)
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_weights.dtype == torch.float16:
            attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_probs = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            attn_probs = layer_head_mask.view(1, -1, 1, 1) * attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
            attn_probs = attn_probs.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_reshaped = (attn_probs.view(bsz, self.num_heads, tgt_len, src_len) if output_attentions else None)

        attn_output = torch.matmul(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.compressed_dim)
        attn_output = self.out_proj_up(attn_output)

        # --- DEBUG CACHE ---
        self.debug_cache["attn_output_mean"] = float(attn_output.mean().item())
        self.debug_cache["attn_output_std"] = float(attn_output.std().item())

        return attn_output, attn_weights_reshaped, past_key_value

    OPTTuckerTTAttention.forward = patched_forward

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_text_standard(model, tokenizer, prompt: str, device, max_new_tokens: int = 40, debug: bool = False) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            return_dict_in_generate=True, output_scores=True, output_attentions=True,
        )
    output_ids = outputs.sequences
    generated = output_ids[0, inputs["input_ids"].shape[1]:]

    if debug:
        print("\n  [DEBUG] Next-token prediction probabilities:")
        scores = outputs.scores
        for step, score in enumerate(scores):
            probs = torch.softmax(score[0], dim=-1)
            top_probs, top_indices = torch.topk(probs, 5)
            print(f"    Step {step + 1}:")
            for i in range(5):
                token_str = tokenizer.decode(top_indices[i]).replace('\n', '\\n')
                print(f"      Top {i + 1}: '{token_str}' (id: {top_indices[i].item():>5}) - prob: {top_probs[i].item():.4f}")
            chosen_token = generated[step]
            chosen_str = tokenizer.decode(chosen_token).replace('\n', '\\n')
            print(f"      -> Chosen: '{chosen_str}' (id: {chosen_token.item()})")
            print()

    return tokenizer.decode(generated, skip_special_tokens=True)

def generate_text_debug_deep(
    model, baseline_model, tokenizer, prompt: str, device, 
    max_new_tokens: int, save_path: str = "", compare_baseline: bool = False
):
    model.eval()
    if compare_baseline and baseline_model:
        baseline_model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    base_input_ids = input_ids.clone() if compare_baseline else None

    trace_data = []
    past_key_values = None
    baseline_past = None

    import json
    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values, output_attentions=True)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            past_key_values = outputs.past_key_values

            base_token = None
            if compare_baseline and baseline_model:
                base_outputs = baseline_model(input_ids=base_input_ids, use_cache=True, past_key_values=baseline_past)
                base_logits = base_outputs.logits[:, -1, :]
                base_token = torch.argmax(base_logits, dim=-1).unsqueeze(-1)
                baseline_past = base_outputs.past_key_values
                base_input_ids = torch.cat([base_input_ids, base_token], dim=-1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            step_trace = {
                "step": step,
                "token_id": next_token.item(),
                "token_str": tokenizer.decode(next_token[0]),
                "baseline_token": tokenizer.decode(base_token[0]) if base_token is not None else None,
                "layer_stats": {}
            }

            for name, module in model.named_modules():
                if hasattr(module, 'debug_cache') and module.debug_cache:
                    step_trace["layer_stats"][name] = dict(module.debug_cache)
                    module.debug_cache.clear()

            trace_data.append(step_trace)
            base_info = f" | Base -> {step_trace['baseline_token']!r}" if base_token else ""
            print(f"Step {step}: Model -> {step_trace['token_str']!r}{base_info}")

    if save_path:
        with open(save_path, 'w') as f:
            for td in trace_data:
                f.write(json.dumps(td) + "\n")
        print(f"\nSaved deep debug trace to {save_path}")

    generated = input_ids[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tucker+TT OPT inference smoke-test")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--cache_dir", type=str, default="./HF_cache")
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--avg_dim", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--compression_ratio", type=float, default=0.7)
    parser.add_argument("--num_factors", type=int, default=5)
    parser.add_argument("--prompt", type=str, default="The quick brown fox jumps over the lazy dog,")
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--no_baseline", action="store_true")
    parser.add_argument("--error_budget", type=float, default=0.025)

    parser.add_argument("--debug_deep", action="store_true", help="Enable deep debug manual generation loop")
    parser.add_argument("--debug_save_path", type=str, default="", help="Save JSONL trace of deep debug metrics")
    parser.add_argument("--debug_compare_baseline", action="store_true", help="Step-by-step token compare against baseline model")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Tucker+TT OPT Inference Test")
    print(f"{'='*60}")
    print(f"  Model    : {args.model}")
    print(f"  Device   : {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    config = AutoConfig.from_pretrained(args.model, attn_implementation="eager", cache_dir=args.cache_dir)
    config.use_cache = False

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, device_map="cpu", torch_dtype=torch.float16, cache_dir=args.cache_dir
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    baseline_params = count_params(base_model)
    print(f"  Baseline parameters: {baseline_params / 1e6:.2f} M")

    baseline_text = None
    if not args.no_baseline and not args.debug_deep:
        print(f"\n[2/4] Baseline generation on {device} ...")
        base_model_dev = base_model.to(device)
        t0 = time.time()
        baseline_text = generate_text_standard(base_model_dev, tokenizer, args.prompt, device, args.max_new_tokens)
        print(f"  Baseline output  : {baseline_text!r}")
        base_model_dev.cpu()
        torch.cuda.empty_cache()
    elif not args.no_baseline:
        print("\n[2/4] Baseline generation delayed for deep debug step-by-step compare.")
    else:
        print("\n[2/4] Baseline generation SKIPPED (--no_baseline).")

    class _LMWrapper:
        def __init__(self, model, dev):
            self.model = model
            self._device = dev
            self.seqlen = model.config.max_position_embeddings
        @property
        def device(self): return self._device

    lm = _LMWrapper(base_model.to(device), device)
    import types
    ea_args = types.SimpleNamespace(
        net="opt-125m", nsamples=args.nsamples, error_budget=args.error_budget, low_rank=False,
        eigen_attn_params={
            "threshold": 0.98,
            "avg_dim_features": args.avg_dim,
            "error_budget": args.error_budget,
        },
    )

    dataloader = load_calib_data(args.model, args.nsamples, args.seed, lm.seqlen, args.cache_dir)

    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("tucker_tt_test")

    print("\n[3/4] Running Tucker+TT decomposition (this may take a while) ...")
    from decompose.eigen_attn import eigenattn
    eigenattn(lm, ea_args, dataloader, logger)

    compressed_model = lm.model
    compressed_model.eval()

    compressed_params = count_params(compressed_model)
    print(f"\n  Baseline parameters  : {baseline_params / 1e6:.2f} M")
    print(f"  Compressed parameters: {compressed_params / 1e6:.2f} M")
    print(f"  Param ratio          : {compressed_params / baseline_params:.4f}")

    if args.debug_deep:
        apply_tucker_tt_attention_patch()
        infer_device = model_device(compressed_model)
        compressed_model.config.use_cache = True
        compressed_model = compressed_model.to(infer_device)

        baseline_model_compare = None
        if args.debug_compare_baseline:
            baseline_model_compare = AutoModelForCausalLM.from_pretrained(
                args.model, config=config, device_map="cpu", torch_dtype=torch.float16, cache_dir=args.cache_dir
            ).to(infer_device)
            baseline_model_compare.eval()

        print(f"\n[4/4] Deep Debug Generation (manual greedy loop) ...")
        compressed_text = generate_text_debug_deep(
            compressed_model, baseline_model_compare, tokenizer, args.prompt,
            infer_device, args.max_new_tokens, args.debug_save_path, args.debug_compare_baseline
        )

    else:
        infer_device = model_device(compressed_model)
        print(f"\n[4/4] Compressed model generation (model on {infer_device}) ...")
        compressed_model.config.use_cache = True
        compressed_model = compressed_model.to(infer_device)
        compressed_text = generate_text_standard(
            compressed_model, tokenizer, args.prompt, infer_device, args.max_new_tokens, debug=True
        )

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Prompt           : {args.prompt!r}")
    if baseline_text is not None:
        print(f"  Baseline output  : {baseline_text!r}")
    print(f"  Compressed output: {compressed_text!r}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
