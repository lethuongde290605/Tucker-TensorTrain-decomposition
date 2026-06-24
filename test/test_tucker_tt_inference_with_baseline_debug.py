"""
test_tucker_tt_inference.py
----------------------------
Quick smoke-test for the Tucker+TT-compressed OPT model with Deep Debugging support.

Extra debug features:
- Compressed model layer-level Q/K/V/pre-softmax/attn-output stats.
- Optional baseline model layer-level Q/K/V/pre-softmax/attn-output stats.
- Optional layer-by-layer compressed-vs-baseline comparison in JSONL.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Ensure project root is on path so local modules are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.decompose_modules import OPTTuckerTTAttention


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _safe_float(x) -> float:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().item()
    return float(x)


def tensor_stats(x: Optional[torch.Tensor]) -> Dict[str, float]:
    if x is None:
        return {}
    x = x.detach().float()
    return {
        "mean": _safe_float(x.mean()),
        "std": _safe_float(x.std()),
        "min": _safe_float(x.min()),
        "max": _safe_float(x.max()),
        "norm": _safe_float(x.norm()),
    }


def cosine_sim(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> float:
    if a is None or b is None:
        return 0.0
    a = a.detach().float().view(-1)
    b = b.detach().float().view(-1)
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return _safe_float(F.cosine_similarity(a, b, dim=0))


def kl_div_from_log_probs(log_p: Optional[torch.Tensor], log_q: Optional[torch.Tensor]) -> float:
    """
    Return KL(P || Q), where log_p and log_q are log-prob tensors.
    Shape expected: [batch, vocab].
    """
    if log_p is None or log_q is None:
        return 0.0
    log_p = log_p.detach().float()
    log_q = log_q.detach().float()
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1).mean()
    return _safe_float(kl)


def entropy_from_probs(probs: torch.Tensor) -> float:
    probs = probs.detach().float()
    ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()
    return _safe_float(ent)


def safe_ratio(num, den, eps: float = 1e-8):
    if num is None or den is None:
        return None
    den = float(den)
    if abs(den) < eps:
        return None
    return float(num) / den


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
    """Load a small wikitext2 calibration batch."""
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
# Debug cache collection and layer comparison
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(r"(?:^|\.)(?:decoder\.)?layers\.(\d+)\.self_attn$")


def normalize_attn_layer_name(name: str) -> str:
    """Normalize module names like model.decoder.layers.3.self_attn -> layer_3."""
    m = _LAYER_RE.search(name)
    if m:
        return f"layer_{int(m.group(1))}"
    return name


def collect_debug_cache(model) -> Dict[str, Dict[str, float]]:
    """
    Collect debug_cache from modules and clear it.

    Returns keys normalized as layer_0, layer_1, ... when possible.
    The original module name is stored inside each dict under `_module_name`.
    """
    out = {}
    for name, module in model.named_modules():
        cache = getattr(module, "debug_cache", None)
        if cache:
            key = normalize_attn_layer_name(name)
            value = dict(cache)
            value["_module_name"] = name
            out[key] = value
            module.debug_cache.clear()
    return out


def compare_layer_stats(
    compressed_stats: Dict[str, Dict[str, float]],
    baseline_stats: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Build compact layer-by-layer compressed-vs-baseline comparisons."""
    compared = {}
    common_layers = sorted(
        set(compressed_stats.keys()).intersection(baseline_stats.keys()),
        key=lambda k: int(k.split("_")[-1]) if k.startswith("layer_") else 10**9,
    )

    for layer_key in common_layers:
        c = compressed_stats[layer_key]
        b = baseline_stats[layer_key]

        row = {
            "compressed_module_name": c.get("_module_name"),
            "baseline_module_name": b.get("_module_name"),
        }

        for field in [
            "Q_norm",
            "K_norm",
            "V_norm",
            "attn_output_norm",
            "attn_entropy",
            "pre_softmax_attn_weights_mean",
            "pre_softmax_attn_weights_max",
        ]:
            cv = c.get(field)
            bv = b.get(field)
            row[f"compressed_{field}"] = cv
            row[f"baseline_{field}"] = bv
            row[f"{field}_ratio"] = safe_ratio(cv, bv)
            if cv is not None and bv is not None:
                row[f"{field}_diff"] = float(cv) - float(bv)

        compared[layer_key] = row

    return compared


def summarize_layer_compare(layer_compare: Dict[str, Dict[str, float]]) -> str:
    """Short one-line console summary for layer_compare."""
    if not layer_compare:
        return ""

    def max_by_abs(field):
        best_key, best_val = None, None
        for layer_key, row in layer_compare.items():
            val = row.get(field)
            if val is None:
                continue
            if best_val is None or abs(val) > abs(best_val):
                best_key, best_val = layer_key, val
        return best_key, best_val

    attn_layer, attn_ratio = max_by_abs("attn_output_norm_ratio")
    q_layer, q_ratio = max_by_abs("Q_norm_ratio")
    score_layer, score_diff = max_by_abs("pre_softmax_attn_weights_max_diff")

    parts = []
    if attn_layer is not None:
        parts.append(f"max attn_out ratio: {attn_layer}={attn_ratio:.3f}")
    if q_layer is not None:
        parts.append(f"max Q ratio: {q_layer}={q_ratio:.3f}")
    if score_layer is not None:
        parts.append(f"max pre-softmax max diff: {score_layer}={score_diff:.3f}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Monkey patches for compressed Tucker+TT and baseline OPT attention
# ---------------------------------------------------------------------------

def apply_tucker_tt_attention_patch():
    """Patch OPTTuckerTTAttention.forward to store layer-level debug_cache."""
    if getattr(OPTTuckerTTAttention, "_debug_patched", False):
        return

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if not hasattr(self, "debug_cache"):
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
        pre_softmax_scores = torch.matmul(query_states, key_states.transpose(1, 2))

        # Debug before mask/softmax, focusing on last query token.
        last_scores = pre_softmax_scores[:, -1, :].detach().float()
        self.debug_cache.update({
            "Q_norm": _safe_float(query_states[:, -1, :].detach().float().norm()),
            "K_norm": _safe_float(key_states.detach().float().norm()),
            "V_norm": _safe_float(value_states.detach().float().norm()),
            "Q_mean": _safe_float(query_states[:, -1, :].detach().float().mean()),
            "K_mean": _safe_float(key_states.detach().float().mean()),
            "V_mean": _safe_float(value_states.detach().float().mean()),
            "pre_softmax_attn_weights_mean": _safe_float(last_scores.mean()),
            "pre_softmax_attn_weights_max": _safe_float(last_scores.max()),
            "pre_softmax_attn_weights_min": _safe_float(last_scores.min()),
            "pre_softmax_attn_weights_std": _safe_float(last_scores.std()),
        })

        attn_weights = pre_softmax_scores
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, "
                f"but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, "
                    f"but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_weights.dtype == torch.float16:
            attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_probs = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, "
                    f"but is {layer_head_mask.size()}"
                )
            attn_probs = layer_head_mask.view(1, -1, 1, 1) * attn_probs.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_probs = attn_probs.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_reshaped = (
            attn_probs.view(bsz, self.num_heads, tgt_len, src_len) if output_attentions else None
        )

        recent_probs = attn_probs.view(bsz, self.num_heads, tgt_len, src_len)[:, :, -1, :].detach().float()
        self.debug_cache["attn_entropy"] = entropy_from_probs(recent_probs)
        self.debug_cache["attn_prob_max"] = _safe_float(recent_probs.max())
        self.debug_cache["attn_prob_mean"] = _safe_float(recent_probs.mean())

        attn_output_low = torch.matmul(attn_probs, value_states)
        attn_output_low = attn_output_low.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_low = attn_output_low.transpose(1, 2)
        attn_output_low = attn_output_low.reshape(bsz, tgt_len, self.compressed_dim)
        attn_output = self.out_proj_up(attn_output_low)

        last_attn_output = attn_output[:, -1, :].detach().float()
        self.debug_cache["attn_output_norm"] = _safe_float(last_attn_output.norm())
        self.debug_cache["attn_output_mean"] = _safe_float(last_attn_output.mean())
        self.debug_cache["attn_output_std"] = _safe_float(last_attn_output.std())
        self.debug_cache["attn_output_min"] = _safe_float(last_attn_output.min())
        self.debug_cache["attn_output_max"] = _safe_float(last_attn_output.max())

        return attn_output, attn_weights_reshaped, past_key_value

    OPTTuckerTTAttention.forward = patched_forward
    OPTTuckerTTAttention._debug_patched = True


def apply_opt_baseline_attention_patch():
    """
    Patch HuggingFace OPTAttention.forward to store baseline layer-level debug_cache.

    This is only used for debugging and tries to mirror HF OPTAttention behavior.
    """
    from transformers.models.opt.modeling_opt import OPTAttention

    if getattr(OPTAttention, "_debug_patched", False):
        return

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if not hasattr(self, "debug_cache"):
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
        pre_softmax_scores = torch.bmm(query_states, key_states.transpose(1, 2))

        last_scores = pre_softmax_scores[:, -1, :].detach().float()
        self.debug_cache.update({
            "Q_norm": _safe_float(query_states[:, -1, :].detach().float().norm()),
            "K_norm": _safe_float(key_states.detach().float().norm()),
            "V_norm": _safe_float(value_states.detach().float().norm()),
            "Q_mean": _safe_float(query_states[:, -1, :].detach().float().mean()),
            "K_mean": _safe_float(key_states.detach().float().mean()),
            "V_mean": _safe_float(value_states.detach().float().mean()),
            "pre_softmax_attn_weights_mean": _safe_float(last_scores.mean()),
            "pre_softmax_attn_weights_max": _safe_float(last_scores.max()),
            "pre_softmax_attn_weights_min": _safe_float(last_scores.min()),
            "pre_softmax_attn_weights_std": _safe_float(last_scores.std()),
        })

        attn_weights = pre_softmax_scores
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, "
                f"but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, "
                    f"but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_weights.dtype == torch.float16:
            attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_probs = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, "
                    f"but is {layer_head_mask.size()}"
                )
            attn_probs = layer_head_mask.view(1, -1, 1, 1) * attn_probs.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_probs = attn_probs.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            attn_weights_reshaped = attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
            attn_probs_for_output = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
            attn_probs_for_output = attn_probs

        recent_probs = attn_probs_for_output.view(bsz, self.num_heads, tgt_len, src_len)[:, :, -1, :].detach().float()
        self.debug_cache["attn_entropy"] = entropy_from_probs(recent_probs)
        self.debug_cache["attn_prob_max"] = _safe_float(recent_probs.max())
        self.debug_cache["attn_prob_mean"] = _safe_float(recent_probs.mean())

        attn_output = torch.bmm(attn_probs_for_output, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        last_attn_output = attn_output[:, -1, :].detach().float()
        self.debug_cache["attn_output_norm"] = _safe_float(last_attn_output.norm())
        self.debug_cache["attn_output_mean"] = _safe_float(last_attn_output.mean())
        self.debug_cache["attn_output_std"] = _safe_float(last_attn_output.std())
        self.debug_cache["attn_output_min"] = _safe_float(last_attn_output.min())
        self.debug_cache["attn_output_max"] = _safe_float(last_attn_output.max())

        return attn_output, attn_weights_reshaped, past_key_value

    OPTAttention.forward = patched_forward
    OPTAttention._debug_patched = True


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_text_standard(
    model,
    tokenizer,
    prompt: str,
    device,
    max_new_tokens: int = 40,
    debug: bool = False,
) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=True,
        )

    output_ids = outputs.sequences
    generated = output_ids[0, inputs["input_ids"].shape[1]:]

    if debug:
        print("\n  [DEBUG] Next-token prediction probabilities:")
        for step, score in enumerate(outputs.scores):
            probs = torch.softmax(score[0].float(), dim=-1)
            top_probs, top_indices = torch.topk(probs, 5)
            print(f"    Step {step + 1}:")
            for i in range(5):
                token_str = tokenizer.decode([top_indices[i].item()]).replace("\n", "\\n")
                print(
                    f"      Top {i + 1}: '{token_str}' "
                    f"(id: {top_indices[i].item():>5}) - prob: {top_probs[i].item():.4f}"
                )
            chosen_token = generated[step]
            chosen_str = tokenizer.decode([chosen_token.item()]).replace("\n", "\\n")
            print(f"      -> Chosen: '{chosen_str}' (id: {chosen_token.item()})\n")

    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_text_debug_deep(
    model,
    baseline_model,
    tokenizer,
    prompt: str,
    device,
    max_new_tokens: int,
    save_path: str = "",
    compare_baseline: bool = False,
    debug_baseline_layer_stats: bool = False,
):
    model.eval()
    if compare_baseline and baseline_model is not None:
        baseline_model.eval()

    if debug_baseline_layer_stats and (not compare_baseline or baseline_model is None):
        print("[WARNING] --debug_baseline_layer_stats requires --debug_compare_baseline; disabling baseline layer stats.")
        debug_baseline_layer_stats = False

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    trace_data = []
    prev_logits = None
    prev_final_hidden = None
    prev_layer_hiddens = None

    watched_tokens = {
        "comma": 6,
        "period": 4,
        "and": 8,
        "dash": 12,
        "newline": 50118,
        "eos": tokenizer.eos_token_id,
    }

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

            logits = outputs.logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)

            top_probs, top_indices = torch.topk(probs, 5)
            top_logits = logits.gather(1, top_indices)

            final_hidden = outputs.hidden_states[-1][:, -1, :].detach().float()
            hidden_stats = tensor_stats(final_hidden)
            layer_hiddens = [h[:, -1, :].detach().float() for h in outputs.hidden_states]

            logits_stats = tensor_stats(logits)
            logits_stats["entropy"] = entropy_from_probs(probs)
            logits_stats["margin_top1_top2"] = _safe_float(top_logits[0, 0] - top_logits[0, 1])

            watched_stats = {}
            for name, tid in watched_tokens.items():
                if tid is not None:
                    watched_stats[name] = {
                        "logit": _safe_float(logits[0, tid]),
                        "prob": _safe_float(probs[0, tid]),
                    }

            logits_vs_prev = {}
            hidden_vs_prev = {}
            layer_hiddens_vs_prev = []

            if prev_logits is not None:
                logits_vs_prev["cosine_sim"] = cosine_sim(logits, prev_logits)
                logits_vs_prev["kl_div"] = kl_div_from_log_probs(log_probs, torch.log_softmax(prev_logits.float(), dim=-1))

                hidden_vs_prev["cosine_sim"] = cosine_sim(final_hidden, prev_final_hidden)
                hidden_vs_prev["max_abs_diff"] = _safe_float(torch.max(torch.abs(final_hidden - prev_final_hidden)))

                for i, (lh, ph) in enumerate(zip(layer_hiddens, prev_layer_hiddens)):
                    h_sim = cosine_sim(lh, ph)
                    layer_hiddens_vs_prev.append({
                        "layer": i,
                        "norm": _safe_float(lh.norm()),
                        "cosine_sim_prev": h_sim,
                    })
                    if h_sim > 0.999 or h_sim < 0.1:
                        print(f"    [WARNING] Layer {i} hidden sim abnormal: {h_sim:.4f}")

                if logits_vs_prev["cosine_sim"] > 0.999:
                    print("    [WARNING] logits distribution collapsed / nearly identical across steps")
                if hidden_vs_prev["cosine_sim"] > 0.999:
                    print("    [WARNING] final hidden state is almost unchanged across steps")

            # Collect compressed attention debug cache immediately after compressed forward.
            compressed_layer_stats = collect_debug_cache(model)

            # Baseline teacher-forced compare on the same input_ids/context.
            baseline_compare = {}
            baseline_layer_stats = {}
            layer_compare = {}

            if compare_baseline and baseline_model is not None:
                base_outputs = baseline_model(
                    input_ids=input_ids,
                    use_cache=False,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                base_logits = base_outputs.logits[:, -1, :].float()
                base_probs = torch.softmax(base_logits, dim=-1)
                base_log_probs = torch.log_softmax(base_logits, dim=-1)
                base_final_hidden = base_outputs.hidden_states[-1][:, -1, :].detach().float()
                base_top_p, base_top_idx = torch.topk(base_probs, 5)

                baseline_compare = {
                    "logits_cosine_sim": cosine_sim(logits, base_logits),
                    "kl_div": kl_div_from_log_probs(log_probs, base_log_probs),
                    "final_hidden_cosine_sim": cosine_sim(final_hidden, base_final_hidden),
                    "comma_logit_diff": _safe_float(logits[0, 6] - base_logits[0, 6]),
                    "comma_prob_compressed": _safe_float(probs[0, 6]),
                    "comma_prob_baseline": _safe_float(base_probs[0, 6]),
                    "top_probs_baseline": [_safe_float(p) for p in base_top_p[0]],
                    "top_indices_baseline": [int(idx) for idx in base_top_idx[0]],
                    "top_tokens_baseline": [tokenizer.decode([int(idx)]).replace("\n", "\\n") for idx in base_top_idx[0]],
                }

                if debug_baseline_layer_stats:
                    baseline_layer_stats = collect_debug_cache(baseline_model)
                    layer_compare = compare_layer_stats(compressed_layer_stats, baseline_layer_stats)

            prev_logits = logits.clone()
            prev_final_hidden = final_hidden.clone()
            prev_layer_hiddens = [h.clone() for h in layer_hiddens]

            # Build trace before appending next_token, so context_tokens is exactly the context used.
            step_trace = {
                "step": step,
                "context_token_ids": [int(t) for t in input_ids[0]],
                "context_tokens": [tokenizer.decode([int(t)]).replace("\n", "\\n") for t in input_ids[0]],
                "chosen_token_id": int(next_token.item()),
                "chosen_token": tokenizer.decode([int(next_token.item())]).replace("\n", "\\n"),
                "top_k_indices": [int(idx) for idx in top_indices[0]],
                "top_k_tokens": [tokenizer.decode([int(idx)]).replace("\n", "\\n") for idx in top_indices[0]],
                "top_k_probs": [_safe_float(p) for p in top_probs[0]],
                "top_k_logits": [_safe_float(l) for l in top_logits[0]],
                "logits_stats": logits_stats,
                "watched_token_stats": watched_stats,
                "hidden_stats": hidden_stats,
                "logits_vs_prev": logits_vs_prev,
                "hidden_vs_prev": hidden_vs_prev,
                "layer_hidden_stats": layer_hiddens_vs_prev,
                "layer_stats": compressed_layer_stats,
                "baseline_compare": baseline_compare,
                "baseline_layer_stats": baseline_layer_stats,
                "layer_compare": layer_compare,
            }

            trace_data.append(step_trace)

            base_info = f" | KL_base: {baseline_compare.get('kl_div', 0):.4f}" if baseline_compare else ""
            layer_summary = summarize_layer_compare(layer_compare)
            layer_info = f" | {layer_summary}" if layer_summary else ""
            print(f"Step {step}: Model -> {step_trace['chosen_token']!r}{base_info}{layer_info}")

            input_ids = torch.cat([input_ids, next_token], dim=-1)

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for td in trace_data:
                f.write(json.dumps(td, ensure_ascii=False) + "\n")
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
    parser.add_argument("--debug_compare_baseline", action="store_true", help="Step-by-step teacher-forced compare against baseline model")
    parser.add_argument("--debug_baseline_layer_stats", action="store_true", help="Save baseline Q/K/V/attention stats and layer_compare in JSONL")

    args = parser.parse_args()

    if args.debug_baseline_layer_stats and not args.debug_compare_baseline:
        print("[WARNING] --debug_baseline_layer_stats requires --debug_compare_baseline; disabling baseline layer stats.")
        args.debug_baseline_layer_stats = False

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print("  Tucker+TT OPT Inference Test")
    print(f"{'=' * 60}")
    print(f"  Model    : {args.model}")
    print(f"  Device   : {device}")
    print(f"  nsamples : {args.nsamples}  |  avg_dim : {args.avg_dim}")
    print(f"  Prompt   : {args.prompt!r}")
    print(f"{'=' * 60}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    config = AutoConfig.from_pretrained(args.model, attn_implementation="eager", cache_dir=args.cache_dir)
    config.use_cache = False

    print("[1/4] Loading tokenizer and base OPT model ...")
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

    baseline_text = None
    if not args.no_baseline and not args.debug_deep:
        print(f"\n[2/4] Baseline generation on {device} ...")
        base_model_dev = base_model.to(device)
        t0 = time.time()
        baseline_text = generate_text_standard(base_model_dev, tokenizer, args.prompt, device, args.max_new_tokens)
        print(f"  Baseline output  : {baseline_text!r}")
        print(f"  Time             : {time.time() - t0:.2f}s")
        base_model_dev.cpu()
        torch.cuda.empty_cache()
    elif not args.no_baseline:
        print("\n[2/4] Baseline generation delayed for deep debug teacher-forced compare.")
    else:
        print("\n[2/4] Baseline generation SKIPPED (--no_baseline).")

    class _LMWrapper:
        def __init__(self, model, dev):
            self.model = model
            self._device = dev
            self.seqlen = model.config.max_position_embeddings

        @property
        def device(self):
            return self._device

    lm = _LMWrapper(base_model.to(device), device)

    import types

    ea_args = types.SimpleNamespace(
        net="opt-125m",
        nsamples=args.nsamples,
        error_budget=args.error_budget,
        low_rank=False,
        eigen_attn_params={
            "threshold": 0.98,
            "avg_dim_features": args.avg_dim,
            "error_budget": args.error_budget,
        },
    )

    dataloader = load_calib_data(args.model, args.nsamples, args.seed, lm.seqlen, args.cache_dir)

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
        if args.debug_baseline_layer_stats:
            apply_opt_baseline_attention_patch()

        infer_device = model_device(compressed_model)
        compressed_model.config.use_cache = False
        compressed_model = compressed_model.to(infer_device)

        baseline_model_compare = None
        if args.debug_compare_baseline:
            baseline_model_compare = AutoModelForCausalLM.from_pretrained(
                args.model,
                config=config,
                device_map="cpu",
                torch_dtype=torch.float16,
                cache_dir=args.cache_dir,
            ).to(infer_device)
            baseline_model_compare.eval()
            for p in baseline_model_compare.parameters():
                p.requires_grad_(False)
            baseline_model_compare.config.use_cache = False

        print("\n[4/4] Deep Debug Generation (manual greedy loop) ...")
        t0 = time.time()
        compressed_text = generate_text_debug_deep(
            compressed_model,
            baseline_model_compare,
            tokenizer,
            args.prompt,
            infer_device,
            args.max_new_tokens,
            save_path=args.debug_save_path,
            compare_baseline=args.debug_compare_baseline,
            debug_baseline_layer_stats=args.debug_baseline_layer_stats,
        )
        print(f"  Time             : {time.time() - t0:.2f}s")

    else:
        infer_device = model_device(compressed_model)
        print(f"\n[4/4] Compressed model generation (model on {infer_device}) ...")
        compressed_model.config.use_cache = True
        compressed_model = compressed_model.to(infer_device)
        t0 = time.time()
        compressed_text = generate_text_standard(
            compressed_model,
            tokenizer,
            args.prompt,
            infer_device,
            args.max_new_tokens,
            debug=True,
        )
        print(f"  Time             : {time.time() - t0:.2f}s")

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Prompt           : {args.prompt!r}")
    if baseline_text is not None:
        print(f"  Baseline output  : {baseline_text!r}")
    print(f"  Compressed output: {compressed_text!r}")
    print(f"  Param ratio      : {compressed_params / baseline_params:.4f}  "
          f"({baseline_params / 1e6:.1f}M → {compressed_params / 1e6:.1f}M)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
