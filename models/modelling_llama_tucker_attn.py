import math
from typing import Optional, Tuple

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from models.modelling_llama_eigen_attn import (
    LlamaEigenAttnDecoderLayer,
    LlamaForCausalLM_EigenAttn,
    LlamaModel_EigenAttn,
)


class LlamaTuckerAttention(nn.Module):
    """LLaMA attention module with Tucker-compressed K/V projections."""

    def __init__(self, rank_kq: int, rank_v: int, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.rank_kq = int(rank_kq)
        self.rank_v = int(rank_v)
        if self.rank_kq % self.num_key_value_heads != 0:
            raise ValueError(f"Tucker K rank {self.rank_kq} must be divisible by num_key_value_heads={self.num_key_value_heads}")
        if self.rank_v % self.num_key_value_heads != 0:
            raise ValueError(f"Tucker V rank {self.rank_v} must be divisible by num_key_value_heads={self.num_key_value_heads}")

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.k_proj_low = nn.Linear(self.hidden_size, self.rank_kq, bias=config.attention_bias)
        self.v_proj_low = nn.Linear(self.hidden_size, self.rank_v, bias=config.attention_bias)

        k_per_head = self.rank_kq // self.num_key_value_heads
        v_per_head = self.rank_v // self.num_key_value_heads
        self.k_proj_up_weight = nn.Parameter(torch.empty(self.num_key_value_heads, self.head_dim, k_per_head))
        self.o_proj_up = nn.Linear(self.num_heads * v_per_head, self.hidden_size, bias=config.attention_bias)

        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _apply_k_proj_up(self, key_states_low: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bhsr,hdr->bhsd", key_states_low, self.k_proj_up_weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states_low = self.k_proj_low(hidden_states)
        value_states_low = self.v_proj_low(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states_low = key_states_low.view(
            bsz, q_len, self.num_key_value_heads, self.rank_kq // self.num_key_value_heads
        ).transpose(1, 2)
        value_states_low = value_states_low.view(
            bsz, q_len, self.num_key_value_heads, self.rank_v // self.num_key_value_heads
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states_low, position_ids)
        past_key_value = getattr(self, "past_key_value", past_key_value)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states_low, value_states_low = past_key_value.update(
                key_states_low, value_states_low, self.layer_idx, cache_kwargs
            )

        key_states = self._apply_k_proj_up(key_states_low)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states_low = repeat_kv(value_states_low, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states_low)

        v_per_head = self.rank_v // self.num_key_value_heads
        if attn_output.size() != (bsz, self.num_heads, q_len, v_per_head):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, v_per_head)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * v_per_head)
        attn_output = self.o_proj_up(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaTuckerAttnDecoderLayer(LlamaEigenAttnDecoderLayer):
    def __init__(self, low_rank_config, config: LlamaConfig, layer_idx: int):
        super(LlamaEigenAttnDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        if config._attn_implementation != "eager":
            import warnings
            warnings.warn("attention implementation was set to " + config._attn_implementation + ", changing it to eager!!")
            config._attn_implementation = "eager"
        assert config._attn_implementation == "eager"
        self.self_attn = LlamaTuckerAttention(
            rank_kq=int(low_rank_config[0]),
            rank_v=int(low_rank_config[1]),
            config=config,
            layer_idx=layer_idx,
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LlamaModel_TuckerAttn(LlamaModel_EigenAttn):
    def __init__(self, low_rank_config, config: LlamaConfig):
        super(LlamaModel_EigenAttn, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                LlamaTuckerAttnDecoderLayer(
                    low_rank_config=low_rank_config[layer_idx],
                    config=config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()


class LlamaForCausalLM_TuckerAttn(LlamaForCausalLM_EigenAttn):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        low_rank_config = kwargs.pop("low_rank_config")
        super(LlamaForCausalLM_EigenAttn, self).__init__(config=config)
        self.model = LlamaModel_TuckerAttn(low_rank_config=low_rank_config, config=config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.idx = 0
        self.post_init()
