from typing import Optional, Tuple

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.models.opt.configuration_opt import OPTConfig

from models.modelling_opt_eigen_attn import (
    OPTDecoder_EigenAttn,
    OPTForCausalLM_EigenAttn,
    OPTModel_EigenAttn,
)


class OPTTuckerAttentionForCausalLM(nn.Module):
    """OPT attention module that runs materialized Tucker-projected weights."""

    def __init__(
        self,
        config: OPTConfig,
        rank_kq: int,
        rank_v: int,
        is_decoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.enable_bias = config.enable_bias

        self.head_dim = self.embed_dim // self.num_heads
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.rank_kq = int(rank_kq)
        self.rank_v = int(rank_v)
        if self.rank_kq % self.num_heads != 0:
            raise ValueError(f"Tucker Q/K rank {self.rank_kq} must be divisible by num_heads={self.num_heads}")
        if self.rank_v % self.num_heads != 0:
            raise ValueError(f"Tucker V rank {self.rank_v} must be divisible by num_heads={self.num_heads}")

        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj_low = nn.Linear(self.embed_dim, self.rank_kq, bias=self.enable_bias)
        self.q_proj_low = nn.Linear(self.embed_dim, self.rank_kq, bias=self.enable_bias)
        self.v_proj_low = nn.Linear(self.embed_dim, self.rank_v, bias=self.enable_bias)
        self.out_proj_up = nn.Linear(self.rank_v, self.embed_dim, bias=self.enable_bias)

    def _new_shape(self, tensor: torch.Tensor, seq_len: int, bsz: int, low_dim: int):
        return tensor.view(bsz, seq_len, self.num_heads, low_dim // self.num_heads).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj_low(hidden_states) * self.scaling

        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self.k_proj_low(key_value_states)
            key_states = self._new_shape(key_states, -1, bsz, self.rank_kq)
            value_states = self._new_shape(self.v_proj_low(key_value_states), -1, bsz, self.rank_v)
        elif past_key_value is not None:
            key_states = self.k_proj_low(hidden_states)
            key_states = self._new_shape(key_states, -1, bsz, self.rank_kq)
            value_states = self.v_proj_low(hidden_states)
            value_states = self._new_shape(value_states, -1, bsz, self.rank_v)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self.k_proj_low(hidden_states)
            key_states = self._new_shape(key_states, -1, bsz, self.rank_kq)
            value_states = self.v_proj_low(hidden_states)
            value_states = self._new_shape(value_states, -1, bsz, self.rank_v)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape_kq = (bsz * self.num_heads, -1, self.rank_kq // self.num_heads)
        proj_shape_v = (bsz * self.num_heads, -1, self.rank_v // self.num_heads)

        query_states = self._new_shape(query_states, tgt_len, bsz, self.rank_kq).view(*proj_shape_kq)
        key_states = key_states.view(*proj_shape_kq)
        value_states = value_states.view(*proj_shape_v)

        src_len = key_states.size(1)
        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2))

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.rank_v // self.num_heads):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.rank_v // self.num_heads)},"
                f" but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.rank_v // self.num_heads)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.rank_v)
        attn_output = self.out_proj_up(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class OPTDecoderLayer_TuckerAttn(nn.Module):
    def __init__(self, low_rank_config, config: OPTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        assert config._attn_implementation == "eager"

        self.self_attn = OPTTuckerAttentionForCausalLM(
            config=config,
            is_decoder=True,
            rank_kq=int(low_rank_config[0]),
            rank_v=int(low_rank_config[1]),
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        residual = hidden_states

        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class OPTDecoder_TuckerAttn(OPTDecoder_EigenAttn):
    def __init__(self, low_rank_config, config: OPTConfig):
        super().__init__(low_rank_config=low_rank_config, config=config)
        self.layers = nn.ModuleList(
            [OPTDecoderLayer_TuckerAttn(config=config, low_rank_config=low_rank_config[i]) for i in range(config.num_hidden_layers)]
        )
        self.post_init()


class OPTModel_TuckerAttn(OPTModel_EigenAttn):
    def __init__(self, config: OPTConfig, low_rank_config):
        super().__init__(config=config, low_rank_config=low_rank_config)
        self.decoder = OPTDecoder_TuckerAttn(config=config, low_rank_config=low_rank_config)
        self.post_init()


class OPTForCausalLM_TuckerAttn(OPTForCausalLM_EigenAttn):
    def __init__(self, config, low_rank_config):
        super().__init__(config=config, low_rank_config=low_rank_config)
        self.model = OPTModel_TuckerAttn(config=config, low_rank_config=low_rank_config)
        self.post_init()
