import torch
from .models_utils import BaseLM, find_layers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import pdb
from models.modelling_opt_eigen_attn import OPTForCausalLM_EigenAttn
from models.modelling_opt_tucker_attn import OPTForCausalLM_TuckerAttn
from models.modelling_mpt_eigen_attn import MptForCausalLM_EigenAttn
from models.modelling_llama_eigen_attn import LlamaForCausalLM_EigenAttn
from models.modelling_llama_tucker_attn import LlamaForCausalLM_TuckerAttn
import os
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)
from peft import PeftModel


def _dtype_from_name(name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype name: {name}")


def _is_quantized_args(args):
    return bool(getattr(args, "load_in_4bit", False) or getattr(args, "load_in_8bit", False))


def _model_load_kwargs(args):
    if getattr(args, "load_in_4bit", False) and getattr(args, "load_in_8bit", False):
        raise ValueError("Choose only one of --load_in_4bit or --load_in_8bit")

    if getattr(args, "load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=getattr(args, "bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=_dtype_from_name(getattr(args, "bnb_4bit_compute_dtype", "float16")),
            bnb_4bit_use_double_quant=bool(getattr(args, "bnb_4bit_use_double_quant", False)),
        )
        return {
            "device_map": getattr(args, "quant_device_map", "auto"),
            "torch_dtype": _dtype_from_name(getattr(args, "bnb_4bit_compute_dtype", "float16")),
            "quantization_config": quantization_config,
            "cache_dir": args.cache_dir,
        }

    if getattr(args, "load_in_8bit", False):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        return {
            "device_map": getattr(args, "quant_device_map", "auto"),
            "torch_dtype": torch.float16,
            "quantization_config": quantization_config,
            "cache_dir": args.cache_dir,
        }

    return {
        "device_map": "cpu",
        "torch_dtype": torch.float16,
        "cache_dir": args.cache_dir,
    }


@contextmanager
def _force_accelerate_hooks_for_quantized_load(enabled):
    if not enabled:
        yield
        return

    try:
        import accelerate.big_modeling as accelerate_big_modeling
        import transformers.modeling_utils as transformers_modeling_utils
    except Exception:
        yield
        return

    original_accelerate_dispatch = accelerate_big_modeling.dispatch_model
    original_transformers_dispatch = getattr(transformers_modeling_utils, "dispatch_model", None)

    def dispatch_model_with_forced_hooks(model, *args, **kwargs):
        kwargs["force_hooks"] = True
        return original_accelerate_dispatch(model, *args, **kwargs)

    accelerate_big_modeling.dispatch_model = dispatch_model_with_forced_hooks
    if original_transformers_dispatch is not None:
        transformers_modeling_utils.dispatch_model = dispatch_model_with_forced_hooks

    try:
        yield
    finally:
        accelerate_big_modeling.dispatch_model = original_accelerate_dispatch
        if original_transformers_dispatch is not None:
            transformers_modeling_utils.dispatch_model = original_transformers_dispatch


class LMClass(BaseLM):
    def __init__(self, args):

        super().__init__()
        self._rank = 0
        self._world_size = 1
        self.args = args
        self.is_quantized = _is_quantized_args(args)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size

        self.model_config = args.model

        config = AutoConfig.from_pretrained(
            args.model, attn_implementation=args.attn_implementation, cache_dir = args.cache_dir
        )
        config.use_cache = False
        use_fast = False
        if 'mpt' in args.net.lower():
            use_fast = True
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=use_fast,legacy=False, cache_dir = args.cache_dir)
        model_load_kwargs = _model_load_kwargs(args)
        with _force_accelerate_hooks_for_quantized_load(self.is_quantized):
            if args.load_low_rank :
                low_rank_config = torch.load(os.path.join(args.save_dir,'low_rank_config.pt'))
                if 'mpt' in args.net.lower():
                    self.model = MptForCausalLM_EigenAttn.from_pretrained(args.save_dir, config=config, low_rank_config=low_rank_config, **model_load_kwargs)
                    if args.load_peft_model:
                        self.model = PeftModel.from_pretrained(self.model, args.peft_model_path)
                        
                elif 'opt' in args.net.lower():
                    low_rank_type_path = os.path.join(args.save_dir, 'low_rank_type.pt')
                    if os.path.exists(low_rank_type_path):
                        low_rank_type = torch.load(low_rank_type_path)
                    else:
                        low_rank_type = "tucker" if getattr(args, "use_tucker_attn", False) else "eigen"

                    if low_rank_type == "tucker":
                        self.model = OPTForCausalLM_TuckerAttn.from_pretrained(args.save_dir, config=config, low_rank_config = low_rank_config, **model_load_kwargs)
                    else:
                        self.model = OPTForCausalLM_EigenAttn.from_pretrained(args.save_dir, config=config, low_rank_config = low_rank_config, **model_load_kwargs)
                    if args.load_peft_model:
                        self.model = PeftModel.from_pretrained(self.model, args.peft_model_path)

                elif 'llama' in args.net.lower():
                    low_rank_type_path = os.path.join(args.save_dir, 'low_rank_type.pt')
                    if os.path.exists(low_rank_type_path):
                        low_rank_type = torch.load(low_rank_type_path)
                    else:
                        low_rank_type = "tucker" if getattr(args, "use_tucker_attn", False) else "eigen"

                    if low_rank_type == "tucker":
                        self.model = LlamaForCausalLM_TuckerAttn.from_pretrained(args.save_dir, config=config, low_rank_config=low_rank_config, **model_load_kwargs)
                    else:
                        self.model = LlamaForCausalLM_EigenAttn.from_pretrained(args.save_dir, config=config, low_rank_config=low_rank_config, **model_load_kwargs)
                    if args.load_peft_model:
                        self.model = PeftModel.from_pretrained(self.model, args.peft_model_path)
                        self.model = self.model.model
                else:
                    raise NotImplementedError
            else:
                self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, **model_load_kwargs)
        if 'mpt' in args.net.lower():
            self.seqlen = self.model.config.max_seq_len
        else:
            self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        print("vocab size: ", self.vocab_size)

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            
            return self.model.config.max_position_embeddings
            # return self.model.config.max_seq_len

    @property
    def max_gen_toks(self):
        print("max_gen_toks fn")
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    @property
    def rank(self):
        # used in the case of parallelism. Hardcoded to
        # ensure no errors arise using API models which do
        # not support multi-device parallelism nor expect it.
        return self._rank

    @property
    def world_size(self):
        # used in the case of parallelism. Hardcoded to
        # ensure no errors arise using API models which do
        # not support multi-device parallelism nor expect it.
        return self._world_size


    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    # def tok_encode_batch(self, strings):
    #     return self.tokenizer(
    #         strings,
    #         padding=True,
    #         add_special_tokens=False,
    #         return_tensors="pt",
    #     )

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        # add_special_tokens = {}
        # add_special_tokens = {self.tokenizer.bos_token}
        self.tokenizer.pad_token = self.tokenizer.bos_token
        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            # padding_side="left",
        )
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]
    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():

            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )

