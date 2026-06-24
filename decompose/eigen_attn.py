import torch
import torch.nn as nn
import copy
import gc

from decompose.eigen_attn_utils import (decompose_opt_layer, decompose_mpt_layer, decompose_llama_layer,
                                         tensor_train_decompose_opt_layer,
                                         tensor_train_decompose_opt_layer_out_proj,
                                         apply_tucker_factors_to_tt_cores, reconstruct_combined_tt_cores,
                                         project_bias_with_tucker_factors,
                                        tensor_train_decompose_opt_layer_bias)
from decompose.tucker_utils import (
    tucker_decompose_opt_layer,
    tucker_analyze_opt_kq,
    get_tucker_attn_config,
    get_opt_tucker_activations,
    build_opt_tucker_projection_config,
    tucker_attention_param_ratio,
)
from models.decompose_modules import (OPTEigenAttnDecoderLayer, MptBlockEigenAttn, LlamaEigenAttnDecoderLayer,
                                      OPTTuckerTTDecoderLayer, OPTTuckerAttnDecoderLayer)


def _use_opt_tucker_attention(args) -> bool:
    eigen_params = getattr(args, "eigen_attn_params", {}) or {}
    tucker_params = {}
    if isinstance(eigen_params.get("tucker"), dict):
        tucker_params.update(eigen_params["tucker"])
    if hasattr(args, "tucker_attn_params") and isinstance(args.tucker_attn_params, dict):
        tucker_params.update(args.tucker_attn_params)
    return bool(tucker_params.get("enabled", tucker_params.get("use_tucker", False)))


def _log_tucker_candidate(logger, layer_id, tucker_config, output_error, compression_ratio, memory_mb):
    qk = tucker_config["qk"]
    v = tucker_config["v"]
    logger.info(
        f"layer {layer_id} Tucker candidate "
        f"threshold:{tucker_config['threshold']} "
        f"qk_ranks:{qk['ranks']} v_ranks:{v['ranks']} "
        f"qk_latent:{qk['latent_dim']} v_latent:{v['latent_dim']} "
        f"qk_energy:{qk['retained_energies']} v_energy:{v['retained_energies']} "
        f"output_error:{output_error} compression_ratio:{compression_ratio:.4f} "
        f"max memory_allocated {memory_mb}"
    )
    if qk["reconstruction_error"] is not None or v["reconstruction_error"] is not None:
        logger.info(
            f"layer {layer_id} Tucker debug reconstruction_error "
            f"qk:{qk['reconstruction_error']} v:{v['reconstruction_error']} "
            f"(not used for rank selection)"
        )


def eigenattn(
    lm,
    args,
    dataloader,
    logger=None,
):
    logger.info("Starting ...")
    #no quantization Only low rank
    assert(('opt' in args.net.lower()) or ('mpt' in args.net.lower()) or ('llama' in args.net.lower()) or ('mistral' in args.net.lower()) or ('qwen' in args.net.lower())) # only support OPT, MPT, Llama2 model for now

    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    is_mpt = False
    is_opt = False
    is_mistral = False
    if "llama" in args.net.lower() :
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = LlamaEigenAttnDecoderLayer
        
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        is_opt = True
        DecoderLayer = OPTEigenAttnDecoderLayer
        
    elif "mpt" in args.net.lower():
        is_mpt = True
        layers = model.transformer.blocks
        model.transformer.wte = model.transformer.wte.to(dev)
        
        DecoderLayer = MptBlockEigenAttn
    
    else:
        raise ValueError("Only support for opt/mpt/llama-2/llama-3.0 now")
    
    
    layers[0] = layers[0].to(dev)
    dtype = torch.float16
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False
            self.is_mpt = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            if self.is_mpt :
                cache["position_bias"] = kwargs["position_bias"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    layers[0].is_mpt = is_mpt

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if is_llama :
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif is_opt:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif is_mpt:
        model.transformer.wte = model.transformer.wte.cpu()
    
    else:
        raise ValueError("Only support for opt/mpt/llama-2/llama-3.0 now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    
    attention_mask = cache["attention_mask"]


    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    if is_mpt:
        position_bias = cache["position_bias"]
    else:
        position_bias = None


    for i in range(len(layers)):
        layer = layers[i].to(dev)

        if is_opt:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    num_heads = lm.model.config.num_attention_heads
                    output_hr = torch.stack([layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0] for j in range(args.nsamples)])

                    if _use_opt_tucker_attention(args):
                        tucker_cfg = get_tucker_attn_config(args, lm.model.config.hidden_size)
                        if (
                            tucker_cfg["preserve_heads"]
                            and not tucker_cfg["modes_were_explicit"]
                            and len(tucker_cfg["factor_dims"]) >= 2
                            and tucker_cfg["factor_dims"][0] == int(num_heads)
                        ):
                            tucker_cfg["modes"] = list(range(2, len(tucker_cfg["factor_dims"]) + 1))
                        logger.info(
                            f"layer {i} starting Tucker-only attention search "
                            f"factor_dims:{tucker_cfg['factor_dims']} modes:{tucker_cfg['modes']} "
                            f"initial_threshold:{tucker_cfg['initial_threshold']} "
                            f"step:{tucker_cfg['threshold_step']} min_threshold:{tucker_cfg['min_threshold']}"
                        )

                        tensor_k, tensor_q, tensor_v = get_opt_tucker_activations(layer, inps, args)
                        threshold = tucker_cfg["initial_threshold"]
                        best_config = None
                        best_error = None
                        best_threshold = None

                        while threshold >= tucker_cfg["min_threshold"]:
                            candidate_config = build_opt_tucker_projection_config(
                                tensor_k,
                                tensor_q,
                                tensor_v,
                                args,
                                num_heads,
                                threshold,
                            )
                            qlayer_candidate = OPTTuckerAttnDecoderLayer(
                                layer,
                                args,
                                candidate_config,
                                lm.model.config,
                            ).to(dev)
                            output_lr = torch.stack([
                                qlayer_candidate(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                                for j in range(args.nsamples)
                            ])
                            error = torch.norm(output_hr - output_lr) / torch.norm(output_hr)
                            compression_ratio = tucker_attention_param_ratio(
                                lm.model.config.hidden_size,
                                candidate_config["qk"]["latent_dim"],
                                candidate_config["v"]["latent_dim"],
                            )
                            _log_tucker_candidate(
                                logger,
                                i,
                                candidate_config,
                                error,
                                compression_ratio,
                                torch.cuda.max_memory_allocated(lm._device) / 1024**2,
                            )

                            candidate_valid = error <= args.error_budget
                            if candidate_valid:
                                best_config = candidate_config
                                best_error = error
                                best_threshold = threshold
                                threshold -= tucker_cfg["threshold_step"]

                            del qlayer_candidate, output_lr
                            torch.cuda.empty_cache()
                            if not candidate_valid:
                                break

                        if best_config is None:
                            logger.info(
                                f"layer {i} no Tucker candidate met output error budget; "
                                f"falling back to full Tucker ranks at threshold 1.0"
                            )
                            best_config = build_opt_tucker_projection_config(
                                tensor_k,
                                tensor_q,
                                tensor_v,
                                args,
                                num_heads,
                                1.0,
                            )
                            qlayer = OPTTuckerAttnDecoderLayer(layer, args, best_config, lm.model.config).to(dev)
                            output_lr = torch.stack([
                                qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                                for j in range(args.nsamples)
                            ])
                            error = torch.norm(output_hr - output_lr) / torch.norm(output_hr)
                            best_error = error
                            best_threshold = 1.0
                        else:
                            qlayer = OPTTuckerAttnDecoderLayer(layer, args, best_config, lm.model.config).to(dev)
                            error = best_error

                        compression_ratio = tucker_attention_param_ratio(
                            lm.model.config.hidden_size,
                            best_config["qk"]["latent_dim"],
                            best_config["v"]["latent_dim"],
                        )
                        logger.info(
                            f"layer {i} selected Tucker output_error:{error} "
                            f"threshold:{best_threshold} "
                            f"qk_ranks:{best_config['qk']['ranks']} "
                            f"v_ranks:{best_config['v']['ranks']} "
                            f"qk_latent:{best_config['qk']['latent_dim']} "
                            f"v_latent:{best_config['v']['latent_dim']} "
                            f"compression_ratio:{compression_ratio:.4f}"
                        )

                        del tensor_k, tensor_q, tensor_v
                        torch.cuda.empty_cache()

                    else:
                        args.eigen_attn_params['threshold'] = 0.98
                        error = 0.0
                        basis_kq, eval_kq, basis_v, eval_v = decompose_opt_layer(layer, inps, args, num_heads, i)

                        rank_kq = num_heads * torch.amax((torch.cumsum(eval_kq, dim = 1) < args.eigen_attn_params['threshold']).sum(1))
                        rank_v = num_heads * torch.amax((torch.cumsum(eval_v, dim = 1) < args.eigen_attn_params['threshold']).sum(1))

                        while error < args.error_budget and args.eigen_attn_params['threshold']> 0.3 and rank_kq > 64 and rank_v > 64:
                            qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config).to(dev)
                            output_lr = torch.stack([qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0] for j in range(args.nsamples)])
                            error = (torch.norm(output_hr - output_lr)/torch.norm(output_hr))
                            args.eigen_attn_params['threshold'] -= 0.02
                            rank_kq = num_heads * torch.amax((torch.cumsum(eval_kq, dim = 1) < args.eigen_attn_params['threshold']).sum(1))
                            rank_v = num_heads * torch.amax((torch.cumsum(eval_v, dim = 1) < args.eigen_attn_params['threshold']).sum(1))

                            logger.info(f"layer {i} error:{error} threshold:{args.eigen_attn_params['threshold']} rank_kq: {rank_kq} rank_v: {rank_v} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")

                        #error budget has been reached, revert back to the previous SVD threshold
                        args.eigen_attn_params['threshold'] += 0.04
                        rank_kq = num_heads * torch.amax((torch.cumsum(eval_kq, dim = 1) < args.eigen_attn_params['threshold']).sum(1))
                        rank_v = num_heads * torch.amax((torch.cumsum(eval_v, dim = 1) < args.eigen_attn_params['threshold']).sum(1))
                        
                        qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config).to(dev)

                        output_lr = torch.stack([qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0] for j in range(args.nsamples)])
                        error = (torch.norm(output_hr - output_lr)/torch.norm(output_hr))
                        logger.info(
                            f"layer {i} final eigenattention error:{error} "
                            f"threshold:{args.eigen_attn_params['threshold']} "
                            f"rank_kq:{rank_kq} rank_v:{rank_v}"
                        )

                        del basis_kq, basis_v

                    del output_hr
                    torch.cuda.empty_cache()

                    # obtain output of model for propagation to next layer
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            for j in range(args.nsamples):
                                inps[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        elif is_mpt:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    args.eigen_attn_params['threshold'] = 0.98
                    error = 0.0
                    num_heads = lm.model.config.num_attention_heads
                    basis_kq, eval_kq, basis_v, eval_v = decompose_mpt_layer(layer, inps, args, num_heads, i, attention_mask,position_ids)
                    output_hr = torch.stack([layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_bias = position_bias)[0] for j in range(args.nsamples)])
                    
                    while error < args.error_budget and args.eigen_attn_params['threshold']> 0.3:
                        rank_kq = num_heads * torch.amax((torch.cumsum(eval_kq, dim = 1) < args.eigen_attn_params['threshold']).sum(1))
                        rank_v = num_heads * torch.amax((torch.cumsum(eval_v, dim = 1) < args.eigen_attn_params['threshold']).sum(1))

                        qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config).to(dev)
                        error = 0
                        output_lr = torch.stack([qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_bias = position_bias)[0] for j in range(args.nsamples)])
                        error = (torch.norm(output_hr - output_lr)/torch.norm(output_hr))
                        args.eigen_attn_params['threshold'] -= 0.02
                        
                        logger.info(f"layer {i} error:{error} threshold:{args.eigen_attn_params['threshold']} rank_kq: {rank_kq} rank_v: {rank_v} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")

                    args.eigen_attn_params['threshold'] += 0.04
                    rank_kq = num_heads * torch.amax((torch.cumsum(eval_kq, dim = 1) < args.eigen_attn_params['threshold']).sum(1))
                    rank_v = num_heads * torch.amax((torch.cumsum(eval_v, dim = 1) < args.eigen_attn_params['threshold']).sum(1))
                    
                    qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config).to(dev)

                    output_lr = torch.stack([qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_bias = position_bias)[0] for j in range(args.nsamples)])
                    error = (torch.norm(output_hr - output_lr)/torch.norm(output_hr))

                    # obtain output of model for propagation to next layer
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            for j in range(args.nsamples):
                                inps[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_bias = position_bias)[0]
                                
                    del basis_kq, basis_v, eval_kq, eval_v, output_hr, output_lr
                    torch.cuda.empty_cache()
                    
        elif is_llama:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    args.eigen_attn_params['threshold'] = 1.0
                    error = 0.0
                    num_heads = lm.model.config.num_key_value_heads
                    basis_kq, eval_kq, basis_v, eval_v = decompose_llama_layer(layer, inps, args, num_heads, i, attention_mask, position_ids)

                    max_rank_kq = layer.self_attn.k_proj.weight.shape[0]
                    max_rank_v = layer.self_attn.v_proj.weight.shape[0]

                    output_hr = torch.stack([layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids = position_ids, layer_idx = i)[0] for j in range(args.nsamples)])
                    
                    rank_kq = max_rank_kq
                    rank_v = max_rank_v
                    min_thresh = 0.45
                    while error < args.error_budget and (rank_kq > min_thresh * max_rank_kq) and (rank_v > min_thresh * max_rank_v):
                        rank_kq = num_heads * ((torch.cumsum(eval_kq, dim = 0) < args.eigen_attn_params['threshold']).sum())
                        rank_v = num_heads * torch.amax((torch.cumsum(eval_v, dim = 1) < args.eigen_attn_params['threshold']).sum(1))

                        qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config, i).to(dev)
                        qlayer = qlayer.to(dev)

                        output_lr = torch.stack([qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids = position_ids, layer_idx = i)[0] for j in range(args.nsamples)])
                        error = (torch.norm(output_hr - output_lr)/torch.norm(output_hr))
                        args.eigen_attn_params['threshold'] -= 0.02

                    args.eigen_attn_params['threshold'] += 0.04
                    rank_v = num_heads * torch.amax((torch.cumsum(eval_v, dim = 1) < args.eigen_attn_params['threshold']).sum(1))
                    rank_kq = num_heads * ((torch.cumsum(eval_kq, dim = 0) < args.eigen_attn_params['threshold']).sum())
                    qlayer = DecoderLayer(layer, args, basis_kq, rank_kq, basis_v, rank_v, lm.model.config, i).to(dev)

                    # del basis_kq, basis_v, eval_kq, eval_v, output_hr, output_lr
                    del basis_kq, basis_v, eval_kq, eval_v
                    torch.cuda.empty_cache()

                    output_lr = torch.stack([qlayer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids = position_ids, layer_idx = i)[0] for j in range(args.nsamples)])
                    error = (torch.norm(output_hr - output_lr)/torch.norm(output_hr))

                    # obtain output of model for propagation to next layer
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            for j in range(args.nsamples):
                                inps[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids = position_ids, layer_idx = i)[0]
        
        # logger.info(f"layer {i} error:{error} threshold:{args.eigen_attn_params['threshold']} rank_kq: {rank_kq} rank_v: {rank_v} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
        
    

        qlayer.half() 

        layers[i] = qlayer.to("cpu")
        del layer
        torch.cuda.empty_cache()

    del inps
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model
