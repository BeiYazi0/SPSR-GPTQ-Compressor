from copy import deepcopy

import torch
import torch.nn as nn

from datas import get_examples
import torch.nn.functional as F

INTERVAL = 1
LOWEST_LAY = 0
THRESHOLD = 0.98


def recover_layers(model, merge_base_lay, merge_layer_num, recover_layers):
    layers = model.model.layers
    new_layers = nn.ModuleList(layers[:merge_base_lay] + recover_layers + layers[merge_base_lay + 1:])
    model.model.layers = new_layers


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def merge_layers(model, merge_indices, lamda=1., model_type="llama"):
    merge_indices = sorted([int(idx) for idx in merge_indices])

    # 找到原始层列表和 config 的字段
    if "Llama" in model_type or "llama" in model_type:
        layers = model.model.layers
    elif "opt" in model_type:
        layers = model.model.decoder.layers
    elif "Qwen" in model_type:
        layers = model.transformer.h
    else:
        layers = model.model.layers

    # 构造新的 ModuleList，只保留未被移除的层
    merge_set = set()
    base_layers = {}
    for merge_idx in merge_indices:
        merge_set.add(merge_idx)
        base_idx = merge_idx - 1
        while base_idx in merge_set:
            base_idx -= 1
        if base_idx not in base_layers:
            base_layers[base_idx] = find_layers(deepcopy(layers[base_idx]))

        base_subset = base_layers[base_idx]
        target_subset = find_layers(layers[base_idx])
        subset = find_layers(layers[merge_idx])
        for name in target_subset:
            # lamda = torch.cosine_similarity(base_subset[name].weight.data.flatten().unsqueeze(0), subset[name].weight.data.flatten().unsqueeze(0))
            # print(lamda)
            target_subset[name].weight.data.add_(
                (subset[name].weight.data - base_subset[name].weight.data) * lamda
            )
    del base_layers


def correct_merge_layers_return_model(model, merge_base_lay, merge_layer_num):
    merge_layer_num = min(merge_layer_num, len(model.model.layers) - merge_base_lay - 1)
    merge_indices = list(range(merge_base_lay + 1, merge_base_lay + 1 + merge_layer_num))
    # merge_layers(model, merge_indices, lamda=1., model_type="llama")
    prune_layers = [model.model.layers[i] for i in range(merge_base_lay, merge_base_lay + 1 + merge_layer_num)]
    model.model.layers = nn.ModuleList([layer for idx, layer in enumerate(model.model.layers) if idx not in merge_indices])
    return prune_layers


def merge_layers_return_model(model, merge_base_lay, merge_layer_num):
    merge_layer_num = min(merge_layer_num, len(model.model.layers) - merge_base_lay - 1)

    model_copy = model
    for diff_lay in range(merge_base_lay + 1, merge_base_lay + 1 + merge_layer_num):
        # gate_proj
        model_copy.model.layers[merge_base_lay].mlp.gate_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.gate_proj.weight.data - model_copy.model.layers[
                merge_base_lay].mlp.gate_proj.weight.data
        )
        # down_proj
        model_copy.model.layers[merge_base_lay].mlp.down_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.down_proj.weight.data - model_copy.model.layers[
                merge_base_lay].mlp.down_proj.weight.data
        )
        # up_proj
        model_copy.model.layers[merge_base_lay].mlp.up_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.up_proj.weight.data - model_copy.model.layers[
                merge_base_lay].mlp.up_proj.weight.data
        )

        # q_proj
        model_copy.model.layers[merge_base_lay].self_attn.q_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.q_proj.weight.data - model_copy.model.layers[
                merge_base_lay].self_attn.q_proj.weight.data
        )

        # k_proj
        model_copy.model.layers[merge_base_lay].self_attn.k_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.k_proj.weight.data - model_copy.model.layers[
                merge_base_lay].self_attn.k_proj.weight.data
        )

        # v_proj
        model_copy.model.layers[merge_base_lay].self_attn.v_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.v_proj.weight.data - model_copy.model.layers[
                merge_base_lay].self_attn.v_proj.weight.data
        )

        # o_proj
        model_copy.model.layers[merge_base_lay].self_attn.o_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.o_proj.weight.data - model_copy.model.layers[
                merge_base_lay].self_attn.o_proj.weight.data
        )

    prune_layers = [model_copy.model.layers[i] for i in range(merge_base_lay, merge_base_lay + 1 + merge_layer_num)]
    model.model.layers = nn.ModuleList([layer for idx, layer in enumerate(model.model.layers) if idx not in list(
        range(merge_base_lay + 1, merge_base_lay + 1 + merge_layer_num))])
    return prune_layers


def prepare_calibration_input(model, layers, dataloader, device):
    model.seqlen = 2048
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                cache['attention_mask'] = kwargs['attention_mask']
            if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
                cache['position_ids'] = kwargs['position_ids']
            if "position_embeddings" in kwargs and kwargs['position_embeddings'] is not None:
                cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.unsqueeze(0).to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    model.config.use_cache = use_cache
    del cache["i"]

    return inps, cache


def cal_last_hidden_sim(model, origin_output, testenc):
    sim_ls = []
    nsamples, seqlen = testenc.shape[0], testenc.shape[1]
    device = model.device
    # assert not torch.any(torch.isnan(outputs)), 'nan exists!'
    for i in range(nsamples):
        inputs = testenc[i].to(device).unsqueeze(0)
        outs = model.model(inputs)[0].squeeze(0)
        sim_ls.append(
            (F.normalize(origin_output[i], p=2, dim=1) * F.normalize(outs, p=2, dim=1)).sum(dim=1).mean(
                dim=0).item())
        # sim_ls[i] = torch.cosine_similarity(outputs[i].flatten().unsqueeze(0), origin_output[i].flatten().unsqueeze(0))
    res = torch.mean(torch.tensor(sim_ls))
    print(sim_ls, res)
    return res


class LacoPruner:  # one shot
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def prune(self, args):
        model, tokenizer = self.model, self.tokenizer
        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        use_cache = model.config.use_cache
        model.config.use_cache = False

        device = model.device
        print("loading calibdation data")
        dataloader = get_examples("c4", tokenizer, n_samples=args.num_examples, seq_len=2048)
        print("dataset loading complete")

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers

        dtype = next(iter(model.parameters())).dtype
        model.seqlen = 2048
        origin_output = torch.zeros((len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype,
                                    device=device)
        nsamples, seqlen = dataloader.shape[0], dataloader.shape[1]
        for i in range(0, nsamples):
            inputs = dataloader[i].to(device).unsqueeze(0)
            origin_output[i] = model.model(inputs)[0]
        assert not torch.any(torch.isnan(origin_output)), 'nan exists!'

        target_layers = len(model.model.layers) - int(len(model.model.layers) * args.final_s)
        HIGHEST_LAY = len(layers) - 1
        MERGE_LAYERS = int(args.final_s * len(layers)) + 1
        lay = HIGHEST_LAY - MERGE_LAYERS
        while lay >= LOWEST_LAY:
            print(lay)
            print('current model layer', len(model.model.layers))
            if target_layers >= len(model.model.layers): break
            prune_layers = correct_merge_layers_return_model(model, lay, MERGE_LAYERS - 1)
            sim_value = cal_last_hidden_sim(model, origin_output, dataloader)
            if sim_value > THRESHOLD:
                lay -= INTERVAL
                if lay >= len(model.model.layers):
                    lay = len(model.model.layers) - 1 - MERGE_LAYERS
                del prune_layers
            else:
                recover_layers(model, lay, MERGE_LAYERS - 1, prune_layers)
                lay -= 1
            torch.cuda.empty_cache()

        after_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("After prune, #parameters: {}".format(after_pruning_parameters))

        model.config.use_cache = use_cache
        if args.save_path is not None:
            self.tokenizer.save_pretrained(args.save_path)
            self.model.save_pretrained(args.save_path)
