import torch
import torch.nn as nn

from datas import get_examples


def remove_layers_and_update_config(model, remove_indices, model_type="llama"):
    # remove_indices: 可迭代的要删除的层索引（int）
    remove_set = set(int(i) for i in remove_indices)

    # 找到原始层列表和 config 的字段
    if "Llama" in model_type or "llama" in model_type:
        layers = model.model.layers
        cfg = model.config
        num_layers_attr = "num_hidden_layers"
    elif "opt" in model_type:
        layers = model.model.decoder.layers
        cfg = model.config
        num_layers_attr = "num_hidden_layers"
    elif "Qwen" in model_type:
        layers = model.transformer.h
        cfg = model.config
        num_layers_attr = "n_layer"  # 视 Qwen config 而定
    else:
        layers = model.model.layers
        cfg = model.config
        num_layers_attr = "num_hidden_layers"

    # 构造新的 ModuleList，只保留未被移除的层
    new_layers = nn.ModuleList([layer for idx, layer in enumerate(layers) if idx not in remove_set])

    # 替换到模型中（注意 opt 路径不同）
    if "opt" in model_type:
        model.model.decoder.layers = new_layers
    elif "Qwen" in model_type:
        model.transformer.h = new_layers
    else:
        model.model.layers = new_layers

    # 更新 config 中记录的层数
    new_n = len(new_layers)
    if hasattr(cfg, num_layers_attr):
        setattr(cfg, num_layers_attr, new_n)
    else:
        # 如果找不到常见字段，打印提醒，并可选择设置 config.layers 或其它自定义
        print(f"Warning: config has no attribute {num_layers_attr}; config keys: {list(cfg.__dict__.keys())}")

    return new_layers


def recover_layers(model, index, recover_layers):
    layers = model.model.layers
    new_layers = nn.ModuleList(layers[:index] + recover_layers + layers[index:])
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


class TaylorPruner:  # one shot
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def prune(self, args):
        model, tokenizer = self.model, self.tokenizer
        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        use_cache = model.config.use_cache
        model.config.use_cache = False

        device = model.device
        print("loading calibdation data")
        dataloader = get_examples("c4", tokenizer, n_samples=args.num_examples, seq_len=128).to(device)
        print("dataset loading complete")

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers

        pruning_num = int(len(layers) * args.final_s)
        if len(args.remove_list) > 0:
            layers = remove_layers_and_update_config(self.model, args.remove_list,
                                                     model_type=args.base_model)
            pruning_num -= len(args.remove_list)
            model = self.model

        # masks = torch.ones((len(layers),), requires_grad=True, device=device, dtype=dtype)
        # for i in range(0, len(layers)):
        #     layer = layers[i]
        #     subset = find_layers(layer)
        #     for name in subset:
        #         module = subset[name]
        #         module.mask = masks[i]
        #         module.forward = types.MethodType(mask_forward, module)

        # get_grad(model, dataloader, device)
        for j in range(args.num_examples):
            batch_input = dataloader[j].unsqueeze(0)
            loss = model(batch_input, labels=batch_input).loss
            loss.backward()

        score = torch.zeros((len(layers),), device=device, dtype=torch.float16 if args.fp16 else torch.float32)

        with torch.no_grad():
            for i in range(0, len(layers)):
                layer = layers[i]
                subset = find_layers(layer)
                for name in subset:
                    weight = subset[name].weight
                    score[i] += torch.sum(torch.abs(weight * weight.grad)).to(device)
            model.zero_grad()

        print(score)
        score[0] = score[1] = score[-2] = score[-1] = torch.inf  # not remove
        pruning_idxs = torch.sort(score)[1][:pruning_num].tolist()
        print(pruning_idxs)

        remove_layers_and_update_config(self.model, pruning_idxs,
                                        model_type=args.base_model)
        after_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("After prune, #parameters: {}".format(after_pruning_parameters))

        model.config.use_cache = use_cache
        if args.save_path is not None:
            self.tokenizer.save_pretrained(args.save_path)
            self.model.save_pretrained(args.save_path)


class TaylorIterPruner:  # one shot
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def prune(self, args):
        model, tokenizer = self.model, self.tokenizer
        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        use_cache = model.config.use_cache
        model.config.use_cache = False

        device = model.device
        print("loading calibdation data")
        dataloader = get_examples("bookcorpus", tokenizer, n_samples=args.num_examples, seq_len=64).to(device)
        print("dataset loading complete")

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers

        pruning_num = int(len(layers) * args.final_s)
        if len(args.remove_list) > 0:
            layers = remove_layers_and_update_config(self.model, args.remove_list,
                                                     model_type=args.base_model)
            pruning_num -= len(args.remove_list)
            model = self.model

        # masks = torch.ones((len(layers),), requires_grad=True, device=device, dtype=dtype)
        # for i in range(0, len(layers)):
        #     layer = layers[i]
        #     subset = find_layers(layer)
        #     for name in subset:
        #         module = subset[name]
        #         module.mask = masks[i]
        #         module.forward = types.MethodType(mask_forward, module)

        pruning_lst = []
        for q in range(pruning_num):
            # get_grad(model, dataloader, device)
            for j in range(args.num_examples):
                batch_input = dataloader[j].unsqueeze(0)
                loss = model(batch_input, labels=batch_input).loss
                loss.backward()

            score = torch.zeros((len(layers),), device=device, dtype=torch.float16 if args.fp16 else torch.float32)

            with torch.no_grad():
                for i in range(0, len(layers)):
                    layer = layers[i]
                    subset = find_layers(layer)
                    for name in subset:
                        weight = subset[name].weight
                        score[i] += torch.sum(torch.abs(weight * weight.grad)).to(device)
                model.zero_grad()

            print(score)

            pruning_idx = torch.argmin(score)
            print(pruning_idx)
            pruning_lst.append(pruning_idx)

            layers = remove_layers_and_update_config(self.model, [pruning_idx], model_type=args.base_model)

        after_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("After prune, #parameters: {}".format(after_pruning_parameters))

        model.config.use_cache = use_cache
        if args.save_path is not None:
            self.tokenizer.save_pretrained(args.save_path)
            self.model.save_pretrained(args.save_path)
