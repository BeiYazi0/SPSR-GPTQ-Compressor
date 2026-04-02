import torch
import torch.nn as nn
from tqdm import tqdm

from datas import get_examples
from utils import block_replace, turn_on_mha, turn_off_mha, turn_off_mlp, turn_on_mlp


@torch.no_grad()
def get_loss(model, testenc, device=None):
    # Calculate number of samples
    nsamples, seqlen = testenc.shape[0], testenc.shape[1]

    # List to store negative log likelihoods
    losses = []
    # print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0, nsamples):
        # Calculate end index

        # Prepare inputs and move to device
        inputs = testenc[i].to(device).unsqueeze(0)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        loss = loss.float() * seqlen

        # Append to list of negative log likelihoods
        losses.append(loss)

    # Compute sum of negative log_likelihood
    loss_sum = torch.exp(torch.stack(losses).sum() / (nsamples * seqlen))

    return loss_sum.item()


def get_model_params(model):
    return sum(int(p.numel()) for p in model.parameters())


class BlockPruner:
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

        # replace with onoff llama
        block_replace(model)

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers

        mha_num, mlp_num = get_model_params(layers[0].self_attn), get_model_params(layers[0].mlp)
        layer_num = mlp_num + mha_num
        pruning_num = int(layer_num * len(layers) * args.final_s)
        pruning_lst = [] + args.remove_list
        cur_remove_num = 0
        if len(args.remove_list) > 0:
            for idx in args.remove_list:
                layer_idx = idx // 2
                if idx % 2 == 0:
                    turn_off_mha(model, layer_idx)
                    cur_remove_num += mha_num
                else:
                    turn_off_mlp(model, layer_idx)
                    cur_remove_num += mlp_num
            print(get_loss(model, dataloader, device))

        while cur_remove_num + 1e8 < pruning_num:
            min_loss = torch.inf
            min_block_idx = None
            for block_idx in tqdm(range(2 * len(layers)), desc="block search"):
                if block_idx in pruning_lst: continue

                layer_idx = block_idx // 2
                if block_idx % 2 == 0:
                    turn_off_mha(model, layer_idx)
                    loss = get_loss(model, dataloader, device)
                    turn_on_mha(model, layer_idx)
                else:
                    turn_off_mlp(model, layer_idx)
                    loss = get_loss(model, dataloader, device)
                    turn_on_mlp(model, layer_idx)

                if loss < min_loss:
                    min_loss = loss
                    min_block_idx = block_idx

            print(min_loss, min_block_idx)
            min_layer_idx = min_block_idx // 2
            if min_block_idx % 2 == 0:
                turn_off_mha(model, min_layer_idx)
                cur_remove_num += mha_num
            else:
                turn_off_mlp(model, min_layer_idx)
                cur_remove_num += mlp_num
            pruning_lst.append(min_block_idx)

        print(pruning_lst)
        after_pruning_parameters = sum(p.numel() for p in self.model.parameters()) - cur_remove_num
        print("After prune, #parameters: {}".format(after_pruning_parameters))

        model.config.use_cache = use_cache
        if args.save_path is not None:
            self.tokenizer.save_pretrained(args.save_path)
            self.model.save_pretrained(args.save_path)


class BlockOneShotPruner:  # one shot
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

        # replace with onoff llama
        block_replace(model)

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers

        mha_num, mlp_num = get_model_params(layers[0].self_attn), get_model_params(layers[0].mlp)
        layer_num = mlp_num + mha_num
        pruning_num = int(layer_num * len(layers) * args.final_s)
        pruning_lst = [] + args.remove_list
        cur_remove_num = 0

        losses = torch.zeros((2 * len(layers), ), device=device)
        for block_idx in range(0, 2 * len(layers)):
            if block_idx in pruning_lst: continue
            print(f"turn off block {block_idx}")

            layer_idx = block_idx // 2
            if block_idx % 2 == 0:
                turn_off_mha(model, layer_idx)
                losses[block_idx] = get_loss(model, dataloader, device)
                turn_on_mha(model, layer_idx)
            else:
                turn_off_mlp(model, layer_idx)
                losses[block_idx] = get_loss(model, dataloader, device)
                turn_on_mlp(model, layer_idx)


        print(losses)
        sorted_block_idx = torch.sort(losses)[1]
        i = 0
        while cur_remove_num < pruning_num:
            min_block_idx = sorted_block_idx[i]
            min_layer_idx = min_block_idx // 2
            if min_block_idx % 2 == 0:
                turn_off_mha(model, min_layer_idx)
                cur_remove_num += mha_num
            else:
                turn_off_mlp(model, min_layer_idx)
                cur_remove_num += mlp_num
            pruning_lst.append(min_block_idx)
            i += 1

        print(sorted_block_idx[:i])

        after_pruning_parameters = sum(p.numel() for p in self.model.parameters()) - cur_remove_num
        print("After prune, #parameters: {}".format(after_pruning_parameters))

        model.config.use_cache = use_cache
        if args.save_path is not None:
            self.tokenizer.save_pretrained(args.save_path)
            self.model.save_pretrained(args.save_path)
