from copy import deepcopy
import random

import torch
import torch.nn as nn
from tqdm import tqdm

from datas import get_examples
from utils import block_replace, turn_on_layer, turn_off_layer


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


class SLEBPruner:
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

        pruning_num = int(len(layers) * args.final_s)
        if len(args.remove_list) > 0:
            for idx in args.remove_list:
                turn_off_layer(model, idx)
            pruning_num -= len(args.remove_list)
            print(get_loss(model, dataloader, device))

        pruning_lst = [] + args.remove_list
        for q in range(pruning_num):
            min_loss = torch.inf
            min_layer_idx = None
            for i in tqdm(range(len(layers)), desc="layer search"):
                if i in pruning_lst: continue
                turn_off_layer(model, i)
                loss = get_loss(model, dataloader, device)
                if loss < min_loss:
                    min_loss = loss
                    min_layer_idx = i
                turn_on_layer(model, i)

            print(min_loss, min_layer_idx)
            turn_off_layer(model, min_layer_idx)
            pruning_lst.append(min_layer_idx)

        print(pruning_lst)
        for idx in sorted(pruning_lst)[::-1]:
            del layers[idx]

        layer_num = sum(p.numel() for p in layers[0].parameters())
        after_pruning_parameters = sum(p.numel() for p in self.model.parameters()) - layer_num * pruning_num
        print("After prune, #parameters: {}".format(after_pruning_parameters))

        model.config.use_cache = use_cache
        if args.save_path is not None:
            self.tokenizer.save_pretrained(args.save_path)
            self.model.save_pretrained(args.save_path)


class SLEBOneShotPruner:  # one shot
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

        pruning_num = int(len(layers) * args.final_s)
        losses = torch.zeros((len(layers),), device=device)
        for i in range(0, len(layers)):
            print(f"turn off layer {i}")
            turn_off_layer(model, i)
            losses[i] = get_loss(model, dataloader, device)
            turn_on_layer(model, i)

        print(losses)
        pruning_lst = torch.sort(losses)[1][:pruning_num].tolist()
        print(pruning_lst)
        for idx in pruning_lst:
            turn_off_layer(model, idx)

        layer_num = sum(p.numel() for p in layers[0].parameters())
        after_pruning_parameters = sum(p.numel() for p in self.model.parameters()) - layer_num * pruning_num
        print("After prune, #parameters: {}".format(after_pruning_parameters))

        model.config.use_cache = use_cache
        if args.save_path is not None:
            self.tokenizer.save_pretrained(args.save_path)
            self.model.save_pretrained(args.save_path)


