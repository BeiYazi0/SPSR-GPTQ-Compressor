from copy import deepcopy

import torch
import time
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from datas import get_examples


class ShortGptPruner:
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
        dataloader = get_examples(args.dataset, tokenizer, n_samples=args.num_examples, seq_len=2048)
        print("dataset loading complete")

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers
        n = len(layers)

        pruning_num = int(n * args.final_s)
        score = torch.zeros((n,), device=device)
        for j in tqdm(range(len(dataloader)), desc="Collecting hidden states"):
            input_ids = dataloader[j].unsqueeze(0).to(device)
            hidden_states = model(input_ids, output_hidden_states=True).hidden_states
            for i in range(len(score)):
                score[i] += (F.normalize(hidden_states[i][0], p=2, dim=1) *
                             F.normalize(hidden_states[i + 1][0], p=2, dim=1)).sum(dim=1).mean(dim=0)

        for i in range(len(score)):
            score[i] = 1 - score[i] / args.num_examples
        print(score)

        sort_index = torch.argsort(score)
        pruning_layers = sort_index[:int(n * args.final_s)].tolist()
        pruning_layers.sort()
        print(pruning_layers)

        # 删除层
        for idx in sorted(pruning_layers)[::-1]:
            del layers[idx]

        after_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("After prune, #parameters: {}".format(after_pruning_parameters))

        model.config.use_cache = use_cache
        if args.save_path is not None:
            self.tokenizer.save_pretrained(args.save_path)
            self.model.save_pretrained(args.save_path)
