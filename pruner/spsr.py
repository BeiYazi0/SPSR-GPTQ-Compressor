import datetime
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Subset, Dataset
from tqdm import tqdm

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from datas import get_examples


class DiagCalPruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def select(self, args, dataloader):
        model, tokenizer = self.model, self.tokenizer
        if not args.fp16 and not args.bf16:
            model.half()

        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        use_cache = model.config.use_cache
        model.config.use_cache = False

        device = model.device

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers
        num_layers = len(layers)

        pruning_num = int(num_layers * args.final_s)
        # pruning_num = 1
        hidden_dim = model.config.hidden_size
        A_acc = [torch.zeros(hidden_dim, hidden_dim).to(device) for _ in range(num_layers)]
        B_acc = [torch.zeros(hidden_dim, hidden_dim).to(device) for _ in range(num_layers)]
        token_counts = [0] * (num_layers - pruning_num + 1)  # 各层对处理的token总数
        ratios = [0] * (num_layers - pruning_num + 1)  # 存储各层对角化程度
        for j in tqdm(range(len(dataloader)), desc="Collecting hidden states"):
            input_ids = dataloader[j].unsqueeze(0).to(device)
            hidden_states = model(input_ids, output_hidden_states=True).hidden_states
            for i in range(len(token_counts)):
                # 获取第i层和第i+pruning_num层的隐藏状态
                h1 = hidden_states[i][0].to(torch.float32)  # [seq_len, hidden_dim]
                h2 = hidden_states[i + pruning_num][0].to(torch.float32)  # [seq_len, hidden_dim]
                assert len(h2.shape) == 2

                # 中心化处理
                h1_centered = h1 - h1.mean(0, keepdim=True)
                h2_centered = h2 - h2.mean(0, keepdim=True)

                # 在线累加协方差矩阵
                A_acc[i] += h1_centered.T @ h1_centered  # [hidden_dim, hidden_dim]
                B_acc[i] += h1_centered.T @ h2_centered  # [hidden_dim, hidden_dim]
                token_counts[i] += h1.shape[0]  # 累加token数

        # 计算各层对角化程度
        lambda_val = 1e-6
        for i in tqdm(range(num_layers - pruning_num + 1), desc="Computing diagonal ratios"):
            if token_counts[i] == 0: continue

            # 正则化协方差矩阵
            A = A_acc[i] / token_counts[i] + lambda_val * torch.eye(hidden_dim, device=device)
            B = B_acc[i] / token_counts[i]

            # 岭回归求解线性映射 w [hidden_dim, hidden_dim]
            try:
                # Cholesky分解 (高效稳定)
                L = torch.linalg.cholesky(A)
                w = torch.cholesky_solve(B, L)
            except RuntimeError:
                # SVD回退 (应对病态矩阵)
                U, S, Vh = torch.linalg.svd(A, full_matrices=False)
                w = Vh.t() @ ((U.t() @ B) / S.unsqueeze(1))

            # 计算对角化程度 = (对角元素平方和) / (全部元素平方和)
            diag_sq = w.diagonal().pow(2).sum()
            full_sq = w.pow(2).sum()
            ratios[i] = (diag_sq / full_sq).item()

        print(ratios)

        model.config.use_cache = use_cache
        if not args.fp16 and not args.bf16:
            model.float()

    def prune(self, args):
        model, tokenizer = self.model, self.tokenizer

        device = model.device
        print("loading calibdation data")
        dataloader = get_examples(args.dataset, tokenizer, n_samples=args.num_examples, seq_len=2048)
        print("dataset loading complete")

        self.select(args, dataloader)
        torch.cuda.empty_cache()


class SPSRCIPruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def select(self, args, dataloader):
        model, tokenizer = self.model, self.tokenizer
        if not args.fp16 and not args.bf16:
            model.half()

        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        use_cache = model.config.use_cache
        model.config.use_cache = False

        device = model.device

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
        if len(args.remove_list) == 0:
            score = torch.zeros((n - pruning_num + 1,), device=device)
            for j in tqdm(range(len(dataloader)), desc="Collecting hidden states"):
                input_ids = dataloader[j].unsqueeze(0).to(device)
                hidden_states = model(input_ids, output_hidden_states=True).hidden_states
                for i in range(len(score)):
                    score[i] += (F.normalize(hidden_states[i][0], p=2, dim=1) *
                                 F.normalize(hidden_states[i + pruning_num][0], p=2, dim=1)).sum(dim=1).mean(dim=0)

            for i in range(len(score)):
                score[i] = 1 - score[i] / args.num_examples
            print(score)

            start_index = torch.argmin(score)
            pruning_layers = list(range(start_index, start_index + pruning_num))
            print(pruning_layers)
        else:
            start_index = args.remove_list[0]
            pruning_layers = list(range(start_index, start_index + pruning_num))
            print(pruning_layers)


        hidden_inputs = torch.zeros((len(dataloader), len(dataloader[0]), model.config.hidden_size), device=device,
                                    dtype=torch.bfloat16 if args.bf16 else torch.float16)
        out_norm = torch.zeros((model.config.hidden_size,), device=device, dtype=torch.float32)
        inp_norm = torch.zeros_like(out_norm)
        for j in tqdm(range(len(dataloader)), desc="Collecting hidden states"):
            input_ids = dataloader[j].unsqueeze(0).to(device)
            hidden_states = model(input_ids, output_hidden_states=True).hidden_states
            hidden_inputs[j] = hidden_states[start_index][0]
            inp_norm += torch.mean(hidden_states[start_index][0].to(dtype=torch.float32) ** 2, dim=0).to(
                device) / args.num_examples
            out_norm += torch.mean(hidden_states[start_index + pruning_num][0].to(dtype=torch.float32) ** 2,
                                   dim=0).to(device) / args.num_examples

        s = ((out_norm / inp_norm) ** 0.5)
        b = torch.zeros_like(s)

        if args.bf16:
            s = s.to(torch.bfloat16)
            b = b.to(torch.bfloat16)
        print(torch.mean(s.abs()), torch.mean(b.abs()))

        new_layers = nn.ModuleList(layers[:start_index] +
                                   [IdentityNormLike(s, b)] + layers[start_index + pruning_num:])
        if "Llama" in args.base_model or "llama" in args.base_model:
            model.model.layers = new_layers
        elif "opt" in args.base_model:
            model.model.decoder.layers = new_layers
        del layers[start_index:start_index + pruning_num]

        model.config.use_cache = use_cache
        if not args.fp16 and not args.bf16:
            model.float()

        return start_index, hidden_inputs

    def prune(self, args):
        model, tokenizer = self.model, self.tokenizer

        device = model.device
        print("loading calibdation data")
        dataloader = get_examples(args.dataset, tokenizer, n_samples=args.num_examples, seq_len=2048)
        print("dataset loading complete")

        target_idx, hidden_inputs = self.select(args, dataloader)
        torch.cuda.empty_cache()

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers

        # 冻结
        for param in model.named_parameters():
            param[1].requires_grad = False

        for layer in layers:
            if isinstance(layer, IdentityNormLike):
                layer.s = nn.Parameter(layer.s)
                layer.bias = nn.Parameter(layer.bias)

        dataloader = [(inps, outs) for inps, outs in zip(hidden_inputs.cpu(), dataloader.cpu())]
        total_size = len(dataloader)
        train_size = int(0.9 * total_size)
        train_dataset = Subset(dataloader, range(train_size))
        eval_dataset = Subset(dataloader, range(train_size, total_size))

        data_collator = DataCollator(tokenizer)

        output_dir = f'{args.output_dir}/{args.epochs}'
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            warmup_steps=int(args.epochs * len(train_dataset) * 0.175),
            weight_decay=args.wd,
            logging_dir='./logs',
            eval_strategy='epoch',
            logging_steps=500,
            # lr_scheduler_type="cosine",
            # warmup_ratio=0.175,
            # lr_scheduler_kwargs={"num_cycles": 5},
            learning_rate=args.lr,
            save_strategy="no",
        )

        trainer = SPSRLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        trainer.position_ids = torch.arange(0, 2048).unsqueeze(0).to(device)
        trainer.start_idx = target_idx

        trainer.train()

        for param in model.named_parameters():
            param[1].requires_grad = False
        torch.cuda.empty_cache()

        model.half()
        for layer in layers:
            if isinstance(layer, IdentityNormLike):
                if not isinstance(layer.bias, nn.Parameter):
                    layer.bias = nn.Parameter(layer.bias)
                print(torch.mean(layer.s.data.abs()), torch.mean(layer.bias.data.abs()))
                print(layer.bias)


class SPSRPearsonSinglePruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def select(self, args, dataloader):
        model, tokenizer = self.model, self.tokenizer
        if not args.fp16 and not args.bf16:
            model.half()

        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        use_cache = model.config.use_cache
        model.config.use_cache = False

        device = model.device

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers
        n = len(layers)

        pruning_num = 1
        convs = torch.zeros((n - pruning_num + 1, model.config.hidden_size), device=device)
        std1s = torch.zeros((n - pruning_num + 1, model.config.hidden_size), device=device)
        std2s = torch.zeros((n - pruning_num + 1, model.config.hidden_size), device=device)
        score = torch.zeros((n - pruning_num + 1,), device=device)
        for j in tqdm(range(len(dataloader)), desc="Collecting hidden states"):
            input_ids = dataloader[j].unsqueeze(0).to(device)
            hidden_states = model(input_ids, output_hidden_states=True).hidden_states
            for i in range(len(score)):
                # 获取第i层和第i+pruning_num层的隐藏状态
                h1 = hidden_states[i][0].to(torch.float32)  # [seq_len, hidden_dim]
                h2 = hidden_states[i + pruning_num][0].to(torch.float32)  # [seq_len, hidden_dim]
                assert len(h2.shape) == 2

                # 中心化处理
                h1_centered = h1 - h1.mean(0, keepdim=True)
                h2_centered = h2 - h2.mean(0, keepdim=True)

                # 计算皮尔逊相关系数
                cov = (h1_centered * h2_centered).sum(0)  # 协方差 [hidden_dim]
                std1 = torch.norm(h1_centered, p=2, dim=0)  # 标准差 [hidden_dim]
                std2 = torch.norm(h2_centered, p=2, dim=0)  # 标准差 [hidden_dim]

                convs[i] += cov
                std1s[i] += std1 ** 2
                std2s[i] += std2 ** 2

        corr = convs / ((std1s * std2s) ** 0.5 + 1e-8)
        score = (1 - corr ** 2).max(dim=1)[0]

        start_index = torch.argmin(score)
        pscore = (score - score[start_index]).abs()
        pscore[start_index] = torch.inf
        if (pscore.min()) < 1e-3:
            score = ((1 - corr ** 2).min(dim=1)[0] + (1 - corr ** 2).max(dim=1)[0]) / 2
            start_index = torch.argmin(score)
        print(score)

        score[0] = score[1] = score[-2] = score[-1] = torch.inf  # not remove

        sort_index = torch.argsort(score)
        pruning_layers = sort_index[:int(n * args.final_s)].tolist()
        print(pruning_layers)

        model.config.use_cache = use_cache
        if not args.fp16 and not args.bf16:
            model.float()

        return pruning_layers

    def prune(self, args, dataloader):
        pruning_layers = self.select(args, dataloader)
        torch.cuda.empty_cache()

        return pruning_layers


class SPSRLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = nn.CrossEntropyLoss()  # top-k label
        self.start_idx = -1
        self.position_ids = None
        print("direct loss")

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        We override the compute_loss method of the Trainer class
        to use our custom loss instead of the default
        cross entropy.
        """
        # res = model(**inputs)
        #
        # return (res.loss, res.logits) if return_outputs else res.loss

        labels = inputs.get("labels")[:, 1:]
        hidden_states = inputs.get("input_ids")

        layers = model.model.layers
        for decoder_layer in layers[self.start_idx:]:
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=self.position_ids,
            )
            hidden_states = layer_outputs[0]

        hidden_states = model.model.norm(hidden_states)

        logits = model.lm_head(hidden_states)[:, :-1, :].contiguous()
        loss = self.loss_fct(logits.reshape(-1, logits.size(-1)).to(labels.device), labels.reshape(-1))

        return (loss, logits) if return_outputs else loss


class SPSRPlusPruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def select(self, args, dataloader, remove_list=None, delect_layers=True):
        model, tokenizer = self.model, self.tokenizer
        if not args.fp16 and not args.bf16:
            model.half()

        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        use_cache = model.config.use_cache
        model.config.use_cache = False

        device = model.device

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers
        n = len(layers)

        if remove_list is None:
            pruning_num = int(n * args.final_s)
            assert len(args.remove_list) == pruning_num, "removing layers not enough"
            pruning_lst = sorted(args.remove_list)
        else:
            pruning_lst = sorted(remove_list)
            pruning_num = len(pruning_lst)
        start = pruning_lst[0]
        candidates = []
        for i in range(1, pruning_num):  # merge
            if pruning_lst[i] - pruning_lst[i - 1] > 1:
                candidates.append((start, pruning_lst[i - 1] + 1))
                start = pruning_lst[i]
        candidates.append((start, pruning_lst[pruning_num - 1] + 1))
        print(candidates)

        norm_ratio = {}
        hidden_inputs = torch.zeros((len(dataloader), len(dataloader[0]), model.config.hidden_size), device=device, dtype=torch.bfloat16 if args.bf16 else torch.float16)
        out_norms = {i: torch.zeros((model.config.hidden_size,), device=device, dtype=torch.float32) for i in candidates}
        inp_norms = {i: torch.zeros_like(out_norms[i]) for i in candidates}
        target_idx = pruning_lst[0]
        for j in tqdm(range(len(dataloader)), desc="Collecting hidden states"):
            input_ids = dataloader[j].unsqueeze(0).to(device)
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            hidden_inputs[j] = hidden_states[target_idx][0]

            for idx_lst in candidates:
                start_index, end_index = idx_lst
                inp_norms[idx_lst] += torch.mean(hidden_states[start_index][0].to(dtype=torch.float32) ** 2, dim=0).to(
                    device) / args.num_examples
                out_norms[idx_lst] += torch.mean(hidden_states[end_index][0].to(dtype=torch.float32) ** 2,
                                       dim=0).to(device) / args.num_examples

        for idx_lst in candidates:
            s = (out_norms[idx_lst] / inp_norms[idx_lst]) ** 0.5
            b = torch.zeros_like(s)

            if args.bf16:
                s = s.to(torch.bfloat16)
                b = b.to(torch.bfloat16)
            norm_ratio[idx_lst] = (s, b)

        model.config.use_cache = use_cache
        if not args.fp16 and not args.bf16:
            model.float()

        if not delect_layers:
            return norm_ratio

        new_layers = layers
        for idx_lst in candidates[::-1]:
            start_index, end_index = idx_lst
            s, b = norm_ratio[idx_lst]
            print(torch.mean(s.abs()), torch.mean(b.abs()))
            new_layers = nn.ModuleList(new_layers[:start_index] +
                                       [IdentityNormLike(s, b)] + new_layers[end_index:])
            del layers[start_index:end_index]
        # model.model.layers = new_layers
        if "Llama" in args.base_model or "llama" in args.base_model:
            model.model.layers = new_layers
        elif "opt" in args.base_model:
            model.model.decoder.layers = new_layers

        return target_idx, hidden_inputs, candidates, norm_ratio

    def prune(self, args):
        model, tokenizer = self.model, self.tokenizer

        device = model.device
        print("loading calibdation data")
        dataloader = get_examples(args.dataset, tokenizer, n_samples=args.num_examples, seq_len=2048)
        print("dataset loading complete")

        if args.pruner == "spsrs":
            pruner = SPSRPearsonSinglePruner(model, tokenizer)
            args.remove_list = pruner.prune(args, dataloader)

        target_idx, hidden_inputs, candidates, norm_ratio = self.select(args, dataloader)
        torch.cuda.empty_cache()

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers

        # 冻结
        for param in model.named_parameters():
            param[1].requires_grad = False

        # 映射候选范围到 IdentityNormLike 实例
        identity_layer_map = {}
        for layer in layers:
            if isinstance(layer, IdentityNormLike):
                layer.s = nn.Parameter(layer.s)
                layer.bias = nn.Parameter(layer.bias)

        dataloader = [(inps, outs) for inps, outs in zip(hidden_inputs.cpu(), dataloader.cpu())]
        total_size = len(dataloader)
        train_size = int(0.9 * total_size)
        train_dataset = Subset(dataloader, range(train_size))
        eval_dataset = Subset(dataloader, range(train_size, total_size))

        data_collator = DataCollator(tokenizer)

        output_dir = f'{args.output_dir}/{args.epochs}'
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            warmup_steps=int(args.epochs * len(train_dataset) * 0.175),
            weight_decay=args.wd,
            logging_dir='./logs',
            eval_strategy='epoch',
            logging_steps=500,
            # lr_scheduler_type="cosine",
            # warmup_ratio=0.175,
            # lr_scheduler_kwargs={"num_cycles": 5},
            learning_rate=args.lr,
            save_strategy="no",
        )

        trainer = SPSRLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        trainer.position_ids = torch.arange(0, 2048).unsqueeze(0).to(device)
        trainer.start_idx = target_idx

        trainer.train()
        print(f"max memory_allocated  {torch.cuda.max_memory_allocated(device) / 1024 ** 2}")

        for param in model.named_parameters():
            param[1].requires_grad = False
        torch.cuda.empty_cache()

        model.half()
        for layer in layers:
            if isinstance(layer, IdentityNormLike):
                if not isinstance(layer.bias, nn.Parameter):
                    layer.bias = nn.Parameter(layer.bias)
                print(torch.mean(layer.s.data.abs()), torch.mean(layer.bias.data.abs()))
                print(layer.bias)
        
        if args.save_path is not None:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)

            # 保存 IdentityNormLike 层的权重和替换关系
            identity_layers_info = []
            
            # 从 layers 中收集所有 IdentityNormLike 实例（按出现顺序）
            identity_instances = []
            for layer in layers:
                if isinstance(layer, IdentityNormLike):
                    identity_instances.append(layer)
            
            # 验证 IdentityNormLike 的数量与候选数量相匹配
            if len(identity_instances) != len(candidates):
                print(f"Warning: IdentityNormLike count ({len(identity_instances)}) "
                    f"!= candidates count ({len(candidates)})")
            
            # 将 IdentityNormLike 实例与候选范围关联
            for idx, (start_index, end_index) in enumerate(sorted(candidates)):
                if idx < len(identity_instances):
                    layer_instance = identity_instances[idx]
                    identity_layers_info.append({
                        'start_index': start_index,
                        'end_index': end_index,
                        's': layer_instance.s.data.cpu() if isinstance(layer_instance.s, nn.Parameter) else layer_instance.s.cpu(),
                        'bias': layer_instance.bias.data.cpu() if isinstance(layer_instance.bias, nn.Parameter) else layer_instance.bias.cpu(),
                        'relu': layer_instance.relu
                    })
                    print(f"Saved IdentityNormLike for range [{start_index}:{end_index}] "
                        f"with s.shape={layer_instance.s.shape}, bias.shape={layer_instance.bias.shape}")
            
            # 计算原始的层数
            num_layers_removed = sum((end - start - 1) for start, end in candidates)
            original_num_layers = len(layers) + num_layers_removed
            
            # ======== 新增：保存完整的模型配置 ========
            import json
            from transformers import AutoConfig
            
            # 1. 获取基础模型的配置
            base_config = model.config
            
            # 2. 创建SPSR特定的配置
            spsr_config = {
                'identity_layers': identity_layers_info,
                'model_type': args.base_model,
                'original_num_layers': original_num_layers,
                'candidates': candidates,  # 保存候选范围信息
                'pruning_ratio': getattr(args, 'pruning_ratio', 0.0),
                'sparsity_pattern': getattr(args, 'sparsity_pattern', 'structured'),
            }
            
            # 3. 保存权重文件（保持原有格式）
            weight_path = os.path.join(args.save_path, "spsr_layers.pth")
            torch.save(spsr_config, weight_path)
            print(f"IdentityNormLike layers saved to {weight_path}")
            
            # 4. 创建并保存config.json文件
            config_path = os.path.join(args.save_path, "config.json")
            
            # 构建完整的配置字典
            full_config = {
                # 基础模型配置
                **base_config.to_dict(),
                
                # SPSR特定配置
                "architectures": ["SPSRLlamaForCausalLM"],
                "auto_map": {
                    "AutoModelForCausalLM": "modeling_spsr_llama.SPSRLlamaForCausalLM"
                },
                "model_type": "spsr_llama",
                "base_model_path": args.base_model,  # 指向基础模型
                "spsr_config_path": os.path.basename(args.save_path),  # SPSR权重文件名
                
                # 元数据
                "spsr_metadata": {
                    "original_model": args.base_model,
                    "total_layers_replaced": len(candidates),
                    "layers_removed": num_layers_removed,
                    "final_layer_count": len(layers),
                    "original_layer_count": original_num_layers,
                    "pruning_ratio": getattr(args, 'final_s', 0.0),
                    "weight_type": "scale",
                },
                
                # vLLM兼容性配置
                "torch_dtype": str(base_config.torch_dtype) if hasattr(base_config, 'torch_dtype') else "float16",
                "use_cache": True,
                "tie_word_embeddings": getattr(base_config, 'tie_word_embeddings', True),
            }
            
            # 5. 保存config.json
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(full_config, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Complete config saved to {config_path}")
            print(f"📊 Model summary:")
            print(f"   - Original layers: {original_num_layers}")
            print(f"   - Final layers: {len(layers)}")
            print(f"   - Layers replaced: {len(candidates)}")
            print(f"   - Layers removed: {num_layers_removed}")
            print(f"   - Config saved: {config_path}")
            
            # 6. 保存tokenizer配置（如果存在）
            try:
                tokenizer.save_pretrained(args.save_path)
                print(f"✅ Tokenizer saved to {args.save_path}")
            except Exception as e:
                print(f"⚠️  Failed to save tokenizer: {e}")

            generation_config = model.generation_config
            generation_config.save_pretrained(args.save_path) 

        # new_layers = []
        # for layer in layers:
        #     if isinstance(layer, IdentityNormLike):
        #         new_layers[-1] = coff_LlamaDecoderLayer(new_layers[-1], layer.s, layer.bias)
        #     else:
        #         new_layers.append(layer)
        #
        # if "Llama" in args.base_model or "llama" in args.base_model:
        #     model.model.layers = torch.nn.ModuleList(new_layers)
        # elif "opt" in args.base_model:
        #     model.model.decoder.layers = torch.nn.ModuleList(new_layers)
        # elif "Qwen" in args.base_model:
        #     model.transformer.h = torch.nn.ModuleList(new_layers)
        # else:
        #     model.model.layers = torch.nn.ModuleList(new_layers)

class coff_LlamaDecoderLayer(nn.Module):
    def __init__(self, original_decoder_layer, s=torch.tensor([1.]), b=torch.tensor([0.])):
        super().__init__()
        self.hidden_size = original_decoder_layer.hidden_size

        self.self_attn = original_decoder_layer.self_attn
        self.mlp = original_decoder_layer.mlp
        self.input_layernorm = original_decoder_layer.input_layernorm
        self.post_attention_layernorm = original_decoder_layer.post_attention_layernorm

        self.s = s.to(device=self.mlp.down_proj.weight.device)
        self.bias = b.to(device=self.mlp.down_proj.weight.device)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual.to(hidden_states.device) + hidden_states

        # mlp
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual.to(hidden_states.device) + hidden_states

        outputs = (hidden_states * self.s + self.bias,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class IdentityNormLike(nn.Module):
    def __init__(self, s, bias, relu=False):
        super().__init__()

        self.s = s
        self.bias = bias
        self.relu = relu

    def forward(self, hidden_states, *args, **kwargs):
        use_cache = kwargs["use_cache"] if "use_cache" in kwargs else False
        output_attentions = kwargs["output_attentions"] if "output_attentions" in kwargs else False
        past_key_value = kwargs["past_key_value"] if "past_key_value" in kwargs else None

        outputs = (
            (hidden_states * self.s.to(device=hidden_states.device) + self.bias.to(device=hidden_states.device)),)

        if self.relu:
            outputs = (F.relu(outputs[0]),)

        if output_attentions:
            outputs += (None,)

        if use_cache:
            outputs += (past_key_value,)
        return outputs


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        labels = torch.cat([example[1].unsqueeze(0) for example in examples], dim=0)
        input_ids = torch.cat([example[0].unsqueeze(0) for example in examples], dim=0)
        output_dict = dict(labels=labels, input_ids=input_ids)
        return output_dict