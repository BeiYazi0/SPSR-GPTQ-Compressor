import copy
import gc
import random
import types
from itertools import chain

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Subset, Dataset
from tqdm import tqdm

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets, load_from_disk

from torch.utils.data import DataLoader
from accelerate import Accelerator

from datas import get_examples

import math
from functools import partial
from typing import Optional, Tuple

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


class CLPruner:
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

        # 删除层
        new_layers = nn.ModuleList(layers[:start_index] + layers[start_index + pruning_num:])
        model.model.layers = new_layers
        del layers[start_index:start_index + pruning_num]

        after_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("After prune, #parameters: {}".format(after_pruning_parameters))

        model.config.use_cache = use_cache
        if args.save_path is not None:
            self.tokenizer.save_pretrained(args.save_path)
            self.model.save_pretrained(args.save_path)


class StreamLinePruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def select(self, args, dataloader):
        model, tokenizer = self.model, self.tokenizer
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

        pruning_num = int(n * args.final_s) + 1  # 多移除一层，保留被移除层的首层用于训练
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

        start_index = torch.argmin(score).item()
        pruning_layers = list(range(start_index, start_index + pruning_num))
        print(pruning_layers)

        model.config.use_cache = use_cache
        return start_index

    def prune(self, args):
        model, tokenizer = self.model, self.tokenizer

        device = model.device
        print("loading calibdation data")
        dataloader = get_examples("c4", tokenizer, n_samples=args.num_examples, seq_len=2048)
        print("dataset loading complete")

        if len(args.remove_list) == 0:
            best_layer = self.select(args, dataloader)
        else:
            best_layer = args.remove_list[0]
        torch.cuda.empty_cache()

        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = model.transformer.h
        else:
            layers = model.model.layers

        layer_intervals = int(len(layers) * args.final_s) + 1
        torch.cuda.empty_cache()

        total_size = len(dataloader)
        train_size = int(0.9 * total_size)
        train_dataset = Subset(dataloader, range(train_size))
        eval_dataset = Subset(dataloader, range(train_size, total_size))

        # c4 v
        if not args.origin:
            replace_model = lightweight_model_train(model, tokenizer, (train_dataset, eval_dataset), device,
                                                    layer_intervals, best_layer,
                                                    train_num_data=117, batch_size=1, epochs=args.epochs, lr=args.lr,
                                                    min_lr=5e-5, wd=0.01,
                                                    gradient_accumulation_step=16)
        else:
            # origin
            replace_model = lightweight_model_train(model, tokenizer, None, device,
                                                    layer_intervals, best_layer,
                                                    train_num_data=10000, batch_size=8, epochs=20, lr=2e-4,
                                                    min_lr=5e-5, wd=1e-3,
                                                    gradient_accumulation_step=16)

        new_layers = nn.ModuleList(layers[:best_layer] + [replace_model] + layers[best_layer + layer_intervals:])
        if "Llama" in args.base_model or "llama" in args.base_model:
            model.model.layers = new_layers
        elif "opt" in args.base_model:
            model.model.decoder.layers = new_layers
        print(f"max memory_allocated  {torch.cuda.max_memory_allocated(device) / 1024 ** 2}")

        if args.save_path is not None:
            config = model.config
            setattr(config, "num_hidden_layers", len(new_layers))
            self.tokenizer.save_pretrained(args.save_path)
            self.model.save_pretrained(args.save_path, max_shard_size="10GB")


@torch.no_grad()
def get_data(model, dataset, device, layer_intervals, best_layer, tokenizer, batch_size):
    input_list = []
    output_list = []

    accelerator = Accelerator()
    # device = accelerator.device
    #
    # model = model.to(device)
    model.eval()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size
    )
    dataloader = accelerator.prepare(dataloader)

    try:
        for batch in tqdm(dataloader, total=len(dataloader)):
            hidden_states = model(
                input_ids=batch['input_ids'].to(device=model.device),
                # attention_mask=batch['attention_mask'],
                output_hidden_states=True
            ).hidden_states

            input_tensor = hidden_states[best_layer].to(torch.float16).cpu()
            output_tensor = hidden_states[best_layer + layer_intervals].to(torch.float16).cpu()

            input_list += torch.unbind(input_tensor, dim=0)
            output_list += torch.unbind(output_tensor, dim=0)

            del hidden_states

    finally:
        accelerator.free_memory()
        torch.cuda.empty_cache()
        # model.cpu()
        # del model
        gc.collect()

    return input_list, output_list


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        max_learning_rate: float,
        min_learning_rate: float,
        num_cycles: float = 0.5,
        last_epoch: int = -1
):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_schedule_with_warmup_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float,
        max_learning_rate: float,
        min_learning_rate: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    _lambda = max(0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return (min_learning_rate + _lambda * (max_learning_rate - min_learning_rate)) / max_learning_rate


def lightweight_model_train(model, tokenizer, dataloader, device, layer_intervals, best_layer,
                            train_num_data, batch_size, epochs, lr, min_lr, wd,
                            gradient_accumulation_step, replace_model=None):
    if dataloader is None:
        dataset = load_dataset('DKYoon/SlimPajama-6B')['train']
        dataset, test_dataset = process_datasets(dataset, train_num_data, tokenizer)
    else:
        dataset, test_dataset = dataloader

    def prepare_dataset_for_training(dataset, model, device):
        input_list, output_list = get_data(model, dataset, device, layer_intervals, best_layer, tokenizer, batch_size)
        return CustomDataset(input_list, output_list)

    test_dataset = prepare_dataset_for_training(test_dataset, model, device)
    train_dataset = prepare_dataset_for_training(dataset, model, device)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=0)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=0)

    # 冻结
    for param in model.named_parameters():
        param[1].requires_grad = False

    if replace_model is None:
        if hasattr(model.model, "layers"):
            replace_model = model.model.layers[best_layer]
        else:
            replace_model = model.model.decoder.layers[best_layer]
    replace_model.float()
    for param in replace_model.named_parameters():
        param[1].requires_grad = True

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(replace_model.parameters(), lr=lr, weight_decay=wd)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_dataloader) * epochs * 0.01 * 0.5,
        num_training_steps=len(train_dataloader) * epochs * 0.5,
        max_learning_rate=lr,
        min_learning_rate=min_lr,
    )

    best_loss = valid_model(replace_model, test_dataloader, device)
    print("Before training, Validation_Loss:", best_loss)
    print("Starting training...")
    best_state_dict = None

    for epoch in range(epochs):
        replace_model.train()
        step = 0
        optimizer.zero_grad()

        for input_data, output_data in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            input_data = input_data.to(device).to(torch.float32)
            output_data = output_data.to(device).to(torch.float32)
            position_ids = torch.arange(0, 2048).repeat(input_data.shape[0], 1).to(device)

            output = replace_model(hidden_states=input_data, position_ids=position_ids)
            output = output[0] if isinstance(output, tuple) else output

            loss = criterion(output.to(device), output_data)
            loss /= gradient_accumulation_step
            loss.backward()

            if (step + 1) % gradient_accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(replace_model.parameters(), max_norm=5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            del input_data, output_data, output, loss
            torch.cuda.empty_cache()

            step += 1

        valid_loss = valid_model(replace_model, test_dataloader, device)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state_dict = replace_model.state_dict()

        print(f"Epoch: {epoch}, Validation Loss: {valid_loss:.6f}")

        torch.cuda.empty_cache()
        gc.collect()

    replace_model.load_state_dict(best_state_dict)

    for param in replace_model.named_parameters():
        param[1].requires_grad = False
    replace_model.half()

    return replace_model


def process_datasets(dataset, train_num_data, tokenizer):
    '''
    We divided the proportions of RedPajamaCommonCrawl, RedPajamaArXiv,
    and RedPajamaBook by a normalization value because the data length
    in these domains is higher than in other domains.
    '''
    proportions = {
        "RedPajamaC4": 0.492,
        "RedPajamaStackExchange": 0.01,
        "RedPajamaCommonCrawl": 0.361 / 3,
        "RedPajamaGithub": 0.008,
        "RedPajamaWikipedia": 0.031,
        "RedPajamaArXiv": 0.007 / 20,
        "RedPajamaBook": 0.091 / 200
    }

    filtered_datasets = {
        name: dataset.filter(lambda x: x['meta'] == {"redpajama_set_name": f"{name}"})
        for name in proportions.keys()
    }

    test_datasets = []
    train_datasets = []

    for name, proportion in proportions.items():
        split = filtered_datasets[name].train_test_split(test_size=(3000 * proportion) / len(filtered_datasets[name]))
        test_datasets.append(split['test'])
        train_split = \
            split['train'].train_test_split(test_size=1 - (train_num_data * proportion) / len(split['train']))['train']
        train_datasets.append(train_split)

    dataset, test_dataset = concatenate_datasets(train_datasets), concatenate_datasets(test_datasets)

    tokenizer.pad_token = tokenizer.eos_token

    column_names = dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    block_size = 2048

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    test_dataset = test_dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return dataset, test_dataset


def valid_model(model, test_dataloader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = []

    with torch.no_grad():
        for input_data, output_data in tqdm(test_dataloader):
            input_data = input_data.to(device).to(torch.float32)
            output_data = output_data.to(device).to(torch.float32)
            position_ids = torch.arange(0, 2048).repeat(input_data.shape[0], 1).to(device)
            pred = model(hidden_states=input_data, position_ids=position_ids)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = loss_fn(pred.to(device), output_data)
            total_loss.append(loss.item())

    return sum(total_loss) / len(total_loss)


class CustomDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def __getitem__(self, index):
        return self.input_data[index].to(dtype=torch.float32), self.output_data[index].to(dtype=torch.float32)

    def __len__(self):
        return len(self.input_data)
