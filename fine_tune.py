import json
import math
import os

os.environ["WANDB_API_KEY"] = "4d02f488c1172d6526b17a48144ae7754db9e4dc"

import torch
import wandb
import argparse
from torch import nn

from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, PeftModel

from utils import Prompter, ZeroPrompter, create_llama_groups

wandb.login()

wandb.init(project="llama3")


# def load_model_tokenizer(ckpt_dir, config_dir):
#     tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.pad_token_id = tokenizer.eos_token_id
#
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,  # 在4bit上，进行量化
#         bnb_4bit_use_double_quant=True,  # 嵌套量化，每个参数可以多节省0.4位
#         bnb_4bit_quant_type="nf4",  # NF4（normalized float）或纯FP4量化 博客说推荐NF4
#         bnb_4bit_compute_dtype=torch.float16
#     )
#
#     model = AutoModelForCausalLM.from_pretrained(
#         ckpt_dir,
#         # quantization_config=bnb_config,  # 上面本地模型的配置
#         device_map="auto",  # 使用GPU的编号
#         torch_dtype=torch.float32
#     )
#
#     return tokenizer, model


def train(args, model, tokenizer):
    if not args.no_instruction:
        prompter = Prompter(args.prompt_template_name)
    else:
        prompter = ZeroPrompter()

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < args.cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if 'lamini' in args.data_path.lower():
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                None,
                data_point["response"],
            )
        elif 'alpaca' in args.data_path.lower():
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )
        elif "ptb" in args.dataset[0].lower():
            full_prompt = data_point["sentence"]
        else:
            raise NotImplementedError

        tokenized_full_prompt = tokenize(full_prompt)
        if not args.train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"] if 'input' in data_point.keys() else None,
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=args.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if args.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    max_seq_length = 512
    # tokenizer, model = load_model_tokenizer(args.base_model, args.prune_config)

    dataset = load_dataset(args.data_path)
    train_val = dataset["train"].train_test_split(test_size=args.val_set_size, shuffle=True, seed=42)
    train_data = (
        train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    )
    val_data = {
        args.data_path: train_val["test"].shuffle().map(generate_and_tokenize_prompt),
    }

    # test_data = dataset["test"]["sentence"]
    # calculate_perplexity(model, tokenizer, test_data)

    gradient_accumulation_steps = args.tune_batch_size // args.micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    training_args = TrainingArguments(
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        logging_first_step=True,
        optim="adamw_torch",
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=200,
        output_dir=args.output_dir,
        save_total_limit=20,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=None,
        group_by_length=args.group_by_length,
        report_to="wandb",
        run_name=args.output_dir.split('/')[-1],
        metric_for_best_model="{}_loss".format(args.data_path),
    )

    # 配置QLora
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    new_layers = []
    from pruner import IdentityNormLike
    for layer in model.model.model.layers:
        if isinstance(layer, IdentityNormLike):
            replace_layer = IdentityLinearLike(None, None)

            replace_layer.s = torch.nn.Parameter(torch.diag(layer.s.to(torch.float32)))
            replace_layer.bias = torch.nn.Parameter(layer.bias.to(torch.float32))

            replace_layer.s.requires_grad_(True)
            replace_layer.bias.requires_grad_(True)
            new_layers.append(replace_layer)
        else:
            new_layers.append(layer)
    model.model.model.layers = nn.ModuleList(new_layers)

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # calculate_perplexity(model, tokenizer, test_data)

    # model = model.merge_and_unload()  # 不合并 LoRA
    # model.save_pretrained(args.output_dir)
    return model


def save_finetuned_model_with_identity_linear_like(model, tokenizer, save_path, load_spsr_path, model_type="llama"):
    """
    保存微调后的模型，将IdentityLinearLike的权重替换到spsr_layers.pth中，修改config的weight_type为linear，并保存LoRA适配器。
    
    Args:
        model: 微调后的模型（包含LoRA）
        tokenizer: 分词器
        save_path: 保存路径
        load_spsr_path: 原始spsr路径
        model_type: 模型类型
    """

    import os
    import json
    from torch import nn
    
    # 载入 spsr_layers.pth
    spsr_path = os.path.join(load_spsr_path, "spsr_layers.pth")
    save_data = torch.load(spsr_path, map_location='cpu')
    identity_layers_info = save_data['identity_layers']
    
    # 获取模型中的 IdentityLinearLike
    if "Llama" in model_type or "llama" in model_type:
        layers = model.model.model.layers
    elif "opt" in model_type:
        layers = model.model.model.decoder.layers
    elif "Qwen" in model_type:
        layers = model.model.transformer.h
    else:
        layers = model.model.model.layers
    
    # 假设一一对应，替换 s 和 b
    idx = 0
    for layer in layers:
        if isinstance(layer, IdentityLinearLike):
            identity_layers_info[idx]['s'] = layer.s.data.cpu()
            identity_layers_info[idx]['bias'] = layer.bias.data.cpu()
            idx += 1
    
    # 保存修改后的 spsr_layers.pth
    os.makedirs(save_path, exist_ok=True)
    torch.save({'identity_layers': identity_layers_info}, os.path.join(save_path, "spsr_layers.pth"))
    
    # 载入 config.json
    config_path = os.path.join(load_spsr_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if 'spsr_metadata' in config:
        config['spsr_metadata']['weight_type'] = 'linear'
    
    # 保存 config
    with open(os.path.join(save_path, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 保存 LoRA 适配器（不合并）
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # 保存 generation_config.json
    generation_config = model.generation_config
    generation_config.save_pretrained(save_path)    
    print(f"Saved to {save_path}")


class IdentityLinearLike(nn.Module):
    def __init__(self, s, bias):
        super().__init__()

        self.s = s
        self.bias = bias

    def forward(self, hidden_states, *args, **kwargs):
        use_cache = kwargs["use_cache"] if "use_cache" in kwargs else False
        output_attentions = kwargs["output_attentions"] if "output_attentions" in kwargs else False
        past_key_value = kwargs["past_key_value"] if "past_key_value" in kwargs else None

        outputs = (
            (hidden_states @ self.s.to(device=hidden_states.device) + self.bias.to(device=hidden_states.device)),)


        if output_attentions:
            outputs += (None,)

        if use_cache:
            outputs += (past_key_value,)
        return outputs


def finetune(model, tokenizer, parser):
    # parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    # parser.add_argument('--base_model', type=str, help='model name')
    parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned", help='data path')
    parser.add_argument('--cache_dataset', action="store_true", default=False)
    parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
    # parser.add_argument('--output_dir', type=str, default="./lora-alpaca", help='output directory')
    # parser.add_argument('--load_spsr_path', type=str, default=None, help='path to load SPSR layers checkpoint')

    # Training Hyperparameters
    parser.add_argument('--tune_batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='validation set size')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca",
                        help="The prompt template to use, will default to alpaca.")
    parser.add_argument('--no_instruction', action='store_true', default=False,
                        help="Whether to use the instruction template or not.")

    # Lora Configuration
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", help='lora target modules')

    # llm hyperparameters
    parser.add_argument('--train_on_inputs', default=False, action="store_true",
                        help='Train on inputs. If False, masks out inputs in loss')
    # parser.add_argument('--no_prune', default=False, action="store_true",
    #                     help='tune with no prune')
    parser.add_argument('--add_eos_token', default=False, action="store_true")
    parser.add_argument('--group_by_length', default=False, action="store_true",
                        help="faster, but produces an odd training loss curve")

    # wandb params
    parser.add_argument('--wandb_project', type=str, default="llama_tune")
    parser.add_argument('--resume_from_checkpoint', type=str, help="either training checkpoint or final adapter")

    # ddp
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version

    if "Qwen3" in args.base_model or "Qwen2.5" in args.base_model or "Qwen1.5" in args.base_model:
        args.base_model = args.base_model.replace("Qwen", "LlamaQwen")

    model = train(args, model, tokenizer)
    
    # 保存微调后的模型
    if args.save_path is not None:
        save_finetuned_model_with_identity_linear_like(model, tokenizer, args.save_path, args.load_spsr_path, model_type=args.base_model if hasattr(args, 'base_model') else "llama")
    
    model = model.merge_and_unload() # 保存好LoRA适配器后再合并，节省存储空间
    return model
