import gc
import json
import os
import argparse
import fnmatch

import numpy as np
import torch
import random
import time


np.random.seed(0)
torch.manual_seed(0)

from transformers import AutoModelForCausalLM, AutoTokenizer

from pruner import *
from layersp import *
from lm_eval import tasks, evaluator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


@torch.no_grad()
def eval_zero(args, model, tokenizer, task_names):
    if "gsm8k" in task_names:
        from utils import eval_gsm8k_zero_shot
        eval_gsm8k_zero_shot(model, tokenizer)
        task_names.pop(task_names.index("gsm8k"))

    if len(task_names) == 0:
        return

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model_type=args.model_type,
        model=(tokenizer, model),
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
    )

    if results is None:
        return

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        import os
        directory_path = os.path.dirname(args.output_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(
        f"{args.model_type} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    )
    print(evaluator.make_table(results))

    task_acc = results["results"]
    acc_res = []
    if "storycloze" in task_names:
        acc_res.append(task_acc["storycloze"]["acc"] * 100)
        task_names.pop(task_names.index("storycloze"))
    if "rte" in task_names:
        acc_res.append(task_acc["rte"]["acc"] * 100)
        task_names.pop(task_names.index("rte"))
    if len(acc_res) > 0:
        print("|".join(["%.2f" % c for c in acc_res]))
        acc_res = []

    need_task_name = ["winogrande", "hellaswag", "boolq", "piqa", "openbookqa", "arc_easy", "arc_challenge"]
    for task_name in need_task_name:
        if task_name not in task_acc:
            acc_res.append(0)
            continue
        if "acc_norm" in task_acc[task_name] and task_name != "piqa":
            acc_res.append(task_acc[task_name]["acc_norm"] * 100)
        else:
            acc_res.append(task_acc[task_name]["acc"] * 100)
    mean_acc = sum(acc_res) / len(acc_res)
    acc_res.append(mean_acc)
    print("|".join(["%.2f" % c for c in acc_res]))


def make_parser():
    parser = argparse.ArgumentParser("Pruner Processor")

    parser.add_argument("-p", "--pruner", default=None, type=str, help="pruner")
    parser.add_argument("-s", "--final_s", default=0., type=float, help="final sparsity")
    parser.add_argument("-d", "--dataset", default="c4", type=str, help="dataset")

    parser.add_argument('--base_model', type=str, help='model name')
    parser.add_argument('--save_path', type=str, default=None,
                        help='save path')
    parser.add_argument('--load_spsr_path', type=str, default=None,
                        help='path to load SPSR layers checkpoint')
    parser.add_argument('--insert', type=str, default=None,
                        help='path to load IdentityLinearLike layers for insertion')
    parser.add_argument('--load_quant', type=str, default=None, help='path to load quantized layers checkpoint')

    parser.add_argument("--gradient_path", type=str, default=None, help="Path to save the gradient.")

    # llm pruner
    parser.add_argument("-slen", "--seq_len", default=2048, type=int, help="sequence length")
    parser.add_argument("-n_exa", "--num_examples", default=128, type=int, help="prune examples num")
    parser.add_argument("-it", "--iters", default=1, type=int, help="prune iters")
    parser.add_argument("-start", "--start_layer", default=3, type=int, help="layer start to prune")
    parser.add_argument("-end", "--end_layer", default=30, type=int, help="layer end prune")
    parser.add_argument("-m", "--mode", default="none", type=str, help="channel")
    parser.add_argument("-i", "--imp", default="none", type=str, help="imp type")

    # \beta
    parser.add_argument("--Lamda", default=0.5, type=float, help="lambda")

    # remove
    parser.add_argument("--remove_list", nargs='+', default=[], help='remove transformer list')

    # control
    parser.add_argument("--fp16", action='store_true', help='use fp16')
    parser.add_argument("--bf16", action='store_true', help='use bf16')
    parser.add_argument("--block_dense", action='store_true', help='use blockwise dense')
    parser.add_argument("--origin", action='store_true', help='not align dense input to prune')
    parser.add_argument("--dense", action='store_true', help='dense cal')
    parser.add_argument("--use_variant", action='store_true',
                        help='whether to use the wanda variant described in the appendix')
    parser.add_argument("--block", default=0, type=int, help='use block')

    parser.add_argument("-prune_n", "--N", default=0, type=int, help="prune N")
    parser.add_argument("-prune_m", "--M", default=0, type=int, help="prune M")

    ## eval zero
    parser.add_argument('--model_type', type=str, default="hf-causal-experimental", help='model type')
    parser.add_argument("--model_args", default="pretrained=facebook/opt-125m")
    parser.add_argument("--tasks", default=None, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")

    parser.add_argument("--seed", default=0, type=int, help='seed')

    ## osscar
    parser.add_argument(
        '--sp_out',
        type=float, default=-1, help='Sparsity level for output layers'
    )
    parser.add_argument(
        '--parallel', action='store_true',
        help='Whether parallel or sequential.'
    )
    parser.add_argument(
        '--local_out', action='store_true',
        help='Whether perform local search on output layers.'
    )
    parser.add_argument(
        '--local_fc', action='store_true',
        help='Whether perform local search on fc layers'
    )
    parser.add_argument(
        '--local_iter', type=int, default=30,
        help='number of iterations of local search'
    )
    parser.add_argument(
        '--local_test', type=int, default=5,
        help='number of tests of local search'
    )
    parser.add_argument(
        '--update_iter', type=int, default=10,
        help='Support update frequency.'
    )
    parser.add_argument(
        '--update_iter2', type=int, default=2,
        help='Support update frequency.'
    )
    parser.add_argument(
        '--lambda2', type=float, default=1e-2,
        help='Regular term'
    )

    ## non-uniform
    parser.add_argument("--layer", default="uniform", type=str, help="layer sp")
    parser.add_argument("--all_layer_ratio", nargs='+', default=[], help='use layer_ratio')

    ## gptq
    parser.add_argument("--gptq", action='store_true', help='gptq')
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')

    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8],
                        help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')

    ## spsr
    parser.add_argument("--output_dir", default="/media/data/yzg", type=str, help="output_dir")
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument("--lr", default=3e-4, type=float, help="lr")
    parser.add_argument("--cnt", type=int, default=10, help='random try cnt')
    parser.add_argument("--nbias", action='store_true', help='not use bias')

    #linear_patch
    parser.add_argument("--train_size", type=int, default=5000, help="Number of training data samples.")
    parser.add_argument("--val_size", type=int, default=16, help="Number of validation data samples.")
    parser.add_argument('--insert_type', type=str, default='rotate', help='insert type')
    parser.add_argument("--min_lr_factor", type=float, default=20, help="min_lr = lr/min_lr_factor")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--early_stop", type=int, default=0, help="early stoping after validation loss do not decrease")

    parser.add_argument('--load_router', action='store_true', help='load from local')
    parser.add_argument("--router_path", type=str, default="/media/data/yzg/router",
                        help="Output directory for model checkpoints.")
    parser.add_argument("--repochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--rbatch_size", type=int, default=16,
                        help="Batch size.")
    parser.add_argument("--rlr", type=float, default=1e-5,
                        help="Learning rate.")

    # fine-tune
    parser.add_argument('--tune', action='store_true', help='fine tune the pruned model')

    ## onnx
    parser.add_argument('--onnx_export_path', type=str, default=None,
                        help='onnx export path')
    parser.add_argument("--onnx", action='store_true', help='convert to onnx')
    parser.add_argument('--opset', required=False, type=int, default=18)
    parser.add_argument('--add_topk_warper', required=False, type=int, default=0)
    parser.add_argument('--topk', required=False, type=int, default=4)

    return parser


def load_model_tokenizer(args):
    ckpt_dir = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.float32
    if args.fp16:
        dtype = torch.float16
    if args.bf16:
        dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        # quantization_config=bnb_config,  # 上面本地模型的配置
        # device_map="cpu" if args.deepsparse else "auto",  # 使用GPU的编号
        device_map="auto",
        torch_dtype=dtype,
    )
    return tokenizer, model


def load_spsr_layers(model, load_path, model_type="llama"):
    """
    Load and apply IdentityNormLike layers to the model based on saved checkpoint.
    This handles layer range replacement (from start_index to end_index).
    
    Args:
        model: Pre-trained model to apply layers to
        load_path: Path to saved checkpoint containing identity layer information
        model_type: Type of model ("llama", "opt", "qwen", etc.)
    """
    from torch import nn
    
    # 获取设备
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # 加载保存的数据
    load_path = os.path.join(load_path, "spsr_layers.pth")
    save_data = torch.load(load_path, map_location=device)
    identity_layers_info = save_data['identity_layers']
    
    # 获取层列表
    if "Llama" in model_type or "llama" in model_type:
        layers = model.model.layers
    elif "opt" in model_type:
        layers = model.model.decoder.layers
    elif "Qwen" in model_type:
        layers = model.transformer.h
    else:
        layers = model.model.layers
    
    # 导入 IdentityNormLike 类
    from pruner import IdentityNormLike
    
    # 从后往前替换（与保存时的逻辑一致），避免索引混乱
    for layer_info in reversed(identity_layers_info):
        start_index = layer_info['start_index']
        end_index = layer_info['end_index']
        s = nn.Parameter(layer_info['s'].to(device, dtype))
        bias = nn.Parameter(layer_info['bias'].to(device, dtype))
        relu = layer_info['relu']
        
        # 创建 IdentityNormLike 实例
        identity_layer = IdentityNormLike(s, bias, relu=relu)
        
        # 删除原始层范围内的层并插入 IdentityNormLike
        new_layers_list = list(layers[:start_index]) + [identity_layer] + list(layers[end_index:])
        layers = nn.ModuleList(new_layers_list)
        
        print(f"Replaced layers [{start_index}:{end_index}] with IdentityNormLike")
    
    # 更新模型的层
    if "Llama" in model_type or "llama" in model_type:
        model.model.layers = layers
    elif "opt" in model_type:
        model.model.decoder.layers = layers
    elif "Qwen" in model_type:
        model.transformer.h = layers
    else:
        model.model.layers = layers
    
    # 清除GPU缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Successfully loaded IdentityNormLike layers from {load_path}")
    return model


def load_identity_linear_like_layers(model, load_path, model_type="llama"):
    """
    Load and insert IdentityLinearLike layers into the model.
    
    Args:
        model: Pre-trained model to apply layers to
        load_path: Path to saved checkpoint containing identity linear layer information
        model_type: Type of model ("llama", "opt", "qwen", etc.)
    """
    from torch import nn
    from fine_tune import IdentityLinearLike
    
    # 获取设备
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # 加载保存的数据
    save_data = torch.load(load_path, map_location=device)
    identity_linear_layers = save_data['identity_linear_layers']
    
    # 获取层列表
    if "Llama" in model_type or "llama" in model_type:
        layers = model.model.layers
    elif "opt" in model_type:
        layers = model.model.decoder.layers
    elif "Qwen" in model_type:
        layers = model.transformer.h
    else:
        layers = model.model.layers
    
    # 插入IdentityLinearLike层
    new_layers = list(layers)
    for layer_info in identity_linear_layers:
        index = layer_info['index']
        s = nn.Parameter(layer_info['s'].to(device, dtype))
        bias = nn.Parameter(layer_info['bias'].to(device, dtype))
        
        # 创建IdentityLinearLike实例
        identity_linear_layer = IdentityLinearLike(s, bias)
        
        # 插入到指定位置
        new_layers.insert(index, identity_linear_layer)
        
        print(f"Inserted IdentityLinearLike at index {index}")
    
    # 更新模型的层
    if "Llama" in model_type or "llama" in model_type:
        model.model.layers = nn.ModuleList(new_layers)
    elif "opt" in model_type:
        model.model.decoder.layers = nn.ModuleList(new_layers)
    elif "Qwen" in model_type:
        model.transformer.h = nn.ModuleList(new_layers)
    else:
        model.model.layers = nn.ModuleList(new_layers)
    
    # 更新config中的层数
    cfg = model.config
    if "Llama" in model_type or "llama" in model_type:
        num_layers_attr = "num_hidden_layers"
    elif "opt" in model_type:
        num_layers_attr = "num_hidden_layers"
    elif "Qwen" in model_type:
        num_layers_attr = "n_layer"
    else:
        num_layers_attr = "num_hidden_layers"
    
    if hasattr(cfg, num_layers_attr):
        setattr(cfg, num_layers_attr, len(new_layers))
    
    # 清除GPU缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Successfully loaded IdentityLinearLike layers from {load_path}")
    return model


def remove_layers_and_update_config(model, remove_indices, model_type="llama"):
    # remove_indices: 可迭代的要删除的层索引（int）
    remove_indices = [int(i) for i in remove_indices]

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
    for idx in sorted(remove_indices)[::-1]:
        del layers[idx]

    model.model.layers = layers
    gc.collect()

    # 更新 config 中记录的层数
    new_n = len(layers)
    if hasattr(cfg, num_layers_attr):
        setattr(cfg, num_layers_attr, new_n)
    else:
        # 如果找不到常见字段，打印提醒，并可选择设置 config.layers 或其它自定义
        print(f"Warning: config has no attribute {num_layers_attr}; config keys: {list(cfg.__dict__.keys())}")

    return layers


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()

    tokenizer, model = load_model_tokenizer(args)

    # 如果指定了加载SPSR层的路径，则加载并替换
    if args.load_spsr_path is not None:
        model = load_spsr_layers(model, args.load_spsr_path, model_type=args.base_model)
        torch.cuda.empty_cache()
        # 如果仅指定加载，则在加载后返回，不进行其他操作
        if args.pruner is None and len(args.remove_list) == 0:
            print(f"Model with loaded SPSR layers ready for evaluation or fine-tuning")
            # 可继续进行后续的gptq, tune, eval等操作

    # 如果指定了插入IdentityLinearLike层的路径，则加载并插入
    if args.insert is not None:
        model = load_identity_linear_like_layers(model, args.insert, model_type=args.base_model)
        torch.cuda.empty_cache()
        print(f"Model with inserted IdentityLinearLike layers ready for evaluation or fine-tuning")

    layer_sp_dic = {"uniform": Uniform, "owl": OWL, "dlp": DLP, "atp": ATP}

    pruner_dic = {
        # structured
        "llm": LLMPruner, "osscar": OSSCARPruner,
        # unstructured
        "wanda": WandaPruner,
        "sgpt": SparseGPTPruner, 
        "prunerzero": PrunerZeroPruner, "alps": ALPSPruner,
        "mag": MagPruner,
        "ria": RIAPruner,
        # layer drop
        "short": ShortGptPruner,
        "laco": LacoPruner,
        "cl": CLPruner, 
        "reme": ReplaceMePruner,
        "spsrcl": SPSRCIPruner, "spsrp": SPSRPlusPruner,
        "spsrs": SPSRPlusPruner,
        "stream": StreamLinePruner,
        "patch": LinearPatchPruner, "patchp": LinearPatchPlusPruner,
        "sleb": SLEBPruner, "slebo": SLEBOneShotPruner,
        "block": BlockPruner,
        "taylor": TaylorPruner, "taylori": TaylorIterPruner,
    }
    random.seed(args.seed)
    if "Qwen3" or "Qwen2.5" or "Qwen1.5" in args.base_model:
        args.base_model = args.base_model.replace("Qwen", "LlamaQwen")
    if args.pruner is not None:
        if len(args.all_layer_ratio) == 0:
            layer_sp = layer_sp_dic[args.layer](model, tokenizer)
            args.all_layer_ratio = layer_sp.get_layer_sp(args)
        else:
            args.all_layer_ratio = [float(ratio) for ratio in args.all_layer_ratio]
            print(args.all_layer_ratio)
        args.remove_list = [int(i) for i in args.remove_list]

        start_time = time.time()
        pruner = pruner_dic[args.pruner](model, tokenizer)
        pruner.prune(args)
        prune_time = time.time() - start_time
        print("overall time cost: %.5f sec" % prune_time)
        del pruner
    elif len(args.remove_list) > 0:
        remove_layers_and_update_config(model, args.remove_list, model_type=args.base_model)
    torch.cuda.empty_cache()

    if args.gptq:
        from gptq import llama_gptq
        model = llama_gptq(model, tokenizer, dev=model.device, args=args)

    if args.tune:
        model.half()
        for param in model.named_parameters():
            param[1].requires_grad = False

        from fine_tune import finetune
        model = finetune(model, tokenizer, parser)
        model.half()

    if args.tasks is not None:
        from utils import eval_ppl

        eval_tasks = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)
        print(f"Selected Tasks: {eval_tasks}")

        if "wikitext" in eval_tasks:
            eval_tasks.pop(eval_tasks.index("wikitext"))
            eval_ppl(model, tokenizer, "wikitext")
        if "ptb" in eval_tasks:
            eval_tasks.pop(eval_tasks.index("ptb"))
            eval_ppl(model, tokenizer, "ptb")
        if "c4" in eval_tasks:
            eval_tasks.pop(eval_tasks.index("c4"))
            eval_ppl(model, tokenizer, "c4")
        if len(eval_tasks) > 0:
            eval_zero(args, model, tokenizer, eval_tasks)

    if args.onnx:
        from utils import export_llama_to_onnx, export_qwen_to_onnx
        model = model.cpu()
        model.float()
        if "Qwen" in args.base_model:
            onnx_file_name = export_qwen_to_onnx(model, model.config, torch.float32, args)
        else:
            onnx_file_name = export_llama_to_onnx(model, model.config, torch.float32, args)
