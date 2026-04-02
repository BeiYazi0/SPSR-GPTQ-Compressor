# SPSR-GPTQ-Compressor
## Introduction

本仓库旨在利用层剪枝技术对 llama 和 Qwen 模型进行剪枝，并使用 GPTQ 进行量化。

高效层剪枝SPSR通过最小皮尔逊相关系数估测层的恢复能力，用 scale + bias 架构替换恢复能力强的层，并采用next-token prediction loss训练 scale/bias vectors，仅需 128 samples 和数分钟即可达到较好的性能。

 [SPSR: Achieving Superior Large Language Model Layer Pruning Performance by Super-fast Recovery](https://openreview.net/forum?id=NlkPKk37bT)

LoRA 微调
nvidia供了一份关于使用剪枝和蒸馏技术将Llama 3.1 8B和Mistral NeMo 12B 模型分别压缩为Llama-3.1-Minitron-4B和MN-Minitron-8B的全面报告（LLM Pruning and Distillation in Practice: The Minitron Approach. 2024），
指出通过修剪带来的损失是可以通过大量数据的训练来恢复的。
 
稀疏分配技术LSA通过最小线性重建误差估测层的冗余性，将该技术应用到混合精度量化中，对高冗余的层采用低精度量化。

 [LSA: Layer-wise Sparsity Allocation for Large Language Model Pruning Based on Minimal Linear Reconstruction Error](https://openreview.net/forum?id=xq3lza5IjN).

量化技术GPTQ

推理引擎vllm

## Requirements
- Python 3.8 or higher
- transformers==4.4.51.0
- torchao==0.14.1
- deepsparse==1.8.0
- torch==2.6.0+cu124
- onnx==1.16.0
- onnxruntime==1.16.0
- onnx-ir==0.1.6
- onnxscript==0.3.2
- onnx-graphsurgeon==0.5.2
- protobuf==6.32.0

## Usage
### SPSR 层剪枝

下面是一个使用 SPSR 进行剪枝的示例，移除模型Qwen3-8B的8层，并测试PPL和zero-shot：

```shell
CUDA_VISIBLE_DEVICES=0 python main.py \
--base_model Qwen/Qwen3-8B \
-p spsrs -s 0.23 \
--num_examples 128 \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag,gsm8k \
--epochs 10 --lr 3e-2 \
--bf16 \
--save_path ./checkpoints/Qwen3-8B-spsr-8
```

在七个zero-shot 任务上与其它层剪枝组件替换方法的比较

| 方法 | LLaMA 2-7B | LLaMA 3-8B | Qwen 2.5-7B | Qwen 3-8B |
| :--- | :---: | :---: | :---: | :---: |
| Dense | 58.96 | 69.87 | 70.20 | 69.28 |
| ShortGPT | 51.41 | 39.92 | 51.58 | 48.80 |
| CL | 51.81 | 39.92 | 52.03 | 49.04 |
| ReplaceMe | 53.53 | 56.37 | 54.75 | 50.65 |
| Linear-Patch | 54.44 | 60.52 | 55.00 | 54.80 |
| LLM-Streamline | 54.42 | 61.83 | 55.15 | 50.49 |
| SPSR(CL) | 53.90 | 58.29 | 56.22 | 54.23 |
| SPSR | 54.16 | 60.84 | 56.34 | 55.42 |

相比于 [LLM-Streamline](./pruner/cl.py)、[Linear-Patch](./pruner/patch.py)、[ReplaceMe](./pruner/replaceme.py) 等基于连续层余弦相似度的层剪枝方法，SPSR 的替换组件更加轻量，仅需少量样本即可快速恢复模型性能。

此外，SPSR 不要求移除的层必须连续，因此可以与其他层剪枝方法结合使用。

SLEB 移除的层使用 SPSR vectors 替换。

```shell
python main.py \
--base_model Qwen/Qwen3-8B \
-p spsrp -s 0.23 \
--num_examples 128 \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag \
--epochs 10 --lr 3e-2 \
--remove_list 16 20 2 15 21 32 26 14 \
--bf16
```   

| 模型 | 方法 | WikiText | PTB | C4 | WinoGrande | HellaSwag | BoolQ | PIQA | OBQA | ARC-e | ARC-c | 平均 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LLaMA2-7B | SLEB | 10.00 | 44.92 | 12.59 | 55.01 | 57.55 | 60.89 | 70.13 | 36.60 | 43.43 | 32.68 | 50.90 |
| | +SPSR | 8.93 | 38.52 | 11.47 | 55.80 | 60.51 | 61.74 | 72.14 | 36.80 | 45.79 | 32.34 | 52.16 |
| | Taylor+ | 19.28 | 71.85 | 21.60 | 58.01 | 57.23 | 62.78 | 69.48 | 35.80 | 42.26 | 32.51 | 51.15 |
| | +SPSR | 10.90 | 37.61 | 11.89 | 58.72 | 61.29 | 65.81 | 72.36 | 36.40 | 44.15 | 33.28 | 53.14 |
| LLaMA3-8B | SLEB | 15.27 | 22.02 | 21.20 | 56.75 | 60.31 | 46.94 | 71.06 | 33.20 | 57.87 | 34.64 | 51.54 |
| | +SPSR | 11.74 | 17.35 | 17.59 | 58.96 | 63.97 | 63.27 | 73.78 | 37.00 | 65.99 | 37.97 | 57.28 |
| | Taylor+ | 561.52 | 656.17 | 824.25 | 56.12 | 25.74 | 62.17 | 55.71 | 32.60 | 33.71 | 28.84 | 42.13 |
| | +SPSR | 16.17 | 29.90 | 20.42 | 64.64 | 66.29 | 66.54 | 74.10 | 37.60 | 64.94 | 39.16 | 59.04 |
 
### vllm 部署

vllm 是一个用于部署大语言模型的推理引擎，支持量化、稀疏、多卡推理等特性。

SPSR 模型的部署详情参见 [VLLM-server](./VLLM-server/README.md)。

### LoRA 微调

英伟达发布了一份详尽的报告[1]，详细介绍了如何运用剪枝和蒸馏技术来将 Llama 3.1 8B 和 Mistral NeMo-12B 分别压缩为 Llama-3.1-Minitron-4B 和 MN-Minitron-8B。该报告表明，再剪枝后通过大量训练，剪枝导致的性能下降是可以恢复的。实验结果表明，MN-Minitron-8B 的性能与原始的 Mistral NeMo 12B 相当，甚至在使用 40 倍更少的训练令牌（380 亿 vs. 15 万亿）的情况下还超过了 Llama 3.1 8B。同样，Llama-3.1-Minitron-4B 不仅超过了其教师 Llama 3.1 8B 的性能，而且与上一代的 Minitron 4B 相比，仅使用了 150 倍更少的训练令牌（94 亿 vs. 15 万亿）就表现更优。

[1] LLM Pruning and Distillation in Practice: The Minitron Approach. 2024.

尽管在有限的计算资源下，使用数十B的token进行蒸馏依然不可行，但使用 alpaca-cleaned 数据集进行 LoRA 进行微调，可以显著提高模型性能。

alpaca-cleaned 是斯坦福大学发布的原始 Alpaca 数据集的清洗版本。识别并修复了原始发布中的以下问题：
- 虚构内容（Hallucinations）: 原始数据集中许多指令引用了互联网上的数据，导致 GPT3 虚构答案。
- 合并指令: 原始数据集中有多个指令因某种原因被合并。
- 空输出
- 缺少代码示例
- 生成图像的指令: 原始数据集中一些描述包含了生成图像的指令，这显然是不可能实现的。
- N/A 输出
- 不一致的输入字段: 原始数据集中，在应为空时对输入字段的使用不一致。<no input> vs. No input.
- 错误答案: 大约 80% 的数学问题估计有错误答案。
- 无意义/不清楚的指令: 许多指令不清晰，我们尝试澄清（或重写）非逻辑性的指令。略显模糊但能推断其意的指令则未作改动。
- 多余的转义和控制字符: 原始数据集中有多条包含多余转义和控制字符的记录。

下面是一个使用 SPSR 进行剪枝后微调的示例：

```shell
python main.py \
--base_model Qwen/Qwen3-8B \
--load_spsr_path ./checkpoints/Qwen3-8B-spsr-8 \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag,gsm8k \
--fp16 \
--tune \
--save_path ./checkpoints/Qwen3-8B-spsrs-8-tune
```

在数据集 yahma/alpaca-cleaned 上训练2个 epoch后，zero-shot表现如下。

| model           | Acc   |
|-----------------|-------|
| **Qwen2.5-7B**  | 70.20 |
| Streamline+LoRA | 58.78 |
| SPSR+LoRA       | 59.74 |
| **Qwen3-8B**    | 69.28 |
| Streamline+LoRA | 61.63 |
| SPSR+LoRA       | 61.57 |
| **LLaMA2-7B**   | 58.96 |
| Streamline+LoRA | 57.19 |
| SPSR+LoRA       | 57.51 |
| **LLaMA3-8B**   | 69.87 |
| Streamline+LoRA | 65.83 |
| SPSR+LoRA       | 65.37 |

在GSM8K上的zero-shot结果如下。

| model           | GSM8K |
|-----------------|-------|
| **Qwen2.5-7B**  | 35.48 |
| SPSR            | 15.77 |
| SPSR+LoRA       | 31.99 |
| **Qwen3-8B**    | 51.40 |
| SPSR            | 6.90  |
| SPSR+LoRA       | 47.69 |
| **Qwen3-4B**    | 49.73 |

### 量化

保存后，内存占用减少了。推理速度很慢，只有短序列（<128）调用 CUDA kernel，考虑更换为 triton 实现。

模型量化示例：

```shell
python main.py \
--base_model Qwen/Qwen3-8B \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag,gsm8k \
--fp16 \
--gptq --wbits 4 --act-order
```

暂未实现 SPSR 模型量化

下面是一个使用 SPSR 进行剪枝后量化的示例：

```shell
python main.py \
--base_model ./checkpoints/Qwen3-8B-spsrs-8-tune \
--insert ./checkpoints/Qwen3-8B-spsrs-8-tune/linear.pth \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag,gsm8k \
--fp16 \
--gptq --wbits 4 --act-order
```

加载量化模型

```shell
python main.py \
--base_model ./checkpoints/Qwen3-8B-spsrs-8-tune \
--insert ./checkpoints/Qwen3-8B-spsrs-8-tune/linear.pth \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag,gsm8k \
--fp16 \
--load_quant ./checkpoints/Qwen3-8B-spsrs-8-tune/quantized.pth
```