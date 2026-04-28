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

spsr layers 单独保存：

```
/path/to/spsr-model/          # SPSR 处理后的模型目录
├── config.json              # 包含 SPSR 元数据的配置
├── generation_config.json
├── special_tokens_map.json
├── spsr_layers.pth          # SPSR 层信息
├── tokenizer.json
└── tokenizer_config.json
```

在七个zero-shot 任务上与其它层剪枝组件替换方法的比较（修剪8层）

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

spsr layers 和 lora adapter 单独保存：

```
/path/to/spsr-model/          # SPSR 处理后的模型目录
├── adapter_config.json
├── adapter_model.safetensors  # LoRA adapter 参数
├── config.json                # 包含 SPSR 元数据的配置
├── generation_config.json
├── special_tokens_map.json
├── spsr_layers.pth           # SPSR 层信息
├── tokenizer.json
└── tokenizer_config.json
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
| Streamline+LoRA | 6.07 |
| SPSR            | 15.77 |
| SPSR+LoRA       | 31.99 |
| **Qwen3-8B**    | 51.40 |
| Streamline+LoRA | 41.17 |
| SPSR            | 6.90  |
| SPSR+LoRA       | 47.69 |
| **Qwen3-4B**    | 49.73 |

在GSM8K上的8-shot结果如下。

| model           | GSM8K |
|-----------------|-------|
| **Qwen3-8B**    | 91.43 |
| Streamline+LoRA | 57.62 |
| SPSR            | 26.08 |
| SPSR+LoRA       | 62.47 |
| **Qwen3-4B**    | 82.18 |



### vllm 部署

vllm 是一个用于部署大语言模型的推理引擎，支持量化、稀疏、多卡推理等特性。

SPSR 模型的部署详情参见 [VLLM-server](./VLLM-server/README.md)。

性能测试

TTFT（Time to First Token）表示从请求开始到第一个 token 返回的时间。
TPOT（Time per Output Token）表示每个输出 token 的平均生成时间。
ITL（Inter-token Latency）表示连续两个 token 之间的延迟。

| 模型 | Request throughput (req/s) | Output token throughput (tok/s) | TTFT(ms) | TPOT(ms) | ITL(ms) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Llama3-8B | 37.61   | 150.42 | 63.89 | 22.63 |23.49  |
| +spsr-8|  44.68 | 178.74 |52.56  | 19.06   |20.32|
| +AutoAWQ-4bit|  61.29 | 245.17 |38.62  | 13.97   |14.60|


### 量化

在 CPU Intel(R)Xeon (R)CPUE5-2699 和 GPU NVIDIA GeForce RTX 3090 上进行测试，相比 fp16，4-bit 量化减少了 60% 的内存占用，同时性能损失在 10% 以内。

量化后推理速度没有提升，甚至更慢。

当 Batch Size 较小时，推理过程主要受限于内存带宽。这意味着GPU从显存中读取庞大模型权重的速度是瓶颈。量化将模型权重的体积显著缩小（例如缩小3倍），大大减少了需要从内存中搬运的数据量，从而加快了数据读取速度。

当 Batch Size 较大时，推理过程主要受限于计算能力。此时，GPU的大部分时间都花在了矩阵乘法运算上。量化模型（如W4A16）虽然存储的是INT4格式的权重，但在计算时需要先将其反量化为FP16格式。这个额外的转换步骤会带来计算开销，反而拖慢了整体的生成速度。

如 SmoothQuant 这种对权重和激活都量化的方法，中间可以省去反量化步骤，从而避免了额外的计算开销，可以比未量化模型快。

使用 AutoAWQ 量化，vLLM 部署，短序列、小批量场景下推理速度相比原模型有提升。

| LLaMA3-8B | Bits | group-size | memory(MiB) | TPOT(ms) | C4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| fp16 | - | -| 15580 | 46.9| 6.54 | 
| AutoAWQ | 4 | 128 | 5726 | 94.1 | 6.86 |
| GPTQ(me) | 4 | 128 | 5832 | 47.2 | 11.20 |
| GPTQ(me) | 4 | -1 | 5705 | 56.2 | 7.98 |

cuda kernel 

```shell
cd quant4bit
python setup.py install
```

GPTQ 实现见 [quant4bit](./quant4bit/README.md)。

模型量化示例：

```shell
python main.py \
--base_model Qwen/Qwen3-8B \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag,gsm8k \
--fp16 \
--gptq --wbits 4 --act-order
--save_path ./checkpoints/Qwen3-8B-spsrs-8-4bit
```

加载量化模型

```shell
python main.py \
--base_model Qwen/Qwen3-8B \
--tasks wikitext,ptb,c4,storycloze,rte,openbookqa,arc_easy,winogrande,arc_challenge,piqa,boolq,hellaswag,gsm8k \
--fp16 \
--gptq --wbits 4 --act-order
--load_quant ./checkpoints/Qwen3-8B-spsrs-8-4bit
```