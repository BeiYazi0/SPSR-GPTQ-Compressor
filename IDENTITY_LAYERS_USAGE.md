# IdentityNormLike 和 IdentityLinearLike 层保存和加载指南

## 概述

`SPSRPlusPruner` 将模型的某些**层范围**替换为单个 `IdentityNormLike` 层（一种轻量级的适配层）。在微调过程中，`IdentityNormLike` 会被替换为 `IdentityLinearLike` 层进行训练。本文档说明如何以兼容的方式保存和加载这些层。

## 关键概念

### 层范围替换 (IdentityNormLike)

`SPSRPlusPruner` 不是替换单个层，而是替换连续的层范围。例如：
- 将原始模型的**第 5-8 层**替换为**一个 IdentityNormLike 层**
- 将原始模型的**第 20-22 层**替换为**另一个 IdentityNormLike 层**


### 层插入 (IdentityLinearLike)

在微调时，`IdentityNormLike` 层会被替换为 `IdentityLinearLike` 层，这些层在微调完成后可以被清除并单独保存，然后在需要时重新插入到微调模型中。


## 工作流程

### 第一部分：SPSR 剪枝和 IdentityNormLike 层

#### 第一步：训练/剪枝并保存 IdentityNormLike 层

运行剪枝时，指定 `--save_path` 参数来保存 IdentityNormLike 层的权重和替换信息：

```bash
python main.py \
  --base_model meta-llama/Llama-2-7b \
  --pruner spsrs \
  --final_s 0.25 \
  --save_path ./checkpoints/identity_layers.pth \
  --dataset c4 \
  --num_examples 128 \
  --epochs 10 \
  --lr 3e-4
```

**说明：**
- `--pruner spsrp` 或 `spsrs`：使用 SPSR+ 剪枝器
- `--final_s 0.25`：剪枝比例（保留 75% 的层）
- `--save_path ./checkpoints/identity_layers.pth`：保存 IdentityNormLike 权重的路径

**输出：**
保存的文件包含所有被替换的层范围及其对应权重：
```
identity_layers: [
    {
        'start_index': 5,           # 原始模型中被替换层范围的起始索引
        'end_index': 8,             # 原始模型中被替换层范围的结束索引（不含）
        's': tensor(...),           # 缩放因子
        'bias': tensor(...),        # 偏置
        'relu': False
    },
    {
        'start_index': 20,
        'end_index': 22,
        's': tensor(...),
        'bias': tensor(...),
        'relu': False
    },
    ...
]
model_type: 'meta-llama/Llama-2-7b'
original_num_layers: 32
```

#### 第二步：加载 IdentityNormLike 层进行推理/微调

在新的运行中加载保存的 IdentityNormLike 层：

```bash
python main.py \
  --base_model meta-llama/Llama-2-7b \
  --load_spsr_path ./checkpoints/identity_layers.pth
```

**说明：**
- `--load_spsr_path`：指定保存的 IdentityNormLike 权重文件路径
- 加载函数会：
  1. 加载原始的完整模型
  2. **从后往前**逐一替换对应的层范围
  3. 清理 GPU 缓存
- 加载后可继续进行 GPTQ 量化、微调或评估

### 第二部分：微调和 IdentityLinearLike 层

### 加载后进行额外操作

#### 加载后进行 GPTQ 量化

```bash
python main.py \
  --base_model meta-llama/Llama-2-7b \
  --load_spsr_path ./checkpoints/identity_layers.pth \
  --gptq \
  --wbits 4 \
  --act-order
```

#### 加载后进行微调

```bash
python main.py \
  --base_model meta-llama/Llama-2-7b \
  --load_spsr_path ./checkpoints/identity_layers.pth \
  --tune \
  --output_dir ./finetuned_model
```

**说明：**
- 微调过程中 `IdentityNormLike` 会自动替换为 `IdentityLinearLike`
- 微调完成后会自动保存标准模型和 `IdentityLinearLike` 参数

#### 加载后进行评估

```bash
python main.py \
  --base_model meta-llama/Llama-2-7b \
  --load_spsr_path ./checkpoints/identity_layers.pth \
  --tasks winogrande,hellaswag,arc_easy
```

## 保存的文件格式

保存文件是 PyTorch 字典，包含以下结构：

```python
{
    'identity_layers': [
        {
            'start_index': 5,
            'end_index': 8,
            's': tensor([...]),
            'bias': tensor([...]),
            'relu': False
        },
        {
            'start_index': 20,
            'end_index': 22,
            's': tensor([...]),
            'bias': tensor([...]),
            'relu': False
        },
        ...
    ],
    'model_type': 'meta-llama/Llama-2-7b',
    'original_num_layers': 32
}
```

## 技术细节

### IdentityNormLike 层的作用

IdentityNormLike 层在剪枝的层之间执行线性适配：

```python
output = input * s + bias
```

其中 s 是缩放因子（按元素），bias 是偏置向量。这样可以补偿被移除层对输出分布的影响。

### 为什么使用层范围而不是单个层

使用层范围的原因：
1. **连贯性**：删除的层通常是连续的
2. **效率**：减少需要训练的参数
3. **稳定性**：简化模型结构

### 加载时为什么从后往前替换

从后往前替换的原因：
- **避免索引混乱**：如果从前往后替换，每次替换都会改变后续层的索引
- 例如：替换 `[5-8]` 后，原本的 `[20-22]` 现在变成了 `[16-18]`
- 从后往前避免了这个问题

### 支持的模型架构

- **Llama**: 自动检测 `model.model.layers`
- **OPT**: 自动检测 `model.model.decoder.layers`
- **Qwen**: 自动检测 `model.transformer.h`
- 其他：默认使用 `model.model.layers`

## 常见问题

### Q: 是否可以在不同的模型间转移 IdentityNormLike 层？

**A:** 不建议。IdentityNormLike 权重是针对特定的基础模型训练的。如果要在不同模型上使用，需要重新训练。

### Q: GPU 缓存清理是否必要？

**A:** 是的。加载过程自动调用 `torch.cuda.empty_cache()`，确保 GPU 内存得到正确释放。

### Q: 如何验证加载是否成功？

**A:** 检查控制台输出，应该显示类似以下信息：
```
Replaced layers [5:8] with IdentityNormLike
Replaced layers [20:22] with IdentityNormLike
...
Successfully loaded IdentityNormLike layers from ./checkpoints/identity_layers.pth
```

## IdentityLinearLike 层保存和加载

### 概述

在微调过程中，`IdentityNormLike` 层会被替换为 `IdentityLinearLike` 层进行训练。微调完成后，可以清除 `IdentityLinearLike` 层，保存标准模型，并单独保存 `IdentityLinearLike` 参数供后续插入使用。

### 工作流程

#### 第一步：微调并保存 IdentityLinearLike 层

运行微调时，系统会自动将 `IdentityNormLike` 替换为 `IdentityLinearLike`，并在微调完成后保存：

```bash
python main.py \
  --base_model meta-llama/Llama-2-7b \
  --load_spsr_path ./checkpoints/identity_layers.pth \
  --tune \
  --save_path ./finetuned_model
```

**说明：**
- `--load_spsr_path`：加载预剪枝的 IdentityNormLike 层
- `--tune`：进行微调
- `--save_path`：保存路径（模型保存到此路径，IdentityLinearLike保存到此路径+"_linear.pth"）

**输出：**
- `./finetuned_model/`：标准模型文件（移除了 IdentityLinearLike）
- `./finetuned_model_linear.pth`：IdentityLinearLike 参数文件

#### 第二步：加载 IdentityLinearLike 层进行推理

在新的运行中加载保存的 IdentityLinearLike 层：

```bash
python main.py \
  --base_model ./finetuned_model \
  --insert ./finetuned_model_linear.pth
```

**说明：**
- `--base_model`：标准模型路径
- `--insert`：IdentityLinearLike 参数文件路径
- 加载函数会：
  1. 加载标准模型
  2. 在对应位置插入 IdentityLinearLike 层
  3. 更新模型配置

### 保存的文件格式

IdentityLinearLike 保存文件包含：

```python
{
    'identity_linear_layers': [
        {
            'index': 5,              # 插入位置的索引
            's': tensor([...]),      # 缩放矩阵
            'bias': tensor([...])    # 偏置向量
        },
        ...
    ],
    'model_type': 'llama',
    'original_num_layers': 28
}
```

### IdentityLinearLike 层的作用

IdentityLinearLike 层执行线性变换：

```python
output = (input @ s) + bias
```

其中 s 是可训练的权重矩阵，bias 是偏置向量。与 IdentityNormLike 不同，IdentityLinearLike 使用矩阵乘法而不是元素级操作。

### 技术细节

- **插入模式**：IdentityLinearLike 通过插入方式添加，不会替换现有层
- **位置保持**：插入位置与微调时的位置一致
- **配置更新**：插入后自动更新模型的 `num_hidden_layers` 配置

### Q: 如果层范围重叠会怎样？

**A:** 在正常的 SPSR+ 工作流中不会发生重叠。如果自定义代码中发生重叠，从后往前替换可以避免大多数索引问题。

## 内存使用提示

- **保存时**：IdentityNormLike 层的权重被移到 CPU 然后保存，不占用 GPU 内存
- **加载时**：权重被加载到与模型相同的设备，通常在 GPU 上
- **缩放**：每个 IdentityNormLike 层通常占用极少的内存（通常 < 1MB）

## 性能影响

- **推理延迟**：IdentityNormLike 层的计算非常快（简单的元素级操作）
- **模型大小**：保存的 IdentityNormLike 权重文件通常很小（通常 < 100MB）
- **吞吐量**：与完整模型相比，模型IdentityNormLike 层不会显著降低吞吐量

### Q: IdentityLinearLike 和 IdentityNormLike 有什么区别？

**A:** 
- `IdentityNormLike`：用于剪枝后的层范围替换，执行元素级缩放和偏置操作
- `IdentityLinearLike`：用于微调时的层替换，执行矩阵乘法和偏置操作，具有更多可训练参数

### Q: 为什么要清除 IdentityLinearLike 后保存模型？

**A:** 这样可以保存一个标准的 transformer 模型，便于在不同场景下使用。当需要恢复微调后的性能时，可以通过 `--insert` 参数重新插入 IdentityLinearLike 层。
