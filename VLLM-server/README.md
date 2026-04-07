# SPSR-GPTQ-Compressor VLLM 部署指南

## 概述

本项目实现了将 SPSR 压缩后的 Llama 模型与 vLLM 推理引擎的集成部署。通过自定义模型注册和权重加载机制，实现了在 vLLM 中直接加载和运行 SPSR 修改过的模型结构。

基于 vLLM 高性能推理框架，充分利用其 Continuous Batching 机制，实现对海量并发请求的高吞吐批量推理能力。系统架构采用 uvicorn + FastAPI 构建异步 HTTP 服务，主线程通过 asyncio 将用户请求提交至 vLLM 推理队列，由独立推理线程完成动态合批计算，并异步返回结果。

同时，项目支持流式推理输出：vLLM 原生提供逐 token 生成能力，结合 FastAPI 可实现按 chunk 的流式响应，客户端可通过 requests 等库逐步接收并实时展示生成过程，从而带来低延迟、顺滑的交互体验。

### 核心特性

- **自定义模型注册**: 注册 `SPSRLlamaForCausalLM` 作为 vLLM 支持的模型类型
- **动态层替换**: 在模型加载时自动应用 SPSR 层的 IdentityNormLike 替换
- **兼容性**: 保持与标准 Llama 模型的 API 兼容性
- **高性能推理**: 利用 vLLM 的异步推理引擎和 GPU 优化

## 文件结构

```
VLLM-server/
├── modeling_spsr_llama.py    # SPSR Llama 模型实现
├── vllm_server_llama.py     # FastAPI 服务器和 vLLM 引擎配置
├── prompt_utils.py           # 提示构建和停止词处理工具
├── vllm_client.py           # 客户端测试脚本
└── README.md                # 本文档
```

## 技术细节

### 1. 自定义模型实现 (`modeling_spsr_llama.py`)

#### IdentityNormLike 层

```python
class IdentityNormLike(nn.Module):
    def __init__(self, s: torch.Tensor, bias: torch.Tensor, relu: bool = False):
        super().__init__()
        self.s = nn.Parameter(s)
        self.bias = nn.Parameter(bias)
        self.relu = relu
```

- **作用**: 实现 SPSR 中的层替换，将多个连续的 Transformer 层替换为一个简单的缩放+偏置操作
- **前向传播**: `hidden_states = hidden_states * s + bias`，注意vllm的llama模型中，decoder layer 的输入为 positions, hidden_states, residuals, 后两者拆分开来以便加速运算，在RMSNorm层中，vLLM实现了特殊的融合操作。IdentityNormLike 需要对两者都进行缩放+偏置操作。

#### SPSRLlamaForCausalLM 类

```python
class SPSRLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        # 从配置中提取 SPSR 元数据
        hf_config = vllm_config.model_config.hf_config
        self.base_model_path = hf_config.spsr_metadata["original_model"]
        self.spsr_path = vllm_config.model_config.model
        
        # 修改配置指向基础模型
        setattr(vllm_config.model_config, "model", self.base_model_path)
        
        super().__init__(vllm_config=vllm_config, prefix=prefix)
```

- **继承关系**: 继承自 vLLM 的标准 `LlamaForCausalLM`
- **配置处理**: 从 vLLM 配置中提取 SPSR 相关路径
- **模型路径重定向**: 将模型路径重定向到基础模型，确保权重加载正确

#### 权重加载流程

```python
def load_weights(self, weights: Dict[str, torch.Tensor]):
    """重写加载权重方法，先加载基础权重，再应用SPSR"""
    print("Loading base model weights into SPSR model...")
    super().load_weights(weights)
    
    if self.spsr_path is not None:
        print(f"Applying SPSR layers from: {self.spsr_path}")
        self.apply_spsr_layers(self.spsr_path)
```

1. **基础权重加载**: 调用父类的 `load_weights` 加载原始 Llama 模型权重
2. **SPSR 层应用**: 如果存在 SPSR 配置，则应用层替换

#### SPSR 层应用 (`apply_spsr_layers`)

```python
def apply_spsr_layers(self, spsr_path: str):
    """应用SPSR层替换 - 优化版本"""
    device = next(self.parameters()).device
    dtype = next(self.parameters()).dtype
    
    # 加载 SPSR 配置
    save_data = torch.load(spsr_path + "/spsr_layers.pth", map_location=device)
    identity_layers_info = save_data['identity_layers']
    
    # 从后往前替换层
    for layer_info in reversed(identity_layers_info):
        start_index = layer_info['start_index']
        end_index = layer_info['end_index']
        s = layer_info['s'].to(device, dtype)
        bias = layer_info['bias'].to(device, dtype)
        relu = layer_info['relu']
        
        identity_layer = IdentityNormLike(s, bias, relu=relu)
        
        # 替换层
        new_layers_list = list(layers[:start_index]) + [identity_layer] + list(layers[end_index:])
        self.model.layers = nn.ModuleList(new_layers_list)
        layers = self.model.layers
        
        print(f"Replaced layers [{start_index}:{end_index}] with IdentityNormLike")
```

- **配置加载**: 从 `spsr_layers.pth` 文件加载 SPSR 层信息
- **层替换策略**: 从后往前替换，避免索引混乱
- **参数转换**: 确保参数在正确的设备和数据类型上

### 2. 服务器实现 (`vllm_server_llama.py`)

#### 模型注册

```python
from modeling_spsr_llama import SPSRLlamaForCausalLM
from vllm import ModelRegistry
ModelRegistry.register_model("SPSRLlamaForCausalLM", SPSRLlamaForCausalLM)
```

- **注册机制**: 将自定义模型类注册到 vLLM 的模型注册表中

#### 配置设置

```python
# 路径配置
base_model_path = "/home/jim/nas/yzg/Llama-3-8b/base"  # 基础模型路径
model_dir = "/media/ssd/yzg/SPSR-GPTQ-Compressor/checkpoints/Llama3-8B-spsr-8"  # SPSR 模型目录

# vLLM 参数配置
args = AsyncEngineArgs(model_dir)
args.worker_use_ray = False
args.engine_use_ray = False
args.tensor_parallel_size = tensor_parallel_size
args.trust_remote_code = True
args.gpu_memory_utilization = gpu_memory_utilization
args.dtype = dtype
args.max_num_seqs = 20
```

- **路径分离**: 
  - `base_model_path`: 原始 Llama 模型权重
  - `model_dir`: SPSR 处理后的模型目录，包含配置和元数据

#### FastAPI 服务

- **聊天接口**: `/chat` POST 接口，支持流式和非流式响应
- **参数支持**: 
  - `query`: 用户查询
  - `history`: 对话历史
  - `system`: 系统提示
  - `stream`: 是否流式输出
  - `user_stop_words`: 自定义停止词

### 3. 工具模块

#### prompt_utils.py

- **`_build_prompt`**: 构建 ChatML 格式的提示，支持多轮对话和历史管理
- **`remove_stop_words`**: 移除响应中的停止词 token

#### vllm_client.py

- **测试客户端**: 简单的命令行客户端，用于测试服务器功能
- **流式处理**: 支持流式响应显示

## 安装和依赖

### 环境要求

- Python 3.11+
- PyTorch 2.6.0+cu124
- vLLM 0.8.5.post1
- Transformers 4.57.6
- FastAPI
- Uvicorn


## 配置和运行

### 1. 准备模型文件

确保以下文件和目录存在：

```
/path/to/base-model/          # 原始 Llama 模型目录
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── pytorch_model.bin (或 safetensors)

/path/to/spsr-model/          # SPSR 处理后的模型目录
├── config.json              # 包含 SPSR 元数据的配置
├── tokenizer.json
├── tokenizer_config.json
└── spsr_layers.pth          # SPSR 层信息
```

### 2. 修改配置路径

在 `vllm_server_llama.py` 中更新路径：

```python
model_dir = "/path/to/spsr-model"        # SPSR 模型目录
```

### 3. 启动服务器

```bash
python vllm_server_llama.py
```

服务器将在 `http://localhost:8000` 启动。

### 4. 测试服务

使用提供的客户端：

```bash
python vllm_client.py
```

或使用 curl：

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "Hello, how are you?", "stream": false}'
```

## SPSR 配置格式

### spsr_layers.pth 结构

```python
{
    'identity_layers': [
        {
            'start_index': int,    # 替换起始层索引
            'end_index': int,      # 替换结束层索引
            's': torch.Tensor,     # 缩放参数
            'bias': torch.Tensor,  # 偏置参数
            'relu': bool          # 是否使用 ReLU
        },
        # ... 更多层信息
    ],
    'model_type': str,           # 模型类型 ('llama', 'opt', 等)
    'original_num_layers': int   # 原始模型层数
}
```

### 模型配置 (config.json)

SPSR 模型的 `config.json` 需要包含 SPSR 元数据：

```json
{
    "architectures": ["SPSRLlamaForCausalLM"],
    "model_type": "llama",
    "spsr_metadata": {
        "original_model": "/path/to/base-model",
        "spsr_layers": "/path/to/spsr-model/spsr_layers.pth"
    },
    // ... 其他标准 Llama 配置
}
```

## 技术优势

### 1. 内存效率

- **动态替换**: 只在需要时加载和替换层，避免预加载所有变体
- **权重复用**: 基础模型权重与 SPSR 层分离，减少存储开销

### 2. 推理性能

- **vLLM 优化**: 利用 vLLM 的 KV 缓存、连续批处理和 GPU 优化
- **异步处理**: 支持高并发请求处理

### 3. 兼容性

- **API 兼容**: 与标准 Transformers 模型 API 保持一致
- **配置灵活**: 支持不同 SPSR 配置的动态加载

## 故障排除

### 调试技巧

参见 [troubleshooting](https://github.com/vllm-project/vllm/blob/c6f722b93e8e795065751172812ee6a5540e5901/docs/usage/troubleshooting.md)

普通断点 pdb 无效，使用 fpdb：
__import__('fpdb').ForkedPdb().set_trace()

## 性能基准

vllm bench serve     --model /home/jim/nas/yzg/Llama-3-8b/base     --host localhost     --port 8000     --random-input-len 32     --random-output-len 4      --num-prompts  5

llama3-8B-spsr-8 19730m
============ Serving Benchmark Result ============
Successful requests:                     5         
Benchmark duration (s):                  0.11      
Total input tokens:                      160       
Total generated tokens:                  20        
Request throughput (req/s):              44.68     
Output token throughput (tok/s):         178.74    
Total Token throughput (tok/s):          1608.64   
---------------Time to First Token----------------
Mean TTFT (ms):                          47.32     
Median TTFT (ms):                        50.26     
P99 TTFT (ms):                           52.56     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          18.89     
Median TPOT (ms):                        18.84     
P99 TPOT (ms):                           19.06     
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.16     
Median ITL (ms):                         18.41     
P99 ITL (ms):                            20.32     
==================================================


llama3-8B 22398m
============ Serving Benchmark Result ============
Successful requests:                     5         
Benchmark duration (s):                  0.13      
Total input tokens:                      160       
Total generated tokens:                  20        
Request throughput (req/s):              37.61     
Output token throughput (tok/s):         150.42    
Total Token throughput (tok/s):          1353.82   
---------------Time to First Token----------------
Mean TTFT (ms):                          58.04     
Median TTFT (ms):                        61.91     
P99 TTFT (ms):                           63.89     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          22.47     
Median TPOT (ms):                        22.48     
P99 TPOT (ms):                           22.63     
---------------Inter-token Latency----------------
Mean ITL (ms):                           22.47     
Median ITL (ms):                         22.73     
P99 ITL (ms):                            23.49     
==================================================




### 典型配置

- **模型**: Llama-3-8B with SPSR (20% 层压缩)
- **并发**: 20 个请求
- **吞吐量**: ~150 tokens/second
- **延迟**: ~50ms/token (首token), ~20ms/token (后续)

### 优化建议

1. **批处理**: 增加 `max_num_seqs` 以提高吞吐量
2. **量化**: 使用 vLLM 的量化选项进一步压缩模型
3. **并行**: 设置 `tensor_parallel_size > 1` 利用多GPU