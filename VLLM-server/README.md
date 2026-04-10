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

#### Chat 流式输出实现原理

**下面详细讲解如何实现实时流式推理输出**：

##### 1. 核心架构设计

系统采用三层异步架构，实现客户端实时接收生成过程：

```
客户端请求 (stream=true)
    ↓
FastAPI 接收请求 (async def chat)
    ↓
创建异步生成器 streaming_resp()
    ↓
vLLM 异步引擎 engine.generate() 逐步产生 token
    ↓
StreamingResponse 将生成器结果流式发送给客户端
```

##### 2. vLLM 异步引擎核心

```python
# vLLM 使用 AsyncLLMEngine，支持逐 token 异步生成
results_iter = engine.generate(
    prompt=prompt_text,
    sampling_params=sampling_params,
    request_id=request_id, 
    lora_request=lora_request
)
```

- **engine.generate()** 返回的是异步迭代器 `AsyncIterator`
- 每次迭代时，返回一个 `RequestOutput` 对象，包含：
  - `outputs[0].token_ids`: 截止当前的所有 token ID 列表
  - `outputs[0].text`: 完整的解码文本
  - `finish_reason`: 生成是否完成

##### 3. 流式返回的关键步骤

**第一步：请求接收与参数解析**

```python
@app.post("/chat")
async def chat(request: Request):
    request = await request.json()
    query = request.get('query', None)
    stream = request.get("stream", False)  # 关键：检查是否启用流式
    user_stop_words = request.get("user_stop_words", [])
```

**第二步：创建异步生成器函数**

```python
if stream:
    async def streaming_resp():
        async for result in results_iter:  # 异步遍历每个生成步骤
            # result 是 RequestOutput 对象
            # 每个循环代表生成了一个新 token
            ...
    return StreamingResponse(streaming_resp())
```

**第三步：处理每个 token**

```python
async def streaming_resp():
    async for result in results_iter:
        # 关键：每次 results_iter 迭代，model 都会生成一个新 token
        # 此时 result.outputs[0].token_ids 包含所有已生成的 token
        
        # 移除系统停止词（如 eos_token, im_end 等）
        token_ids = remove_stop_words(result.outputs[0].token_ids, stop_words_ids)
        
        # 解码所有 token 为文本（从头开始累积）
        text = tokenizer.decode(token_ids)
        
        # 将当前状态转为 JSON 并编码为字节，以 UTF-8 null 分隔符结尾
        yield (json.dumps({'text': text}) + '\0').encode('utf-8')
        
        # 检查是否匹配用户自定义停止词（例如 "Observation:", "Action:"）
        if match_user_stop_words(token_ids, user_stop_tokens):
            await engine.abort(request_id)  # 立即停止推理
            break
```

**第四步：流式响应发送**

```python
return StreamingResponse(streaming_resp())
```

StreamingResponse 会：
1. 设置 HTTP 状态码 200
2. 将生成器逐个产生的数据块发送给客户端
3. 保持 HTTP 连接打开，直到生成完成
4. 遇到 `break` 立即关闭连接

##### 4. 数据流动过程示意

以推理文本 "Hello World" 为例，流式生成过程：

```
初始状态: token_ids = []

第 1 次迭代:
  ├─ vLLM 生成第 1 个 token："Hello"
  ├─ token_ids = [7592]
  ├─ text = "Hello"
  └─ 发送给客户端: {"text": "Hello"}

第 2 次迭代:
  ├─ vLLM 生成第 2 个 token：" World"
  ├─ token_ids = [7592, 3354]
  ├─ text = "Hello World"
  └─ 发送给客户端: {"text": "Hello World"}

第 3 次迭代:
  ├─ vLLM 生成第 3 个 token：eos_token
  ├─ token_ids = [7592, 3354, 128009]
  ├─ 移除停止词后：token_ids = [7592, 3354]
  ├─ text = "Hello World"
  └─ 发送给客户端: {"text": "Hello World"}
  
✓ 推理完成，连接关闭
```

**关键点**：
- 每次 yield 发送的是**完整到目前为止的文本**（从头开始），不仅是增量差
- 客户端可通过比较前后内容提取增量（delta）用于实时展示
- 支持中途停止：通过用户停止词或 token 数上限实现

##### 5. 与非流式模式的区别

**非流式模式**（`stream=False`）：

```python
# 整体一次性返回模式
final_text = None
async for result in results_iter:
    # 仍然遍历所有推理步骤，但不发送给客户端
    token_ids = remove_stop_words(result.outputs[0].token_ids, stop_words_ids)
    final_text = tokenizer.decode(token_ids)
    if match_user_stop_words(token_ids, user_stop_tokens):
        await engine.abort(request_id)
        break

# 最后才返回完整结果
return JSONResponse({"text": final_text})
```

**对比总结**：

| 特性 | 流式模式 | 非流式模式 |
|------|--------|---------|
| 返回方式 | 每个 token 产生时立即发送 | 所有 token 生成完成后一次性返回 |
| 客户端延迟 | 低延迟，立即看到首个 token | 高延迟，需等待完整推理 |
| 网络传输 | 多次小包传输 | 一次大包传输 |
| 实时体验 | 顺滑，可实时显示生成过程 | 无进度反馈 |
| 带宽效率 | 较低（多次 HTTP 头） | 较高（单次 HTTP 头） |

##### 6. 客户端接收流式数据示例

**Python 客户端**：

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/chat",
    json={"query": "你好，请自我介绍", "stream": True},
    stream=True  # 启用流式响应
)

for chunk in response.iter_lines():
    if chunk:
        data = json.loads(chunk.decode().rstrip('\0'))
        print(data['text'], end='', flush=True)  # 实时显示
print()  # 新行
```

**JavaScript 客户端**：

```javascript
const response = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
        query: "你好，请自我介绍",
        stream: true
    })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const text = decoder.decode(value);
    const lines = text.split('\0').filter(l => l);
    
    for (const line of lines) {
        const data = JSON.parse(line);
        process.stdout.write(data.text);  // 实时显示
    }
}
```

##### 7. 停止词机制深度剖析

**用户自定义停止词**：

```python
# 客户端请求
{
    "query": "计算 2+2",
    "stream": true,
    "user_stop_words": ["计算过程:", "最终答案:"]  # 定义 2 个停止词
}

# 服务器处理
user_stop_tokens = []
for words in user_stop_words:
    # 将停止词编码为 token ID 列表
    user_stop_tokens.append(tokenizer.encode(words))

# 推理过程中动态检查
async for result in results_iter:
    token_ids = remove_stop_words(result.outputs[0].token_ids, stop_words_ids)
    text = tokenizer.decode(token_ids)
    yield (json.dumps({'text': text}) + '\0').encode('utf-8')
    
    # 检查是否命中停止词
    if match_user_stop_words(token_ids, user_stop_tokens):
        await engine.abort(request_id)  # vLLM 立即停止推理
        break
```

**match_user_stop_words 实现逻辑**：

```python
def match_user_stop_words(response_token_ids, user_stop_tokens):
    for stop_tokens in user_stop_tokens:
        if len(response_token_ids) < len(stop_tokens):
            continue  # token 个数不足，跳过
        
        # 检查响应末尾是否匹配停止词
        if response_token_ids[-len(stop_tokens):] == stop_tokens:
            return True  # 命中停止词
    
    return False
```

**应用场景**：
- Agent 系统中的 Thought/Action/Observation 分界
- 多步推理中的阶段划分
- 对话系统中的自动停止

##### 8. 性能优化要点

**为什么这种设计高效**：

1. **异步非阻塞**：FastAPI 的异步处理确保单个小线程可处理多个并发连接
2. **vLLM 并发**：vLLM 内部使用 Ray 支持多请求的动态合批
3. **增量更新**：虽然每次发送完整文本，但网络只传输增量部分
4. **早期终止**：支持用户停止词实现动态提前停止
5. **内存高效**：StreamingResponse 不缓存整个响应，逐步流式发送

**可能的性能瓶颈**：

- **吞吐量**：每个 token 编码+解码+JSON 序列化的开销
- **解决方案**：可优化为仅发送增量 token 而非完整文本
- **延迟**：token 生成延迟受模型推理速度限制

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

```python
vllm bench serve     --model /home/jim/nas/yzg/Llama-3-8b/base     --host localhost     --port 8000     --random-input-len 32     --random-output-len 4      --num-prompts  5
```

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


autoawq 19720m
============ Serving Benchmark Result ============
Successful requests:                     5         
Benchmark duration (s):                  0.08      
Total input tokens:                      160       
Total generated tokens:                  20        
Request throughput (req/s):              61.29     
Output token throughput (tok/s):         245.17    
Total Token throughput (tok/s):          2206.52   
---------------Time to First Token----------------
Mean TTFT (ms):                          35.75     
Median TTFT (ms):                        37.87     
P99 TTFT (ms):                           38.62     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          13.64     
Median TPOT (ms):                        13.84     
P99 TPOT (ms):                           13.97     
---------------Inter-token Latency----------------
Mean ITL (ms):                           10.23     
Median ITL (ms):                         13.53     
P99 ITL (ms):                            14.60     
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