# nano-vllm

[源码深度解剖](https://www.toutiao.com/article/7602678840405574144/)

nano-vllm 是由 DeepSeek 工程师开源的极简 LLM 推理引擎实现。
- 代码约 1200 行 Python，覆盖从配置、调度、KV 管理到模型前向的完整链路。
- 在 GitHub 上获得约 12.6k+ stars，说明其作为 vLLM 思想的可运行缩影，被大量开发者用于学习。
- 它不是为了替代 vLLM 上生产，而是为了用可维护的代码量讲清推理系统的主干。

example.py

```python
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
```

## 总体架构

### 目录结构

```
nanovllm/
├── __init__.py          # 导出 LLM, SamplingParams
├── config.py            # Config dataclass
├── llm.py               # LLM(LLMEngine) 空子类或薄封装
├── sampling_params.py   # SamplingParams dataclass
├── engine/
│   ├── llm_engine.py    # 引擎主循环
│   ├── scheduler.py     # Prefill/Decode 调度
│   ├── sequence.py      # 序列状态管理
│   ├── block_manager.py # KV Cache 块管理
│   └── model_runner.py  # GPU 上执行模型
├── layers/
│   ├── attention.py     # FlashAttention + Triton KV 存储
│   ├── linear.py        # 列/行/QKV 并行 Linear
│   ├── embed_head.py    # 词表并行 Embedding + LMHead
│   ├── layernorm.py     # RMSNorm
│   ├── activation.py    # SiluAndMul (SwiGLU)
│   ├── rotary_embedding.py
│   └── sampler.py       # 温度采样
├── models/
│   └── qwen3.py         # Qwen3 模型实现
└── utils/
    ├── context.py       # 全局推理上下文（prefill/decode、cu_seqlens 等）
    └── loader.py        # safetensors 权重加载
```

### 四层架构

#### 接口层

由 `llm.py` + `sampling_params.py` + `config.py` 构成，用户只接触 `LLM` 与 `SamplingParams`。Config 聚合显存、batch、TP、KV 块等全局约束，在引擎初始化时一次性生效。

#### 引擎层

由 `engine/llm_engine.py` 构成，负责调度与模型执行。从 `Scheduler` 取可运行批次，调用 `ModelRunner`，再采样、更新序列状态。

#### 调度层

由 `scheduler.py` + `sequence.py` + `block_manager.py` 构成，负责调度与序列状态管理。
- `Scheduler` 决定本轮是 prefill 还是 decode，拼接 batch，受 `max_num_batched_tokens` 等约束。
- `Sequence` 跟踪每条请求 token、步数、是否结束。
- `BlockManager` 为 KV 分配物理块，与 Attention 里的 `block_table` 呼应。

#### 执行层

由 `engine/model_runner.py` + `models/qwen3.py` + `layers/*` 构成。

- `ModelRunner`：准备 输入张量、设置 全局 `context`（`is_prefill`、`cu_seqlens_q` 等），调用 `Qwen3`。
- `layers`：算子级实现；`attention.py` 同时承担写 KV 与 FlashAttention 前向。

## 配置

`dataclass` 自动生成 `__init__` 方法，并支持类型检查。

### config.py

```python
@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512   # 并发序列条数上限
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1 # 张量并行卡数，1~8
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1  # 默认 -1 表示后续再设
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
```

Config 在初始化阶段被 `LLMEngine` / `ModelRunner` / `BlockManager` 消费。

- max_num_batched_tokens：单步前向批内 token 总数上限，限制算子输入规模，防止 OOM；需能覆盖单条最长序列（max_num_batched_tokens >= max_model_len）。

- gpu_memory_utilization：KV等占用 GPU 显存的比例，留余量给框架、碎片、临时 tensor。

- enforce_eager：True 时不用 CUDA Graph，便于调试。

- kvcache_block_size：KV 分页块大小，需 256 的倍数（与内核/对齐有关）。

- num_kvcache_blocks：块总数，-1 常表示自动算，结合显存利用率与块大小推算池大小。

### sampling_params.py

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

为简化采样路径、与温度采样统一，禁止 temperature=0（贪心），贪心需 argmax 分支。

## 序列并行

### embedding

当嵌入矩阵过大（大词表 × 隐藏维），可在词表维度切分。

第 r 张卡只保存 行索引区间 [v_start, v_end) 对应的 V/tp_size 行。前向时若某 token id 落在本卡区间，用本地行查表，否则为 0。

每张卡得到部分向量，需 All-reduce（求和） 合并。每张卡输出的嵌入向量数量与输入 token 数量相同，直接按卡数划分张量的 rank 归属，执行 reduce-scatter 即可使得每张卡对应 rank 位置的嵌入完整（同一 token 只在一张卡上非0），接着执行 All-Gather，使得每张卡都获得所有 token 对应的嵌入。

```
V=4
tokens = [0, 3, 1, 2]

卡1 (0~1)  |    卡2 (2~3)

embedding
卡1 [[0.3, 0.4], [0., 0.], [0.1, 0.1], [0., 0.]]
卡2 [[0., 0.], [0.5, 0.7], [0., 0.], [0.6, 0.9]]

reduce-scatter
卡1 [[0.3, 0.4], [0., 0.], [0.1, 0.1], [0.6, 0.9]]
卡2 [[0.3, 0.4], [0.5, 0.7], [0., 0.], [0.6, 0.9]]

all-gather
卡1 [[0.3, 0.4], [0.5, 0.7], [0.1, 0.1], [0.6, 0.9]]
卡2 [[0.3, 0.4], [0.5, 0.7], [0.1, 0.1], [0.6, 0.9]]
```

### LMHead

许多模型输出层与输入嵌入共享权重。

- Prefill：序列中每个位置都有 hidden，但 训练/推理目标 常是 最后一个位置预测下一 token；nano-vllm 用 cu_seqlens_q 取 每条序列最后一个 query 位置。
- Decode：通常每序列 1 个 token，形状与上下文 is_prefill 由 model_runner 设置。


```python
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
```

LMHead 的输出只包含 rank 对应词表区域的 logits，需要 gather 到 0 卡，并将它们存储在一个张量列表中，其它卡返回 None。

Prefill 时 LMHead 取 last：
```
batch 内 3 条序列，展平后 hidden 下标:
  seq0: [0,1,2]
  seq1: [3,4]
  seq2: [5,6,7,8]

cu_seqlens_q 类似 [0,3,5,9]
last_indices = [2,4,8]  -> 只取这三处 hidden 做 logits
```
LMHead 的输入由 batch 内各序列拼接而成。

cu_seqlens的长度是 batch_size+1，其实也就是正好对应了这个长度 [0, batch_1的序列长度, batch_1 + batch_2 的序列长度, ....+ batch_size的序列长度]。

## 模型

### Attention

FlashAttention 通过 分块（tiling）、在 SRAM 上完成 softmax 规约，减少 HBM 访问量，在长序列上显著加速。调用参见[flash-attn库中三种推理形式](https://zhuanlan.zhihu.com/p/2002424125411569756)，原理参见[flash_attention简要笔记](https://qinganzhang.github.io/posts/flash_attention%E7%AE%80%E8%A6%81%E7%AC%94%E8%AE%B0/)。

- `flash_attn_varlen_func`：输入 **多条序列拼接** 成的张量，配合 cu_seqlens_* 描述边界。适合 Prefill，因果掩码 为 True。可选 block_table，把逻辑 token 映射到 物理块，支持 非连续 KV 存储。

- `flash_attn_with_kvcache`：针对 已有 KV cache、当前步 短 Q（通常每序列 1 token）优化。适合 Decode，从 k_cache / v_cache 读取历史，cache_seqlens 描述当前已缓存长度等。

```python
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
```

这里 k_cache 和 v_cache 在 model_runner 初始化时开辟连续空间，该层 attention 有且只有这么多可用。所谓非连续的存储应当指的是 对于具体的某个序列 而言，整体是连续的。**先 store 再 attention**，prefill 和 decode 阶段都会调用 store_kvcache，将 k/v 存入 cache，这里的 k/v 都已经做了 rope。

Prefix cache：当 部分前缀的 KV 已存在于 cache（例如 共享系统提示、多轮对话复用），本轮 不必重新计算 前缀 K/V。显然，这里的 k/v 不是完整的 prompt，而是除了共享前缀外的部分，前面调用 store_kvcache 将这部分 k/v 存入 cache，完整的 prompt 应该凭借 block_table 从 cache 中读取。

`store_kvcache_kernel`：slot_mapping 大小为 N，为所有序列的每个 token k/v 指定存储位置。slot = -1 表明 padding 或 不归属本步写入 的位置。 没有具体说明 slot_mapping 如何调度，这不是该函数的职责，其只需要将 k/v 存入 cache 的指定位置。

### ROPE

RoPE 的核心思想：对每个位置 (m)，把 head 维度上的向量看成若干二维子空间上的向量，对每个子空间施加依赖位置 (m) 的旋转。这样，内积 $\langle R_m q, R_n k\rangle$ 会自然依赖相对位置 (m-n)。

二维向量 $(x_1, x_2)$ 旋转 $\theta$ 角：

$\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} =
\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} $

等价地，把 $(x_1, x_2)$ 看成复数 $x_1 + i x_2$，旋转即乘以 $e^{i\theta}$。

对 RoPE，不同频率对应 head 内不同「子对」：第 $j$ 对使用频率 $\theta_j$，位置 (m) 处相位为 $m\theta_j$。

设 rotary_dim = d，通常取偶数维，按对处理。第 $j$ 个频率（代码里 arange(0, rotary_dim, 2) 给出 $j=0,1,\ldots$）：

$ \theta_j = \mathrm{base}^{-2j/d} $

即 inv_freq[j] = 1 / base^(2j/d)。位置 $t$ 上该频率的相位为 $t \cdot \theta_j$（代码用 einsum("i,j->ij", t, inv_freq) 对所有位置、所有频率一次性算完）。

base（如 10000）控制频率谱：高频分量编码细粒度相对位置，低频对应长程模式。

```python
from functools import lru_cache

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
```

预先算好 [max_position, 1, rotary_dim] 的 cache（cos 与 sin 在最后一维拼接），前向只做索引 cos_sin_cache[positions]，避免重复三角函数与广播开销。

@torch.compile 把 forward 编成更高效的融合内核路径，减少 Python 调度开销；与静态形状的 RoPE 查表+chunk 很契合。动态形状极端变化时重编译

@lru_cache(1)：同一 (head_size, rotary_dim, max_position, base) 只构造一份 RotaryEmbedding 模块实例，避免每层重复创建与重复 buffer。

positions 为当前 batch 各 token 的位置 id，支持非连续位置（如解码步）。

nano-vllm rope 的实现有问题，`x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)` 获得的是 `x1 = [..., 0:head_dim//2], x2 = [..., head_dim//2:head_dim]`，这与 rope 定义不符合，正确应当是 `x1 = x[..., 0::2], x2 = x[..., 1::2]`。

nano-vllm rope 采用与 hf 相似的 `apply_rotary_emb`，但后者在加载权重的时候会对 q 和 k 权重做 [permute](https://github.com/huggingface/transformers/blob/7028c30b6df3369f984469a089d26037e4ce3c1b/src/transformers/models/llama/convert_llama_weights_to_hf.py#L220)：`w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)`。这种情况下，chunk 的结果是正确的。

### Layernorm

```python
class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
```

GPU 上 torch.rsqrt 常是单指令或融合更好的路径，数值上避免先 sqrt 再除带来的额外舍入步骤（实现细节因硬件而异）。

rms_forward：fp16/bf16 上直接平方、求均值、再 rsqrt 容易溢出或下溢；先在 fp32 算统计量再回写，是混合精度训练/推理的常见模式。

add_rms_forward：融合残差的 Pre-Norm 变体。decoder layer 的输入为 positions, hidden_states, residuals, 这与 vllm 的实现一致，后两者拆分开来以便加速运算，它们是上一个 block 的线性层输出和残差流。add_rms_forward 实际返回 rms(x + residual) 和 x + residual。

```python
if residual is None:
    hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)
```

### Qwen3

nano-vllm 中 Qwen3ForCausalLM.forward 只返回 最后一层隐状态，与部分框架「forward 直接返回 logits」不同，logits 由 compute_logits 单独调用计算得到。批量时由 scheduler 统一调度 compute_logits，利于与采样器、CUDA Graph 等模块解耦。注意最后的隐层输出通常要过一个 rmsnorm。

GQA 与张量并行设定，这里并没有调整 k/v 的 shape 就传入 self.attn，flash_attn 可以直接 GQA ？？

```python
## model/qwen3.py
class Qwen3Attention(nn.Module):

    def __init__():
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = getattr(config, 'head_dim', hidden_size // config.num_attention_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5  ## softmax

        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output

class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x

## layers/activation.py
class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y        
```

Qwen3 系列在「无 bias」配置下对 Q、K 做 per-head 归一化，注意缩放粒度是每个头。

nano-vllm 将多个线性层 融合 成 qkv_proj 与 gate_up_proj。加载时根据 Qwen3ForCausalLM.packed_modules_mapping，告诉 loader 如何把 HF 张量 拼接/切片 到融合层。好处：一次 GEMM、更好利用 Tensor Core、减少权重碎片。

输入嵌入与 输出 logits 投影 共享权重矩阵。

```python
if config.tie_word_embeddings:
    self.lm_head.weight.data = self.model.embed_tokens.weight.data
```

## PagedAttention

### kv cache 分配

ModelRunner.allocate_kv_cache

```python
def allocate_kv_cache(self):
    config = self.config
    hf_config = config.hf_config
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
    config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
    assert config.num_kvcache_blocks > 0
    self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

显存余量：total * gpu_memory_utilization - used - peak + current
- mem_get_info：当前设备「空闲/总」显存。
- used = total - free：非空闲部分（含框架缓存等，语义以 CUDA 运行时为准）。
- peak / current：分配器统计的峰值与当前分配，用于修正「warmup 已分配但未必常驻」等差异。

按层绑定到 Attention。k/v cache 可以做量化。

### BlockManager

PagedAttention的核心是将KV缓存不再存储于连续空间，而是分割成一个个固定大小的物理块。每个物理块可以存储固定数量token的K和V状态。

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id    # 物理块的唯一标识符
        self.ref_count = 0          # 引用计数
        self.hash = -1             
        self.token_ids = []   
    
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1      
```

hash_to_block_id 是全局哈希表，从 内容哈希 映射到 物理块 id，实现「若见过同一段 token 序列且块内容一致，则复用物理块」。

compute_hash 中，prefix 通常为 前一块的哈希（链式）。这样整块内容不仅依赖本块 token，还依赖上文，降低「不同上下文碰巧 token 相同」的碰撞风险。xxHash 是一种极快的非加密哈希算法。

allocate 为整条序列初次占位

- 对序列的 每一个块索引 i，取出 seq.block(i) 的 token_ids。
- 仅当 len(token_ids) == block_size 时，才计算链式哈希 h；否则 h = -1（未满块不参与全局复用索引）。
- 查 hash_to_block_id.get(h)；若不存在或块内 token_ids 不一致 → cache miss，从 free_block_ids 取新物理块。
- cache hit 时：增加 seq 的 num_cached_tokens，并增加引用计数（共享块）。
- 最后把选中的 block_id 追加到 seq.block_table。

deallocate 释放序列，只有当一个块的ref_count降为0，此时它才会被真正归还到空闲列表。

can_append 表示当 长度对块大小取模为 1 时，下一步 may_append 需要新物理块，要求 free_block_ids 至少 1 个。否则不需要为新块预留，>= 0 恒成立。

may_append 

- 序列长度对块大小取模为 1 时，需要新物理块，从 free_block_ids 取新物理块。
- 序列长度对块大小取模为 0 时，说明最后一个块是满的，需要计算哈希，并更新 hash_to_block_id。


### Seqence

推理引擎需要跟踪每条序列的以下信息：

- 状态：这条序列当前是等待调度、正在执行、还是已经完成？
- Token 列表：包含原始 prompt 的 token 和已经生成的 completion token。
- KV Cache 映射：序列的 Key/Value 缓存数据分布在哪些物理 block 中？
- 采样参数：温度（temperature）、最大生成长度（max_tokens）、是否忽略 EOS 等。
- 缓存信息：有多少 token 的 KV Cache 已经被计算并缓存？

如果不用统一的数据结构管理这些信息，Scheduler 和 ModelRunner 之间就无法高效协作。

```
用户请求（文本）
  ↓  tokenize
LLMEngine.add_request()
  ↓  创建 Sequence 对象
Scheduler.add(seq)         ← seq 进入 waiting 队列
  ↓
Scheduler.schedule()       ← seq 被调度，分配 KV Cache block
  ↓
ModelRunner.run(seqs, ...)  ← 读取 seq 的 token_ids / block_table
  ↓
Scheduler.postprocess()     ← 追加新 token，判断是否结束
  ↓
seq.status == FINISHED → 返回结果给用户
```

```python
from itertools import count

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0    # tokens that don't need prefill
        self.num_scheduled_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.token_ids if self.num_completion_tokens == 0 or self.num_cached_tokens < self.num_tokens else self.last_token
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

```

| 状态 | 含义 | 进入条件 | 退出条件 |
|------|------|---------|---------|
| `WAITING` | 在等待队列中，尚未被调度 | 新创建 / 被抢占（preempt） | 被 `schedule()` 选中并分配 block |
| `RUNNING` | 正在参与推理（prefill 或 decode） | `schedule()` 分配 block 后 | 生成完毕（EOS / max_tokens） / 被抢占 |
| `FINISHED` | 生成完毕，等待回收 | 命中 EOS 或达到 max_tokens | 终态，不再变化 |

vLLM 的正式版本中有 SWAPPED 状态，用于区分被交换到 CPU 内存的序列，而非直接丢弃。nano-vllm 简化了这一设计，抢占后的序列直接回到 WAITING。被抢占的序列通过 appendleft 放到 waiting 队列头部，保证它们优先被重新调度。

设定 block_size 为类变量，因为所有序列共享同一套物理 block 管理系统，block 大小必须一致。使用 itertools.count() 生成全局唯一 seq_id，这比 UUID 更轻量，且在单进程环境下足够用。在分布式场景中，由于 Sequence 只在 rank 0 创建，不存在 ID 冲突问题。

block_table 是 Sequence 与 KV Cache 物理存储之间的桥梁。它是一个整数列表，每个元素是一个物理 block 的索引号。

```
假设 block_size = 256, 序列有 600 个 token:

block_table = [3, 7, 15]  # 3个物理block

Block 3: token 0-255 的 KV Cache
Block 7: token 256-511 的 KV Cache
Block 15: token 512-599 的 KV Cache（未满）
```

decode 阶段只需要最后一个 token 作为输入，因此 维护 last_token。 last_block_num_tokens 计算最后一个 block 中有多少个 token。这个信息对 decode 阶段的 slot_mapping 计算至关重要。

```python
slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
```

注意 append_token 不会修改 block_table。block 的分配是 BlockManager 的职责，在 Scheduler.schedule() 中通过 block_manager.may_append() 完成。

num_cached_tokens 记录了序列中有多少 token 的 KV Cache 已经可用（来自前缀缓存或之前的计算）。当序列被抢占（preempt）时，它的 KV Cache 被释放，num_cached_tokens 会被重置为 0（由 BlockManager 处理）。重新调度时需要重新计算所有 token 的 KV Cache。

改进参考思路：

- 添加 SWAPPED 状态，支持将 KV Cache 交换到 CPU 内存（而非直接丢弃）
- 支持 beam search，添加 parent_seq_id 和 fork() 方法
- 添加 LoRA adapter ID，支持多 LoRA 推理
- 添加请求级别的 priority 字段，支持优先级调度
- 添加 arrival_time，支持基于等待时间的公平调度
- 支持 stop_sequences（停止词列表），而不仅是 EOS

#### Sequence 与其他模块的交互

与 Scheduler 的交互
``` python
# Scheduler 创建时机
scheduler.add(seq)  # seq 加入 waiting 队列

# Scheduler 调度时
scheduler.schedule()
    → seq.status = SequenceStatus.RUNNING
    → block_manager.allocate(seq)  # 填充 seq.block_table

# Scheduler 后处理时
scheduler.postprocess(seqs, token_ids)
    → seq.append_token(token_id)
    → 检查 seq.num_completion_tokens == seq.max_tokens
```

与 ModelRunner 的交互
```python
# Prefill 准备
model_runner.prepare_prefill(seqs)
    → 读取 seq.token_ids[seq.num_cached_tokens:]  # 跳过已缓存的 token
    → 读取 seq.block_table  # 获取 KV Cache 写入位置

# Decode 准备
model_runner.prepare_decode(seqs)
    → 读取 seq.last_token  # 只需最后一个 token
    → 读取 seq.block_table[-1]  # 最后一个 block
    → 读取 seq.last_block_num_tokens  # 计算 slot_mapping
```

与 BlockManager 的交互
```python
# 分配 block
block_manager.allocate(seq)
    → 计算 seq.num_blocks
    → 分配物理 block
    → 填充 seq.block_table

# 释放 block
block_manager.deallocate(seq)
    → 回收 seq.block_table 中的物理 block
    → seq.block_table = []

# 追加 block
block_manager.may_append(seq)
    → 如果最后一个 block 满了，分配新 block
    → 追加到 seq.block_table
```


### Scheduler 

调度器的核心职责：

- 决定每一步执行哪些序列（选取 + 排序）
- 区分 prefill 和 decode 阶段（不同阶段的资源特征截然不同）
- 管理 KV Cache 资源（通过 BlockManager 分配 / 释放物理 block）
- 处理资源不足（抢占低优先级序列，释放 block 给高优先级序列）
- 后处理（追加 token、判断终止条件、清理已完成序列）

```python
class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0 or (not seq.block_table and not self.block_manager.can_allocate(seq)):    # no budget
                break
            if remaining < num_tokens and scheduled_seqs:    # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            if seq.num_scheduled_tokens == num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
            num_batched_tokens += seq.num_scheduled_tokens
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            if is_prefill:
                seq.num_cached_tokens = min(seq.num_cached_tokens + seq.num_scheduled_tokens, seq.num_tokens)
                if seq.num_cached_tokens < seq.num_tokens or seq.num_completion_tokens > 0:    # chunked prefill or re prefill after preemption
                    seq.num_scheduled_tokens = 0
                    continue
            seq.append_token(token_id)
            seq.num_cached_tokens += 1
            seq.num_scheduled_tokens = 0
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
```

**BlockManager 在 Scheduler 中初始化**，负责分配和回收所有 seq 的 block。

schedule() 分两个阶段执行。第一阶段尝试从 waiting 队列调度 prefill 序列：按 FCFS（先来先服务） 顺序逐个取出，检查 token 数限制和 block 可用性，通过检查则分配 block、修改状态为 RUNNING、移入 running 队列。如果成功调度了 prefill 序列，直接返回。第二阶段处理 decode：从 running 队列逐个取出，检查是否能追加新 token（可能需要新 block），如果 block 不足则通过 LIFO(Last In First Out) 策略抢占其他序列释放资源。

prefill 优先:

- 用户体验：新请求需要先完成 prefill 才能开始生成，prefill 越快，用户等待首个 token 的时间越短（Time To First Token, TTFT）。 decode 优先则每个输出 token 的生成时间（Time Per Output Token, TPOT）更短。TTFT 对用户体验影响更大，用户更在意"什么时候开始有回复"而非"回复速度有多快"。
- 计算效率：prefill 是计算密集型，可以高效利用 GPU 算力
- 避免饥饿：如果 decode 优先，新请求可能长时间无法得到处理

prefill break 条件：单步总 token 上限 max_num_batched_tokens；can_allocate(seq) 物理 KV block 是否够整条序列当前所需块数，这里没有考虑缓存的 token 数，因为只有在 allocate 中才能确定前缀缓存的 block 数，这里只判断一般无前缀缓存的情况。

decode 阶段，如果 KV 无法再 append（例如缺 block）时，通过 LIFO running.pop() **抢占队尾序列**，直到能 append 或只能抢占当前 seq 并 break。被抢占的序列状态设为 WAITING，block 回收，**重新入队 waiting 队列队首**，确保其在下一轮调度中优先被重新调度，避免某个序列被反复抢占却永远无法完成的情况。

最后加入的序列可能才刚开始生成，抢占它浪费的计算量最少。被抢占的序列需要完全重新计算。在 vLLM 的完整版本中，还有 swap 策略：将 KV Cache 从 GPU 交换到 CPU 内存，避免重复计算。

通过 can_append(seq) 检查序列的最后一个 block 是否已满，如果满了是否有 1 个空闲 block。通过 may_append 条件性分配。

postprocess 对于本步参与 decode 的每个序列，追加新 token，遇到 EOS token 且未设置 ignore_eos，或达到 max_tokens，则标记为 FINISHED，block 回收，从 running 队列移除。remove(seq) 是因为完成的序列不一定在队列头部。

调度器与 BlockManager 的协作
```python
# 调度器不直接管理物理 block，而是委托给 BlockManager

# Prefill 时：
self.block_manager.can_allocate(seq)   # 询问：有足够 block 给这个序列吗？
self.block_manager.allocate(seq)       # 执行：分配 block 并填充 seq.block_table

# Decode 时：
self.block_manager.can_append(seq)     # 询问：能追加一个 token 吗？
self.block_manager.may_append(seq)     # 执行：如果需要，分配新 block

# 抢占/完成时：
self.block_manager.deallocate(seq)     # 执行：释放该序列的所有 block
```

改进 nano-vllm 的调度器思路：

- 实现 chunked prefill，平衡 TTFT 和 TPOT
- 添加 swap 策略减少抢占浪费
- 支持优先级调度（priority queue）
- 支持 prefix-aware 调度，共享前缀的请求一起调度以最大化缓存命中
- 支持投机解码的两阶段调度
- 添加公平性保障（基于等待时间的优先级提升）

#### Chunked Prefill

Prefill 和 decode 的计算模式不同。prefill 处理多个连续 token，使用的是变长序列的注意力计算；decode 每个序列只处理一个 token，使用的是 KV Cache 加速的注意力计算。不混合执行简化了 ModelRunner 的实现和 CUDA kernel 的选择。实际上高性能系统如 vLLM 已支持混合执行以提高 GPU 利用率。

当 Prefill 处理长 prompt 时，Decode 被完全阻塞，已有序列无法产出新 token。用户感知到的现象是：正在生成的回复突然"卡住"了。长 prompt 的 prefill 可以被分成多个 chunk，与 decode 序列交错执行。这样既不会因为长 prompt 阻塞 decode 序列，又能保持较低的 TTFT。

```
Step 1: [A(prefill chunk1 512 tokens), B(decode), C(decode)]
Step 2: [A(prefill chunk2 512 tokens), B(decode), C(decode)]
Step 3: [A(prefill chunk3 512 tokens), B(decode), C(decode)]
Step 4: [A(prefill chunk4 464 tokens), B(decode), C(decode)]
```

nano-vllm 最新实现疑似添加 [chunked prefill](https://github.com/GeeeekExplorer/nano-vllm/commit/8d63a98c03805e54e9a422fd83fff7a4780c17dc)。

Chunked Prefill 的优点：

- Decode 不会被 Prefill 阻塞
- 更均匀的每步执行时间
- 更好的尾延迟（P99 latency）
Chunked Prefill 的代价：

- 实现复杂度增加
- 注意力核需要支持混合模式
- Prefill 的总耗时可能增加（因为分多步处理）


## 批处理

在 Decode 阶段，每一步只处理 1 个 token，但需要从显存中加载全部模型权重。批处理的核心思想：将多个请求的计算合并到一次 GPU 操作中。

### 静态批处理（Static Batching）

收集固定数量的请求组成一个 batch，所有请求同时开始 Prefill，所有请求同时进入 Decode。必须等 batch 中最慢的请求完成，整个 batch 才能结束，结束后才能开始处理下一个 batch。

问题：

- GPU 气泡（Bubble）：当 batch 中的请求生成长度差异较大时，先完成的请求对应的 GPU 计算资源被浪费。
- TTFT 延迟增大：所有请求必须凑齐一个 batch 才开始处理。如果第一个请求到达后，需要等待其他请求到达才能组成 batch，TTFT 会显著增加。
- 吞吐量受限：已完成的请求位置不能被新请求复用。整个 batch 的有效吞吐量取决于最慢的请求。
- 显存浪费：必须为 batch 中每个请求按最大可能生成长度预分配 KV Cache 空间。实际生成长度通常远小于最大值，造成大量显存浪费。

适用场景：
- 离线批量推理：所有请求已知，无需实时响应
- 固定长度生成：如摘要生成（输出长度可控）
- 教学/原型验证：实现简单，便于理解

### 连续批处理（Continuous Batching）

连续批处理由 Orca 论文（Yu et al., 2022）提出，vLLM 和 nano-vllm 都实现了这一策略。其核心思想可以总结为三点：

- 迭代级调度（Iteration-level Scheduling）：每一次前向传播（iteration）都重新决定哪些序列参与计算
- 即时加入与退出：新请求可以在任意 iteration 加入，完成的请求立即退出
- Prefill 与 Decode 分离：不同阶段的序列可以分开处理（或在 chunked-prefill 中混合处理）

|静态批处理的问题|连续批处理的解决方案|
|---|---|
|GPU 气泡|请求完成后立即退出，位置被新请求填入|
|TTFT 延迟大|新请求可以在下一个 iteration 立即做 prefill|
|吞吐量受限|GPU 始终保持满载，有效计算比例高|
|显存浪费|配合 PagedAttention 按需分配 KV Cache|

#### nano-vllm 中的连续批处理实现

```python
## engine/llm_engine.py
def generate(self, prompts, sampling_params, use_tqdm=True):
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)
    outputs = {}
    while not self.is_finished():
        output, num_tokens = self.step()
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids
    outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
    outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
    return outputs

def step(self):
    seqs, is_prefill = self.scheduler.schedule()
    num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
    token_ids = self.model_runner.call("run", seqs, is_prefill)  ## 多进程调用
    self.scheduler.postprocess(seqs, token_ids, is_prefill)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```

- `generate`：添加请求并不断调用 `step`，直到所有请求完成
- `step`：先调度、再推理、再后处理更新序列状态，返回已完成序列的输出和 token 数

即时加入：每次 schedule() 调用，只要 waiting 非空且资源允许，就立即处理新请求，新请求不需要等待当前 batch 结束，多个新请求可以在同一步中一起做 prefill。

动态 batch 组合：每步 decode 时，从 running 队列中选取序列组成 batch，如果资源不足，通过抢占释放 block，batch 的组成在每一步都可能不同。

即时退出：一旦序列生成了 EOS 或达到 max_tokens，立即释放资源，不需要等待 batch 中其他序列完成，释放的 block 在下一步就可以被新请求使用。

Prefill 优先：新请求的 TTFT 被最小化，但代价是已有请求的 TPOT 可能增加。

避免 Padding：Prefill 阶段使用 flash_attn_varlen_func，所有序列的 token 拼接成一个一维张量，通过 cu_seqlens 记录每个序列的边界。FlashAttention 的变长接口能够处理这种格式，无需 Padding。Decode 阶段每个序列只有 1 个 token，天然等长，无需 Padding。

vLLM 在 Orca 的基础上引入了 PagedAttention 和更完善的调度机制：

| 特性 | vLLM | nano-vllm |
|------|------|-----------|
| 连续批处理 | ✅ | ✅ |
| PagedAttention | ✅ | ✅ |
| Chunked Prefill | ✅ | ❌（存疑） |
| Swap（CPU↔GPU） | ✅ | ❌（用 recompute） |
| 前缀缓存 | ✅ | ✅ |
| Beam Search | ✅ | ❌ |
| SequenceGroup | ✅ | ❌ |
| Speculative Decoding | ✅ | ❌ |
| 多节点推理 | ✅ | ❌ |

#### 超参数

- max_num_seqs：限制 batch 中最多包含多少个序列。主要受限于 KV Cache 总量和 CUDA Graph 的 batch size 限制。
- max_num_batched_tokens：限制单步 prefill 中最多处理多少个 token。主要受限于 GPU 显存（激活值大小）和计算耗时（影响 TTFT）。

CUDA Graph 可以大幅减少 GPU kernel launch 开销，但要求输入形状固定。在连续批处理中，每步的 batch size 可能不同，这与 CUDA Graph 的要求矛盾。

nano-vllm 的解决方案：预先为多种 batch size 捕获 CUDA Graph，运行时选择最接近的。Decode 阶段的 batch size 通常在 1~max_num_seqs 之间变化。通过预捕获，可以覆盖大部分情况。

```python
# model_runner.py
def capture_cudagraph(self):
    for bs in capture_batch_sizes():   # 枚举可能的 batch size
        self.graph_runners[bs] = CUDAGraphRunner(self.model, bs)
```

max_num_seqs 的最优值取决于多个因素：

- KV Cache 总量：num_kvcache_blocks 除以每个序列平均需要的 block 数，得到上限
- GPU 算力：batch 过大会使 Decode 从 memory-bound 变为 compute-bound，TPOT 增加
- CUDA Graph 限制：需要为所有可能的 batch size 预捕获 graph
- 延迟 SLA：TPOT 必须满足服务级别协议

一般做法是通过 profiling 确定：从小到大增加 max_num_seqs，观察吞吐量和 TPOT，选择吞吐量饱和且 TPOT 满足 SLA 的点。


## ModelRuner

engine/model_runner.py  ModelRunner 负责将调度器选出的序列转化为 GPU 可执行的张量输入，驱动模型前向传播，并返回采样结果。

### 初始化

```python
class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier() ## 同步
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()
```

初始化 NCCL 必须在加载模型之前完成，因为张量并行的线性层（如 ColumnParallelLinear）在初始化时需要知道 tp_rank 和 tp_size 来确定权重分片方式。

加载模型后进行 warmup，分配 KV Cache，捕获 CUDA Graph，非 rank-0 永久阻塞。

### allocate kv cache

```python
def allocate_kv_cache(self):
    config = self.config
    hf_config = config.hf_config
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
    config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
    assert config.num_kvcache_blocks > 0
    self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

KV Cache 张量的形状解读
```
kv_cache 的 6 维张量：
  维度 0: [K, V]                     → 2
  维度 1: [layer_0, ..., layer_N]    → num_hidden_layers
  维度 2: [block_0, ..., block_M]    → num_kvcache_blocks
  维度 3: [token_0, ..., token_B]    → block_size
  维度 4: [head_0, ..., head_H]      → num_kv_heads
  维度 5: [dim_0, ..., dim_D]        → head_dim
```

### prepare_prefill

```python
def prepare_prefill(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    block_tables = None
    for seq in seqs:
        seqlen = len(seq)
        start = min(seq.num_cached_tokens, seqlen - 1)
        seqlen_q = seq.num_scheduled_tokens   # Q 的长度（需计算的部分）
        seqlen_k = seqlen                     # K 的长度（包含缓存部分）
        end = start + seqlen_q
        input_ids.extend(seq[start:end])      # 构造 input_ids：只包含未缓存的 token
        positions.extend(range(start, end))   # 构造 positions：从 num_cached_tokens 开始
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)  # 更新累积序列长度
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)
        if not seq.block_table:    # warmup   没分配 blocks 就跳过写入 slot_mapping
            continue

        # 构造 slot_mapping
        start_block = start // self.block_size
        end_block = (end + self.block_size - 1) // self.block_size
        for i in range(start_block, end_block):
            slot_start = seq.block_table[i] * self.block_size
            if i == start_block:
                slot_start += start % self.block_size
            if i != end_block - 1:
                slot_end = seq.block_table[i] * self.block_size + self.block_size
            else:
                slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
            slot_mapping.extend(range(slot_start, slot_end))
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
        block_tables = self.prepare_block_tables(seqs)
    input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    ...
    set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
    return input_ids, positions
```

cu_seqlens（cumulative sequence lengths）是 FlashAttention 变长接口的核心输入，用于标记每个序列在拼接后张量中的边界。

cu_seqlens_q 和 cu_seqlens_k 可能不同，当存在前缀缓存命中时：缓存部分的 KV 已经在 KV Cache 中，不需要重新计算 Q，但注意力仍需要与它们交互。此时 FlashAttention 走带 block_table 的分页注意力路径。还有可能是 Encoder-Decoder 架构的 交叉注意力？？

slot_mapping 将每个 token 映射到 KV Cache 中的物理位置（slot），在 Attention 层的前向传播中，新计算的 K 和 V 会被写入 slot_mapping 指定的位置。

构造完所有输入后，将元数据设置到全局上下文中，Attention 层通过 get_context() 获取这些信息，决定使用哪种注意力计算路径。

### prepare_decode

```python
def prepare_block_tables(self, seqs: list[Sequence]):
    max_len = max(len(seq.block_table) for seq in seqs)
    block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
    block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    return block_tables

def prepare_decode(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []
    for seq in seqs:
        input_ids.append(seq.last_token) # 输入只有 1 个 token：最后生成的 token
        positions.append(len(seq) - 1)   # 位置是序列总长度 - 1
        context_lens.append(len(seq))    
        # slot_mapping：新 token 写入最后一个 block 的下一个位置
        slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
    input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    block_tables = self.prepare_block_tables(seqs)
    set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
    return input_ids, positions
```

在 LLMEngine.step() 中，先 Scheduler.schedule() 后 modelruner.run()，这意味着 prepare_decode 在 may_append 之后被调用，最新的 token 在上一轮 run 后生成，如果超出了最后 block 的范围，会分配新的 block，并更新 block_table，prepare_decode 中的 slot_mapping 不会越界。

所有序列的 block_table 会被合并为一个二维张量，padding 值为 -1，传递给 Attention 核。


### run model

```python
## layers/sampler.py
class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        ## 采用 Gumbel-Softmax 采样
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens

## engine/model_runner.py
@torch.inference_mode()
def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    reset_context()
    return token_ids
```

Eager 模式：直接调用 PyTorch 模型前向传播。每次都会重新发起所有 CUDA kernel，有 CPU→GPU launch 开销。Prefill 阶段始终使用 Eager 模式，因为 Prefill 的 batch 形状（总 token 数）变化很大，无法枚举所有可能的输入形状，难以预先捕获 CUDA Graph。Prefill 的 kernel 执行时间远大于 launch 开销，CUDA Graph 的收益微小。变长注意力（flash_attn_varlen_func）对 CUDA Graph 的支持有限。

CUDA Graph 模式：Decode 阶段使用。预先捕获好的 CUDA Graph 包含了所有 kernel 的调用序列，运行时只需一次 graph.replay()，消除了 CPU→GPU 的 launch 开销。对于 Decode 这种单 token 计算量小但 kernel 数量多的场景，CUDA Graph 的加速效果显著。

只有 rank 0 做采样。最终的 logits 在 lm_head（VocabParallelEmbedding）的 forward 中通过 all_gather 汇总到所有 rank。但采样只需要做一次——由 rank 0 执行，然后通过 SharedMemory 或其他机制将结果分发给其他 rank。这避免了重复计算号和不一致的随机采样结果（不同 rank 的随机种子可能不同）。

流程图
```
run(seqs, is_prefill)
    │
    ├── is_prefill == True
    │       └── prepare_prefill(seqs)
    │               ├── 构造 input_ids（拼接所有序列的未缓存 token）
    │               ├── 构造 positions（位置编码索引）
    │               ├── 构造 cu_seqlens_q, cu_seqlens_k（序列边界）
    │               ├── 构造 slot_mapping（KV Cache 写入位置）
    │               └── set_context(is_prefill=True, ...)
    │
    ├── is_prefill == False
    │       └── prepare_decode(seqs)
    │               ├── 构造 input_ids（每序列 1 个 last_token）
    │               ├── 构造 positions（每序列 1 个位置）
    │               ├── 构造 slot_mapping（1 个新 slot）
    │               ├── 构造 context_lens, block_tables
    │               └── set_context(is_prefill=False, ...)
    │
    ├── run_model(input_ids, positions, is_prefill)
    │       ├── Prefill/Eager → self.model(input_ids, positions)
    │       └── Decode/Graph  → self.graph_runners[bs].run(...)
    │       └── 返回 logits: [num_tokens, vocab_size]
    │
    └── sampler(logits, temperatures)
            ├── logits / temperature
            ├── softmax → 概率分布
            ├── multinomial 采样
            └── 返回 token_ids: [num_seqs]
```

### 多进程通信（SharedMemory）

```python
def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

def read_shm(self):
    assert self.world_size > 1 and self.rank > 0
    self.event.wait()
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()
    return method_name, args

def write_shm(self, method_name, *args):
    assert self.world_size > 1 and self.rank == 0
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.event:
        event.set()

def call(self, method_name, *args):
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)
    method = getattr(self, method_name, None)
    return method(*args)
```

在张量并行（TP > 1）中，每个 GPU 运行一个独立进程。调度器运行在 rank 0 的进程中，它需要将序列信息传递给其他 rank 的 ModelRunner。

Rank 0 将方法名和参数序列化写入 SharedMemory，通过 Event 通知其他 rank，所有 rank 调用相同的方法（如 run(seqs, is_prefill)），各 rank 独立执行前向传播（张量并行自动处理权重分片和 AllReduce）。

Sequence 的序列化在通过 SharedMemory 传递时，Sequence 对象需要被 pickle.dumps() 序列化。这就是 Sequence.__getstate__ 优化的用武之地。

### Warmup 与显存管理

```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
    seq_len = min(max_num_batched_tokens, max_model_len)
    num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
    seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
    for seq in seqs:
        seq.num_scheduled_tokens = seq_len
    self.run(seqs, True)
    torch.cuda.empty_cache()
```

Warmup 的目的：

- Triton Kernel 编译：首次执行时，Triton 会 JIT 编译自定义的注意力核、LayerNorm 核等。编译耗时可达数秒，warmup 将这个延迟前置。
- CUDA 内存池初始化：PyTorch 的 CUDA 内存分配器在首次分配时会建立内存池。warmup 后，后续分配会从池中获取，更快。
- cuBLAS 句柄初始化：矩阵乘法库 cuBLAS 在首次调用时需要初始化，warmup 将这个开销前置。
- 确定显存峰值：warmup 后调用 torch.cuda.max_memory_allocated() 可以得到前向传播的显存峰值，用于后续 KV Cache 的分配计算。

## 张量并行

为什么需要张量并行：显存瓶颈/计算瓶颈。

| 策略 | 切分维度 | 通信模式 | 适用场景 |
|------|---------|---------|---------|
| **数据并行（DP）** | 按 batch 切分 | AllReduce 梯度 | 训练 |
| **张量并行（TP）** | 按权重矩阵切分 | AllReduce 激活值 | 推理（同节点） |
| **流水线并行（PP）** | 按层切分 | 点对点发送激活值 | 训练 + 推理（跨节点） |
| **序列并行（SP）** | 按序列长度切分 | AllGather/ReduceScatter | 长序列场景 |

推理场景下，TP 相比其他策略有独特优势：

- 低延迟：所有 GPU 同时计算，一层的计算时间几乎是 ( 1/N )（N 为 GPU 数）。PP 则需要按层串行，延迟无法降低。
- 高利用率：每个 GPU 始终有活干。PP 在推理中存在"气泡"（等待上一层输出）。
- AllReduce 高效：同一节点内的 GPU 通过 NVLink 连接，AllReduce 带宽高达 600 GB/s。
- 实现相对简单：只需修改 Linear 层的权重划分方式。

### 列并行（ColumnParallelLinear）

```python
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, tp_dim=0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
```

ColumnParallelLinear 将权重矩阵按列切分，每个 GPU 负责一部分列。forward 时，直接计算即可，如果没有行并行（RowParallelLinear），则还需要 AllGather 激活值。

### 行并行（RowParallelLinear）

```python
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
```

RowParallelLinear 将权重矩阵按行切分，每个 GPU 负责一部分行。forward 时，计算结果需要 AllReduce。

Bias 只在 rank 0 加：因为 AllReduce 会对所有 GPU 的结果求和。如果每个 GPU 都加 bias，最终结果会被加了 N 次 bias。

每层 Transformer 需要 2 次 AllReduce（一次来自 Attention 的 o_proj，一次来自 FFN 的 down_proj）。

单次 AllReduce 的通信量： $\text{通信量} = 2 \times (N_{\text{tp}} - 1) / N_{\text{tp}} \times M \times D \times \text{sizeof(dtype)}$

对于 M=1（Decode 单 token）、D=4096、FP16、TP=4： $ = 2 \times 3/4 \times 1 \times 4096 \times 2 = 12 \text{ KB} $

非常小！ 这就是为什么 TP 在同节点内（NVLink）效率极高。

对于 M=1000（Prefill 长序列）： $ = 2 \times 3/4 \times 1000 \times 4096 \times 2 = 12 \text{ MB} $

仍然可以接受（NVLink 的单向带宽 300+ GB/s，传输 12 MB 仅需 ~40 μs）。

### Merged

#### QKVParallelLinear 

为了减少 kernel launch 开销，通常将 Q、K、V 的权重合并为一个大矩阵，一次矩阵乘法同时得到 Q、K、V。

当使用 GQA（Grouped Query Attention）时，Q 的头数（如 32）和 KV 的头数（如 4）不同，权重矩阵的 Q 部分和 KV 部分大小不一致。加上张量并行的切分，权重加载时需要特殊的分片逻辑。

```python
def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
    param_data = param.data
    assert loaded_shard_id in ["q", "k", "v"]
    if loaded_shard_id == "q":
        shard_size = self.num_heads * self.head_size
        shard_offset = 0
    elif loaded_shard_id == "k":
        shard_size = self.num_kv_heads * self.head_size
        shard_offset = self.num_heads * self.head_size
    else:
        shard_size = self.num_kv_heads * self.head_size
        shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
    param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
    loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
    param_data.copy_(loaded_weight)
```

每个 GPU 的权重布局：
```
GPU 的 QKV 权重：[Q_shard | K_shard | V_shard]
                  ↑         ↑         ↑
           num_heads×D  num_kv×D  num_kv×D
           (16×128)     (2×128)   (2×128)
```

前向传播得到 QKV 的拼接结果后，需要拆分为 Q、K、V 三个张量：
```python
# layers/attention.py
qkv = self.qkv_proj(x)  # [M, (num_heads + 2×num_kv_heads) × head_dim]
q, k, v = qkv.split([
    self.num_heads * self.head_dim,
    self.num_kv_heads * self.head_dim,
    self.num_kv_heads * self.head_dim
], dim=-1)
```

#### MergedColumnParallelLinear

在 Transformer FFN 中，通常有两个列并行的线性层需要同时计算（如 Qwen3 的 gate_proj 和 up_proj）。为了减少 kernel launch 开销，将它们合并为一个大矩阵乘法。

### NCCL 

nano-vllm 使用 NCCL（NVIDIA Collective Communication Library）作为通信后端。选择原因：

- 专为 NVIDIA GPU 优化：自动利用 NVLink、NVSwitch 等硬件特性
- 支持所有集合通信原语：AllReduce、AllGather、ReduceScatter 等
- GPU Direct：数据直接在 GPU 间传输，不经过 CPU
- Ring 和 Tree 算法自适应：根据拓扑和数据量自动选择最优算法
- 与 PyTorch 深度集成：torch.distributed 原生支持 NCCL 后端

nano-vllm 使用 Python 的 multiprocessing 启动多个进程：

```python
## engine/llm_engine.py

import torch.multiprocessing as mp

class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)
    
    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
```

Rank 0 在主进程中运行，负责调度和采样，Rank 1~N 在子进程中运行，创建 ModelRunner 后进入 loop() 等待指令。

