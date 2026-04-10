# 量化

```shell
cd ./quant4bit
python setup.py install
```

## GPTQ 实现

基于校准数据估计输入 Hessian，用二阶近似在逐列量化时做误差补偿（通过 \(H^{-1}\) 将当前量化误差传播到未量化列），从而在低比特下显著降低输出误差。**

### 1. 核心原理

对某一层线性变换 \(y = Wx\)，GPTQ不是只看权重本身做就地四舍五入，而是最小化输出误差近似：


$\min_{\hat W} \ \mathbb{E}\|W x - \hat W x\|_2^2$

理论源自于OBS和OBD，修剪参数后对未修剪参数进行补偿，在量化中，参数变化为$w_i - quatnt(w_i)$，故补偿量为

$\Delta_i = -\frac{w_i - quatnt(w_i)}{H^{-1}_{ii}} \cdot H^{-1}_{:,i}$

GPTQ 采用顺序量化（逐列），每列量化后，用 Hessian 逆做误差补偿，再量化下一列，直到所有列量化完成。这样的好处在于
不用更新 Hessian，反之，采用 $(quatnt(w_i) - w_i)^2 / {H^{-1}_{ii}}$ 估计重要性，并对低重要性参数先量化，则需要更新。
从GPTQ结果来看，前者的计算量大大减少，效果上不差于后者。

### 2. Hessian 的构建：`add_batch`

```python
# Hessian H = 2 X XT + λ I
self.H += inp.matmul(inp.t())
```

- 对每层收集校准样本的输入 `inp`；
- 统一展开成二维“特征维 × 样本数”形式；
- `H` 的维度是 `[columns, columns]`，其中 `columns` 是该层的输入维度大小；
- 在线更新，期望是 $2/n * XX^T$。


### 3. 量化前预处理：数值稳定 + 可选重排

在 `fasterquant` 里：

1. **dead 列处理**  
   ```python
   dead = torch.diag(H) == 0
   H[dead, dead] = 1
   W[:, dead] = 0
   ```
   防止奇异维度导致数值问题。

2. **actorder（可选）**  
   ```python
   perm = torch.argsort(torch.diag(H), descending=True)
   ```
   按 Hessian 对角线从大到小排序，优先量化 $H^{-1}_{ii}$ 小的列，通常能降误差。

3. **damping**  
   ```python
   damp = percdamp * mean(diag(H))
   H[diag, diag] += damp
   ```
   防止 Hessian 病态，提升数值稳定性。

### 4. 逐列量化 + 误差补偿

block 内层循环：

```python
w = W1[:, i]
d = Hinv1[i, i]
q = quantize(w)
err1 = (w - q) / d
W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
```

损失估计近似每列量化引起的代价评估（最终汇总成 `error`）。

```python
Losses1[:, i] = (w - q)**2 / d**2
```

block 间外层循环：

- 按 `blocksize` 切块；
- 块内做逐列补偿（`W1[:, i:] -= ...`）；
- 块间再做一次跨块更新（懒更新）：
  ```python
  W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
  ```

`groupsize` 与 `blocksize` 不同，逻辑是量化器参数共享策略（每组一套 scale/zero），影响 `quantize(w)` 这一步的离散化粒度与灵活性。注意，每组的共享参数由该组所有参数没量化时确定，组内每列计算量化反量化引入的误差时不修改量化参数。**组内共享指的是同一输出通道的权重，不同输出通道的量化参数通常不一致**，即默认 per_channel 的量化方式，量化参数共计 $2 * out_f * ceil(in_f // groupsize)$。

`groupsize = -1` 即所有列共享一套参数。

```python
if groupsize != -1:
    if (i1 + i) % groupsize == 0:
        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

    if ((i1 + i) // groupsize) - now_idx == -1: # 本组量化共享参数，now_idx 用来跟踪已经缓存了多少组参数
        scale.append(self.quantizer.scale)
        zero.append(self.quantizer.zero)
        now_idx += 1
```

### 5. 在 LLaMA 中的执行流程（`llama_sequential`）

**layer-wise GPTQ** 的标准实践：

1. 先用 `Catcher` 截获第一层输入，拿到校准激活；
2. 对每一层：
   - 给层内 Linear 子模块挂 hook，收集输入并累计 Hessian；
   - 运行 `fasterquant` 得到量化参数；
   - 用量化后层重新前向，生成下一层校准输入；
3. 逐层推进，避免全模型同时处理，节省显存。

---

## 量化原理

### 均匀量化

均匀量化将浮点数离散到 [0, maxq] 的整数上，其中 `maxq = 2**bits - 1`。量化反量化代码如下

```python
q = clamp(round(x / scale) + zero, 0, maxq)
x_hat = scale * (q - zero)
```

对应公式：
$
\hat{x} = s\cdot \big(\mathrm{clip}(\mathrm{round}(x/s)+z,\,0,\,Q_{\max})-z\big)
$
- \(s\): scale
- \(z\): zero-point

### `find_params` 的主流程：求 scale / zero

- `perchannel=True`：每个输出通道（或每行）各自算一组参数  
- `perchannel=False`：全张量共享一组参数

```python
tmp = torch.zeros(x.shape[0], device=dev)
xmin = torch.minimum(x.min(1)[0], tmp)
xmax = torch.maximum(x.max(1)[0], tmp)
```

这样**保证 0 一定在区间里**，很关键，避免 zero-point 偏移异常。根据公示，零点 zero-point 是整数域中对应浮点数0的位置，
如果0不在xmin和xmax范围内，则 zero-point 也不在 [0, maxq] 范围内。

对称量化 `sym=True`，对于xmin < 0 的情况，量化的浮点数范围为 [-a, a]，其中 a=max(abs(xmin), xmax)，则 scale 为 2a / maxq。
对于xmin = 0 的情况，量化的浮点数范围为 [0, xmax]，则 scale 为 xmax / maxq，浮点数范围限定为[-xmax/2, xmax/2]？？

```python
xmax = torch.maximum(torch.abs(xmin), xmax)
tmp = xmin < 0
xmin[tmp] = -xmax[tmp]
scale = (xmax - xmin) / maxq
zero = torch.full_like(scale, (maxq + 1) / 2)
```

非对称量化 `sym=False`，默认方式。

```python
scale = (xmax - xmin) / maxq
zero = torch.round(-xmin / scale)
```

输出参数形状重排

- weight reshape 成 `[-1,1,1,...]`（按输出通道/行广播）
- activation 按 2D/3D/4D reshape 成可直接逐通道广播的形状


### MSE 网格搜索

当 `mse=True` 时，代码不直接使用初始 min/max，而是做网格搜索：

```python
for i in range(int(maxshrink * grid)):
    p = 1 - i / grid
    xmin1 = p * xmin
    xmax1 = p * xmax
    scale1 = (xmax1 - xmin1) / maxq
    ...
    err = sum(|q(x)-x|^norm)
    选误差最小的 scale/zero
```

#### 核心思想
- outlier会拉大量化区间，导致大量“普通值”分辨率变粗。
- 通过收缩区间（`p<1`），允许截断一部分极值，换取主体分布更小误差。

#### 搜索空间
- `grid=100`：把收缩系数 \(p\) 离散成 100 格（0~1）
- `maxshrink=.8`：最多搜索到 \(p=0.2\)
- 每个输出通道独立比较误差并更新最优参数

#### 误差度量

`norm=2.4` 默认不是严格 MSE(=2)，而是 $L_p$ 风格，工程上常用于平衡离群点与主体误差。

$\text{err} = \sum | \hat{x} - x |^{\text{norm}}$

### 推理

#### `pack`：把 FP 权重压成 4bit 紧凑表示

`QuantLinear` 存的不是 fp16 权重，而是：

- `qweight`：打包后的 4bit 权重（按 int32 存） (infeatures // 32 * self.bits, outfeatures), dtype=torch.int32
- `qzeros`：打包后的 zero-point（按 group、按输出通道存）(infeatures / self.groupsize, outfeatures // 32 * self.bits), dtype=torch.int32
- `scales`：每 group × out_channel 的缩放因子 (infeatures / self.groupsize, outfeatures), dtype=torch.float16
- `g_idx`：每个输入维属于哪个 group（支持 act-order 重排后非连续分组）

先把权重转成整数码字 `intweight`，代码核心：

```python
for idx in range(self.infeatures):
    intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]]).to(torch.int)[:, None])
intweight = torch.cat(intweight, dim=1)
intweight = intweight.t().contiguous()
intweight = intweight.to(torch.int32) 
```

这里 `scale_zeros = zeros * scales`，注意 `g_idx[idx]`：第 `idx` 个输入列使用其所属 group 的 `(scale, zero)`。

4bit 打包到 int32：`qweight`。4bit 下，一个 int32 能装 8 个值：

```python
qweight[row] |= intweight[j] << (4 * (j - i))
```

`qzeros` 也同样位打包，`zeros` 先做了：
```python
zeros -= 1
```
这是和底层 CUDA kernel 的约定，读取时会按约定还原？？
然后同样每 8 个 4bit zero-point 打成一个 int32，这里是在output channel维度上打包，可能是因为 input channel 维度上做了分组，再打包读取上繁琐？？

#### `forward`：根据 batch 大小选 kernel 做量化矩阵乘

输入 `x` 先展平成 `[N, infeatures]`，输出目标 `[N, outfeatures]`。

小 batch（`x.shape[0] <= kernelswitch`）走自定义 CUDA 向量-矩阵乘核：

- `act_order=True`：  
  ```python
  vecquant4matmul_g(..., g_idx, ...)
  ```
  使用 `g_idx`，适配 GPTQ act-order 后的分组映射（输入列顺序可能被打乱）。

- `act_order=False`：  
  ```python
  vecquant4matmul(..., groupsize, ...)
  ```
  假设规则连续分组，不需要 `g_idx` 映射，kernel 更简单。

通常针对 token-by-token / 小并发推理，延迟更低。kernelswitch = 1？？

大 batch 走 triton 实现。

```python
out = QuantLinearFunction.apply(...)
```

---

## 算子

### fuse

把量化后的算子组织成更少的 kernel / 更少中间张量，目标是降低延迟和显存带宽压力。

#### `fused_attn`：把 Q/K/V 三个 QuantLinear 融成一个 qkv_proj

`make_quant_attn` 里把原本 `q_proj`, `k_proj`, `v_proj` 拼成一个 `qkv_layer: QuantLinear(in, out_q+out_k+out_v)`

具体是对量化参数直接拼接：
- `qweight` 按输出通道维拼
- `qzeros`、`scales` 同样按输出通道拼
- `bias` 也拼
- `g_idx` 代码里也拼了（工程上常见；本质是给 fused kernel 用分组索引）

原来要做 3 次量化 matmul；现在 1 次拿到 QKV，大幅减少 kernel launch 和读输入 `hidden_states` 的次数。

RoPE 融合点（删除）
`QuantLlamaAttention.forward` 里：
1. `qkv_proj(hidden_states)` 得到 `[B, T, 3, H, D]`
2. 对前两块（Q,K）直接调用 `triton_rotate_half_` 原地做 RoPE
3. 再走 `scaled_dot_product_attention`

这比“先拆 q/k，再单独模块做 rotary”更省中间张量和访存。  

#### `fused_mlp`：把 gate_proj + up_proj 融成一个 Triton kernel

LLaMA 的 FFN 核心是：$\text{down\_proj}(\mathrm{silu}(xW_g)\odot (xW_u))$

这里把前半段两个投影融合为一个 kernel：
- 同时算 `A*B1`（gate）和 `A*B2`（up）
- kernel 内直接做 `silu(acc1) * acc2`
- 输出 intermediate，再喂给 `down_proj`

Triton kernel 里关键点，`fusedmatmul_248_kernel` 中：

- `B1/B2` 是 4bit 打包权重（int32）
- 每个 K block 内：
  - 按位解包 4bit (`>> shifter & maxq`)
  - 读取 group-wise `scales/zeros/g_idx`
  - 反量化 `(q - zero) * scale`
  - 两路分别累加 dot
- 最后 kernel 内完成 `silu(acc1) * acc2`

这样做避免了：
- 两次独立 matmul 的重复读 A
- 写回/读回两个中间激活

所以吞吐和带宽都更友好。

#### 精度/实现细节

- `zeros` 在 pack 时 `-1`，kernel 里再 `+1` 还原（和量化存储对应）
- 矩阵乘积累加用 fp32，输出转 fp16（常见折中）
- `autotune_warmup_fused` 预热不同 `(M,K,N)` 配置，减少首次推理抖动

然而 fuse 后似乎跑的更慢了？？

| LLaMA3-8B | Bits | group-size | memory(MiB) | TPOT(ms) | C4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| fp16 | - | -| 15580 | 46.9| 6.54 | 
| AutoAWQ | 4 | 128 | 5726 | 94.1 | 6.86 |
| GPTQ(me) | 4 | 128 | 5832 | 47.2 | 11.20 |
| GPTQ(me) | 4 | -1 | 5705 | 56.2 | 7.98 |
| GPTQ(me)+fuse | 4 | -1 | 5705 | 120.9 | 7.98 |


### kernel

#### CUDA kernel（`vecquant4matmul` / `_g`）

并行映射

- grid.z = batch
- grid.y = 输出列块（N 方向）
- grid.x = K 方向块（按 `BLOCKHEIGHT4=32` 个 packed row）
- 每个线程负责一个输出列 `w` 的部分 K 累加  
=> 因为 K 被分块，最终用 `atomicAdd` 写回 `mul[b,w]`

高速技巧

- `half2` 向量化：一次处理两个 fp16
- `deq2[256][8]` 查表：把一个 byte 里的两个4bit直接映射成 half2，避免重复位操作转 half
- `__hfma2` 做 fused multiply-add，吞吐高

`atomicAdd` 是核心瓶颈之一（多个 block 累加同一输出元素）


#### Triton kernel（`matmul_248_kernel`）

这是标准 tile GEMM 形态：

- 一个 program 负责 `C[BLOCK_M, BLOCK_N]`
- 在 K 维循环，逐块加载 A / packed B
- 内部解包4bit、应用 zero/scale，再 `tl.dot` 累加到 fp32 accumulator
- 最后一次性 store 输出块

通常更适合大 batch

- 不用 atomic 汇总（一个 tile 由一个 program 完整负责）
- 访存与算子调度更接近高效 GEMM
- 有 autotune 选 block 配置，吞吐更稳

`transpose_matmul_248_kernel` 用于 backward 求 `grad_input = grad_output * W^T`。  

推理一般不走 backward，但这个实现让 `QuantLinearFunction` 在训练/梯度场景也可工作。

关键一致性细节

- `qzeros` 打包时 `-1`，kernel 解包后 `+1`
- `qweight`/`qzeros` 都按 32bit 容器 + 4bit lane 提取
- `g_idx` 在 CUDA `_g` 和 Triton 都参与索引 `scales/zeros`
- 累加精度：中间基本用 fp32（CUDA `_g` 最终原子加到 half，精度略弱于纯 fp32 输出路径）
