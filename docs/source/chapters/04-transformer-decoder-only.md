# Transformer (decoder-only) 模型实现

```{contents} 本页目录
---
depth: 2
local: true
---
```

## 现代 Transformer 架构与超参数的工程默认值

开始从零写出一个 decoder-only Transformer 之前，真正难的部分不是“照着论文把模块堆起来”，而是判断哪些结构选择值得成为默认值。现代大模型论文里会出现很多变体：pre-norm、RMSNorm、SwiGLU、RoPE、GQA、滑动窗口注意力、QK norm、z-loss、logit soft-capping。它们看起来分散，但背后都在回答同一个问题：在模型继续变大、上下文继续变长、训练规模继续上升时，怎样让梯度、算力、显存访问和推理延迟保持可控。

不是每个默认值都有严格的理论最优证明，但很多选择已经被大量模型和工程经验反复验证。更重要的是，不只是记住结论，还要知道每个结论是从什么约束推出来的：残差路径为什么要尽量保持干净，RMSNorm 为什么可能比 FLOPs 数字看起来更重要，GLU 为什么要配合缩小 `d_ff`，RoPE 为什么应该作用在 attention 的 query/key 上，以及 GQA/MQA 为什么主要是在解决推理阶段的 KV cache 读写。

<callout emoji="💡">
用一句话概括这组默认值：现代 decoder-only LLM 的架构优化不是为了把 Transformer 改得更花，而是为了让 residual 主干更稳定、FFN 更有表达力、位置关系更贴近 attention、推理缓存更轻，同时把实验预算留给数据、训练和系统实现。
</callout>

### 先确定 baseline：现代 Transformer 和原始 Transformer 差在哪里

一个最朴素的 Transformer block 可以写成两段：先做 self-attention，再做 FFN，中间都带 residual connection。早期 Transformer 常见的是 post-norm，也就是先把子层输出加回 residual，再做 LayerNorm。用符号写就是：

```text
h'      = LayerNorm(h + Attention(h))
h_next  = LayerNorm(h' + FFN(h'))
```

现代 decoder-only LLM 更常见的是 pre-norm。它先归一化输入，再把子层输出加回原始 residual 流：

```text
h'      = h + Attention(Norm(h))
h_next  = h' + FFN(Norm(h'))
```

这两个式子看起来只换了 norm 的位置，但训练含义很不一样。post-norm 里，残差相加之后马上被 LayerNorm 改写；也就是说，所谓的 identity path 不再是完全的 identity。pre-norm 里，`h` 可以沿着 residual 分支直接传到下一层，子层只是额外加上一个增量。对深层网络来说，这条“干净的高速路”很关键，因为反向传播时梯度至少有一部分可以沿 identity path 传回去，而不是每层都被 norm 的雅可比矩阵重新缩放。

所以，当我们实现一个现代 baseline 时，会先采用这组组合：pre-norm 或 non-residual norm、RoPE、SwiGLU、线性层和 norm 不带 bias。它不保证在所有数据集上最优，但它是一个风险很低的起点。

![图：一个简单但现代的 decoder-only Transformer baseline](../assets/images/04-transformer-decoder-only/image-01.png)

深层网络很难训练，一个原因是每层都可能放大、缩小或扭曲梯度。residual 的作用是给信息和梯度提供一条近似恒等的路径：即使某一层子模块暂时学得不好，模型也可以先把输入原样传下去。pre-norm 的价值正在这里：它让 norm 服务于子层计算，而不是持续改写 residual 主干。

如果写得更直观一点，post-norm 的更新是 `Norm(h + F(h))`。这里 `h` 和 `F(h)` 相加后整体被重新中心化和缩放。pre-norm 的更新是 `h + F(Norm(h))`。这里 `h` 直接进入下一层，归一化只发生在子层输入上。大模型层数加深以后，这个差异会放大成训练稳定性的差异：pre-norm 更容易避免梯度衰减和尖峰，也更容易使用较大的学习率。

这并不意味着 post-norm 永远错误。BERT 这类较早模型就使用 post-norm，有些新模型也会在 residual 之外再加一层 non-residual postnorm 或 double norm。但我会把它们看成在“保护 residual 主干”这个原则下的变体，而不是回到早期 post-norm 的简单结构。

![图：pre-norm 把 LayerNorm 移出 residual 主路径](../assets/images/04-transformer-decoder-only/image-02.png)

### RMSNorm 和无 bias：小操作为什么能影响 wall-clock time

LayerNorm 会对一个 token 的 hidden vector 计算均值和方差，再做归一化：

```text
mean = average(x)
var  = average((x - mean)^2)
y    = (x - mean) / sqrt(var + eps) * gamma + beta
```

RMSNorm 删掉了减均值这一步，只保留均方根缩放：

```text
rms = sqrt(average(x^2) + eps)
y   = x / rms * gamma
```

单从 FLOPs 看，norm 在整个 Transformer 里不是最大头，矩阵乘法才是主要计算量。因此“RMSNorm 少算一点”并不是完整解释。更关键的是，大模型运行时经常受数据搬运影响：norm 每层都做、每个 token 都做，虽然算术量小，但要读写 activation、参数和中间统计量。RMSNorm 少读写一个均值和 bias 路径，kernel 也更简单，所以它可能在 wall-clock time 上产生可见收益。

无 bias 也是类似逻辑。线性层 `y = xW + b` 里的 `b` 参数量看起来很小，但它需要存储、加载、参与优化器状态，并且在有 norm 的网络里，很多平移自由度会被归一化抵消。对大模型来说，这类参数的表达收益通常不如它带来的实现复杂度和数据搬运成本。因此现代 LLM 经常在线性层和 norm 层都去掉 bias。

| 选择 | 常见现代做法 | 我会怎么理解 |
|-|-|-|
| Norm 位置 | pre-norm / non-residual norm | 让 residual 主干更接近 identity path，改善深层训练稳定性 |
| Norm 类型 | RMSNorm | 保留缩放稳定性，减少均值相关计算和数据搬运 |
| Bias | 多数线性层和 norm 不带 bias | 小参数收益有限，但会增加状态、读写和优化复杂度 |

### FFN 的演化：SwiGLU 为什么不是“换个激活函数”这么简单

Transformer block 里 FFN 通常占据很大一部分参数和 FLOPs。最普通的 FFN 是先升维、激活、再降维：

```text
FFN(x) = activation(xW_up) W_down
```

如果 `x` 的维度是 `d_model`，中间维度是 `d_ff`，那么两层矩阵大约有 `2 * d_model * d_ff` 个参数。传统经验会取 `d_ff = 4 * d_model`，所以 FFN 参数量大约是 `8 * d_model^2`。这也是为什么 FFN 宽度不是一个小超参数：它直接决定每层很大一部分计算和参数预算。

GLU 系列把 FFN 改成“内容分支 × 门控分支”的形式。以 SwiGLU 为例，可以写成：

```text
FFN_GLU(x) = (activation(xW_gate) * xW_up) W_down
```

它多了一组上投影矩阵，所以如果还保持 `d_ff = 4 * d_model`，参数和计算都会明显增加。为了让预算接近普通 FFN，我们可以做一个简单换算：普通 FFN 是 `2 * d_model * 4d_model = 8d_model^2`；GLU 有三个矩阵，约为 `3 * d_model * d_ff_glu`。令二者接近，就得到 `d_ff_glu ≈ 8/3 * d_model`。这就是为什么很多使用 SwiGLU/GeGLU 的模型，会把 FFN hidden size 调到约 `2.66 * d_model`。

从功能上看，GLU 不是简单把 ReLU 换成 Swish 或 GeLU。门控分支让模型可以按通道决定哪些信息通过，等价于给 FFN 加了输入相关的选择能力。经验上，SwiGLU/GeGLU 在很多模型中比 ReLU/GeLU 更稳定地带来收益；从工程上看，只要把中间维度按 `8/3` 调整，它的预算也能保持在可接受范围内。

![图：GeGLU / SwiGLU 代表的 gated FFN 变体](../assets/images/04-transformer-decoder-only/image-03.png)

### RoPE：把相对位置写进 attention 内积

位置编码要解决的问题是：同一个词出现在不同位置时，attention 应该知道它的位置关系。绝对位置 embedding 的做法是在 token embedding 上加一个位置向量；正弦位置 embedding 也是加法，只是位置向量有固定频率结构。加法方案简单，但它很难保证 attention score 只依赖相对距离，因为 query/key 内积里会混入 token 内容、绝对位置和交叉项。

我希望的位置形式是：

```text
score(i, j) = f(x_i, i)^T f(x_j, j)
            = g(x_i, x_j, i - j)
```

RoPE 的关键是把向量坐标两两配对，在二维平面里按位置角度旋转。设第 `i` 个位置对应旋转矩阵 `R_i`，attention 里实际参与内积的是 `R_i q` 和 `R_j k`。于是：

```text
(R_i q)^T (R_j k)
= q^T R_i^T R_j k
= q^T R_{j-i} k
```

这一步就是 RoPE 的核心。位置 `i` 和 `j` 没有以两个独立标签进入打分，而是通过 `j - i` 进入内积。也因此，我在实现 RoPE 时会把它放在每一层 attention 的 query/key 路径上，而不是只在 embedding 层加一次位置向量。它不是“给词加位置”，而是让 attention 打分天然带有相对位置信息。

![图：RoPE 让 attention 更直接依赖相对位置](../assets/images/04-transformer-decoder-only/image-04.png)

### 超参数共识：默认值背后是预算分配

看模型表格时，很容易把超参数当成经验数字背下来。但更有用的方式是问：这个数字控制了哪类预算？FFN 宽度控制每层参数和 MLP FLOPs；head 数和 head_dim 控制 attention 表达和 kernel 形态；模型深宽比影响 pipeline parallelism、latency 和每层吞吐；词表大小影响序列长度、embedding/softmax 参数和多语覆盖。

| 问题 | 保守默认值 | 推导和取舍 |
|-|-|-|
| FFN 要多宽 | 非 GLU：`4 * d_model`；GLU：约 `8/3 * d_model` | 普通 FFN 有两个矩阵，GLU 有三个矩阵；为了维持接近预算，GLU 中间维度需要缩小 |
| head 怎么配 | `head_dim * num_heads ≈ d_model` | 保持投影维度和模型宽度接近，方便实现、并行和 checkpoint 迁移 |
| 深还是宽 | `d_model / n_layer` 常落在约 100-200 | 更深会增加串行层数和推理延迟；更宽会改变单层 GEMM、通信和显存压力 |
| 词表多大 | 单语约 30k-50k；多语/生产系统约 100k-250k | 大词表减少 token 数，但会增加 embedding 和输出 softmax 成本，也会改变稀有词覆盖 |
| 预训练要不要 dropout | 新模型常不用 dropout，但保留 weight decay | 海量 token 下过拟合不是唯一矛盾，weight decay 更多影响优化动力学和学习率 schedule |

以 FFN 宽度为例，`4 * d_model` 不是不可违反的规律。T5 的 11B 版本曾经使用非常大的 FFN multiplier，说明极端设置也可以训练起来。但后续许多模型回到更保守的比例，原因很现实：每次扩大 FFN 都会吃掉参数、计算和通信预算。如果收益没有稳定超过这些成本，默认值就应该保守。

![图：GLU 变体下 FFN 宽度通常缩放到 8/3 * d_model 附近](../assets/images/04-transformer-decoder-only/image-05.png)

词表大小也是同样的预算问题。小词表让 embedding 和 softmax 更便宜，但可能把一个词切成更多 token，拉长序列；大词表能减少 token 数、改善多语和特殊领域覆盖，但输出层更大、低频 token 更稀疏。单语模型常见 30k-50k，多语或生产系统常见 100k-250k，并不是谁更高级，而是面向语种、数据分布和服务成本的不同折中。

dropout 和 weight decay 则提醒我，不要把“regularization”只理解成防止过拟合。预训练数据通常有数万亿 token，模型很少在同一批数据上反复训练到纯记忆，dropout 的必要性下降；但 weight decay 仍然常见，因为它会和学习率、cosine schedule、参数范数演化相互作用，影响训练动力学。

### 稳定性技巧：softmax 是最容易放大问题的地方

当模型规模变大以后，很多训练不稳定会集中暴露在 softmax 周围。softmax 先对 logit 做指数，再按总和归一化。如果 logit 尺度过大，指数会迅速放大差异；如果分布过尖，梯度会集中到少数位置；如果 attention score 或 vocab logit 在训练中突然失控，就会出现 loss spike 甚至训练崩溃。

![图：大规模训练中的稳定性问题常表现为 loss 尖峰或崩溃](../assets/images/04-transformer-decoder-only/image-06.png)

输出端的 z-loss 可以这样理解。设 `z = log(sum(exp(logits)))`，这是 softmax 归一化分母的 log 形式。z-loss 会惩罚过大的 `z`，让输出 logit 的整体尺度不要无限膨胀。它不改变语言建模目标的主方向，但给数值尺度加了一个软约束。

attention 端的 QK norm 处理的是另一个位置：在 query 和 key 做内积之前，先对它们做 LayerNorm 或 RMSNorm。这样 `QK^T` 的分布不会轻易因为某些向量范数变大而变得极端。logit soft-capping 则更直接：用 `tanh` 等函数把 logit 压到一个上限内。三者都不是为了让模型结构更复杂，而是为了避免 softmax 把尺度问题放大成训练事故。

### GQA / MQA：推理瓶颈为什么会从计算转向 KV cache

训练时，attention 可以并行处理整段序列。矩阵乘法足够大，GPU 容易保持高利用率。自回归生成不一样：第 `t` 个 token 生成时，只能使用前面已经生成的 KV cache；第 `t+1` 个 token 又要在新缓存上继续算。这个过程是逐 token 推进的，batch 和 sequence 的形状也更不稳定。

标准 multi-head attention 会为每个 head 都保存一套 K/V。假设有 `h` 个 heads，每个 head_dim 是 `k`，那么每个 token 的 KV cache 大小和 `h * k` 相关，也就是接近 `d_model`。上下文越长，缓存按 token 数线性增长；并发 batch 越大，缓存再按 batch 线性增长。生成时每一步都要读过去的 K/V，所以瓶颈很容易从“算不动”变成“搬不动”。

MQA 的做法是让多个 query heads 共享同一组 K/V heads。这样 query 仍然可以有多个 heads，但 K/V cache 的份数显著减少。GQA 是折中：不是所有 query 都共享一组 K/V，而是一组 query heads 共享一组 K/V heads。这个 knob 很实用，因为它允许在表达能力和推理成本之间连续调节，而不是只能在 full MHA 和 MQA 两端二选一。

这也是为什么 GQA/MQA 更像推理系统优化，而不是单纯架构审美。训练 FLOPs 上它们未必显得惊人，但在长上下文、较大 batch、服务端持续生成的场景里，KV cache 读写量经常直接决定吞吐和成本。

![图：MQA 通过共享 K/V 降低 KV cache 读写成本](../assets/images/04-transformer-decoder-only/image-07.png)

### 滑动窗口和混合注意力：用图结构控制长上下文成本

full attention 可以看成一个完全图：每个 token 都连到所有历史 token。表达能力强，但边数是 `O(n^2)`。滑动窗口 attention 把图变成一条带宽有限的局部图：每个 token 只看附近窗口。这样每层成本更低，但远距离信息不能在一层内直接传播。

如果所有层都只做局部窗口，长程依赖就要靠多层逐步传递，路径会变长。因此现代长上下文模型常用混合结构：大部分层用 sliding window 或 local attention，少数层保留 full attention，让远距离信息周期性地重新连通。这样做的核心不是某个固定间隔，而是在成本和表达能力之间选一张合适的 attention graph。

需要把这个策略和前面的 GQA/MQA 区分开：GQA/MQA 主要减少每个 token 的 K/V 状态大小，滑动窗口主要减少每层 attention 连接数。前者压缩缓存宽度，后者压缩上下文连接密度。两者可以叠加，也会共同影响推理系统的 batch、cache 和 kernel 设计。

## Embedding layer

tokenizer 之后，模型入口变成整数张量 `(batch_size, sequence_length)`。Embedding 层把每个 token ID 查成一个 `d_model` 维向量，输出 `(batch_size, sequence_length, d_model)`。从这一步开始，我会强制自己用形状来检查每个模块：最后一维是特征维，前面的维度都可以看作 batch-like 维度。这样写 Linear、RMSNorm、FFN、attention 时，代码自然支持 batch、sequence、head 等额外维度。

PyTorch 默认是 row-major 内存布局，常见线性层权重存成 `(out_features, in_features)`。数学上如果写列向量会得到 `y = W x`，但代码里更常见的是对最后一维做 `x @ W.T`。这不是符号细节，而是实现 bug 的高发点：权重应该按什么形状存、forward 里是否转置、输入的最后一维是否等于 `in_features`，都要一致。

整个 decoder-only Transformer LM 可以写成下面这条路径：

```text
token_ids:        (B, T)
token_embedding:  (B, T, D)
for each block:
    residual stream stays (B, T, D)
final RMSNorm:    (B, T, D)
LM head:          (B, T, V)
logits:           next-token scores for every position
```

Embedding layer实现：

```python
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.embeddings = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.embeddings, mean=0.0, std=1 / math.sqrt(embedding_dim))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]

```

## FFN & PreNorm Layer

### Pre-norm block：残差主干必须保持干净

现代 decoder-only LM 通常使用 pre-norm，而不是原始 Transformer 的 post-norm。pre-norm 的计算顺序可以写成：

```text
z = x + MultiHeadSelfAttention(RMSNorm(x))
y = z + FFN(RMSNorm(z))
```

这两行公式的重点是 residual stream。主干 `x -> z -> y` 上没有被 normalization 直接截断；RMSNorm 只放在进入子层之前。这样残差路径更像一条干净的信息高速路，attention 和 FFN 往里面追加更新量。训练深层 Transformer 时，这种结构通常比 post-norm 更稳定，因为梯度可以沿着残差路径更直接地传播。

### RMSNorm

RMSNorm 本身也值得拆开。LayerNorm 会减均值再除标准差，RMSNorm 则只用均方根缩放：

```text
RMS(a) = sqrt(mean(a_i^2) + eps)
RMSNorm(a_i) = a_i / RMS(a) * g_i
```

实现时先把输入 upcast 到 float32 再平方求和，避免低精度下 overflow 或精度损失；最后再 cast 回原 dtype。

RMSNorm 实现：

```python
class RmsNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.g = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.g * x).to(input_dtype)

```

### GLU 和 SwiGLU：FFN 的门控不是换激活函数这么简单

#### Linear layer实现

```python
class Linear(torch.nn.Module):
    def __init__(
        self, in_features, out_features, weights: Float[Tensor, " out in"] | None = None, device=None, dtype=None
    ):
        super().__init__()
        if weights is None:
            sigma = math.sqrt(2.0 / (in_features + out_features))
            self.w = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
            torch.nn.init.trunc_normal_(self.w, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)
        else:
            self.w = torch.nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einx.dot("... [in], out [in] -> ... out", x, self.w)


```

> Weight 矩阵的 shape 为 [out_feature, in_feature] 是因为这样在做矩阵乘法时，weight矩阵(右矩阵) 可以按行取数，对缓存更友好。

#### SiLU（Sigmoid Linear Unit）

又称 Swish（Google Brain 2017 提出）。SiLU = ReLU 的平滑升级版：用微小的计算代价，换取梯度稳定性、表达能力与现代架构的深度契合。在资源允许时，它是 ReLU 的现代化替代方案——这也是 LLaMA、Mistral 等顶尖大模型集体选择它的原因。

```python
class SiLu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * x
```

**GLU（Gated Linear Units）**，它是一类带门控机制的激活函数。数学形式如下：

```text
# 标准 GLU（以输入 x 为例）
a = W₁x + b₁          # 线性变换（内容路径）
b = W₂x + b₂          # 线性变换（门控路径）
output = a ⊙ σ(b)     # ⊙ = 逐元素乘，σ = sigmoid（或其他激活函数）
```

本质：用门控信号 `σ(b)` 动态调节内容信号 `a` 的通过强度

- 关键特性：

  - ✅ 非线性增强：门控机制提供比单激活函数更强的表达能力
  - ✅ 梯度友好：门控路径保留梯度流（缓解梯度消失）
  - ✅ 参数可控：通过调整门控激活函数衍生多种高效变体

#### 常用变体（实际工程中更主流）


| 变体名称 | 公式 | 特点 |
| --- | --- | --- |
| SwiGLU | SwiGLU(x)=a⊗SiLU(b) | 用 SiLU 替代 Sigmoid，梯度更优，GPT/LLaMA 均采用 |
| ReGLU | ReGLU(x)=a⊗ReLU(b) | 计算更快，适合轻量级模型 |


原始 Transformer 的 FFN 大致是 `W2 ReLU(W1 x)`，中间维度常取 `4 * d_model`。现代 LLM 更常用 SwiGLU，并且通常去掉 linear bias。SwiGLU 可以写成：

```text
SiLU(u) = u * sigmoid(u)
FFN(x) = W2( SiLU(W1 x) * W3 x )
```

这不是单纯把 ReLU 换成 SiLU。这里有两条投影分支：一条经过 SiLU 产生平滑的门控信号，另一条产生被门控的值；两者逐元素相乘，再投影回 `d_model`。可以把它理解成“让每个 hidden dimension 自己决定信息通过多少”。相比普通 FFN，SwiGLU 多了一组矩阵，因此为了让参数量和计算量可比，中间维度通常不是 `4 * d_model`，而是接近 `8/3 * d_model`，再向 64 的倍数取整以适配硬件。

![图：SiLU 与 ReLU 的形状对比，是理解 SwiGLU 门控的入口](../assets/images/04-transformer-decoder-only/image-08.png)

这个细节对初学者很重要：如果在 ablation 里比较 SwiGLU 和普通 SiLU FFN，却让两者参数量差很多，那么最后看到的 loss 差异就不一定来自 gating。正确的比较方式是让 SiLU baseline 用 `4 * d_model`，让 SwiGLU 用约 `8/3 * d_model`，尽量匹配参数规模后再看曲线。

原始 transform 中使用 Linear Layer + ReLU 的方式实现FFN，现代变体通常使用SwiGLU。

GLU 实现：

```python
class Glu(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(in_features, out_features, device=device, dtype=dtype)
        self.w2 = Linear(in_features, out_features, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.w1(x)) * self.w2(x)


```

SwiGLU FFN实现：

```python
class SwiGluFFN(torch.nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, device=None, dtype=None) -> None:
        super().__init__()
        self.w1 = Linear(d_in, d_hidden, device=device, dtype=dtype)
        self.w3 = Linear(d_in, d_hidden, device=device, dtype=dtype)
        self.w2 = Linear(d_hidden, d_out, device=device, dtype=dtype)
        self.silu = SiLu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

```

## RoPE (Rotary Position Embedding)

在 decoder-only Transformer 里，attention score 通常写成：

$$s_{m,n}=\mathbf{q}_m^\top\mathbf{k}_n,\qquad \mathbf{q}_m=W_q\mathbf{x}_m,\quad \mathbf{k}_n=W_k\mathbf{x}_n$$

这里的 `m` 是当前 query token 的位置，`n` 是被看的 key token 的位置。如果不加入位置编码，两个 token 的相对顺序不会进入这个内积；模型只能知道“两个内容向量像不像”，不知道“它们隔了多远”。一种直接做法是给输入加绝对位置向量，但加法会把 token 内容、绝对位置和交叉项混在一起，很难保证 attention score 以稳定方式依赖相对距离。

RoPE 的目标更明确：找到一个位置相关变换 $f(\mathbf{x},m)$，使得 query/key 打分可以写成：

$$f_q(\mathbf{q},m)^\top f_k(\mathbf{k},n)=g(\mathbf{q},\mathbf{k},n-m)$$

也就是说，位置最终通过 `n-m` 进入 attention，而不是作为两个互不相关的绝对标签进入。

### 把 head dimension 两两看成二维平面

RoPE的具体做法是：设单个 attention head 的维度为 $d_h$，并且 $d_h$ 是偶数。RoPE 把向量按相邻维度两两分组：

$$(x_0,x_1),\ (x_2,x_3),\ldots,\ (x_{d_h-2},x_{d_h-1})$$

第 $i$ 个二维子空间使用一个固定频率：

$$\omega_i=\theta^{-2i/d_h},\qquad i=0,\ldots,d_h/2-1$$

常见实现里 $\theta=10000$。位置 $m$ 对应的旋转角度就是 $m\omega_i$。二维旋转矩阵为：

$$R_i(m)=\begin{bmatrix}\cos(m\omega_i)&-\sin(m\omega_i)\\\sin(m\omega_i)&\cos(m\omega_i)\end{bmatrix}$$

整个 head 上的旋转矩阵是 block diagonal 结构：

$$R_m=\operatorname{diag}\left(R_0(m),R_1(m),\ldots,R_{d_h/2-1}(m)\right)$$

实际工程不会显式构造这个大矩阵，只会缓存每个 position、每个维度 pair 对应的 `cos` 和 `sin`，再用向量化操作完成旋转。

### 相对位置来自旋转矩阵的群性质

RoPE 不直接改 token embedding，而是在每一层 attention 里旋转 query 和 key：

$$\tilde{\mathbf{q}}_m=R_m\mathbf{q}_m,\qquad \tilde{\mathbf{k}}_n=R_n\mathbf{k}_n$$

新的 attention score 是：

$$s_{m,n}^{\mathrm{RoPE}}=(R_m\mathbf{q}_m)^\top(R_n\mathbf{k}_n)$$

旋转矩阵有两个关键性质：

$$R_m^\top=R_{-m},\qquad R_aR_b=R_{a+b}$$

因此：

$$(R_m\mathbf{q}_m)^\top(R_n\mathbf{k}_n)=\mathbf{q}_m^\top R_m^\top R_n\mathbf{k}_n=\mathbf{q}_m^\top R_{n-m}\mathbf{k}_n$$

这一步就是 RoPE 的数学核心。绝对位置 `m` 和 `n` 在推导中相消，只留下相对位移 `n-m`。所以 RoPE 不是简单地“把位置信息塞进向量”，而是把相对距离写进了 query/key 的内积结构。

### 二维展开后能直接看到 sin 和 cos

只看第 $i$ 个二维子空间，令：

$$\mathbf{q}_i=\begin{bmatrix}a\\b\end{bmatrix},\qquad \mathbf{k}_i=\begin{bmatrix}c\\d\end{bmatrix},\qquad \Delta=(n-m)\omega_i$$

旋转后的二维内积可以展开为：

$$(R_i(m)\mathbf{q}_i)^\top(R_i(n)\mathbf{k}_i)=(ac+bd)\cos\Delta+(bc-ad)\sin\Delta$$

这个式子说明了两个细节。第一，原始内容相似度 $ac+bd$ 仍然保留，只是被相对距离的余弦项调制。第二，$bc-ad$ 是二维方向关系，它被相对距离的正弦项调制。于是同一对 token 内容，在距离不同的时候会得到不同的 attention score；同一距离下，不同频率的二维子空间也会给出不同尺度的位置响应。

### 为什么要用一组频率

如果所有二维子空间都用同一个频率，位置模式很快会周期性重复，表达能力也有限。RoPE 沿用了 sinusoidal position embedding 的多频率设计：小 $i$ 对应较高频率，对短距离变化更敏感；大 $i$ 对应较低频率，对长距离变化更平滑。

| 频率范围 | 数学效果 | 直观作用 |
|-|-|-|
| 高频 | $\omega_i$ 大，$m\omega_i$ 随位置变化快 | 更容易区分近距离 token 的顺序差异 |
| 低频 | $\omega_i$ 小，旋转角度变化慢 | 更适合给长距离依赖提供平滑的位置线索 |

多频率不是装饰，而是在不同二维子空间里提供不同“波长”的相对位置特征。模型后续可以通过注意力头和线性层学习如何组合这些尺度。

### 具体实现：rotate_half 只是矩阵乘法的向量化写法

以相邻维度配对的实现为例，二维旋转可以写成：

$$\begin{bmatrix}x_0'\\x_1'\end{bmatrix}=\begin{bmatrix}x_0\cos\alpha-x_1\sin\alpha\\x_0\sin\alpha+x_1\cos\alpha\end{bmatrix}$$

这等价于：

$$\mathbf{x}'=\mathbf{x}\cos\alpha+\operatorname{rotate_half}(\mathbf{x})\sin\alpha,\qquad \operatorname{rotate_half}([x_0,x_1])=[-x_1,x_0]$$

下面保留完整实现，和上面的数学形式一一对应：`inv_freq` 对应每个二维子空间的 $\omega_i$，`cos_cached` / `sin_cached` 对应位置 $m$ 的旋转角，`x_rotated` 对应 `rotate_half`。

```python
class RoPE(torch.nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # inv_freq: (dim//2,)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))

        # t: (seq_len,)
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # freqs: (seq_len, dim//2)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # outer product

        emb = freqs.repeat_interleave(2, dim=-1)  # (seq_len, dim)

        self.register_buffer("cos_cached", emb.cos().to(dtype))  # (seq_len, dim)
        self.register_buffer("sin_cached", emb.sin().to(dtype))

    def forward(self, x: Float[Tensor, "... seq d_k"], token_positions: Float[Tensor, "... seq"]) -> torch.Tensor:
        # token_positions: (..., seq_len) 任意前缀维度
        # x: (..., seq_len, dim)
        cos = self.cos_cached[token_positions]  # (..., seq_len, dim)
        sin = self.sin_cached[token_positions]  # (..., seq_len, dim)

        x_reshaped = x.view(*x.shape[:-1], -1, 2)  # (..., seq_len, dim//2, 2)
        x_rotated = torch.stack((-x_reshaped[..., 1], x_reshaped[..., 0]), dim=-1)  # rotate: (a,b) -> (-b,a)
        x_rotated = x_rotated.view(*x.shape)  # (..., seq_len, dim)

        if x.ndim == 4:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # print(f"x shape {x.shape}, cos shape {cos.shape}")
        x_rot = x * cos + x_rotated * sin
        return x_rot
```

工程实现里需要注意两件事。第一，`cos` 和 `sin` 的缓存布局必须和维度配对方式一致；有些实现是相邻维度配对，有些实现把前半维当实部、后半维当虚部。第二，`token_positions` 用来索引缓存，实际类型应是整数张量；对于带 KV cache 的增量推理，它通常不是简单的 `0..seq_len-1`，而是当前上下文里的真实位置。

### 为什么 RoPE 只作用在 query 和 key 上

attention 的输出是：

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V$$

位置关系主要决定“当前 token 应该看哪些历史 token”，也就是 softmax 前的打分矩阵 $QK^\top$。因此 RoPE 作用在 query/key 上，让权重计算携带相对位置。value 承载的是被聚合的内容本身，通常不需要被同样旋转；否则会把位置信号进一步混入被读取的信息，反而让内容聚合变得更难解释。



## Attention & Transformer

### ScaledDotProductAttention

```python
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(
        self,
        q: Float[Tensor, "... s d"],
        k: Float[Tensor, "... s d"],
        v: Float[Tensor, "... s d"],
        mask: torch.Tensor
        | None = None,  # true means the item should be covered and not participating to softmax calculating
    ) -> torch.Tensor:
        d_model = q.shape[-1]

        # Compute attention scores
        att = einx.dot("... s_q [d], ... s_k [d] -> ... s_q s_k", q, k)
        att_scale = att / math.sqrt(d_model)

        if mask is not None:
            if mask.ndim < att_scale.ndim:
                mask = mask.reshape((1,) * (att_scale.ndim - mask.ndim) + mask.shape)
            # Apply mask - removed the ~ operator
            att_scale = att_scale.masked_fill(mask, -1e9)

        att_score = self.softmax(att_scale)

        return einx.dot("... s_q [s], ... [s] d -> ... s_q d", att_score, v)

```

### MultiHeadAttention

实际实现中通常使用一个投影矩阵完成对多个head的投影，用来提升计算速度。

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_head: int, max_seq_len=2048, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.out_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.project = Linear(in_features=d_model, out_features=3 * d_model, device=device, dtype=dtype)
        self.dot_product_att = ScaledDotProductAttention()

        # Cache causal mask - removed the ~ operator
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: Float[Tensor, "b s d"]) -> torch.Tensor:
        seq_len = x.shape[1]

        mask = self.causal_mask[:seq_len, :seq_len]

        qkv = self.project(x)
        q, k, v = einx.rearrange("b s (n h d) -> n b h s d", qkv, n=3, h=self.num_head)

        output = self.dot_product_att(q, k, v, mask)
        output = einx.rearrange("b h s d -> b s (h d)", output)
        return self.out_linear(output)

```

### MultiHeadAttentionWithRoPE

multi-head attention 只是把 `d_model` 拆成 `num_heads * d_head`。head 维度应该像 batch 维度一样独立处理：每个 head 都用自己的 Q/K/V 切片做 attention，但 RoPE 的位置旋转规则对所有 head 一样。一个稳妥的实现路径是先分别做 Q、K、V 三个线性投影，再把形状从 `(B, T, D)` rearrange 成 `(B, H, T, d_head)`，对 Q/K 应用 RoPE，对所有 head 并行算 masked attention，最后 concat 回 `(B, T, D)` 并过 output projection。

```python
class MultiHeadAttentionWithRoPE(MultiHeadAttention):
    def __init__(self, d_model: int, num_head: int, theta: float = 10000, max_seq_len=2048, device=None, dtype=None):
        super().__init__(d_model=d_model, num_head=num_head, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.rope = RoPE(d_model // num_head, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[1]
        batch_size = x.shape[0]

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        mask = self.causal_mask[:seq_len, :seq_len]

        qkv = self.project(x)
        q, k, v = einx.rearrange("b s (n h d) -> n b h s d", qkv, n=3, h=self.num_head)

        # Apply RoPE to q and k
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        output = self.dot_product_att(q, k, v, mask)
        output = einx.rearrange("b h s d -> b s (h d)", output)
        return self.out_linear(output)

```

### TransformerBlock

将上述模块堆叠在一起可以实现TransformerBlock，跟原 transformer 论文不同的是，现代 LLM 通常使用pre-norm, layer norm使用更简单的RmsNorm，并且使用 SwiGLU作为 FFN 实现。

```python
class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        theta: float = 10000,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.rms_norm1 = RmsNorm(d_model, device=device, dtype=dtype)
        self.rms_norm2 = RmsNorm(d_model, device=device, dtype=dtype)
        self.mult_head_atten = MultiHeadAttentionWithRoPE(
            d_model, num_heads, theta, max_seq_len=max_seq_len, device=device, dtype=dtype
        )
        self.ffe = FFN(d_model, d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x_norm = self.rms_norm1(x)
        x_atten = self.mult_head_atten(x_norm, token_positions)
        x = x + x_atten
        x_norm = self.rms_norm2(x)
        x_ffe = self.ffe(x_norm)
        return x + x_ffe

```

### Transformer

最终的 trasformer 是由多个 transformer block叠加起来

```python
class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        num_layers: int,
        max_seq_len=2048,
        rope_theta: float = 10000,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.out_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len

    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(token_ids)
        if token_positions is None:
            batch_size, seq_len = token_ids.shape
            token_positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        for block in self.blocks:
            x = block(x, token_positions)
        x_norm = self.norm(x)
        logits = self.out_linear(x_norm)
        return logits


```



## Optimize

### Loss

语言模型训练目标很朴素：给定前缀 `x_1 ... x_i`，预测下一个 token `x_{i+1}`。模型一次前向会对每个位置都输出一个 logits 向量，所以一个长度为 `T` 的输入可以同时产生 `T` 个 next-token 预测。训练 batch 里，输入 `x` 和目标 `y` 的关系就是右移一位。

对单个位置，cross-entropy 可以写成：

```text
loss_i = -log softmax(logits_i)[target_i]
       = log(sum_j exp(logits_i[j] - max_i)) + max_i - logits_i[target_i]
```

需要避免先显式算 softmax 再取 log，因为这样会把两个容易溢出或下溢的操作连在一起。用上面的 logsumexp 形式，最大值被减掉，指数项更稳定；目标 token 的 logit 单独取出，最后对所有 batch-like 位置求平均。perplexity 则是平均 cross-entropy 的指数：`perplexity = exp(mean_loss)`。它可以理解成模型平均每一步还在多少个候选 token 之间困惑。

```python
class CrossEntropyLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Reshape for cross_entropy, handling any shape of logits
        # logits = einops.rearrange(logits, "... c -> (...) c")
        # targets = einops.rearrange(targets, "... -> (...)")
        logits = einx.rearrange("... c -> (...) c", logits)
        targets = einx.rearrange("... -> (...)", targets)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        correct_log_probs = log_probs[torch.arange(len(log_probs)), targets]
        nll = -correct_log_probs

        mean_loss = torch.mean(nll)

        return mean_loss

```

### Optimizer —— AdamW

SGD 的更新很容易理解：参数沿着负梯度方向走一步。但现代 Transformer 通常用 AdamW，因为它会为每个参数维护一阶和二阶 moment，分别估计梯度均值和梯度平方均值。这样每个参数的步长会根据历史梯度尺度自适应调整，训练通常更稳。

AdamW 里的 W 很重要：weight decay 和梯度更新是解耦的。也就是说，参数先被按 `theta = theta - lr * weight_decay * theta` 往 0 拉，再用 moment-adjusted gradient 更新。这样 weight decay 更接近直接的参数正则，而不是混进 Adam 的梯度统计里。



**AdamW**（Adam with decoupled Weight Decay）中**权重衰减不再混入梯度计算，**修正了原始Adam中L2正则化失效的问题。完整更新流程为：

1. 计算梯度：$g_t = \nabla_\theta L(\theta_{t-1})$
2. 一阶矩（动量）更新：$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
3. 二阶矩（梯度平方）更新：$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
4. 偏置校正：(初始状态$\beta_t 和 v_t 累计值较小，需要进行放大，随着step增加，放大倍数逐步减小$）

$$\hat{m}_t = m_t / (1-\beta_1^t)$$

$$\hat{v}_t = v_t / (1-\beta_2^t)$$

**AdamW核心更新**（解耦权重衰减）：

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \cdot \lambda \cdot \theta_{t-1}$$

**与原始Adam的关键区别：**

- **原始Adam**：权重衰减项被包含在梯度内，受自适应学习率缩放，正则效果不稳定

$$\theta_t = \theta_{t-1} - \eta \cdot \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

- **AdamW**：权重衰减独立于梯度，直接作用于参数，与SGD的L2正则行为一致

```python
class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr=1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay=1e-3,
        eps=1e-8,
    ):
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        beta1, beta2 = betas
        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["sm"] = torch.zeros_like(p.data)

                m, sm = state["m"], state["sm"]
                t = state["t"] + 1

                grad = p.grad.data

                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # Update biased second raw moment estimate
                sm.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                m_hat = m / (1.0 - beta1**t)
                sm_hat = sm / (1.0 - beta2**t)

                # Update parameters
                p.data.addcdiv_(m_hat, torch.sqrt(sm_hat) + eps, value=-lr)

                # Weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                state["t"] = t
        return loss


```

**这个优化器的代价是内存。假设参数用 float32，单份参数需要 4 字节；梯度还要一份；AdamW 的 `m` 和 `v` 又各要一份。只看参数相关状态，就已经是参数量的约 4 倍内存，再加上 activation、logits、临时张量和 checkpoint。理解这个成本，才能解释为什么同一个模型在推理时能装下，训练时却爆显存。**

### Consine LR Scheduler

学习率决定每一步走多远，它通常比很多结构细节更先影响训练成败。一个实用策略是 warmup 加 cosine decay：前几个 step 从 0 线性升到最大学习率，让 moment state 和模型激活先进入稳定范围；之后用 cosine 逐步衰减到最小学习率，让训练后期更细地收敛。这里的调度器只是一个纯函数：输入当前 step 和几个超参数，输出本 step 的学习率。

```python
def cos_lr_scheduler(it: int, warmup_iters: int, cos_cycle_iters: int, lr_min: float, lr_max: float) -> float:
    if it <= warmup_iters:
        return lr_max * it / warmup_iters
    elif warmup_iters < it < cos_cycle_iters:
        return lr_min + 0.5 * (lr_max - lr_min) * (
            1 + math.cos(math.pi * (it - warmup_iters) / (cos_cycle_iters - warmup_iters))
        )
    else:
        return lr_min


```

### Gradient Clip

梯度裁剪解决的是另一类问题：偶发 batch 可能产生非常大的梯度，把参数一步推到坏区域。

**L2 范数梯度裁剪**（也叫`max-norm`裁剪）：计算所有参数梯度的全局 L2 范数，若范数超过设定的`max_norm`阈值，则按比例缩小所有梯度，确保梯度的整体范数不超过阈值，从而避免梯度爆炸问题。

```python
def gradient_clip(params: Iterable[torch.nn.Parameter], max_norm: float, delta=1e-6):
    with torch.no_grad():
        grads = [p.grad for p in params if p.grad is not None]
        total_norm = torch.linalg.norm(torch.stack([torch.linalg.norm(g.detach()) for g in grads]))
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + delta)
            for g in grads:
                g.detach().mul_(clip_coef)

```



## 资源核算：矩阵乘法决定了大部分 FLOPs

实现完模型后，除了它能不能跑，还要能算它为什么贵。Transformer 里的主要 FLOPs 来自矩阵乘法，而矩阵乘法 $A\in\mathbb{R}^{m\times n}$ 乘 $B\in\mathbb{R}^{n\times p}$ 大约需要 $2mnp$ FLOPs。根据这个规则，可以把模型前向拆成一张账表。

| 组件 | 主要矩阵乘法 | 增长直觉 |
|-|-|-|
| Q/K/V projections | $(B T, D)\times(D,D)$ 三次 | 随 token 数和 $D^2$ 线性增长 |
| Attention scores | $QK^\top$，每个 head 是 $(T,d_{\mathrm{head}})\times(d_{\mathrm{head}},T)$ | 随 $T^2$ 增长，长上下文时变重 |
| Attention values | $\operatorname{softmax}(QK^\top)V$ | 同样随 $T^2$ 增长 |
| Output projection | $(B T,D)\times(D,D)$ | 随 token 数和 $D^2$ 线性增长 |
| SwiGLU FFN | $W_1$、$W_3$ 上投影和 $W_2$ 下投影 | 通常是 block 内最大 dense compute 来源之一 |
| LM head | $(B T,D)\times(D,V)$ | 词表很大时不可忽略 |

这张账表会直接影响后续实验判断。增大 $d_{\mathrm{model}}$ 会放大大多数 dense projection；增大 $\mathrm{context_length}$ 会让 attention score/value 的 $T^2$ 项快速抬头；增大 $V$ 会抬高 LM head 和 cross-entropy 的成本。也正因为这样，训练一个看似“小”的模型时，如果数据读取、验证 loss 或 checkpoint 写入不当，真实 wall-clock 也可能被非模型部分拖住。
