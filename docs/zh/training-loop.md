---
title: 训练循环与 Checkpoint
summary: 解释 llm/training.py 如何取 batch、初始化多卡环境、执行验证、更新学习率并保存 checkpoint。
slug: training-loop
locale: zh
group: core-stack
order: 4
translationKey: training-loop
sourceFiles:
  - llm/training.py
  - llm/checkpoint.py
  - llm/generating.py
  - llm/args.py
sourceDocs:
  - docs/1.md
---

# 训练循环与 Checkpoint

预训练主路径在 `llm/training.py`。这个文件虽然不长，但覆盖了从进程组初始化到 checkpoint 保存的完整执行生命周期。下面按真实控制流来拆。

## 配置边界

公开入口是 `train()`。它从 `llm.args` 读取全部超参数，先断言：

```python
args.batch_size % args.world_size == 0
```

然后创建 checkpoint 和日志目录，再通过：

```python
mp.spawn(_train, args=(args.world_size, args.backend, args), nprocs=args.world_size, join=True)
```

启动多进程训练。

所以每个 rank 都会执行同一个 `_train()`，只是在 rank id 和设备分配上不同。

## 可复现性路径

`set_random_seed(seed, rank)` 会把全局 seed 与 rank 相加，得到 rank 级 seed，然后设置：

- `torch`
- `torch.cuda`
- `numpy`
- Python `random`

同时还会强制：

- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

这是一个很实际的折中：不同 rank 用不同 seed，但在固定 world size 和启动方式下，整体运行仍然可复现。

## Batch 构造

`get_batch()` 是整个数据加载的核心原语。它接收 token id 数组 `x`，随机采样起点：

```python
ix = torch.randint(0, len(x) - context_length, (batch_size,))
```

对每个起点构造：

- `input_seqs = x[i : i + context_length]`
- `target_seqs = x[i + 1 : i + 1 + context_length]`

这正是 next-token prediction 的标准自回归监督：输入与目标只差一个 token 的右移。

两个实现细节很重要：

- 数据可以通过 `np.load(..., mmap_mode="r")` 做内存映射
- 返回前已经把 tensor 移到目标设备

因此训练循环不会把整个数据集一次性搬进 GPU。

## 进程组初始化

`_setup_process_group()` 负责多卡运行时：

- 设置 `MASTER_ADDR=localhost`
- 设置 `MASTER_PORT=12390`
- 设置 `NCCL_DEBUG=NONE`
- 通过 `rank % device_count` 选择本地 CUDA device
- 调用 `dist.init_process_group(...)`

这说明这个脚本主要面向本地多卡，而不是隐藏在某个集群启动器后面。

`_cleanup_process_group()` 则会在销毁进程组前插入一次 barrier，避免不同 rank 提前拆掉通信状态。

## 模型构造

在 `_train()` 中，模型通过参数显式构造：

```python
model = Transformer(
    d_model=args.d_model,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    vocab_size=args.vocab_size,
    num_layers=args.num_layers,
    max_seq_len=args.max_seq_len,
    device=device,
)
```

如果 `world_size > 1`，模型会立刻被包进自定义 `DDP`。

这不是表面包装，而是会改变后续梯度同步的执行时机。

## 优化器选择

优化器会根据 world size 分叉：

- 单 rank：自定义 `AdamW`
- 多 rank：`ShardedOptimizer(..., AdamW, ...)`

loss 始终使用自定义 `CrossEntropyLoss`。

也就是说，高层训练逻辑没变，变化的是 optimizer state 的拥有方式和同步方式。

## 验证路径

验证只在 rank 0 上执行：

1. 切到 eval 模式
2. 跑 100 个验证 batch
3. 求平均 loss
4. 打印结果
5. 写入 TensorBoard
6. 切回 train 模式

这样做的好处是日志集中，不会让每个 rank 都重复输出一份验证结果。

这个验证路径故意保持简单：没有额外的分布式 metric reduce，只有 rank 0 作为参考视角。

## 主训练步

每次 iteration 的执行顺序是：

1. 取一个训练 batch
2. 前向得到 logits
3. 计算交叉熵 loss
4. `optimizer.zero_grad()`
5. `loss.backward()`
6. `gradient_clip(model.parameters(), max_norm=1.0)`
7. 如果是多卡，调用 `model.finish_gradient_sync()`
8. `optimizer.step()`

最关键的一步是：

```python
if args.world_size > 1:
    model.finish_gradient_sync()
```

因为自定义 DDP 用的是异步 bucket all-reduce，`backward()` 返回并不自动意味着所有梯度都已经全局同步完成。训练脚本把“同步真正结束”的时刻显式写了出来。

## 学习率调度

每次参数更新后，脚本都会重新计算学习率：

```python
cos_lr_scheduler(
    it=i,
    warmup_iters=args.warmup_iters,
    cos_cycle_iters=args.cos_cycle_iters,
    lr_min=args.lr_min,
    lr_max=args.lr_max,
)
```

得到的新学习率会写回所有 optimizer param group。

所以学习率调度器是独立于 optimizer 的显式函数，而不是藏在 optimizer 内部。

## 日志

rank 0 会往 `SummaryWriter` 写三类指标：

- `loss_train`
- `lr`
- `val_loss`

同时每到 `log_interval` 还会在控制台打印训练 loss 和当前学习率。

这个监控面不大，但足够看：

- 训练是否稳定
- 学习率是否按预期变化
- 是否有明显过拟合或发散

## Checkpoint 格式

每到 `checkpoint_interval` 且 `i > 0`，脚本调用：

```python
save_checkpoint(model, optimizer, i, path)
```

`llm/checkpoint.py` 序列化的是一个小字典：

- `model`
- `optimizer`
- `iteration`

这里没有额外 trainer metadata，也没有复杂封装层。

## Checkpoint 如何被消费

`llm/generating.py` 直接消费这个格式：

1. 重新构造相同结构的 `Transformer`
2. 调用 `load_checkpoint(...)`
3. 加载 tokenizer
4. 编码 prompt
5. 每步裁到 `model.max_seq_len`
6. 取最后一个位置的 logits
7. 应用 temperature
8. 应用 top-p 截断
9. 采样下一个 token
10. 遇到 end-of-text token 停止

这很重要，因为它证明训练产物不需要额外转换流程就能进入推理。

## 多卡路径的具体变化

当 `world_size > 1` 时，运行上有两个关键变化：

- 每个 rank 实际使用 `mini_batch_size = args.batch_size // world_size`
- 模型与优化器的同步路径被激活

但训练语义本身没变：仍然是在同一份 tokenized corpus 上做随机固定窗口的 next-token prediction。

## 这个文件真正优化的是什么

按框架标准看，`llm/training.py` 功能并不算多。但它的价值在于职责非常清楚：

- 一个函数切 batch
- 一个函数初始化进程组
- 一个函数做单个 rank 的训练体
- 一个顶层 launcher
- 一种 checkpoint 格式

因此你可以很快回答这类实现问题：

- 数据是什么时候进 GPU 的
- 梯度是什么时候裁剪的
- 分布式通信什么时候被强制等完
- 学习率什么时候变化
- 到底序列化了什么

对于一个强调“把机制讲清楚”的仓库，这种直接性比更复杂的 trainer 抽象更有价值。
