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

预训练主路径在 `llm/training.py`。这个文件负责的不只是一个训练 `for` 循环，而是整个训练生命周期：

- 随机种子
- 数据加载
- 进程组与设备初始化
- 模型构造
- 优化器选择
- 验证节奏
- 梯度同步
- 学习率调度
- checkpoint 保存

## Batch 构造

`get_batch()` 会从内存映射的 token id 数组里随机抽起点，然后构造：

- 输入序列 `x[i : i + context_length]`
- 目标序列 `x[i + 1 : i + 1 + context_length]`

这意味着训练目标始终是固定窗口内的 next-token prediction。

函数返回前就会把 tensor 移到指定 device，所以主训练循环里不需要再做一遍迁移。

## 进程组初始化

`_setup_process_group()` 会显式设置：

- `MASTER_ADDR=localhost`
- `MASTER_PORT=12390`
- `backend=nccl`

然后根据 rank 和可见 CUDA 设备数推导 local device。

这说明脚本主要面向本地多卡训练，而不是依赖某个隐藏的集群启动框架。

## 模型与优化器初始化

模型通过 `llm/args.py` 中的超参数构造 `Transformer`。

当 `world_size > 1` 时：

- 模型包装为自定义 `DDP`
- 优化器切换为 `ShardedOptimizer(..., AdamW, ...)`

当 `world_size == 1` 时：

- 直接用本地 `Transformer`
- 优化器为自定义 `AdamW`

所以单卡和多卡实际上共用一套主训练逻辑。

## 验证循环

每到 `val_interval`，rank 0 会：

1. 切到 eval 模式
2. 跑 100 个验证 batch
3. 求平均 loss
4. 写 TensorBoard
5. 切回 train 模式

这条验证路径不复杂，但足够稳定地给出训练健康度信号。

## 训练步

每个 iteration 的核心步骤是：

1. 取训练 batch
2. 前向得到 logits
3. 计算 loss
4. 清梯度
5. 反向传播
6. 做梯度裁剪
7. 如果是多卡，结束梯度同步
8. `optimizer.step()`

这里最关键的一句是：

```python
if args.world_size > 1:
    model.finish_gradient_sync()
```

它保证自定义 DDP 中异步 bucket 通信在参数更新前真正完成。

## 学习率调度

`cos_lr_scheduler()` 实现了：

- warmup 阶段线性升高
- 中段 cosine decay
- 后段固定到 `lr_min`

训练循环里每一步都会重新计算当前学习率并写回 optimizer param group。

## Checkpoint

`save_checkpoint()` 会保存三类内容：

- 模型参数
- 优化器状态
- 当前 iteration

`load_checkpoint()` 会恢复这些内容，并返回保存时的 iteration。

这也是 `llm/generating.py` 能直接复用训练结果的原因。

## 生成脚本如何衔接训练产物

`llm/generating.py` 的路径很清晰：

1. 构造和训练时同结构的 `Transformer`
2. 加载 checkpoint
3. 加载 tokenizer
4. 编码 prompt
5. 用 temperature + top-p 做自回归采样
6. 遇到 end-of-text token 时停止

所以 checkpoint 不是只为了存档，而是被仓库内的生成流程直接消费。

## 这一层在系统里的作用

这个训练循环的特点是足够朴素：

- mmap 数据读取
- 一个模型构造入口
- 一个验证循环
- 一个调度器
- 一种 checkpoint 格式

这种朴素不是能力不足，而是为了让读者能真正看清训练链路本身。
