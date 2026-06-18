---
title: 项目总览
summary: 从 tokenizer、Transformer、训练循环到 kernel、分布式和对齐流程的整体仓库地图。
slug: project-overview
locale: zh
group: core-stack
order: 1
translationKey: project-overview
sourceFiles:
  - README.md
  - llm/transformer.py
  - llm/bpe_tokenizer.py
  - llm/training.py
  - kernel/flash_attention_triton.py
  - parallel/ddp.py
  - alignment/sft.py
  - alignment/train_rl.py
sourceDocs:
  - docs/1.md
---

# 项目总览

这个仓库的目标不是把大模型训练流程藏在框架抽象后面，而是把现代 decoder-only 语言模型的关键层都摊开给读者看。

仓库主要覆盖七类能力：

- 分词
- Transformer 核心结构
- 训练与优化
- kernel 加速
- 分布式执行
- 数据预处理
- 对齐微调示例

## 模块边界

### `llm/`

这是核心模型层：

- `llm/bpe_tokenizer.py`：从零实现 BPE tokenizer
- `llm/transformer.py`：实现 embedding、RMSNorm、RoPE、attention、SwiGLU、自定义 loss 和优化器工具
- `llm/training.py`：负责训练、验证、日志和 checkpoint
- `llm/generating.py`：加载 checkpoint 并执行 top-p 生成
- `llm/checkpoint.py`：checkpoint 的保存与恢复

### `kernel/`

这是性能优化层：

- `flash_attention_triton.py`：Triton 版本 Flash Attention
- `flash_attention_mock.py`：参考实现
- `bench_mark/`：性能对比脚本

### `parallel/`

这是多卡训练层：

- `ddp.py`：自定义 DDP，同步梯度并做 bucket 通信
- `sharded_optimizer.py`：把优化器状态按参数分片到不同 rank

### `data_processing/`

这是语料处理层，包含：

- HTML 提取
- 语言识别
- 启发式质量过滤
- 去重
- PII masking
- 有害内容识别
- 质量分类

### `alignment/`

这是对齐实验层：

- `sft.py`：监督微调
- `train_rl.py`：强化学习微调
- `grpo.py`：奖励归一化和策略梯度损失
- `drgrpo_grader.py`：格式与答案判分

## 推荐阅读顺序

如果你想按实现边界理解整个仓库，一个更合理的阅读顺序是：

1. `Tokenizer 与词表`
2. `Transformer 核心`
3. `训练循环与 Checkpoint`
4. `Flash Attention 与 Kernel 优化`
5. `分布式训练与 Sharded Optimizer`
6. `数据处理流水线`
7. `gsm8k 上的 SFT`
8. `gsm8k 上的 RLFT`

这个顺序比从 README 线性往下读更接近真实系统结构。

## 为什么这个仓库值得读

它有三个比较少见的优点：

1. 不是只实现过时的最小 Transformer，而是覆盖了现代 decoder-only 常见模块。
2. 不只给出模型本体，还把数据、性能、分布式和对齐都接进来了。
3. 代码仍然保持在能直接阅读的规模，而不是被过度封装。

后面的章节会分别展开这些层。
