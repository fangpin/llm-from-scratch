---
title: Tokenizer 与词表
summary: 解释 llm/bpe_tokenizer.py 如何从正则预切分、byte 初始化、merge 学习一路走到 encode/decode。
slug: tokenizer
locale: zh
group: core-stack
order: 2
translationKey: tokenizer
sourceFiles:
  - llm/bpe_tokenizer.py
  - llm/args.py
sourceDocs:
  - docs/1.md
---

# Tokenizer 与词表

仓库里的 tokenizer 完整实现在 `llm/bpe_tokenizer.py`。它不是对第三方 tokenizer 的轻量封装，而是自己维护了从原始文本到 token id 的整个流程。

## 主要状态

这个实现维护了几组核心状态：

- `vcab2id`：bytes 到整数 id 的映射
- `id2vcab`：反向映射
- `merges`：训练阶段学到的 merge 顺序
- `merge_ranks`：编码阶段按优先级查 merge 的索引

这里用 bytes 作为词表单元，而不是直接把字符串当词表对象。这是因为 BPE 的底层合并本来就是围绕 byte 片段做的。

## 预切分

`_pre_token()` 用正则先把文本拆成较粗的片段，分别覆盖：

- 字母序列
- 数字序列
- 标点片段
- 空白片段

这样后续的 merge 学习不是在整段 Unicode 文本上直接做，而是在这些预切分后的片段内部做 byte 级组合。

## 特殊 Token

如果传入了 `special_tokens`，`_pre_token()` 会先把这些特殊 token 从 byte 流里切出来，并把它们保留为不可拆分的原子单位。

这也是为什么像 `<|endoftext|>` 这种分隔符可以稳定地参与训练、编码和生成。

## 训练流程

`train()` 的核心流程是：

1. 为 special tokens 分配 id
2. 为 256 个原始 byte 分配 id
3. 读取语料并做预切分
4. 统计词级 byte tuple
5. 统计相邻 pair 频次
6. 反复选择最频繁 pair 进行 merge，直到达到目标词表大小

每次 merge 发生时，代码都会显式更新：

- 新词元
- 旧 pair 计数
- 新 pair 计数

所以这里非常适合拿来理解 BPE 的训练机制，而不是只会调用现成 trainer。

## 编码

`encode()` 的思路是：

1. 文本先转成 bytes
2. 做预切分
3. 对每个普通片段按单字节拆开
4. 反复扫描相邻 pair
5. 找出 merge rank 最低的 pair
6. 合并直到没有可用 merge

这里的关键点是：编码时不是重新统计 pair 频率，而是严格复用训练阶段学出来的 merge 顺序。

## 解码

`decode()` 按 id 查回 bytes，拼接后再解码成 UTF-8 字符串。

遇到未知 id 时，会退回 replacement character 对应的 bytes，这样解码不会因为异常 id 直接崩掉。

## 持久化

`save()` 会写出三部分：

- merges
- `id2vcab`
- special tokens

`load()` 再把这些对象恢复出来，并重建派生映射。

这套结构正好够训练数据准备、checkpoint 复用和生成脚本使用。

## 在仓库中的位置

这个 tokenizer 不是一个边缘工具，而是整个训练链路的入口：

- `llm.training` 依赖它把语料转换成 token ids
- `llm.generating` 依赖它做 prompt 编码和结果解码
- 后续 checkpoint 的推理路径也都建立在同一套词表之上

所以理解 tokenizer，实际上就是理解整个模型系统的第一层输入边界。
