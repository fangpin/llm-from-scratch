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

`llm/bpe_tokenizer.py` 从零实现了一个 byte-pair tokenizer。这个文件同时负责离线训练和在线 encode/decode，因此它是整个仓库里第一个真正的系统边界。

## 内部状态与不变量

tokenizer 维护四份核心状态：

- `vcab2id: dict[bytes, int]`
- `id2vcab: dict[int, bytes]`
- `merges: list[tuple[bytes, bytes]]`
- `merge_ranks: dict[tuple[bytes, bytes], int]`

其中：

- `vcab2id` 与 `id2vcab` 是运行时词表
- `merges` 记录训练阶段学到的 merge 顺序
- `merge_ranks` 是从 `merges` 派生出来的，编码时真正依赖的是它

这里最重要的实现选择是：词表的规范表示是 `bytes`，不是 Python 字符串。因为：

- 初始词表就是 256 个 byte
- merge 规则天然作用在 byte 片段上
- decode 本质上只是 byte 拼接再转回 UTF-8

所以这个 tokenizer 不需要再发明一层单独的“subword 对象”。

## 正则预切分

真正做 BPE 之前，代码先用这条正则做预切分：

```python
self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s"""
```

这会把文本粗分成几类片段：

- 英文缩写尾缀
- 字母串
- 数字串
- 标点串
- 空白串

这意味着 merge 学习不是在一整条原始 Unicode 流上进行，而是在这些预切分好的片段内部进行。这样可以减少跨词边界、跨标点边界的病态 merge。

## Special Token

如果传入了 `special_tokens`，`_pre_token()` 会先按这些 byte 序列切开原文。命中 special token 的部分不会再进入正则切分，而是直接作为原子 token 保留。

这就是为什么 `<|endoftext|>` 这种分隔符能稳定参与：

- tokenizer 训练
- 语料编码
- 生成停止条件

如果没有这层分支，special token 会被拆成多个 byte，再被 merge 规则不稳定地重组。

## 训练阶段

`train()` 实现了完整的 BPE 构建流程。

### 第一步：初始化基础词表

函数先安装：

1. 所有 special token
2. 全部 256 个原始 byte

因此在任何 merge 学习发生前，tokenizer 已经拥有完整的 fallback 词表。

### 第二步：统计 token tuple 频次

输入文件会整体读入内存，再做 `_pre_token()`。对每个普通预切分片段，代码构造：

```python
bs = tuple(bytes([b]) for b in pre_token)
```

`tokens_cnt` 统计的是这些 byte tuple 的出现次数。它可以理解成“词频表”，只是这里的“词”是“正则预切分后、按 byte 展开的片段”。

### 第三步：统计相邻 pair

然后代码遍历 `tokens_cnt`，累计所有相邻 pair 的频次到 `pair_cnt`。这就是 BPE 选择下一次 merge 所需的充分统计量。

### 第四步：merge 循环

只要当前词表大小还没达到 `vocab_size`，代码就会不断：

1. 从 `pair_cnt` 里选出频次最高的 pair
2. 把它加入 `self.merges`
3. 为两个 byte 片段拼接后的新 token 分配 id
4. 重写所有受影响的 token tuple
5. 扣掉旧 pair 计数
6. 增加新 pair 计数

这里的关键优化是 `update_pair_counts()`。它不是每轮 merge 后都重新全量扫描统计 pair，而是做增量更新。

这也是整个文件里最核心的算法优化点。

### 第五步：构造运行时映射

训练结束后，代码重建：

```python
self.id2vcab = {id: vcab for vcab, id in self.vcab2id.items()}
self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
```

之后推理阶段就不再依赖频次统计，只依赖 merge 排序。

## 编码阶段

`encode()` 使用训练好的 merge 顺序处理新文本。

### 第一步：转 bytes 并预切分

文本先按 UTF-8 编码，再交给 `_pre_token()`。如果片段本身就是 special token，就直接查 id。

### 第二步：普通片段回到单 byte 状态

对普通预切分片段，初始表示是：

```python
tokens = tuple(bytes([c]) for c in pre_token)
```

也就是说，无论训练时学了多少 merge，编码总是从单 byte 序列重新开始。

### 第三步：贪心回放 merge 顺序

编码循环会不断：

1. 枚举当前所有相邻 pair
2. 找到 `merge_ranks` 最小的 pair
3. 合并这一对
4. 继续下一轮

直到当前 token 序列里没有任何 pair 出现在 `merge_ranks` 里。

这里的要点是：编码时不会重新统计 pair 频率。训练阶段学的是一套全局 merge 顺序，编码阶段只是贪心回放这套顺序。

## 解码阶段

`decode()` 的逻辑非常直接：

1. 按 token id 从 `id2vcab` 查回 bytes
2. 把所有 bytes 拼起来
3. 按 UTF-8 decode 回字符串

如果某个 id 不存在，会退回到 replacement character 的 bytes：

```python
b"\xef\xbf\xbd"
```

这样 decode 遇到损坏输入时不会直接崩掉。

## 持久化与恢复

`save()` 序列化三项：

- `merge`
- `id2vcab`
- `special_tokens`

它不会把所有派生字段都写进去。`load()` 读取后再调用 `from_pretrained()`，重建：

- `vcab2id`
- `merge_ranks`

所以保存产物比较小，但仍然足够恢复完整运行时契约。

## 在仓库里的角色

`__main__` 入口已经把这个 tokenizer 接进了主训练准备流程：

1. 从 `llm.args` 读取参数
2. 训练 tokenizer
3. 保存 tokenizer checkpoint
4. 编码训练文本
5. 把 token id 保存成 `.npy`
6. 编码验证文本
7. 再保存成 `.npy`

所以 tokenizer 不是边缘工具，而是 `llm/training.py` 消费的数据生产者。

## 设计取舍

当前实现有几个很明确的取舍：

- 训练时会把输入语料整体读入内存
- 编码阶段每轮都线性扫描相邻 pair
- 变量名确实拼成了 `vcab`，文档也应当跟着真实代码走

这些取舍在仓库目标下是合理的：优先讲清楚机制，而不是先做工业级规模优化。重要的是，BPE 的每个阶段都在可读代码里摊开了。
