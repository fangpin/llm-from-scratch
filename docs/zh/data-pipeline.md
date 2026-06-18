---
title: 数据处理流水线
summary: 解释 data_processing/ 中从 HTML 提取、语言过滤、去重到 PII masking 和质量分类的完整预处理路径。
slug: data-pipeline
locale: zh
group: scale-performance
order: 7
translationKey: data-pipeline
sourceFiles:
  - data_processing/html_process.py
  - data_processing/language_identification.py
  - data_processing/quality_filter.py
  - data_processing/deduplicate.py
  - data_processing/mask_pii.py
  - data_processing/harmful_detect.py
  - data_processing/quality_classfier.py
sourceDocs:
  - docs/4.md
---

# 数据处理流水线

`data_processing/` 是一组把原始语料整理成 tokenizer 和训练脚本可消费形态的小工具。仓库没有提供一个“大而全”的单体预处理入口，而是把每个阶段拆成独立模块。

## 推荐执行顺序

一个比较合理的处理顺序是：

1. `html_process.py`
2. `language_identification.py`
3. `quality_filter.py`
4. `deduplicate.py`
5. `mask_pii.py`
6. `harmful_detect.py`
7. `quality_classfier.py`

这个顺序背后有明确的成本逻辑：

- 先做结构清理
- 再做便宜、确定的规则过滤
- 后面再跑更重的去重与分类模型

代码保持拆分，就是为了让你能按语料特点裁剪或重排这些阶段。

## `html_process.py`：HTML 到纯文本

`extract_text_from_html(html_content: bytes)` 的职责很窄，但很清楚：

1. 用 `detect_encoding` 检测编码
2. 把原始 bytes decode 成字符串
3. 用 `extract_plain_text` 提取可见文本

所以它不是“完整解析 Common Crawl”的入口，而是一个更基础的原语：给我 HTML bytes，我给你尽量稳健的纯文本。

文件里虽然 import 了 `ArchiveIterator` 和 `WarcRecordType`，暗示它面向 WARC / crawl 数据，但 helper 本身只停留在“单个 HTML 文档”这一层。

## `language_identification.py`：FastText 包装器 + 兼容性补丁

这个文件其实同时做了两件事。

第一件事是在 import 时 monkey-patch `fasttext.FastText._FastText.predict`，绕开 NumPy 2 兼容问题。

第二件事才是定义 `LanguageIdentifier`，它会：

- 加载 `pre_trained/lid.176.bin`
- 把换行与回车替换成空格
- 调用 `model.predict`
- 去掉 FastText label 前缀 `__label__`

`identify(text, k=1)` 的返回值是：

- `k == 1` 时：`(label, prob)`
- `k > 1` 时：`(labels, probs)`

所以这个文件既是语言识别器，也是运行时兼容补丁。

## `quality_filter.py`：低成本启发式过滤

`QualityFilter` 是规则过滤阶段。初始化时会下载 NLTK 资源：

- `punkt`
- `punkt_tab`

它暴露四个过滤器：

- `pass_wc_filter`：token 数在 50 到 100000 之间
- `pass_word_len_filter`：平均 token 长度在 3 到 10 之间
- `pass_alphabetic_filter`：至少 80% 的 token 含字母字符
- `pass_ellipsis_filter`：以 `...` 结尾的行占比小于 30%

`pass_all_filters(content)` 会先 `word_tokenize(content)`，再把这些规则 AND 起来。

这些规则不是在做语义质量判断，而是在用低成本手段剔除明显噪声。

## `deduplicate.py`：精确去重与近似去重

这一层是整个预处理里算法成分最重的部分。

### 精确逐行去重

`exact_line_deduplicate(files, output_dir)` 会：

1. 读取所有文件的所有行
2. 统计 `hash(line)` 的出现次数
3. 只把唯一出现的行写到同名输出文件里

这适合去掉重复 header、模板化片段和大段完全一致的文本。

### `MinHashDeduplicator`

近似去重由 `MinHashDeduplicator` 负责。构造函数参数包括：

- `num_hashes`
- `num_bands`
- `n_gram`
- `jaccard_threshold`

它的大致流程是：

1. 标准化文本
2. 分词
3. 构造 n-gram shingles
4. 用多个带 seed 的 SHA256 hasher 生成 MinHash signature
5. 把文档放进 candidate bucket
6. 用精确 Jaccard 相似度验证 candidate pair
7. 只把保留下来的文件写入输出目录

### 标准化与 shingles

`normalize()` 不只是 lower-case，它依次做：

- Unicode NFD 标准化
- 用 `unidecode` 去掉重音
- 全部转小写
- 删除标点，只保留字母、数字与空格
- 压缩多余空白

之后 `shingle()` 会对标准化文本分词，再构造连续 token n-gram 集合。

这里一个实现细节值得注意：`_tokenize()` 自己也会下载 `punkt` 和 `punkt_tab`，因此如果环境没缓存好，资源准备可能在运行时重复发生。

### signature 与 bucket

`signatures()` 对每个 hasher 取所有 shingle hash 的最小值，得到 MinHash signature。

`deduplicate()` 构建 bucket 的方式是滑动窗口式的：

```python
for i in range(self.num_hashes - self.num_bands):
    bucket_id = (i, tuple(signatures[i : i + self.num_bands]))
```

所以它不是标准教材里那种固定 band 分块，而是对 signature 做重叠切片后分桶。

### candidate 验证与去重决策

拿到 candidate pair 后，代码会对标准化 shingle 集合计算精确 Jaccard 相似度。如果超过阈值，就把其中一个文件标记为 duplicate。

当前实现固定把排序后二元组里的第一个文件加入 `deduplicates`，因此 survivor 选择虽然确定，但也会受文件排序影响。

## `mask_pii.py`：基于正则的脱敏

`PIIMasker` 在初始化时编译了三类正则：

- 邮箱地址
- 电话号码
- IPv4 地址

每个 masking 方法都返回：

```python
(masked_content, count)
```

并把命中内容替换为显式占位符，例如：

- `|||EMAIL_ADDRESS|||`
- `|||PHONE_NUMBER|||`
- `|||IP_ADDRESS|||`

这种做法的好处是：不必整篇丢弃文档，仍然保留大部分可训练文本，同时移除敏感表面形式。

## `harmful_detect.py`：安全分类器

这个文件定义了两个基于 FastText 的包装器：

- `NSFWDetector`
- `ToxicDetector`

两者都做同样的几步：

- 初始化时加载预训练二进制模型
- 推理前把换行替换为空格
- 把输出 label 的 `__label__` 前缀去掉

所以这一层和语言识别层的风格基本一致：薄封装、统一 label 清理、模型文件由构造函数决定。

## `quality_classfier.py`：学习式质量打分

最后一层是模型化质量分类。

这个文件既包含：

- `train(...)`：用标注文本训练 FastText supervised classifier，并保存到 `checkpoints/quality_classifier.bin`

也包含：

- `QualityClassifier.identify(...)`：加载保存好的模型并返回归一化后的标签与置信度

`identify()` 还处理了几个边界情况：

- 空字符串直接返回 `("low-quality", 0.0)`
- 未知标签也回退到 `("low-quality", 0.0)`
- `high_quality` / `low_quality` 会归一化成带短横线的形式

仓库里的文件名确实拼成了 `quality_classfier.py`，文档也保持和真实路径一致。

## 它如何连接到仓库其他部分

这一层的出口在 tokenizer 之前：

- 清洗后的文本进入 `llm/bpe_tokenizer.py`
- 去重与质量过滤提高 token 化前的信息密度
- PII 与 harmful filter 决定模型究竟会从哪些内容里学习

把预处理留在 `llm/` 之外的好处是：你可以单独调整语料策略，而不必碰模型和训练代码。

## 设计取舍

当前这套预处理栈是显式的，而不是平台化封装：

- 没有统一 orchestrator 脚本
- 某些 NLTK 资源会在运行时下载
- 精确去重是基于文件和行哈希的
- 近似去重采用实用的 bucket heuristic，而不是完整服务化方案
- 基于模型的过滤依赖外部 FastText 二进制

这很符合仓库目标。这里展示的是一条真实预处理流水线需要的积木，而不是把所有逻辑藏进一个你看不见的命令里。
