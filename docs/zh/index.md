---
title: 文档
summary: 站内的模块化项目文档，按实现层次组织，而不是按教程顺序组织。
slug: index
locale: zh
group: core-stack
order: 0
translationKey: docs-index
sourceFiles:
  - README.md
  - README_cn.md
sourceDocs:
  - docs/1.md
---

这个文档区不是把仓库里的 Markdown 原样搬过来，而是把代码重新整理成一张实现地图。

每一章都尽量回答四个问题：

- 这一层代码的职责边界是什么
- 关键函数和类是怎么协作的
- 张量、文本或 checkpoint 是怎样流动的
- 当前实现有哪些真实限制，而不是概念上的“理想版本”

章节分成三层：

- 从原始文本到 decoder-only 模型的核心栈
- 让训练扩展到更大规模的性能与系统层
- 围绕 gsm8k 的 SFT 与 RLFT 对齐流程

两种阅读顺序都成立：

- 如果你先想看仓库全貌，从 `项目总览` 开始
- 如果你想沿着真实数据流往下读，从 `Tokenizer 与词表` 开始

可以直接从下方概览卡片跳到你最关心的模块。
