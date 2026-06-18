export type DocLocale = "en" | "zh";
export type DocGroupId = "core-stack" | "scale-performance" | "alignment-workflows";

export interface DocEntry {
  title: string;
  summary: string;
  slug: string;
  locale: DocLocale;
  group: DocGroupId;
  order: number;
  translationKey: string;
  sourceFiles: string[];
  sourceDocs?: string[];
}

export const docGroupOrder: DocGroupId[] = [
  "core-stack",
  "scale-performance",
  "alignment-workflows",
];

const docsByLocale: Record<DocLocale, DocEntry[]> = {
  en: [
    {
      title: "Docs",
      summary: "Module-based documentation for the repository, organized around implementation layers instead of tutorial order.",
      slug: "index",
      locale: "en",
      group: "core-stack",
      order: 0,
      translationKey: "docs-index",
      sourceFiles: ["README.md", "README_cn.md"],
      sourceDocs: ["docs/1.md"],
    },
    {
      title: "Project Overview",
      summary: "A system map of the repository, from tokenizer and transformer internals to kernels, distributed training, and alignment workflows.",
      slug: "project-overview",
      locale: "en",
      group: "core-stack",
      order: 1,
      translationKey: "project-overview",
      sourceFiles: [
        "README.md",
        "llm/transformer.py",
        "llm/bpe_tokenizer.py",
        "llm/training.py",
        "kernel/flash_attention_triton.py",
        "parallel/ddp.py",
        "alignment/sft.py",
        "alignment/train_rl.py",
      ],
      sourceDocs: ["docs/1.md"],
    },
    {
      title: "Tokenizer and Vocabulary",
      summary: "How the repository trains and applies a byte-pair tokenizer, from regex pre-tokenization to merge learning and byte-level encode/decode.",
      slug: "tokenizer",
      locale: "en",
      group: "core-stack",
      order: 2,
      translationKey: "tokenizer",
      sourceFiles: ["llm/bpe_tokenizer.py", "llm/args.py"],
      sourceDocs: ["docs/1.md"],
    },
    {
      title: "Transformer Core",
      summary: "Decoder-only transformer internals implemented in llm/transformer.py, including RMSNorm, RoPE, SwiGLU, custom attention, and custom loss.",
      slug: "transformer-core",
      locale: "en",
      group: "core-stack",
      order: 3,
      translationKey: "transformer-core",
      sourceFiles: ["llm/transformer.py"],
      sourceDocs: ["docs/1.md"],
    },
    {
      title: "Training Loop and Checkpointing",
      summary: "How llm/training.py slices data, initializes distributed execution, runs validation, applies schedules, and saves checkpoints.",
      slug: "training-loop",
      locale: "en",
      group: "core-stack",
      order: 4,
      translationKey: "training-loop",
      sourceFiles: ["llm/training.py", "llm/checkpoint.py", "llm/generating.py", "llm/args.py"],
      sourceDocs: ["docs/1.md"],
    },
    {
      title: "Flash Attention and Kernel Optimization",
      summary: "How the Triton Flash Attention path in kernel/ reduces attention memory pressure and how it is validated against a reference implementation.",
      slug: "flash-attention",
      locale: "en",
      group: "scale-performance",
      order: 5,
      translationKey: "flash-attention",
      sourceFiles: [
        "kernel/flash_attention_triton.py",
        "kernel/flash_attention_mock.py",
        "bench_mark/bench_mark_flash_attention.py",
        "bench_mark/bench_mark_atten.py",
      ],
      sourceDocs: ["docs/2.md"],
    },
    {
      title: "Distributed Training and Sharded Optimizer",
      summary: "How parallel/ddp.py and parallel/sharded_optimizer.py synchronize gradients, bucket communication, and reduce optimizer-state memory.",
      slug: "distributed-training",
      locale: "en",
      group: "scale-performance",
      order: 6,
      translationKey: "distributed-training",
      sourceFiles: ["parallel/ddp.py", "parallel/sharded_optimizer.py", "llm/training.py"],
      sourceDocs: ["docs/3.md", "docs/technical_article3.md"],
    },
    {
      title: "Data Processing Pipeline",
      summary: "The preprocessing stack in data_processing/, from HTML extraction and language filtering to deduplication, PII masking, harmful-content checks, and quality classification.",
      slug: "data-pipeline",
      locale: "en",
      group: "scale-performance",
      order: 7,
      translationKey: "data-pipeline",
      sourceFiles: [
        "data_processing/html_process.py",
        "data_processing/language_identification.py",
        "data_processing/quality_filter.py",
        "data_processing/deduplicate.py",
        "data_processing/mask_pii.py",
        "data_processing/harmful_detect.py",
        "data_processing/quality_classfier.py",
      ],
      sourceDocs: ["docs/4.md"],
    },
    {
      title: "Supervised Fine-Tuning on gsm8k",
      summary: "How alignment/sft.py turns gsm8k examples into prompt-completion training data and measures accuracy plus format compliance.",
      slug: "sft-gsm8k",
      locale: "en",
      group: "alignment-workflows",
      order: 8,
      translationKey: "sft-gsm8k",
      sourceFiles: [
        "alignment/sft.py",
        "alignment/dataset.py",
        "alignment/r1_prompt.py",
        "alignment/evaluate.py",
        "alignment/drgrpo_grader.py",
        "alignment/args.py",
      ],
      sourceDocs: ["docs/5-sft.md"],
    },
    {
      title: "Reinforcement Learning Fine-Tuning on gsm8k",
      summary: "How the repo implements reward shaping, grouped normalization, GRPO-style clipping, and multi-GPU role separation for RLFT.",
      slug: "rlft-gsm8k",
      locale: "en",
      group: "alignment-workflows",
      order: 9,
      translationKey: "rlft-gsm8k",
      sourceFiles: [
        "alignment/train_rl.py",
        "alignment/grpo.py",
        "alignment/drgrpo_grader.py",
        "alignment/evaluate.py",
        "alignment/args.py",
      ],
      sourceDocs: ["docs/qwen25-math-gsm8k-rl-finetune.md"],
    },
  ],
  zh: [
    {
      title: "文档",
      summary: "站内的模块化项目文档，按实现层次组织，而不是按教程顺序组织。",
      slug: "index",
      locale: "zh",
      group: "core-stack",
      order: 0,
      translationKey: "docs-index",
      sourceFiles: ["README.md", "README_cn.md"],
      sourceDocs: ["docs/1.md"],
    },
    {
      title: "项目总览",
      summary: "从 tokenizer、Transformer、训练循环到 kernel、分布式和对齐流程的整体仓库地图。",
      slug: "project-overview",
      locale: "zh",
      group: "core-stack",
      order: 1,
      translationKey: "project-overview",
      sourceFiles: [
        "README.md",
        "llm/transformer.py",
        "llm/bpe_tokenizer.py",
        "llm/training.py",
        "kernel/flash_attention_triton.py",
        "parallel/ddp.py",
        "alignment/sft.py",
        "alignment/train_rl.py",
      ],
      sourceDocs: ["docs/1.md"],
    },
    {
      title: "Tokenizer 与词表",
      summary: "解释 llm/bpe_tokenizer.py 如何从正则预切分、byte 初始化、merge 学习一路走到 encode/decode。",
      slug: "tokenizer",
      locale: "zh",
      group: "core-stack",
      order: 2,
      translationKey: "tokenizer",
      sourceFiles: ["llm/bpe_tokenizer.py", "llm/args.py"],
      sourceDocs: ["docs/1.md"],
    },
    {
      title: "Transformer 核心",
      summary: "解释 llm/transformer.py 中的 decoder-only 结构，包括 RMSNorm、RoPE、SwiGLU、自定义 attention 和自定义 loss。",
      slug: "transformer-core",
      locale: "zh",
      group: "core-stack",
      order: 3,
      translationKey: "transformer-core",
      sourceFiles: ["llm/transformer.py"],
      sourceDocs: ["docs/1.md"],
    },
    {
      title: "训练循环与 Checkpoint",
      summary: "解释 llm/training.py 如何取 batch、初始化多卡环境、执行验证、更新学习率并保存 checkpoint。",
      slug: "training-loop",
      locale: "zh",
      group: "core-stack",
      order: 4,
      translationKey: "training-loop",
      sourceFiles: ["llm/training.py", "llm/checkpoint.py", "llm/generating.py", "llm/args.py"],
      sourceDocs: ["docs/1.md"],
    },
    {
      title: "Flash Attention 与 Kernel 优化",
      summary: "解释 kernel/ 中的 Triton Flash Attention 路径、参考实现和性能对比脚本。",
      slug: "flash-attention",
      locale: "zh",
      group: "scale-performance",
      order: 5,
      translationKey: "flash-attention",
      sourceFiles: [
        "kernel/flash_attention_triton.py",
        "kernel/flash_attention_mock.py",
        "bench_mark/bench_mark_flash_attention.py",
        "bench_mark/bench_mark_atten.py",
      ],
      sourceDocs: ["docs/2.md"],
    },
    {
      title: "分布式训练与 Sharded Optimizer",
      summary: "解释 parallel/ddp.py 和 parallel/sharded_optimizer.py 如何同步梯度、做 bucket 通信并分摊优化器状态内存。",
      slug: "distributed-training",
      locale: "zh",
      group: "scale-performance",
      order: 6,
      translationKey: "distributed-training",
      sourceFiles: ["parallel/ddp.py", "parallel/sharded_optimizer.py", "llm/training.py"],
      sourceDocs: ["docs/3.md", "docs/technical_article3.md"],
    },
    {
      title: "数据处理流水线",
      summary: "解释 data_processing/ 中从 HTML 提取、语言过滤、去重到 PII masking 和质量分类的完整预处理路径。",
      slug: "data-pipeline",
      locale: "zh",
      group: "scale-performance",
      order: 7,
      translationKey: "data-pipeline",
      sourceFiles: [
        "data_processing/html_process.py",
        "data_processing/language_identification.py",
        "data_processing/quality_filter.py",
        "data_processing/deduplicate.py",
        "data_processing/mask_pii.py",
        "data_processing/harmful_detect.py",
        "data_processing/quality_classfier.py",
      ],
      sourceDocs: ["docs/4.md"],
    },
    {
      title: "gsm8k 上的监督微调",
      summary: "解释 alignment/sft.py 如何构造 prompt-completion 数据、做 completion-only loss，并评估准确率与格式遵循率。",
      slug: "sft-gsm8k",
      locale: "zh",
      group: "alignment-workflows",
      order: 8,
      translationKey: "sft-gsm8k",
      sourceFiles: [
        "alignment/sft.py",
        "alignment/dataset.py",
        "alignment/r1_prompt.py",
        "alignment/evaluate.py",
        "alignment/drgrpo_grader.py",
        "alignment/args.py",
      ],
      sourceDocs: ["docs/5-sft.md"],
    },
    {
      title: "gsm8k 上的强化学习微调",
      summary: "解释仓库如何实现 reward 设计、group normalization、GRPO clip loss 和多 GPU 角色拆分。",
      slug: "rlft-gsm8k",
      locale: "zh",
      group: "alignment-workflows",
      order: 9,
      translationKey: "rlft-gsm8k",
      sourceFiles: [
        "alignment/train_rl.py",
        "alignment/grpo.py",
        "alignment/drgrpo_grader.py",
        "alignment/evaluate.py",
        "alignment/args.py",
      ],
      sourceDocs: ["docs/qwen25-math-gsm8k-rl-finetune.md"],
    },
  ],
};

export function getAllDocs(locale: DocLocale) {
  return docsByLocale[locale];
}

export function getDocOverview(locale: DocLocale) {
  return getAllDocs(locale).find((entry) => entry.slug === "index");
}

export function getDocs(locale: DocLocale) {
  return getAllDocs(locale).filter((entry) => entry.slug !== "index");
}

export function getDoc(locale: DocLocale, slug: string) {
  return getDocs(locale).find((entry) => entry.slug === slug);
}

export function getDocsByGroup(locale: DocLocale) {
  const docs = getDocs(locale);
  return docGroupOrder.map((group) => ({
    group,
    entries: docs.filter((entry) => entry.group === group),
  }));
}

export function getDocHref(locale: DocLocale, slug: string) {
  return locale === "en" ? `/docs/${slug}` : `/zh/docs/${slug}`;
}

export function getDocSiblings(locale: DocLocale, slug: string) {
  const docs = getDocs(locale);
  const currentIndex = docs.findIndex((entry) => entry.slug === slug);
  if (currentIndex === -1) {
    return { previous: undefined, next: undefined };
  }

  return {
    previous: docs[currentIndex - 1],
    next: docs[currentIndex + 1],
  };
}

export function getRelatedDocs(locale: DocLocale, slug: string) {
  const current = getDoc(locale, slug);
  if (!current) {
    return [];
  }

  return getDocs(locale)
    .filter((entry) => entry.group === current.group && entry.slug !== slug)
    .slice(0, 3);
}
