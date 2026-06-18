export const siteContent = {
  en: {
    navigation: {
      items: [
        { label: "Architecture", href: "/architecture" },
        { label: "SFT & RLFT", href: "/sft-rlft" },
        { label: "Benchmarks", href: "/benchmarks" },
        { label: "Docs", href: "/docs" },
      ],
      githubLabel: "GitHub",
      docsLabel: "Docs",
      localeSwitchLabel: "中文",
    },
    pages: {
      home: {
        title: "LLM from Scratch",
        description: "A bilingual GitHub Pages project site for a from-scratch decoder-only transformer in PyTorch.",
        eyebrow: "Decoder-only Transformer • PyTorch • From Scratch",
        subtitle: "Build, train, benchmark, and fine-tune a modern language model from first principles.",
        actions: [
          { label: "View on GitHub", href: "https://github.com/fangpin/llm-from-scratch", variant: "primary" },
          { label: "Explore Docs", href: "/docs", variant: "secondary" },
        ],
        proofMetrics: [
          { value: "1.56% -> 62.9%", label: "Zero-shot accuracy on gsm8k" },
          { value: "18.9% -> 100%", label: "Format compliance" },
          { value: "Tokenizer / Training / Parallel / Kernel", label: "Implemented stack" },
        ],
        portals: [
          { title: "Architecture", href: "/architecture", summary: "Tokenizer, transformer blocks, training loop, kernels, and distributed training." },
          { title: "SFT & RLFT", href: "/sft-rlft", summary: "Qwen2.5-Math-1.5B fine-tuning story with concrete results on gsm8k." },
          { title: "Benchmarks", href: "/benchmarks", summary: "Loss curves, learning-rate schedule, and performance context." },
          { title: "Docs", href: "/docs", summary: "Structured markdown chapters for tokenizer, transformer internals, training, scaling, and alignment." },
        ],
        capabilityTags: ["RoPE", "RMSNorm", "SwiGLU", "Flash Attention 2", "DDP", "SFT", "RLFT"],
      },
      architecture: {
        title: "Architecture",
        description: "See how tokenizer, transformer blocks, training, kernels, and distributed execution fit together.",
        sections: [
          { title: "Tokenizer", summary: "A from-scratch BPE tokenizer that learns merges and special tokens from raw text." },
          { title: "Transformer Core", summary: "Decoder-only blocks with RMSNorm, RoPE, SwiGLU, custom attention, and custom loss." },
          { title: "Training Loop", summary: "Checkpointing, validation, generation, cosine scheduling, and optimizer utilities." },
          { title: "Parallel + Kernel", summary: "Custom DDP, sharded optimizer, and Triton Flash Attention 2 support." },
          { title: "Data Pipeline", summary: "HTML extraction, filtering, deduplication, masking, and quality tooling." },
        ],
      },
      sftRlft: {
        title: "SFT & RLFT",
        description: "The strongest proof page in the repo: Qwen2.5-Math-1.5B on gsm8k with measured gains.",
        metrics: [
          { value: "1.56%", label: "Zero-shot baseline" },
          { value: "62.9%", label: "After SFT" },
          { value: "100%", label: "Format compliance" },
        ],
        steps: [
          "Load gsm8k train and test sets.",
          "Evaluate Qwen2.5-Math-1.5B zero-shot behavior.",
          "Run supervised fine-tuning for answer-format alignment.",
          "Run reinforcement fine-tuning to improve reward-shaped reasoning behavior.",
        ],
      },
      benchmarks: {
        title: "Benchmarks",
        description: "Training curves and implementation notes that ground the project in measurable behavior.",
        cards: [
          { value: "Flash Attention 2", label: "Triton kernel path included" },
          { value: "Loss Curve", label: "TinyStories training convergence" },
          { value: "LR Schedule", label: "Cosine warmup and decay" },
        ],
        charts: [
          { src: "/images/loss.png", alt: "Loss curve" },
          { src: "/images/lr.png", alt: "Learning rate schedule" },
        ],
      },
      docs: {
        title: "Docs",
        description: "Module-based project documentation rendered inside the site from mirrored English and Chinese Markdown chapters.",
        eyebrow: "Repository Map",
        overviewLead: "Read the repository by subsystem instead of by tutorial sequence. The docs are split into the core model stack, scale-and-performance topics, and alignment workflows.",
        systemMap: ["Tokenizer", "Transformer", "Training", "Scale", "Alignment"],
        groups: {
          "core-stack": {
            title: "Core Stack",
            summary: "The baseline path from raw text to a trainable decoder-only model.",
          },
          "scale-performance": {
            title: "Scale & Performance",
            summary: "The layers that make larger runs practical: kernels, distributed execution, and data preparation.",
          },
          "alignment-workflows": {
            title: "Alignment Workflows",
            summary: "Concrete SFT and RLFT implementations built around gsm8k and Qwen2.5-Math-1.5B.",
          },
        },
        labels: {
          docsHome: "Docs",
          sourceFiles: "Source Files",
          relatedChapters: "Related Chapters",
          previous: "Previous",
          next: "Next",
        },
      },
    },
  },
  zh: {
    navigation: {
      items: [
        { label: "架构", href: "/zh/architecture" },
        { label: "SFT 与 RLFT", href: "/zh/sft-rlft" },
        { label: "基准", href: "/zh/benchmarks" },
        { label: "文档", href: "/zh/docs" },
      ],
      githubLabel: "GitHub",
      docsLabel: "文档",
      localeSwitchLabel: "EN",
    },
    pages: {
      home: {
        title: "从零开始的 LLM",
        description: "一个为从零实现的 PyTorch decoder-only Transformer 打造的双语 GitHub Pages 项目站。",
        eyebrow: "Decoder-only Transformer • PyTorch • 从零实现",
        subtitle: "从分词器、训练、基准到微调，完整展示这个语言模型项目的核心实现。",
        actions: [
          { label: "查看 GitHub", href: "https://github.com/fangpin/llm-from-scratch", variant: "primary" },
          { label: "浏览文档", href: "/zh/docs", variant: "secondary" },
        ],
        proofMetrics: [
          { value: "1.56% -> 62.9%", label: "gsm8k 零样本准确率" },
          { value: "18.9% -> 100%", label: "输出格式遵循率" },
          { value: "Tokenizer / Training / Parallel / Kernel", label: "实现覆盖范围" },
        ],
        portals: [
          { title: "架构", href: "/zh/architecture", summary: "Tokenizer、Transformer blocks、训练循环、Kernel 与分布式训练。" },
          { title: "SFT 与 RLFT", href: "/zh/sft-rlft", summary: "Qwen2.5-Math-1.5B 在 gsm8k 上的微调与强化学习结果。" },
          { title: "基准", href: "/zh/benchmarks", summary: "Loss 曲线、学习率调度和性能说明。" },
          { title: "文档", href: "/zh/docs", summary: "按模块组织的 Markdown 章节，覆盖 tokenizer、训练、分布式、数据与对齐实现。" },
        ],
        capabilityTags: ["RoPE", "RMSNorm", "SwiGLU", "Flash Attention 2", "DDP", "SFT", "RLFT"],
      },
      architecture: {
        title: "架构",
        description: "从 tokenizer、transformer blocks 到训练、kernel 与分布式执行，整体说明这个项目是如何组成的。",
        sections: [
          { title: "Tokenizer", summary: "从零实现的 BPE tokenizer，可从原始文本中学习 merge 规则和特殊 token。" },
          { title: "Transformer Core", summary: "带有 RMSNorm、RoPE、SwiGLU、自定义 attention 与 loss 的 decoder-only blocks。" },
          { title: "Training Loop", summary: "包含 checkpoint、validation、generation、cosine scheduler 与优化器工具。" },
          { title: "Parallel + Kernel", summary: "自定义 DDP、sharded optimizer，以及 Triton Flash Attention 2 支持。" },
          { title: "Data Pipeline", summary: "HTML 抽取、过滤、去重、PII masking 和质量筛选工具。" },
        ],
      },
      sftRlft: {
        title: "SFT 与 RLFT",
        description: "仓库里最强的结果展示页：Qwen2.5-Math-1.5B 在 gsm8k 上的微调与强化学习结果。",
        metrics: [
          { value: "1.56%", label: "零样本基线" },
          { value: "62.9%", label: "SFT 后准确率" },
          { value: "100%", label: "格式遵循率" },
        ],
        steps: [
          "准备 gsm8k 训练集与测试集。",
          "评估 Qwen2.5-Math-1.5B 的零样本表现。",
          "运行监督微调以稳定答案格式。",
          "运行强化学习微调以提升奖励驱动的推理表现。",
        ],
      },
      benchmarks: {
        title: "基准",
        description: "用训练曲线和实现说明把这个项目落到可观测的性能与训练行为上。",
        cards: [
          { value: "Flash Attention 2", label: "包含 Triton kernel 路径" },
          { value: "Loss Curve", label: "TinyStories 训练收敛过程" },
          { value: "LR Schedule", label: "Cosine warmup 与 decay" },
        ],
        charts: [
          { src: "/images/loss.png", alt: "损失曲线" },
          { src: "/images/lr.png", alt: "学习率调度" },
        ],
      },
      docs: {
        title: "文档",
        description: "站内的模块化项目文档，直接由中英文 Markdown 章节驱动。",
        eyebrow: "仓库结构图",
        overviewLead: "文档不再按教程顺序组织，而是按项目模块拆开阅读：从 tokenizer、Transformer、训练循环，到分布式、数据处理和对齐流程。",
        systemMap: ["Tokenizer", "Transformer", "训练", "扩展", "对齐"],
        groups: {
          "core-stack": {
            title: "核心栈",
            summary: "从原始文本到 decoder-only 模型训练的基础主路径。",
          },
          "scale-performance": {
            title: "规模与性能",
            summary: "让更大训练可行的几层能力：kernel、分布式执行和数据准备。",
          },
          "alignment-workflows": {
            title: "对齐流程",
            summary: "围绕 gsm8k 和 Qwen2.5-Math-1.5B 的 SFT 与 RLFT 实现。",
          },
        },
        labels: {
          docsHome: "文档",
          sourceFiles: "关键源码",
          relatedChapters: "相关章节",
          previous: "上一章",
          next: "下一章",
        },
      },
    },
  },
} as const;
