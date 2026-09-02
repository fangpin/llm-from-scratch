# 模型输出采样

```{contents} 本页目录
---
depth: 2
local: true
---
```

大语言模型（LLM）生成文本时，需从每一步的 logits 概率分布中选择下一个 token。**采样策略直接决定输出的多样性、连贯性与可控性**。



训练完成后，生成文本的基本循环很短：把 prompt 编码成 token ID，取模型最后一个位置的 logits，变成概率分布，采样下一个 token，拼回上下文。这个过程重复到达到最大 token 数，或采到 `<|endoftext|>`。

temperature 会把 logits 除以 `t` 再做 softmax。`t` 越小，最大 logit 越占优势，输出更确定；`t` 越大，分布更平，输出更随机。top-p 进一步把低概率尾部截掉：先按概率排序，保留累计概率达到 `p` 的最小 token 集合，再在这个集合内重新归一化采样。它的直觉是保留当前模型认为“合情合理”的候选，而不是让长尾噪声频繁进入输出。

## 核心采样方法（按使用频率排序）

### **Temperature 调节（基础预处理，必用）**

- **原理**：`scaled_logits = logits / temperature` → softmax  

  - `T < 1.0`：分布更尖锐（高概率词更突出，输出更确定）  
  - `T > 1.0`：分布更平滑（低概率词机会增加，输出更多样）
- **典型值**：  

  - `T=0.7~1.0`：创意写作、对话（平衡多样性与质量）  
  - `T=0.1~0.3`：代码生成、事实问答（追求确定性）  
  - `T=0`：等价于贪婪搜索（非随机）
- ✅ **几乎总与其他采样方法组合使用**

---

### **Top-p 采样（Nucleus Sampling）— 当前工业界首选**

- **原理**：  

  1. 将 token 按概率降序排列  
  2. 累加概率直至 ≥ `p`（如 `p=0.9`）  
  3. 仅从该动态集合中按概率采样
- **优势**：  

  - 动态调整候选集大小（高频词多时集合小，低频词多时集合大）  
  - 避免 Top-k 在分布平坦时包含无关词，或在尖锐时遗漏合理词
- **典型配置**：`top_p=0.9, temperature=0.8`（GPT 系列、Claude 默认）  

---

### **Top-k 采样**

- **原理**：仅保留概率最高的 `k` 个 token（如 `k=50`），重新归一化后采样  
- **适用场景**：  

  - 语言分布较均匀的任务（如诗歌生成）  
  - 与 Top-p 组合使用（`top_k=50, top_p=0.9`）
- **缺点**：  

  - `k` 固定 → 分布尖锐时包含低质词，分布平坦时遗漏合理词  
  - 需人工调参（不同任务最优 `k` 差异大）

---

### **贪婪搜索（Greedy Search）**

- **原理**：每步选概率最高的 token  
- **优点**：计算快、结果确定  
- **致命缺陷**：极易陷入重复循环（如 "I love you I love you..."）  
- **仅适用**：`temperature=0` 的确定性任务（如数学推理、结构化代码生成）

---

### **Beam Search（搜索算法，非采样）**

- **原理**：维护 `beam_width` 条候选路径，每步扩展并保留最优  
- **适用**：  

  - 机器翻译、摘要（需全局最优）  
  - 配合 `length_penalty` 避免过短
- **不适用**：开放式生成（对话、故事）→ 输出呆板、缺乏创意  
- ⚠️ 注意：**LLM 通常不直接使用 Beam Search，有时会使用 Beam Search + top-k/top-p的方案**

---

## 实现

Temperature 调节 + Top-p 采样实现

```python
from torch.optim import AdamW
from llm.args import get_parser
from llm.checkpoint import load_checkpoint
from llm.transformer import Transformer, Softmax
from llm.bpe_tokenizer import BpeTokenizer
import torch
import os


def generate(prompt: str) -> tuple[str, list[int]]:
    parser = get_parser()
    args = parser.parse_args()

    model = Transformer(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        device=args.device,
    ).to(args.device)

    load_checkpoint(os.path.join(args.checkpoint_path, f"chpt_{str(args.iterations)}.pt"), model)

    tokenizer = BpeTokenizer()
    tokenizer.load(args.tokenizer_checkpoint)

    # Encode the prompt
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=args.device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for _ in range(args.max_seq_len):
            # Get the last context_length tokens
            input_ids_cond = input_ids[:, -model.max_seq_len :]

            # The positions should be relative to the current context window
            token_positions = torch.arange(input_ids_cond.shape[1], device=args.device).unsqueeze(0)

            # Get the logits from the model
            logits = model(input_ids_cond, token_positions)
            # Take the logits for the last token
            logits = logits[:, -1, :]
            # print(logits)

            # Apply temperature scaling
            logits = logits / args.temperature

            # Apply top-p sampling
            probs = Softmax()(logits)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > args.top_p
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[:, indices_to_remove] = 0

            # Re-normalize the probabilities
            probs = probs / torch.sum(probs, dim=-1, keepdim=True)

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the new token to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for end-of-text token
            if next_token.item() == tokenizer.vcab2id[tokenizer.special_tokens[0]]:
                break

    # Decode the generated tokens
    prompt_len = len(token_ids)
    generated_ids = input_ids[0, prompt_len:].tolist()
    if generated_ids and generated_ids[-1] == tokenizer.vcab2id[tokenizer.special_tokens[0]]:
        generated_ids.pop()
    return tokenizer.decode(generated_ids), generated_ids


if __name__ == "__main__":
    prompt = "tell you a story"
    print(f"Prompt: {prompt}")
    output, output_token_ids = generate(prompt)
    print(f"Completion: {output}")
    print(output_token_ids)


```
