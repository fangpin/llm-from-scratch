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
            current_len = input_ids.shape[1]
            start_idx = max(0, current_len - model.max_seq_len)
            input_ids_cond = input_ids[:, start_idx:]

            # The positions should be absolute, tracking the full sequence length
            token_positions = torch.arange(start_idx, current_len, device=args.device).unsqueeze(0)

            # Get the logits from the model
            logits = model(input_ids_cond, token_positions)
            # Take the logits for the last token
            logits = logits[:, -1, :]

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
    prompt_len = len(tokenizer.encode(prompt))
    generated_ids = input_ids[0, prompt_len:].tolist()
    return tokenizer.decode(generated_ids), generated_ids


if __name__ == "__main__":
    prompt = "tell you a story, "
    print(f"Prompt: {prompt}")
    output, output_token_ids = generate(prompt)
    print(f"Completion: {output}")
    print(output_token_ids)