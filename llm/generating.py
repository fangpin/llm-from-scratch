from transformer import Transformer, Softmax
from bpe_tokenizer import BpeTokenizer
import torch


def generate(
    model: Transformer,
    tokenizer: BpeTokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> tuple[str, list[int]]:
    """
    Generates a text completion from a prompt using a trained transformer model.

    Args:
        model: The trained Transformer model.
        tokenizer: The BPE tokenizer.
        prompt: The input prompt string.
        max_tokens: The maximum number of tokens to generate.
        temperature: The temperature for softmax scaling.
        top_p: The threshold for top-p (nucleus) sampling.
        device: The PyTorch device to run the generation on.

    Returns:
        The generated text completion.
    """
    # Encode the prompt
    token_ids = tokenizer.encode(prompt.encode("utf-8"))
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get the last context_length tokens
            input_ids_cond = input_ids[:, -model.max_seq_len :]

            # Get the logits from the model
            logits = model(input_ids_cond, train=False)
            # Take the logits for the last token
            logits = logits[:, -1, :]

            # Apply temperature scaling
            logits = logits / temperature

            # Apply top-p sampling
            probs = Softmax()(logits)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
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
            if next_token.item() == tokenizer.special_tokens[0]:  # Assuming the first special token is <|endoftext|>
                break

    # Decode the generated tokens
    generated_ids = input_ids[0].tolist()
    return tokenizer.decode(generated_ids), generated_ids
