import argparse
from llm.transformer import Transformer, CrossEntropyLoss
import torch
import timeit


def benchmark(
    d_model: int,
    num_heads: int,
    d_ff: int,
    vocab_size: int,
    num_layers: int,
    batch_size: int,
    seq_len: int,
    warmup_steps: int,
    forward_only: bool,
    device: str,
    dtype: torch.dtype,
    steps=100,
):
    model = Transformer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        num_layers=num_layers,
        max_seq_len=seq_len,
        rope_theta=10000,
        device=device,
        dtype=dtype,
    )
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    criterion = CrossEntropyLoss()

    for _ in range(warmup_steps):
        if forward_only:
            with torch.no_grad():
                model(input_ids)
        else:
            logits = model(input_ids)
            loss = criterion(logits, targets)
            loss.backward()

    torch.cuda.synchronize()

    start = timeit.default_timer()
    for _ in range(steps):
        if forward_only:
            with torch.no_grad():
                model(input_ids)
        else:
            logits = model(input_ids)
            loss = criterion(logits, targets)
            loss.backward()

    torch.cuda.synchronize()
    end = timeit.default_timer()

    print(f"Time per iteration: {(end - start) * 1000 / steps:.3f} ms, total time: {(end - start) * 1000:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a Transformer model.")

    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1024, help="Feed-forward dimension")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default=32")
    parser.add_argument("--seq_len", type=int, default=128, help="Context length, default=128")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps, default=1000")
    parser.add_argument("--forward_only", type=bool, default=False, help="only benchmark forward pass, default=false")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")

