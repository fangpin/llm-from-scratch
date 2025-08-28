import torch
import numpy as np
import argparse
from llm.args import get_parser
from llm.checkpoint import save_checkpoint
from llm.transformer import CrossEntropyLoss, Transformer, AdamW, gradient_clip, cos_lr_scheduler
import os
from torch.utils.tensorboard.writer import SummaryWriter


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of input and target sequences from the tokenized data.

    Args:
        x: A numpy array of token IDs.
        batch_size: The number of sequences in a batch.
        context_length: The length of each sequence.
        device: The PyTorch device to place the tensors on (e.g., 'cpu', 'cuda:0').

    Returns:
        A tuple containing the input and target sequences as PyTorch tensors.
    """
    # Generate random starting indices for the batches
    ix = torch.randint(0, len(x) - context_length, (batch_size,))

    # Create the input and target sequences
    input_seqs = torch.stack([torch.from_numpy(x[i : i + context_length].astype(np.int64)) for i in ix])
    target_seqs = torch.stack([torch.from_numpy(x[i + 1 : i + 1 + context_length].astype(np.int64)) for i in ix])

    # Move the tensors to the specified device
    return input_seqs.to(device), target_seqs.to(device)


def train():
    parser = argparse.ArgumentParser(description="Train a Transformer model.")

    parser = get_parser()
    args = parser.parse_args()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=args.log_dir)

    # Data Loading
    print("Loading data...")
    # train_data = np.load(args.train_data)
    # val_data = np.load(args.val_data)
    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")

    # Model Initialization
    print("Initializing model...")
    model = Transformer(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        device=args.device,
    ).to(args.device)

    # Optimizer Initialization
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    criterion = CrossEntropyLoss()
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    # Training Loop
    print("Starting training...")
    for i in range(args.iterations + 1):
        # Validation
        if i % args.val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for _ in range(100):  # 100 batches for validation
                    val_inputs, val_targets = get_batch(val_data, args.batch_size, args.max_seq_len, args.device)
                    val_logits = model(val_inputs)
                    val_loss += criterion(val_logits, val_targets).item()
            val_loss /= 100
            print(f"Iteration {i}, Validation Loss: {val_loss:.4f}")
            model.train()
            writer.add_scalar("val_loss", val_loss, i)

        # Get a batch of training data
        inputs, targets = get_batch(train_data, args.batch_size, args.max_seq_len, args.device)

        # Forward pass
        logits = model(inputs)
        loss = criterion(logits, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        gradient_clip(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Learning rate decay
        lr = cos_lr_scheduler(
            it=i,
            warmup_iters=args.warmup_iters,
            cos_cycle_iters=args.cos_cycle_iters,
            lr_min=args.lr_min,
            lr_max=args.lr_max,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        writer.add_scalar("loss_train", loss.item(), i)
        writer.add_scalar("lr", lr, i)
        # Logging
        if i % args.log_interval == 0:
            print(f"Iteration {i}, Training Loss: {loss.item():.4f}, LR: {lr:.6f}")

        # Checkpointing
        if i % args.checkpoint_interval == 0 and i > 0:
            save_checkpoint(model, optimizer, i, os.path.join(args.checkpoint_path, f"chpt_{i}.pt"))
            print(f"Saved checkpoint at iteration {i}")


if __name__ == "__main__":
    train()
