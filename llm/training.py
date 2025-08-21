import torch
import numpy as np
import argparse
from llm.checkpoint import save_checkpoint
from .transformer import *


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of input and target sequences from the tokenized data.

    Args:
        x: A numpy array of token IDs.
        batch_size: The number of sequences in a batch.
        context_length: The length of each sequence.
        device: The PyTorch device to place the tensors on (e.g., 'cpu', 'cuda:0').

    Returns:
        A tuple containing the input sequences and target sequences as PyTorch tensors.
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

    # Model Hyperparameters
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")

    # Optimizer Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="AdamW weight decay")

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--context_length", type=int, default=256, help="Context length")
    parser.add_argument("--iterations", type=int, default=10000, help="Number of training iterations")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to train on (e.g., 'cpu', 'cuda:0', 'mps')",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to the training data file (numpy array)",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to the validation data file (numpy array)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints",
        help="Path to save checkpoints",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Interval for logging training loss",
    )
    parser.add_argument("--val_interval", type=int, default=500, help="Interval for running validation")
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
        help="Interval for saving checkpoints",
    )

    args = parser.parse_args()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Data Loading
    print("Loading data...")
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode="r")

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
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
    )

    # Loss Function
    criterion = CrossEntropyLoss()

    # Training Loop
    print("Starting training...")
    for i in range(args.iterations):
        # Get a batch of training data
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, args.device)

        # Forward pass
        logits = model(inputs, train=True)
        loss = criterion(logits, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        gradient_clip(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Learning rate decay
        lr = cos_lr_scheduler(1000, args.iterations, i, args.lr, 1e-5)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Logging
        if i % args.log_interval == 0:
            print(f"Iteration {i}, Training Loss: {loss.item():.4f}, LR: {lr:.6f}")

        # Validation
        if i % args.val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for _ in range(100):  # 100 batches for validation
                    val_inputs, val_targets = get_batch(val_data, args.batch_size, args.context_length, args.device)
                    val_logits = model(val_inputs, train=False)
                    val_loss += criterion(val_logits, val_targets).item()
            val_loss /= 100
            print(f"Iteration {i}, Validation Loss: {val_loss:.4f}")
            model.train()

        # Checkpointing
        if i % args.checkpoint_interval == 0 and i > 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": i,
            }
            save_checkpoint(model, optimizer, i, os.path.join(args.checkpoint_path, f"chpt_{i}.pt"))
            print(f"Saved checkpoint at iteration {i}")


if __name__ == "__main__":
    # Create a dummy data file
    dummy_data = np.arange(1000, dtype=np.int64)
    dummy_data_path = "dummy_data.bin"
    dummy_data.tofile(dummy_data_path)

    # Load the data using memory mapping
    # The dtype must match the data type used to save the file
    memmapped_data = np.memmap(dummy_data_path, dtype=np.int64, mode="r")

    # Define parameters
    batch_size = 4
    context_length = 10
    device = "cpu"

    # Get a batch from the memory-mapped data
    inputs, targets = get_batch(memmapped_data, batch_size, context_length, device)

    print("Successfully sampled a batch from the memory-mapped file.")
    print("Input shape:", inputs.shape)
    print("Target shape:", targets.shape)

    # Clean up the dummy file
    import os

    os.remove(dummy_data_path)

    train()
