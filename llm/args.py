import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Train a Transformer model.")

    # Model Hyperparameters
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1024, help="Feed-forward dimension")
    parser.add_argument("--vocab_size", type=int, default=5000, help="Vocabulary size")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length")

    # Optimizer Hyperparameters
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lr_min", type=float, default=5e-5, help="Min learning rate in cos lr scheduler")
    parser.add_argument("--lr_max", type=float, default=5e-4, help="Max learning rate in cos lr scheduler")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--context_length", type=int, default=256, help="Context length")
    parser.add_argument("--iterations", type=int, default=30000, help="Number of training iterations")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="Number of warmup iterations")
    parser.add_argument("--cos_cycle_iters", type=int, default=25000, help="Number of cos cycle iterations")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to train on (e.g., 'cpu', 'cuda:0', 'mps')",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/training_data.npy",
        # required=True,
        help="Path to the training data file (numpy array)",
    )
    parser.add_argument(
        "--tokenizer_checkpoint",
        type=str,
        default="data/tokenizer",
        help="Path to the tokenizer checpoint path",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="data/validation_data.npy",
        # required=True,
        help="Path to the validation data file (numpy array)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints",
        help="Path to save checkpoints",
    )
    parser.add_argument(
        "--train_source_file",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="path to the train source file, used to generated token ids",
    )
    parser.add_argument(
        "--valid_source_file",
        type=str,
        default="data/TinyStoriesV2-GPT4-valid.txt",
        help="path to the valid source file, used to generated token ids",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="log",
        help="Path to store log",
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

    # Inferencing Hyperparameters
    parser.add_argument("--temperature", type=float, default=0.8, help="temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Inferencing top_p kernel search")
    return parser