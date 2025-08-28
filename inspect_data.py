import numpy as np
from collections import Counter
from llm.bpe_tokenizer import BpeTokenizer


def inspect_token_distribution(data_path: str, tokenizer_path: str, sample_size: int = 10000):
    """
    检查数据集中token的分布，特别是特殊token的频率
    """
    # Load tokenizer
    tokenizer = BpeTokenizer()
    tokenizer.load(tokenizer_path)

    # Get endoftext token ID
    endoftext_id = None
    if tokenizer.special_tokens:
        endoftext_token = tokenizer.special_tokens[0]  # <|endoftext|>
        endoftext_id = tokenizer.vcab2id[endoftext_token]
        print(f"Endoftext token ID: {endoftext_id}")

    # Load data
    data = np.load(data_path, mmap_mode="r")

    # Sample data
    sample = data[:sample_size] if len(data) > sample_size else data

    # Count token frequencies
    token_counts = Counter(sample.tolist())

    # Get top 20 most common tokens
    print("\nTop 20 most frequent tokens:")
    for token_id, count in token_counts.most_common(20):
        token_text = tokenizer.decode([token_id])
        percentage = (count / len(sample)) * 100
        print(f"Token {token_id} ('{token_text}'): {count} times ({percentage:.2f}%)")

    # Check endoftext frequency
    if endoftext_id is not None:
        endoftext_count = token_counts.get(endoftext_id, 0)
        endoftext_percentage = (endoftext_count / len(sample)) * 100
        print(f"\n<|endoftext|> appears {endoftext_count} times ({endoftext_percentage:.2f}% of tokens)")

        # Check if endoftext appears too frequently
        if endoftext_percentage > 10:
            print("⚠️ WARNING: <|endoftext|> token appears very frequently in the dataset!")
            print("This might cause the model to overfit to this token.")

    # Check sequence patterns
    print("\nChecking for consecutive <|endoftext|> tokens...")
    consecutive_endoftext = 0
    for i in range(len(sample) - 1):
        if endoftext_id and sample[i] == endoftext_id and sample[i + 1] == endoftext_id:
            consecutive_endoftext += 1

    if consecutive_endoftext > 0:
        print(f"⚠️ Found {consecutive_endoftext} consecutive <|endoftext|> tokens!")

    return token_counts


def check_batch_diversity(data_path: str, batch_size: int = 64, context_length: int = 64, num_batches: int = 10):
    """
    检查批次中的token多样性
    """
    data = np.load(data_path, mmap_mode="r")

    print(f"\nChecking {num_batches} random batches...")
    for batch_idx in range(num_batches):
        # Random batch
        ix = np.random.randint(0, len(data) - context_length, batch_size)
        batch = np.array([data[i : i + context_length] for i in ix])

        # Check first token diversity
        first_tokens = batch[:, 0]
        unique_first_tokens = len(np.unique(first_tokens))

        print(f"Batch {batch_idx}: {unique_first_tokens}/{batch_size} unique first tokens")

        if unique_first_tokens < batch_size // 4:
            print(f"  ⚠️ Low diversity in first tokens!")


if __name__ == "__main__":
    # 检查训练数据
    print("=== Inspecting Training Data ===")
    inspect_token_distribution("data/training_data.npy", "data/tokenizer")
    check_batch_diversity("data/training_data.npy")

    print("\n=== Inspecting Validation Data ===")
    inspect_token_distribution("data/validation_data.npy", "data/tokenizer")
