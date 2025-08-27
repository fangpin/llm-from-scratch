import sys
import os
import numpy as np

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from llm.bpe_tokenizer import BpeTokenizer

def inspect_data():
    """
    Loads the tokenizer and a sample of the training data,
    then prints a detailed analysis of how the data is structured and tokenized.
    """
    tokenizer_path = "data/tokenizer"
    data_path = "data/training_data.npy"

    # Load the tokenizer
    tokenizer = BpeTokenizer()
    try:
        tokenizer.load(tokenizer_path)
    except FileNotFoundError:
        print(f"Error: Tokenizer file not found at '{tokenizer_path}'")
        return
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Load a sample of the training data
    try:
        token_ids = np.load(data_path)
        sample_token_ids = token_ids[:200]  # Take the first 200 tokens
    except FileNotFoundError:
        print(f"Error: Training data file not found at '{data_path}'")
        return
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # Decode the sample
    sample_text = tokenizer.decode(sample_token_ids.tolist())
    decoded_tokens = [tokenizer.decode([token_id]) for token_id in sample_token_ids]


    # Print the analysis
    print("--- Data Inspection ---")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Data sample: {data_path}")
    print("-" * 20)
    print(f"Vocabulary size: {len(tokenizer.vcab2id)}")
    special_tokens_decoded = [st.decode('utf-8', errors='ignore') for st in tokenizer.special_tokens]
    print(f"Special tokens: {special_tokens_decoded}")
    if special_tokens_decoded:
        endoftext_token = special_tokens_decoded[0]
        endoftext_id = tokenizer.vcab2id[tokenizer.special_tokens[0]]
        print(f"The ID for '{endoftext_token}' is: {endoftext_id}")

    print("-" * 20)
    print("Sample Token IDs:")
    print(sample_token_ids)
    print("-" * 20)
    print("Decoded Text:")
    print(sample_text)
    print("-" * 20)
    print("Decoded Tokens (one by one):")
    print(decoded_tokens)
    print("--- End of Inspection ---")

if __name__ == "__main__":
    inspect_data()