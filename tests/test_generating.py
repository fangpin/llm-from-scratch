import pytest
from unittest.mock import patch, MagicMock
import torch

from llm.generating import generate

@patch('llm.generating.get_parser')
@patch('llm.generating.Transformer')
@patch('llm.generating.load_checkpoint')
@patch('llm.generating.BpeTokenizer')
def test_generate_produces_output(mock_bpe_tokenizer, mock_load_checkpoint, mock_transformer, mock_get_parser):
    # Arrange
    # Mock arguments
    mock_args = MagicMock()
    mock_args.max_seq_len = 10
    mock_args.temperature = 0.1 # Low temperature to make it deterministic
    mock_args.top_p = 1.0 # No top-p filtering
    mock_args.checkpoint_path = 'dummy'
    mock_args.iterations = 1
    mock_args.tokenizer_checkpoint = 'dummy'
    mock_args.device = 'cpu'
    mock_args.vocab_size = 50
    mock_args.d_model = 128
    mock_args.num_heads = 4
    mock_args.d_ff = 256
    mock_args.num_layers = 2
    mock_get_parser.return_value.parse_args.return_value = mock_args

    # Mock tokenizer
    mock_tokenizer_instance = MagicMock()
    prompt_tokens = [10, 20, 30]
    mock_tokenizer_instance.encode.return_value = prompt_tokens
    # The decode function will be called with the generated IDs
    mock_tokenizer_instance.decode.return_value = "4041"
    mock_tokenizer_instance.vcab2id = {'<|endoftext|>': 0}
    mock_tokenizer_instance.special_tokens = ['<|endoftext|>']
    mock_bpe_tokenizer.return_value = mock_tokenizer_instance

    # Mock model
    # The model is a callable object. Let's mock the whole instance.
    mock_model_instance = MagicMock()
    
    # vocab size is 50
    # let's make it output token 40, then 41, then 0 (endoftext)
    logits1 = torch.full((1, 50), -float('inf'))
    logits1[0, 40] = 1.0
    logits2 = torch.full((1, 50), -float('inf'))
    logits2[0, 41] = 1.0
    logits3 = torch.full((1, 50), -float('inf'))
    logits3[0, 0] = 1.0
    
    # The model is called inside a loop. We need to mock its return value for each call.
    mock_model_instance.side_effect = [logits1, logits2, logits3]
    mock_transformer.return_value.to.return_value = mock_model_instance

    # Act
    output, output_token_ids = generate("a prompt")

    # Assert
    assert output == "4041"
    assert output_token_ids == [40, 41]
    
    # Check that the tokenizer was called correctly
    mock_tokenizer_instance.encode.assert_called_once_with("a prompt")
    mock_tokenizer_instance.decode.assert_called_once_with([40, 41])
    
    # Check that the model was called
    assert mock_model_instance.call_count == 3
