
# Simple Language Model

This repository contains a simple implementation of a language model using PyTorch. The model is built using transformer blocks, which include multi-head attention and feed-forward networks.

## Files

- `main.py`: Contains the main implementation of the language model, including the `MultiHeadAttention`, `FeedForward`, and `TransformerBlock` classes.
- `single.py`: Contains an alternative implementation of the language model with additional features such as rotary positional embeddings and nucleus sampling for text generation.
- `ui.py`: Implements a simple custom Python notebook UI using Tkinter for running and displaying code.

## Usage

### Running the Language Model

To run the language model, you can use the `main.py` file. Here is an example of how to initialize and use the model:

```python
import torch
from main import SimpleLanguageModel, create_padding_mask

# Model initialization
vocab_size = 10000
model = SimpleLanguageModel(vocab_size)

# Sample batch generation
batch_size = 32
seq_length = 50
sample_input = torch.randint(0, vocab_size, (batch_size, seq_length))
mask = create_padding_mask(sample_input)

# Forward pass demonstration
output = model(sample_input, mask)
print(f"Output shape: {output.shape}")