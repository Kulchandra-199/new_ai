
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

# Resources 

https://www.youtube.com/watch?v=dA-NhCtrrVE&ab_channel=ChrisAlexiuk



flowchart TD
    A[Start: load_model_and_tokenizer] --> B[Initialize Tokenizer]
    B --> C{Pad Token Exists?}
    
    C -->|No| D[Set pad_token = eos_token]
    C -->|Yes| E[Keep existing pad_token]
    
    D --> F[Initialize Model]
    E --> F
    
    F --> G[Configure Model Parameters]
    
    subgraph Model Configuration
        G --> H[Set device_map: CPU]
        H --> I[Set trust_remote_code: True]
        I --> J[Set dtype: float32]
        J --> K[Set offload folder]
    end
    
    K --> L[Return model and tokenizer]


    ########## Main 

    flowchart TD
    A[Start: main] --> B[Configure Model Name]
    B --> C[Load Model & Tokenizer]
    
    subgraph Model Initialization
        C --> D[Load DeepSeek Model]
        D --> E[Load Tokenizer]
    end
    
    E --> F[Configure LoRA]
    F --> G[Print Trainable Parameters]
    
    subgraph Data Processing
        H[Load JSON Dataset] --> I[Tokenize Dataset]
        I --> J[Create Data Collator]
    end
    
    G --> H
    
    subgraph Training Setup
        J --> K[Setup Training Arguments]
        K --> L[Initialize Trainer]
    end
    
    subgraph Training Process
        L --> M[Train Model]
        M --> N[Save LoRA Adapters]
        N --> O[Save Tokenizer]
    end
    
    style Model Initialization fill:#e1f5fe
    style Data Processing fill:#f3e5f5
    style Training Setup fill:#e8f5e9
    style Training Process fill:#fff3e0