import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

# Multi-Head Attention Module
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import string

class SimpleTokenizer:
    """Basic character-level tokenizer for demo purposes"""
    def __init__(self):
        self.chars = string.printable
        self.vocab_size = len(self.chars) + 2  # +2 for padding and unknown
        self.char2idx = {c: i+2 for i, c in enumerate(self.chars)}
        self.char2idx['<pad>'] = 0
        self.char2idx['<unk>'] = 1
        self.idx2char = {v: k for k, v in self.char2idx.items()}

    def encode(self, text, max_len=50):
        encoded = [self.char2idx.get(c, 1) for c in text[:max_len]]
        return encoded + [0] * (max_len - len(encoded))
    
    def decode(self, tokens):
        return ''.join([self.idx2char.get(t, '�') for t in tokens if t > 1])

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_probs = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, V)
        return output, attention_probs

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        output, attention_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output), attention_probs

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_output, attention_probs = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x, attention_probs

class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1000, d_model))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        seq_length = x.size(1)
        x = self.embedding(x)
        x = x + self.pos_encoding[:seq_length, :]
        x = self.dropout(x)
        all_attention_probs = []
        for transformer in self.transformer_blocks:
            x, attention_probs = transformer(x, mask)
            all_attention_probs.append(attention_probs.detach())
        output = self.fc(x)
        return output, all_attention_probs

def create_padding_mask(seq):
    return (seq != 0).unsqueeze(1).unsqueeze(2)

def chat(model, tokenizer, max_length=50):
    print("Start chatting with the AI (type 'quit' to exit)")
    while True:
        input_text = input("You: ")
        if input_text.lower() == 'quit':
            break
        
        # Encode input
        input_ids = tokenizer.encode(input_text, max_length)
        input_tensor = torch.tensor([input_ids])
        
        # Generate response
        with torch.no_grad():
            logits, _ = model(input_tensor)
            # Simple random sampling from last output
            probs = F.softmax(logits[0, -1], dim=-1)
            response_id = torch.multinomial(probs, num_samples=1).item()
            response = tokenizer.decode([response_id])
            
        print(f"AI: {response}\n")

if __name__ == "__main__":
    # Initialize components
    tokenizer = SimpleTokenizer()
    model = SimpleLanguageModel(tokenizer.vocab_size)
    
    # Start chat session
    chat(model, tokenizer)
# print(f"Output shape: {output.shape}")

# # Visualization Examples
# # 1. Plot attention weights for the first layer and first head
# plot_attention_weights(all_attention_probs[0][0, 0].cpu().numpy(), list(range(seq_length)))

# # 2. Plot token embeddings for the first 100 tokens
# plot_token_embeddings(model.embedding, list(range(100)))

# # 3. Plot attention heads for the first layer
# plot_attention_heads(all_attention_probs[0][0].cpu().numpy(), num_heads=8)

# # 3D Token Embeddings Visualization
# plot_3d_token_embeddings(model.embedding, list(range(100)))

# # 3D Attention Weights Visualization
# plot_3d_attention_heatmap(all_attention_probs[0][0], num_heads=8)




# # Visualization Functions
# def plot_attention_weights(attention_probs, input_tokens):
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(attention_probs, xticklabels=input_tokens, yticklabels=input_tokens, cmap="viridis")
#     plt.xlabel("Key Tokens")
#     plt.ylabel("Query Tokens")
#     plt.title("Attention Weights")
#     plt.show()

# def plot_loss_curves(train_losses, val_losses):
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label="Training Loss")
#     plt.plot(val_losses, label="Validation Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.title("Training and Validation Loss Curves")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def plot_gradient_flow(named_parameters):
#     ave_grads = []
#     layers = []
#     for n, p in named_parameters:
#         if p.grad is not None:
#             layers.append(n)
#             ave_grads.append(p.grad.abs().mean().item())
#     plt.figure(figsize=(10, 6))
#     plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.7, lw=1, color="b")
#     plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
#     plt.xticks(np.arange(len(ave_grads)), layers, rotation="vertical")
#     plt.xlabel("Layers")
#     plt.ylabel("Average Gradient")
#     plt.title("Gradient Flow")
#     plt.grid(True)
#     plt.show()

# def plot_token_embeddings(embedding_layer, token_ids):
#     embeddings = embedding_layer(torch.tensor(token_ids)).detach().numpy()
#     tsne = TSNE(n_components=2, random_state=42)
#     embeddings_2d = tsne.fit_transform(embeddings)
#     plt.figure(figsize=(10, 8))
#     plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
#     for i, token_id in enumerate(token_ids):
#         plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], str(token_id), fontsize=9)
#     plt.xlabel("t-SNE Component 1")
#     plt.ylabel("t-SNE Component 2")
#     plt.title("Token Embeddings Visualization")
#     plt.grid(True)
#     plt.show()

# def plot_attention_heads(attention_probs, num_heads):
#     fig, axes = plt.subplots(1, num_heads, figsize=(20, 4))
#     for i in range(num_heads):
#         sns.heatmap(attention_probs[i], ax=axes[i], cmap="viridis")
#         axes[i].set_title(f"Head {i+1}")
#     plt.tight_layout()
#     plt.show()

# def plot_3d_token_embeddings(embedding_layer, token_ids):
#     """
#     Create a 3D visualization of token embeddings using t-SNE
    
#     Args:
#         embedding_layer (nn.Embedding): Embedding layer of the model
#         token_ids (list): List of token IDs to visualize
#     """
#     # Get embeddings
#     embeddings = embedding_layer(torch.tensor(token_ids)).detach().numpy()
    
#     # Use t-SNE to reduce to 3D
#     tsne = TSNE(n_components=3, random_state=42)
#     embeddings_3d = tsne.fit_transform(embeddings)
    
#     # Create 3D plot
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Scatter plot with color gradient
#     scatter = ax.scatter(embeddings_3d[:, 0], 
#                          embeddings_3d[:, 1], 
#                          embeddings_3d[:, 2], 
#                          c=np.arange(len(token_ids)), 
#                          cmap='viridis', 
#                          alpha=0.7)
    
#     # Annotate points with token IDs
#     for i, token_id in enumerate(token_ids):
#         ax.text(embeddings_3d[i, 0], 
#                 embeddings_3d[i, 1], 
#                 embeddings_3d[i, 2], 
#                 str(token_id), 
#                 fontsize=8)
    
#     plt.colorbar(scatter, label='Token Index')
#     ax.set_xlabel('t-SNE Component 1')
#     ax.set_ylabel('t-SNE Component 2')
#     ax.set_zlabel('t-SNE Component 3')
#     ax.set_title('3D Token Embeddings Visualization')
#     plt.tight_layout()
#     plt.show()

# def plot_3d_attention_heatmap(attention_probs, num_heads):
#     """
#     Create a 3D heatmap visualization of attention weights
    
#     Args:
#         attention_probs (torch.Tensor): Attention probabilities
#         num_heads (int): Number of attention heads
#     """
#     fig = plt.figure(figsize=(20, 6))
    
#     for i in range(num_heads):
#         ax = fig.add_subplot(1, num_heads, i+1, projection='3d')
        
#         # Create meshgrid for 3D surface
#         x = y = np.arange(attention_probs.shape[1])
#         X, Y = np.meshgrid(x, y)
#         Z = attention_probs[i].cpu().numpy()
        
#         # Plot 3D surface
#         surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
#                                 linewidth=0, antialiased=False)
        
#         ax.set_title(f'Attention Head {i+1}')
#         ax.set_xlabel('Key Tokens')
#         ax.set_ylabel('Query Tokens')
#         ax.set_zlabel('Attention Weight')
        
#         # Add a color bar
#         fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
#     plt.tight_layout()
#     plt.show()
