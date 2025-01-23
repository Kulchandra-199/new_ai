import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import math

class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.process_conversations(conversations)
        
    def process_conversations(self, conversations):
        processed = []
        for conv in conversations:
            history = []
            for i in range(0, len(conv)-1, 2):
                input_text = " [SEP] ".join(history + [conv[i]])
                target_text = conv[i+1]
                processed.append((input_text, target_text))
                history.append(conv[i])
                history.append(conv[i+1])
        return processed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_text, target_text = self.data[idx]
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length, padding='max_length', truncation=True)
        target_ids = self.tokenizer.encode(target_text, max_length=self.max_length, padding='max_length', truncation=True)
        return torch.tensor(input_ids), torch.tensor(target_ids)

class DialogueTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(d_model, vocab_size)
def forward(self, src, tgt):
    # Ensure matching batch sizes
    src = self.pos_encoder(self.embedding(src))
    tgt = self.pos_encoder(self.embedding(tgt))
    
    # Compute attention mask
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
    
    # Pass through transformer
    output = self.transformer(src, tgt, tgt_mask=tgt_mask)
    return self.fc(output)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class ConversationTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
    def train(self, dataset, epochs=10, batch_size=8):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for src, tgt in loader:
                self.optimizer.zero_grad()
                
                # Slice off the last token from target for teacher forcing
                output = self.model(src, tgt[:, :-1])
                
                # Reshape for loss calculation
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)), 
                    tgt[:, 1:].reshape(-1)
                )
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

            
class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = {word: idx+4 for idx, word in enumerate(vocab)}
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3
        }
        self.vocab.update(self.special_tokens)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = self.special_tokens['[PAD]']
        
    def encode(self, text, max_length=128, padding=True, truncation=True):
        tokens = [self.vocab.get(word, self.special_tokens['[UNK]']) 
                for word in text.split()]
        
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
            
        if padding and len(tokens) < max_length:
            tokens += [self.pad_token_id] * (max_length - len(tokens))
            
        return tokens
    
    def decode(self, token_ids):
        return ' '.join([self.inverse_vocab.get(tid, '[UNK]') for tid in token_ids])

# Example conversation data
conversations = [
    [
        "Hello, how are you?",
        "I'm doing well, thank you!",
        "What's your name?",
        "I'm an AI assistant. How can I help you today?"
    ],
    [
        "Tell me a joke",
        "Why don't scientists trust atoms? Because they make up everything!",
        "That's funny!",
        "Glad you liked it! Want another one?"
    ]
]

# Initialize components
tokenizer = SimpleTokenizer(vocab=["Hello", "how", "are", "you", "I'm", "doing", "well", "thank", 
                                 "What's", "your", "name", "AI", "assistant", "help", "today",
                                 "Tell", "me", "a", "joke", "Why", "don't", "scientists", "trust",
                                 "atoms", "Because", "they", "make", "up", "everything", "That's",
                                 "funny", "Glad", "you", "liked", "it", "Want", "another", "one"])

dataset = ConversationDataset(conversations, tokenizer)
model = DialogueTransformer(vocab_size=len(tokenizer.vocab))
trainer = ConversationTrainer(model, tokenizer)

# Train the model
trainer.train(dataset, epochs=20, batch_size=2)

# Conversation interface
def chat(model, tokenizer, max_length=50):
    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        history.append(user_input)
        context = " [SEP] ".join(history[-3:])  # Keep last 3 turns
        input_ids = tokenizer.encode(context, max_length=tokenizer.max_length)
        
        # Generate response
        model.eval()
        with torch.no_grad():
            output = model(torch.tensor([input_ids]), torch.tensor([[tokenizer.special_tokens['[CLS]']]]))
            response_ids = output.argmax(-1).squeeze().tolist()
            response = tokenizer.decode(response_ids).split('[SEP]')[0].strip()
            
        print(f"AI: {response}")
        history.append(response)

# Start chatting
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
