import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Multi-Head Attention Module
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0  # Ensure d_model is divisible by num_heads
        
        self.d_model = d_model  # Dimensionality of the model
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d_model // num_heads  # Dimensionality of each head
        
        # Single matrix for more efficient parallel computation
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)  # Project input into Q, K, V
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # Output projection
        
        # Dropout layers for regularization
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Optional key-value cache for inference (e.g., autoregressive generation)
        self.kv_cache = {}
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Compute attention scores (Q * K^T / sqrt(d_k))
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask to prevent attending to future tokens (for causal attention)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        # Compute attention probabilities using softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)  # Apply dropout
        return torch.matmul(attention_probs, V)  # Weighted sum of values
        
    def forward(self, x, mask=None, use_cache=False, layer_id=None):
        batch_size = x.size(0)
        
        # Efficient parallel projection of Q, K, V
        qkv = self.qkv_proj(x).chunk(3, dim=-1)  # Split into Q, K, V
        Q, K, V = [x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) 
                   for x in qkv]  # Reshape for multi-head attention
        
        # Use KV cache during inference if enabled
        if use_cache and layer_id is not None:
            cache_key = f"layer_{layer_id}"
            if cache_key in self.kv_cache:
                K_cache, V_cache = self.kv_cache[cache_key]
                K = torch.cat([K_cache, K], dim=2)  # Concatenate cached keys with new keys
                V = torch.cat([V_cache, V], dim=2)  # Concatenate cached values with new values
            self.kv_cache[cache_key] = (K, V)  # Update cache
        
        # Apply scaled dot-product attention
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # Reshape back
        output = self.W_o(output)  # Project back to original dimensionality
        return self.output_dropout(output)  # Apply dropout


# Feed-Forward Network Module
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # First linear transformation
        self.linear2 = nn.Linear(d_ff, d_model)  # Second linear transformation
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization
        self.activation = nn.GELU()  # GELU activation (smoother alternative to ReLU)
        
        # Layer scale parameters for better optimization
        self.layer_scale = nn.Parameter(torch.ones(1, 1, d_model) * 0.1)
        
    def forward(self, x):
        # Apply feed-forward network: Linear -> GELU -> Linear
        return self.layer_scale * self.linear2(self.dropout(self.activation(self.linear1(x))))


# Transformer Block Module
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # Pre-norm architecture
        self.norm1 = nn.LayerNorm(d_model)  # Normalization before attention
        self.norm2 = nn.LayerNorm(d_model)  # Normalization before feed-forward
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)  # Multi-head attention
        self.feed_forward = FeedForward(d_model, d_ff, dropout)  # Feed-forward network
        
        # Parallel attention and feed-forward branches
        self.parallel_mode = True
        
    def forward(self, x, mask=None, use_cache=False, layer_id=None):
        # Pre-norm architecture with parallel processing
        if self.parallel_mode:
            attn_norm = self.norm1(x)  # Normalize input for attention
            ff_norm = self.norm2(x)  # Normalize input for feed-forward
            
            attn_output = self.attention(attn_norm, mask, use_cache, layer_id)  # Apply attention
            ff_output = self.feed_forward(ff_norm)  # Apply feed-forward
            
            return x + attn_output + ff_output  # Residual connections
        else:
            # Sequential processing fallback
            attn_output = x + self.attention(self.norm1(x), mask, use_cache, layer_id)
            return attn_output + self.feed_forward(self.norm2(attn_output))


# Simple Language Model
class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=6, d_ff=1024, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  # Token embeddings
        
        # Rotary positional embeddings (RoPE)
        self.max_seq_length = 2048
        self.rope_cache = None
        
        # Stack of Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)  # Final normalization layer
        self.fc = nn.Linear(d_model, vocab_size)  # Output projection to vocabulary
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization
        
    def _get_rope_cache(self, device):
        # Generate cache for Rotary Positional Embeddings (RoPE)
        if self.rope_cache is None:
            positions = torch.arange(self.max_seq_length, device=device)
            dim_pairs = torch.arange(0, self.d_model, 2, device=device)
            
            # Generate rotation matrices
            freqs = torch.exp(-math.log(10000.0) * dim_pairs / self.d_model)
            angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
            
            self.rope_cache = torch.stack([
                torch.cos(angles),
                torch.sin(angles)
            ], dim=-1)
        
        return self.rope_cache
    
    def _apply_rope(self, x, start_pos=0):
        # Apply Rotary Positional Embeddings (RoPE)
        seq_len = x.size(1)
        rope_cache = self._get_rope_cache(x.device)[start_pos:start_pos + seq_len]
        
        # Reshape input to apply rotary embeddings
        x_reshape = x.view(*x.shape[:-1], -1, 2)
        
        # Apply rotation
        x_rot = torch.stack([
            x_reshape[..., 0] * rope_cache[..., 0] - x_reshape[..., 1] * rope_cache[..., 1],
            x_reshape[..., 1] * rope_cache[..., 0] + x_reshape[..., 0] * rope_cache[..., 1]
        ], dim=-1)
        
        return x_rot.view(*x.shape)
    
    def forward(self, x, mask=None, use_cache=False, start_pos=0):
        # Forward pass through the model
        x = self.embedding(x)  # Token embeddings
        x = self._apply_rope(x, start_pos)  # Apply Rotary Positional Embeddings
        x = self.dropout(x)  # Apply dropout
        
        # Pass through each Transformer block
        for i, transformer in enumerate(self.transformer_blocks):
            x = transformer(x, mask, use_cache, i)
            
        x = self.norm(x)  # Final normalization
        return self.fc(x)  # Project to vocabulary size


# Utility function to create a causal mask
def create_causal_mask(seq_length, device):
    """Create causal mask for autoregressive generation"""
    mask = torch.triu(torch.ones((seq_length, seq_length), device=device), diagonal=1).bool()
    return ~mask.unsqueeze(0)


# Text generation function using nucleus sampling
def generate(model, input_ids, max_length=100, temperature=0.8, top_p=0.9):
    """Nucleus sampling with temperature for text generation"""
    model.eval()  # Set model to evaluation mode
    
    for _ in range(max_length):
        # Create causal mask
        seq_length = input_ids.size(1)
        mask = create_causal_mask(seq_length, input_ids.device)
        
        # Forward pass with caching
        with torch.no_grad():
            logits = model(input_ids, mask=mask, use_cache=True)
            next_token_logits = logits[:, -1, :] / temperature  # Apply temperature scaling
            
            # Apply nucleus sampling (top-p sampling)
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # Sample next token
        
        input_ids = torch.cat([input_ids, next_token], dim=1)  # Append generated token
    
    return input_ids