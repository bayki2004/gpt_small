
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

# Updated configuration for a T5-like model.
@dataclass
class ModelConfig:
    vocab_size: int = 32128            # T5 uses a 32k vocab by default.
    n_encoder_layers: int = 6          # For demonstration, using 6 layers (can be increased)
    n_decoder_layers: int = 6
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True
    block_size: int = 1024             # Used for causal masking in the decoder.

# Modified Head that can work with or without causal masking.
class Head(nn.Module):
    def __init__(self, head_size, config, causal=False):
        super().__init__()
        self.causal = causal
        self.key = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.query = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value = nn.Linear(config.n_embd, head_size, bias=config.bias)
        if self.causal:
            # Only needed for causal self-attention.
            self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (q.size(-1) ** -0.5)
        if self.causal:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

# Modified MultiHeadAttention accepts a flag for causal masking.
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, config, causal=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, config, causal=causal) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_head * head_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# FeedForward remains largely the same.
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.ReLU(),
            nn.Linear(config.n_embd * 4, config.n_embd),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        return self.net(x)

# Encoder block (no causal mask in self-attention)
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        # No causal mask here.
        self.sa = MultiHeadAttention(head_size, config, causal=False)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Cross-attention layer used in the decoder.
class CrossAttention(nn.Module):
    def __init__(self, head_size, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.query = nn.Linear(config.n_embd, head_size, bias=config.bias)
        # Project back to the model dimensionality
        self.proj = nn.Linear(head_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, encoder_out):
        # x: decoder input, encoder_out: encoder hidden states
        B, T, _ = x.shape
        # encoder_out shape: (B, S, config.n_embd)
        k = self.key(encoder_out)
        v = self.value(encoder_out)
        q = self.query(x)
        attn_weights = q @ k.transpose(-2, -1) * (q.size(-1) ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ v
        out = self.proj(out)
        return out

# Decoder block with causal self-attention, cross-attention, and feed-forward.
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(head_size, config, causal=True)
        self.cross_attn = CrossAttention(head_size, config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
    
    def forward(self, x, encoder_out):
        x = x + self.sa(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), encoder_out)
        x = x + self.ffwd(self.ln3(x))
        return x

# T5-like model with a shared embedding and separate encoder and decoder.
class T5LikeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # T5 typically ties the encoder and decoder embeddings.
        self.shared_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.encoder_blocks = nn.ModuleList([Block(config) for _ in range(config.n_encoder_layers)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_decoder_layers)])
        self.decoder_ln = nn.LayerNorm(config.n_embd)
        # Output projection tied to embeddings (optional, but common in T5)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Optionally tie the weights:
        self.lm_head.weight = self.shared_embedding.weight

    def encode(self, encoder_input_ids):
        # T5 does not use explicit positional embeddings.
        x = self.shared_embedding(encoder_input_ids)
        for block in self.encoder_blocks:
            x = block(x)
        return x

    def decode(self, encoder_hidden_states, decoder_input_ids):
        x = self.shared_embedding(decoder_input_ids)
        for block in self.decoder_blocks:
            x = block(x, encoder_hidden_states)
        x = self.decoder_ln(x)
        logits = self.lm_head(x)
        return logits

    def forward(self, encoder_input_ids, decoder_input_ids, decoder_targets=None):
        encoder_hidden_states = self.encode(encoder_input_ids)
        logits = self.decode(encoder_hidden_states, decoder_input_ids)
        if decoder_targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            decoder_targets_flat = decoder_targets.view(B * T)
            loss = F.cross_entropy(logits_flat, decoder_targets_flat)
        return logits, loss

    def generate(self, encoder_input_ids, decoder_start_token, max_new_tokens):
        encoder_hidden_states = self.encode(encoder_input_ids)
        decoder_input_ids = decoder_start_token.unsqueeze(0)  # shape: (1, 1)
        for _ in range(max_new_tokens):
            logits = self.decode(encoder_hidden_states, decoder_input_ids)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
        return decoder_input_ids


import torch
import torch.optim as optim

# Create the configuration and model.
config = ModelConfig()
model = T5LikeModel(config)
model.train()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 3
batch_size = 2
encoder_seq_len = 10   # Source sequence length.
decoder_seq_len = 8    # Target sequence length.

# Define a start token (for example, 0; choose according to your vocabulary)
start_token_id = 0

for epoch in range(num_epochs):
    # Dummy encoder inputs using config.vocab_size.
    encoder_input_ids = torch.randint(0, config.vocab_size, (batch_size, encoder_seq_len))
    
    # Full target sequence for the decoder.
    full_decoder_targets = torch.randint(0, config.vocab_size, (batch_size, decoder_seq_len))
    
    # Create shifted decoder inputs by prepending the start token and dropping the last target token.
    decoder_input_ids = torch.cat(
        [torch.full((batch_size, 1), start_token_id, dtype=torch.long), full_decoder_targets[:, :-1]],
        dim=1
    )
    
    # The decoder targets remain as the full target sequence.
    decoder_targets = full_decoder_targets.clone()
    
    optimizer.zero_grad()
    logits, loss = model(encoder_input_ids, decoder_input_ids, decoder_targets)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
