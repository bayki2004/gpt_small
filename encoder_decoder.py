import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from transformers import T5EncoderModel

import math
#encoder = BertModel.from_pretrained('bert-base-uncased')
t5_encoder = T5EncoderModel.from_pretrained('t5-small')  # or t5-base, etc.
t5_hidden_size = t5_encoder.config.d_model  # For t5-small, this is typically 512
class EncoderProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.proj(x)

# Example: if your GPT-2 decoder uses 768 and t5_hidden_size is 512:
if t5_hidden_size != 768:
    encoder_proj = EncoderProjection(t5_hidden_size, 768)
else:
    encoder_proj = nn.Identity()

# Transformer model components (unchanged)
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

class Head(nn.Module):
    def __init__(self, head_size, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.query = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class CrossAttention(nn.Module):
    def __init__(self, head_size, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.query = nn.Linear(config.n_embd, head_size, bias=config.bias)
        # Added projection to map from head_size back to config.n_embd
        self.proj = nn.Linear(head_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, encoder_out):
        # x: decoder input; encoder_out: encoder's hidden states
        B, T, _ = x.shape
        # Assume encoder_out has shape (B, S, config.n_embd)
        k = self.key(encoder_out)    # shape: (B, S, head_size)
        v = self.value(encoder_out)  # shape: (B, S, head_size)
        q = self.query(x)            # shape: (B, T, head_size)
        # Use head_size for scaling instead of full embedding dimension
        attn_weights = q @ k.transpose(-2, -1) * (q.size(-1) ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ v       # shape: (B, T, head_size)
        # Project back to config.n_embd
        out = self.proj(out)         # shape: (B, T, config.n_embd)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(head_size, config)  # Self-attention (masked)
        self.cross_attn = CrossAttention(head_size, config)  # Cross-attention layer
        self.ffwd = FeedForward(config)
        # Three layer norms for the three sub-layers
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
    
    def forward(self, x, encoder_out):
        # Self-attention with residual connection
        x = x + self.sa(self.ln1(x))
        # Cross-attention over encoder output with residual connection
        x = x + self.cross_attn(self.ln2(x), encoder_out)
        # Feed-forward network with residual connection
        x = x + self.ffwd(self.ln3(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,  head_size, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_head * head_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention( head_size, config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

@dataclass
class ModelConfig:
    block_size: int = 1024
    vocab_size: int = 32128 # GPT-2 config.vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class EncoderDecoderModel(nn.Module):
    def __init__(self, config, t5_encoder, encoder_proj):
        super().__init__()
        self.encoder = t5_encoder  # Pretrained encoder
        # Use your existing decoder token/position embeddings or create new ones
        self.encoder_proj = encoder_proj

        self.decoder_token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.decoder_position_embedding = nn.Embedding(config.block_size, config.n_embd)
        # Replace the old blocks with the new decoder blocks that include cross-attention
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.config = config

    def forward(self, encoder_input_ids, decoder_input_ids, decoder_targets=None):
        # Pass the encoder input through the pretrained encoder
        # Assume the encoder returns a tuple where the first element is the hidden states
        encoder_outputs = self.encoder(encoder_input_ids)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        # Project if necessary
        encoder_hidden_states = self.encoder_proj(encoder_hidden_states)
        
        B, T = decoder_input_ids.shape
        device = decoder_input_ids.device
        token_embd = self.decoder_token_embedding(decoder_input_ids)
        pos_embd = self.decoder_position_embedding(torch.arange(T, device=device))
        x = token_embd + pos_embd
        
        # Pass through each decoder block along with the encoder output
        for block in self.decoder_blocks:
            x = block(x, encoder_hidden_states)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if decoder_targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            decoder_targets = decoder_targets.view(B * T)
            loss = F.cross_entropy(logits, decoder_targets)
        return logits, loss

    def generate(self, encoder_input_ids, decoder_start_token, max_new_tokens):
        # Use last_hidden_state and project encoder outputs:
        encoder_hidden_states = self.encoder(encoder_input_ids).last_hidden_state
        encoder_hidden_states = self.encoder_proj(encoder_hidden_states)
        # Start the decoder with a given start token; shape: (batch_size, 1)
        decoder_input_ids = decoder_start_token.unsqueeze(0)  
        for _ in range(max_new_tokens):
            token_embd = self.decoder_token_embedding(decoder_input_ids)
            T = decoder_input_ids.size(1)
            pos_embd = self.decoder_position_embedding(torch.arange(T, device=decoder_input_ids.device))
            x = token_embd + pos_embd
            for block in self.decoder_blocks:
                x = block(x, encoder_hidden_states)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
        return decoder_input_ids

