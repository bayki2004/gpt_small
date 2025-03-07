import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math


# Transformer model components
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

class EncoderHead(nn.Module):
    """Encoder attention head with no causal mask"""
    def __init__(self, head_size, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.query = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        
        # Apply padding mask if provided
        if mask is not None:
            wei = wei.masked_fill(mask == 0, float('-inf'))
            
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class DecoderHead(nn.Module):
    """Decoder attention head with causal mask"""
    def __init__(self, head_size, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.query = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        
        # Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Also apply padding mask if provided
        if mask is not None:
            wei = wei.masked_fill(mask == 0, float('-inf'))
            
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class CrossAttentionHead(nn.Module):
    """Cross-attention for decoder to attend to encoder outputs"""
    def __init__(self, head_size, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.query = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, encoder_output, encoder_mask=None):
        B, T, C = x.shape
        k = self.key(encoder_output)  # keys from encoder
        q = self.query(x)             # queries from decoder
        v = self.value(encoder_output)  # values from encoder
        
        wei = q @ k.transpose(-2, -1) * C**-0.5
        
        # Apply encoder padding mask if provided
        if encoder_mask is not None:
            wei = wei.masked_fill(encoder_mask == 0, float('-inf'))
            
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class EncoderMultiHeadAttention(nn.Module):
    def __init__(self, head_size, config):
        super().__init__()
        self.heads = nn.ModuleList([EncoderHead(head_size, config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_head * head_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class DecoderMultiHeadAttention(nn.Module):
    def __init__(self, head_size, config):
        super().__init__()
        self.heads = nn.ModuleList([DecoderHead(head_size, config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_head * head_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class CrossMultiHeadAttention(nn.Module):
    def __init__(self, head_size, config):
        super().__init__()
        self.heads = nn.ModuleList([CrossAttentionHead(head_size, config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_head * head_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, encoder_output, encoder_mask=None):
        out = torch.cat([h(x, encoder_output, encoder_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = EncoderMultiHeadAttention(head_size, config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = DecoderMultiHeadAttention(head_size, config)
        self.ca = CrossMultiHeadAttention(head_size, config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)

    def forward(self, x, encoder_output, mask=None, encoder_mask=None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ca(self.ln2(x), encoder_output, encoder_mask)
        x = x + self.ffwd(self.ln3(x))
        return x

@dataclass
class ModelConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 config.vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_encoder_layer: int = 6
    n_decoder_layer: int = 6
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True
    
class EncoderDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)    
        self.encoder_position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.decoder_position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_encoder_layer)])
        self.encoder_ln_f = nn.LayerNorm(config.n_embd)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_decoder_layer)])
        self.decoder_ln_f = nn.LayerNorm(config.n_embd)
        
        # Output projection
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
    def encode(self, src, src_mask=None):
        B, T = src.shape
        device = src.device
        
        # Embed tokens and positions
        token_embd = self.token_embedding_table(src)
        pos_embd = self.encoder_position_embedding(torch.arange(T, device=device))
        x = token_embd + pos_embd
        
        # Pass through encoder blocks
        for block in self.encoder_blocks:
            x = block(x, src_mask)
        
        x = self.encoder_ln_f(x)
        return x
    
    def decode(self, tgt, encoder_output, tgt_mask=None, encoder_mask=None):
        B, T = tgt.shape
        device = tgt.device
        
        # Embed tokens and positions
        token_embd = self.token_embedding_table(tgt)
        pos_embd = self.decoder_position_embedding(torch.arange(T, device=device))
        x = token_embd + pos_embd
        
        # Pass through decoder blocks
        for block in self.decoder_blocks:
            x = block(x, encoder_output, tgt_mask, encoder_mask)
        
        x = self.decoder_ln_f(x)
        return x
    
    def forward(self, src, tgt,  src_mask=None, tgt_mask=None):
    
        encoder_output = self.encode(src, src_mask)
        # Adjust src_mask for cross-attention:
        if src_mask is not None:
            # If src_mask is of shape [B, T_src, T_src] (from your training loop),
            # extract the original mask from the first row.
            if src_mask.dim() == 3:
                orig_src_mask = src_mask[:, 0, :]  # shape: [B, T_src]
            else:
                # Otherwise, assume it's still [B, 1, 1, T_src]
                orig_src_mask = src_mask.squeeze(1).squeeze(1)  # shape: [B, T_src]
            T_tgt = tgt.size(1)  # e.g., 31
            encoder_mask = orig_src_mask.unsqueeze(1).expand(-1, T_tgt, -1)  # shape: [B, T_tgt, T_src]
        else:
            encoder_mask = None
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, encoder_mask)
        
        
        logits = self.lm_head(decoder_output)
        
        return logits
    
    def compute_loss(self,src, tgt, tgt_y, src_mask=None, tgt_mask=None):
        logits = self.forward(src, tgt,src_mask, tgt_mask) 
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        tgt_y = tgt_y.reshape(B*T)
        loss = F.cross_entropy(logits, tgt_y)
        return logits, loss
    
    def generate(self, src, max_len, bos_token_id, eos_token_id, src_mask=None):
        
        device = src.device
        B = src.shape[0]
        
        if src_mask is not None and src_mask.dim() == 4:
            # Get original vector mask: shape [B, T_src]
            orig_src_mask = src_mask.squeeze(1).squeeze(1)  # [B, T_src]
            # Create encoder mask for self-attention (square mask): [B, T_src, T_src]
            T_src = orig_src_mask.size(1)
            enc_mask = orig_src_mask.unsqueeze(1).expand(-1, T_src, -1)
        else:
            orig_src_mask = None
            enc_mask = None

        # Encode the source using the square mask for encoder self-attention
        encoder_output = self.encode(src, enc_mask)
        
        # Initialize decoder input with BOS token
        decoder_input = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_len):
            # For cross-attention, expand the original source mask to shape [B, T_dec, T_src]
            T_dec = decoder_input.size(1)
            if orig_src_mask is not None:
                cross_mask = orig_src_mask.unsqueeze(1).expand(-1, T_dec, -1)
            else:
                cross_mask = None
            
            # Decode using cross_mask for cross-attention
            decoder_output = self.decode(decoder_input, encoder_output, tgt_mask=None, encoder_mask=cross_mask)
            logits = self.lm_head(decoder_output[:, -1:])  # Get logits for the last token
            probs = F.softmax(logits.squeeze(1), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # Stop if every sequence in the batch produces an EOS token
            if (decoder_input == eos_token_id).any(dim=1).all():
                break
                
        return decoder_input

    
    # Helper function to create padding masks
    @staticmethod
    def create_padding_mask(seq, pad_token_id):
        """Create mask for padding tokens: 0 for pad tokens, 1 for others"""
        return (seq != pad_token_id).unsqueeze(1).unsqueeze(2)  # Shape: [B, 1, 1, T]