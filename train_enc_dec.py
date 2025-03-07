import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from tqdm import tqdm
from dataclasses import dataclass
from encoder_decoder import ModelConfig, EncoderDecoder
import os
import time
import random

# Hyperparameters
batch_size = 32
block_size = 32  # Context length for both encoder and decoder
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
n_embd = 768
n_head = 8
n_encoder_layer = 12
n_decoder_layer = 12
dropout = 0.0
vocab_size = 50257  # GPT2 vocab size
bias = False
enc = tiktoken.get_encoding("gpt2")

gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
outdir = os.path.join(os.path.dirname(__file__), "out")
model_dir = os.path.join(os.path.dirname(__file__), "models")
data_name = ""
init_from = 'scratch'  # 'scratch' or 'resume'
resume_from = 'encdec_512_30000_True_end.pth'
description = f"{block_size}_{init_from}_{data_name}"

# Special tokens
PAD_TOKEN_ID = 0  # Assuming 0 is reserved for padding
BOS_TOKEN_ID = 1  # Beginning of sequence
EOS_TOKEN_ID = 2  # End of sequence

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
print(f"Using device: {device}")

data_dir = os.path.join(os.path.dirname(__file__), 'data')
df = pd.read_csv('data/data_no_dup.csv', nrows=100)
df = df.dropna(subset=['Author', 'Text'])
train_src_data = df.loc[:90]['Author'].values
train_tgt_data = df.loc[:90]['Text'].values
val_src_data = df.loc[91:]['Author'].values
val_tgt_data = df.loc[91:]['Text'].values

# Function to get a batch of data for encoder-decoder
def get_batch(split):
    if split == 'train':
        src_data = train_src_data
        tgt_data = train_tgt_data
    else:
        src_data = val_src_data
        tgt_data = val_tgt_data

    # Create batches for source and target
    src_batch = []
    tgt_batch = []
    tgt_y_batch = []
    max_src_len = 0
    max_tgt_len = 0
    max_seq_len = 32  # Choose an appropriate value

    for _ in range(batch_size):
        # For simplicity, we'll assume src_data and tgt_data are paired
        # In a real implementation, you'd need to ensure proper alignment
        src_idx = random.randint(0, len(src_data) - block_size - 1)
        
        # Get source sequence (encoder input)
        src_seq = src_data[src_idx:src_idx + block_size][0]
        
        src_seq = enc.encode(src_seq)
        src_seq = np.array(src_seq).astype(np.int64)
        # Trim to actual content by finding EOS token or taking full length
        eos_positions = np.where(src_seq == EOS_TOKEN_ID)[0]
        if len(eos_positions) > 0:
            src_len = eos_positions[0] + 1  # Include the EOS token
            src_seq = src_seq[:src_len]
        else:
            src_len = len(src_seq)
        max_src_len = max(max_src_len, src_len)
        
        # Get corresponding target sequence
        tgt_idx = random.randint(0, len(tgt_data) - block_size - 1)
        tgt_seq = tgt_data[tgt_idx:tgt_idx + block_size][0]
        tgt_seq = enc.encode(tgt_seq)
        tgt_seq = np.array(tgt_seq).astype(np.int64)
        # Add BOS at the beginning
        tgt_seq = np.concatenate(([BOS_TOKEN_ID], tgt_seq))
        # Trim to actual content
        eos_positions = np.where(tgt_seq == EOS_TOKEN_ID)[0]
        if len(eos_positions) > 0:
            tgt_len = eos_positions[0] + 1  # Include the EOS token
            tgt_seq = tgt_seq[:tgt_len]
        else:
            tgt_len = len(tgt_seq)
        max_tgt_len = max(max_tgt_len, tgt_len)
        
        # Create target for loss computation (shifted right)
        tgt_seq = tgt_seq[:min(len(tgt_seq), max_seq_len)]
        tgt_y = tgt_seq[1:]  # Target is input shifted by 1
        tgt_seq = tgt_seq[:-1]  # Input is everything except the last token
        
        src_batch.append(src_seq)
        tgt_batch.append(tgt_seq)
        tgt_y_batch.append(tgt_y)
    
    # Pad sequences to the max length in the batch
    src_tensor = torch.zeros((batch_size, max_src_len), dtype=torch.long)
    tgt_tensor = torch.zeros((batch_size, max_seq_len-1), dtype=torch.long)  # -1 because we removed the last token
    tgt_y_tensor = torch.zeros((batch_size, max_seq_len-1), dtype=torch.long)
    
    # Create padding masks
    src_mask = torch.zeros((batch_size, 1, 1, max_src_len), dtype=torch.bool)
    tgt_mask = torch.zeros((batch_size, 1, 1, max_seq_len-1), dtype=torch.bool)
    
    for i in range(batch_size):
        src_len = len(src_batch[i])
        tgt_len = len(tgt_batch[i])
        tgt_y_len = len(tgt_y_batch[i])
        
        src_tensor[i, :src_len] = torch.tensor(src_batch[i])
        tgt_tensor[i, :tgt_len] = torch.tensor(tgt_batch[i])
        tgt_y_tensor[i, :tgt_y_len] = torch.tensor(tgt_y_batch[i])
        
        # Create masks (1 for tokens, 0 for padding)
        src_mask[i, 0, 0, :src_len] = 1
        tgt_mask[i, 0, 0, :max_seq_len] = 1
    # Move to device
    if device_type == 'cuda':
        # Pin arrays for asynchronous transfer
        src_tensor = src_tensor.pin_memory().to(device, non_blocking=True)
        tgt_tensor = tgt_tensor.pin_memory().to(device, non_blocking=True)
        tgt_y_tensor = tgt_y_tensor.pin_memory().to(device, non_blocking=True)
        src_mask = src_mask.pin_memory().to(device, non_blocking=True)
        tgt_mask = tgt_mask.pin_memory().to(device, non_blocking=True)
    else:
        src_tensor = src_tensor.to(device)
        tgt_tensor = tgt_tensor.to(device)
        tgt_y_tensor = tgt_y_tensor.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
    print(src_tensor.shape, tgt_tensor.shape, tgt_y_tensor.shape, src_mask.shape, tgt_mask.shape)
    return src_tensor, tgt_tensor, tgt_y_tensor, src_mask, tgt_mask

# Loss estimation function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_interval)
        for k in range(eval_interval):
            print(k)
            src, tgt, tgt_y, src_mask, tgt_mask = get_batch(split)
            _, loss = model.compute_loss(src, tgt, tgt_y, src_mask, tgt_mask)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# Model initialization
model_args = dict(
    block_size=block_size,
    vocab_size=vocab_size,
    n_encoder_layer=n_encoder_layer,
    n_decoder_layer=n_decoder_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias
)

iter_num = 0
best_val_loss = 1e9
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"Tokens per iteration will be: {tokens_per_iter:,}")
training_losses = []
validation_losses = []

# Initialize or resume model
if init_from == 'scratch':
    config = ModelConfig(**model_args)
    model = EncoderDecoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# Handle block size mismatch
if block_size < config.block_size:
    print(f"Cropping block size from {config.block_size} to {block_size}")
    # Implement a method similar to crop_block_size in your EncoderDecoder model if needed
    # model.crop_block_size(block_size)
    model_args['block_size'] = block_size
    
print(config)
print(description)

# Training loop
if __name__ == "__main__":
    print("Starting training")
    
    # Make sure output directories exist
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize progress tracking
    start_time = time.time()
    
    for iter in range(iter_num, max_iters):
        # Evaluate the model
        if iter % eval_interval == 0 and iter > 0:
            losses = estimate_loss()
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            validation_losses.append(losses['val'])
            
            # Save best model
            if best_val_loss > losses['val']:
                best_val_loss = losses['val']
                
        src, tgt, tgt_y, src_mask, tgt_mask = get_batch('train')
        if src_mask is not None and src_mask.dim() == 4:
            # For encoder self-attention, we need [B, 1, T, T]
            # We can expand the last dimension to T
            B, _, _, T = src_mask.shape
            src_mask = src_mask.squeeze(1).squeeze(1).unsqueeze(-2)  # [B, 1, T]
            src_mask = src_mask.expand(-1, T, -1)  # [B, T, T]

        if tgt_mask is not None and tgt_mask.dim() == 4:
            # For decoder self-attention, we need [B, 1, T, T]
            B, _, _, T = tgt_mask.shape
            tgt_mask = tgt_mask.squeeze(1).squeeze(1).unsqueeze(-2)  # [B, 1, T]
            tgt_mask = tgt_mask.expand(-1, T, -1)  # [B, T, T]

        logits, loss = model.compute_loss(src, tgt, tgt_y, src_mask, tgt_mask)
        
        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        print(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Print progress
        if iter % 10 == 0:
            print(f"Iteration {iter}: loss = {loss.item():.4f}")
        
        training_losses.append(loss.item())
        
    
    
    
    # Generate an example translation
    print("Generating sample translation:")
    
    # Example input (customize based on your data)
    prompt = "Hello, how are you today?"
    
    # Tokenize input
    src_tokens = torch.tensor([enc.encode(prompt) + [EOS_TOKEN_ID]], dtype=torch.long).to(device)
    src_mask = torch.ones((1, 1, 1, src_tokens.size(1)), dtype=torch.bool).to(device)
    
    # Generate translation
    generated = model.generate(
        src_tokens, 
        max_len=10, 
        bos_token_id=BOS_TOKEN_ID, 
        eos_token_id=EOS_TOKEN_ID, 
        src_mask=src_mask
    )
    
    # Decode the generated tokens
    generated_text = enc.decode(generated[0].tolist())
    print(f"Input: {prompt}")
    print(f"Generated: {generated_text}")
    
    print(f"Training completed in {(time.time() - start_time)/60:.2f} minutes")