import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from tqdm import tqdm
from dataclasses import dataclass
from model import *
from configs import *
import os
# Hyperparameters
batch_size = 8
block_size = 8  # Increase block size for better context
max_iters = 10
eval_interval = 300
learning_rate = 1e-3
n_embd = 8
n_head = 8
n_layer = 4
dropout = 0.0
vocab_size = 50257  # GPT2 vocab size
init_from = 'scratch'
bias = False
enc = tiktoken.get_encoding("gpt2")

gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
print(device)


# Function to get a batch of data
data_dir = os.path.join(os.path.dirname(__file__), 'data')
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir,'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


iter_num = 0
best_val_loss = 1e9
tokens_per_iter = gradient_accumulation_steps * 1 * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
# Loss estimation function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_interval)
        for k in range(eval_interval):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
# Initialize and train the model
config = ModelConfig(**llm_2)
print(config)
model = LanguageModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
if __name__ == "__main__":
    print("starting training")
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    #torch.save(model.state_dict(), "language_model.pth")
    # Generate text conditioned on an author
    prompt = "Frank M. Chapman "
    context = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    generated_ids = model.generate(context, max_new_tokens=1024)[0].tolist()
    generated_text = enc.decode(generated_ids)
    print(generated_text)