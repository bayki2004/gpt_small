import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from tqdm import tqdm
from dataclasses import dataclass
from model import *
import os
# Hyperparameters
batch_size = 32
block_size = 32  # Increase block size for better context
max_iters = 20000
eval_interval = 300
learning_rate = 1e-4
n_embd = 768
n_head = 8
n_layer = 4
dropout = 0.0
vocab_size = 50257  # GPT2 vocab size
init_from = 'scratch'
bias = False
enc = tiktoken.get_encoding("gpt2")
out_dir = os.path.join(os.path.dirname(__file__), "out")
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
print(device)
train_data = np.memmap(os.path.join('train_small_2.bin'), dtype=np.uint16, mode='r')
validation_data = np.memmap(os.path.join('val_small_2.bin'), dtype=np.uint16, mode='r')
# Function to get a batch of data
data_dir = os.path.join(os.path.dirname(__file__))
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = train_data
    else:
        data = validation_data
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
training_losses = []
validation_losses = []
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
config = ModelConfig(**model_args)
print(config)
model = LanguageModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
current_loss = 1e9
current_val_loss = 1e9
checkpoint = {
    "model_state_dict": model.state_dict(),
    "model_args": model_args,  # Dictionary containing hyperparameters
    "current_loss": current_loss,
    "current_val_loss": current_val_loss,
    "learning_rate": learning_rate,
    "optimizer": optimizer.state_dict(),
    "best_val_loss": best_val_loss
}

if __name__ == "__main__":
    print("starting training")
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            validation_losses.append(losses['val'])
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint["model_state_dict"] = model.state_dict()
                checkpoint["learning_rate"] = learning_rate
                checkpoint["current_loss"] = losses['train']
                checkpoint["current_val_loss"] = losses['val']
                torch.save(checkpoint, os.path.join(out_dir, "best_model.pth"))
                
        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        print(iter, loss.item())
        training_losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    torch.save(checkpoint,  os.path.join(out_dir, "finished_model.pth"))
    # Generate text conditioned on an author
    np.save("training_losses_original_train.npy", np.array(training_losses))
    np.save("validation_losses_original_train.npy", np.array(validation_losses))
    prompt = "Frank M. Chapman <>"
    context = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    generated_ids = model.generate(context, max_new_tokens=10)[0].tolist()
    generated_text = enc.decode(generated_ids)
    print(generated_text)
