import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from tqdm import tqdm
from dataclasses import dataclass
from model import *
from sklearn.model_selection import train_test_split
import os
import gc
# Hyperparameters
batch_size = 32
block_size = 48  # Increase block size for better context
max_iters = 150000
eval_interval = 1000
eval_iters = 500
learning_rate = 3e-4
n_embd = 768
n_head = 12
n_layer = 12
dropout = 0.0
vocab_size = 50257  # GPT2 vocab size
vocab_size = 100278
bias = False
encoding = 'gpt-4'
enc = tiktoken.get_encoding("gpt2")
enc = tiktoken.encoding_for_model('gpt-4')
eop_token = 100276
save_checkpoints = True
simple_learning_rate = True
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
outdir = os.path.join(os.path.dirname(__file__), "out")
model_dir = os.path.join(os.path.dirname(__file__), "models")
data_name = "_no_dup_small"
init_from = 'scratch'
resume_from = 'lm_48_scratch_names_end.pth'
description = f"{block_size}_{init_from}_middle_text"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
print(device)


# Function to get a batch of data
data_dir = os.path.join(os.path.dirname(__file__), 'data')
#arr = np.memmap(os.path.join(data_dir, 'data_author_names.bin'), dtype=np.float32, mode='r')
#offsets = np.load("/itet-stor/kbayraktar/net_scratch/training_ml/data/author_names_offsets.npy")
#reconstructed_tensors = [arr[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]

#train_memmap, test_memmap = train_test_split(reconstructed_tensors, test_size=0.005, random_state=42)
#print(f"train memmap shape: {len(train_memmap)}, test memmap shape: {len(test_memmap)}")

train_tensors_middle_text = torch.load(os.path.join(data_dir, 'train_tensors_middle_text.pt'))
test_tensors_middle_text = torch.load(os.path.join(data_dir, 'test_tensors_middle_text.pt'))

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, f"train{data_name}.bin"), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, f"val{data_name}.bin"), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
def get_batch_pairs(split):
    if split=='train':
        data = train_memmap
    else:
        data = test_memmap
    ix = torch.randint(len(data), (1,))
    data = data[ix]
    print(data.shape)
    data = data[0]
    if len(data) < block_size:
        data = np.concatenate(( np.zeros(block_size - len(data) +3), data))
    ix = torch.randint(1, len(data) - block_size-1, (batch_size-1,))
    ix = np.concatenate(([0], ix))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    if device=='cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x,y
def get_batch_tensor(split):
    if split=='train':
        data = train_memmap
    else:
        data = test_memmap
    ix = torch.randint(len(data), (1,))
    data = data[ix]
    ix = torch.randint(1, len(data) - block_size-1, (batch_size-1,))
    ix = np.concatenate(([0], ix))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type=='cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x,y
def get_batch_tensor_middle(split):
    if split=='train':
        data = train_tensors_middle_text
    else:
        data = test_tensors_middle_text
    ix = torch.randint(len(data), (1,))
    data = data[ix]
    ix = torch.randint(1, len(data) - block_size-1, (batch_size-1,))
    ix = np.concatenate(([0], ix))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    if device_type=='cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x,y
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
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_tensor_middle(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
# Initialize and train the model


if init_from == 'scratch':
    config = ModelConfig(**model_args)
    model = LanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(model.config.block_size)
    print(model.config.n_embd)
    print(model.config.n_layer)
    print(model.config.n_head)
    print(model.config.vocab_size)

elif init_from == 'resume':
    ckpt_path = os.path.join(model_dir, resume_from)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    
    config = ModelConfig(**model_args)
    model = LanguageModel(config)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    optimizer_state_dict = checkpoint['optimizer']
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    optimizer.load_state_dict(optimizer_state_dict)
    iter_num = checkpoint['iter_num']
    iter_num = 0
    best_val_loss = checkpoint['best_val_loss']
    best_val_loss = 1e9
    training_losses = checkpoint['training_losses']
    validation_losses = checkpoint['validation_losses']
if block_size < model.config.block_size:
    print(f"cropping block size from {model.config.block_size} to {block_size}")
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)
    config = ModelConfig(**model_args)
print(model.config.block_size)
print(description)
if __name__ == "__main__":
    print("starting training")
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            if math.isnan(losses['val']):
                print("NaN loss detected, stopping training")
                break
            if best_val_loss > losses['val']:
                best_val_loss = losses['val']
    
                if iter>0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'current_loss': losses['train'],
                        'best_val_loss': losses['val'],
                        'training_losses': training_losses,
                        'validation_losses': validation_losses,
                    }
                    
                    if save_checkpoints:
                        print('saving checkpoint at out directory')
                        torch.save(checkpoint, os.path.join(outdir, f"ckpt_{description}.pt"))
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            validation_losses.append(losses['val'])
        xb, yb = get_batch_tensor_middle('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        #print(iter, loss.item())
        training_losses.append(loss.item())
        loss_ = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'current_loss': loss_,
        'best_val_loss': best_val_loss,
        'training_losses': training_losses,
        'validation_losses': validation_losses,
    }

    if save_checkpoints:
        torch.save(checkpoint, os.path.join(model_dir, f'lm_{description}_end.pth'))
    # Generate text conditioned on an author
    prompt = "Frank M. Chapman "
    context = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    generated_ids = model.generate(context, max_new_tokens=10)[0].tolist()
    generated_text = enc.decode(generated_ids)
    print(generated_text)
