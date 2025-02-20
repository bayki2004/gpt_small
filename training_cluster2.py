import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from tqdm import tqdm
from dataclasses import dataclass
from model import *
import os
import math
from contextlib import nullcontext
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

# Hyperparametsers
batch_size = 8
block_size = 8  # Increase block size for better context
max_iters = 30000
eval_interval = 2000
eval_iters = 1
learning_rate = 3e-4
n_embd = 768
n_head = 8
n_layer = 4
dropout = 0.0
vocab_size = 50257  # GPT2 vocab size
init_from = 'scratch'
bias = False
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = 1e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
grad_clip = 1.0
enc = tiktoken.get_encoding("gpt2")
gradient_accumulation_steps = 5 * 5 # used to simulate larger batch sizes
outdir = os.path.join(os.path.dirname(__file__), "out")
model_dir = os.path.join(outdir, "models")
description = f"{block_size}_{learning_rate}"
save_checkpoints = False
learning_rates = []
training_losses = []
validation_losses = []

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
print(device)
print(max_iters, block_size, batch_size, learning_rate, n_embd)


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

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

iter_num = 0
best_val_loss = 1e9
current_val_loss = 1e9
tokens_per_iter = gradient_accumulation_steps * 1 * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
# Loss estimation function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
# Initialize and train the model
config = ModelConfig(**model_args)
print(config)
if init_from == 'scratch':
    model = LanguageModel(config).to(device)
elif init_from == 'resume':
    ckpt_path = os.path.join(outdir, 'best_model.pth')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = ModelConfig(**model_args)
    model = LanguageModel(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    training_losses = checkpoint['training_losses']
    validation_losses = checkpoint['validation_losses']
    learning_rates = checkpoint['learning_rates']
else:
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = LanguageModel.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

if block_size < model.config.block_size:
    print(f"cropping block size from {model.config.block_size} to {block_size}")
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
#scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)  # Reduces LR by 10x every 1000 steps
scheduler = CosineAnnealingLR(optimizer, T_max=lr_decay_iters, eta_min=min_lr)



if __name__ == "__main__":
    print("starting training")
    xb, yb = get_batch('train')
    
    while iter_num < max_iters:
        """lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr"""
        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            val_loss = losses['val']
            validation_losses.append(val_loss)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if math.isnan(val_loss) or math.isnan(losses['train']) or current_val_loss<val_loss:
                assert False, 'diverging'
            current_val_loss = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if iter_num>0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter,
                        'current_loss': losses['train'],
                        'best_val_loss': val_loss,
                        'learning_rates': learning_rates,
                        'training_losses': training_losses,
                        'validation_losses': validation_losses,
                    }
                    print('saving checkpoint at intermediate models')
                    if save_checkpoints:
                        torch.save(checkpoint, os.path.join(outdir, f"checkpoint_{description}.pt"))
        with torch.amp.autocast('cuda'):
            logits, loss = model(xb, yb)
        scaler.scale(loss).backward()
        for micro_step in range(gradient_accumulation_steps): 
            with ctx:
                logits, loss = model(xb, yb)
                loss = loss / gradient_accumulation_steps
            xb,yb = get_batch('train')
            scaler.scale(loss).backward()
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        loss_ = loss.item() * gradient_accumulation_steps
        print(loss_)
        training_losses.append(loss_)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        iter_num += 1
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter,
        'current_loss': loss_,
        'best_val_loss': best_val_loss,
        'learning_rates': learning_rates,
        'training_losses': training_losses,
        'validation_losses': validation_losses,
    }
    if save_checkpoints:
        torch.save(checkpoint, os.path.join(model_dir, f'lm_{description}_end.pth'))
    # Generate text conditioned on an author
    prompt = "Frank M. Chapman "
    prompt = prompt + ' <> '
    context = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    generated_ids = model.generate(context, max_new_tokens=10)[0].tolist()
    generated_text = enc.decode(generated_ids)
    print(generated_text)
    
