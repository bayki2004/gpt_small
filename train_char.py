import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
# Hyperparameters
batch_size = 32
block_size = 32
max_iters = 300
eval_iters = 100
eval_interval = 100
learning_rate = 1e-4
n_embd = 64
n_head = 8
n_layer = 4
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# Load dataset
df = pd.read_csv('gutenberg_data_with_text.csv', nrows=10)  # Replace with your actual CSV file name
df = df.dropna(subset=['Author', 'Text'])  # Ensure there are no missing values
print('loaded data')
# Prepare the combined text: "Author: <author_name>\nText: <text>"
texts = ["Author: " + row['Author'] + "\nText: " + row['Text'] for _, row in df.iterrows()]
combined_text = '\n'.join(texts)
print("combined text")
# Build vocabulary
chars = sorted(list(set(combined_text)))
vocab_size = len(chars)
print('vocab size')
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)
print('encoded')
# Encode the entire text and split into train/validation sets
data = torch.tensor(encode(combined_text), dtype=torch.long).to(device)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print('prepared data')
# Function to get a batch of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
    return x, y

# Loss estimation function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# Transformer model components
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device=device)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-1e9'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
model = LanguageModel().to(device)
# Initialize and train the model
print('training')
if __name__ == "__main__":
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    # Save the model after training
    torch.save(model.state_dict(), "language_model.pth")


# Generate text conditioned on an author
context = torch.tensor(encode("Author: AuthorX\nText: "), dtype=torch.long).unsqueeze(0).to(device)
generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print(generated_text)
