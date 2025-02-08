import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer

# Hyperparameters
batch_size = 16
block_size = 128  # Increase block size for better context
max_iters = 3000
eval_interval = 300
learning_rate = 1e-4
n_embd = 256
n_head = 8
n_layer = 4
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# Load dataset
df = pd.read_csv('gutenberg_data_with_text.csv', nrows=10)  # Replace with your actual CSV file name
df = df.dropna(subset=['Author', 'Text'])  # Ensure there are no missing values

# Prepare text data: "Author: <author_name>\nText: <text>"
texts = ["Author: " + row['Author'] + "\nText: " + row['Text'] for _, row in df.iterrows()]
combined_text = '\n'.join(texts)

# Tokenizer setup using GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token exists
vocab_size = tokenizer.vocab_size
# Truncate sequences that exceed the maximum length
max_length = tokenizer.model_max_length  # Default is 1024 for GPT-2

# Encode text
data = tokenizer.encode(combined_text, truncation=True, padding=True, max_length=max_length)
data = torch.tensor(data, dtype=torch.long).to(device)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

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
        losses = torch.zeros(eval_interval)
        for k in range(eval_interval):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# Transformer model components (unchanged)
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
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
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
        self.proj = nn.Linear(num_heads * head_size, n_embd)
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

# Initialize and train the model
model = LanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print("starting training")
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

# Generate text conditioned on an author
prompt = "Author: AuthorX\nText: "
context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
generated_ids = model.generate(context, max_new_tokens=200)[0].tolist()
generated_text = tokenizer.decode(generated_ids)
print(generated_text)
