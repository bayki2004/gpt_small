import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

#hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-4
n_embd = 8
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

print(device)
eval_iters = 200
n_head = 8
n_layer = 6
dropout = 0.2
torch.manual_seed(1337)


df = pd.read_csv('gutenberg_data_with_text.csv', nrows=100)
df = df.drop(labels=['Link'], axis=1)
df['encoder_input'] = df['Title'] + " " + df['Author'] + " " + df['Bookshelf']
df = df[~df['encoder_input'].isna()]
chars = sorted(list(set(' '.join(df['encoder_input'].astype(str).values) + ' '.join(df['Text'].astype(str).values))))
vocab_size = len(chars)
print(vocab_size)
#encoder decoder mapping
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda i: ''.join(itos[l] for l in i)
 

max_length = 512  # Or any other fixed length you'd like to use


# Convert the data into tokenized format
train_data = []
for idx, row in df.iterrows():
    encoder_input = encode(row['encoder_input'])
    decoder_input = encode(row['Text'])
    train_data.append((torch.tensor(encoder_input), torch.tensor(decoder_input)))

def get_batch():
    # Randomly pick a sample
    ix = torch.randint(len(train_data), (batch_size,))
    encoder_inputs = [train_data[i][0] for i in ix]
    decoder_inputs = [train_data[i][1] for i in ix]
    # Pad the sequences to the same length
    x = pad_sequence(encoder_inputs, batch_first=True, padding_value=0).to(device)
    y = pad_sequence(decoder_inputs, batch_first=True, padding_value=0).to(device)
    return x, y

xb, yb = get_batch()
print(xb.shape, yb.shape)
block_size_encoder = xb.shape[-1]
print(block_size_encoder)

@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train
    return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, n_embd*4), 
                                nn.ReLU(), 
                                nn.Linear(n_embd *4, n_embd), 
                                nn.Dropout(dropout),
                                )
    def forward(self,x):
        return self.net(x)

class Head(nn.Module):
    # one head of self-attention

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device=device)))
        self.dropout = nn.Dropout(dropout)


    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        #compute attention scores 'affinities'
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-1e9'))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
        #perform weighted aggregation of the values
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
        out = torch.cat([h(x) for h in self.heads], dim= -1)
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x+ self.sa(self.ln1(x))
        x = x+ self.ffwd(self.ln2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layers, n_head):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embd)
    
    def forward(self, src):
        B, T = src.shape
        print(B,T)
        token_emb = self.token_embedding(src)  # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=device))  # (T, n_embd)
        x = token_emb + pos_emb  # (B, T, n_embd)
        x = self.blocks(x)  # Pass through transformer blocks
        x = self.ln(x)  # Final layer normalization
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layers, n_head):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, tgt, encoder_output):
        B, T = tgt.shape
        token_emb = self.token_embedding(tgt)  # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=device))  # (T, n_embd)
        x = token_emb + pos_emb  # (B, T, n_embd)
        x = self.blocks(x)  # Pass through transformer blocks
        x = self.ln(x)  # Final layer normalization
        logits = self.lm_head(x)  # Map to vocabulary size
        return logits

class EncoderDecoderModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, n_embd, block_size, n_layers, n_head):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, n_embd, block_size, n_layers, n_head)
        self.decoder = Decoder(tgt_vocab_size, n_embd, block_size, n_layers, n_head)
    
    def forward(self, src, tgt):
        encoder_output = self.encoder(src)  # Encode the source
        logits = self.decoder(tgt, encoder_output)  # Decode using the encoder output
        return logits

model = EncoderDecoderModel(vocab_size, vocab_size, n_embd, block_size, n_layer, n_head).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print('starting training')
for iter in range(max_iters):
    model.train()
    xb_src, yb_tgt = get_batch()  # `src` and `tgt` batches
    logits = model(xb_src, yb_tgt)
    
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    yb_tgt = yb_tgt.view(B * T)
    loss = F.cross_entropy(logits, yb_tgt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if iter % eval_interval == 0:
        print(f"Iter {iter}, Loss: {loss.item()}")




print('done with training')
"""context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))"""

 

