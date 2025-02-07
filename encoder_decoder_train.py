import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32 # How many sentences are grouped together during training
block_size = 8 # Sentence Length with which one does training. Maximum Context length == Block Size
max_iters = 300
eval_interval = 300
learning_rate = 1e-4
n_embd = 64 # Information about one word
device = 'mps' if False else 'cpu'
print(device)
eval_iters = 200
n_head = 8
n_layer = 4
dropout = 0.2
torch.manual_seed(1337)

with open('books.txt', 'r', encoding= 'UTF -8') as f:
    text = f.read()
print("input read")


## Vocabulary in our context is the individual characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)


#Encode and Decode our vocabulary to integers
## Encode: String -> Integer
## Decode: Integer -> String
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda i: ''.join(itos[l] for l in i)
 

#get train and test split
## First encode the whole data to integers
## split that vector of integers into a train and test set
data = torch.tensor(encode(text), dtype=torch.long)
data = data.to(device)
n = int(0.9*len(text))
train_data = data[:n]
val_data = data[n:]

print(train_data.shape, val_data.shape, vocab_size)
## Getting a random batch of training data with corresponding targets
def get_batch(split):
    #generate a small batch of data of inputs x and y
    data = train_data if split == 'train' else val_data
    #generate a random tensor of integers of batch size. 
    #The elements of ix are the indices of the starting position of each block of this specific batch.
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+1+block_size] for i in ix]).to(device)
    return x, y

xb, yb = get_batch('train')


@torch.no_grad
def estimate_loss(encoder=False):
    out = {}
    model.eval()
    if encoder:
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X,Y = get_batch_encoder_decoder(split)
                logits, loss = model(X,Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train
        return out
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
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, 4),
            Block(n_embd, 4),
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, src):
        B, T = src.shape
        token_emb = self.token_embedding(src)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        return x  # Encoder output (context representation)

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # token_embedding_table stores a vector of size n_embd for each word in the vocabulary
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        # position_embedding_table stores a vector of size n_emd for each position that is possible. 
        # Only positions which are possible are 0 -> Block size
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, 4),
            Block(n_embd, 4),
            Block(n_embd, 4),
            nn.LayerNorm(n_embd),
        )
        
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B,T = idx.shape

        #idx and targets are both tensors of size B,T. 
        # Essentially: B = Batch Size and T = Block Size. One also refers to T as the Sequence Length
        # token_embd is now the 3-dimensional tensor (B,T,C) and now stores the embeddings of each element in the idx that we have.
        token_embd = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embd + pos_embd
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:    
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits, loss
   

    def generate(self, idx, max_new_tokens):
        #idx is (B,T) array of indices in current context
        for _ in range(max_new_tokens):
            #get the predictions
            idx_cond = idx[:,-block_size:]
            logits, loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:, -1, :] #becomes (B,C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            # append sampled index to running sequence
            idx = torch.cat((idx,idx_next), dim = 1) # (B, T+1)
        return idx

class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.encoder = Encoder(vocab_size, n_embd, block_size)
        self.decoder = LanguageModel()

    def forward(self, x, y):
        x = self.encoder(x)
        print('x', x.shape)
        x = x[:, :, 0:2].mean(dim=-1)
        print('x', x.shape)
        x = x.long()
        logits, loss = self.decoder(x, y)
        return logits, loss

model = LanguageModel()

m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

print('starting training')
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)

    optimizer.step()



print('done with training')
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))



print("done with training \n")


print('starting with encoder-decoder training')
df = pd.read_csv('dataframe_testing.csv')
df = df.drop(columns=['Link'])


def prepare_data(df):
    train_data = []

    for _, row in df.iterrows():
        encoder_input = f"{row['Title']} | {row['Author']} | {row['Bookshelf']}"
        decoder_input = row['Text']  # Take a chunk of the text

        enc_tokens = encode(encoder_input)
        dec_tokens = encode(decoder_input)

        train_data.append((torch.tensor(enc_tokens, dtype=torch.long), torch.tensor(dec_tokens, dtype=torch.long)))

    return train_data

tokenized_data = prepare_data(df)
data_len = len(tokenized_data)
print(len(train_data))

n = int(0.8*data_len)
train_data = tokenized_data[:n]
val_data = tokenized_data[n:]

def get_batch_encoder_decoder(split):
    if split == 'train':
        data = train_data
    else:
        data = val_data
    data_idx = torch.randint(len(data), (batch_size,))
    x = torch.stack([data[i][0][:block_size] for i in data_idx])
    y = torch.stack([data[i][1][:block_size] for i in data_idx])
    return x, y

xb, yb = get_batch_encoder_decoder('train')

xb, yb = xb.to(device), yb.to(device)
print(xb.shape, yb.shape)

encoder_decoder = EncoderDecoder(vocab_size, n_embd, block_size)
model = encoder_decoder.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print('starting training')
for iter in range(max_iters):
    xb, yb = get_batch_encoder_decoder('train')
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    if iter % eval_interval == 0:
        losses = estimate_loss(encoder=True)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")