import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from collections import Counter
import re
from tqdm import tqdm

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dim_feedforward, max_len, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, d_model))
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if src.dim() == 1:
            src = src.unsqueeze(0)
        if tgt.dim() == 1:
            tgt = tgt.unsqueeze(0)
            
        src_emb = self.embedding(src) + self.positional_encoding[:src.size(1), :].unsqueeze(0)
        tgt_emb = self.embedding(tgt) + self.positional_encoding[:tgt.size(1), :].unsqueeze(0)
        
        if src_mask is None:
            src_mask = torch.ones((src.size(0), src.size(1)), device=src.device).bool()
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
            
        output = self.transformer(src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.fc_out(output)
    
    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from collections import Counter
import re
from tqdm import tqdm
import os

class TextDataset(IterableDataset):
    def __init__(self, file_path, seq_len, chunk_size=1024*1024):
        self.file_path = file_path
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.vocab = self.build_vocab()
        
        # Calculate approximate number of sequences
        file_size = os.path.getsize(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            sample_chunk = f.read(min(chunk_size, file_size))
            words_per_byte = len(sample_chunk.split()) / len(sample_chunk)
        
        self.approx_total_words = file_size * words_per_byte
        self.approx_total_sequences = max(0, self.approx_total_words - seq_len)
        print(f"Approximate number of sequences per epoch: {int(self.approx_total_sequences)}")
        
    def build_vocab(self):
        print("Building vocabulary...")
        word_counts = Counter()
        with open(self.file_path, 'r', encoding='utf-8') as f:
            chunk = f.read(self.chunk_size)
            while chunk:
                words = re.sub(r'[^a-z0-9 ]+', '', chunk.lower()).split()
                word_counts.update(words)
                chunk = f.read(self.chunk_size)
        
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for i, (word, count) in enumerate(word_counts.most_common(9998), start=2):
            vocab[word] = i
        print(f"Vocabulary size: {len(vocab)}")
        return vocab
    
    def __iter__(self):
        buffer = []
        sequence_count = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                    
                words = re.sub(r'[^a-z0-9 ]+', '', chunk.lower()).split()
                indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
                
                indices = buffer + indices
                buffer = indices[-self.seq_len:]
                
                for i in range(0, len(indices) - self.seq_len):
                    sequence_count += 1
                    src = torch.tensor(indices[i:i+self.seq_len])
                    tgt = torch.tensor(indices[i+1:i+self.seq_len+1])
                    yield src, tgt

def train(model, dataset, optimizer, criterion, num_epochs=10, batch_size=32, device='mps' if torch.backends.mps.is_available() else 'cpu'):
    model = model.to(device)
    model.train()
    print(f"Training started... Using device: {device}")
    
    # Calculate steps per epoch
    steps_per_epoch = dataset.approx_total_sequences // batch_size
    print(f"Approximate steps per epoch: {steps_per_epoch}")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        data_loader = DataLoader(dataset, batch_size=batch_size)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        pbar = tqdm(total=steps_per_epoch, desc="Training")
        
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            
            decoder_input = tgt[:, :-1]
            target = tgt[:, 1:]
            
            output = model(src, decoder_input)
            output = output.view(-1, output.size(-1))
            target = target.reshape(-1)
            
            loss = criterion(output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:  # Update progress bar every 10 batches
                pbar.update(10)
                pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}',
                                'batch': f'{num_batches}/{steps_per_epoch}'})
                
            # Optional: break epoch after expected number of steps
            if num_batches >= steps_per_epoch:
                break
                
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")
        pbar.close()

# Example usage
file_path = 'books.txt'
seq_len = 32
batch_size = 32
num_epochs = 10

# Calculate and print dataset size
file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
print(f"Dataset file size: {file_size_mb:.2f} MB")

# Create dataset
dataset = TextDataset(file_path, seq_len)

# Initialize model (previous model initialization code remains the same)
vocab_size = 10000
d_model = 128
num_heads = 4
num_layers = 3
dim_feedforward = 512
max_len = 50
dropout = 0.1

model = SimpleTransformer(vocab_size, d_model, num_heads, num_layers, dim_feedforward, max_len, dropout)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train
train(model, dataset, optimizer, criterion, num_epochs=num_epochs, batch_size=batch_size)