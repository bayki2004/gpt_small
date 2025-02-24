import torch
from torch.nn import functional as F
from model import *
from dataclasses import dataclass
import pandas as pd
import math
import numpy as np
import seaborn as sns
import tiktoken
import matplotlib.pyplot as plt
import gc 
import time
import os

start_time = time.time()
enc = tiktoken.get_encoding("gpt2")
block_size = 256
n_layer = 4
n_head = 8
n_embd = 768
bias = False
vocab_size = 50257
dropout = 0.0
batch_size = 50
normalization_factor = 100
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
model_dir = os.path.join(os.path.dirname(__file__), 'models')
model_name = 'lm_512_30000_True_end.pth'
# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print('at use.py')
print(device)
train_data_name = 'data/train.bin'
delimiter = "*** START OF THE PROJECT GUTENBERG EBOOK"

ckpt_path = os.path.join(model_dir, model_name)
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']
# force these config attributes to be equal otherwise we can't even resume training
# the rest of the attributes (e.g. dropout) can stay as desired from command line
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]
block_size = model_args['block_size']    
gptconf = ModelConfig(**model_args)
model = LanguageModel(gptconf)
state_dict = checkpoint['model']

unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
# Function to generate text from a prompt
def generate_text(prompt, max_new_tokens=10):
    # Encode the prompt and convert it to a tensor
    prompt = enc.encode(prompt)
    prompt = prompt + enc.encode(' <> ')
    context = torch.tensor(prompt, dtype=torch.long).unsqueeze(0).to(device)
    generated_ids = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    generated_text = enc.decode(generated_ids)
    # Decode the generated indices to text
    return generated_text

def compute_log_probability(model, x, y):
        log_prob = 0.0
        idx = x  # Start with the given prefix x
        y_len = y.shape[1]
        ctxt_length = min(y_len, 20)
        for i in range(ctxt_length):
            idx_cond = idx[:, -block_size:]  # Truncate to block size
            logits, _ = model(idx_cond)  # Get logits
            logits = logits[:, -1, :]  # Get logits for the last token in the sequence
            probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
            y_next = y[:, i]  # The next token from the target sequence y
            token_prob = probs[:, y_next].item()  # Get the probability of generating y_next
            max_value, max_index = torch.max(probs, dim=1)
            #print(max_value.item(), token_prob, enc.decode(max_index.tolist()), enc.decode(y_next.tolist()))
            log_prob += -1 * math.log(token_prob + 1e-9)  # Add log probability (avoid log(0) with 1e-9)
            idx = torch.cat((idx, y_next.unsqueeze(1)), dim=1)  # Append y_next to the sequence
        return log_prob/ctxt_length

def get_batch_training_set(batch_size, train_data_name=train_data_name):
    data = np.memmap(os.path.join(train_data_name), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+block_size:i+block_size+ block_size]).astype(np.int64)) for i in ix])
    del data
    gc.collect()
    return (x.to(device), y.to(device))



def plot_heat(log_prob_matrix):
    col_min = log_prob_matrix.min(axis=0, keepdims=True)  # Min per column
    col_max = log_prob_matrix.max(axis=0, keepdims=True)  # Max per column
    norm_matrix = (log_prob_matrix - col_min) / (col_max - col_min + 1e-8)  # Avoid div by zero
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(norm_matrix, annot=False, cmap='coolwarm', linewidths=0.5)

    # Highlight diagonal elements where i == j
    for i in range(log_prob_matrix.shape[0]):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='yellow', lw=2))
    min_indices = np.argmin(log_prob_matrix, axis=0)  # Indices of min values per column
    for col, row in enumerate(min_indices):
        plt.scatter(col + 0.5, row + 0.5, color='lime', s=100, edgecolors='black', lw=1.5, marker='o')  # Green markers
    plt.title("Log Probabilities Heatmap (Author Prompt vs Target Text)")
    plt.xlabel("Author Prompt Index (i)")
    plt.ylabel("Target Text Index (j)")
    plt.show()

if __name__ == "__main__":
    i = 0
    prompt = "Color Photography, Vol. 1, No. 1 Various"
    #print('generating text')
    #print(generate_text(prompt))

    log_probs_matrix = np.zeros((batch_size, batch_size))
    x, y = get_batch_training_set(batch_size)
    print(x.size(), y.size())
    for i in range(batch_size):
        for j in range(batch_size):
            log_probs_matrix[i, j] = compute_log_probability(
                                    model, 
                                    x[i].unsqueeze(0).clone().detach().to(device),  # Clone and detach
                                    y[j].unsqueeze(0).clone().detach().to(device)   # Clone and detach
                                )
    print(log_probs_matrix)
    plot_heat(log_probs_matrix)
    
    df = pd.read_csv('data/dataframe_testing.csv', nrows=100).dropna(subset=['Author', 'Text'])
    df = df.sample(n=batch_size, random_state=42)
    print(df.shape)
    log_probs_matrix = np.zeros((df.shape[0], df.shape[0]))
    for k in range(normalization_factor):
        for j in range(df.shape[0]):
            target_text = df.iloc[j]['Text']
            ids = enc.encode_ordinary(target_text)
            print(enc.decode(ids[:10]))
            for i in range(0, df.shape[0]):
                author_prompt = f"{df.iloc[i]['Author']} <> "
                author_prompt = enc.encode(author_prompt)
                target_text_enc = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                author_prompt = torch.tensor(author_prompt, dtype=torch.long).unsqueeze(0).to(device)
                probs = compute_log_probability(model, author_prompt, target_text_enc)
                log_probs_matrix[j, i] += probs
            print('----------')
        print(k)
    log_probs_matrix /= normalization_factor
    print(log_probs_matrix)
    plot_heat(log_probs_matrix)

print("--- %s seconds ---" % (time.time() - start_time))

    