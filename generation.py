import torch
from torch.nn import functional as F
from model import *
from configs import llm_3, llm_4
import pandas as pd
import math
import numpy as np
import seaborn as sns
import tiktoken
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from functools import partial
from loading_text import extract_text_between_markers, end_of_text
import time
import os

start_time = time.time()
enc = tiktoken.get_encoding("gpt2")
block_size = 8
n_layer = 4
n_head = 8
n_embd = 768
bias = False
vocab_size = 50257
dropout = 0.0
batch_size = 30
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
# Device configuration
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print('at use.py')
print(device)
configs = ModelConfig(**model_args)
delimiter = "*** START OF THE PROJECT GUTENBERG EBOOK"
# Load the trained model
model = LanguageModel(configs)
model.to(device)
model.load_state_dict(torch.load("language_model_original_small_32.pth", map_location=torch.device('mps')),strict=False )
model.eval()

# Function to generate text from a prompt
def generate_text(prompt, max_new_tokens=10):
    # Encode the prompt and convert it to a tensor

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
train_data = np.memmap(os.path.join('train_small_2.bin'), dtype=np.uint16, mode='r')
def plot_heat(log_prob_matrix):
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(log_probs_matrix, annot=False, cmap='coolwarm', linewidths=0.5)

    # Highlight diagonal elements where i == j
    for i in range(log_prob_matrix.shape[0]):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='yellow', lw=2))

    plt.title("Log Probabilities Heatmap (Author Prompt vs Target Text)")
    plt.xlabel("Author Prompt Index (i)")
    plt.ylabel("Target Text Index (j)")
    plt.show()

if __name__ == "__main__":
    i = 0
    prompt = "Color Photography, Vol. 1, No. 1 Various"
    prompt = enc.encode(prompt)
    prompt = prompt + enc.encode(' <> ')
    #print(generate_text(prompt))
    df = pd.read_csv('data/dataframe_testing.csv', nrows=20).dropna(subset=['Author', 'Text'])
    log_probs_matrix = np.zeros((batch_size, batch_size))
    print(df.shape)
    
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((train_data[i:i+block_size]).astype(np.int64)) for i in ix])
    target_index = ix[0].item()
    y = torch.stack([torch.from_numpy((train_data[i+block_size:i+block_size+ block_size]).astype(np.int64)) for i in ix])

    for i in range(batch_size):
        print(i)
        for j in range(batch_size):
            log_probs_matrix[i,j] = compute_log_probability(model, torch.tensor(x[i]).unsqueeze(0).to(device), torch.tensor(y[j]).unsqueeze(0).to(device))
    print(log_probs_matrix)
    plot_heat(log_probs_matrix)
    
    log_probs_matrix = np.zeros((20, 20))

    for k in range(1):

        for j in range(df.shape[0]):
            
            #target_text = extract_text_between_markers(df.iloc[j]['Text'])
            target_text = df.iloc[j]['Text']
            ids = enc.encode_ordinary(target_text)
            print(len(ids))
            print(enc.decode(ids[:10]))
            #target_text_enc = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)  
            #print(target_text_enc.size())   
            for i in range(0, df.shape[0]):
                author_prompt = f"{df.iloc[i]['Author']} <> "
                author_prompt = enc.encode(author_prompt)
                fill_text = df.iloc[i]['Text']
                fill_text = enc.encode_ordinary(fill_text)
                len_author = len(author_prompt)
                if len_author <32:
                    print(len_author)
                    author_prompt = author_prompt #+ fill_text[:32-len(author_prompt)]
                    #ids = ids[32-len_author:]
                else:
                    author_prompt = author_prompt[:32]
                    ids = ids
                print(enc.decode(author_prompt),"------", enc.decode(ids[:10]))
                target_text_enc = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                author_prompt = torch.tensor(author_prompt, dtype=torch.long).unsqueeze(0).to(device)
                probs = compute_log_probability(model, author_prompt, target_text_enc)
                log_probs_matrix[j, i] = probs
            print('----------')
        print(k)
    log_probs_matrix /= 1
    print(log_probs_matrix)
    

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(log_probs_matrix, annot=False, cmap='coolwarm', linewidths=0.5)

    # Highlight diagonal elements where i == j
    for i in range(df.shape[0]):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='yellow', lw=2))
    min_indices = np.argmin(log_probs_matrix, axis=0)  # Find the row index of the min value in each column
    # Find second smallest values
    sorted_indices = np.argsort(log_probs_matrix, axis=0)  # Sort indices column-wise
    second_min_indices = sorted_indices[0]  # Second smallest is at index 1 after sorting

    # Highlight second smallest values
    for j, i in enumerate(second_min_indices):
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='green', lw=2))

    plt.title("Log Probabilities Heatmap (Author Prompt vs Target Text)")
    plt.xlabel("Author Prompt Index (i)")
    plt.ylabel("Target Text Index (j)")
    plt.show()

print("--- %s seconds ---" % (time.time() - start_time))

    