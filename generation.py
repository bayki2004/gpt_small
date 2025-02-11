import torch
from torch.nn import functional as F
from model import *
from configs import llm_3
import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
enc = tiktoken.get_encoding("gpt2")
block_size = llm_3['block_size']
# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('at use.py')
configs = ModelConfig(**llm_3)
# Load the trained model
model = LanguageModel(configs)

model.load_state_dict(torch.load("models/language_model_big_2.pth", map_location=torch.device('mps')),strict=False )
model.eval()

# Function to generate text from a prompt
def generate_text(prompt, max_new_tokens=50):
    # Encode the prompt and convert it to a tensor
    print('generating')
    context = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    print('context')
    generated_ids = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    print('generated_ids')
    generated_text = enc.decode(generated_ids)
    print('generated_text')
    # Decode the generated indices to text
    
    return generated_text

def compute_log_probability(model, x, y):
        log_prob = 0.0
        idx = x  # Start with the given prefix x
        y_len = y.size(1)
        
        for i in range(min(y_len, 5000)):
            idx_cond = idx[:, -block_size:]  # Truncate to block size
            logits, _ = model(idx_cond)  # Get logits
            logits = logits[:, -1, :]  # Get logits for the last token in the sequence
            probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
            y_next = y[:, i]  # The next token from the target sequence y
            token_prob = probs[:, y_next].item()  # Get the probability of generating y_next
            log_prob += math.log(token_prob + 1e-9)  # Add log probability (avoid log(0) with 1e-9)
            idx = torch.cat((idx, y_next.unsqueeze(1)), dim=1)  # Append y_next to the sequence
        
        return log_prob
# Example usage

if __name__ == "__main__":
    i = 0
    df = pd.read_csv('data/dataframe_testing.csv').dropna(subset=['Author', 'Text'])
    log_probs_matrix = np.zeros((df.shape[0], df.shape[0]))
    print(df.shape)
    for j in range(df.shape[0]):
        target_text = df['Text'][i]
        target_text = torch.tensor(enc.encode(target_text), dtype=torch.long).unsqueeze(0).to(device)
        for i in range(df.shape[0]):
            author_prompt = df.iloc[i]['Author'] + '\n'  + ' '
            author_prompt = torch.tensor(enc.encode(author_prompt), dtype=torch.long).unsqueeze(0).to(device)
            probability = (compute_log_probability(model, author_prompt, target_text))
            print(probability)
            log_probs_matrix[j, i] = probability
    
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(log_probs_matrix, annot=False, cmap='coolwarm', linewidths=0.5)

    # Highlight diagonal elements where i == j
    for i in range(df.shape[0]):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='yellow', lw=2))

    plt.title("Log Probabilities Heatmap (Author Prompt vs Target Text)")
    plt.xlabel("Author Prompt Index (i)")
    plt.ylabel("Target Text Index (j)")
    plt.show()
            

    
    