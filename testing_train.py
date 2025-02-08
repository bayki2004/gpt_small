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
print(device)