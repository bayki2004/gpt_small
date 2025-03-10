import pandas as pd
import torch
import numpy as np
import tiktoken
enc = tiktoken.encoding_for_model('gpt-4')
df = pd.read_csv('/itet-stor/kbayraktar/net_scratch/training_ml/data/dataset_full_pre_text.csv')
df = df.dropna(subset=['Author', 'Text'])
df = df.reset_index(drop=True)
df__ = df.drop_duplicates(subset=['Author'], keep='first')
author_mapping = df__.set_index(df__.index)['Author'].to_dict()
author_to_index = dict(zip(df__['Author'], df__.index))
eop_token = 100276

def tokenize_row(author, text):
    tokenized_index = enc.encode(str(author))
    tokenized_text = enc.encode(str(text))
    full_sequence = tokenized_index + [eop_token] + tokenized_text + [enc.eot_token]
    return torch.tensor(full_sequence, dtype=torch.long)

# Function to process the dataframe in chunks
def process_in_chunks(df, chunk_size=1000):
    tensor_list = []
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))  # Ensure we don't go out of bounds
        chunk = df.iloc[start:end]
        chunk_tensors = chunk.apply(lambda row: tokenize_row(row['Author'], row["Text"]), axis=1).tolist()
        tensor_list.extend(chunk_tensors)
        print(start)
    return tensor_list

# Process the dataframe in chunks
print('starting preprocessing')
tensor_list = process_in_chunks(df, chunk_size=1000)
print(f"Number of tensors: {len(tensor_list)}")
flattened_np = np.concatenate([tensor.numpy() for tensor in tensor_list])
offsets = np.cumsum([0] + [t.shape[0] for t in tensor_list[:-1]])
filename = "data/data_author_names.bin"
arr = np.memmap(filename, dtype=np.float32, mode='w+', shape=(len(flattened_np),))
arr[:] = flattened_np[:]
arr.flush()
np.save("data/author_names_offsets.npy", offsets)