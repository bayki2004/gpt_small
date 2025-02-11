import pandas as pd
import numpy as np
import tiktoken
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Load and prepare the dataset
df = pd.read_csv('gutenberg_data_with_text.csv').dropna(subset=['Author', 'Text'])
train_df, val_df = train_test_split(df, test_size=0.0005, random_state=2357, shuffle=True)

# Initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")

# Generator function for processing rows
def process_generator(dataframe):
    for _, example in dataframe.iterrows():
        combined_text = f"{example['Author']} '\n' {example['Text']}"
        ids = enc.encode_ordinary(combined_text)
        ids.append(enc.eot_token)  # Add end-of-text token
        yield {'ids': ids, 'len': len(ids)}

# Function to write tokenized data to a binary file using a generator
def write_to_bin_file(dataframe, filename, batch_size=1000):
    dtype = np.uint16  # Use np.uint16 since token values are < 65536
    total_len = sum(len(item['ids']) for item in process_generator(dataframe))
    
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(total_len,))
    idx = 0

    batch = []
    for processed in tqdm(process_generator(dataframe), desc=f'Writing {filename}'):
        batch.append(processed['ids'])
        if len(batch) >= batch_size:
            batch_arr = np.concatenate(batch)
            arr[idx:idx + len(batch_arr)] = batch_arr
            idx += len(batch_arr)
            batch = []  # Clear batch

    if batch:  # Write any remaining data
        batch_arr = np.concatenate(batch)
        arr[idx:idx + len(batch_arr)] = batch_arr
    arr.flush()

# Write the train and validation splits to binary files
write_to_bin_file(train_df, "train.bin")
write_to_bin_file(val_df, "val.bin")

