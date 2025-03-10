import pandas as pd
import numpy as np
import tiktoken
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")
delimiter = "*** START OF THE PROJECT GUTENBERG EBOOK"
chunk_size = 1000  # Number of rows to read in each chunk
end_of_text = "*** END OF THE PROJECT GUTENBERG EBOOK"
max_text_length = 64
small_samples = True
only_text = False
def extract_text_between_markers(text):
    start_idx = text.find(delimiter)
    end_idx = text.find(end_of_text)

    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        extracted_text = text[start_idx+ len(delimiter): end_idx].strip()
        #print(len(extracted_text), len(enc.encode_ordinary(extracted_text)))
        return extracted_text
    return text  # Return None if markers are missing


# Generator function for processing rows within a chunk
def process_generator(dataframe):
    for _, example in dataframe.iterrows():
        author_str = example['Author']
        author_encoded = enc.encode(author_str)
        text_str = example['Text']
        text_encoded = enc.encode(text_str)
        if small_samples:
            text_encoded = text_encoded[:max_text_length]
        combined_text = author_encoded + text_encoded
        if only_text:
            combined_text = text_encoded
        ids = combined_text
        ids.append(enc.eot_token)
        yield ids


# Function to write tokenized data from a chunk to a binary file
def write_to_bin_file_in_chunks(csv_file, filename, batch_size=1000):
    dtype = np.uint16
    all_ids = []

    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        chunk = chunk.dropna(subset=['Author', 'Text'])
        train_chunk, val_chunk = train_test_split(chunk, test_size=0.0005, random_state=2357, shuffle=True)

        for dataframe, split_name in [(train_chunk, "train"), (val_chunk, "val")]:
            if split_name not in filename:
                continue
            for ids in tqdm(process_generator(dataframe), desc=f'Processing {split_name} chunk'):
                all_ids.extend(ids)  # Accumulate all tokenized data
            print(f"processed {split_name}")
    # Convert the accumulated list to a NumPy array and save it using np.memmap
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(len(all_ids),))
    arr[:] = all_ids
    arr.flush()

if __name__ == "__main__":
# Run the preprocessing and write to binary files
    write_to_bin_file_in_chunks("data/data_no_dup.csv", "data/train_no_dup_small.bin")
    write_to_bin_file_in_chunks("data/data_no_dup.csv", "data/val_no_dup_small.bin")