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
max_prompt_length = 0
def extract_text_between_markers(text):
    start_idx = text.find(delimiter)
    end_idx = text.find(end_of_text)

    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        extracted_text = text[start_idx+ len(delimiter): end_idx-10000].strip()
        #print(len(extracted_text), len(enc.encode_ordinary(extracted_text)))
        return extracted_text
    return None  # Return None if markers are missing

authors = set()
i, j = 0, 0
# Generator function for processing rows within a chunk
def process_generator(dataframe):
    global authors, i, j
    for _, example in dataframe.iterrows():
        text = extract_text_between_markers(example['Text'])
        combined_text = f"{example['Author']} <> {text}"
        if authors.intersection({example['Author']}):
            print(example['Author'])
            i = 1 +i
            print(i)
            continue
        j = 1 +j
        
        authors.add(example['Author'])

        author_encoded = enc.encode(example['Author'])
        
        combined_text = enc.encode(f" <> {text}")
        combined_text = author_encoded + combined_text
        ids = (combined_text[:len(author_encoded)+20])
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
    write_to_bin_file_in_chunks("data/gutenberg_data_with_text.csv", "train_small_2.bin")
    write_to_bin_file_in_chunks("data/gutenberg_data_with_text.csv", "val_small_2.bin")
    print(max_prompt_length)
    print(i, j)