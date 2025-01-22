import pandas as pd

# Read the CSV in chunks to avoid loading everything into memory at once
chunk_size = 5000  # adjust this based on your memory
text_file_path = "books_text_xl.txt"

# Open the text file to write incrementally
with open(text_file_path, "w") as text_file:
    # Iterate over CSV in chunks
    for chunk in pd.read_csv('gutenberg_data_with_text.csv', chunksize=chunk_size):
        # Filter rows where 'Text' is not NaN and concatenate texts from each chunk
        valid_rows = chunk.dropna(subset=['Text'])
        text = " ".join(valid_rows['Text'])
        
        # Write to the file
        text_file.write(text)
        print("chunk done")
