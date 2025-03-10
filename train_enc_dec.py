from encoder_decoder import *
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

# Assume config is defined as per your ModelConfig
config = ModelConfig()  # defaults: block_size=1024, vocab_size=50304, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True

# Instantiate the encoder-decoder model with your pretrained encoder (BERT)
model = EncoderDecoderModel(config, t5_encoder, encoder_proj)
for param in model.encoder.parameters():
    param.requires_grad = False

model.train()

# Create a simple optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)


num_epochs = 3
batch_size = 3
encoder_seq_len = 10   # length of source sequences
decoder_seq_len = 8 
start_token_id = 101


# Assume you saved your decoder-only model's state dict in 'decoder_state_dict.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
saved_decoder_state = torch.load("models/lm_32_scratch_indices_end.pth", map_location=device)

# Inspect the keys (for example purposes)
print("Saved state keys:", saved_decoder_state.keys())

# Create a new dictionary to hold the mapped weights for the new model.
mapped_decoder_state = {}

# Suppose your old decoder model had these keys:
#   "token_embedding_table", "position_embedding_table", "blocks.0.ln1.weight", "blocks.0.ln1.bias", etc.
# And your new model expects:
#   "decoder_token_embedding.weight", "decoder_position_embedding.weight", "decoder_blocks.0.ln1.weight", etc.
saved_decoder_state = saved_decoder_state['model']
for k, v in saved_decoder_state.items():
    # Map token embedding table:
    print(k,v.shape)
    if k.startswith("token_embedding_table"):
        new_key = k.replace("token_embedding_table", "decoder_token_embedding")
        mapped_decoder_state[new_key] = v
    # Map position embedding table:
    elif k.startswith("position_embedding_table"):
        new_key = k.replace("position_embedding_table", "decoder_position_embedding")
        mapped_decoder_state[new_key] = v
    # Map the transformer blocks:
    elif k.startswith("blocks"):
        # For example, "blocks.0.ln1.weight" becomes "decoder_blocks.0.ln1.weight"
        new_key = "decoder_blocks" + k[len("blocks"):]
        mapped_decoder_state[new_key] = v
    # Map the final layer norm and LM head directly if names are unchanged:
    else:
        # e.g. "ln_f" and "lm_head" may be the same.
        mapped_decoder_state[k] = v

# Now, instantiate your new encoder–decoder model.
print('done')
model = EncoderDecoderModel(config, t5_encoder, encoder_proj)

# Freeze the encoder parameters, if not already frozen.
for param in model.encoder.parameters():
    param.requires_grad = False

model.to(device)
model.train()

# Get the current state dict of the new model.
new_state = model.state_dict()

# Update only the decoder parts with the mapped weights.
new_state.update(mapped_decoder_state)

# Load the updated state dict into the model.
model.load_state_dict(new_state)

print("Loaded pretrained decoder weights into the encoder-decoder model.")

# Now you can continue training with the encoder frozen.
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)



# --- Sample DataFrame ---
df = pd.read_csv('data/sample.csv')

# --- Set sequence lengths ---
encoder_seq_len = 10
decoder_seq_len = 12

# --- Initialize T5 tokenizer ---
tokenizer = T5Tokenizer.from_pretrained('t5-small')
# Use T5 encoder's decoder_start_token_id if available; otherwise fallback to pad_token_id.
decoder_start_token_id = t5_encoder.config.decoder_start_token_id
if decoder_start_token_id is None:
    decoder_start_token_id = tokenizer.pad_token_id

# --- Create a custom dataset ---
class AuthorTextDataset(Dataset):
    def __init__(self, df, tokenizer, encoder_seq_len, decoder_seq_len, decoder_start_token_id):
        self.df = df
        self.tokenizer = tokenizer
        self.encoder_seq_len = encoder_seq_len
        self.decoder_seq_len = decoder_seq_len
        self.decoder_start_token_id = decoder_start_token_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        author = row['Author']
        text = row['Text']
        # Tokenize encoder input (Author)
        encoder_enc = self.tokenizer(
            author,
            padding='max_length',
            max_length=self.encoder_seq_len,
            truncation=True,
            return_tensors='pt'
        )
        # Tokenize decoder target (Text)
        decoder_enc = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.decoder_seq_len,
            truncation=True,
            return_tensors='pt'
        )
        encoder_input_ids = encoder_enc['input_ids'].squeeze(0)  # shape: (encoder_seq_len,)
        full_decoder_targets = decoder_enc['input_ids'].squeeze(0)  # shape: (decoder_seq_len,)

        # Create shifted decoder inputs: prepend the start token and drop the last token.
        decoder_input_ids = torch.cat(
            [torch.tensor([self.decoder_start_token_id]), full_decoder_targets[:-1]],
            dim=0
        )
        return encoder_input_ids, decoder_input_ids, full_decoder_targets


class PreprocessedDataset(Dataset):
    def __init__(self, data_tuples, decoder_start_token_id):
        self.data = data_tuples
        self.decoder_start_token_id = decoder_start_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Unpack the tuple: encoder tensor and full target sentence tensor.
        encoder_input_ids, full_decoder_targets = self.data[idx]
        # Create shifted decoder input: prepend start token and remove the last token.
        decoder_input_ids = torch.cat(
            [torch.tensor([self.decoder_start_token_id], dtype=torch.long), full_decoder_targets[:-1]]
        )
        return encoder_input_ids, decoder_input_ids, full_decoder_targets

# Create dataset and dataloader (batch_size can be adjusted)
dataset = AuthorTextDataset(df, tokenizer, encoder_seq_len, decoder_seq_len, decoder_start_token_id)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
train_dataset = PreprocessedDataset(training_tuples, decoder_start_token_id)
val_dataset = PreprocessedDataset(validation_tuples, decoder_start_token_id)

batch_size = 2  # adjust as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# --- Training Setup ---
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'mps'
model.to(device)
print(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- Training Loop ---
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch in dataloader:
        encoder_input_ids, decoder_input_ids, decoder_targets = batch
        # Move tensors to device
        encoder_input_ids = encoder_input_ids.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
        decoder_targets = decoder_targets.to(device)
        print(encoder_input_ids.shape, decoder_input_ids.shape, decoder_targets.shape)
        print(tokenizer.decode(encoder_input_ids[0].squeeze().to(device), skip_special_tokens=True))
        print(tokenizer.decode(decoder_input_ids[0].squeeze().to(device), skip_special_tokens=True)) 
        optimizer.zero_grad()
        logits, loss = model(encoder_input_ids, decoder_input_ids, decoder_targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

# --- Generation Sample ---
# Use the first sample's encoder input to generate new text.
sample_encoder_input_ids, _, _ = dataset[0]
sample_encoder_input_ids = sample_encoder_input_ids.unsqueeze(0).to(device)  # add batch dimension
print(sample_encoder_input_ids.shape)  # torch.Size([1, 10])
print(tokenizer.decode(sample_encoder_input_ids.squeeze().to(device), skip_special_tokens=True))
generated_ids = model.generate(
    sample_encoder_input_ids,
    torch.tensor([decoder_start_token_id]).to(device),
    max_new_tokens=20
)

# Decode generated IDs back to text.
output_text = tokenizer.decode(generated_ids.squeeze().to(device), skip_special_tokens=True)
print("Input (Author):", tokenizer.decode(sample_encoder_input_ids.squeeze().to(device), skip_special_tokens=True))
print("Generated Text:", output_text)
