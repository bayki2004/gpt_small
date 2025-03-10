from encoder_decoder import *
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

# Assume config is defined as per your ModelConfig
config = ModelConfig()  # defaults: block_size=1024, vocab_size=50304, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True
init_from = 'scratch'
config.block_size = 512
num_epochs = 20
batch_size = 8
start_token_id = 101
best_val_loss = 1e9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate the encoder-decoder model with your pretrained encoder (BERT)
if init_from == 'scratch':
    model = EncoderDecoderModel(config, t5_encoder, encoder_proj)
    for param in model.encoder.parameters():
        param.requires_grad = False

    model.train()
if init_from == 'resume':
    # Assume you saved your decoder-only model's state dict in 'decoder_state_dict.pth'

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
  
    model = EncoderDecoderModel(config, t5_encoder, encoder_proj)
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Get the current state dict of the new model.
    new_state = model.state_dict()

    # Update only the decoder parts with the mapped weights.
    new_state.update(mapped_decoder_state)

    # Load the updated state dict into the model.
    model.load_state_dict(new_state)
print('done')
print("Loaded pretrained decoder weights into the encoder-decoder model.")

# --- Initialize T5 tokenizer ---
tokenizer = T5Tokenizer.from_pretrained('t5-small')
# Use T5 encoder's decoder_start_token_id if available; otherwise fallback to pad_token_id.
decoder_start_token_id = t5_encoder.config.decoder_start_token_id
if decoder_start_token_id is None:
    decoder_start_token_id = tokenizer.pad_token_id

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

# --- Load and preprocess data ---
training_tuples = torch.load('data/train_tensors_enc.pt')
validation_tuples = torch.load('data/test_tensors_enc.pt')
train_dataset = PreprocessedDataset(training_tuples, decoder_start_token_id)
val_dataset = PreprocessedDataset(validation_tuples, decoder_start_token_id)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

model.to(device)
# Now you can continue training with the encoder frozen.
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
print('loaded device')
print(device)
# --- Training Loop with Validation Pass ---
print("Starting training loop...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        encoder_input_ids, decoder_input_ids, decoder_targets = batch
        encoder_input_ids = encoder_input_ids.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
        decoder_targets = decoder_targets.to(device)
        optimizer.zero_grad()
        # Assume your model returns (logits, loss) when given inputs
        logits, loss = model(encoder_input_ids, decoder_input_ids, decoder_targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
    
    # --- Validation Pass ---
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            encoder_input_ids, decoder_input_ids, decoder_targets = batch
            encoder_input_ids = encoder_input_ids.to(device)
            decoder_input_ids = decoder_input_ids.to(device)
            decoder_targets = decoder_targets.to(device)
            # Forward pass; no gradient computation
            logits, loss = model(encoder_input_ids, decoder_input_ids, decoder_targets)
            val_running_loss += loss.item()
    avg_val_loss = val_running_loss / len(val_loader)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_config": config
        }
        torch.save(checkpoint, "out/enc_dec_512_long.pt")
        print("Checkpoint saved.")
    print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
print("------------------")
print("Training complete.")
checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_config": config
        }
torch.save(checkpoint, "models/enc_dec_512_long.pt")
# --- Generation Sample (Optional) ---
# Use the first sample's encoder input to generate new text.
sample_encoder_input_ids, _, _ = train_dataset[0]
sample_encoder_input_ids = sample_encoder_input_ids.unsqueeze(0).to(device)  # add batch dimension
print("Input (Author):", tokenizer.decode(sample_encoder_input_ids.squeeze(), skip_special_tokens=True))
generated_ids = model.generate(
    sample_encoder_input_ids,
    torch.tensor([decoder_start_token_id]).to(device),
    max_new_tokens=20
)
output_text = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
print("Generated Text:", output_text)