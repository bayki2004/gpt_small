llm_3 = {
    "block_size": 256,  # Increase block size for better context
    "n_embd": 768,
    "n_head": 8,
    "n_layer": 4,
    "dropout": 0.0,
    "bias": False,  
    "vocab_size": 50257  # GPT2 vocab size
}
llm_2 = {
    "batch_size": 32,
    "block_size": 256,  # Increase block size for better context
    "max_iters": 10000,
    "eval_interval": 300,
    "learning_rate": 1e-3,
    "n_embd": 512,
    "n_head": 8,
    "n_layer": 4,
    "dropout": 0.0,
    "bias": False,
    "vocab_size": 50257  # GPT2 vocab size
}
llm_1 = {
    "batch_size": 32,
    "block_size": 32,
    "max_iters": 300,
    "eval_iters": 100,
    "eval_interval": 100,
    "learning_rate": 1e-4,
    "n_embd": 64,
    "n_head": 8,
    "n_layer": 4,
    "dropout": 0.2
    
}