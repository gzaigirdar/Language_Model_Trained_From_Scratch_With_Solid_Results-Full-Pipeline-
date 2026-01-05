from Transformer import GModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = GModel(
                vocab_size=20,      # Requires 'tokenizer' from data setup
                d_model=20,
                num_heads=2,
                num_layers=1,
                d_ff=1,
                max_seq_length=5,      # Requires 'BLOCK_SIZE' from data setup
                dropout=0.05
)
model.to(device) # Move model to GPU if available

print(f"\nModel initialized successfully with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
print(f"Model on device: {next(model.parameters()).device}")

x = torch.tensor(
   [[1,2,3,5,-1],
     [5,6,7,-1,-1]]
)
attention_mask = torch.tensor(
    [[1,1,1,1,0],
    [1,1,1,0,0]]
)
mask = model.create_mask(x,pad_token=-1)
print(mask)
z = (x != -1)
#print(z)
#print(model.forward(x,attention_mask))