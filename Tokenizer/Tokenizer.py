# %%
from transformers import AutoTokenizer 

# google t5's sentence piece tokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small",use_fast=False)

special_tokens =  {
    "pad_token": "<pad>",
    "eos_token": "<end>",
    "bos_token": "<start>",
    "additional_special_tokens":["<sep>","<user>","<bot>"]

    }
tokenizer.add_special_tokens(special_tokens)

# %%
save_path = 'Saved_tokenizer/t5_Tokenizer'
tokenizer.save_pretrained(save_path)
 
# %%
