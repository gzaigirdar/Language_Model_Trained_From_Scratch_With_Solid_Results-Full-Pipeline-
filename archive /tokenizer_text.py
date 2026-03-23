# %%
from transformers import AutoTokenizer
# %%
tokenizer_path = 'Tokenizer/Saved_tokenizer/t5_Tokenizer'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,use_fast=False)


# %%
tokenizer.