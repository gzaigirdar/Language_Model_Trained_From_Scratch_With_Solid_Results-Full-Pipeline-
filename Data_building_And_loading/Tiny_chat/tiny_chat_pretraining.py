#%%
from datasets import load_dataset

dataset = load_dataset("starhopp3r/TinyChat")
# %%
dataset

# %%

import re

specials = {"<user>", "<bot>"}

def clean_text(text):
    # replace role markers first
    text = text.replace("[INST]", "<user>").replace("[/INST]", "<bot>")
    text = text.lower()
    # match <user>/<bot> first, then words, then punctuation
    tokens = re.findall(r"<user>|<bot>|\w+(?:'\w+)*|[?.!,;]", text)
    return " ".join(tokens)


def preprocess_func(batch, tokenizer):
    texts = []
    for text in batch["text"]:
        cleaned = clean_text(text)
        texts.append(f"{tokenizer.bos_token} {cleaned} {tokenizer.eos_token}")

    tokens = tokenizer(texts, padding=False, truncation=False, return_attention_mask=False,add_special_tokens=False)

   
    input_ids = [list(ids) for ids in tokens["input_ids"]]

    return {
        "input_ids": input_ids,
        "raw_text": texts
    }


# =================================================================
# %%
from transformers import AutoTokenizer
# load the saved tokenizer 
tokenizer_path = '/home/gz/Documents/Full Pipeline(LLM)/Saved_tokenizer/t5_Tokinzer'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,use_fast=True)
# %%
ds = dataset["train"].map(
    lambda batch: preprocess_func(batch, tokenizer),
    batched=True,
    batch_size=800,
    num_proc=6,
    remove_columns=dataset["train"].column_names,
)



# %%
for sample in ds:
    print(sample['raw_text'])
    print(sample['input_ids'])
    break
# %%
dataset = dataset['train']
for sample in dataset['text']:
    demo_text = sample
    break
demo_text = f"{tokenizer.bos_token} {clean_text(demo_text)} {tokenizer.eos_token}"
# %%
print(demo_text)

# %%
lengths = [len(row['input_ids']) for row in ds]

# Compute stats
avg_len = sum(lengths) / len(lengths)
min_len = min(lengths)
max_len = max(lengths)

print(f"Number of sequences: {len(dataset)}")
print(f"Average tokens per sequence: {avg_len:.1f}")
print(f"Minimum tokens: {min_len}")
print(f"Maximum tokens: {max_len}")


# %%
print(tokenizer.decode(1))
# %%

ds.save_to_disk('/home/gz/Documents/Full Pipeline(LLM)/Saved_Data/Tiny_chat_pretrained_dataset')
# %%
