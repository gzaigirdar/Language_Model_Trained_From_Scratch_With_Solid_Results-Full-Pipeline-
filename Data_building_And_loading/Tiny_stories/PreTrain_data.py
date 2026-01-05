#%%
from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories",split="train[:500000]")
# %%
dataset

# %%

for sample in dataset:
    print(sample)
    break
# %%
from transformers import AutoTokenizer
tokenizer_path = '/home/gz/Documents/Full Pipeline(LLM)/Saved_tokenizer/t5_Tokinzer'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,use_fast=False)

# %%
print(len(tokenizer))


# %%
import re
def clean_text(text):
    text = text.lower()

    tokens = re.findall(r"\w+(?:'\w+)*|[^\w\s]", text)
    tokens = " ".join(tokens)

    return tokens

# %%
def reduce_length(text,max_len=100):
    text = text.split()
    text = text[:max_len]
    text = ' '.join(text)
    return text
# %%
def PreProcess(example,tokenizer,clean_text):
    example = example['text']
    # example = reduce_length(example,max_len=max_len)
    sample = f"{tokenizer.bos_token} {clean_text(example)} {tokenizer.eos_token}"
    input_ids = tokenizer(sample,add_special_tokens=False,return_attention_mask=False)['input_ids']

    return {
        'input_ids': input_ids,
        'raw_text' : sample
    }


# %%

ds = dataset.map(lambda example :PreProcess(example,tokenizer=tokenizer,clean_text=clean_text),batched=False,num_proc=4)

ds = ds.remove_columns(['text'])

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
for sample in ds:
    print(sample['raw_text'])
    print(sample['input_ids'])
    break

# %%
ds.save_to_disk('/home/gz/Documents/Full Pipeline(LLM)/Saved_Data/PretrainedProccessedData')
# %%
