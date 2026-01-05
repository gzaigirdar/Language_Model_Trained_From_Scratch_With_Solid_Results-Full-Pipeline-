# %%
from datasets import load_from_disk

saved_path = '/home/gz/Documents/Full Pipeline(LLM)/Saved_Data/pairs_dataset'

pairs_dataset = load_from_disk(saved_path)

for sample in pairs_dataset:
    print(sample['prompt'])
    print(sample['response'])
    break
#===============================================================================


# %%
# clean function to lower text and seperate punc
import re

def clean_text(text):
    text = text.lower()

    tokens = re.findall(r"\w+(?:'\w+)*|[^\w\s]", text)
    tokens = " ".join(tokens)

    return tokens
print(clean_text('Hello World!.'))
# %%
def preprocess_func(batch,tokenizer,clean_text):

    
    input_samples = [f"{tokenizer.bos_token} <user> {clean_text(prompt)} {tokenizer.eos_token} <bot> {clean_text(response)} {tokenizer.eos_token}"
                     for prompt,response in zip(batch['prompt'],batch['response'])
                     ]
    
    tokens = tokenizer(input_samples, padding=False, truncation=False, return_attention_mask=False,add_special_tokens=False)
    


    return{
        'input_ids': tokens['input_ids'],
        'text_sample': input_samples

    }

# =================================================================
# %%
from transformers import AutoTokenizer
# load the saved tokenizer 
tokenizer_path = '/home/gz/Documents/Full Pipeline(LLM)/Saved_tokenizer/t5_Tokinzer'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,use_fast=False)

# %%
# apply the preprocess function and transfor the dataset

preprocessed_dataset = pairs_dataset.map(lambda batch: preprocess_func(batch,tokenizer,clean_text),batched=True)

# %%
for sample in preprocessed_dataset:
    print(sample['input_ids'])
    print(sample['text_sample'])
    break

# %%

preprocessed_dataset = preprocessed_dataset.remove_columns(['prompt','response'])
for sample in preprocessed_dataset:
    print(sample)
    break

# %%
processed_path = '/home/gz/Documents/Full Pipeline(LLM)/Saved_Data/processedDataset'

preprocessed_dataset.save_to_disk(processed_path)

# %%
len(preprocessed_dataset)
# %%
