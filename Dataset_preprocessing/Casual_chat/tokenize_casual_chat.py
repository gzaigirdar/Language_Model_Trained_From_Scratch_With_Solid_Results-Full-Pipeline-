# %%
from datasets import load_from_disk,concatenate_datasets

saved_path = '../../Saved_Data/Casual_chat/casual_chat_raw_dataset'

pairs_dataset = load_from_disk(saved_path)
print(pairs_dataset)
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
        'raw_text': input_samples

    }

# =================================================================
# %%
from transformers import AutoTokenizer
# load the saved tokenizer 
tokenizer_path = '../../Tokenizer/Saved_tokenizer/t5_Tokenizer'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,use_fast=False)

# %%
# apply the preprocess function and transfor the dataset

preprocessed_dataset = pairs_dataset.map(lambda batch: preprocess_func(batch,tokenizer,clean_text),batched=True)

# %%
for sample in preprocessed_dataset:
    print(sample['input_ids'])
    print(sample['raw_text'])
    break

# %%

preprocessed_dataset = preprocessed_dataset.remove_columns(['prompt','response'])
for sample in preprocessed_dataset:
    print(sample)
    break
# %%

tiny_chat_ds = load_from_disk('../../Saved_Data/Tiny_chat__dataset_tokenized')
tiny_chat_ds = tiny_chat_ds.shuffle(seed=42)
tiny_chat_ds = tiny_chat_ds.select(range(92613))
print(tiny_chat_ds)

# %%
print(preprocessed_dataset)
preprocessed_dataset = concatenate_datasets([preprocessed_dataset,tiny_chat_ds])
print(preprocessed_dataset)
# %%
print(preprocessed_dataset['raw_text'][11600])
# %%
processed_path = '../../Saved_Data/Casual_chat/Casual_chat_tokenized_Dataset'

preprocessed_dataset.save_to_disk(processed_path)

# %%
len(preprocessed_dataset)
# %%
