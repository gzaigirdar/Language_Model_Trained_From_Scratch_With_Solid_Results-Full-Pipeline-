# %%
from datasets import load_dataset,DatasetDict

# pairs holds all the input and output pairs from every datasets
pairs = []


# %%
''' daily dialog dataset loading'''
dataset = load_dataset("roskoN/dailydialog")
dataset


# %%
# go through train,val and test and extract pairs for it

# Train
train_data = dataset['train']
test_data = dataset['test']
val_data = dataset['validation']

# functionw will extract all the pairs from each train,test,validation set in daily dialog ds.
def extract_pairs_daily_dialog(data_set):
    for dialogs in data_set['utterances']:
        for i in range(0,len(dialogs)-1):
            input= dialogs[i]
            output = dialogs[i+1]
            pairs.append([input,output])
extract_pairs_daily_dialog(train_data)
extract_pairs_daily_dialog(val_data)
extract_pairs_daily_dialog(test_data)



# %%
print(len(pairs))
pairs[:10]

# %%
# loading cosmos ai mixed conversation dataset and extracting pairs
cosmosai_ds = load_dataset('cosmosai471/General_Conversation_Mixed_Dataset')
cosmosai_ds


# %%
# replace name of the ai from luna for gchat using regex
import re
replace = lambda text: re.sub(r"luna","Gchat",text,flags=re.IGNORECASE)

for sample in cosmosai_ds['train']:
    
    prompt = replace(sample['text'])
    response = replace(sample['response'])
    pairs.append([prompt,response])
   

# %%
print(len(pairs))
# %%
# now loading kaggle pairs data from local text files
'''___________________________________________'''


# %%
# load kaggle text data and extract pairs
print(len(pairs))
kaggle_data_path = '/home/gz/Documents/Full Pipeline(LLM)/Saved_Data/Kaggle_dialogs_data.txt'

with open(kaggle_data_path,'r') as f:
    lines = f.readlines()
    for line in lines:
        prompt,output= line.strip().split('\t')
        pairs.append([prompt,output])


#---------------------------------------------------------

# %%
len(pairs)
#--------------------------------------------


# %%
# load hard coded samples and extract pairs
import pickle
samples_path = '/home/gz/Documents/Full Pipeline(LLM)/Saved_Data/harcoded_samples.pkl'
with open(samples_path,'rb') as f:
    samples = pickle.load(f)
    for sample in samples:
        for index in range(0,len(sample)-1):
            prompt = sample[index]
            response = sample [index+1]
            pairs.append([prompt,response])
            
# --------------------------------------
# %%
len(pairs)
#-----------------------------------------
# %%
everyday_convo = load_dataset('HuggingFaceTB/everyday-conversations-llama3.1-2k')
everyday_convo = everyday_convo['train_sft']

# %%

for sample in everyday_convo:
    text = sample['completion']
    lines = [line.strip() for line in text.split("\n") if line.strip()]
   
    for i in range(0, len(lines)-1, 2):
        prompt = lines[i]
        response = lines[i+1]
        if prompt.startswith("User: "):
            prompt = prompt[len("User: "):]
        if response.startswith("AI: "):
            response = response[len("AI: "):]
        pairs.append([prompt,response])
        



# %%
# create dataset dict using the pairs list, each row has prompt and response
from datasets import Dataset
print(len(pairs))
# list of dictionaries 
dict_list = [{"prompt":row[0],"response":row[1]} for row in pairs]
pairs_dataset = Dataset.from_list(dict_list)
#---------------------------------------------

# %%
for sample in pairs_dataset:
    print(sample)
    break
#----------------------------------------------------------------------
# %% 
# save the parirs dataset 
saved_path = '/home/gz/Documents/Full Pipeline(LLM)/Saved_Data/pairs_dataset'
pairs_dataset.save_to_disk(saved_path)
# %%
