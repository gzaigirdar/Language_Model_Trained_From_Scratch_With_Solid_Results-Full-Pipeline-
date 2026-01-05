from Training_utils.buildModel import BuildModel
from transformers import AutoTokenizer
import torch
import re
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--Top_K",type=int,default=0)
parser.add_argument("--Temp",type=float,default=0.0)
args = parser.parse_args()
model_config = {
        "D_Model": 420,
        "Num_Heads": 6,
        "Num_Layers": 6,
        "Dropout": 0.001,
        "Vocab_size": 32105,
        "FeedForward_size": 2000,
        "Context_size": 80
    }
tokenizer_path = '../Final_version_SLM/Saved_tokenizer/t5_Tokinzer'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,use_fast=True)
def clean_text(text):
    text = text.lower()

    tokens = re.findall(r"\w+(?:'\w+)*|[^\w\s]", text)
    
    tokens = " ".join(tokens)
    return tokens



model_path ="Saved_Models/Full_trained_Model/80t_41m_b2_wo_pre.pth"
device = ('cuda' if torch.cuda.is_available() else 'cpu')
builder = BuildModel()
model = builder.createModel(model_config,Model_type='None')
model.to(device)

builder.load_weights(path=model_path)



def gen_text(prompt,model,tokenizer,max_tokens=80,pad_token=0,temperature=0.0,top_k=0,device=device):
    prompt = clean_text(prompt)
    prompt = f"{tokenizer.bos_token} <user> {prompt} {tokenizer.eos_token} <bot> "
    tokenized_text = tokenizer(prompt,return_tensors='pt',add_special_tokens=False,return_attention_mask=False,padding=False,truncation=False,)
    input_ids = tokenized_text['input_ids']
    pad_index = tokenizer.pad_token
    generated_tokens = []
    input_ids = input_ids.to(device)
    model.eval()
   
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(input_ids,pad_token=pad_token)
            logits = logits[:,-1,:]
           
            if temperature > 0.0:
               logits= logits/temperature
            
            if top_k != 0:
                top_logits,top_pos = torch.topk(logits,top_k)
                min_val = top_logits[:,-1]
               
                logits= torch.where(logits<min_val,
                                          torch.tensor(float('-inf'),device=device),
                                          logits
                                          ) 
                
            if top_k != 0 or temperature > 0.0:
                probs = torch.softmax(logits,dim=-1)
                pred_index = torch.multinomial(probs,num_samples=1)
                pred_token = tokenizer.convert_ids_to_tokens(pred_index.tolist()[0])
                if pred_token[0] == '<end>':
                    break
                generated_tokens.append(pred_index.squeeze(0).tolist()[0])
            else:
                pred_index = torch.argmax(logits,dim=-1)
                pred_token = tokenizer.convert_ids_to_tokens(pred_index.tolist())
                if pred_token[0] == '<end>':
                    break
                generated_tokens.append(pred_index.tolist()[0])
                pred_index = pred_index.unsqueeze(0)
            
            input_ids = torch.cat([input_ids,pred_index],dim=-1)
    
    text = tokenizer.decode(generated_tokens)

    for word in text.split():
        print(word, end=" ", flush=True)  
        time.sleep(0.07)  
    



print('Hello! please enter a prompt to Chat')
print('enter ## to exit the chat.')
while True:
    
    prompt = input('\nMe: \n')
    if prompt == '##':
        print('Bye!')
        break
    else:
        print('AI:')
        gen_text(prompt,model,tokenizer,top_k=args.Top_K,temperature=args.Temp)
    