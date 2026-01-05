import sys
import os
import torch
from pathlib import Path
import json

# finds the current folder path and  goes up a level
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# adds the path so python can search it
sys.path.append(parent_folder)

from DecoderOnly_Transformer_Custom_implementation.Transformer import GModel,GModel2

class BuildModel:
    def __init__(self,model_type='GModel'):
        self.D_Model = None
        self.Num_Heads = None
        self.Num_Layers = None
        self.Context_size = None
        self.Dropout = None 
        self.FeedForward_size = None 
        self.Vocab_size = None
        self.Model = None
        self.Num_of_params = None
        self.Parent_folder = None
        self.Model_Type = None
    
        


    
    def createModel(self,config,Model_type,ff_multi_factor=None):
        self.D_Model = config['D_Model']
        self.Num_Heads = config['Num_Heads']
        self.Num_Layers = config['Num_Layers'] 
        self.Dropout = config['Dropout']
        self.Vocab_size = config['Vocab_size']
        self.Model_Type = Model_type
        if ff_multi_factor is None:
            self.FeedForward_size = config['FeedForward_size'] 
        else:
            self.FeedForward_size = config['FeedForward_size'] * ff_multi_factor
        self.Context_size = config['Context_size']
        if self.Model_Type == 'GModel':
            self.Model = GModel(
                vocab_size=self.Vocab_size,
                d_model=self.D_Model,
                num_heads=self.Num_Heads,
                num_layers=self.Num_Layers,
                d_ff=self.FeedForward_size,
                max_seq_length=self.Context_size,
                dropout=self.Dropout
            )
        else:
            self.Model = GModel2(
                vocab_size=self.Vocab_size,
                d_model=self.D_Model,
                num_heads=self.Num_Heads,
                num_layers=self.Num_Layers,
                d_ff=self.FeedForward_size,
                max_seq_length=self.Context_size,
                dropout=self.Dropout
            )
            
        self.Num_of_params = sum(p.numel() for p in self.Model.parameters() if p.requires_grad)
        #print(f'Model initialized: {self.Num_Layers} layers, {self.Num_Heads} heads, d_model={self.D_Model}, d_ff={self.FeedForward_size}')

        return self.Model
    def get_total_params(self,in_millons=False):
        if in_millons:
            # 1e6 --> 1 millon
            return self.Num_of_params/1e6
    
        return self.Num_of_params
    def Save_model(self,model_name,folder_name):
        
        save_folder = Path(f"../Full Pipeline(LLM)/Saved_Models/{folder_name}")
        save_folder.mkdir(parents=True, exist_ok=True)

        torch.save(self.Model.state_dict(),save_folder/f'{model_name}.pth')

        print(f'model has been saved in folder {save_folder}')
        print(f'Model weights saved at: {save_folder}/{model_name}.pth')


    
        
     
    
    def get_config_dict(self):
        config = {
            "D_Model": self.D_Model,
            "Num_Heads": self.Num_Heads,
            "Num_Layers": self.Num_Layers,
            "Context_size": self.Context_size,
            "Dropout": self.Dropout,
            "FeedForward_size": self.FeedForward_size,
            "Vocab_size": self.Vocab_size
        }
        return config
    
    def save_config(self,folder_name,file_name,model_config,train_config):
        

        config_path = f"../Full Pipeline(LLM)/Saved_Models/{folder_name}/{file_name}.json"

        all_configs = {
            "model_config": model_config,
            "train_config": train_config
        }

        with open(config_path, 'w') as f:
            json.dump(all_configs, f, indent=4)

    def load_weights(self,path):
      
        self.Model.load_state_dict(torch.load(path, weights_only=True))
        print('model weights has been loaded')
   
    


    def set_parent_folder_path(self,path):
        self.Parent_folder = path
        return self.Parent_folder

