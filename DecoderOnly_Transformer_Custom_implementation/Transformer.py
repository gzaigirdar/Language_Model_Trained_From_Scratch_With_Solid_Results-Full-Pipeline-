import torch
import torch.nn as nn
import math
import sys
sys.path.append('../DecoderOnly_Transformer_Custom_implementation')
from DecoderOnly_Transformer_Custom_implementation.TransformerBlock import TransformerBlock
from DecoderOnly_Transformer_Custom_implementation.RMSNorm import RMSNorm
from DecoderOnly_Transformer_Custom_implementation.PositionalEncoding import PositionalEncoding




class GModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(GModel, self).__init__()
        self.d_model = d_model

 
        self.token_embedding = nn.Embedding(vocab_size, d_model)
       
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

      
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

       
        self.final_norm = RMSNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size,bias=False)
        self.dropout = nn.Dropout(dropout)

        
        #  weight Initialization 
        self.apply(self._init_weights)
        self.fc.weight = self.token_embedding.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
   
 
    
    def create_mask(self,input_ids,pad_token):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        # shape of the inputs_ids are batch_size,seq_len
        pad_mask = (input_ids != pad_token) # ---> batch_size,seq_len
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2) #---> batch_size,1,1,seq_len
        casual_mask = torch.tril(torch.ones(seq_len,seq_len,device=device)).bool() # (seq_len,seq_len)
        full_mask = casual_mask * pad_mask
        return full_mask
        


  
    def forward(self, input_ids,pad_token):
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        attn_mask = self.create_mask(input_ids,pad_token)
        

        for block in self.transformer_blocks:
            x = block(x, attn_mask)

        x = self.final_norm(x)
        output = self.fc(x)
        return output
    

# model with out weight tieing 
class GModel2(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(GModel2, self).__init__()
        self.d_model = d_model

 
        self.token_embedding = nn.Embedding(vocab_size, d_model)
       
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

      
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

       
        self.final_norm = RMSNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size,bias=True)
        self.dropout = nn.Dropout(dropout)

        
        #  weight Initialization 
        self.apply(self._init_weights)
        

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
   
    
    
    def create_mask(self,input_ids,pad_token):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        # shape of the inputs_ids are batch_size,seq_len
        pad_mask = (input_ids != pad_token) # ---> batch_size,seq_len
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2) #---> batch_size,1,1,seq_len
        casual_mask = torch.tril(torch.ones(seq_len,seq_len,device=device)).bool() # (seq_len,seq_len)
        full_mask = casual_mask * pad_mask
        return full_mask
        


  
    def forward(self, input_ids,pad_token):
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        attn_mask = self.create_mask(input_ids,pad_token)
        

        for block in self.transformer_blocks:
            x = block(x, attn_mask)

        x = self.final_norm(x)
        output = self.fc(x)
        return output
    

print("GPTModel class defined.")