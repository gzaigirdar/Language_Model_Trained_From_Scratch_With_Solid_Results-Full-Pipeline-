import torch
import torch.nn as nn

from DecoderOnly_Transformer_Custom_implementation.MultiHeadAttention import MultiHeadAttention
from DecoderOnly_Transformer_Custom_implementation.RMSNorm import RMSNorm
from DecoderOnly_Transformer_Custom_implementation.PointwiseFF import PointwiseFeedForward

class TransformerBlock(nn.Module): 
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TransformerBlock, self).__init__() 
        self.Attention = MultiHeadAttention(d_model, num_heads)
        self.FeedForward = PointwiseFeedForward(d_model, d_ff)
        self.RMSNorm1 = RMSNorm(d_model) 
       
        self.RMSNorm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

 
    def forward(self, x, target_mask):
        norm_x_for_attn = self.RMSNorm1(x)
        attn_output = self.Attention(norm_x_for_attn, norm_x_for_attn, norm_x_for_attn, target_mask)
        x = x + self.dropout(attn_output)

      
        norm_x_for_ffn = self.RMSNorm2(x) 
        ff_output = self.FeedForward(norm_x_for_ffn)
        x = x + self.dropout(ff_output)
        
        return x