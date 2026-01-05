import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.Final_layer = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)  # Dropout for attention probabilities

    def Attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
           scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)


        scores_prob = torch.softmax(scores, dim=-1)
        scores_prob = self.dropout(scores_prob)  # Dropout applied here

        output = torch.matmul(scores_prob, V)
        return output

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        # returns shaep (batch size, num_head,seq_len,self.dk
        return x.view(batch_size, seq_len, self.num_heads, self.dk).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, dk = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.WQ(Q))
        K = self.split_heads(self.WK(K))
        V = self.split_heads(self.WV(V))

        scores = self.Attention(Q, K, V, mask)

        output = self.Final_layer(self.combine_heads(scores))
        return output

