import torch
'''x = torch.tensor( [[1,2,3,0,0],


     [1,2,3,4,0],
     [1,2,3,0,0],
     [1,2,3,4,5],
     [0,0,3,4,5]])


# batch_size,seq_len (5,5)
print(f'shape of x is {x.shape}')

#x = x.unsqueeze(1) 
# shape (5,1,5)

casual_mask = torch.tril(torch.ones(5,5))
# shape (1,1,1,5,5)
print(casual_mask)
pad_mask = (x != 0)
# shape (5,1,1,1,5)
print(pad_mask.shape)
print(pad_mask)
combined_mask = casual_mask * pad_mask
print(combined_mask)
'''

batch = 1
num_tokens = 2
word_dim = 4
num_heads = 2 
dk = 2
print('origina query:')
query = torch.randn(num_tokens,word_dim)
print(query)
print('after adding num_heada and dk')
query = query.view(num_tokens,num_heads,dk)
print(f'before tranpose {query}')
print(f'after transposing:')
print(query.transpose(1,2))
''' query = query.reshape(1,2)
print(query)'''