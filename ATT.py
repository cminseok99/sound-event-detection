import numpy as np
from torch.utils.data import DataLoader
import torch
from torch import nn


class ATT(nn.Module):
    def __init__(self, n_in_channel, activation="Relu", conv_dropout=0,
                 kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)]
                 ):
        super(ATT,self).__init__()
        #들어올 입력은 628x128
        self.linear = nn.Linear(628,157)
        self.act = nn.Sigmoid()
        # self.linear1 = nn.Linear(628,157)
        # self.linear2 = nn.Linear(628,157)
        # self.linear3 = nn.Linear(628,157)
        # self.input = nn.Linear(628,157)    
        # self.query = nn.Linear(628,157)
        # self.key = nn.Linear(628,157)
        # self.val = nn.Linear(628,157) 

        # self.attn = nn.MultiheadAttention(embed_dim=157, num_heads=8)

            
    # def load_state_dict(self, state_dict, strict=True): 
    #     self.att.load_state_dict(state_dict) 

    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     return self.att.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
    
    def forward(self,x):
         # input size : (batch_size, n_channels, n_frames, n_freq
        # x = nn.functional.scaled_dot_product_attention(self.linear1(x) ,self.linear2(x) 8,self.linear3(x))
        # x = self.acti(x)
        # x,y = self.attn(self.input,self.query,self.key,self.val)
        # x = x * y
        x = self.linear(x)
        x = self.act(x)

        return x
    
  
    




     
