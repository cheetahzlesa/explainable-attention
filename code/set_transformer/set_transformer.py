from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn
from modules import *

class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input = 9,
        num_outputs=1,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=False,
    ):
        super(SetTransformer, self).__init__()
        
        self.dim_input = dim_input
        self.enc = nn.Sequential(
            SAB(self.dim_input, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads,  ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, 1),
        )
    

    def forward(self, X):
        return self.dec(self.enc(X)).view(X.size(0))




