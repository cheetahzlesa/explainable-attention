from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn


class TabTransformer(nn.Module):
    def __init__(
        self,
        num_cat: int,
        num_cont: int,
        d_model: int = 32,
        col_id_dim: int = 8,          
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        mlp_hidden_mult: tuple[int, int] = (4, 2), 
    ):
        super().__init__()
        assert 0 < col_id_dim < d_model
        self.num_cat = num_cat
        self.num_cont = num_cont
        self.d_model = d_model
        self.col_id_dim = col_id_dim
        self.val_dim = d_model - col_id_dim

        self.cardinalities = [2] * num_cat
        offsets = np.cumsum([0] + self.cardinalities[:-1]).astype(np.int64)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        self.value_emb = nn.Embedding(sum(self.cardinalities), self.val_dim)
        self.col_emb = nn.Embedding(num_cat, col_id_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        mlp_in = num_cat * d_model + num_cont
        h1 = mlp_hidden_mult[0] * mlp_in
        h2 = mlp_hidden_mult[1] * mlp_in
        self.mlp = nn.Sequential(
            nn.LayerNorm(mlp_in),
            nn.Linear(mlp_in, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),  
        )

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        B, M = x_cat.shape
        assert M == self.num_cat

        x = x_cat + self.offsets.unsqueeze(0)  
        val = self.value_emb(x)                

        cols = torch.arange(M, device=x_cat.device)
        col = self.col_emb(cols).unsqueeze(0).expand(B, -1, -1)  

        tok = torch.cat([col, val], dim=-1)     

        ctx = self.transformer(tok)             
        flat = ctx.reshape(B, M * self.d_model) 

        z = torch.cat([flat, x_cont], dim=-1)   
        logit = self.mlp(z).squeeze(-1)         
        return logit