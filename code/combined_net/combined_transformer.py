from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from modules import SAB, PMA


class TabTransformerCLS(nn.Module):
    """Tabular branch that returns one CLS representation per PE sample.

    Categorical features are represented exactly as in the original
    TabTransformer (column embedding + value embedding). The continuous
    features are projected into one additional token so the CLS token can
    attend to both categorical and numerical information.
    """

    def __init__(
        self,
        num_cat: int,
        num_cont: int,
        d_model: int = 32,
        col_id_dim: int = 8,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not (0 < col_id_dim < d_model):
            raise ValueError("col_id_dim must satisfy 0 < col_id_dim < d_model")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.num_cat = num_cat
        self.num_cont = num_cont
        self.d_model = d_model
        self.col_id_dim = col_id_dim
        self.val_dim = d_model - col_id_dim

        # Same categorical encoding as the original TabTransformer.
        self.cardinalities = [2] * num_cat
        offsets = np.cumsum([0] + self.cardinalities[:-1]).astype(np.int64)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        self.value_emb = nn.Embedding(sum(self.cardinalities), self.val_dim)
        self.col_emb = nn.Embedding(num_cat, col_id_dim)

        # Learnable class token.
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        # Put all continuous values into one token so they participate in
        # self-attention and therefore influence the final CLS representation.
        self.numeric_token = nn.Sequential(
            nn.LayerNorm(num_cont),
            nn.Linear(num_cont, d_model),
        )

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
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        """Return shape [B, d_model]."""
        B, M = x_cat.shape
        if M != self.num_cat:
            raise ValueError(f"Expected {self.num_cat} categorical features, got {M}")

        x = x_cat + self.offsets.unsqueeze(0)
        val = self.value_emb(x)

        cols = torch.arange(M, device=x_cat.device)
        col = self.col_emb(cols).unsqueeze(0).expand(B, -1, -1)
        cat_tokens = torch.cat([col, val], dim=-1)  # [B, M, d_model]

        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        num = self.numeric_token(x_cont).unsqueeze(1)  # [B, 1, d_model]

        tokens = torch.cat([cls, cat_tokens, num], dim=1)
        ctx = self.transformer(tokens)

        return self.out_norm(ctx[:, 0, :])


class SetTransformerCLS(nn.Module):
    """Set-transformer branch that returns the PMA seed as a CLS-like token."""

    def __init__(
        self,
        dim_input: int = 9,
        dim_hidden: int = 128,
        num_heads: int = 4,
        ln: bool = True,
    ):
        super().__init__()

        if dim_hidden % num_heads != 0:
            raise ValueError("dim_hidden must be divisible by num_heads")

        self.dim_hidden = dim_hidden
        self.enc = nn.Sequential(
            SAB(dim_input, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        )

        # Keep the same `dec.1` naming as the original SetTransformer so an
        # existing checkpoint can reuse the PMA weights with strict=False.
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_seeds=1, ln=ln),
            nn.Dropout(),
        )
        self.out_norm = nn.LayerNorm(dim_hidden)

    def forward(self, x_set: torch.Tensor) -> torch.Tensor:
        """Return shape [B, dim_hidden]."""
        encoded = self.enc(x_set)
        cls_like = self.dec(encoded).squeeze(1)
        return self.out_norm(cls_like)


class CombinedTransformer(nn.Module):
    """One classifier built from concatenated representations of both branches."""

    def __init__(
        self,
        num_cat: int,
        num_cont: int = 7,
        tab_d_model: int = 32,
        tab_col_id_dim: int = 8,
        tab_heads: int = 8,
        tab_layers: int = 6,
        set_input_dim: int = 9,
        set_hidden: int = 128,
        set_heads: int = 4,
        dropout: float = 0.1,
        fusion_hidden: int = 256,
    ):
        super().__init__()

        self.tab = TabTransformerCLS(
            num_cat=num_cat,
            num_cont=num_cont,
            d_model=tab_d_model,
            col_id_dim=tab_col_id_dim,
            n_heads=tab_heads,
            n_layers=tab_layers,
            dropout=dropout,
        )

        self.set = SetTransformerCLS(
            dim_input=set_input_dim,
            dim_hidden=set_hidden,
            num_heads=set_heads,
            ln=True,
        )

        fusion_dim = tab_d_model + set_hidden
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
        x_set: torch.Tensor,
        return_features: bool = False,
    ):
        tab_cls = self.tab(x_cat, x_cont)  # [B, tab_d_model]
        set_cls = self.set(x_set)          # [B, set_hidden]

        fused = torch.cat([tab_cls, set_cls], dim=-1)
        logit = self.head(fused).squeeze(-1)

        if return_features:
            return logit, {
                "tab_cls": tab_cls,
                "set_cls": set_cls,
                "fused": fused,
            }
        return logit

    def load_pretrained_branches(
        self,
        tab_checkpoint: str | None = None,
        set_checkpoint: str | None = None,
        map_location: str | torch.device = "cpu",
    ):
        """Optionally initialize both branches from old single-model checkpoints.

        The old classification heads are intentionally ignored. The original
        TabTransformer did not contain CLS/numeric tokens, so those parameters
        remain newly initialized and should be fine-tuned jointly.
        """
        reports = {}

        if tab_checkpoint is not None:
            ckpt = torch.load(tab_checkpoint, map_location=map_location)
            state = ckpt.get("model_state_dict", ckpt)
            reports["tab"] = self.tab.load_state_dict(state, strict=False)

        if set_checkpoint is not None:
            ckpt = torch.load(set_checkpoint, map_location=map_location)
            state = ckpt.get("model_state_dict", ckpt)
            reports["set"] = self.set.load_state_dict(state, strict=False)

        return reports
