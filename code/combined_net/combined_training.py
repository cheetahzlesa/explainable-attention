from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from combined_transformer import CombinedTransformer
from prepare_data import PEMalwareOntologyTabular
from prepare_set_data import PEMalwareOntologySet, BucketBatchSampler

import wandb 

class CombinedPEMalwareDataset(Dataset):

    def __init__(self, raw_json: str, actions_json: str, binar_entropy: bool = False):
        self.tab = PEMalwareOntologyTabular(
            raw_json,
            actions_json,
            normalize_numeric=True,
            log1p_numeric=True,
        )
        self.set = PEMalwareOntologySet(raw_json, binar_entropy=binar_entropy)

        if len(self.tab) != len(self.set):
            raise ValueError(
                f"Dataset length mismatch: tab={len(self.tab)}, set={len(self.set)}. "
                "Both dataset classes must index exactly the same samples."
            )
        if not np.array_equal(self.tab.y, self.set.y):
            raise ValueError("Label/index mismatch between tabular and set datasets.")

        self.y = self.tab.y
        self.lengths = self.set.lenghts
        self.num_cat = self.tab.num_cat
        self.num_cont = self.tab.num_cont

    def __len__(self):
        return len(self.tab)

    def __getitem__(self, idx):
        x_cat, x_num, y_tab = self.tab[idx]
        x_set, y_set = self.set[idx]

        if y_tab.item() != y_set.item():
            raise RuntimeError(f"Label mismatch at index {idx}")

        return x_cat, x_num, x_set, y_tab


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split_indices(y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    from sklearn.model_selection import StratifiedShuffleSplit

    idx = np.arange(len(y))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(splitter.split(idx, y))
    return tr_idx, te_idx


def make_loaders(ds, tr_idx, te_idx, batch_size, eval_batch_size, device):
    tr = torch.utils.data.Subset(ds, tr_idx)
    te = torch.utils.data.Subset(ds, te_idx)

    # x_set has variable sequence length, so keep samples with equal number of
    # sections in the same batch. Then PyTorch's default collate is sufficient.
    tr_sampler = BucketBatchSampler(
        buckets=[ds.lengths[i] for i in tr_idx],
        batch_size=batch_size,
        shuffle=True,
    )
    te_sampler = BucketBatchSampler(
        buckets=[ds.lengths[i] for i in te_idx],
        batch_size=eval_batch_size,
        shuffle=False,
    )

    pin = device.type == "cuda"
    return (
        DataLoader(tr, batch_sampler=tr_sampler, num_workers=0, pin_memory=pin),
        DataLoader(te, batch_sampler=te_sampler, num_workers=0, pin_memory=pin),
    )


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, use_amp=True, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    n = 0
    amp_enabled = use_amp and device.type == "cuda"

    for x_cat, x_num, x_set, y in loader:
        x_cat = x_cat.to(device, non_blocking=True)
        x_num = x_num.to(device, non_blocking=True)
        x_set = x_set.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=amp_enabled):
            logit = model(x_cat, x_num, x_set)
            loss = loss_fn(logit, y)

        if amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item()) * y.size(0)
        n += y.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def eval_metrics(model, loader, device):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    model.eval()
    ys, ps = [], []

    for x_cat, x_num, x_set, y in loader:
        x_cat = x_cat.to(device, non_blocking=True)
        x_num = x_num.to(device, non_blocking=True)
        x_set = x_set.to(device, non_blocking=True)

        prob = torch.sigmoid(model(x_cat, x_num, x_set)).cpu().numpy()
        ps.append(prob)
        ys.append(y.numpy())

    y = np.concatenate(ys).astype(np.float32)
    p = np.concatenate(ps).astype(np.float32)
    pred = (p >= 0.5).astype(np.int64)

    return {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else float("nan"),
        "acc": float(accuracy_score(y.astype(np.int64), pred)),
        "f1": float(f1_score(y.astype(np.int64), pred, zero_division=0)),
    }


def main():

    with wandb.init(project="combined-transformer-1") as run:
        config = run.config

        epochs = config.epochs
        batch_size = config.batch_size
        eval_batch_size = config.batch_size
        lr = config.lr
        weight_decay = config.weight_decay
        

        tab_d_model= config.tab_d_model
        tab_col_id_dim = config.tab_col_id_dim
        tab_heads = config.tab_heads
        tab_layers = config.tab_layers
        set_hidden= config.set_hidden
        set_heads = config.set_heads
        fusion_hidden = config.fusion_hidden
        
        raw_json = RAW_JSON
        actions_json = ACTIONS_JSON
        test_size = 0.2
        seed = 42
        dropout = 0.1
        save_path = "combined_best.pt"
        use_amp = True



        
    
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True



        ds = CombinedPEMalwareDataset(raw_json, actions_json)
        tr_idx, te_idx = stratified_split_indices(ds.y, test_size=test_size, seed=seed)
        tr_loader, te_loader = make_loaders(
            ds, tr_idx, te_idx, batch_size, eval_batch_size, device
        )

        model = CombinedTransformer(
            num_cat=ds.num_cat,
            num_cont=ds.num_cont,
            tab_d_model=tab_d_model,
            tab_col_id_dim=tab_col_id_dim,
            tab_heads=tab_heads,
            tab_layers=tab_layers,
            set_input_dim=9,
            set_hidden=set_hidden,
            set_heads=set_heads,
            dropout=dropout,
            fusion_hidden=fusion_hidden,
        ).to(device)

        y_tr = ds.y[tr_idx]
        pos = int((y_tr == 1).sum())
        neg = int((y_tr == 0).sum())
        pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = GradScaler(enabled=use_amp and device.type == "cuda")

        best_auc = -1.0

        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(
                model, tr_loader, optimizer, loss_fn, device, scaler, use_amp=use_amp
            )
            metrics = eval_metrics(model, te_loader, device)

            print(
                f"epoch={epoch:03d} loss={loss:.4f} "
                f"auc={metrics['auc']:.4f} acc={metrics['acc']:.4f} f1={metrics['f1']:.4f}"
            )

            run.log({"auc": metrics['auc'], "acc": metrics['acc'], "f1": metrics['f1']})

        del model, loss_fn, optimizer
        import gc; gc.collect()
        torch.cuda.empty_cache()
        


if __name__ == "__main__":
    RAW_JSON = ""
    ACTIONS_JSON = "actions.json"
    


    random_sweep_config = {
    "name": "stage-1-random-broad",
    "method": "random",
    "metric": {
        "name": "acc",
        "goal": "maximize",
    },
    "run_cap": 50,
    "parameters": {
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-2,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-7,
            "max": 1e-2,
        },
        "batch_size": {
            "values": [16, 32, 64, 128, 256],
        },
        "epochs": {
            "values": [5, 10, 15, 20]
        },
        "tab_d_model":{
            "values": [16, 32, 64, 128]
        },
        
        "tab_col_id_dim":{
            "values": [4, 8, 16, 32]
        },
        
        "tab_heads":{
            "values":[4, 8, 16, 32]
        },
                
        "tab_layers":{
            "values":[2, 4, 6, 8]
        },
                
        "set_hidden":{
            "values": [128, 256, 512, 1024]
        },
        
        "set_heads":{
            "values" : [2, 4, 8, 16, 32]

        },
                
        "fusion_hidden":{
            "values": [128, 256, 512, 1024]
        }

    },
}   









    # Initialize the sweep by passing in the config dictionary
    sweep_id = wandb.sweep(sweep=random_sweep_config, project="transformer-1")

    # Start the sweep job
    wandb.agent(sweep_id, function=main)
