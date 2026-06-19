from __future__ import annotations

import os
import random
from typing import Optional, List, Dict, Any
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import wandb


from set_transformer import SetTransformer
from prepare_set_data import PEMalwareOntologySet, BucketBatchSampler


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = True,
    grad_clip: Optional[float] = 1.0,
) -> float:
    
    model.train()
    total = 0.0
    n = 0

    amp_enabled = use_amp and (device.type == "cuda") and (scaler is not None)

    for x_features, y in loader:
        x_features = x_features.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if amp_enabled:
            with autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
                logit = model(x_features)
                loss = loss_fn(logit, y)


            scaler.scale(loss).backward()

            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            logit = model(x_features)
            loss = loss_fn(logit, y)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        total += float(loss.item()) * y.size(0)
        n += y.size(0)

    return total / max(1, n)


@torch.no_grad()
def eval_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []

    for x_features, y in loader:
        x_features = x_features.to(device, non_blocking=True)

        logit = model(x_features)
        prob = torch.sigmoid(logit).detach().cpu().numpy()

        ps.append(prob)
        ys.append(y.detach().cpu().numpy())

    y = np.concatenate(ys).astype(np.float32)
    p = np.concatenate(ps).astype(np.float32)

    out = {}

    # AUC
    if len(np.unique(y)) >= 2:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(y, p))
    else:   
        out["auc"] = float("nan")

    # Threshold metrics at 0.5
    pred = (p >= 0.5).astype(np.int64)
    y_int = y.astype(np.int64)

    from sklearn.metrics import accuracy_score, f1_score
    out["acc"] = float(accuracy_score(y_int, pred))
    out["f1"] = float(f1_score(y_int, pred, zero_division=0))

    return out


def stratified_split_indices(y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    from sklearn.model_selection import StratifiedShuffleSplit
    idx = np.arange(len(y))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(splitter.split(idx, y))
    return tr_idx, te_idx

    



def main():
    with wandb.init(project="transformer-1") as run:
        config = run.config
        
        epochs = config.epochs
        batch_size = config.batch_size
        eval_batch_size = config.batch_size
        lr = config.lr
        weight_decay = config.weight_decay
        dim_hidden = config.dim_hidden
        num_heads = config.num_heads


        raw_json = RAW_JSON
        seed= 42
        use_amp= True
        grad_clip = 1.0
        test_size = 0.2
        binary_entropy = False



        set_seed(seed)

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            

        # Dataset (OWL parsing happens here)
        ds = PEMalwareOntologySet(
            raw_json,
            binar_entropy=binary_entropy,
        )

        


        tr_idx, te_idx = stratified_split_indices(ds.y, test_size=test_size, seed=seed)
        tr = torch.utils.data.Subset(ds, tr_idx)
        te = torch.utils.data.Subset(ds, te_idx)

        tr_bucket_sampler = BucketBatchSampler(
            buckets=[ds.lenghts[i] for i in tr_idx],
            batch_size=batch_size,
            shuffle=True
        )

        te_bucket_sampler = BucketBatchSampler(
            buckets=[ds.lenghts[j] for j in te_idx],
            batch_size=eval_batch_size,
            shuffle=False
        )

        
        pin = (device.type == "cuda")
        tr_loader = DataLoader(tr, batch_sampler=tr_bucket_sampler, num_workers=0, pin_memory=pin)
        te_loader = DataLoader(te, batch_sampler=te_bucket_sampler, num_workers=0, pin_memory=pin)

        

        model = SetTransformer(dim_input = 9, dim_hidden=dim_hidden, num_heads=num_heads,ln=True).to(device)

        y_tr = ds.y[tr_idx]
        pos = int((y_tr == 1).sum())
        neg = int((y_tr == 0).sum())
        if pos == 0:
            raise RuntimeError("Training split has 0 positives. Check your dataset / split.")

        pos_weight = torch.tensor([neg / max(1, pos)], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))

        print("\n==============================")
        print(f"RUN: test_run")
        print(f"device={device} | amp={use_amp and device.type=='cuda'}")
        print(f"seed={seed} | epochs={epochs} | batch={batch_size} | lr={lr} wd={weight_decay}")
        print(f"Train size={len(tr)} | Test size={len(te)} | pos={pos} neg={neg} pos_weight={float(pos_weight.item()):.4f}")
        

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model, tr_loader, optimizer, loss_fn, device,
                scaler=scaler, use_amp=use_amp, grad_clip=grad_clip
            )
            metrics = eval_metrics(model, te_loader, device)

            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} "
                f"val_auc={metrics['auc']:.4f} "
                f"val_acc={metrics['acc']:.4f} "
                f"val_f1={metrics['f1']:.4f}"
            )
            run.log({"auc": metrics['auc'], "acc": metrics['acc'], "f1": metrics['f1']})
        
        del model, optimizer, loss_fn
        import gc; gc.collect()
        torch.cuda.empty_cache()

    

if __name__ == "__main__":
    # Change these paths
    RAW_JSON = ""
    

    # Define a sweep config dictionary
    grid_sweep_configuration = {
        "method": "grid",
        "name": "sweep-1",
        # Metric that you want to optimize
        # For example, if you want to maximize validation
        # accuracy set "goal": "maximize" and the name of the variable 
        # you want to optimize for, in this case "val_acc"
        
        "metric": {
            "goal": "maximize",
            "name": "acc"
            },
        "parameters": {
            "batch_size": {"values": [16, 32, 64, 128, 256]},
            "epochs": {"values": [5, 10, 15, 20]},
            "lr": {"values": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2,]},
            "weight_decay" : {"values" :  [0.0, 1e-5, 1e-4, 1e-3]}
            
            }
        }


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
        "dim_hidden": {
            "values": [128, 256, 512, 1024],
        },
        "dropout": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.5,
        },
        "epochs": {
            "values": [5, 10, 15, 20]
        },
        "num_heads" :{
            "values" : [2, 4, 8, 16, 32]
        }

    },
}   


    # Initialize the sweep by passing in the config dictionary
    sweep_id = wandb.sweep(sweep=random_sweep_config, project="transformer-1")

    # Start the sweep job
    wandb.agent(sweep_id, function=main)






    

    
    
