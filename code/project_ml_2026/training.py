from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tab_transformer import TabTransformer
from prepare_data import PEMalwareOntologyTabular


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, optimizer, loss_fn) -> float:
    model.train()
    total = 0.0
    n = 0

    for x_cat, x_num, y in loader:
        optimizer.zero_grad(set_to_none=True)
        logit = model(x_cat, x_num)
        loss = loss_fn(logit, y)
        loss.backward()
        optimizer.step()

        total += float(loss.item()) * y.size(0)
        n += y.size(0)

    return total / max(1, n)


@torch.no_grad()
def eval_metrics(model, loader) -> dict:
    model.eval()
    ys, ps = [], []

    for x_cat, x_num, y in loader:
        logit = model(x_cat, x_num)
        prob = torch.sigmoid(logit).cpu().numpy()
        ps.append(prob)
        ys.append(y.cpu().numpy())

    y = np.concatenate(ys).astype(np.float32)
    p = np.concatenate(ps).astype(np.float32)

    out = {}

    if len(np.unique(y)) >= 2:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(y, p))
    else:
        out["auc"] = float("nan")

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


def main(
    owl_path: str,
    examples_path: str,
    *,
    epochs: int = 20,
    batch_size: int = 128,
    eval_batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    test_size: float = 0.2,
    seed: int = 42,
    # model
    d_model: int = 32,
    col_id_dim: int = 8,
    n_heads: int = 8,
    n_layers: int = 6,
    dropout: float = 0.1,
    mlp_hidden_mult: tuple[int, int] = (4, 2),
    
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    
    set_seed(seed)

    ds = PEMalwareOntologyTabular(
        owl_path,
        examples_path,
        normalize_numeric=True,
        log1p_numeric=True,
    )

    tr_idx, te_idx = stratified_split_indices(ds.y, test_size=test_size, seed=seed)
    tr = torch.utils.data.Subset(ds, tr_idx)
    te = torch.utils.data.Subset(ds, te_idx)

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0)
    te_loader = DataLoader(te, batch_size=eval_batch_size, shuffle=False, num_workers=0)

    num_cat = len(ds.cat_feature_names)
    num_cont = ds.x_num.shape[1]

    
    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

    model = TabTransformer(
        num_cat=num_cat,
        num_cont=num_cont,
        d_model=d_model,
        col_id_dim=col_id_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        mlp_hidden_mult=mlp_hidden_mult,
    )

    
    y_tr = ds.y[tr_idx]
    pos = int((y_tr == 1).sum())
    neg = int((y_tr == 0).sum())
    if pos == 0:
        raise RuntimeError("Training split has 0 positives. Check your dataset / split.")
    pos_weight = torch.tensor([neg / max(1, pos)], dtype=torch.float32)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    
    best_auc = -1.0
    best_epoch = -1
    best_metrics = None

    print("\n==============================")
    print(f"RUN: {run_name}")
    print(f"seed={seed} | epochs={epochs} | batch={batch_size} | lr={lr} wd={weight_decay}")
    print(f"d_model={d_model} col_id_dim={col_id_dim} heads={n_heads} layers={n_layers} drop={dropout} mlp={mlp_hidden_mult}")
    print(f"Train size={len(tr)} | Test size={len(te)} | pos={pos} neg={neg} pos_weight={float(pos_weight.item()):.4f}")
    print(f"num_cat={num_cat} | num_cont={num_cont}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, tr_loader, optimizer, loss_fn)
        metrics = eval_metrics(model, te_loader)
        auc = metrics["auc"]

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_auc={auc:.4f} "
            f"val_acc={metrics['acc']:.4f} "
            f"val_f1={metrics['f1']:.4f}"
        )

        

    summary = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_auc": float(best_auc),
        "best_metrics": best_metrics,
        "num_cat": num_cat,
        "num_cont": num_cont,
        "train_pos": pos,
        "train_neg": neg,
        "hyperparams": {
            "epochs": epochs,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "test_size": test_size,
            "seed": seed,
            "d_model": d_model,
            "col_id_dim": col_id_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
            "mlp_hidden_mult": mlp_hidden_mult,
        },
    }

    

    print(f"Best AUC: {best_auc:.4f} at epoch {best_epoch}")
    return summary


if __name__ == "__main__":
    OWL_PATH = "/home/pythoninnutshell/Desktop/gabo/gabo/magisterke/diplomovka/pe-malware-ontology-main/example/dataset_8_1000.owl"
    EXAMPLES_PATH = "/home/pythoninnutshell/Desktop/gabo/gabo/magisterke/diplomovka/pe-malware-ontology-main/example/dataset_8_1000_examples.json"

    
    

    RUNS = [
        
        dict(run_name="base_32d_6L_8H", d_model=32, col_id_dim=8, n_heads=8, n_layers=6, dropout=0.1,
             lr=3e-4, weight_decay=1e-4, batch_size=128, epochs=20, mlp_hidden_mult=(2, 1)),

        
        dict(run_name="small_16d_4L_4H", d_model=16, col_id_dim=4, n_heads=4, n_layers=4, dropout=0.1,
             lr=5e-4, weight_decay=1e-4, batch_size=128, epochs=20, mlp_hidden_mult=(2, 1)),

        
        dict(run_name="wide_64d_2L_8H", d_model=64, col_id_dim=16, n_heads=8, n_layers=2, dropout=0.2,
             lr=3e-4, weight_decay=1e-4, batch_size=64, epochs=20, mlp_hidden_mult=(1, 1)),

        
        dict(run_name="reg_32d_6L_8H", d_model=32, col_id_dim=8, n_heads=8, n_layers=6, dropout=0.2,
             lr=1e-4, weight_decay=1e-3, batch_size=128, epochs=25, mlp_hidden_mult=(1, 1)),
    ]

    all_summaries = []
    best = None

    for i, cfg in enumerate(RUNS):
        run_name = cfg.get("run_name", f"run_{i:02d}")

        if cfg["d_model"] % cfg["n_heads"] != 0:
            print(f"Skipping {run_name}: d_model not divisible by n_heads")
            continue

        summary = main(
            OWL_PATH,
            EXAMPLES_PATH,
            seed=42 + i,         
            test_size=0.2,
            eval_batch_size=256,
            **cfg,
        )
        all_summaries.append(summary)

        if best is None or (not np.isnan(summary["best_auc"]) and summary["best_auc"] > best["best_auc"]):
            best = summary

    print("\n==============================")
    if best:
        print(f"BEST RUN: {best['run_name']} | AUC={best['best_auc']:.4f} | ckpt={best['save_path']}")
    else:
        print("No successful runs.")
