"""
gnn_baselines.py
================
Comparaison équitable de GasGAT avec trois baselines GNN classiques :
  - GCN  : Graph Convolutional Network (Kipf & Welling, 2017)
  - GraphSAGE : Graph Sample and Aggregate (Hamilton et al., 2017)
  - GIN  : Graph Isomorphism Network (Xu et al., 2019)
  - GasGAT : notre modèle (Said et al., 2025)

Protocole expérimental identique pour tous les modèles :
  - Mêmes graphes (.pt), mêmes labels, même seed
  - Cross-validation 5-fold stratifiée (StratifiedKFold, seed=42)
  - Même architecture MLP de classification (pool → Linear)
  - Même optimiseur (Adam, lr=0.001, weight_decay=1e-4)
  - Même early stopping (patience=20)
  - Même class weights
  - CPU forcé (compatibilité Apple Silicon)

Usage :
    python gnn_baselines.py \\
        --graphs_dir data/graphs_slither/ \\
        --output_dir results/gnn_comparison/ \\
        --gasgat_cv  results/slither_cv/cv_results.json

    # Sur le dataset heuristique original :
    python gnn_baselines.py \\
        --graphs_dir data/graphs/ \\
        --output_dir results/gnn_comparison_heuristic/ \\
        --gasgat_cv  results/cv/cv_results.json

Requirements :
    pip install torch torch-geometric scikit-learn matplotlib seaborn tqdm
"""

import os
import json
import argparse
import logging
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv, SAGEConv, GINConv,
    global_mean_pool, global_max_pool,
)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("gnn_baselines.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Config commune à tous les modèles ─────────────────────────────────────────
COMMON_CONFIG = {
    "node_feature_dim": 16,
    "hidden_dim":       128,
    "num_layers":       3,
    "dropout":          0.3,
    "num_classes":      2,
    "lr":               0.001,
    "weight_decay":     1e-4,
    "epochs":           200,
    "batch_size":       32,
    "patience":         20,
    "n_folds":          5,
    "random_seed":      42,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATASET (identique à GasGAT)
# ══════════════════════════════════════════════════════════════════════════════

class GasGATDataset(Dataset):
    def __init__(self, graphs_dir: str, labeled_only: bool = True):
        super().__init__()
        self.graphs_dir   = Path(graphs_dir)
        self.labeled_only = labeled_only
        self.graph_files  = self._load_file_list()

    def _load_file_list(self) -> list:
        files = sorted(self.graphs_dir.glob("*.pt"))
        if self.labeled_only:
            valid = []
            for f in files:
                try:
                    data = torch.load(f, weights_only=False)
                    if hasattr(data, "y") and data.y.item() != -1:
                        valid.append(f)
                except Exception:
                    pass
            logger.info("Graphes labellisés : %d / %d", len(valid), len(files))
            return valid
        return files

    def len(self) -> int:
        return len(self.graph_files)

    def get(self, idx: int) -> Data:
        return torch.load(self.graph_files[idx], weights_only=False)

    def get_labels(self) -> list:
        labels = []
        for f in self.graph_files:
            try:
                data = torch.load(f, weights_only=False)
                labels.append(int(data.y.item()))
            except Exception:
                labels.append(0)
        return labels


# ══════════════════════════════════════════════════════════════════════════════
# 2. ARCHITECTURES GNN
# ══════════════════════════════════════════════════════════════════════════════

class GCNModel(nn.Module):
    """
    Graph Convolutional Network (Kipf & Welling, ICLR 2017).
    h_i = σ(Σ_{j∈N(i)} (1/√(d_i·d_j)) · W · h_j)
    """
    def __init__(self, config: dict):
        super().__init__()
        in_dim  = config["node_feature_dim"]
        hidden  = config["hidden_dim"]
        n_layers = config["num_layers"]
        dropout = config["dropout"]
        n_cls   = config["num_classes"]

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ELU(), nn.Dropout(dropout)
        )
        self.convs = nn.ModuleList([
            GCNConv(hidden, hidden) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(n_layers)
        ])
        self.dropout   = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_cls),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch):
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = F.elu(conv(h, edge_index))
            h = self.dropout(norm(h))
        h_pool = torch.cat([global_mean_pool(h, batch),
                            global_max_pool(h, batch)], dim=-1)
        return self.classifier(h_pool)


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE — Graph Sample and Aggregate (Hamilton et al., NeurIPS 2017).
    h_i = σ(W · [h_i || mean_{j∈N(i)} h_j])
    """
    def __init__(self, config: dict):
        super().__init__()
        in_dim   = config["node_feature_dim"]
        hidden   = config["hidden_dim"]
        n_layers = config["num_layers"]
        dropout  = config["dropout"]
        n_cls    = config["num_classes"]

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ELU(), nn.Dropout(dropout)
        )
        self.convs = nn.ModuleList([
            SAGEConv(hidden, hidden) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(n_layers)
        ])
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_cls),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch):
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = F.elu(conv(h, edge_index))
            h = self.dropout(norm(h))
        h_pool = torch.cat([global_mean_pool(h, batch),
                            global_max_pool(h, batch)], dim=-1)
        return self.classifier(h_pool)


class GINModel(nn.Module):
    """
    Graph Isomorphism Network (Xu et al., ICLR 2019).
    h_i = MLP((1+ε) · h_i + Σ_{j∈N(i)} h_j)
    Théoriquement aussi puissant que le test de Weisfeiler-Leman.
    """
    def __init__(self, config: dict):
        super().__init__()
        in_dim   = config["node_feature_dim"]
        hidden   = config["hidden_dim"]
        n_layers = config["num_layers"]
        dropout  = config["dropout"]
        n_cls    = config["num_classes"]

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ELU(), nn.Dropout(dropout)
        )

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden * 2),
                nn.BatchNorm1d(hidden * 2),
                nn.ReLU(),
                nn.Linear(hidden * 2, hidden),
            )
            self.convs.append(GINConv(mlp, train_eps=True))

        self.norms   = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_cls),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch):
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = F.elu(conv(h, edge_index))
            h = self.dropout(norm(h))
        h_pool = torch.cat([global_mean_pool(h, batch),
                            global_max_pool(h, batch)], dim=-1)
        return self.classifier(h_pool)


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAINER COMMUN
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """Pipeline d'entraînement identique pour tous les modèles GNN."""

    def __init__(self, model, config, output_dir, device="cpu",
                 class_counts=None, model_name="model"):
        self.model      = model
        self.config     = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device     = device
        self.model_name = model_name
        self.model.to(device)

        self.optimizer = Adam(model.parameters(),
                              lr=config["lr"],
                              weight_decay=config["weight_decay"])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", patience=10, factor=0.5
        )

        # Class weights
        if class_counts is not None:
            n0, n1  = float(class_counts[0]), float(class_counts[1])
            total   = n0 + n1
            w0, w1  = total / (2 * n0), total / (2 * n1)
            weights = torch.tensor([w0, w1], dtype=torch.float).to(device)
        else:
            weights = None
        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def _run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                batch       = batch.to(self.device)
                labels_flat = batch.y.view(-1)
                mask        = labels_flat >= 0
                if mask.sum() == 0:
                    continue
                valid_size   = mask.sum().item()
                logits       = self.model(batch.x, batch.edge_index, batch.batch)
                valid_logits = logits[mask]
                valid_labels = labels_flat[mask]
                loss         = self.criterion(valid_logits, valid_labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item() * valid_size
                all_preds.extend(valid_logits.argmax(dim=-1).cpu().tolist())
                all_labels.extend(valid_labels.cpu().tolist())

        n    = len(all_labels)
        loss = total_loss / n if n > 0 else 0.0
        acc  = accuracy_score(all_labels, all_preds) if n > 0 else 0.0
        return loss, acc

    def evaluate(self, loader) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch       = batch.to(self.device)
                labels_flat = batch.y.view(-1)
                mask        = labels_flat >= 0
                if mask.sum() == 0:
                    continue
                logits       = self.model(batch.x, batch.edge_index, batch.batch)
                valid_logits = logits[mask]
                valid_labels = labels_flat[mask]
                all_preds.extend(valid_logits.argmax(dim=-1).cpu().tolist())
                all_labels.extend(valid_labels.cpu().tolist())

        if not all_labels:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

        return {
            "accuracy":  float(accuracy_score(all_labels, all_preds)),
            "precision": float(precision_score(all_labels, all_preds,
                                               average="macro", zero_division=0)),
            "recall":    float(recall_score(all_labels, all_preds,
                                            average="macro", zero_division=0)),
            "f1":        float(f1_score(all_labels, all_preds,
                                        average="macro", zero_division=0)),
        }

    def train(self, train_loader, val_loader, fold=0) -> dict:
        best_f1, best_epoch, patience_ctr, best_state = 0.0, 0, 0, None
        epochs = self.config["epochs"]

        pbar = tqdm(range(1, epochs + 1),
                    desc=f"{self.model_name} Fold {fold}")
        for epoch in pbar:
            self._run_epoch(train_loader, train=True)
            val_metrics = self.evaluate(val_loader)
            val_f1      = val_metrics["f1"]
            val_acc     = val_metrics["accuracy"]

            self.scheduler.step(val_f1)
            pbar.set_postfix({"val_acc": f"{val_acc:.4f}",
                              "val_f1": f"{val_f1:.4f}"})

            if val_f1 > best_f1:
                best_f1      = val_f1
                best_epoch   = epoch
                patience_ctr = 0
                best_state   = {k: v.clone()
                                for k, v in self.model.state_dict().items()}
                ckpt = self.output_dir / f"{self.model_name}_fold{fold}.pt"
                torch.save(best_state, ckpt)
            else:
                patience_ctr += 1
                if patience_ctr >= self.config["patience"]:
                    logger.info("%s Early stopping epoch %d (best=%d)",
                                self.model_name, epoch, best_epoch)
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return {"best_epoch": best_epoch, "best_val_f1": best_f1}


# ══════════════════════════════════════════════════════════════════════════════
# 4. CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "GCN":       GCNModel,
    "GraphSAGE": GraphSAGEModel,
    "GIN":       GINModel,
}


def run_cv_for_model(model_name: str, dataset, config, output_dir, device="cpu"):
    """Cross-validation 5-fold pour un modèle GNN donné."""
    labels = dataset.get_labels()
    skf    = StratifiedKFold(n_splits=config["n_folds"], shuffle=True,
                             random_state=config["random_seed"])
    indices      = list(range(len(dataset)))
    fold_metrics = []

    n0 = labels.count(0)
    n1 = labels.count(1)
    logger.info("%s — Distribution : efficient=%d | inefficient=%d",
                model_name, n0, n1)

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        logger.info("── %s Fold %d/%d ──", model_name, fold + 1, config["n_folds"])

        train_data = [dataset.get(i) for i in train_idx]
        val_data   = [dataset.get(i) for i in val_idx]

        train_labels = [labels[i] for i in train_idx]
        fold_n0 = train_labels.count(0)
        fold_n1 = train_labels.count(1)

        train_loader = DataLoader(train_data, batch_size=config["batch_size"],
                                  shuffle=True)
        val_loader   = DataLoader(val_data,   batch_size=config["batch_size"],
                                  shuffle=False)

        ModelClass = MODEL_REGISTRY[model_name]
        model      = ModelClass(config)
        trainer    = Trainer(
            model, config,
            output_dir   = str(output_dir / model_name),
            device       = device,
            class_counts = [fold_n0, fold_n1],
            model_name   = model_name,
        )
        trainer.train(train_loader, val_loader, fold=fold + 1)

        metrics        = trainer.evaluate(val_loader)
        metrics["fold"] = fold + 1
        fold_metrics.append(metrics)
        logger.info("%s Fold %d | Acc=%.4f | F1=%.4f",
                    model_name, fold + 1, metrics["accuracy"], metrics["f1"])

    keys = ("accuracy", "precision", "recall", "f1")
    avg  = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    std  = {f"{k}_std": float(np.std([m[k] for m in fold_metrics])) for k in keys}

    logger.info("=" * 55)
    logger.info("%s — CROSS-VALIDATION RÉSULTATS", model_name)
    logger.info("Accuracy  : %.4f ± %.4f", avg["accuracy"],  std["accuracy_std"])
    logger.info("Precision : %.4f ± %.4f", avg["precision"], std["precision_std"])
    logger.info("Recall    : %.4f ± %.4f", avg["recall"],    std["recall_std"])
    logger.info("F1-Score  : %.4f ± %.4f", avg["f1"],        std["f1_std"])
    logger.info("=" * 55)

    results = {**avg, **std, "folds": fold_metrics, "model": model_name}
    with open(output_dir / f"{model_name}_cv_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison_table(all_results: dict, output_dir: Path):
    """
    Reproduit la Table 2 du paper : GCN / GraphSAGE / GIN / GasGAT.
    """
    models  = list(all_results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]

    x     = np.arange(len(metrics))
    width = 0.20
    colors = ["steelblue", "seagreen", "darkorange", "tomato"]

    fig, ax = plt.subplots(figsize=(13, 7))

    for i, (model_name, color) in enumerate(zip(models, colors)):
        vals = [all_results[model_name].get(m, 0) for m in metrics]
        stds = [all_results[model_name].get(f"{m}_std", 0) for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width

        bars = ax.bar(x + offset, vals, width,
                      label=model_name, color=color,
                      edgecolor="white", linewidth=0.5,
                      yerr=stds, capsize=3,
                      error_kw={"elinewidth": 1.2})

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.2%}", ha="center", va="bottom",
                    fontsize=7, rotation=45)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "GNN Model Comparison — GasGAT vs Baselines\n"
        "(5-fold cross-validation, same dataset and protocol)",
        fontsize=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.4, 1.10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = output_dir / "gnn_comparison_chart.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Graphique → %s", path)


def print_latex_table(all_results: dict):
    """Génère la Table 2 en LaTeX prête pour Overleaf."""
    print("\n" + "=" * 70)
    print("  TABLE 2 — LaTeX (prête pour Overleaf)")
    print("=" * 70)
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Performance Comparison of GNN Models for Smart Contract Gas Classification}")
    print(r"\label{tab:gnn_comparison}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"\textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\")
    print(r"\hline")

    for model_name, results in all_results.items():
        acc  = results.get("accuracy",  0)
        prec = results.get("precision", 0)
        rec  = results.get("recall",    0)
        f1   = results.get("f1",        0)
        acc_s  = results.get("accuracy_std",  0)
        prec_s = results.get("precision_std", 0)
        rec_s  = results.get("recall_std",    0)
        f1_s   = results.get("f1_std",        0)

        bold_start = r"\textbf{" if model_name == "GasGAT" else ""
        bold_end   = "}"         if model_name == "GasGAT" else ""

        print(
            f"{bold_start}{model_name}{bold_end} & "
            f"{bold_start}{acc:.2%} $\\pm$ {acc_s:.2%}{bold_end} & "
            f"{bold_start}{prec:.2%} $\\pm$ {prec_s:.2%}{bold_end} & "
            f"{bold_start}{rec:.2%} $\\pm$ {rec_s:.2%}{bold_end} & "
            f"{bold_start}{f1:.2%} $\\pm$ {f1_s:.2%}{bold_end} \\\\"
        )

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("=" * 70 + "\n")


def print_summary_table(all_results: dict):
    """Table de comparaison console."""
    print("\n" + "=" * 78)
    print("  GNN COMPARISON — 5-fold Cross-Validation")
    print("=" * 78)
    print(f"{'Model':<14} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("-" * 78)
    for model_name, results in all_results.items():
        mark = " ← OURS" if model_name == "GasGAT" else ""
        print(
            f"{model_name:<14} "
            f"{results.get('accuracy',0):>7.2%}±{results.get('accuracy_std',0):.2%}  "
            f"{results.get('precision',0):>7.2%}±{results.get('precision_std',0):.2%}  "
            f"{results.get('recall',0):>7.2%}±{results.get('recall_std',0):.2%}  "
            f"{results.get('f1',0):>7.2%}±{results.get('f1_std',0):.2%}"
            f"{mark}"
        )
    print("=" * 78)


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Comparaison GasGAT vs GCN, GraphSAGE, GIN — même protocole."
    )
    p.add_argument("--graphs_dir",  default="data/graphs_slither/",
                   help="Graphes .pt à utiliser (défaut: data/graphs_slither/)")
    p.add_argument("--output_dir",  default="results/gnn_comparison/")
    p.add_argument("--gasgat_cv",   default=None,
                   help="Fichier cv_results.json de GasGAT (optionnel)")
    p.add_argument("--models",      nargs="+",
                   default=["GCN", "GraphSAGE", "GIN"],
                   choices=["GCN", "GraphSAGE", "GIN"],
                   help="Modèles à entraîner (défaut: tous)")
    p.add_argument("--epochs",      type=int,   default=COMMON_CONFIG["epochs"])
    p.add_argument("--batch_size",  type=int,   default=COMMON_CONFIG["batch_size"])
    p.add_argument("--hidden_dim",  type=int,   default=COMMON_CONFIG["hidden_dim"])
    p.add_argument("--patience",    type=int,   default=COMMON_CONFIG["patience"])
    p.add_argument("--seed",        type=int,   default=COMMON_CONFIG["random_seed"])
    return p.parse_args()


def main():
    args = parse_args()

    config = {**COMMON_CONFIG,
              "epochs":      args.epochs,
              "batch_size":  args.batch_size,
              "hidden_dim":  args.hidden_dim,
              "patience":    args.patience,
              "random_seed": args.seed}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = "cpu"  # forcé CPU (Apple Silicon MPS incompatible)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = GasGATDataset(args.graphs_dir, labeled_only=True)
    logger.info("Dataset : %d graphes", len(dataset))
    if len(dataset) == 0:
        logger.error("Aucun graphe trouvé dans %s", args.graphs_dir)
        return

    labels = dataset.get_labels()
    n0, n1 = labels.count(0), labels.count(1)
    logger.info("Distribution — efficient(0)=%d | inefficient(1)=%d", n0, n1)

    # ── Entraînement des baselines ────────────────────────────────────────────
    all_results = {}

    for model_name in args.models:
        logger.info("\n" + "=" * 55)
        logger.info("MODÈLE : %s", model_name)
        logger.info("=" * 55)
        results = run_cv_for_model(
            model_name = model_name,
            dataset    = dataset,
            config     = config,
            output_dir = out_dir,
            device     = device,
        )
        all_results[model_name] = results

    # ── Charger résultats GasGAT ──────────────────────────────────────────────
    if args.gasgat_cv and Path(args.gasgat_cv).exists():
        with open(args.gasgat_cv) as f:
            gasgat_results = json.load(f)
        all_results["GasGAT"] = gasgat_results
        logger.info("GasGAT CV chargé depuis %s", args.gasgat_cv)
    else:
        logger.warning(
            "Résultats GasGAT non trouvés. Lancez d'abord :\n"
            "  python gasgat_model.py --graphs_dir %s --mode cv "
            "--output_dir results/slither_cv/",
            args.graphs_dir
        )

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    with open(out_dir / "all_models_cv_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Affichage ─────────────────────────────────────────────────────────────
    print_summary_table(all_results)
    print_latex_table(all_results)
    plot_comparison_table(all_results, out_dir)

    logger.info("✅ Comparaison terminée → %s", out_dir)


if __name__ == "__main__":
    main()
