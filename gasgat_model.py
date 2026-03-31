"""
gasgat_model.py
===============
Architecture complète du modèle GasGAT :
  - Graph Attention Network (GAT) multi-têtes
  - Pipeline d'entraînement avec cross-validation
  - Class weights pour corriger le déséquilibre efficient/inefficient
  - Évaluation (accuracy, precision, recall, F1, McNemar)
  - Visualisation des attention weights

Fixes v2 :
  - Forcé CPU (MPS incompatible avec scatter_reduce sur Apple Silicon)
  - Architecture GAT corrigée : dimensions inter-couches explicites
  - Class weights automatiques depuis la distribution du dataset

Usage :
    python gasgat_model.py --graphs_dir data/graphs/ --mode train
    python gasgat_model.py --graphs_dir data/graphs/ --mode cv
    python gasgat_model.py --graphs_dir data/graphs/ --mode eval --model_path results/gasgat_best_fold0.pt

Requirements :
    pip install torch torch-geometric scikit-learn matplotlib seaborn tqdm
"""

import os
import json
import argparse
import logging
from pathlib import Path

# ── Fix MPS : fallback CPU pour les opérateurs non supportés ──────────────────
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("gasgat_training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Hyperparamètres par défaut ─────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Architecture
    "node_feature_dim": 16,    # dimension features nœud (graph_builder.py)
    "hidden_dim":       128,   # dimension cachée couches GAT
    "num_heads":        4,     # têtes d'attention
    "num_layers":       3,     # couches GAT empilées
    "dropout":          0.3,   # taux de dropout
    "num_classes":      2,     # 0=efficient, 1=inefficient

    # Entraînement
    "lr":               0.001,
    "weight_decay":     1e-4,
    "epochs":           200,
    "batch_size":       32,
    "patience":         20,    # early stopping

    # Cross-validation
    "n_folds":          5,
    "random_seed":      42,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATASET
# ══════════════════════════════════════════════════════════════════════════════

class GasGATDataset(Dataset):
    """
    Dataset PyTorch Geometric pour les graphes GasGAT.
    Charge les fichiers .pt générés par graph_builder.py.
    """

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
        """Retourne tous les labels (pour stratification et class weights)."""
        labels = []
        for f in self.graph_files:
            try:
                data = torch.load(f, weights_only=False)
                labels.append(int(data.y.item()))
            except Exception:
                labels.append(0)
        return labels


# ══════════════════════════════════════════════════════════════════════════════
# 2. ARCHITECTURE GASGAT
# ══════════════════════════════════════════════════════════════════════════════

class GATLayer(nn.Module):
    """
    Couche GAT avec connexion résiduelle et LayerNorm.
    h'_i = σ( Σ_{j∈N(i)} α_ij · W · h_j )
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        heads:        int   = 4,
        dropout:      float = 0.3,
        concat:       bool  = True,
        residual:     bool  = True,
    ):
        super().__init__()
        self.concat   = concat
        self.residual = residual

        self.gat = GATConv(
            in_channels  = in_channels,
            out_channels = out_channels,
            heads        = heads,
            dropout      = dropout,
            concat       = concat,
        )

        # Dimension de sortie réelle
        out_dim = out_channels * heads if concat else out_channels

        # Projection résiduelle si dimensions différentes
        if residual and in_channels != out_dim:
            self.res_proj = nn.Linear(in_channels, out_dim, bias=False)
        else:
            self.res_proj = None

        self.norm    = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ):
        if return_attention_weights:
            out, attn = self.gat(x, edge_index, return_attention_weights=True)
        else:
            out  = self.gat(x, edge_index)
            attn = None

        out = F.elu(out)
        out = self.dropout(out)

        if self.residual:
            res = self.res_proj(x) if self.res_proj is not None else x
            if res.shape[-1] == out.shape[-1]:
                out = out + res

        out = self.norm(out)

        if return_attention_weights:
            return out, attn
        return out


class GasGAT(nn.Module):
    """
    Modèle GasGAT complet :
      Input  : graphes sémantiques G=(V,E) avec features nœuds [N, 16]
      Output : classification binaire [B, 2] (efficient / inefficient)

    Architecture :
      Linear(16→hidden) → GATLayer×L → GlobalPool(mean+max) → MLP → softmax

    Fix dimensions :
      - Couche i>0 reçoit hidden*heads (concat=True des couches précédentes)
      - Dernière couche : concat=False → out_dim = hidden
    """

    def __init__(self, config: dict = None):
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.config = cfg

        in_dim      = cfg["node_feature_dim"]
        hidden      = cfg["hidden_dim"]
        heads       = cfg["num_heads"]
        num_layers  = cfg["num_layers"]
        dropout     = cfg["dropout"]
        num_classes = cfg["num_classes"]

        # ── Projection initiale ───────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # ── Couches GAT avec dimensions explicites ────────────────────────────
        # Règle : out_dim d'une couche concat=True = out_ch * heads = hidden
        #          out_dim d'une couche concat=False = out_ch = hidden
        # → toutes les couches sortent 'hidden', donc in_ch = hidden partout
        self.gat_layers = nn.ModuleList()
        prev_out = hidden   # sortie de input_proj

        for i in range(num_layers):
            is_last = (i == num_layers - 1)

            in_ch  = prev_out                           # toujours = hidden
            out_ch = hidden if is_last else hidden // heads  # 128 ou 32

            layer = GATLayer(
                in_channels  = in_ch,
                out_channels = out_ch,
                heads        = 1 if is_last else heads,
                dropout      = dropout,
                concat       = not is_last,
                residual     = True,
            )
            self.gat_layers.append(layer)
            # out_dim = out_ch*heads si concat, sinon out_ch → toujours = hidden
            prev_out = out_ch * heads if not is_last else out_ch

        # Pooling : mean + max → 2*hidden
        pool_dim = prev_out * 2

        # ── Classificateur MLP ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, pool_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(pool_dim // 2, pool_dim // 4),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(pool_dim // 4, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        batch:      torch.Tensor,
        return_attention: bool = False,
    ):
        # Projection initiale
        h = self.input_proj(x)

        # Couches GAT
        attention_weights = None
        for i, layer in enumerate(self.gat_layers):
            is_last = (i == len(self.gat_layers) - 1)
            if return_attention and is_last:
                h, attention_weights = layer(h, edge_index,
                                             return_attention_weights=True)
            else:
                h = layer(h, edge_index)

        # Pooling global : mean + max
        h_mean = global_mean_pool(h, batch)
        h_max  = global_max_pool(h, batch)
        h_pool = torch.cat([h_mean, h_max], dim=-1)

        logits = self.classifier(h_pool)

        if return_attention:
            return logits, attention_weights
        return logits

    def predict(self, data: Data, device: str = "cpu"):
        """Prédit la classe pour un seul graphe."""
        self.eval()
        with torch.no_grad():
            data  = data.to(device)
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            logits = self(data.x, data.edge_index, batch)
            probs  = F.softmax(logits, dim=-1)
            pred   = probs.argmax(dim=-1).item()
        return pred, probs


# ══════════════════════════════════════════════════════════════════════════════
# 3. ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """Pipeline d'entraînement avec early stopping, class weights et sauvegarde."""

    def __init__(
        self,
        model:        GasGAT,
        config:       dict,
        output_dir:   str  = "results/",
        device:       str  = None,
        class_counts: list = None,   # [n_class0, n_class1]
    ):
        self.model      = model
        self.config     = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Device : CPU forcé (MPS incompatible avec scatter_reduce) ─────────
        self.device = device or "cpu"
        logger.info("Device : %s", self.device)
        self.model.to(self.device)

        self.optimizer = Adam(
            model.parameters(),
            lr           = config["lr"],
            weight_decay = config["weight_decay"],
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", patience=10, factor=0.5
        )

        # ── Class weights pour corriger le déséquilibre ───────────────────────
        # Dataset GasGAT : ~23K efficient vs ~4K inefficient → ratio 5.4:1
        # Formule : weight_c = total / (n_classes * n_c)
        if class_counts is not None:
            n0, n1  = float(class_counts[0]), float(class_counts[1])
            total   = n0 + n1
            w0      = total / (2.0 * n0)   # ≈ 0.59
            w1      = total / (2.0 * n1)   # ≈ 3.19
            weights = torch.tensor([w0, w1], dtype=torch.float).to(self.device)
            logger.info(
                "Class weights — efficient(0): %.4f | inefficient(1): %.4f  "
                "[ratio %d:%d]", w0, w1, int(n0), int(n1)
            )
        else:
            weights = None
            logger.info("Pas de class weights (distribution inconnue)")

        self.criterion = nn.CrossEntropyLoss(weight=weights)

        # Historique
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "val_f1":     [],
        }

    def _run_epoch(self, loader, train: bool = True):
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        all_preds  = []
        all_labels = []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                batch = batch.to(self.device)
                # Filtrer les labels ambigus (-1) — robuste pour datasets Slither
                labels_flat = batch.y.view(-1)         # toujours [B]
                mask        = labels_flat >= 0
                if mask.sum() == 0:
                    continue
                valid_size = mask.sum().item()

                logits       = self.model(batch.x, batch.edge_index, batch.batch)
                valid_logits = logits[mask]             # [valid_B, num_classes]
                valid_labels = labels_flat[mask]        # [valid_B]
                loss         = self.criterion(valid_logits, valid_labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item() * valid_size
                preds = valid_logits.argmax(dim=-1).cpu().numpy()
                labs  = valid_labels.cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labs.tolist())

        n    = len(all_labels)
        loss = total_loss / n if n > 0 else 0.0
        acc  = accuracy_score(all_labels, all_preds)
        return loss, acc

    def train(self, train_loader, val_loader, fold: int = 0) -> dict:
        best_val_f1  = 0.0
        best_epoch   = 0
        patience_ctr = 0
        best_state   = None
        epochs       = self.config["epochs"]

        pbar = tqdm(range(1, epochs + 1), desc=f"Fold {fold} Training")
        for epoch in pbar:
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss,   val_acc   = self._run_epoch(val_loader,   train=False)
            val_metrics = self.evaluate(val_loader)
            val_f1      = val_metrics["f1"]

            self.scheduler.step(val_f1)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)

            pbar.set_postfix({
                "loss":    f"{train_loss:.4f}",
                "val_acc": f"{val_acc:.4f}",
                "val_f1":  f"{val_f1:.4f}",
            })

            if val_f1 > best_val_f1:
                best_val_f1  = val_f1
                best_epoch   = epoch
                patience_ctr = 0
                best_state   = {k: v.clone() for k, v in self.model.state_dict().items()}
                ckpt = self.output_dir / f"gasgat_best_fold{fold}.pt"
                torch.save(best_state, ckpt)
            else:
                patience_ctr += 1
                if patience_ctr >= self.config["patience"]:
                    logger.info("Early stopping — epoch %d (best=%d)", epoch, best_epoch)
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        logger.info("Fold %d | Best epoch=%d | Val F1=%.4f | Val Acc=%.4f",
                    fold, best_epoch, best_val_f1,
                    self.history["val_acc"][best_epoch - 1])
        return {"best_epoch": best_epoch, "best_val_f1": best_val_f1}

    def evaluate(self, loader) -> dict:
        self.model.eval()
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                batch  = batch.to(self.device)
                labels_flat = batch.y.view(-1)
                mask        = labels_flat >= 0
                if mask.sum() == 0:
                    continue
                logits       = self.model(batch.x, batch.edge_index, batch.batch)
                valid_logits = logits[mask]
                valid_labels = labels_flat[mask]
                preds = valid_logits.argmax(dim=-1).cpu().numpy()
                labs  = valid_labels.cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labs.tolist())

        return {
            "accuracy":  accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average="macro",
                                         zero_division=0),
            "recall":    recall_score(all_labels, all_preds, average="macro",
                                      zero_division=0),
            "f1":        f1_score(all_labels, all_preds, average="macro",
                                  zero_division=0),
        }

    def plot_history(self, fold: int = 0):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(self.history["train_loss"], label="Train Loss", color="steelblue")
        axes[0].plot(self.history["val_loss"],   label="Val Loss",   color="tomato")
        axes[0].set_title(f"Training Loss — Fold {fold}")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history["train_acc"], label="Train Accuracy", color="steelblue")
        axes[1].plot(self.history["val_acc"],   label="Val Accuracy",   color="tomato")
        axes[1].plot(self.history["val_f1"],    label="Val F1",         color="green",
                     linestyle="--")
        axes[1].set_title(f"Accuracy & F1 — Fold {fold}")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / f"training_curves_fold{fold}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Courbes → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# 4. ÉVALUATION FINALE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, loader, output_dir, device="cpu"):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            labels_flat = batch.y.view(-1)
            mask        = labels_flat >= 0
            if mask.sum() == 0:
                continue
            logits       = model(batch.x, batch.edge_index, batch.batch)
            valid_logits = logits[mask]
            valid_labels = labels_flat[mask]
            preds = valid_logits.argmax(dim=-1).cpu().numpy()
            labs  = valid_labels.cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labs.tolist())

    report = classification_report(
        all_labels, all_preds,
        target_names=["Gas-Efficient", "Gas-Inefficient"],
        digits=4,
    )
    logger.info("\n%s", report)
    (output_dir / "classification_report.txt").write_text(report)

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Efficient", "Inefficient"],
                yticklabels=["Efficient", "Inefficient"])
    ax.set_title("Confusion Matrix — GasGAT")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    metrics = {
        "accuracy":  float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, average="macro",
                                            zero_division=0)),
        "recall":    float(recall_score(all_labels, all_preds, average="macro",
                                         zero_division=0)),
        "f1":        float(f1_score(all_labels, all_preds, average="macro",
                                     zero_division=0)),
    }

    # McNemar vs baseline majoritaire
    majority       = max(set(all_labels), key=all_labels.count)
    baseline_preds = [majority] * len(all_labels)
    b = sum(1 for t, p, b in zip(all_labels, all_preds, baseline_preds)
            if t != p and t == b)
    c = sum(1 for t, p, b in zip(all_labels, all_preds, baseline_preds)
            if t == p and t != b)
    mcnemar = float((abs(b - c) - 1) ** 2 / (b + c)) if (b + c) > 0 else 0.0
    metrics["mcnemar"] = mcnemar

    logger.info("Résultats : %s", metrics)
    logger.info("McNemar statistic : %.4f", mcnemar)
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 5. CROSS-VALIDATION 5-FOLD
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate(dataset, config, output_dir="results/"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = dataset.get_labels()
    skf    = StratifiedKFold(
        n_splits     = config["n_folds"],
        shuffle      = True,
        random_state = config["random_seed"],
    )

    # Distribution globale
    n0_global = labels.count(0)
    n1_global = labels.count(1)
    logger.info("Distribution — efficient(0): %d | inefficient(1): %d",
                n0_global, n1_global)

    fold_metrics = []
    device       = "cpu"   # forcé CPU (MPS incompatible)
    indices      = list(range(len(dataset)))

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        logger.info("=" * 50)
        logger.info("FOLD %d / %d", fold + 1, config["n_folds"])
        logger.info("=" * 50)

        train_data = [dataset.get(i) for i in train_idx]
        val_data   = [dataset.get(i) for i in val_idx]

        # Class counts sur le fold d'entraînement
        train_labels = [labels[i] for i in train_idx]
        fold_n0 = train_labels.count(0)
        fold_n1 = train_labels.count(1)

        train_loader = DataLoader(train_data, batch_size=config["batch_size"],
                                  shuffle=True)
        val_loader   = DataLoader(val_data,   batch_size=config["batch_size"],
                                  shuffle=False)

        model   = GasGAT(config)
        trainer = Trainer(
            model, config,
            output_dir   = str(out_dir),
            device       = device,
            class_counts = [fold_n0, fold_n1],
        )
        trainer.train(train_loader, val_loader, fold=fold + 1)
        trainer.plot_history(fold=fold + 1)

        metrics = trainer.evaluate(val_loader)
        metrics["fold"] = fold + 1
        fold_metrics.append(metrics)
        logger.info("Fold %d | Acc=%.4f | F1=%.4f",
                    fold + 1, metrics["accuracy"], metrics["f1"])

    # Agrégation
    keys = ("accuracy", "precision", "recall", "f1")
    avg  = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    std  = {f"{k}_std": float(np.std([m[k] for m in fold_metrics])) for k in keys}

    logger.info("=" * 50)
    logger.info("CROSS-VALIDATION RÉSULTATS (%d folds)", config["n_folds"])
    logger.info("Accuracy  : %.4f ± %.4f", avg["accuracy"],  std["accuracy_std"])
    logger.info("Precision : %.4f ± %.4f", avg["precision"], std["precision_std"])
    logger.info("Recall    : %.4f ± %.4f", avg["recall"],    std["recall_std"])
    logger.info("F1-Score  : %.4f ± %.4f", avg["f1"],        std["f1_std"])
    logger.info("=" * 50)

    results = {**avg, **std, "folds": fold_metrics}
    with open(out_dir / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATION ATTENTION
# ══════════════════════════════════════════════════════════════════════════════

def visualize_attention(model, data, output_dir, device="cpu"):
    model.eval()
    data  = data.to(device)
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        _, attn_output = model(data.x, data.edge_index, batch,
                               return_attention=True)

    if attn_output is None:
        logger.warning("Pas de poids d'attention disponibles.")
        return

    edge_index, attn_weights = attn_output
    if attn_weights.dim() == 2:
        attn_weights = attn_weights.mean(dim=-1)

    attn_np   = attn_weights.cpu().numpy()
    edge_np   = edge_index.cpu().numpy()
    top_k     = min(10, len(attn_np))
    top_idx   = np.argsort(attn_np)[-top_k:][::-1]
    top_edges = [(edge_np[0, i], edge_np[1, i], float(attn_np[i]))
                 for i in top_idx]

    logger.info("Top %d arêtes par attention :", top_k)
    for src, dst, w in top_edges:
        logger.info("  nœud %d → nœud %d | attention = %.4f", src, dst, w)

    fig, ax = plt.subplots(figsize=(10, 6))
    weights = [w for _, _, w in top_edges]
    labels  = [f"{s}→{d}" for s, d, _ in top_edges]
    ax.barh(np.arange(top_k), weights,
            color=plt.cm.RdYlGn_r(np.array(weights) / max(weights)))
    ax.set_yticks(np.arange(top_k))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Attention Weight")
    ax.set_title(f"Top {top_k} Attention Weights\n"
                 f"{getattr(data, 'address', 'contract')}")
    ax.axvline(x=np.mean(weights), color="navy", linestyle="--", alpha=0.7,
               label=f"Mean = {np.mean(weights):.3f}")
    ax.legend()
    plt.tight_layout()
    addr = getattr(data, "address", "contract")
    path = output_dir / f"attention_{addr}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Attention → %s", path)
    return top_edges


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="GasGAT — Graph Attention Network pour l'optimisation gas."
    )
    p.add_argument("--graphs_dir",  type=str, default="data/graphs/")
    p.add_argument("--output_dir",  type=str, default="results/")
    p.add_argument("--mode",        type=str, default="train",
                   choices=["train", "eval", "cv"])
    p.add_argument("--model_path",  type=str, default=None)
    p.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["lr"])
    p.add_argument("--batch_size",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--hidden_dim",  type=int,   default=DEFAULT_CONFIG["hidden_dim"])
    p.add_argument("--num_heads",   type=int,   default=DEFAULT_CONFIG["num_heads"])
    p.add_argument("--num_layers",  type=int,   default=DEFAULT_CONFIG["num_layers"])
    p.add_argument("--dropout",     type=float, default=DEFAULT_CONFIG["dropout"])
    p.add_argument("--n_folds",     type=int,   default=DEFAULT_CONFIG["n_folds"])
    p.add_argument("--seed",        type=int,   default=DEFAULT_CONFIG["random_seed"])
    p.add_argument("--visualize",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = {
        **DEFAULT_CONFIG,
        "epochs":      args.epochs,
        "lr":          args.lr,
        "batch_size":  args.batch_size,
        "hidden_dim":  args.hidden_dim,
        "num_heads":   args.num_heads,
        "num_layers":  args.num_layers,
        "dropout":     args.dropout,
        "n_folds":     args.n_folds,
        "random_seed": args.seed,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = "cpu"   # forcé CPU — MPS incompatible avec scatter_reduce (PyG/Apple Silicon)

    dataset = GasGATDataset(args.graphs_dir, labeled_only=True)
    logger.info("Dataset : %d graphes", len(dataset))
    if len(dataset) == 0:
        logger.error("Aucun graphe trouvé dans %s", args.graphs_dir)
        return

    # ── Cross-validation ──────────────────────────────────────────────────────
    if args.mode == "cv":
        cross_validate(dataset, config, output_dir=args.output_dir)
        return

    # ── Split train / test (80/20) ────────────────────────────────────────────
    labels  = dataset.get_labels()
    n       = len(dataset)
    n_train = int(n * 0.8)
    indices = list(range(n))

    n0_total = labels.count(0)
    n1_total = labels.count(1)
    logger.info("Dataset — efficient(0): %d | inefficient(1): %d",
                n0_total, n1_total)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)

    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    train_data = [dataset.get(i) for i in train_idx]
    test_data  = [dataset.get(i) for i in test_idx]

    train_loader = DataLoader(train_data, batch_size=config["batch_size"],
                              shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=config["batch_size"],
                              shuffle=False)

    logger.info("Train=%d | Test=%d", len(train_data), len(test_data))

    # ── Mode train ────────────────────────────────────────────────────────────
    if args.mode == "train":
        model = GasGAT(config)
        logger.info("Paramètres du modèle : %d",
                    sum(p.numel() for p in model.parameters()))

        train_labels_list = [labels[i] for i in train_idx]
        train_n0 = train_labels_list.count(0)
        train_n1 = train_labels_list.count(1)

        trainer = Trainer(
            model, config,
            output_dir   = args.output_dir,
            device       = device,
            class_counts = [train_n0, train_n1],
        )
        trainer.train(train_loader, test_loader, fold=0)
        trainer.plot_history(fold=0)

        logger.info("Évaluation finale :")
        evaluate_model(model, test_loader, out_dir, device=device)

        if args.visualize and len(test_data) > 0:
            visualize_attention(model, test_data[0], out_dir, device=device)

    # ── Mode eval ─────────────────────────────────────────────────────────────
    elif args.mode == "eval":
        if not args.model_path:
            logger.error("--model_path requis pour le mode eval")
            return
        model = GasGAT(config)
        state = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        logger.info("Modèle chargé : %s", args.model_path)
        evaluate_model(model, test_loader, out_dir, device=device)
        if args.visualize and len(test_data) > 0:
            visualize_attention(model, test_data[0], out_dir, device=device)


if __name__ == "__main__":
    main()
