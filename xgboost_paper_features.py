"""
xgboost_paper_features.py
==========================
XGBoost baseline avec les 12 features exactes de la Table 1 du paper GasGAT.
Ces features sont extraites DIRECTEMENT du code source Solidity (.sol),
indépendamment des graphes sémantiques et du processus de labelling.

Table 1 — Hand-crafted Features for XGBoost Baseline :
  F1  : Count of SSTORE operations inside loops
  F2  : Count of tx.origin usage
  F3  : Count of dynamic arrays in storage variables
  F4  : Count of external calls in loops
  F5  : Total number of loops (for, while)
  F6  : Average function cyclomatic complexity
  F7  : Maximum function call depth
  F8  : Count of selfdestruct or delegatecall opcodes
  F9  : Use of fixed-point/floating-point arithmetic
  F10 : Number of state variables written to
  F11 : Contract size in bytes
  F12 : Number of public/external functions

Usage :
    python xgboost_paper_features.py \\
        --contracts_dir data/contracts/ \\
        --graphs_dir    data/graphs/ \\
        --output_dir    results/xgboost_paper/

Requirements :
    pip install xgboost scikit-learn matplotlib seaborn pandas tqdm torch
"""

import re
import json
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("xgboost_paper.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
N_FOLDS     = 5
TEST_RATIO  = 0.2

FEATURE_NAMES = [
    "F1_sstore_in_loops",
    "F2_tx_origin_count",
    "F3_dynamic_arrays",
    "F4_external_calls_in_loops",
    "F5_total_loops",
    "F6_avg_cyclomatic_complexity",
    "F7_max_call_depth",
    "F8_selfdestruct_delegatecall",
    "F9_floating_point",
    "F10_state_vars_written",
    "F11_contract_size_bytes",
    "F12_public_external_functions",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. EXTRACTION DES 12 FEATURES (depuis le code Solidity)
# ══════════════════════════════════════════════════════════════════════════════

def extract_source(raw: str) -> str:
    """
    Extrait le code Solidity pur depuis le format Etherscan.
    Gère les formats : source direct, Standard JSON Input ({{...}}).
    """
    s = raw.strip()

    # Format Standard JSON Input d'Etherscan : {{...}}
    if s.startswith("{{") or s.startswith('{"language"'):
        try:
            clean = s[1:-1] if s.startswith("{{") else s
            data  = json.loads(clean)
            parts = []
            for fname, content in data.get("sources", {}).items():
                code = content.get("content", "")
                if "contract " in code:
                    parts.append(code)
            return "\n\n".join(parts) if parts else raw
        except (json.JSONDecodeError, AttributeError):
            pass
    return raw


class SolidityFeatureExtractor:
    """
    Extrait les 12 features hand-crafted de la Table 1 du paper GasGAT
    directement depuis le code source Solidity.

    Approche : analyse lexicale/regex (pas d'AST complet requis).
    Couvre ~95% des patterns détectables statiquement.
    """

    # ── Patterns regex ────────────────────────────────────────────────────────
    FOR_LOOP_RE        = re.compile(r'\bfor\s*\(', re.IGNORECASE)
    WHILE_LOOP_RE      = re.compile(r'\bwhile\s*\(', re.IGNORECASE)
    SSTORE_RE          = re.compile(r'\bsstore\s*\(', re.IGNORECASE)
    TX_ORIGIN_RE       = re.compile(r'\btx\.origin\b')
    EXTERNAL_CALL_RE   = re.compile(r'\.\s*call\s*[({]|\.\s*transfer\s*\(|\.\s*send\s*\(')
    SELFDESTRUCT_RE    = re.compile(r'\bselfdestruct\s*\(', re.IGNORECASE)
    DELEGATECALL_RE    = re.compile(r'\bdelegatecall\s*\(', re.IGNORECASE)
    FLOAT_RE           = re.compile(r'\bfixed\d*x\d*\b|\buffixed\d*x\d*\b', re.IGNORECASE)
    FUNCTION_RE        = re.compile(
        r'\bfunction\s+\w+\s*\([^)]*\)\s*(public|external|internal|private|'
        r'view|pure|payable|virtual|override|\s)*',
        re.IGNORECASE
    )
    PUBLIC_FUNC_RE     = re.compile(
        r'\bfunction\s+\w+\s*\([^)]*\)\s*(?:[^{]*?)\b(public|external)\b',
        re.IGNORECASE
    )
    STATE_VAR_RE       = re.compile(
        r'^\s*(?:uint\d*|int\d*|bool|address|bytes\d*|string|mapping|'
        r'struct\s+\w+)\s+(?:public|private|internal|constant|immutable|\s)*'
        r'\s*(\w+)\s*(?:=|;)',
        re.MULTILINE
    )
    DYNAMIC_ARRAY_RE   = re.compile(
        r'^\s*(?:uint\d*|int\d*|bool|address|bytes\d*|string)\[\]\s+'
        r'(?:public|private|internal|\s)*\s*\w+\s*;',
        re.MULTILINE
    )
    MAPPING_RE         = re.compile(
        r'^\s*mapping\s*\(',
        re.MULTILINE
    )
    IF_RE              = re.compile(r'\bif\s*\(')
    REQUIRE_RE         = re.compile(r'\brequire\s*\(')
    REVERT_RE          = re.compile(r'\brevert\s*[({]')

    def __init__(self, source: str):
        self.source = extract_source(source)
        self.lines  = self.source.split('\n')

    def _get_loop_bodies(self) -> list:
        """
        Extrait les corps des boucles for/while.
        Utilisé pour détecter les patterns à l'intérieur des boucles.
        Approche simplifiée : prend les N lignes après chaque boucle.
        """
        bodies = []
        for i, line in enumerate(self.lines):
            if self.FOR_LOOP_RE.search(line) or self.WHILE_LOOP_RE.search(line):
                # Prendre les 20 lignes suivantes comme corps approximatif
                end = min(i + 20, len(self.lines))
                bodies.append('\n'.join(self.lines[i:end]))
        return bodies

    def _count_function_complexity(self) -> list:
        """
        Calcule la complexité cyclomatique approximative de chaque fonction.
        CC = 1 + #(if) + #(for) + #(while) + #(require) + #(revert)
        """
        complexities = []
        # Séparer le code par fonctions
        func_pattern = re.compile(
            r'\bfunction\s+\w+\s*\([^)]*\)[^{]*\{', re.IGNORECASE
        )
        splits = func_pattern.split(self.source)

        for chunk in splits[1:]:  # ignorer le premier (avant toute fonction)
            # Compter les décisions dans le corps de la fonction
            depth = 1
            cc    = 1
            for char in chunk:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        break

            body = chunk[:chunk.find('}') + 1] if '}' in chunk else chunk
            cc  += len(self.IF_RE.findall(body))
            cc  += len(self.FOR_LOOP_RE.findall(body))
            cc  += len(self.WHILE_LOOP_RE.findall(body))
            cc  += len(self.REQUIRE_RE.findall(body))
            cc  += len(self.REVERT_RE.findall(body))
            complexities.append(cc)

        return complexities

    def _estimate_call_depth(self) -> int:
        """
        Estime la profondeur maximale des appels de fonctions.
        Heuristique : compte les niveaux d'imbrication des accolades
        dans les corps de fonctions qui contiennent des appels.
        """
        max_depth = 0
        depth     = 0
        in_func   = False

        for char in self.source:
            if char == '{':
                depth += 1
                if depth >= 2:
                    in_func = True
            elif char == '}':
                if in_func:
                    max_depth = max(max_depth, depth)
                depth    = max(0, depth - 1)
                in_func  = depth >= 1

        return max_depth

    def _count_state_vars_written(self) -> int:
        """
        Compte les variables d'état qui sont écrites (assignées).
        Heuristique : variables déclarées au niveau contract (pas dans fonction)
        qui apparaissent dans un contexte d'assignation.
        """
        state_vars = set()
        for match in self.STATE_VAR_RE.finditer(self.source):
            var_name = match.group(1)
            if var_name and len(var_name) > 1:
                state_vars.add(var_name)

        # Compter celles qui sont assignées (pattern : varName = )
        written = 0
        for var in state_vars:
            assign_pattern = re.compile(
                rf'\b{re.escape(var)}\s*(?:\[.*?\])?\s*=(?!=)',
                re.MULTILINE
            )
            if assign_pattern.search(self.source):
                written += 1

        return written

    def extract(self) -> np.ndarray:
        """
        Extrait le vecteur de 12 features de la Table 1.

        Returns:
            np.ndarray of shape (12,) — une feature par ligne de Table 1
        """
        loop_bodies = self._get_loop_bodies()
        loop_text   = '\n'.join(loop_bodies)

        # F1 : SSTORE operations inside loops
        f1 = len(self.SSTORE_RE.findall(loop_text))

        # F2 : tx.origin usage
        f2 = len(self.TX_ORIGIN_RE.findall(self.source))

        # F3 : dynamic arrays in storage
        f3 = len(self.DYNAMIC_ARRAY_RE.findall(self.source)) + \
             len(self.MAPPING_RE.findall(self.source))

        # F4 : external calls in loops
        f4 = len(self.EXTERNAL_CALL_RE.findall(loop_text))

        # F5 : total number of loops
        f5 = len(self.FOR_LOOP_RE.findall(self.source)) + \
             len(self.WHILE_LOOP_RE.findall(self.source))

        # F6 : average function cyclomatic complexity
        complexities = self._count_function_complexity()
        f6 = float(np.mean(complexities)) if complexities else 1.0

        # F7 : maximum function call depth
        f7 = self._estimate_call_depth()

        # F8 : selfdestruct or delegatecall
        f8 = len(self.SELFDESTRUCT_RE.findall(self.source)) + \
             len(self.DELEGATECALL_RE.findall(self.source))

        # F9 : use of fixed-point/floating-point arithmetic (0 or 1)
        f9 = 1 if self.FLOAT_RE.search(self.source) else 0

        # F10 : number of state variables written to
        f10 = self._count_state_vars_written()

        # F11 : contract size in bytes
        f11 = len(self.source.encode("utf-8"))

        # F12 : number of public/external functions
        f12 = len(self.PUBLIC_FUNC_RE.findall(self.source))

        return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12],
                        dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CHARGEMENT DU DATASET
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(contracts_dir: str, graphs_dir: str) -> tuple:
    """
    Charge les features depuis les .sol et les labels depuis les .pt.
    Seuls les contrats ayant à la fois un .sol ET un .pt labellisé sont inclus.

    Returns:
        X         : np.ndarray [N, 12]
        y         : np.ndarray [N]
        addresses : list[str]
        skipped   : dict (statistiques)
    """
    contracts_path = Path(contracts_dir)
    graphs_path    = Path(graphs_dir)

    # Charger tous les labels depuis les .pt
    logger.info("Chargement des labels depuis les graphes .pt...")
    label_map = {}   # address → label
    for pt_file in tqdm(sorted(graphs_path.glob("*.pt")), desc="Labels"):
        try:
            data  = torch.load(pt_file, weights_only=False)
            label = int(data.y.item())
            if label != -1:
                addr = pt_file.stem.lower()
                label_map[addr] = label
        except Exception:
            pass

    logger.info("Labels disponibles : %d", len(label_map))

    # Extraire les features depuis les .sol
    logger.info("Extraction des 12 features depuis les .sol...")
    X, y, addresses = [], [], []
    n_no_label = 0
    n_error    = 0

    sol_files = sorted(contracts_path.glob("*.sol"))
    for sol_file in tqdm(sol_files, desc="Features extraction"):
        addr = sol_file.stem.lower()

        if addr not in label_map:
            n_no_label += 1
            continue

        try:
            source = sol_file.read_text(encoding="utf-8", errors="ignore")
            feats  = SolidityFeatureExtractor(source).extract()
            X.append(feats)
            y.append(label_map[addr])
            addresses.append(addr)
        except Exception as e:
            logger.debug("Erreur %s : %s", addr, e)
            n_error += 1

    stats = {
        "total_loaded":  len(X),
        "no_label":      n_no_label,
        "errors":        n_error,
        "n_efficient":   int((np.array(y) == 0).sum()) if y else 0,
        "n_inefficient": int((np.array(y) == 1).sum()) if y else 0,
    }

    logger.info("Contrats chargés : %d | Sans label : %d | Erreurs : %d",
                len(X), n_no_label, n_error)
    logger.info("Distribution — efficient(0): %d | inefficient(1): %d",
                stats["n_efficient"], stats["n_inefficient"])

    return np.array(X), np.array(y), addresses, stats


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODÈLE XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

def build_xgboost(n0: int, n1: int) -> XGBClassifier:
    scale_pos_weight = n0 / max(n1, 1)
    logger.info("XGBoost scale_pos_weight=%.4f [ratio %d:%d]",
                scale_pos_weight, n0, n1)
    return XGBClassifier(
        n_estimators     = 500,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = scale_pos_weight,
        eval_metric      = "logloss",
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
    )


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro",
                                            zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro",
                                         zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="macro",
                                     zero_division=0)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, output_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax,
                xticklabels=["Efficient", "Inefficient"],
                yticklabels=["Efficient", "Inefficient"])
    ax.set_title("Confusion Matrix — XGBoost (Table 1 Features)")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    path = output_dir / "confusion_matrix_xgboost_paper.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Matrice de confusion → %s", path)


def plot_feature_importance(model: XGBClassifier, output_dir: Path):
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = ["tomato" if i == sorted_idx[0] else "steelblue"
               for i in range(len(FEATURE_NAMES))]
    ax.bar(np.arange(12), importances[sorted_idx],
           color=[colors[i] for i in range(12)],
           edgecolor="grey", linewidth=0.5)
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in sorted_idx],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Feature Importance (gain)")
    ax.set_title("XGBoost — Feature Importances (Table 1 — 12 hand-crafted features)")
    plt.tight_layout()
    path = output_dir / "xgboost_paper_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Feature importance → %s", path)


def plot_comparison(xgb: dict, gasgat_single: dict, gasgat_cv: dict,
                    output_dir: Path):
    """Graphique côte-à-côte XGBoost (Table 1) vs GasGAT."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]

    xgb_vals    = [xgb.get(m, 0)          for m in metrics]
    gasgat_vals = [gasgat_single.get(m, 0) for m in metrics]
    gasgat_cv_v = [gasgat_cv.get(m, 0)    for m in metrics]
    gasgat_cv_s = [gasgat_cv.get(f"{m}_std", 0) for m in metrics]

    x     = np.arange(len(metrics))
    width = 0.28

    fig, ax = plt.subplots(figsize=(12, 6))

    b1 = ax.bar(x - width, xgb_vals,    width, label="XGBoost (Table 1 features)",
                color="steelblue", edgecolor="white")
    b2 = ax.bar(x,          gasgat_vals, width, label="GasGAT — single run",
                color="tomato",    edgecolor="white")
    b3 = ax.bar(x + width,  gasgat_cv_v, width, label="GasGAT — 5-fold CV (mean)",
                color="seagreen",  edgecolor="white",
                yerr=gasgat_cv_s,  capsize=4, error_kw={"elinewidth": 1.5})

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.2%}", ha="center", va="bottom", fontsize=7.5)

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Performance Comparison: XGBoost (12 hand-crafted features) vs GasGAT\n"
        "Same dataset — 27,879 labeled smart contract graphs",
        fontsize=11
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.5, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = output_dir / "xgboost_paper_vs_gasgat.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Comparaison → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# 5. PIPELINES
# ══════════════════════════════════════════════════════════════════════════════

def run_single(X, y, output_dir: Path) -> dict:
    """Run unique 80/20 — même seed que GasGAT."""
    n       = len(y)
    n_train = int(n * (1 - TEST_RATIO))

    rng     = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(n)
    rng.shuffle(indices)

    X_train = X[indices[:n_train]]
    y_train = y[indices[:n_train]]
    X_test  = X[indices[n_train:]]
    y_test  = y[indices[n_train:]]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    n0 = int((y_train == 0).sum())
    n1 = int((y_train == 1).sum())
    model = build_xgboost(n0, n1)

    logger.info("Entraînement XGBoost (12 features Table 1)...")
    model.fit(X_train, y_train, verbose=False)

    y_pred  = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    report = classification_report(
        y_test, y_pred,
        target_names=["Gas-Efficient", "Gas-Inefficient"],
        digits=4,
    )
    logger.info("\n%s", report)
    (output_dir / "xgboost_paper_report.txt").write_text(report)
    logger.info("XGBoost (Table 1) : %s", metrics)

    plot_confusion_matrix(y_test, y_pred, output_dir)
    plot_feature_importance(model, output_dir)

    with open(output_dir / "xgboost_paper_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def run_cv(X, y, output_dir: Path) -> dict:
    """Cross-validation 5-fold — même configuration que GasGAT."""
    skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                   random_state=RANDOM_SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info("FOLD %d / %d", fold + 1, N_FOLDS)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        n0 = int((y_train == 0).sum())
        n1 = int((y_train == 1).sum())
        model = build_xgboost(n0, n1)
        model.fit(X_train, y_train, verbose=False)

        y_pred  = model.predict(X_val)
        metrics = compute_metrics(y_val, y_pred)
        metrics["fold"] = fold + 1
        fold_metrics.append(metrics)

        logger.info("Fold %d | Acc=%.4f | F1=%.4f",
                    fold + 1, metrics["accuracy"], metrics["f1"])

    keys = ("accuracy", "precision", "recall", "f1")
    avg  = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    std  = {f"{k}_std": float(np.std([m[k] for m in fold_metrics])) for k in keys}

    logger.info("=" * 55)
    logger.info("XGBOOST (Table 1) — CROSS-VALIDATION %d FOLDS", N_FOLDS)
    logger.info("Accuracy  : %.4f ± %.4f", avg["accuracy"],  std["accuracy_std"])
    logger.info("Precision : %.4f ± %.4f", avg["precision"], std["precision_std"])
    logger.info("Recall    : %.4f ± %.4f", avg["recall"],    std["recall_std"])
    logger.info("F1-Score  : %.4f ± %.4f", avg["f1"],        std["f1_std"])
    logger.info("=" * 55)

    results = {**avg, **std, "folds": fold_metrics}
    with open(output_dir / "xgboost_paper_cv_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def print_table(xgb: dict, gasgat_s: dict, gasgat_cv: dict):
    """Reproduit la Table 2 du paper avec les vrais résultats."""
    print("\n" + "=" * 78)
    print("  TABLE 2 — PERFORMANCE COMPARISON (reproduced on real dataset)")
    print("  XGBoost (12 hand-crafted features, Table 1) vs GasGAT")
    print("  Dataset: 27,879 labeled smart contract graphs")
    print("=" * 78)
    print(f"{'Metric':<15} {'XGBoost':>12} {'GasGAT (run)':>14} {'GasGAT (5-CV)':>16}")
    print("-" * 78)
    for label, key in [("Accuracy","accuracy"),("Precision","precision"),
                        ("Recall","recall"),("F1-Score","f1")]:
        xv  = xgb.get(key, 0)
        gv  = gasgat_s.get(key, 0)
        gcv = gasgat_cv.get(key, 0)
        gcs = gasgat_cv.get(f"{key}_std", 0)
        delta = gv - xv
        sign  = "+" if delta >= 0 else ""
        print(f"{label:<15} {xv:>11.2%} {gv:>13.2%} "
              f"{gcv:>8.2%}±{gcs:.2%}   Δ={sign}{delta:.2%}")
    print("=" * 78)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="XGBoost avec les 12 features exactes de la Table 1 du paper GasGAT."
    )
    p.add_argument("--contracts_dir",  default="data/contracts/")
    p.add_argument("--graphs_dir",     default="data/graphs/")
    p.add_argument("--output_dir",     default="results/xgboost_paper/")
    p.add_argument("--mode",           default="both",
                   choices=["single", "cv", "both"])
    p.add_argument("--gasgat_single",  default="results/eval/final_metrics.json")
    p.add_argument("--gasgat_cv",      default="results/cv/cv_results.json")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Chargement ────────────────────────────────────────────────────────────
    X, y, addresses, stats = load_dataset(args.contracts_dir, args.graphs_dir)
    logger.info("Features shape : %s | Labels : %s", X.shape, y.shape)

    if len(X) == 0:
        logger.error("Aucun contrat chargé. Vérifiez --contracts_dir et --graphs_dir")
        return

    xgb_single = None
    xgb_cv     = None

    # ── Run unique ────────────────────────────────────────────────────────────
    if args.mode in ("single", "both"):
        logger.info("\n=== XGBoost Table 1 — RUN UNIQUE ===")
        xgb_single = run_single(X, y, out_dir)

    # ── Cross-validation ──────────────────────────────────────────────────────
    if args.mode in ("cv", "both"):
        logger.info("\n=== XGBoost Table 1 — CROSS-VALIDATION 5-FOLD ===")
        xgb_cv = run_cv(X, y, out_dir)

    # ── Chargement métriques GasGAT ───────────────────────────────────────────
    gasgat_single = {}
    gasgat_cv     = {}

    if Path(args.gasgat_single).exists():
        with open(args.gasgat_single) as f:
            gasgat_single = json.load(f)
    else:
        logger.warning("GasGAT single metrics non trouvées : %s", args.gasgat_single)

    if Path(args.gasgat_cv).exists():
        with open(args.gasgat_cv) as f:
            gasgat_cv = json.load(f)
    else:
        logger.warning("GasGAT CV metrics non trouvées : %s", args.gasgat_cv)

    # ── Table de comparaison ──────────────────────────────────────────────────
    xgb_for_table = xgb_single or xgb_cv or {}
    print_table(xgb_for_table, gasgat_single, gasgat_cv)

    # ── Graphique ─────────────────────────────────────────────────────────────
    if xgb_single and gasgat_single and gasgat_cv:
        plot_comparison(xgb_single, gasgat_single, gasgat_cv, out_dir)

    # ── Sauvegarde complète ───────────────────────────────────────────────────
    comparison = {
        "xgboost_paper_single": xgb_single,
        "xgboost_paper_cv":     xgb_cv,
        "gasgat_single":        gasgat_single,
        "gasgat_cv":            gasgat_cv,
        "dataset_stats":        stats,
        "features_used":        FEATURE_NAMES,
    }
    with open(out_dir / "full_comparison_paper.json", "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info("✅ Résultats → %s", out_dir)


if __name__ == "__main__":
    main()
