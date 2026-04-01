# GasGAT: A Graph Attention Network Framework for Smart Contract Gas Optimization

[![PeerJ Computer Science](https://img.shields.io/badge/PeerJ-Computer%20Science-blue)](https://peerj.com/computer-science/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-green)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Dataset: Zenodo](https://img.shields.io/badge/Dataset-Zenodo-lightblue)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## 📖 Description

**GasGAT** is a deep learning framework that models Ethereum smart contracts as **semantic graphs** and applies a **Graph Attention Network (GAT)** to detect gas-intensive code patterns. Unlike rule-based static analysis tools, GasGAT captures **non-local, inter-procedural dependencies** that drive gas inefficiency — patterns invisible to traditional heuristics.

> This repository accompanies the paper:
> **"GasGAT: A Graph Attention Network Framework for Smart Contract Gas Optimization"**
> Youssef Said, Al Mahdi Khaddar, Lahcen Hassine, Ahmed Eddaoui, Tarik Chafiq
> *Submitted to PeerJ Computer Science, 2025*

### Key Contributions

- **Semantic graph representation** of smart contracts derived from Solidity ASTs
- **Graph Attention Network** with multi-head attention for non-local pattern learning
- **Interpretable attention maps** identifying which inter-procedural dependencies cause gas inefficiency
- Evaluation on **40,000 real-world Ethereum smart contracts** (Solidity ≥ 0.8.0, ≥ 50 transactions)
- Labels generated via **Slither static analysis** (independent of model features)
- **5-fold cross-validation**: Accuracy 94.92% ±1.04%, F1 88.69% ±2.70%, McNemar = 633.12 (p ≪ 0.05)

---

## 📂 Repository Structure

```
GasGAT/
│
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
│
├── etherscan_downloader.py         # Download verified contracts from Etherscan API
├── address_fetcher.py              # Collect contract addresses from public sources
├── graph_builder.py                # Solidity source → Semantic graph (.pt)
├── slither_labeler.py              # Slither-based labeling pipeline (3 steps)
├── gasgat_model.py                 # GasGAT architecture + training + CV + evaluation
├── gnn_baselines.py                # GCN / GraphSAGE / GIN baselines (same protocol)
├── xgboost_paper_features.py       # XGBoost baseline (12 hand-crafted features, Table 1)
│
└── data/
    └── codebook.csv                # Codebook: numerical codes → categorical labels
```

---

## 📊 Dataset

### Collection
- **Source**: Ethereum mainnet via [Etherscan API V2](https://etherscan.io/apis)
- **Filter**: Solidity version ≥ 0.8.0, ≥ 50 confirmed transactions
- **Total collected**: 40,000 verified smart contracts

### Labeling (Slither-based, independent of model features)

| Label | Code | Criterion |
|-------|------|-----------|
| Gas-efficient | 0 | Bottom 25% of Slither inefficiency score |
| Gas-inefficient | 1 | Top 25% of Slither inefficiency score |
| Ambiguous | -1 | Middle 50% — excluded from primary task |

### Final labeled dataset (Slither labels)

| Gas-efficient (0) | Gas-inefficient (1) | Total labeled |
|-------------------|---------------------|---------------|
| 2,794             | 2,795               | 5,589         |

### Codebook (categorical variables)

| Variable | Code | Label |
|----------|------|-------|
| Label | 0 | Gas-efficient |
| Label | 1 | Gas-inefficient |
| Label | -1 | Ambiguous (excluded) |
| NodeType | 0 | Function node |
| NodeType | 1 | Control-flow node (loop, conditional) |
| NodeType | 2 | State variable node |
| NodeType | 3 | Operation node |
| EdgeType | 0 | Function call dependency |
| EdgeType | 1 | Control-flow relation |
| EdgeType | 2 | State read/write access |
| EdgeType | 3 | Inter-procedural dependency |

### Access

The full dataset (40,000 contracts + labeled graphs) is publicly available on Zenodo:
📦 **DOI**: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## 💻 Scripts

### 1. `etherscan_downloader.py` — Contract collection

```bash
python etherscan_downloader.py \
    --addresses_file addresses.txt \
    --output_dir     data/contracts/ \
    --limit          40000
```

### 2. `address_fetcher.py` — Address collection

```bash
python address_fetcher.py --source bigquery --output addresses.txt
```

### 3. `graph_builder.py` — Semantic graph construction

Parses Solidity AST and builds directed semantic graphs (PyTorch Geometric `.pt` format).

```bash
python graph_builder.py \
    --contracts_dir data/contracts/ \
    --output_dir    data/graphs/ \
    --workers       4
```

### 4. `slither_labeler.py` — Slither-based labeling (3 steps)

```bash
# Step 1: Analyze contracts with Slither
python slither_labeler.py --step analyze \
    --contracts_dir data/contracts/ \
    --output_dir    data/slither_results/

# Step 2: Compute scores and labels (strict 25/50/25 distribution)
python slither_labeler.py --step label \
    --slither_dir   data/slither_results/ \
    --output_file   data/slither_labels.json

# Step 3: Rebuild graphs with Slither labels
python slither_labeler.py --step rebuild_graphs \
    --labels_file   data/slither_labels.json \
    --graphs_out    data/graphs_slither/
```

### 5. `gasgat_model.py` — GasGAT training & evaluation

```bash
# 5-fold cross-validation
python gasgat_model.py \
    --graphs_dir data/graphs_slither/ \
    --mode       cv \
    --epochs     200 \
    --output_dir results/cv/

# Evaluation with attention visualization
python gasgat_model.py \
    --graphs_dir  data/graphs_slither/ \
    --mode        eval \
    --model_path  results/cv/gasgat_best_fold2.pt \
    --output_dir  results/eval/ \
    --visualize
```

### 6. `gnn_baselines.py` — GCN / GraphSAGE / GIN comparison

Trains all three baselines with the **exact same protocol** as GasGAT (same graphs, same splits, same seed).

```bash
python gnn_baselines.py \
    --graphs_dir data/graphs_slither/ \
    --output_dir results/gnn_comparison/ \
    --gasgat_cv  results/cv/cv_results.json
```

### 7. `xgboost_paper_features.py` — XGBoost baseline (Table 1)

Uses the 12 hand-crafted features from Table 1 of the paper, extracted directly from `.sol` source files.

```bash
python xgboost_paper_features.py \
    --contracts_dir data/contracts/ \
    --graphs_dir    data/graphs_slither/ \
    --output_dir    results/xgboost/
```

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/Ysaid7810/Youssef.git
cd Youssef

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Slither and solc
pip install slither-analyzer solc-select
solc-select install 0.8.17
solc-select use 0.8.17

# 4. Download dataset from Zenodo (graphs ready-to-use)
# https://doi.org/10.5281/zenodo.XXXXXXX
# → unzip GasGAT_semantic_graphs.zip into data/graphs_slither/

# 5. Train GasGAT (5-fold CV)
python gasgat_model.py \
    --graphs_dir data/graphs_slither/ \
    --mode cv \
    --epochs 200 \
    --output_dir results/cv/
```

---

## ⚙️ Requirements

```
Python           >= 3.9
PyTorch          >= 2.0.0
PyTorch Geometric >= 2.3.0
slither-analyzer >= 0.11.0
solc-select      >= 0.9.0
xgboost          >= 1.7.0
scikit-learn     >= 1.2.0
pandas           >= 1.5.0
numpy            >= 1.23.0
matplotlib       >= 3.6.0
seaborn          >= 0.12.0
tqdm             >= 4.65.0
requests         >= 2.31.0
huggingface_hub  >= 1.0.0
```

```bash
pip install -r requirements.txt
```

### Computing Infrastructure

| Component | Specification |
|-----------|---------------|
| OS | macOS 14 (Apple Silicon) |
| CPU | Apple M-series (10 cores) |
| RAM | 16 GB unified memory |
| Device | CPU (MPS incompatible with `scatter_reduce` in PyG) |
| Storage | ~50 GB (full dataset + graphs) |

> **Apple Silicon note**: `gasgat_model.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` automatically. All training runs on CPU.

---

## 📈 Results

### GasGAT — 5-fold Cross-Validation (Slither labels)

| Metric | Mean | ± Std |
|--------|------|-------|
| Accuracy | **94.92%** | ±1.04% |
| Precision | **96.94%** | ±0.43% |
| Recall | **83.77%** | ±3.46% |
| F1-Score | **88.69%** | ±2.70% |
| McNemar | **633.12** | p ≪ 0.05 |

### Per-Fold Breakdown

| Fold | Accuracy | F1 | Best Epoch |
|------|----------|----|------------|
| 1 | 94.21% | 86.88% | 6 |
| 2 | **96.52%** | **92.72%** | 7 |
| 3 | 95.52% | 90.33% | 5 |
| 4 | 94.84% | 88.60% | 7 |
| 5 | 93.51% | 84.92% | 11 |

### Attention Weight Case Study

Contract `0xe5b2b3038e9e5fb42be2100ceed784874ea4634b` (Gas-Inefficient):
- 2,026 inter-node edges out of 2,583 total
- **Node 127** = hub function: 7 outgoing edges with α > 0.96
- **Top edge**: node 105 → node 115, α = **0.9965**

This inter-procedural pattern — a loop delegating to state-modifying helpers causing cascading `SSTORE` operations — is invisible to local static analysis but captured by GasGAT's attention mechanism.

---

## 🔬 Evaluation Methodology

| Method | Purpose |
|--------|---------|
| 5-fold stratified CV (seed=42) | Robust performance estimation |
| McNemar's test vs majority baseline | Statistical significance |
| Attention weight visualization | Interpretability validation |
| Error analysis (local baseline failures) | Evidence against label circularity |
| Extended experiments (3-class, regression) | Generalization beyond polarized setting |
| Cross-version evaluation (Solidity 0.5–0.7) | Version generalizability |

---

## 📧 Contact

**Corresponding author:** Youssef Said
📧 `youssef.said3-etu@etu.univh2c.ma`
🏛️ Information Processing Laboratory (LTI), Univh2c, Casablanca, Morocco
