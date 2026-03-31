"""
slither_labeler.py
==================
Relabellisation correcte des smart contracts avec Slither.
Reproduit exactement la méthodologie du paper GasGAT :
  - Score d'inefficacité basé sur les détecteurs Slither
  - Labels : top 25% = inefficient (1), bottom 25% = efficient (0)
  - Middle 50% = ambiguous (-1), exclu du dataset polarisé

Détecteurs Slither utilisés (gas-related) :
  - costly-loop          : opérations coûteuses dans des boucles
  - reentrancy-eth       : reentrancy avec transfert d'ETH
  - reentrancy-no-eth    : reentrancy sans ETH
  - calls-loop           : appels externes dans des boucles
  - unused-return        : retours non utilisés
  - tx-origin            : usage de tx.origin
  - uninitialized-local  : variables locales non initialisées
  - write-after-write    : écriture après écriture inutile

Installation :
    pip install slither-analyzer
    pip install solc-select
    solc-select install 0.8.0
    solc-select use 0.8.0

Usage :
    # Étape 1 : analyser tous les contrats avec Slither
    python slither_labeler.py --step analyze \\
        --contracts_dir data/contracts/ \\
        --output_dir    data/slither_results/

    # Étape 2 : calculer les scores et générer les labels
    python slither_labeler.py --step label \\
        --slither_dir   data/slither_results/ \\
        --output_file   data/slither_labels.json

    # Étape 3 : régénérer les graphes avec les nouveaux labels
    python slither_labeler.py --step rebuild_graphs \\
        --contracts_dir data/contracts/ \\
        --labels_file   data/slither_labels.json \\
        --output_dir    data/graphs_slither/

Requirements :
    pip install slither-analyzer solc-select tqdm pandas
"""

import os
import re
import json
import argparse
import logging
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("slither_labeler.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Poids des détecteurs Slither (gas-related) ────────────────────────────────
# Plus le poids est élevé, plus le pattern est gas-intensif
DETECTOR_WEIGHTS = {
    # Haute priorité gas
    "costly-loop":            5.0,   # SSTORE/external calls in loops
    "calls-loop":             4.0,   # external calls in loops
    "reentrancy-eth":         3.0,   # reentrancy avec ETH
    "reentrancy-no-eth":      2.0,   # reentrancy sans ETH
    # Moyenne priorité gas
    "tx-origin":              2.0,   # tx.origin usage
    "unused-return":          1.5,   # retours non utilisés
    "write-after-write":      1.5,   # double écriture inutile
    "uninitialized-local":    1.0,   # variables non initialisées
    # Basse priorité gas
    "dead-code":              1.0,   # code mort
    "constable-states":       0.5,   # variables d'état pouvant être const
    "immutable-states":       0.5,   # variables pouvant être immutable
    "boolean-equality":       0.3,   # comparaison booléenne explicite
    "low-level-calls":        1.0,   # appels bas niveau
    "delegatecall-loop":      4.0,   # delegatecall dans boucle
    "msg-value-loop":         2.0,   # msg.value dans boucle
    "storage-array":          2.0,   # tableau de storage dans boucle
    "tautology":              0.3,   # tautologies
    "incorrect-equality":     0.5,   # égalité incorrecte
}

# Seuils de labelling (comme dans le paper)
LOW_THRESHOLD  = 0.25   # bottom 25% → efficient (0)
HIGH_THRESHOLD = 0.75   # top 25%    → inefficient (1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ANALYSE SLITHER
# ══════════════════════════════════════════════════════════════════════════════

def check_slither_installed() -> bool:
    """Vérifie que Slither est installé."""
    try:
        result = subprocess.run(
            ["slither", "--version"],
            capture_output=True, text=True, timeout=10
        )
        version = result.stdout.strip() or result.stderr.strip()
        logger.info("Slither détecté : %s", version)
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.error(
            "Slither non trouvé.\n"
            "Installation :\n"
            "  pip install slither-analyzer\n"
            "  pip install solc-select\n"
            "  solc-select install 0.8.0\n"
            "  solc-select use 0.8.0"
        )
        return False


def check_solc_installed() -> bool:
    """Vérifie que solc est installé."""
    try:
        result = subprocess.run(
            ["solc", "--version"],
            capture_output=True, text=True, timeout=10
        )
        logger.info("solc détecté : %s", result.stdout.split('\n')[1].strip())
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning(
            "solc non trouvé. Installation via solc-select :\n"
            "  pip install solc-select\n"
            "  solc-select install 0.8.17\n"
            "  solc-select use 0.8.17"
        )
        return False


def extract_solidity_version(source: str) -> str:
    """Extrait la version Solidity depuis le pragma."""
    match = re.search(r'pragma\s+solidity\s+([^;]+);', source)
    if not match:
        return "0.8.17"
    version_str = match.group(1).strip()
    # Extraire x.y.z
    ver_match = re.search(r'(\d+\.\d+\.\d+)', version_str)
    if ver_match:
        return ver_match.group(1)
    # Extraire x.y
    ver_match = re.search(r'(\d+\.\d+)', version_str)
    if ver_match:
        return ver_match.group(1) + ".0"
    return "0.8.17"


def extract_main_source(raw: str) -> str:
    """Extrait le code Solidity depuis le format Etherscan Standard JSON."""
    s = raw.strip()
    if s.startswith("{{") or s.startswith('{"language"'):
        try:
            clean = s[1:-1] if s.startswith("{{") else s
            data  = json.loads(clean)
            parts = []
            for fname, content in data.get("sources", {}).items():
                code = content.get("content", "")
                if "contract " in code and not fname.startswith("@"):
                    parts.append(f"// === {fname} ===\n{code}")
            if parts:
                return "\n\n".join(parts)
        except (json.JSONDecodeError, AttributeError):
            pass
    return raw


def run_slither(sol_path: Path, timeout: int = 60) -> dict:
    """
    Lance Slither sur un fichier .sol et retourne les détections.

    Returns:
        dict avec clés : 'detectors', 'error', 'success'
    """
    cmd = [
        "slither", str(sol_path),
        "--json", "-",                  # output JSON sur stdout
        "--disable-color",
        "--filter-paths", "node_modules",
        "--solc-remaps", "@openzeppelin=node_modules/@openzeppelin",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=sol_path.parent,
        )

        # Slither retourne 0 (succès) ou 1 (détections trouvées) — les deux sont OK
        if result.stdout:
            try:
                data = json.loads(result.stdout)
                detectors = data.get("results", {}).get("detectors", [])
                return {"success": True, "detectors": detectors, "error": None}
            except json.JSONDecodeError:
                pass

        return {"success": False, "detectors": [], "error": result.stderr[:200]}

    except subprocess.TimeoutExpired:
        return {"success": False, "detectors": [], "error": "timeout"}
    except Exception as e:
        return {"success": False, "detectors": [], "error": str(e)[:200]}


def analyze_contracts(contracts_dir: str, output_dir: str,
                       limit: int = None, resume: bool = True):
    """
    Étape 1 : Analyse tous les contrats avec Slither.
    Sauvegarde les résultats JSON par contrat.
    """
    if not check_slither_installed():
        return

    contracts_path = Path(contracts_dir)
    out_path       = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sol_files = sorted(contracts_path.glob("*.sol"))
    if limit:
        sol_files = sol_files[:limit]

    # Résumé
    done = {f.stem for f in out_path.glob("*.json")} if resume else set()
    if done:
        logger.info("Déjà analysés : %d — reprise...", len(done))

    n_ok      = 0
    n_error   = 0
    n_timeout = 0

    pbar = tqdm(sol_files, desc="Analyse Slither")
    for sol_file in pbar:
        addr = sol_file.stem

        if addr in done:
            continue

        # Créer un fichier temporaire avec le code source extrait
        try:
            raw    = sol_file.read_text(encoding="utf-8", errors="ignore")
            source = extract_main_source(raw)
            ver    = extract_solidity_version(source)

            # Écrire dans un fichier temporaire propre
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sol",
                dir=str(out_path), delete=False,
                prefix=f"{addr}_"
            ) as tmp:
                tmp.write(source)
                tmp_path = Path(tmp.name)

        except Exception as e:
            logger.debug("Lecture échouée %s : %s", addr, e)
            n_error += 1
            continue

        # Lancer Slither
        result = run_slither(tmp_path)
        tmp_path.unlink(missing_ok=True)   # nettoyage

        # Sauvegarder le résultat
        result["address"] = addr
        result["version"] = ver
        result_path = out_path / f"{addr}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        if result["success"]:
            n_ok += 1
        elif result["error"] == "timeout":
            n_timeout += 1
        else:
            n_error += 1

        pbar.set_postfix(ok=n_ok, err=n_error, timeout=n_timeout)

    logger.info("Analyse terminée | OK=%d | Erreurs=%d | Timeouts=%d",
                n_ok, n_error, n_timeout)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CALCUL DES SCORES ET LABELS
# ══════════════════════════════════════════════════════════════════════════════

def compute_inefficiency_score(detectors: list) -> float:
    """
    Calcule le score d'inefficacité gas à partir des détections Slither.
    Score = somme pondérée des détections normalisée sur [0, 1].
    """
    if not detectors:
        return 0.0

    raw_score = 0.0
    for detection in detectors:
        check_name = detection.get("check", "")
        # Pondération selon le détecteur
        weight     = DETECTOR_WEIGHTS.get(check_name, 0.5)
        # Pondération selon l'impact
        impact     = detection.get("impact", "Informational")
        impact_mult = {"High": 2.0, "Medium": 1.5, "Low": 1.0,
                       "Informational": 0.3, "Optimization": 0.8}.get(impact, 1.0)
        raw_score += weight * impact_mult

    # Normalisation sigmoid-like : score max raisonnable ≈ 20
    normalized = min(raw_score / 20.0, 1.0)
    return normalized


def assign_label(score: float, low: float = LOW_THRESHOLD,
                 high: float = HIGH_THRESHOLD) -> int:
    """
    Assigne un label basé sur le percentile du score.
    Returns: 0 (efficient), 1 (inefficient), -1 (ambiguous)
    """
    if score <= low:
        return 0
    elif score >= high:
        return 1
    return -1


def generate_labels(slither_dir: str, output_file: str):
    """
    Étape 2 : Calcule les scores et génère les labels depuis les résultats Slither.
    Applique la stratégie de labelling du paper (top/bottom 25%).
    """
    slither_path = Path(slither_dir)
    result_files = sorted(slither_path.glob("*.json"))

    logger.info("Calcul des scores depuis %d résultats Slither...", len(result_files))

    records = []
    n_failed = 0
    for rf in tqdm(result_files, desc="Scoring"):
        try:
            with open(rf) as f:
                data = json.load(f)
        except Exception:
            continue

        # ── FILTRE CRITIQUE : ignorer les contrats où Slither a échoué ──────
        # Un échec Slither (import manquant, etc.) donne detectors=[]
        # ce qui produirait un score=0 et un label "efficient" artificiel.
        if not data.get("success", False):
            n_failed += 1
            continue

        addr      = data.get("address", rf.stem)
        detectors = data.get("detectors", [])
        score     = compute_inefficiency_score(detectors)

        # Détails des détections
        detection_summary = defaultdict(int)
        for d in detectors:
            detection_summary[d.get("check", "unknown")] += 1

        records.append({
            "address":           addr,
            "score":             score,
            "n_detections":      len(detectors),
            "detection_summary": dict(detection_summary),
            "slither_success":   data.get("success", False),
        })

    logger.info("Contrats échoués ignorés : %d / %d", n_failed, len(result_files))

    if not records:
        logger.error("Aucun résultat Slither valide trouvé dans %s", slither_dir)
        logger.info("Vérifiez que l\'analyse Slither a réussi sur au moins quelques contrats.")
        return

    df = pd.DataFrame(records)

    # Calcul des seuils par percentile strict (comme dans le paper)
    # On s'assure d'avoir exactement 25% par classe en utilisant rank
    low_threshold  = df["score"].quantile(LOW_THRESHOLD)
    high_threshold = df["score"].quantile(HIGH_THRESHOLD)

    logger.info("Score distribution :")
    logger.info("  Min    : %.4f", df["score"].min())
    logger.info("  25%%ile : %.4f (seuil efficient)", low_threshold)
    logger.info("  Median : %.4f", df["score"].median())
    logger.info("  75%%ile : %.4f (seuil inefficient)", high_threshold)
    logger.info("  Max    : %.4f", df["score"].max())

    # Assignation des labels via rank strict (method='first' brise les ex-aequo)
    # Garantit exactement 25%/50%/25% même si beaucoup de scores sont identiques
    df["rank_pct"] = df["score"].rank(pct=True, method="first")

    df["label"] = -1   # middle 50% → ambiguous par défaut
    df.loc[df["rank_pct"] <= LOW_THRESHOLD,  "label"] = 0   # bottom 25% → efficient
    df.loc[df["rank_pct"] > HIGH_THRESHOLD,  "label"] = 1   # top 25%    → inefficient

    # Statistiques
    n0    = (df["label"] == 0).sum()
    n1    = (df["label"] == 1).sum()
    n_amb = (df["label"] == -1).sum()

    logger.info("\nDistribution des labels :")
    logger.info("  Efficient (0)   : %d (%.1f%%)", n0, 100*n0/len(df))
    logger.info("  Inefficient (1) : %d (%.1f%%)", n1, 100*n1/len(df))
    logger.info("  Ambiguous (-1)  : %d (%.1f%%)", n_amb, 100*n_amb/len(df))

    # Sauvegarde
    label_dict = df.set_index("address")[["score", "label",
                                          "n_detections"]].to_dict(orient="index")

    with open(output_file, "w") as f:
        json.dump({
            "thresholds": {
                "low":  float(low_threshold),
                "high": float(high_threshold),
            },
            "stats": {
                "n_efficient":   int(n0),
                "n_inefficient": int(n1),
                "n_ambiguous":   int(n_amb),
                "total":         len(df),
            },
            "labels": label_dict,
        }, f, indent=2)

    # Sauvegarde CSV pour inspection
    csv_path = Path(output_file).with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    logger.info("Labels sauvegardés → %s", output_file)
    logger.info("CSV inspection   → %s", csv_path)

    return label_dict


# ══════════════════════════════════════════════════════════════════════════════
# 3. RÉGÉNÉRATION DES GRAPHES AVEC LES NOUVEAUX LABELS
# ══════════════════════════════════════════════════════════════════════════════

def rebuild_graphs(contracts_dir: str, labels_file: str, output_dir: str):
    """
    Étape 3 : Met à jour les labels dans les graphes .pt existants.
    Utilise les nouveaux labels Slither au lieu des labels heuristiques.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        logger.error("pip install torch torch-geometric")
        return

    # Charger les labels Slither
    with open(labels_file) as f:
        data = json.load(f)

    label_map  = {addr: info["label"] for addr, info in data["labels"].items()}
    thresholds = data.get("thresholds", {})

    logger.info("Labels Slither chargés : %d contrats", len(label_map))
    logger.info("Seuils utilisés — low=%.4f | high=%.4f",
                thresholds.get("low", LOW_THRESHOLD),
                thresholds.get("high", HIGH_THRESHOLD))

    # Chercher les graphes existants
    contracts_path = Path(contracts_dir)
    out_path       = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Chercher les graphes existants dans le dossier parent de contracts_dir
    # ou dans graph_builder.py par défaut (data/graphs/)
    default_graphs_dir = contracts_path.parent / "graphs"
    if default_graphs_dir.exists():
        existing_graphs = sorted(default_graphs_dir.glob("*.pt"))
    else:
        existing_graphs = []

    if not existing_graphs:
        logger.warning("Aucun graphe .pt trouvé dans %s", default_graphs_dir)
        logger.info(
            "Lancez d'abord : python graph_builder.py "
            "--contracts_dir %s --output_dir %s/graphs/",
            contracts_dir, contracts_path.parent
        )
        return

    n_updated  = 0
    n_skipped  = 0
    n_ambig    = 0

    pbar = tqdm(existing_graphs, desc="Mise à jour labels")
    for pt_file in pbar:
        addr = pt_file.stem.lower()

        if addr not in label_map:
            n_skipped += 1
            continue

        new_label = label_map[addr]

        try:
            graph = torch.load(pt_file, weights_only=False)

            # Mettre à jour le label
            graph.y              = torch.tensor([new_label], dtype=torch.long)
            graph.label_source   = "slither"

            # Sauvegarder dans le nouveau dossier
            torch.save(graph, out_path / f"{addr}.pt")
            n_updated += 1

            if new_label == -1:
                n_ambig += 1

        except Exception as e:
            logger.debug("Erreur %s : %s", addr, e)
            n_skipped += 1

        pbar.set_postfix(updated=n_updated, skipped=n_skipped)

    n_labeled = n_updated - n_ambig
    logger.info("=" * 55)
    logger.info("Graphes mis à jour : %d", n_updated)
    logger.info("  → Labellisés (0 ou 1) : %d", n_labeled)
    logger.info("  → Ambigus (-1)        : %d", n_ambig)
    logger.info("  → Ignorés (pas Slither): %d", n_skipped)
    logger.info("Sortie → %s", out_path.resolve())
    logger.info("=" * 55)
    logger.info("\nProchaine étape :")
    logger.info("  python gasgat_model.py --graphs_dir %s --mode train", out_path)
    logger.info("  python xgboost_paper_features.py --graphs_dir %s --mode both", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 4. INSTALLATION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def print_install_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  INSTALLATION DE SLITHER                                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Installer Slither et solc-select :                           ║
║     pip install slither-analyzer solc-select                     ║
║                                                                  ║
║  2. Installer le compilateur Solidity 0.8.x :                    ║
║     solc-select install 0.8.17                                   ║
║     solc-select use 0.8.17                                       ║
║                                                                  ║
║  3. Vérifier l'installation :                                    ║
║     slither --version                                            ║
║     solc --version                                               ║
║                                                                  ║
║  4. Lancer l'analyse :                                           ║
║     python slither_labeler.py --step analyze \\                  ║
║         --contracts_dir data/contracts/ \\                       ║
║         --output_dir data/slither_results/                       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ══════════════════════════════════════════════════════════════════════════════
# 5. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Relabellisation des smart contracts avec Slither (méthodologie GasGAT paper)."
    )
    p.add_argument("--step", required=True,
                   choices=["analyze", "label", "rebuild_graphs", "install"],
                   help=(
                       "analyze       : analyser les contrats avec Slither\n"
                       "label         : calculer scores et labels\n"
                       "rebuild_graphs: mettre à jour les labels dans les .pt\n"
                       "install       : afficher les instructions d'installation"
                   ))
    p.add_argument("--contracts_dir", default="data/contracts/")
    p.add_argument("--slither_dir",   default="data/slither_results/")
    p.add_argument("--output_dir",    default="data/slither_results/")
    p.add_argument("--output_file",   default="data/slither_labels.json")
    p.add_argument("--labels_file",   default="data/slither_labels.json")
    p.add_argument("--graphs_out",    default="data/graphs_slither/")
    p.add_argument("--limit",         type=int, default=None,
                   help="Limiter le nombre de contrats analysés")
    p.add_argument("--no_resume",     action="store_true",
                   help="Recommencer depuis zéro")
    return p.parse_args()


def main():
    args = parse_args()

    if args.step == "install":
        print_install_instructions()
        return

    elif args.step == "analyze":
        if not check_slither_installed():
            print_install_instructions()
            return
        logger.info("=== ÉTAPE 1 : Analyse Slither ===")
        logger.info("Contrats : %s", args.contracts_dir)
        logger.info("Sortie   : %s", args.output_dir)
        analyze_contracts(
            contracts_dir = args.contracts_dir,
            output_dir    = args.output_dir,
            limit         = args.limit,
            resume        = not args.no_resume,
        )

    elif args.step == "label":
        logger.info("=== ÉTAPE 2 : Calcul des scores et labels ===")
        generate_labels(
            slither_dir = args.slither_dir,
            output_file = args.output_file,
        )

    elif args.step == "rebuild_graphs":
        logger.info("=== ÉTAPE 3 : Régénération des graphes ===")
        rebuild_graphs(
            contracts_dir = args.contracts_dir,
            labels_file   = args.labels_file,
            output_dir    = args.graphs_out,
        )


if __name__ == "__main__":
    main()
