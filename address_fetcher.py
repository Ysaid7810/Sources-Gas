"""
address_fetcher.py
==================
Génère une liste massive d'adresses de smart contracts Ethereum vérifiés.

Sources disponibles :
  clone     → Clone Smart Contract Sanctuary (~200MB, 500K+ adresses) RECOMMANDÉ
  etherscan → Scrape Etherscan contractsVerified (~5K adresses)
  disl      → HuggingFace ASSERT-KTH/DISL (514K contrats)
  csv       → Fichier CSV externe (BigQuery, Dune Analytics...)
  bigquery  → Affiche les instructions SQL Google BigQuery
  dune      → Affiche les instructions SQL Dune Analytics

Usage:
    python address_fetcher.py --source clone      --output addresses.txt
    python address_fetcher.py --source etherscan  --output addresses.txt
    python address_fetcher.py --source disl       --output addresses.txt
    python address_fetcher.py --source csv        --csv_file bq_result.csv
    python address_fetcher.py --source bigquery
    python address_fetcher.py --source dune

Requirements:
    pip install requests pandas tqdm
"""

import re
import time
import argparse
import logging
import subprocess
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 : Clone Smart Contract Sanctuary (RECOMMANDÉE)
# Repo : tintinweb/smart-contract-sanctuary-ethereum
# Fichiers nommés : <address>_<ContractName>.sol
# ══════════════════════════════════════════════════════════════════════════════

def fetch_from_clone(clone_dir: str = "sanctuary_clone", limit: int = 200_000) -> list:
    """
    Clone le Smart Contract Sanctuary en mode sparse (noms de fichiers seulement)
    et extrait les adresses depuis les noms de fichiers .sol.

    Format des fichiers : <40-char-hex>_ContractName.sol
    → adresse = "0x" + les 40 premiers caractères hex du nom de fichier
    """
    repo_url  = "https://github.com/tintinweb/smart-contract-sanctuary-ethereum.git"
    clone_path = Path(clone_dir)

    if not clone_path.exists():
        logger.info("Clonage du Sanctuary (sparse, pas de contenu — juste les noms)...")
        logger.info("Repo : %s", repo_url)

        # Initialise le repo sparse
        subprocess.run(["git", "init", str(clone_path)], check=True, capture_output=True)
        subprocess.run(["git", "remote", "add", "origin", repo_url],
                       cwd=clone_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "core.sparseCheckout", "true"],
                       cwd=clone_path, check=True, capture_output=True)

        # On veut uniquement les noms de fichiers, pas leur contenu
        sparse_file = clone_path / ".git" / "info" / "sparse-checkout"
        sparse_file.write_text("contracts/mainnet/\n")

        logger.info("Téléchargement des métadonnées (peut prendre 2-5 min)...")
        result = subprocess.run(
            ["git", "fetch", "--depth=1", "origin", "master"],
            cwd=clone_path, capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            logger.error("Fetch échoué : %s", result.stderr[:500])
            logger.info("Alternative : utilisez --source bigquery ou --source etherscan")
            return []

        subprocess.run(
            ["git", "checkout", "master"],
            cwd=clone_path, capture_output=True, timeout=300
        )
        logger.info("Clone terminé → %s", clone_path.resolve())
    else:
        logger.info("Repo déjà présent : %s", clone_path.resolve())

    # Extraction des adresses depuis les noms de fichiers
    mainnet_dir = clone_path / "contracts" / "mainnet"
    if not mainnet_dir.exists():
        logger.error("Dossier introuvable : %s", mainnet_dir)
        logger.info("Le clone n'a peut-être pas récupéré les fichiers.")
        logger.info("Essayez : git -C %s checkout master", clone_dir)
        return []

    addr_re   = re.compile(r'^([0-9a-fA-F]{40})', re.IGNORECASE)
    addresses = set()
    sol_files = list(mainnet_dir.rglob("*.sol"))
    logger.info("Fichiers .sol trouvés : %d", len(sol_files))

    for f in tqdm(sol_files, desc="Extraction adresses"):
        m = addr_re.match(f.stem)
        if m:
            addresses.add("0x" + m.group(1).lower())
        if len(addresses) >= limit:
            break

    logger.info("Adresses extraites : %d", len(addresses))
    return list(addresses)


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 : Etherscan contractsVerified (scraping, sans clé API)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_from_etherscan(limit: int = 10_000) -> list:
    """Scrape la liste des contrats récemment vérifiés sur Etherscan."""
    logger.info("Scraping Etherscan contractsVerified (max ~5000 adresses)...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        )
    }
    addr_re   = re.compile(r'0x[a-fA-F0-9]{40}')
    addresses = set()

    for page in tqdm(range(1, 501), desc="Pages Etherscan"):
        if len(addresses) >= limit:
            break
        url = f"https://etherscan.io/contractsVerified/{page}?ps=100"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 429:
                logger.warning("Rate limit — pause 30s")
                time.sleep(30)
                continue
            if resp.status_code != 200:
                break
            for addr in addr_re.findall(resp.text):
                if len(addr) == 42:
                    addresses.add(addr.lower())
            time.sleep(1.2)
        except Exception as e:
            logger.warning("Page %d : %s", page, e)
            time.sleep(3)

    logger.info("Adresses récupérées : %d", len(addresses))
    return list(addresses)[:limit]


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3 : DISL (HuggingFace)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_from_disl(limit: int = 200_000) -> list:
    """Télécharge les adresses depuis le dataset DISL (HuggingFace ASSERT-KTH)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("pip install huggingface_hub")
        return []

    logger.info("Téléchargement DISL (HuggingFace ASSERT-KTH/DISL)...")
    for filename in ["metadata.csv", "metadata.parquet", "train.csv", "data.csv"]:
        try:
            path = hf_hub_download(
                repo_id="ASSERT-KTH/DISL",
                filename=filename,
                repo_type="dataset",
            )
            df = pd.read_parquet(path) if filename.endswith(".parquet") else pd.read_csv(path)
            logger.info("Fichier : %s | Colonnes : %s", filename, df.columns.tolist())
            col = next((c for c in df.columns if "address" in c.lower()), None)
            if col:
                addrs = df[col].dropna().str.strip().str.lower().tolist()
                addrs = [a for a in addrs if a.startswith("0x") and len(a) == 42]
                logger.info("Adresses DISL : %d", len(addrs))
                return addrs[:limit]
        except Exception:
            continue

    logger.warning("DISL : aucun fichier compatible trouvé")
    return []


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 4 : CSV externe
# ══════════════════════════════════════════════════════════════════════════════

def fetch_from_csv(csv_file: str, address_col: str = "address") -> list:
    """Charge des adresses depuis un fichier CSV (BigQuery, Dune, etc.)."""
    path = Path(csv_file)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_file}")
    df  = pd.read_csv(path)
    col = next((c for c in df.columns if address_col.lower() in c.lower()), None)
    if col is None:
        raise ValueError(f"Colonne '{address_col}' introuvable. "
                         f"Colonnes disponibles : {df.columns.tolist()}")
    addrs = df[col].dropna().str.strip().str.lower().tolist()
    addrs = [a for a in addrs if a.startswith("0x") and len(a) == 42]
    logger.info("Adresses chargées depuis %s : %d", csv_file, len(addrs))
    return addrs


# ══════════════════════════════════════════════════════════════════════════════
# INSTRUCTIONS SQL
# ══════════════════════════════════════════════════════════════════════════════

BIGQUERY_SQL = """
╔══════════════════════════════════════════════════════════╗
║  GOOGLE BIGQUERY — gratuit, 1TB/mois offert              ║
║  https://console.cloud.google.com/bigquery               ║
╚══════════════════════════════════════════════════════════╝

SELECT DISTINCT c.address
FROM `bigquery-public-data.crypto_ethereum.contracts` AS c
JOIN `bigquery-public-data.crypto_ethereum.transactions` AS t
  ON t.to_address = c.address
WHERE c.block_timestamp >= TIMESTAMP('2021-11-01')
GROUP BY c.address
HAVING COUNT(t.hash) >= 50
LIMIT 200000;

Après exécution → SAVE RESULTS → CSV (local file)
Puis : python address_fetcher.py --source csv --csv_file bq_result.csv
"""

DUNE_SQL = """
╔══════════════════════════════════════════════════════════╗
║  DUNE ANALYTICS — gratuit                                ║
║  https://dune.com/queries  (créez un compte)             ║
╚══════════════════════════════════════════════════════════╝

SELECT DISTINCT "to" AS address
FROM ethereum.transactions
WHERE block_time >= DATE '2021-11-01'
  AND "to" IS NOT NULL
  AND success = true
GROUP BY "to"
HAVING COUNT(*) >= 50
LIMIT 200000;

Après exécution → Download CSV
Puis : python address_fetcher.py --source csv --csv_file dune_result.csv
"""


# ══════════════════════════════════════════════════════════════════════════════
# SAUVEGARDE
# ══════════════════════════════════════════════════════════════════════════════

def save_addresses(addresses: list, output_file: str) -> list:
    out   = Path(output_file)
    clean = list(dict.fromkeys(
        a.lower() for a in addresses
        if isinstance(a, str) and a.startswith("0x") and len(a) == 42
    ))
    out.write_text("\n".join(clean))
    logger.info("=" * 55)
    logger.info("✅  %d adresses → %s", len(clean), out)
    logger.info("=" * 55)
    logger.info("Prochaine étape :")
    logger.info("  python etherscan_downloader.py \\")
    logger.info("      --addresses_file %s --limit 40000", out)
    return clean


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Génère une liste d'adresses de smart contracts Ethereum vérifiés.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--source", default="clone",
        choices=["clone", "etherscan", "disl", "csv", "bigquery", "dune"],
        help=(
            "clone     : clone Sanctuary GitHub (~200MB, 500K+ adresses)  ← RECOMMANDÉ\n"
            "etherscan : scrape contractsVerified (~5K adresses)\n"
            "disl      : HuggingFace ASSERT-KTH/DISL (514K contrats)\n"
            "csv       : fichier CSV externe (BigQuery, Dune...)\n"
            "bigquery  : affiche les instructions SQL BigQuery\n"
            "dune      : affiche les instructions SQL Dune Analytics"
        ),
    )
    p.add_argument("--output",      default="addresses.txt",
                   help="Fichier de sortie (défaut: addresses.txt)")
    p.add_argument("--limit",       type=int, default=200_000,
                   help="Nombre max d'adresses (défaut: 200000)")
    p.add_argument("--clone_dir",   default="sanctuary_clone",
                   help="Dossier local pour le clone git (défaut: sanctuary_clone)")
    p.add_argument("--csv_file",    default=None,
                   help="Fichier CSV source (requis pour --source csv)")
    p.add_argument("--address_col", default="address",
                   help="Nom de la colonne adresse dans le CSV (défaut: address)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.source == "bigquery":
        print(BIGQUERY_SQL)
        return

    if args.source == "dune":
        print(DUNE_SQL)
        return

    addresses = []

    if args.source == "clone":
        addresses = fetch_from_clone(
            clone_dir=args.clone_dir,
            limit=args.limit,
        )

    elif args.source == "etherscan":
        addresses = fetch_from_etherscan(limit=min(args.limit, 10_000))

    elif args.source == "disl":
        addresses = fetch_from_disl(limit=args.limit)

    elif args.source == "csv":
        if not args.csv_file:
            logger.error("--csv_file requis avec --source csv")
            return
        addresses = fetch_from_csv(args.csv_file, args.address_col)

    if not addresses:
        logger.error("Aucune adresse récupérée.")
        logger.info("")
        logger.info("Options :")
        logger.info("  python address_fetcher.py --source bigquery  (SQL gratuit)")
        logger.info("  python address_fetcher.py --source dune      (SQL gratuit)")
        logger.info("  python address_fetcher.py --source etherscan (scraping)")
        return

    save_addresses(addresses[:args.limit], args.output)


if __name__ == "__main__":
    main()
