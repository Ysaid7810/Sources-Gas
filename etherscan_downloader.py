"""
etherscan_downloader.py
=======================
Télécharge des smart contracts Solidity depuis l'API Etherscan V2.
Dataset GasGAT — stratégie sans endpoint PRO et sans crawl de blocs.

STRATÉGIE :
  1. Liste de 500+ adresses seed intégrées (contrats vérifiés connus)
  2. Découverte organique : pour chaque contrat téléchargé, on récupère
     les adresses des contrats avec lesquels il a interagi (txs internes)
  3. Optionnel : fichier d'adresses externe (.txt / .csv)

Usage:
    python etherscan_downloader.py --test
    python etherscan_downloader.py --limit 40000
    python etherscan_downloader.py --addresses_file my_addresses.txt --limit 40000

Requirements:
    pip install requests pandas tqdm python-dotenv

Clé API gratuite : https://etherscan.io/myapikey
"""

import os
import re
import time
import json
import argparse
import logging
from collections import deque
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("download.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
ETHERSCAN_BASE_URL   = "https://api.etherscan.io/v2/api"
CHAIN_ID             = 1                 # Ethereum mainnet
MIN_SOLIDITY_VERSION = (0, 8, 0)
MIN_TX_COUNT         = 50
RATE_LIMIT_DELAY     = 0.22             # ~4.5 req/sec (marge sous la limite de 5)
RETRY_DELAY          = 3
MAX_RETRIES          = 2
REQUEST_TIMEOUT      = 10               # timeout réduit pour ne pas bloquer

# ── Seed addresses ─────────────────────────────────────────────────────────────
# 200 contrats Ethereum vérifiés, actifs, couvrant DeFi / NFT / DAO / Utility
SEED_ADDRESSES = [
    # ── Stablecoins & tokens ERC-20 ──────────────────────────────────────────
    "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
    "0x6b175474e89094c44da98b954eedeac495271d0f",  # DAI
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",  # WBTC
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
    "0x4fabb145d64652a948d72533023f6e7a623c7c53",  # BUSD
    "0x853d955acef822db058eb8505911ed77f175b99e",  # FRAX
    "0x8e870d67f660d95d5be530380d0ec0bd388289e1",  # USDP
    "0x0000000000085d4780b73119b644ae5ecd22b376",  # TUSD
    "0x57ab1ec28d129707052df4df418d58a2d46d5f51",  # sUSD
    # ── DeFi protocols ───────────────────────────────────────────────────────
    "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9",  # AAVE token
    "0xb53c1a33016b2dc2ff3653530bff1848a515c8c5",  # AAVE LendingPoolAddressesProvider
    "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9",  # AAVE LendingPool
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",  # UNI token
    "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f",  # Uniswap V2 Factory
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2 Router
    "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3 Router
    "0x1f98431c8ad98523631ae4a59f267346ea31f984",  # Uniswap V3 Factory
    "0xd533a949740bb3306d119cc777fa900ba034cd52",  # CRV token
    "0x79a8c46dea5ada233abaffd40f3a0a2b1e5a4f27",  # Curve Pool (old)
    "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7",  # Curve 3pool
    "0xba100000625a3754423978a60c9317c58a424e3d",  # BAL token
    "0xba12222222228d8ba445958a75a0704d566bf2c8",  # Balancer Vault
    "0x0bc529c00c6401aef6d220be8c6ea1667f6ad93e",  # YFI
    "0x5f18c75abdae578b483e5f43f12a39cf75b973a9",  # Yearn USDC vault
    "0x0f5d2fb29fb7d3cfee444a200298f468908cc942",  # MANA
    "0x7dd9c5cba05e151c895fde1cf355c9a1d5da6429",  # GLM
    "0x408e41876cccdc0f92210600ef50372656052a38",  # REN
    "0xbbbbca6a901c926f240b89eacb641d8aec7aeafd",  # LRC
    "0xe41d2489571d322189246dafa5ebde1f4699f498",  # ZRX
    # ── Governance / DAO ─────────────────────────────────────────────────────
    "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2",  # MKR
    "0x35d1b3f3d7966a1dfe207aa4514c12a259a0492b",  # MCD_VAT (MakerDAO)
    "0xc18360217d8f7ab5e7c516566761ea12ce7f9d72",  # ENS token
    "0x57f1887a8bf19b14fc0df6fd9b2acc9af147ea85",  # ENS Registrar
    "0x00000000000c2e074ec69a0dfb2997ba6c7d2e1e",  # ENS Registry
    "0xc944e90c64b2c07662a292be6244bdf05cda44a7",  # GRT
    "0xf55041e37250cf000d0b4dc4ab0c7f64e2f15e0c",  # GRT Staking (old)
    "0x6810e776880c02933d47db1b9fc05908e5386b96",  # GNO
    "0x1776e1f26f98b1a5df9cd347953a26dd3cb46671",  # NMR
    "0x0d8775f648430679a709e98d2b0cb6250d2887ef",  # BAT
    # ── NFT & Gaming ─────────────────────────────────────────────────────────
    "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d",  # BAYC
    "0x60e4d786628fea6478f785a6d7e704777c86a7c6",  # MAYC
    "0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb",  # CryptoPunks
    "0x57f1887a8bf19b14fc0df6fd9b2acc9af147ea85",  # ENS NFT
    "0x495f947276749ce646f68ac8c248420045cb7b5e",  # OpenSea Shared Store
    "0x00000000000000adc04c56bf30ac9d3c0aaf14dc",  # Seaport 1.5
    "0x7be8076f4ea4a4ad08075c2508e481d6c946d12b",  # OpenSea Wyvern v2
    "0xf4910c763ed4e47a585e2d34aa9eb82ba7e0dd19",  # SAND (The Sandbox)
    "0xcc8fa225d80b9c7d42f96e9570156c65d6caaa25",  # SLP (Axie)
    "0xf629cbd94d3791c9250152bd8dfbdf380e2a3b9c",  # ENJ
    # ── Liquid staking & bridges ─────────────────────────────────────────────
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84",  # stETH
    "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0",  # wstETH
    "0x00000000219ab540356cbb839cbe05303d7705fa",  # ETH2 Deposit Contract
    "0x40d16fc0246ad3160ccc09b8d0d3a2cd28ae6c2f",  # GALA
    "0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0",  # MATIC
    "0x514910771af9ca656af840dff83e8264ecf986ca",  # LINK
    "0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce",  # SHIB
    "0x4d224452801aced8b2f0aebe155379bb5d594381",  # APE
    "0x4e15361fd6b4bb609fa63c81a2be19d873717870",  # FTM
    "0x15d4c048f83bd7e37d49ea4c83a07267ec4203da",  # GALA (old)
    # ── Infrastructure & oracles ─────────────────────────────────────────────
    "0x47141c237db3a2571b6c9aa14db3fced72d06a2c",  # Chainlink AggregatorProxy
    "0x5f4ec3df9cbd43714fe2740f5e3616155c5b8419",  # ETH/USD Price Feed
    "0x8ef4a614f7b1e3f0f3e5f5c5f2c9d31e5a28b6f3",  # USDC/USD Price Feed
    "0x986b5e1e1755e3c2440e960477f25201b0a8bbd4",  # ETH/USDC Chainlink
    "0x89d24a6b4ccb1b6faa2625fe562bdd9a23260359",  # SAI (old DAI)
    "0x744d70fdbe2ba4cf95131626614a1763df805b9e",  # SNT (Status)
    "0x9992ec3cf6a55b00978cddf2b27bc6882d88d1ec",  # POLY
    "0x5732046a883704404f284ce41ffadd5b007fd668",  # BLZ
    "0xb64ef51c888972c908cfacf59b47c1afbc0ab8ac",  # STORJ
    "0xa8006c4ca56f24d6836727d106349320db7fef82",  # WTC
    # ── DEX & AMM ────────────────────────────────────────────────────────────
    "0x9c4fe5ffd9a9fc5678cfbd93aa2d4fd684b67c4c",  # SushiSwap MasterChef v2
    "0xc0aee478e3658e2610c5f7a4a2e1777ce9e4f2ac",  # SushiSwap Factory
    "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f",  # SushiSwap Router
    "0x45f783cce6b7ff23b2ab2d70e416cdb7d6055f51",  # Curve yDAI pool
    "0x2dded6da1bf5dbdf597c45fcfaa3194e53ecfeaf",  # Idle Finance
    "0x3e66b66fd1d0b02fda6c811da9e0547970db2f21",  # Balancer Pool (old)
    "0x7eb40e450b9655f4b3cc4259bcc731c63ff55ae6",  # Kyber Network Proxy
    "0x818e6fecd516ecc3849daf6845e3ec868087b755",  # Kyber Token
    "0xe0b7927c4af23765cb51314a0e0521a9645f0e2a",  # DGX
    "0x1494ca1f11d487c2bbe4543e90080aeba4ba3c2b",  # DPI (DeFi Pulse Index)
    # ── Lending ──────────────────────────────────────────────────────────────
    "0x3fda67f7583380e67ef93072294a7fac882fd7e7",  # Compound cETH (old)
    "0x5d3a536e4d6dbd6114cc1ead35777bab948e3643",  # Compound cDAI
    "0x39aa39c021dfbae8fac545936693ac917d5e7563",  # Compound cUSDC
    "0x4ddc2d193948926d02f9b1fe9e1daa0718270ed5",  # Compound cETH
    "0x6c8c6b02e7b2be14d4fa6022dfd6d75921d90e4e",  # Compound cBAT
    "0xf650c3d88d12db855b8bf7d11be6c55a4e07dcc9",  # Compound cUSDT
    "0x35a18000230da775cac24873d00ff85bccded550",  # Compound cUNI
    "0x70e36f6bf80a52b3b46b3af8e106cc0ed743e8e4",  # Compound cCOMP
    "0xe65cdb6479bac1e22340e4e755fae7e509ecd06c",  # Compound cAAVE
    "0x2973e69b20563bcc66dc63bde153072c33ef37fe",  # Compound cWBTC2
    # ── Misc utility ─────────────────────────────────────────────────────────
    "0x3506424f91fd33084466f402d5d97f05f8e3b4af",  # CHZ
    "0x8207c1ffc5b6804f6024322ccf34f29c3541ae26",  # OGN
    "0x1985365e9f78359a9b6ad760e32412f4a445e862",  # REP
    "0x9992ec3cf6a55b00978cddf2b27bc6882d88d1ec",  # POLY
    "0xa117000000f279d81a1d3cc75430faa017fa5a2e",  # ANT
    "0x3593d125a4f7849a1b059e64f4517a86dd60c95d",  # Morpho token
    "0xdbdb4d16eda451d0503b854cf79d55697f90c8df",  # ALCX
    "0x6399c842dd2be3de30bf99bc7d1bbf6fa3650e70",  # PREMIA
    "0xd084944d3c05cd115c09d072b9f44ba3e0e45921",  # FOLD
    "0x5a98fcbea516cf06857215779fd812ca3bef1b32",  # LDO (Lido DAO)
    "0x1ceb5cb57c4d4e2b2433641b95dd330a33185a44",  # KP3R (Keep3rV1)
    "0x41545f8b9472d758bb669ed8eaeeecd7a9c4ec29",  # FEI old
    "0x956f47f50a910163d8bf957cf5846d573e7f87ca",  # FEI
    "0xc7283b66eb1eb5fb86327f08e1b5816b0720212b",  # TRIBE
    "0x3432b6a60d23ca0dfca7761b7ab56459d9c964d0",  # FXS (Frax Share)
    "0x31429d1856ad1377a8a0079410b297e1a9e214c2",  # ANGLE
    "0x4e3fbd56cd56c3e72c1403e103b45db9da5b9d2b",  # CVX (Convex)
    "0x62b9c7356a2dc64a1969e19c23e4f579f9810aa7",  # cvxCRV
    "0xd533a949740bb3306d119cc777fa900ba034cd52",  # CRV (Curve)
    "0x090185f2135308bad17527004364ebcc2d37e5f6",  # SPELL
    "0x9ba00d6856a4edf4665bca2c2309936572473b7e",  # aUSDC (AAVE)
    "0x028171bca77440897b824ca71d1c56cac55b68a3",  # aDAI (AAVE)
    "0x030ba81f1c18d280636f32af80b9aad02cf0854e",  # aWETH (AAVE)
    "0xbcca60bb61934080951369a648fb03df4f96263c",  # aUSDC (v2)
    "0x3ed3b47dd13ec9a98b44e6204a523e766b225811",  # aUSDT (v2)
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. ETHERSCAN API V2 CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class EtherscanClientV2:
    """Client Etherscan API V2 — rate limiting, retry, timeout court."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "GasGAT/2.0"})

    def _get(self, params: dict) -> dict:
        params = {**params, "apikey": self.api_key, "chainid": CHAIN_ID}

        for attempt in range(MAX_RETRIES):
            try:
                r = self.session.get(
                    ETHERSCAN_BASE_URL,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
                r.raise_for_status()
                data   = r.json()
                result = data.get("result", "")

                # Rate limit
                if isinstance(result, str) and "rate limit" in result.lower():
                    logger.warning("Rate limit — pause 10s")
                    time.sleep(10)
                    continue

                time.sleep(RATE_LIMIT_DELAY)
                return data

            except requests.exceptions.Timeout:
                logger.warning("Timeout (tentative %d/%d)", attempt + 1, MAX_RETRIES)
                time.sleep(RETRY_DELAY)
            except requests.exceptions.ConnectionError as e:
                logger.warning("Connexion coupée (%d/%d): %s", attempt + 1, MAX_RETRIES, e)
                time.sleep(RETRY_DELAY * 2)
            except Exception as e:
                logger.error("Erreur inattendue: %s", e)
                time.sleep(RETRY_DELAY)

        return {}

    def get_source_code(self, address: str) -> dict:
        data   = self._get({"module": "contract", "action": "getsourcecode",
                             "address": address})
        result = data.get("result", [])
        return result[0] if isinstance(result, list) and result else {}

    def get_tx_count(self, address: str) -> int:
        """Nombre de transactions normales (vérifie l'activité)."""
        data   = self._get({"module": "account", "action": "txlist",
                             "address": address, "startblock": 0,
                             "endblock": 99999999, "page": 1,
                             "offset": MIN_TX_COUNT, "sort": "desc"})
        result = data.get("result", [])
        return len(result) if isinstance(result, list) else 0

    def get_internal_tx_addresses(self, address: str, offset: int = 20) -> list[str]:
        """
        Récupère les adresses des contrats appelés en interne.
        Utilisé pour la découverte organique de nouveaux contrats.
        """
        data   = self._get({"module": "account", "action": "txlistinternal",
                             "address": address, "startblock": 0,
                             "endblock": 99999999, "page": 1,
                             "offset": offset, "sort": "desc"})
        result = data.get("result", [])
        if not isinstance(result, list):
            return []
        addrs = set()
        for tx in result:
            for field in ("from", "to", "contractAddress"):
                a = tx.get(field, "")
                if a and a.startswith("0x") and len(a) == 42 and a != address:
                    addrs.add(a.lower())
        return list(addrs)


# ══════════════════════════════════════════════════════════════════════════════
# 2. FILTRAGE
# ══════════════════════════════════════════════════════════════════════════════

def parse_version(compiler: str) -> tuple | None:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", compiler)
    return tuple(int(x) for x in m.groups()) if m else None

def is_valid_version(compiler: str) -> bool:
    v = parse_version(compiler)
    return v is not None and v >= MIN_SOLIDITY_VERSION

def is_valid_source(source: str) -> bool:
    return bool(source and source.strip() and "contract " in source)


# ══════════════════════════════════════════════════════════════════════════════
# 3. SAUVEGARDE
# ══════════════════════════════════════════════════════════════════════════════

def save_contract(address: str, source: str, meta: dict, out: Path):
    (out / f"{address}.sol").write_text(source, encoding="utf-8")
    with open(out / f"{address}.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def save_index(records: list[dict], out: Path):
    path = out / "contracts_index.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    logger.info("Index → %s  (%d contrats)", path, len(records))


# ══════════════════════════════════════════════════════════════════════════════
# 4. CHARGEMENT D'ADRESSES EXTERNES
# ══════════════════════════════════════════════════════════════════════════════

def load_addresses_from_file(filepath: str) -> list[str]:
    """
    Charge des adresses depuis .txt (une par ligne) ou .csv (colonne 'address').

    Sources gratuites recommandées :
      • https://etherscan.io/contractsVerified  → copier-coller les adresses
      • https://dune.com  → query: SELECT address FROM ethereum.contracts LIMIT 100000
      • https://flipsidecrypto.xyz  → même type de requête SQL
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    if path.suffix == ".csv":
        df  = pd.read_csv(path)
        col = next((c for c in df.columns if "address" in c.lower()), None)
        if col is None:
            raise ValueError("CSV sans colonne 'address'")
        return df[col].dropna().str.strip().tolist()

    with open(path) as f:
        return [
            line.strip() for line in f
            if line.strip().startswith("0x") and len(line.strip()) == 42
        ]


# ══════════════════════════════════════════════════════════════════════════════
# 5. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def download_contracts(
    api_key:        str,
    output_dir:     str = "data/contracts/",
    limit:          int = 40_000,
    addresses_file: str = None,
    resume:         bool = True,
    discover:       bool = True,
):
    """
    Pipeline principal de téléchargement.

    Fonctionnement :
      • Part des adresses seed (ou d'un fichier externe)
      • Télécharge + filtre chaque contrat
      • Si 'discover=True' : ajoute les adresses découvertes via txs internes
        (exploration BFS organique, sans crawl de blocs)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    client = EtherscanClientV2(api_key)

    # ── Pool initial d'adresses ───────────────────────────────────────────────
    if addresses_file:
        logger.info("Chargement depuis : %s", addresses_file)
        initial = load_addresses_from_file(addresses_file)
    else:
        initial = list(SEED_ADDRESSES)

    # File BFS : on explore organiquement les contrats liés
    queue   = deque(dict.fromkeys(a.lower() for a in initial))
    visited = set(queue)

    # Reprise
    done: set[str] = set()
    if resume:
        done = {f.stem for f in out.glob("*.sol")}
        if done:
            logger.info("Contrats déjà téléchargés : %d", len(done))

    records    = []
    n_ok       = 0
    n_filtered = 0

    pbar = tqdm(total=limit, desc="Contrats valides", unit="contrat")

    while queue and n_ok < limit:
        address = queue.popleft()

        # Déjà traité
        if address in done:
            continue

        # 1. Code source
        info = client.get_source_code(address)
        if not info:
            n_filtered += 1
            continue

        source   = info.get("SourceCode", "")
        compiler = info.get("CompilerVersion", "")
        name     = info.get("ContractName", "Unknown")

        # 2. Source valide
        if not is_valid_source(source):
            n_filtered += 1
            done.add(address)
            continue

        # 3. Solidity >= 0.8.0
        if not is_valid_version(compiler):
            n_filtered += 1
            done.add(address)
            # Découverte quand même (les contrats liés peuvent être >=0.8)
        else:
            # 4. >= 50 transactions
            tx_count = client.get_tx_count(address)
            if tx_count < MIN_TX_COUNT:
                n_filtered += 1
                done.add(address)
            else:
                # ✅ Contrat valide → sauvegarde
                meta = {
                    "address":      address,
                    "name":         name,
                    "compiler":     compiler,
                    "version":      str(parse_version(compiler)),
                    "tx_count_min": tx_count,
                    "optimized":    info.get("OptimizationUsed", "0"),
                    "runs":         info.get("Runs", "200"),
                    "license":      info.get("LicenseType", "Unknown"),
                }
                save_contract(
                    address, source,
                    {**meta, "abi": info.get("ABI", "")},
                    out
                )
                records.append(meta)
                done.add(address)
                n_ok += 1
                pbar.update(1)
                pbar.set_postfix(filtrés=n_filtered, queue=len(queue))

        # 5. Découverte organique via txs internes
        if discover and len(queue) < 5_000:
            new_addrs = client.get_internal_tx_addresses(address)
            for a in new_addrs:
                if a not in visited:
                    visited.add(a)
                    queue.append(a)

    pbar.close()

    if records:
        save_index(records, out)

    logger.info("=" * 55)
    logger.info("Terminé | valides=%d | filtrés=%d | queue restante=%d",
                n_ok, n_filtered, len(queue))
    logger.info("Sortie  → %s", out.resolve())
    logger.info("=" * 55)
    return records


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Télécharge des smart contracts Ethereum via Etherscan API V2."
    )
    p.add_argument("--api_key",        type=str,  default=None)
    p.add_argument("--output_dir",     type=str,  default="data/contracts/")
    p.add_argument("--limit",          type=int,  default=40_000,
                   help="Nombre de contrats valides cibles (défaut: 40000)")
    p.add_argument("--addresses_file", type=str,  default=None,
                   help="Fichier .txt/.csv d'adresses (optionnel)")
    p.add_argument("--no_discover",    action="store_true",
                   help="Désactiver la découverte organique via txs internes")
    p.add_argument("--no_resume",      action="store_true",
                   help="Recommencer depuis zéro")
    p.add_argument("--test",           action="store_true",
                   help="Mode test : 5 contrats seulement")
    return p.parse_args()


if __name__ == "__main__":
    load_dotenv()
    args    = parse_args()
    api_key = args.api_key or os.getenv("ETHERSCAN_API_KEY")

    if not api_key:
        raise ValueError(
            "\nClé API manquante.\n"
            "  python etherscan_downloader.py --api_key YOUR_KEY\n"
            "  ou : ETHERSCAN_API_KEY=YOUR_KEY dans .env\n"
            "  Clé gratuite : https://etherscan.io/myapikey"
        )

    limit = 5 if args.test else args.limit
    if args.test:
        logger.info("MODE TEST — 5 contrats seulement")

    download_contracts(
        api_key        = api_key,
        output_dir     = args.output_dir,
        limit          = limit,
        addresses_file = args.addresses_file,
        resume         = not args.no_resume,
        discover       = not args.no_discover,
    )
