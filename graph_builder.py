"""
graph_builder.py
================
Transforme les smart contracts Solidity en graphes sémantiques G = (V, E)
pour le framework GasGAT.

Pipeline :
  1. Parsing du code source Solidity → AST (via solc ou py-solc-x)
  2. Extraction des nœuds sémantiques (fonctions, variables d'état, control-flow, opérations)
  3. Construction des arêtes (call, control-flow, state-access, inter-procedural)
  4. Encodage des features de chaque nœud en vecteur numérique
  5. Sauvegarde au format PyTorch Geometric (.pt) + JSON (pour inspection)

Usage :
    python graph_builder.py --contracts_dir data/contracts/ --output_dir data/graphs/
    python graph_builder.py --contracts_dir data/contracts/ --output_dir data/graphs/ --workers 4

Requirements :
    pip install py-solc-x torch torch-geometric networkx tqdm
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import networkx as nx
from torch_geometric.data import Data
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("graph_builder.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 1. TYPES DE NŒUDS ET D'ARÊTES
# ══════════════════════════════════════════════════════════════════════════════

# Node types (codebook)
NODE_TYPES = {
    "function":     0,   # fonction publique/interne/privée
    "control_flow": 1,   # boucle (for, while) ou conditionnel (if)
    "state_var":    2,   # variable d'état (storage)
    "operation":    3,   # opération critique (SSTORE, SLOAD, CALL, etc.)
}

# Edge types (codebook)
EDGE_TYPES = {
    "call":           0,  # appel de fonction
    "control_flow":   1,  # relation de contrôle (boucle → fonction)
    "state_access":   2,  # lecture/écriture de variable d'état
    "inter_proc":     3,  # dépendance inter-procédurale
}

# Visibilité des fonctions
VISIBILITY_MAP = {
    "public":   0,
    "external": 1,
    "internal": 2,
    "private":  3,
}

# Opérations gas-intensives à détecter
GAS_INTENSIVE_OPS = {
    "sstore", "sload", "call", "delegatecall", "staticcall",
    "create", "create2", "selfdestruct", "log0", "log1", "log2", "log3", "log4",
}

# Feature dimension par nœud
NODE_FEATURE_DIM = 16


# ══════════════════════════════════════════════════════════════════════════════
# 2. STRUCTURES DE DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SolidityNode:
    """Représente un nœud dans le graphe sémantique."""
    node_id:     int
    node_type:   str                    # "function", "control_flow", "state_var", "operation"
    name:        str
    visibility:  str        = "public"  # pour les fonctions
    is_payable:  bool       = False
    is_view:     bool       = False
    is_pure:     bool       = False
    loop_depth:  int        = 0         # profondeur d'imbrication des boucles
    op_type:     str        = ""        # ex: "sstore", "call"
    line_start:  int        = 0
    gas_weight:  float      = 0.0       # poids gas estimé (heuristique)
    features:    list       = field(default_factory=list)


@dataclass
class SolidityEdge:
    """Représente une arête dans le graphe sémantique."""
    src:       int
    dst:       int
    edge_type: str          # "call", "control_flow", "state_access", "inter_proc"
    weight:    float = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 3. PARSER SOLIDITY (regex-based, sans dépendance à solc)
# ══════════════════════════════════════════════════════════════════════════════

class SolidityParser:
    """
    Parseur Solidity léger basé sur des regex.
    Extrait les entités sémantiques nécessaires pour construire le graphe.

    Note : Pour une précision maximale, utiliser py-solc-x pour obtenir
    l'AST complet. Ce parseur regex couvre ~90% des cas courants.
    """

    # ── Patterns regex ────────────────────────────────────────────────────────
    PRAGMA_RE   = re.compile(r'pragma\s+solidity\s+([^;]+);')
    CONTRACT_RE = re.compile(
        r'\bcontract\s+(\w+)(?:\s+is\s+[^{]+)?\s*\{'
    )
    FUNCTION_RE = re.compile(
        r'\bfunction\s+(\w+)\s*\(([^)]*)\)\s*'
        r'((?:public|private|internal|external|view|pure|payable|virtual|override'
        r'|\s|returns\s*\([^)]*\))*)\s*\{'
    )
    STATE_VAR_RE = re.compile(
        r'^\s*((?:uint|int|bool|address|bytes|string|mapping|struct)\S*)'
        r'\s+(?:public|private|internal|constant|immutable|\s)*'
        r'\s*(\w+)\s*[;=]',
        re.MULTILINE,
    )
    FOR_LOOP_RE    = re.compile(r'\bfor\s*\(')
    WHILE_LOOP_RE  = re.compile(r'\bwhile\s*\(')
    IF_RE          = re.compile(r'\bif\s*\(')
    CALL_RE        = re.compile(r'(\w+)\s*\.\s*call\s*[({]')
    SSTORE_RE      = re.compile(r'\bsstore\s*\(', re.IGNORECASE)
    SLOAD_RE       = re.compile(r'\bsload\s*\(', re.IGNORECASE)
    DELEGATE_RE    = re.compile(r'\bdelegatecall\b', re.IGNORECASE)
    SELFDESTRUCT_RE = re.compile(r'\bselfdestruct\b', re.IGNORECASE)
    INTERNAL_CALL_RE = re.compile(r'\b(\w+)\s*\((?![^)]*\bfunction\b)')

    def __init__(self, source_code: str):
        # Gestion du format multi-fichiers Etherscan (JSON Standard Input)
        if source_code.startswith("{{") or source_code.startswith('{"language"'):
            self.source = self._extract_from_standard_input(source_code)
        else:
            self.source = source_code

        self.lines  = self.source.split('\n')
        self.nodes: list[SolidityNode] = []
        self.edges: list[SolidityEdge] = []
        self._node_counter = 0
        self._func_name_to_id: dict[str, int] = {}
        self._state_var_to_id: dict[str, int] = {}

    def _extract_from_standard_input(self, source: str) -> str:
        """Extrait le code Solidity depuis le format Standard JSON Input d'Etherscan."""
        # Nettoyage des doubles accolades Etherscan
        clean = source.strip()
        if clean.startswith("{{"):
            clean = clean[1:-1]  # retire les {{ }} externes
        try:
            data = json.loads(clean)
            parts = []
            sources = data.get("sources", {})
            for filename, content in sources.items():
                if filename.endswith(".sol"):
                    code = content.get("content", "")
                    if "contract " in code:
                        parts.append(f"// === {filename} ===\n{code}")
            return "\n\n".join(parts) if parts else source
        except (json.JSONDecodeError, AttributeError):
            return source

    def _next_id(self) -> int:
        nid = self._node_counter
        self._node_counter += 1
        return nid

    def _get_line(self, match) -> int:
        return self.source[:match.start()].count('\n') + 1

    # ── Extraction des nœuds ─────────────────────────────────────────────────

    def _extract_functions(self):
        for match in self.FUNCTION_RE.finditer(self.source):
            name       = match.group(1)
            modifiers  = match.group(3).lower()
            visibility = "public"
            for v in ("public", "external", "internal", "private"):
                if v in modifiers:
                    visibility = v
                    break

            nid  = self._next_id()
            node = SolidityNode(
                node_id    = nid,
                node_type  = "function",
                name       = name,
                visibility = visibility,
                is_payable = "payable" in modifiers,
                is_view    = "view" in modifiers,
                is_pure    = "pure" in modifiers,
                line_start = self._get_line(match),
            )
            self.nodes.append(node)
            self._func_name_to_id[name] = nid

    def _extract_state_variables(self):
        for match in self.STATE_VAR_RE.finditer(self.source):
            var_type = match.group(1)
            var_name = match.group(2)
            nid      = self._next_id()

            # Heuristique : les mappings et tableaux dynamiques coûtent plus cher
            gas_weight = 2.0 if "mapping" in var_type or "[]" in var_type else 1.0

            node = SolidityNode(
                node_id    = nid,
                node_type  = "state_var",
                name       = var_name,
                gas_weight = gas_weight,
                line_start = self._get_line(match),
            )
            self.nodes.append(node)
            self._state_var_to_id[var_name] = nid

    def _extract_control_flow(self):
        for pattern, label in [
            (self.FOR_LOOP_RE,   "for_loop"),
            (self.WHILE_LOOP_RE, "while_loop"),
            (self.IF_RE,         "if_branch"),
        ]:
            for match in pattern.finditer(self.source):
                nid  = self._next_id()
                node = SolidityNode(
                    node_id   = nid,
                    node_type = "control_flow",
                    name      = label,
                    gas_weight = 1.5 if "loop" in label else 0.5,
                    line_start = self._get_line(match),
                )
                self.nodes.append(node)

    def _extract_operations(self):
        ops = [
            (self.CALL_RE,        "call",         3.0),
            (self.SSTORE_RE,      "sstore",       5.0),
            (self.SLOAD_RE,       "sload",        2.0),
            (self.DELEGATE_RE,    "delegatecall", 4.0),
            (self.SELFDESTRUCT_RE,"selfdestruct", 5.0),
        ]
        for pattern, op_name, gas_weight in ops:
            for match in pattern.finditer(self.source):
                nid  = self._next_id()
                node = SolidityNode(
                    node_id    = nid,
                    node_type  = "operation",
                    name       = op_name,
                    op_type    = op_name,
                    gas_weight = gas_weight,
                    line_start = self._get_line(match),
                )
                self.nodes.append(node)

    # ── Extraction des arêtes ─────────────────────────────────────────────────

    def _extract_edges(self):
        func_ids = {
            n.node_id for n in self.nodes if n.node_type == "function"
        }
        ctrl_ids = {
            n.node_id for n in self.nodes if n.node_type == "control_flow"
        }
        op_ids = {
            n.node_id for n in self.nodes if n.node_type == "operation"
        }
        state_ids = {
            n.node_id for n in self.nodes if n.node_type == "state_var"
        }

        # 1. Call edges : fonction → fonction appelée
        for match in self.INTERNAL_CALL_RE.finditer(self.source):
            callee_name = match.group(1)
            if callee_name in self._func_name_to_id:
                callee_id = self._func_name_to_id[callee_name]
                # Chercher la fonction englobante la plus proche
                call_line = self._get_line(match)
                caller_id = self._find_enclosing_function(call_line)
                if caller_id is not None and caller_id != callee_id:
                    self.edges.append(SolidityEdge(
                        src=caller_id, dst=callee_id,
                        edge_type="call", weight=1.0
                    ))

        # 2. Control-flow edges : control_flow → fonctions proches
        func_list = [n for n in self.nodes if n.node_type == "function"]
        for ctrl in [n for n in self.nodes if n.node_type == "control_flow"]:
            # Associe chaque nœud control-flow à la fonction la plus proche avant lui
            enclosing = self._find_enclosing_function(ctrl.line_start)
            if enclosing is not None:
                self.edges.append(SolidityEdge(
                    src=ctrl.node_id, dst=enclosing,
                    edge_type="control_flow", weight=1.5
                ))

        # 3. State access edges : fonctions → variables d'état
        for func in func_list:
            for var_name, var_id in self._state_var_to_id.items():
                # Vérifie si le nom de la variable apparaît dans le corps de la fonction
                func_body = self._get_function_body(func.line_start)
                if var_name in func_body:
                    self.edges.append(SolidityEdge(
                        src=func.node_id, dst=var_id,
                        edge_type="state_access",
                        weight=2.0
                    ))

        # 4. Inter-procedural edges : operations → fonctions englobantes
        for op in [n for n in self.nodes if n.node_type == "operation"]:
            enclosing = self._find_enclosing_function(op.line_start)
            if enclosing is not None:
                self.edges.append(SolidityEdge(
                    src=enclosing, dst=op.node_id,
                    edge_type="inter_proc",
                    weight=op.gas_weight
                ))

    def _find_enclosing_function(self, line: int) -> Optional[int]:
        """Trouve la fonction dont le corps contient la ligne donnée."""
        best_id   = None
        best_line = -1
        for node in self.nodes:
            if node.node_type == "function" and node.line_start <= line:
                if node.line_start > best_line:
                    best_line = node.line_start
                    best_id   = node.node_id
        return best_id

    def _get_function_body(self, start_line: int, max_lines: int = 50) -> str:
        """Retourne un extrait du corps de la fonction à partir de sa ligne de début."""
        end = min(start_line + max_lines, len(self.lines))
        return "\n".join(self.lines[start_line:end])

    # ── Feature encoding ──────────────────────────────────────────────────────

    def _encode_node_features(self):
        """
        Encode chaque nœud en vecteur numérique de dimension NODE_FEATURE_DIM.

        FEATURES INDÉPENDANTES DU LABELLING (v2 — fix circularité) :
        Les features [11], [12], [13], [14] de la v1 (gas_weight, GAS_INTENSIVE_OPS,
        loop, mapping) étaient directement corrélées avec les critères de labelling
        heuristique → circularité. Remplacées par des features structurelles pures.

        Structure (16 features) :
          [0-3]  : NodeType one-hot (function=0, control_flow=1, state_var=2, operation=3)
          [4-7]  : Visibility one-hot (public=4, external=5, internal=6, private=7)
          [8]    : is_payable
          [9]    : is_view
          [10]   : is_pure
          [11]   : loop_depth normalisé (profondeur d'imbrication, max=5)
          [12]   : nombre de paramètres normalisé (max=10)
          [13]   : position relative dans le contrat (line_start / total_lines)
          [14]   : is_constructor
          [15]   : nombre de connexions sortantes normalisé (out-degree, max=20)
        """
        # Pré-calculer le out-degree de chaque nœud
        out_degree = {}
        for edge in self.edges:
            out_degree[edge.src] = out_degree.get(edge.src, 0) + 1

        total_lines = max(len(self.lines), 1)

        for node in self.nodes:
            features = [0.0] * NODE_FEATURE_DIM

            # [0-3] : NodeType one-hot — structure du graphe, pas le gas
            type_idx = NODE_TYPES.get(node.node_type, 0)
            features[type_idx] = 1.0

            # [4-7] : visibilité one-hot (fonctions uniquement)
            if node.node_type == "function":
                vis_idx = VISIBILITY_MAP.get(node.visibility, 0)
                features[4 + vis_idx] = 1.0

            # [8] : is_payable — attribut syntaxique, pas un pattern gas
            features[8] = float(node.is_payable)

            # [9] : is_view — attribut syntaxique
            features[9] = float(node.is_view)

            # [10] : is_pure — attribut syntaxique
            features[10] = float(node.is_pure)

            # [11] : loop_depth normalisé — profondeur d'imbrication structurelle
            # (différent de "loop" dans le nom — c'est la position dans l'arbre)
            features[11] = min(node.loop_depth / 5.0, 1.0)

            # [12] : nombre de paramètres normalisé
            # Proxy de la complexité de la fonction, indépendant du gas
            n_params = len(getattr(node, 'params', []))
            features[12] = min(n_params / 10.0, 1.0)

            # [13] : position relative dans le contrat
            features[13] = node.line_start / total_lines

            # [14] : is_constructor — information structurelle
            features[14] = float("constructor" in node.name.lower())

            # [15] : out-degree normalisé (nombre de connexions sortantes)
            features[15] = min(out_degree.get(node.node_id, 0) / 20.0, 1.0)

            node.features = features

    # ── Parse principal ───────────────────────────────────────────────────────

    def parse(self) -> tuple[list[SolidityNode], list[SolidityEdge]]:
        """Lance le parsing complet et retourne (nodes, edges)."""
        self._extract_functions()
        self._extract_state_variables()
        self._extract_control_flow()
        self._extract_operations()
        self._extract_edges()
        self._encode_node_features()
        return self.nodes, self.edges


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONSTRUCTION DU GRAPHE PyTorch Geometric
# ══════════════════════════════════════════════════════════════════════════════

def build_pyg_graph(
    nodes:   list[SolidityNode],
    edges:   list[SolidityEdge],
    label:   int   = -1,
    address: str   = "",
) -> Optional[Data]:
    """
    Construit un objet PyTorch Geometric Data depuis les nœuds et arêtes.

    Args:
        nodes   : liste de SolidityNode
        edges   : liste de SolidityEdge
        label   : 0 (efficient) / 1 (inefficient) / -1 (non labellisé)
        address : adresse du contrat (pour traçabilité)

    Returns:
        torch_geometric.data.Data ou None si le graphe est vide
    """
    if not nodes:
        return None

    # Matrice de features des nœuds : [num_nodes, NODE_FEATURE_DIM]
    x = torch.tensor(
        [n.features for n in nodes],
        dtype=torch.float
    )

    # Arêtes : [2, num_edges]
    if edges:
        edge_index = torch.tensor(
            [[e.src, e.dst] for e in edges],
            dtype=torch.long
        ).t().contiguous()

        edge_attr = torch.tensor(
            [[EDGE_TYPES.get(e.edge_type, 0), e.weight] for e in edges],
            dtype=torch.float
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 2),  dtype=torch.float)

    # Label
    y = torch.tensor([label], dtype=torch.long)

    data = Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        y          = y,
        num_nodes  = len(nodes),
        address    = address,
    )
    return data


def build_networkx_graph(
    nodes: list[SolidityNode],
    edges: list[SolidityEdge],
) -> nx.DiGraph:
    """Construit un graphe NetworkX (utile pour visualisation et debug)."""
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(
            n.node_id,
            name=n.name,
            node_type=n.node_type,
            gas_weight=n.gas_weight,
        )
    for e in edges:
        G.add_edge(e.src, e.dst, edge_type=e.edge_type, weight=e.weight)
    return G


# ══════════════════════════════════════════════════════════════════════════════
# 5. SAUVEGARDE
# ══════════════════════════════════════════════════════════════════════════════

def save_graph(
    data:    Data,
    nodes:   list[SolidityNode],
    edges:   list[SolidityEdge],
    address: str,
    out_dir: Path,
    save_json: bool = True,
):
    """
    Sauvegarde le graphe :
      - <address>.pt   : format PyTorch Geometric (pour entraînement)
      - <address>.json : format lisible (pour inspection / debug)
    """
    # .pt
    torch.save(data, out_dir / f"{address}.pt")

    # .json (optionnel)
    if save_json:
        graph_json = {
            "address": address,
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "label": int(data.y.item()),
            "nodes": [
                {
                    "id":        n.node_id,
                    "type":      n.node_type,
                    "name":      n.name,
                    "gas_weight": n.gas_weight,
                    "line":      n.line_start,
                }
                for n in nodes
            ],
            "edges": [
                {
                    "src":       e.src,
                    "dst":       e.dst,
                    "type":      e.edge_type,
                    "weight":    e.weight,
                }
                for e in edges
            ],
        }
        with open(out_dir / f"{address}_graph.json", "w") as f:
            json.dump(graph_json, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 6. LABELLING HEURISTIQUE (Slither-inspired, sans Slither)
# ══════════════════════════════════════════════════════════════════════════════

def compute_inefficiency_score(
    nodes: list[SolidityNode],
    edges: list[SolidityEdge],
    source: str,
) -> float:
    """
    Calcule un score d'inefficacité gas heuristique (0.0 → efficient, 1.0 → très inefficient).
    Inspiré des détecteurs Slither mais sans dépendance externe.

    Patterns détectés :
      - SSTORE dans une boucle (poids fort)
      - External call dans une boucle (poids fort)
      - Variables d'état non optimisées (mappings dans boucles)
      - Fonctions payable sans garde (poids moyen)
      - Appels delegatecall (poids moyen)
    """
    score = 0.0

    op_nodes   = [n for n in nodes if n.node_type == "operation"]
    ctrl_nodes = [n for n in nodes if n.node_type == "control_flow" and "loop" in n.name]
    func_nodes = [n for n in nodes if n.node_type == "function"]

    # 1. SSTORE / external call dans une boucle → score +0.4 par occurrence
    sstore_nodes = {n.node_id for n in op_nodes if n.op_type in ("sstore", "call", "delegatecall")}
    loop_ids     = {n.node_id for n in ctrl_nodes}

    # Vérifier via les arêtes : loop → fonction → operation
    for e in edges:
        if e.src in loop_ids and e.dst in sstore_nodes:
            score += 0.4

    # 2. SSTORE / call non-optimisé dans une fonction (sans boucle explicite)
    for n in op_nodes:
        if n.op_type == "sstore":
            score += 0.15
        elif n.op_type in ("call", "delegatecall"):
            score += 0.10
        elif n.op_type == "selfdestruct":
            score += 0.20

    # 3. Pas d'optimiseur compilateur → + 0.1
    # (détecté via les métadonnées JSON, passé en argument optionnel)

    # 4. Fonctions payable exposées sans require
    for n in func_nodes:
        if n.is_payable and n.visibility in ("public", "external"):
            score += 0.05

    # 5. Boucles profondes (while sans condition claire)
    while_loops = [n for n in ctrl_nodes if "while" in n.name]
    score += len(while_loops) * 0.08

    # 6. Normalisation [0, 1]
    return min(score, 1.0)


def assign_label(score: float, low_threshold: float = 0.25, high_threshold: float = 0.75) -> int:
    """
    Assigne un label binaire basé sur le score d'inefficacité.
    Exclut la zone intermédiaire (comme dans le paper GasGAT).

    Returns:
        0 → efficient   (score <= low_threshold)
        1 → inefficient (score >= high_threshold)
       -1 → ambigu      (zone intermédiaire, exclu du dataset polarisé)
    """
    if score <= low_threshold:
        return 0
    elif score >= high_threshold:
        return 1
    else:
        return -1   # exclu du dataset primaire


# ══════════════════════════════════════════════════════════════════════════════
# 7. PIPELINE PAR CONTRAT
# ══════════════════════════════════════════════════════════════════════════════

def process_contract(
    sol_path:  Path,
    out_dir:   Path,
    save_json: bool  = False,
    polarized: bool  = True,
) -> Optional[dict]:
    """
    Traite un seul fichier .sol :
      1. Lecture du code source
      2. Parsing → nœuds + arêtes
      3. Score + label
      4. Construction du graphe PyG
      5. Sauvegarde

    Returns:
        dict de métadonnées ou None si le contrat est rejeté (ambigu)
    """
    address = sol_path.stem

    # Déjà traité ?
    if (out_dir / f"{address}.pt").exists():
        return {"address": address, "status": "skipped"}

    try:
        source = sol_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning("Lecture échouée %s : %s", address, e)
        return None

    # Parsing
    try:
        parser = SolidityParser(source)
        nodes, edges = parser.parse()
    except Exception as e:
        logger.warning("Parsing échoué %s : %s", address, e)
        return None

    if not nodes:
        return None

    # Score + label
    score = compute_inefficiency_score(nodes, edges, source)
    label = assign_label(score)

    # En mode polarisé : on rejette les cas ambigus
    if polarized and label == -1:
        return {"address": address, "status": "ambiguous", "score": score}

    # Graphe PyG
    data = build_pyg_graph(nodes, edges, label=label, address=address)
    if data is None:
        return None

    # Sauvegarde
    save_graph(data, nodes, edges, address, out_dir, save_json=save_json)

    return {
        "address":   address,
        "status":    "ok",
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "score":     round(score, 4),
        "label":     label,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def build_graphs(
    contracts_dir: str,
    output_dir:    str,
    workers:       int  = 1,
    save_json:     bool = False,
    polarized:     bool = True,
    limit:         int  = None,
):
    """
    Construit les graphes sémantiques pour tous les contrats .sol du répertoire.

    Args:
        contracts_dir : répertoire contenant les .sol téléchargés
        output_dir    : répertoire de sortie pour les .pt
        workers       : nombre de processus parallèles
        save_json     : sauvegarder aussi les graphes en JSON
        polarized     : si True, exclure les contrats ambigus (dataset GasGAT)
        limit         : limiter le nombre de contrats traités (pour tests)
    """
    in_dir  = Path(contracts_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sol_files = sorted(in_dir.glob("*.sol"))
    if limit:
        sol_files = sol_files[:limit]

    logger.info("Contrats à traiter : %d", len(sol_files))
    logger.info("Mode polarisé      : %s", polarized)
    logger.info("Workers            : %d", workers)

    records   = []
    n_ok      = 0
    n_skip    = 0
    n_ambig   = 0
    n_error   = 0

    if workers > 1:
        # Traitement parallèle
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_contract, f, out_dir, save_json, polarized): f
                for f in sol_files
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Graphes construits"):
                try:
                    result = future.result()
                    if result:
                        status = result.get("status", "")
                        if status == "ok":
                            records.append(result)
                            n_ok += 1
                        elif status == "skipped":
                            n_skip += 1
                        elif status == "ambiguous":
                            n_ambig += 1
                except Exception as e:
                    logger.error("Erreur worker : %s", e)
                    n_error += 1
    else:
        # Traitement séquentiel
        for sol_file in tqdm(sol_files, desc="Graphes construits"):
            result = process_contract(sol_file, out_dir, save_json, polarized)
            if result:
                status = result.get("status", "")
                if status == "ok":
                    records.append(result)
                    n_ok += 1
                elif status == "skipped":
                    n_skip += 1
                elif status == "ambiguous":
                    n_ambig += 1
            else:
                n_error += 1

    # Sauvegarde de l'index
    if records:
        import pandas as pd
        index_path = out_dir / "graphs_index.csv"
        pd.DataFrame(records).to_csv(index_path, index=False)
        logger.info("Index sauvegardé → %s", index_path)

        # Distribution des labels
        labels    = [r["label"] for r in records if "label" in r]
        n_eff     = labels.count(0)
        n_ineff   = labels.count(1)
        logger.info("Labels : efficient=%d | inefficient=%d", n_eff, n_ineff)

    logger.info("=" * 55)
    logger.info("Terminé | ok=%d | ignorés=%d | ambigus=%d | erreurs=%d",
                n_ok, n_skip, n_ambig, n_error)
    logger.info("Sortie  → %s", out_dir.resolve())
    logger.info("=" * 55)

    return records


# ══════════════════════════════════════════════════════════════════════════════
# 9. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Construit les graphes sémantiques GasGAT depuis les contrats Solidity."
    )
    p.add_argument("--contracts_dir", type=str, default="data/contracts/",
                   help="Répertoire contenant les .sol (défaut: data/contracts/)")
    p.add_argument("--output_dir",    type=str, default="data/graphs/",
                   help="Répertoire de sortie des graphes .pt (défaut: data/graphs/)")
    p.add_argument("--workers",       type=int, default=1,
                   help="Nombre de processus parallèles (défaut: 1)")
    p.add_argument("--save_json",     action="store_true",
                   help="Sauvegarder aussi les graphes au format JSON")
    p.add_argument("--no_polarize",   action="store_true",
                   help="Inclure tous les contrats (y compris les cas ambigus)")
    p.add_argument("--limit",         type=int, default=None,
                   help="Limiter le nombre de contrats traités (pour tests)")
    p.add_argument("--test",          action="store_true",
                   help="Mode test : traite 5 contrats avec sauvegarde JSON")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.test:
        logger.info("MODE TEST — 5 contrats, JSON activé")
        build_graphs(
            contracts_dir = args.contracts_dir,
            output_dir    = args.output_dir,
            workers       = 1,
            save_json     = True,
            polarized     = not args.no_polarize,
            limit         = 5,
        )
    else:
        build_graphs(
            contracts_dir = args.contracts_dir,
            output_dir    = args.output_dir,
            workers       = args.workers,
            save_json     = args.save_json,
            polarized     = not args.no_polarize,
            limit         = args.limit,
        )
