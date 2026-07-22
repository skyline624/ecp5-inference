"""rtl/cluster.py — Cluster 2× Colorlight i9 (tensor-parallel head-parallel) +
ss_link anneau, avec trafic (all-gather / all-reduce) animé.

Vue pédagogique de l'architecture multi-puce du design (plan §2.1-2.4) : 2 cartes
i9 côte à côte, chaque carte exécute la moitié des 8 heads (NN=2), KV-cache 100%
local (suit la frontière des heads), poids shardés préchargés en backdoor (ne
traversent jamais le lien). Le ss_link anneau (source-synchrone DW=8) ne porte
que des activations O(D)/token : all-gather (W1/W3 row-parallel, ffn_up) et
all-reduce (W2/WO column-parallel, ffn_down + wo — partiels int32 sommés
localement + requantés une fois).

Définition fixe à la main (comme memory.py / pipeline.py — pas d'extraction).
Génère un .dot (clusters Graphviz CARD_A / CARD_B, rankdir=LR) rendu en SVG via
`dot`. ids `node:...` (nœuds), `flow:...` (arêtes intra-carte), `sslink:...`
(arêtes inter-carte). Le frontend (vue Cluster) anime les sslink selon
`link_activity` synchronisé au cycle du pipeline Animate.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

from . import graphviz_svg
from . import memory as rtl_memory

# Couleurs par "phase" de nœud. Les 3 tiers mémoire réutilisent memory.py
# (cohérent avec la vue « Mémoires »). seq = bleu, node (compute TP) = vert.
PHASE_COLORS: Dict[str, Tuple[str, str]] = {
    "seq":   ("#cfe0ff", "#89b4fa"),
    "node":  ("#d8f5d4", "#a6e3a1"),
    "flash": rtl_memory.MEMORY_COLORS["flash"],
    "sdram": rtl_memory.MEMORY_COLORS["sdram"],
    "bram":  rtl_memory.MEMORY_COLORS["bram"],
    "link":  ("#fdd9e2", "#f38ba8"),   # ss_link (rose) — légende seulement
}

CARDS = [
    {"id": "card_a", "label": "FPGA A — i9 #1", "heads": "heads 0..3",
     "desc": "Carte 1 du cluster. Exécute les heads 0..3 (moitié des 8 heads, "
             "NN=2). KV-cache de ces heads 100% local (en SDRAM de cette carte). "
             "Poids du shard (heads 0..3) préchargés en backdoor depuis la flash."},
    {"id": "card_b", "label": "FPGA B — i9 #2", "heads": "heads 4..7",
     "desc": "Carte 2 du cluster. Exécute les heads 4..7 (moitié des 8 heads, "
             "NN=2). KV-cache de ces heads 100% local (en SDRAM de cette carte). "
             "Poids du shard (heads 4..7) préchargés en backdoor depuis la flash."},
]

# Nœuds (10 = 5/carte). id = <prefix>_<kind>, ex "a_seq".
NODES = [
    {"id": "a_seq", "card": "card_a", "kind": "seq", "phase": "seq",
     "label": "Séquenceur TP\nattn/ffn_tp_seq", "desc":
     "Séquenceur tensor-parallel carte A (attn_tp_seq + ffn_tp_seq2, "
     "NN-paramétrés). Pilote le nœud, multipléxe NN liens link. Calcule les "
     "opcodes pour les heads 0..3."},
    {"id": "a_node", "card": "card_a", "kind": "node", "phase": "node",
     "label": "Nœud TP\nnode_top\ntp_node/tp_ar", "desc":
     "Nœud tensor-parallel carte A (node_top + tp_node + tp_ar_node) : matmul "
     "local du shard (heads 0..3) + ring all-gather (W1/W3) / all-reduce "
     "(W2/WO). Poids du shard en SDRAM, KV local en SDRAM, scratch en BRAM."},
    {"id": "a_flash", "card": "card_a", "kind": "flash", "phase": "flash",
     "label": "Flash\nshard poids", "desc":
     "Flash carte A : shard des poids (heads 0..3). RO, préchargé en backdoor "
     "au boot, copié vers la SDRAM. Ne traverse JAMAIS le lien."},
    {"id": "a_sdram", "card": "card_a", "kind": "sdram", "phase": "sdram",
     "label": "SDRAM\nshard + KV local", "desc":
     "SDRAM carte A : miroir du shard de poids + KV-cache LOCAL des heads 0..3 "
     "(layout KV_BASE+(layer*TMAX+pos)<<6, R/W, grandit avec les tokens). Le KV "
     "ne quitte jamais cette carte (suit la frontière des heads)."},
    {"id": "a_bram", "card": "card_a", "kind": "bram", "phase": "bram",
     "label": "BRAM\nscratch", "desc":
     "BRAM carte A : scratchpads du shard — wcache_ffn (du shard), vfile, "
     "xbuf/wbuf, ROMs RoPE, acc. Calculs à latence critique 1 cycle."},
    {"id": "b_seq", "card": "card_b", "kind": "seq", "phase": "seq",
     "label": "Séquenceur TP\nattn/ffn_tp_seq", "desc":
     "Séquenceur tensor-parallel carte B. Pilote le nœud pour les heads 4..7, "
     "multipléxe NN liens link."},
    {"id": "b_node", "card": "card_b", "kind": "node", "phase": "node",
     "label": "Nœud TP\nnode_top\ntp_node/tp_ar", "desc":
     "Nœud tensor-parallel carte B : matmul local du shard (heads 4..7) + ring "
     "all-gather (W1/W3) / all-reduce (W2/WO)."},
    {"id": "b_flash", "card": "card_b", "kind": "flash", "phase": "flash",
     "label": "Flash\nshard poids", "desc":
     "Flash carte B : shard des poids (heads 4..7). RO, préchargé en backdoor, "
     "copié vers SDRAM. Ne traverse jamais le lien."},
    {"id": "b_sdram", "card": "card_b", "kind": "sdram", "phase": "sdram",
     "label": "SDRAM\nshard + KV local", "desc":
     "SDRAM carte B : miroir du shard + KV-cache LOCAL des heads 4..7. R/W, "
     "grandit avec les tokens. Ne quitte jamais cette carte."},
    {"id": "b_bram", "card": "card_b", "kind": "bram", "phase": "bram",
     "label": "BRAM\nscratch", "desc":
     "BRAM carte B : scratchpads du shard — wcache_ffn, vfile, xbuf/wbuf, ROMs "
     "RoPE, acc."},
]

# Arêtes intra-carte (from, to, label). id = flow:<from>__<to>.
CARD_EDGES: List[Tuple[str, str, str]] = [
    ("a_seq", "a_node", "cmd"),
    ("a_node", "a_seq", "resp"),
    ("a_flash", "a_sdram", "boot shard"),
    ("a_sdram", "a_node", "stream shard"),
    ("a_node", "a_bram", "scratch R/W"),
    ("b_seq", "b_node", "cmd"),
    ("b_node", "b_seq", "resp"),
    ("b_flash", "b_sdram", "boot shard"),
    ("b_sdram", "b_node", "stream shard"),
    ("b_node", "b_bram", "scratch R/W"),
]

# ss_link inter-carte (anneau, 2 liens unidirectionnels). id = sslink:<a>__<b>.
# A→B puis B→A = anneau à 2 nœuds (1 saut chacun). DW=8, source-synchrone.
# Chaque edge coule dans sa direction naturelle (from→to) -> l'anneau est rendu
# par l'animation des 2 edges simultanément (A envoie à B, B envoie à A).
SS_LINKS: List[Tuple[str, str, str]] = [
    ("a_node", "b_node", "A→B"),
    ("b_node", "a_node", "B→A"),
]

# Trafic ss_link par étape du pipeline (source de vérité pour l'animation).
# step -> {pattern, desc}. Le lien ne porte que des activations O(D)/token
# (constant, ne grandit pas avec le contexte). Hors ces étapes, lien idle.
LINK_ACTIVITY: Dict[str, dict] = {
    "ffn_up": {"pattern": "all-gather",
               "desc": "W1/W3 row-parallel : chaque carte envoie sa slice de "
                       "sortie au suivant (anneau, 1 saut à NN=2) -> chaque carte "
                       "possède le vecteur complet. All-gather d'activations O(D)."},
    "ffn_down": {"pattern": "all-reduce",
                 "desc": "W2 column-parallel : chaque carte calcule un partiel "
                         "int32 (raw, pas de requant), les partiels sont all-gatherés "
                         "(anneau 32-bit) puis sommés localement + requantés une fois."},
    "wo": {"pattern": "all-reduce",
           "desc": "WO column-parallel : partiels int32 de la projection output de "
                   "l'attention, all-gatherés + sommés localement (comme W2)."},
}

CARD_CLUSTER = {"card_a": "CARD_A", "card_b": "CARD_B"}
CLUSTER_LABEL = {
    "CARD_A": "FPGA A — i9 #1 · heads 0..3",
    "CARD_B": "FPGA B — i9 #2 · heads 4..7",
}
CLUSTER_COLOR = {"CARD_A": "#89b4fa", "CARD_B": "#a6e3a1"}


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _node_dot(n: dict) -> str:
    fill, col = PHASE_COLORS.get(n["phase"], ("#f8f8f8", "#888888"))
    return (f'"{n["id"]}" [label="{_esc(n["label"])}", id="node:{n["id"]}", '
            f'fillcolor="{fill}", color="{col}"];')


def generate_cluster_dot() -> str:
    lines: List[str] = []
    lines.append("digraph G {")
    lines.append("  rankdir=LR;")  # carte A gauche, carte B droite
    lines.append('  graph [fontname="Helvetica", splines=true, nodesep=0.3, ranksep=0.9];')
    lines.append('  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9];')
    lines.append('  edge [fontname="Helvetica", fontsize=8, color="#585b70"];')
    # 2 clusters (cartes)
    for cid, clabel in CLUSTER_LABEL.items():
        col = CLUSTER_COLOR[cid]
        lines.append(f'  subgraph cluster_{cid} {{')
        lines.append(f'    label="{_esc(clabel)}"; style="rounded,filled"; color="{col}"; '
                     f'fillcolor="#1e1e2e0a"; fontname="Helvetica"; fontsize=11;')
        for n in NODES:
            if CARD_CLUSTER[n["card"]] == cid:
                lines.append("    " + _node_dot(n))
        lines.append("  }")
    # arêtes intra-carte
    for frm, to, lbl in CARD_EDGES:
        eid = f"flow:{frm}__{to}"
        lines.append(f'  "{frm}" -> "{to}" [label="{_esc(lbl)}", id="{eid}"];')
    # ss_link inter-carte : rose pointillé, hors-layout (constraint=false)
    for frm, to, lbl in SS_LINKS:
        eid = f"sslink:{frm}__{to}"
        lines.append(f'  "{frm}" -> "{to}" [label="{_esc(lbl)}", id="{eid}", '
                     f'color="#f38ba8", style=dashed, fontcolor="#f38ba8", '
                     f'constraint=false, penwidth=2.0];')
    lines.append("}")
    return "\n".join(lines)


def cluster_svg() -> str:
    return graphviz_svg.render_svg(generate_cluster_dot())


def cluster_json() -> dict:
    return {
        "cards": {c["id"]: c for c in CARDS},
        "nodes": {n["id"]: n for n in NODES},
        "card_edges": [{"from": f, "to": t, "label": l, "id": f"flow:{f}__{t}"}
                        for f, t, l in CARD_EDGES],
        "sslinks": [{"from": f, "to": t, "label": l, "id": f"sslink:{f}__{t}"}
                    for f, t, l in SS_LINKS],
        "link_activity": LINK_ACTIVITY,
        "phases": list(PHASE_COLORS.keys()),
    }


if __name__ == "__main__":
    print(generate_cluster_dot())