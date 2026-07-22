"""rtl/pipeline.py — Phase C : diagramme de DATAFLOW d'un token (vue pipeline).

Contrairement au graphe d'instanciation (extract.py) qui montre la hiérarchie
module/contenance, ce graphe montre le FLUX de données d'un token à travers le
transformer : les nœuds sont les étapes du pipeline (embed → rms → QKV → rope →
attention → FFN → lm_head → argmax), les arêtes sont les vecteurs de données
(XB, XN, Q, K, V, ATT, OUTV, logits, token). C'est fixe pour ce design (pipeline
d'un token en mode génération autonome), donc défini à la main (pas d'extraction).

Le prompt/prefill est représenté en entrée : N token IDs ingérés sans émission,
puis 1er token généré. En sim (gen_seq_top, prompt_len=0) : BOS hardcodé.

Génère un .dot (clusters Graphviz : PROMPT / COUCHE ×5 / LM_HEAD / OUT) rendu en
SVG via `dot`. Les ids `step:...` (nœuds) et `vec:...` (arêtes) sont injectés
pour le frontend (hover -> description de l'étape).
"""
from __future__ import annotations
from typing import Dict, List, Tuple

from . import graphviz_svg
from . import memory as rtl_memory

# Couleur de bord par tier mémoire (réutilise memory.py -> cohérent avec la vue
# « Mémoires »). flash = brick, sdram = mauve, bram = teal.
_TIER_BORDER: Dict[str, str] = {k: v[1] for k, v in rtl_memory.MEMORY_COLORS.items()}
_TIER_FILL: Dict[str, str] = {k: v[0] for k, v in rtl_memory.MEMORY_COLORS.items()}
_TIER_LABEL = {"flash": "flash", "sdram": "SDRAM", "bram": "BRAM"}
# region id -> tier (pour résoudre les accès mémoire d'une étape vers son tier).
_REGION_TIER: Dict[str, str] = {r["id"]: r["tier"] for r in rtl_memory.REGIONS}

# Phases -> (fill, border) pastel Catppuccin.
PHASE_COLORS: Dict[str, Tuple[str, str]] = {
    "io":   ("#d7d9e3", "#6c7086"),
    "mem":  ("#ece0fb", "#cba6f7"),
    "norm": ("#fdf6cf", "#e6d99a"),
    "mac":  ("#d8f5d4", "#a6e3a1"),
    "attn": ("#fdd9e2", "#f38ba8"),
    "act":  ("#fde2d0", "#fab387"),
    "seq":  ("#cfe0ff", "#89b4fa"),
}

# --- Étapes du pipeline (nœuds) ---
# cluster: "" (top), "PROMPT", "COUCHE", "LM", "OUT"
# mem: liste d'accès mémoire [{region, op}] — region = id d'une REGIONS de
# memory.py, op ∈ {"read","write"}. Utilisé pour les pastilles tier sur le
# graphe, le détail sidebar (PipelineView) et la bande mémoire animée (Animate).
STEPS = [
    {"id": "boot", "label": "Boot\nflash→SDRAM", "phase": "mem", "cluster": "BOOT",
     "mem": [{"region": "flash_model", "op": "read"}, {"region": "flash_meta", "op": "read"},
             {"region": "tok_emb", "op": "write"}, {"region": "layers", "op": "write"},
             {"region": "rms_final", "op": "write"}],
     "desc": "Boot (power-up, 1×) : flash_boot.v copie l'image flash (poids int8 "
             "+ 47 shifts + trailer) vers la SDRAM — tok_emb, layers 0..4, rms_final. "
             "Le host lit les shifts via l'opcode SH. Après ça, la flash est idle : "
             "l'inférence tourne sur le miroir SDRAM, la flash n'est plus accédée."},
    {"id": "prompt", "label": "Prompt\n(prefill)", "phase": "io", "cluster": "PROMPT",
     "mem": [],
     "desc": "Prompt (prefill) : N token IDs ingérés en embed+KVWR sans émission, "
             "puis le dernier token du prompt = 1er token généré. En sim "
             "(gen_seq_top, prompt_len=0) : BOS hardcodé (token 1), pas de prefill. "
             "Côté silicium : opcode PR du host."},
    {"id": "embed", "label": "Embedding\nEE[tok]", "phase": "mem", "cluster": "",
     "mem": [{"region": "tok_emb", "op": "read"}],
     "desc": "Embedding : lit la ligne tok_emb[tok] dans la SDRAM (u_sdram) -> "
             "vecteur XB (64 octets). opcode 'EE' envoyé par u_seq, exécuté par "
             "u_n via u_fm_mac/u_sdram."},
    {"id": "rms_att", "label": "RMSNorm\nrms_att", "phase": "norm", "cluster": "COUCHE",
     "mem": [{"region": "vfile", "op": "read"}, {"region": "vfile", "op": "write"}],
     "desc": "RMSNorm rms_att (u_rms) : normalise XB -> XN avant l'attention. "
             "Début de couche (répété 5 fois)."},
    {"id": "qkv", "label": "Projections\nQKV (WQ/WK/WV)", "phase": "mac", "cluster": "COUCHE",
     "mem": [{"region": "layers", "op": "read"}, {"region": "vfile", "op": "write"}],
     "desc": "Projections Q,K,V via WQ/WK/WV (u_fm_mac) : XN -> Q, K, V. "
             "Une matmul par projection."},
    {"id": "rope", "label": "RoPE\n(Q, K)", "phase": "attn", "cluster": "COUCHE",
     "mem": [{"region": "cos_rom", "op": "read"}, {"region": "vfile", "op": "read"},
             {"region": "vfile", "op": "write"}],
     "desc": "RoPE (u_rope) : applique la rotation positionnelle à Q et K selon "
             "pos (freq_cis en BSRAM). -> Q', K'."},
    {"id": "kvwrite", "label": "KV cache\nwrite", "phase": "mem", "cluster": "COUCHE",
     "mem": [{"region": "vfile", "op": "read"}, {"region": "kv", "op": "write"}],
     "desc": "KV cache : écrit K',V' roped dans kvmem[pos] (u_sdram) pour les "
             "tokens passés. Lu par l'attention aux positions précédentes."},
    {"id": "attn", "label": "Attention\nsoftmax(QK)·V", "phase": "attn", "cluster": "COUCHE",
     "mem": [{"region": "kv", "op": "read"}, {"region": "xbuf", "op": "read"},
             {"region": "xbuf", "op": "write"}, {"region": "acc", "op": "write"}],
     "desc": "Attention (u_attn + u_soft) : softmax(Q'·K' sur les positions "
             "passées) puis pondération par V -> ATT. Causal (masque triangulaire)."},
    {"id": "wo", "label": "Projection\nWO", "phase": "mac", "cluster": "COUCHE",
     "mem": [{"region": "layers", "op": "read"}, {"region": "vfile", "op": "write"}],
     "desc": "Projection output WO (u_fm_mac) : ATT -> OUTV."},
    {"id": "resid1", "label": "Résiduel\nXB += OUTV", "phase": "seq", "cluster": "COUCHE",
     "mem": [{"region": "vfile", "op": "read"}, {"region": "vfile", "op": "write"}],
     "desc": "Résiduel (u_alu) : XB = XB + OUTV. Fin du bloc d'attention."},
    {"id": "rms_ffn", "label": "RMSNorm\nrms_ffn", "phase": "norm", "cluster": "COUCHE",
     "mem": [{"region": "vfile", "op": "read"}, {"region": "vfile", "op": "write"}],
     "desc": "RMSNorm rms_ffn (u_rms) : XB -> XN avant le FFN."},
    {"id": "ffn_up", "label": "FFN up\nW1, W3", "phase": "mac", "cluster": "COUCHE",
     "mem": [{"region": "wcache_ffn", "op": "read"}, {"region": "xbuf", "op": "write"}],
     "desc": "FFN up-projection W1 et W3 (u_fm_mac) : XN -> H1, H3 (dim 172, "
             "zero-paddé à 192)."},
    {"id": "silu", "label": "SiLU\nH1 × H3", "phase": "act", "cluster": "COUCHE",
     "mem": [{"region": "xbuf", "op": "read"}, {"region": "xbuf", "op": "write"},
             {"region": "acc", "op": "read"}],
     "desc": "Activation SiLU (u_silu) : SiLU(H1) × H3 -> SG. x·sigmoid(x)."},
    {"id": "ffn_down", "label": "FFN down\nW2", "phase": "mac", "cluster": "COUCHE",
     "mem": [{"region": "wcache_ffn", "op": "read"}, {"region": "xbuf", "op": "read"},
             {"region": "acc", "op": "read"}, {"region": "vfile", "op": "write"}],
     "desc": "FFN down-projection W2 (u_fm_mac) : SG -> W2 (retour dim 64)."},
    {"id": "resid2", "label": "Résiduel\nXB += W2", "phase": "seq", "cluster": "COUCHE",
     "mem": [{"region": "vfile", "op": "read"}, {"region": "vfile", "op": "write"}],
     "desc": "Résiduel (u_alu) : XB = XB + W2. Fin du bloc FFN. -> couche suivante "
             "(×5) ou rms_final."},
    {"id": "rms_final", "label": "RMSNorm\nfinal", "phase": "norm", "cluster": "LM",
     "mem": [{"region": "rms_final", "op": "read"}, {"region": "vfile", "op": "read"},
             {"region": "vfile", "op": "write"}],
     "desc": "RMSNorm final (u_rms, rms_final) : XB -> XN avant lm_head."},
    {"id": "lm_head", "label": "lm_head\ntok_emb", "phase": "mac", "cluster": "LM",
     "mem": [{"region": "tok_emb", "op": "read"}, {"region": "acc", "op": "write"}],
     "desc": "lm_head (u_fm_mac × tok_emb, classifieur partagé avec l'embedding) : "
             "XN -> logits (VOCAB=512)."},
    {"id": "argmax", "label": "Argmax\n(u_seq)", "phase": "seq", "cluster": "LM",
     "mem": [{"region": "acc", "op": "read"}],
     "desc": "Argmax (u_seq, états AMAXC/AMAXW) : scanne les 512 logits -> best_idx "
             "= token émis. token_valid pulse 1 cycle."},
    {"id": "token", "label": "Token\némis", "phase": "seq", "cluster": "OUT",
     "mem": [],
     "desc": "Token émis sur le port token/token_valid (u_seq). pos++, cur_tok = "
             "best_idx, boucle vers embed pour le token suivant (jusqu'à NPOS=17)."},
]

# --- Vecteurs de données (arêtes) ---
# (from, to, label)
VECS: List[Tuple[str, str, str]] = [
    ("boot", "prompt", "poids en SDRAM"),
    ("prompt", "embed", "tok ID"),
    ("embed", "rms_att", "XB (64)"),
    ("rms_att", "qkv", "XN"),
    ("qkv", "rope", "Q, K, V"),
    ("rope", "kvwrite", "K', V'"),
    ("rope", "attn", "Q', V'"),
    ("kvwrite", "attn", "K', V' passés"),
    ("attn", "wo", "ATT"),
    ("wo", "resid1", "OUTV"),
    ("resid1", "rms_ffn", "XB"),
    ("rms_ffn", "ffn_up", "XN"),
    ("ffn_up", "silu", "H1, H3"),
    ("silu", "ffn_down", "SG"),
    ("ffn_down", "resid2", "W2"),
    ("resid2", "rms_att", "couche suivante (×5)"),  # back-edge (loop)
    ("resid2", "rms_final", "XB (après 5 couches)"),
    ("rms_final", "lm_head", "XN"),
    ("lm_head", "argmax", "logits (512)"),
    ("argmax", "token", "best_idx"),
    ("token", "embed", "next token (pos++)"),  # back-edge (boucle de génération)
]

CLUSTER_LABEL = {
    "BOOT": "Boot (power-up, 1×)",
    "PROMPT": "Prompt / prefill",
    "COUCHE": "Couche transformer (×5)",
    "LM": "lm_head + argmax",
    "OUT": "Sortie",
}
CLUSTER_COLOR = {
    "BOOT": "#dc8a78",
    "PROMPT": "#6c7086",
    "COUCHE": "#f38ba8",
    "LM": "#a6e3a1",
    "OUT": "#89b4fa",
}


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _html_esc(s: str) -> str:
    """Échappe les caractères HTML pour les labels HTML-like de Graphviz."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _mem_chip_dot(s: dict) -> str:
    """Ligne de pastilles tier : une cellule colorée (BGCOLOR = bord du tier)
    par tier mémoire accédé par l'étape, avec l'union des ops (R/W/R-W)."""
    ops_by_tier: Dict[str, set] = {}
    for acc in s.get("mem", []):
        tier = _REGION_TIER.get(acc["region"])
        if not tier:
            continue
        ops_by_tier.setdefault(tier, set()).add(acc["op"])
    if not ops_by_tier:
        return ""
    cells = []
    for tier in ("flash", "sdram", "bram"):
        if tier not in ops_by_tier:
            continue
        parts = []
        if "read" in ops_by_tier[tier]:
            parts.append("R")
        if "write" in ops_by_tier[tier]:
            parts.append("W")
        tag = "/".join(parts)
        col = _TIER_BORDER[tier]
        cells.append(
            f'<TD BGCOLOR="{col}" ALIGN="CENTER">'
            f'<FONT POINT-SIZE="8" COLOR="#1e1e2e">{_TIER_LABEL[tier]} {tag}</FONT></TD>'
        )
    return (f'<TR><TD ALIGN="CENTER"><TABLE BORDER="0" CELLBORDER="0" '
            f'CELLSPACING="2"><TR>{"".join(cells)}</TR></TABLE></TD></TR>')


def generate_pipeline_dot() -> str:
    lines: List[str] = []
    lines.append("digraph G {")
    lines.append("  rankdir=TB;")
    lines.append('  graph [fontname="Helvetica", splines=true, nodesep=0.3, ranksep=0.45];')
    lines.append('  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];')
    lines.append('  edge [fontname="Helvetica", fontsize=9, color="#585b70"];')
    # clusters
    for cname, label in CLUSTER_LABEL.items():
        col = CLUSTER_COLOR[cname]
        lines.append(f'  subgraph cluster_{cname} {{')
        lines.append(f'    label="{_esc(label)}"; style="rounded,filled"; color="{col}"; '
                     f'fillcolor="#1e1e2e0a"; fontname="Helvetica"; fontsize=11;')
        for s in STEPS:
            if s["cluster"] == cname:
                lines.append("    " + _step_dot(s))
        lines.append("  }")
    # nœuds hors cluster
    for s in STEPS:
        if not s["cluster"]:
            lines.append("  " + _step_dot(s))
    # arêtes
    for frm, to, lbl in VECS:
        eid = f"vec:{frm}__{to}"
        lines.append(f'  "{frm}" -> "{to}" [label="{_esc(lbl)}", id="{eid}"];')
    lines.append("}")
    return "\n".join(lines)


def _step_dot(s: dict) -> str:
    fill, col = PHASE_COLORS.get(s["phase"], ("#f8f8f8", "#888888"))
    title = _html_esc(s["label"]).replace("\n", "<BR/>")
    chip = _mem_chip_dot(s)
    # Label HTML-like Graphviz : ligne titre + (option) ligne de pastilles tier.
    # L'attribut id reste sur le nœud -> render_svg le met sur le <g> (querySelector).
    label = (f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">'
             f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="10">{title}</FONT></TD></TR>'
             f'{chip}</TABLE>>')
    return (f'"{s["id"]}" [label={label}, id="step:{s["id"]}", '
            f'fillcolor="{fill}", color="{col}"];')


def pipeline_svg() -> str:
    return graphviz_svg.render_svg(generate_pipeline_dot())


def pipeline_json() -> dict:
    return {
        "steps": {s["id"]: s for s in STEPS},
        "vecs": [{"from": f, "to": t, "label": l, "id": f"vec:{f}__{t}"} for f, t, l in VECS],
        "phases": list(PHASE_COLORS.keys()),
        # Mémoires : pour que le frontend rende pastilles/bande sans re-fetch
        # /api/memory. Régions (id -> {label, tier}) + couleurs par tier.
        "mem_regions": {r["id"]: {"label": r["label"].replace("\n", " "), "tier": r["tier"]}
                        for r in rtl_memory.REGIONS},
        "mem_tiers": {k: {"border": v[1], "fill": v[0]}
                      for k, v in rtl_memory.MEMORY_COLORS.items()},
    }


if __name__ == "__main__":
    print(generate_pipeline_dot())