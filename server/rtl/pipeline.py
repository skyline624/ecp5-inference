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
STEPS = [
    {"id": "prompt", "label": "Prompt\n(prefill)", "phase": "io", "cluster": "PROMPT",
     "desc": "Prompt (prefill) : N token IDs ingérés en embed+KVWR sans émission, "
             "puis le dernier token du prompt = 1er token généré. En sim "
             "(gen_seq_top, prompt_len=0) : BOS hardcodé (token 1), pas de prefill. "
             "Côté silicium : opcode PR du host."},
    {"id": "embed", "label": "Embedding\nEE[tok]", "phase": "mem", "cluster": "",
     "desc": "Embedding : lit la ligne tok_emb[tok] dans la SDRAM (u_sdram) -> "
             "vecteur XB (64 octets). opcode 'EE' envoyé par u_seq, exécuté par "
             "u_n via u_fm_mac/u_sdram."},
    {"id": "rms_att", "label": "RMSNorm\nrms_att", "phase": "norm", "cluster": "COUCHE",
     "desc": "RMSNorm rms_att (u_rms) : normalise XB -> XN avant l'attention. "
             "Début de couche (répété 5 fois)."},
    {"id": "qkv", "label": "Projections\nQKV (WQ/WK/WV)", "phase": "mac", "cluster": "COUCHE",
     "desc": "Projections Q,K,V via WQ/WK/WV (u_fm_mac) : XN -> Q, K, V. "
             "Une matmul par projection."},
    {"id": "rope", "label": "RoPE\n(Q, K)", "phase": "attn", "cluster": "COUCHE",
     "desc": "RoPE (u_rope) : applique la rotation positionnelle à Q et K selon "
             "pos (freq_cis en BSRAM). -> Q', K'."},
    {"id": "kvwrite", "label": "KV cache\nwrite", "phase": "mem", "cluster": "COUCHE",
     "desc": "KV cache : écrit K',V' roped dans kvmem[pos] (u_sdram) pour les "
             "tokens passés. Lu par l'attention aux positions précédentes."},
    {"id": "attn", "label": "Attention\nsoftmax(QK)·V", "phase": "attn", "cluster": "COUCHE",
     "desc": "Attention (u_attn + u_soft) : softmax(Q'·K' sur les positions "
             "passées) puis pondération par V -> ATT. Causal (masque triangulaire)."},
    {"id": "wo", "label": "Projection\nWO", "phase": "mac", "cluster": "COUCHE",
     "desc": "Projection output WO (u_fm_mac) : ATT -> OUTV."},
    {"id": "resid1", "label": "Résiduel\nXB += OUTV", "phase": "seq", "cluster": "COUCHE",
     "desc": "Résiduel (u_alu) : XB = XB + OUTV. Fin du bloc d'attention."},
    {"id": "rms_ffn", "label": "RMSNorm\nrms_ffn", "phase": "norm", "cluster": "COUCHE",
     "desc": "RMSNorm rms_ffn (u_rms) : XB -> XN avant le FFN."},
    {"id": "ffn_up", "label": "FFN up\nW1, W3", "phase": "mac", "cluster": "COUCHE",
     "desc": "FFN up-projection W1 et W3 (u_fm_mac) : XN -> H1, H3 (dim 172, "
             "zero-paddé à 192)."},
    {"id": "silu", "label": "SiLU\nH1 × H3", "phase": "act", "cluster": "COUCHE",
     "desc": "Activation SiLU (u_silu) : SiLU(H1) × H3 -> SG. x·sigmoid(x)."},
    {"id": "ffn_down", "label": "FFN down\nW2", "phase": "mac", "cluster": "COUCHE",
     "desc": "FFN down-projection W2 (u_fm_mac) : SG -> W2 (retour dim 64)."},
    {"id": "resid2", "label": "Résiduel\nXB += W2", "phase": "seq", "cluster": "COUCHE",
     "desc": "Résiduel (u_alu) : XB = XB + W2. Fin du bloc FFN. -> couche suivante "
             "(×5) ou rms_final."},
    {"id": "rms_final", "label": "RMSNorm\nfinal", "phase": "norm", "cluster": "LM",
     "desc": "RMSNorm final (u_rms, rms_final) : XB -> XN avant lm_head."},
    {"id": "lm_head", "label": "lm_head\ntok_emb", "phase": "mac", "cluster": "LM",
     "desc": "lm_head (u_fm_mac × tok_emb, classifieur partagé avec l'embedding) : "
             "XN -> logits (VOCAB=512)."},
    {"id": "argmax", "label": "Argmax\n(u_seq)", "phase": "seq", "cluster": "LM",
     "desc": "Argmax (u_seq, états AMAXC/AMAXW) : scanne les 512 logits -> best_idx "
             "= token émis. token_valid pulse 1 cycle."},
    {"id": "token", "label": "Token\némis", "phase": "seq", "cluster": "OUT",
     "desc": "Token émis sur le port token/token_valid (u_seq). pos++, cur_tok = "
             "best_idx, boucle vers embed pour le token suivant (jusqu'à NPOS=17)."},
]

# --- Vecteurs de données (arêtes) ---
# (from, to, label)
VECS: List[Tuple[str, str, str]] = [
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
    "PROMPT": "Prompt / prefill",
    "COUCHE": "Couche transformer (×5)",
    "LM": "lm_head + argmax",
    "OUT": "Sortie",
}
CLUSTER_COLOR = {
    "PROMPT": "#6c7086",
    "COUCHE": "#f38ba8",
    "LM": "#a6e3a1",
    "OUT": "#89b4fa",
}


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


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
    return (f'"{s["id"]}" [label="{_esc(s["label"])}", id="step:{s["id"]}", '
            f'fillcolor="{fill}", color="{col}"];')


def pipeline_svg() -> str:
    return graphviz_svg.render_svg(generate_pipeline_dot())


def pipeline_json() -> dict:
    return {
        "steps": {s["id"]: s for s in STEPS},
        "vecs": [{"from": f, "to": t, "label": l, "id": f"vec:{f}__{t}"} for f, t, l in VECS],
        "phases": list(PHASE_COLORS.keys()),
    }


if __name__ == "__main__":
    print(generate_pipeline_dot())