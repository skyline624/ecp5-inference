"""rtl/memory.py — Carte mémoire 3 tiers (flash / SDRAM / BRAM) + flux de données.

Vue pédagogique de l'architecture mémoire du design (plan §2.2 :
flash = poids RO, SDRAM = KV-cache R/W, BRAM = calculs internes). Contrairement
au graphe d'instanciation (extract.py) qui montre la hiérarchie module/contenance,
ce graphe montre les 3 mémoires, leurs régions/contenus, et les flux entre elles
(boot, cache FFN, KV read/write). Définition fixe du design, à la main (comme
pipeline.py — pas d'extraction Verilog).

Génère un .dot (clusters Graphviz FLASH / SDRAM / BRAM, rankdir=LR pour la
lecture flash -> SDRAM -> BRAM) rendu en SVG via `dot`. Les ids `mem:...`
(nœuds : tiers + régions) et `flow:...` (arêtes) sont injectés pour le frontend
(hover -> description).
"""
from __future__ import annotations
from typing import Dict, List, Tuple

from . import graphviz_svg

# Tier -> (fill, border) pastel Catppuccin.
# flash = brick (boot), sdram = mauve (mem), bram = teal (clk).
MEMORY_COLORS: Dict[str, Tuple[str, str]] = {
    "flash": ("#fde2d0", "#dc8a78"),
    "sdram": ("#ece0fb", "#cba6f7"),
    "bram":  ("#d4e9e8", "#94e2d5"),
}

# cluster d'appartenance d'un tier
TIER_CLUSTER = {"flash": "FLASH", "sdram": "SDRAM", "bram": "BRAM"}

CLUSTER_LABEL = {
    "FLASH": "Flash W25Q64 — SPI NOR 8 Mo, RO, ~0.78 Mo/s",
    "SDRAM": "SDRAM AS4C32M16 — 8 Mo, R/W, refresh 15 µs, clk_sys 31.25 MHz",
    "BRAM":  "BRAM on-chip — DP16KD ×108, ~188 KiB, 1 cycle",
}
CLUSTER_COLOR = {"FLASH": "#dc8a78", "SDRAM": "#cba6f7", "BRAM": "#94e2d5"}

# --- Tiers (nœuds principales, un par cluster) ---
TIERS = [
    {"id": "flash", "label": "Flash\nW25Q64", "phase": "flash", "desc":
     "Flash SPI NOR 8 Mo, read-only, persistante. Stocke l'image du modèle "
     "(poids int8 + 47 shifts + trailer) générée par gen_flash_model.py. Au "
     "power-up, flash_boot.v la copie vers la SDRAM. SCK via la primitive "
     "USRMCLK (pad MCLK dédié sur CABGA381, pas un GPIO). QSPI non câblé "
     "(DQ2/DQ3 absents) -> SPI 1-bit ~0.78 Mo/s."},
    {"id": "sdram", "label": "SDRAM\n8 Mo", "phase": "sdram", "desc":
     "SDRAM SDR externe 8 Mo, R/W, volatile (refresh auto 15 µs). clk_sys "
     "31.25 MHz. DQM hardwired GND sur la i9 -> accès 32-bit word-aligned "
     "obligatoire (piège shadow-word : écrire les 4 bytes d'un mot consécutivement). "
     "2 usages : (1) miroir des poids chargé au boot @ 0x000000-0x060040 (lu "
     "pendant l'inférence), (2) KV-cache R/W qui grandit @ 0x070000+ (étape 1)."},
    {"id": "bram", "label": "BRAM\n~188 KiB", "phase": "bram", "desc":
     "BRAM on-chip Lattice DP16KD (108 blocs), R/W, volatile, 1 cycle. "
     "Scratchpads de calcul à latence critique : cache FFN, register-file vfile, "
     "activations xbuf/wbuf, shifts de re-align ksh/vsh, ROMs RoPE, accumulateur. "
     "Le KV-cache y vivait avant l'étape 1 (10 Ko, 6 DP16KD) -> désormais en SDRAM "
     "(seuls ksh/vsh restent on-chip, latence critique 1 cyc pour le re-align)."},
]

# --- Régions / blocs (sous-nœuds, rattachés à un tier via "tier") ---
REGIONS = [
    # ---- Flash : contenu de l'image ----
    {"id": "flash_model", "tier": "flash",
     "label": "poids modèle\nint8 (~394 Ko)", "phase": "flash", "desc":
     "Poids int8 du modèle stories260K (tok_emb [512,64] + 5 couches + rms_final), "
     "packés en image flash par gen_flash_model.py. ~394 Ko sur les 8 Mo. Copié "
     "vers la SDRAM au boot."},
    {"id": "flash_meta", "tier": "flash",
     "label": "shifts (47)\n+ trailer", "phase": "flash", "desc":
     "47 shifts de quantification (un par matrice de poids) + trailer de boot "
     "(magic, adresses, longueur). Le host lit les shifts via l'opcode SH au "
     "démarrage (payload UART), le reste est lu par flash_boot.v."},
    # ---- SDRAM : régions adressables ----
    {"id": "tok_emb", "tier": "sdram",
     "label": "tok_emb\n0x000000 · 32 KiB", "phase": "sdram", "desc":
     "Table d'embedding [512,64] int8. Partagée avec lm_head (classifieur lié — "
     "weight tying). Lu par l'opcode 'EE' au début de chaque token, et par lm_head "
     "à la fin (XN @ tok_emb -> logits)."},
    {"id": "layers", "tier": "sdram",
     "label": "layers 0..4\n0x010000-0x050000 · 64 KiB/couche", "phase": "sdram", "desc":
     "5 couches transformer (bloc 64 KiB chacune) : rms_att, wq, wk, wv, wo, "
     "rms_ffn, w1, w2, w3. Streamés par u_fm_mac pendant les matvec QKV/WO/lm_head. "
     "W1/W3/W2 sont aussi copiés vers wcache_ffn (BRAM) au 1er token de chaque couche."},
    {"id": "rms_final", "tier": "sdram",
     "label": "rms_final\n0x060000 · 64 o", "phase": "sdram", "desc":
     "RMSNorm final (vecteur de 64 poids int8). Lu juste avant lm_head. Fin du "
     "miroir des poids (0x000000-0x060040)."},
    {"id": "kv", "tier": "sdram",
     "label": "KV-cache\n0x070000 · R/W", "phase": "sdram", "desc":
     "KV-cache (étape 1) : K',V' roped par (layer,pos). Layout "
     "KV_BASE+(layer*TMAX+pos)<<6, K offsets 0-31, V offsets 32-63. R/W, grandit "
     "avec les tokens (NL*TMAX*64 o = 5*32*64 = 10 Ko). Accédé via kv_cdc (CDC "
     "toggle-sync clk 25 / clk_sys 31.25). Avant l'étape 1 ce KV était en BRAM."},
    # ---- BRAM : blocs on-chip ----
    {"id": "wcache_ffn", "tier": "bram",
     "label": "wcache_ffn\n184 KiB · 90 DP16KD", "phase": "bram", "desc":
     "Cache FFN (étape 3) : W1/W3/W2 des 5 couches (184320 octets int8). Rempli "
     "depuis la SDRAM au 1er token de chaque couche, évite les re-reads SDRAM "
     "pendant le FFN (decode 10.96 -> 14.02 tok/s). Gros bloc BRAM."},
    {"id": "vfile", "tier": "bram",
     "label": "vfile\n7×64 o · DP16KD", "phase": "bram", "desc":
     "Register-file (style -49 % LUT) : 7 slots de D=64 — XB, XN, Q, Kc, Vc, ATT, "
     "OUTV. Kc/Vc y sont calculés (FQ + RoPE) puis écrits vers le KV-cache SDRAM "
     "(KVWR, via kv_cdc)."},
    {"id": "xbuf", "tier": "bram",
     "label": "xbuf / wbuf\n128 o each", "phase": "bram", "desc":
     "Scratchpads d'activations 128 octets : rmsnorm, silu, softmax (octets 0..63) "
     "ET K/V multi-head attention (0..127). Le node y reçoit le stream ss_link "
     "depuis le seq (K/V re-aligned y arrivent pour l'attention)."},
    {"id": "ksh", "tier": "bram",
     "label": "ksh / vsh\n160 o each", "phase": "bram", "desc":
     "Shifts de re-align KV (NL×TMAX = 5×32 = 160 octets each). On-chip 1 cycle, "
     "latence critique pour clip8(kraw >>> (sKref - ksh[layer*TMAX+pos])). Sont "
     "restés en BRAM quand le KV a déménagé en SDRAM (étape 1) — le re-align "
     "reste dans le seq."},
    {"id": "cos_rom", "tier": "bram",
     "label": "cos_rom / sin_rom\n68 mots · BSRAM", "phase": "bram", "desc":
     "ROMs RoPE (cos/sin Q1.15, 4 paires × 17 positions = 68 entrées each). "
     "BSRAM chargée à la synthèse par $readmemh depuis lut/freq_cis_{cos,sin}.hex. "
     "Remplace l'ancien bus host cos_q15/sin_q15 (1088-bit) du design Gowin."},
    {"id": "acc", "tier": "bram",
     "label": "acc (vec_alu2)\n64×32-bit · LUT-RAM", "phase": "bram", "desc":
     "Accumulateur serialisé du vec_alu2 (ops elementwise FFN glue : MUL/ADD + "
     "requantify). 64×32-bit en LUT-RAM, boucle 1 élément/cycle (~200 cyc/FFN)."},
]

# --- Flux de données entre mémoires (arêtes) ---
# (from, to, label) — memory-to-memory (boot, cache, KV read/write)
FLOWS: List[Tuple[str, str, str]] = [
    ("flash", "sdram", "boot\n(poids)"),          # flash_boot.v au power-up
    ("layers", "wcache_ffn", "cache FFN\nW1/W3/W2"),  # étape 3, 1er token/couche
    ("kv", "xbuf", "KV read\n(PH_MM)"),           # kv_cdc, attention
    ("vfile", "kv", "KV write\n(KVWR)"),          # étape 1, kv_cdc
]


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _node_dot(n: dict) -> str:
    fill, col = MEMORY_COLORS.get(n["phase"], ("#f8f8f8", "#888888"))
    return (f'"{n["id"]}" [label="{_esc(n["label"])}", id="mem:{n["id"]}", '
            f'fillcolor="{fill}", color="{col}"];')


def _all_nodes() -> List[Tuple[dict, str]]:
    """Tous les nœuds (tiers + régions) avec leur cluster d'appartenance."""
    out: List[Tuple[dict, str]] = []
    for t in TIERS:
        out.append((t, TIER_CLUSTER[t["id"]]))
    for r in REGIONS:
        out.append((r, TIER_CLUSTER[r["tier"]]))
    return out


def generate_memory_dot() -> str:
    lines: List[str] = []
    lines.append("digraph G {")
    lines.append("  rankdir=LR;")  # flash -> SDRAM -> BRAM (lecture gauche -> droite)
    lines.append('  graph [fontname="Helvetica", splines=true, nodesep=0.25, ranksep=0.7];')
    lines.append('  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];')
    lines.append('  edge [fontname="Helvetica", fontsize=9, color="#585b70"];')
    nodes = _all_nodes()
    # clusters
    for cname, label in CLUSTER_LABEL.items():
        col = CLUSTER_COLOR[cname]
        lines.append(f'  subgraph cluster_{cname} {{')
        lines.append(f'    label="{_esc(label)}"; style="rounded,filled"; color="{col}"; '
                     f'fillcolor="#1e1e2e0a"; fontname="Helvetica"; fontsize=11;')
        for n, c in nodes:
            if c == cname:
                lines.append("    " + _node_dot(n))
        lines.append("  }")
    # arêtes (flux)
    for frm, to, lbl in FLOWS:
        eid = f"flow:{frm}__{to}"
        lines.append(f'  "{frm}" -> "{to}" [label="{_esc(lbl)}", id="{eid}"];')
    lines.append("}")
    return "\n".join(lines)


def memory_svg() -> str:
    return graphviz_svg.render_svg(generate_memory_dot())


def memory_json() -> dict:
    return {
        "tiers": {t["id"]: t for t in TIERS},
        "regions": {r["id"]: r for r in REGIONS},
        "flows": [{"from": f, "to": t, "label": l, "id": f"flow:{f}__{t}"}
                  for f, t, l in FLOWS],
        "phases": list(MEMORY_COLORS.keys()),
    }


if __name__ == "__main__":
    print(generate_memory_dot())