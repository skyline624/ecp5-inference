"""rtl/extract.py — Phase A : parser RTL Verilog -> graphe de modules.

Extrait, depuis src/*.v + board/gen_seq_silicon.v :
  - la liste des modules déclarés (nom + fichier + paramètres + ports)
  - les instanciations dans chaque module : `MODULE [#(params)] INST (.port(net),...)`
  - le graphe d'instanciation depuis un top donné (gen_seq_silicon ou gen_seq_top)

L'approche : un mini-parser par équilibrage de parenthèses (pas de dépendance
externe). Ce dépôt n'utilise pas de `generate for` pour les sous-modules
critiques -> un scan `IDENT [#(...)] IDENT (` où IDENT est un nom de module
connu suffit à identifier les instanciations (cf plan, Phase A, option b).

Les "signaux d'intérêt" (sd_addr, cache_addr, clk_sys, ...) étiquetent les
edges pour la visualisation "buses nommées" ; leur hier_path complet est
calculé pour le mapping VCD <-> SVG de la Phase C.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import config

# Signaux à étiqueter sur les edges (buses nommées visibles).
INTEREST_SIGNALS = {
    "sd_addr", "sd_dout", "sd_din", "sd_rd", "sd_wr", "sd_refresh", "sd_busy",
    "cache_addr", "wcache_dout", "is_ffn_addr", "ffn_hit_q",
    "wcache_fill_done", "wcache_fill_done_n", "wcache_filling", "booting",
    "start_pulse", "run_mode", "op_sel", "clk_sys", "clk", "clk_sdram",
    "lk_rx_data", "lk_tx_data", "token", "token_valid", "done",
    "result", "result_sh", "shifts_out", "shifts_valid",
}

IDENT_RE = re.compile(r"[A-Za-z_]\w*")


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def _strip_comments(code: str) -> str:
    """Retire /* */ (bloc) et // (ligne). Préserve les newlines pour les offsets."""
    code = re.sub(r"/\*.*?\*/", lambda m: " " * len(m.group(0)), code, flags=re.S)
    code = re.sub(r"//[^\n]*", " ", code)
    return code


def _strip_directives(code: str) -> str:
    """Retire les lignes de directives `ifdef/`else/`endif/`define/`timescale/...
    On garde le contenu des deux branches ifdef/else (potentiel duplicat, OK pour
    l'extraction de graphe : on dédupliquer les instances ensuite)."""
    return re.sub(r"(?m)^[ \t]*`[A-Za-z_]+.*$", "", code)


def _balance(s: str, open_idx: int) -> int:
    """s[open_idx] == '(' -> index du ')' matché, -1 si déséquilibré."""
    depth = 0
    for i in range(open_idx, len(s)):
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\r\n":
        i += 1
    return i


def _net_head(net: str) -> str:
    """Premier identifiant d'un net (ex 'sw_rms[sh_lyr*8 +: 8]' -> 'sw_rms')."""
    m = IDENT_RE.search(net)
    return m.group(0) if m else ""


def _parse_conns(conns: str) -> Dict[str, str]:
    """Parse `.port(net)` nommé (ce dépôt n'utilise pas le positionnel).
    Retourne {port: net_str} avec net nettoyé (newlines -> espaces)."""
    out: Dict[str, str] = {}
    i = 0
    n = len(conns)
    while i < n:
        c = conns[i]
        if c in " \t\r\n,":
            i += 1
            continue
        if c == ".":
            mname = re.match(r"\.\s*([A-Za-z_]\w*)\s*\(", conns[i:])
            if not mname:
                break
            port = mname.group(1)
            op = conns.index("(", i)
            close = _balance(conns, op)
            if close < 0:
                break
            net = conns[op + 1 : close]
            net = re.sub(r"\s+", " ", net).strip()
            out[port] = net
            i = close + 1
        else:
            # positional ou noise -- on skippe jusqu'à la virgule au depth 0
            depth = 0
            while i < n:
                cc = conns[i]
                if cc == "(":
                    depth += 1
                elif cc == ")":
                    depth -= 1
                elif cc == "," and depth == 0:
                    break
                i += 1
            i += 1
    return out


# ---------------------------------------------------------------------------
# Module + instance dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    module: str          # nom du module instancié (ex "sdram")
    inst: str            # nom d'instance (ex "u_sdram")
    params: Optional[str]
    conns: Dict[str, str] = field(default_factory=dict)  # port -> net
    file: str = ""
    line: int = 0


@dataclass
class Module:
    name: str
    file: str
    params: List[str] = field(default_factory=list)   # noms des paramètres
    ports: List[str] = field(default_factory=list)     # noms des ports
    instances: List[Instance] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_file(path: Path) -> Tuple[List[Module], Dict[str, Module]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    code = _strip_directives(_strip_comments(raw))
    # Split en blocs module...endmodule (ils ne s'imbriquent pas en Verilog).
    mods: List[Module] = []
    for chunk in re.split(r"\bendmodule\b", code):
        m = re.search(r"\bmodule\s+([A-Za-z_]\w*)\b", chunk)
        if not m:
            continue
        name = m.group(1)
        # header = du 1er '(' après `module NAME` jusqu'au ';' qui ferme la liste
        # de ports. On équilibre depuis le premier '(' (#(params) puis (ports)).
        body_start = m.end()
        # Cherche le ';' qui termine le header : on équilibre les () éventuels.
        i = body_start
        # skip jusqu'au premier ';' en sautant les blocs parenthésés
        depth = 0
        hdr_end = -1
        while i < len(chunk):
            c = chunk[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif c == ";" and depth == 0:
                hdr_end = i
                break
            i += 1
        if hdr_end < 0:
            hdr_end = body_start
        header = chunk[body_start:hdr_end]
        body = chunk[hdr_end + 1 :]
        # param names (#(.A(..), .B(..)))
        params: List[str] = []
        pm = re.search(r"#\s*\(", header)
        if pm:
            pc = _balance(header, pm.end() - 1)
            if pc > 0:
                pstr = header[pm.end() : pc]
                for pmn in re.finditer(r"\.\s*([A-Za-z_]\w*)\s*\(", pstr):
                    params.append(pmn.group(1))
        # port names ( liste (...) )
        # on prend le dernier bloc (...) du header (les ports, après #(...) éventuel)
        port_names: List[str] = []
        # trouve le dernier '(' au depth 0 du header
        last_op = header.rfind("(")
        if last_op >= 0:
            pc = _balance(header, last_op)
            if pc > 0:
                pstr = header[last_op + 1 : pc]
                for pn in re.finditer(r"\b(?:input|output|inout)\b.*?\b([A-Za-z_]\w*)\b\s*(?=[,)]|$)", pstr):
                    port_names.append(pn.group(1))
                # fallback : ports sans direction (style non-ANSI déjà géré ci-dessus)
                if not port_names:
                    for pn in re.finditer(r"\b([A-Za-z_]\w*)\b\s*(?=[,)]|$)", pstr):
                        nm = pn.group(1)
                        if nm not in {"wire", "reg", "input", "output", "inout", "signed", "logic"}:
                            port_names.append(nm)

        mod = Module(name=name, file=str(path), params=params, ports=port_names)
        # instanciations dans le body
        known = None  # resolved later (need all module names)
        mod._body = body  # type: ignore[attr-defined]
        mod._line_off = chunk[: m.start()].count("\n") + 1  # type: ignore[attr-defined]
        mods.append(mod)
    return mods, {mm.name: mm for mm in mods}


class RTLGraph:
    def __init__(self) -> None:
        self.modules: Dict[str, Module] = {}
        self.files: List[Path] = []

    @classmethod
    def build(cls, files: List[Path]) -> "RTLGraph":
        g = cls()
        g.files = list(files)
        # passe 1 : collecte tous les noms de modules déclarés
        all_mods: List[Module] = []
        for f in files:
            mods, _ = _parse_file(f)
            all_mods.extend(mods)
        known = {m.name for m in all_mods}
        g.modules = {m.name: m for m in all_mods}
        # passe 2 : résout les instanciations de chaque module
        for m in all_mods:
            m.instances = _find_instances(m._body, known, m.file, m._line_off)  # type: ignore[attr-defined]
        return g


def _find_instances(body: str, known: set, file: str, line_off: int) -> List[Instance]:
    """Scan `IDENT [#(...)] IDENT (` où IDENT est un module connu."""
    out: List[Instance] = []
    seen_spans: List[Tuple[int, int]] = []
    for m in IDENT_RE.finditer(body):
        name = m.group(0)
        if name not in known:
            continue
        start = m.start()
        # évite les chevauchements (un ident déjà consommé dans une inst. précédente)
        if any(s <= start < e for s, e in seen_spans):
            continue
        j = _skip_ws(body, m.end())
        params: Optional[str] = None
        if j < len(body) and body[j] == "#":
            k = _skip_ws(body, j + 1)
            if k < len(body) and body[k] == "(":
                close = _balance(body, k)
                if close < 0:
                    continue
                params = body[k + 1 : close]
                j = _skip_ws(body, close + 1)
        m2 = re.match(r"([A-Za-z_]\w*)\s*\(", body[j:])
        if not m2:
            continue
        inst_name = m2.group(1)
        paren_pos = j + m2.end() - 1
        close = _balance(body, paren_pos)
        if close < 0:
            continue
        conns = _parse_conns(body[paren_pos + 1 : close])
        line = line_off + body[:start].count("\n")
        out.append(Instance(module=name, inst=inst_name, params=params, conns=conns, file=file, line=line))
        seen_spans.append((start, close + 1))
    # dédup (ifdef/else peut dupliquer)
    uniq: Dict[Tuple[str, str], Instance] = {}
    for ins in out:
        uniq.setdefault((ins.module, ins.inst), ins)
    return list(uniq.values())


# ---------------------------------------------------------------------------
# Graphe d'instanciation depuis un top
# ---------------------------------------------------------------------------

@dataclass
class GNode:
    inst: str            # nom d'instance (ex "u_sdram")
    module: str          # type (ex "sdram")
    hier: str            # chemin hiérarchique complet (ex "gen_seq_silicon.u_n.u_sdram")
    parent: Optional[str]
    depth: int
    children: List["GNode"] = field(default_factory=list)
    # buses d'intérêt partagées avec le parent : liste de nets (côté parent)
    buses: List[str] = field(default_factory=list)


def _instance_interest_buses(inst: Instance) -> List[str]:
    """Nets (côté parent) des connexions qui sont des signaux d'intérêt."""
    nets = []
    seen = set()
    for port, net in inst.conns.items():
        head = _net_head(net)
        if head in INTEREST_SIGNALS and head not in seen:
            seen.add(head)
            nets.append(head)
    return nets


def build_inst_tree(top: str, g: RTLGraph, max_depth: int = 3) -> GNode:
    """Construit l'arbre d'instanciation depuis `top` (jusqu'à max_depth)."""
    root = GNode(inst=top, module=top, hier=top, parent=None, depth=0)
    _expand(root, g, max_depth)
    return root


def _expand(node: GNode, g: RTLGraph, max_depth: int) -> None:
    if node.depth >= max_depth:
        return
    mod = g.modules.get(node.module)
    if not mod:
        return
    for ins in mod.instances:
        child = GNode(
            inst=ins.inst,
            module=ins.module,
            hier=f"{node.hier}.{ins.inst}",
            parent=node.inst,
            depth=node.depth + 1,
            buses=_instance_interest_buses(ins),
        )
        node.children.append(child)
        _expand(child, g, max_depth)


def flatten(root: GNode) -> List[GNode]:
    out = [root]
    for c in root.children:
        out.extend(flatten(c))
    return out


def edges(root: GNode) -> List[Tuple[GNode, GNode, List[str]]]:
    """Liste des (parent, child, buses) pour le graphe plat."""
    out = []
    for c in root.children:
        out.append((root, c, c.buses))
        out.extend(edges(c))
    return out


# ---------------------------------------------------------------------------
# Clustering (mapping instance parent->enfant -> label de cluster)
# ---------------------------------------------------------------------------

# (parent_inst, child_inst) -> chemin de clusters (sous-clusters Graphviz)
INSTANCE_CLUSTERS: Dict[Tuple[str, str], Tuple[str, ...]] = {
    # top gen_seq_silicon
    ("gen_seq_silicon", "u_seq"): ("SEQUENCER",),
    ("gen_seq_silicon", "u_cmd"): ("FIFOS",),
    ("gen_seq_silicon", "u_resp"): ("FIFOS",),
    ("gen_seq_silicon", "u_rx"): ("LINK_HOST",),
    ("gen_seq_silicon", "u_tx"): ("LINK_HOST",),
    ("gen_seq_silicon", "u_n"): ("NODE",),
    # dans u_n (module top)
    ("u_n", "u_pll"): ("NODE", "PLL"),
    ("u_n", "u_boot"): ("NODE", "MEM"),
    ("u_n", "u_wc"): ("NODE", "MEM"),
    ("u_n", "u_sdram"): ("NODE", "MEM"),
    ("u_n", "u_rx"): ("NODE", "LINK"),
    ("u_n", "u_tx"): ("NODE", "LINK"),
    ("u_n", "u_rms"): ("NODE", "NN"),
    ("u_n", "u_silu"): ("NODE", "NN"),
    ("u_n", "u_rope"): ("NODE", "NN"),
    ("u_n", "u_soft"): ("NODE", "NN"),
    ("u_n", "u_attn"): ("NODE", "NN"),
    ("u_n", "u_fm_mac"): ("NODE", "NN"),
    # dans u_seq (gen_seq)
    ("u_seq", "u_alu"): ("SEQUENCER", "ALU"),
    ("u_seq", "u_tx"): ("SEQUENCER", "LINK"),
    ("u_seq", "u_rx"): ("SEQUENCER", "LINK"),
    # dans u_alu (vec_alu2)
    ("u_alu", "u_mac"): ("SEQUENCER", "ALU"),
}


def css_id(prefix: str, hier: str) -> str:
    """Identifiant CSS/SVG-safe pour les nœuds et edges (Phase C : querySelector)."""
    return f"{prefix}:{hier.replace('.', '_')}"


def cluster_of(parent_inst: str, child_inst: str) -> Tuple[str, ...]:
    return INSTANCE_CLUSTERS.get((parent_inst, child_inst), ())


# Phase C : méta-données pédagogiques (phase fonctionnelle + description) par
# nom d'instance. Phase = catégorie pour couleur/légende du graphe.
MODULE_META: Dict[str, Tuple[str, str]] = {
    "u_seq": ("seq", "Séquenceur (FSM). Pilote le pipeline phase par phase : "
                       "embed → rmsnorm → QKV → RoPE → attention → FFN → lm_head. "
                       "Envoie des opcodes au node via u_cmd, reçoit les vecteurs "
                       "via u_resp. Gère la boucle de tokens et fait l'argmax final."),
    "u_alu": ("seq", "ALU du séquenceur (vec_alu2). Additionneur sérialisé : "
                      "résiduels (XB += OUTV/W2) et réductions (W2)."),
    "u_mac": ("seq", "Multiplieur mac18 interne à u_alu (distinct de u_fm_mac)."),
    "u_cmd": ("io", "FIFO de commandes u_seq → u_n (octets d'opcode + données)."),
    "u_resp": ("io", "FIFO de réponses u_n → u_seq (vecteurs résultats)."),
    "u_tx": ("io", "Émetteur ss_link (sérialiseur d'octets du lien interne)."),
    "u_rx": ("io", "Récepteur ss_link (désérialiseur d'octets du lien interne)."),
    "u_sdram": ("mem", "Contrôleur SDRAM externe. Lit les poids du modèle "
                       "(rms/wq/wk/wv/wo/w1/w3/w2/tok_emb) et héberge le KV cache "
                       "(kvmem : K,V des tokens passés)."),
    "u_wc": ("mem", "Cache FFN (wcache_ffn, BRAM 184 Ko). Cache W1/W3/W2 du FFN. "
                    "Inactif en sim (path non-FLASH_BOOT), actif sur silicium."),
    "u_rms": ("norm", "RMSNorm (rmsnorm_op). Normalise avant l'attention, avant "
                      "le FFN, et en final avant lm_head. Réutilisé 3 rôles."),
    "u_rope": ("attn", "RoPE (rope_op). Rotation positionnelle des vecteurs Q et K "
                        "selon la position (freq_cis en BSRAM)."),
    "u_attn": ("attn", "Cœur d'attention (attention_head_op). Matmul Q·K puis "
                       "pondération des V : softmax(scores) × V → ATT."),
    "u_soft": ("attn", "Softmax (softmax_op) des scores d'attention, par tête."),
    "u_silu": ("act", "Activation SiLu (silu_op) du FFN : x·sigmoid(x), entre W1 et W2."),
    "u_fm_mac": ("mac", "Unité de matmul (mac18 pipeliné, FQ). Fait TOUTES les "
                        "projections : WQ/WK/WV/WO (attention), W1/W3/W2 (FFN), "
                        "lm_head. C'est l'unité de calcul partagée du node."),
    "u_pll": ("clk", "PLL d'horloge (ECP5_PLL). Génère clk_sys depuis l'horloge "
                     "board (pas de calcul NN)."),
    "u_boot": ("boot", "Boot flash (flash_boot). DMA SPI flash → SDRAM au démarrage "
                       "+ latch des shifts. Absent en sim (ifdef FLASH_BOOT)."),
    "u_n": ("node", "Node (node_top) : le transformer complet. Contient rms, rope, "
                    "attn, softmax, silu, mac, sdram, cache. Exécute les calculs NN "
                    "demandés par u_seq."),
    "gen_seq_top": ("top", "Top SIMULATION (wrapper). Ports propres pilotés par un "
                           "testbench. Build -DLINK_SS (+ -DSDRAM_PRELOAD en sim). "
                           "Instancié par tb_node.v (Packet Tracer)."),
    "gen_seq_silicon": ("top", "Top SILICIUM (wrapper Colorlight i9). UART host "
                               "(LL/SH/GO/PR/RR), boot flash, pins physiques. "
                               "Build -DFLASH_BOOT -DCLUSTER_NODE. Flashé sur le FPGA."),
}


def module_meta(inst: str, module: str) -> Tuple[str, str]:
    if inst in MODULE_META:
        return MODULE_META[inst]
    if module in MODULE_META:
        return MODULE_META[module]
    return ("", "")


# Communications logiques entre modules frères (ABSENTES de l'arbre
# d'instanciation, qui ne montre que la contenance parent→child). Ces arêtes
# superposent les liens runtime réels sur le graphe hiérarchie, pour répondre au
# "tous les modules n'ont que des entrées, aucun ne communique".
# (from_inst, to_inst, label, kind) ; kind ∈ cmd|resp|link|boot.
# - cmd/resp : FIFO parallèle (path CLUSTER_NODE / multi-puce).
# - link     : ss_link série (path LINK_SS / sim gen_seq_top).
# - boot     : latch shifts + DMA flash→SDRAM (silicium, ifdef FLASH_BOOT).
COMM_EDGES: List[Tuple[str, str, str, str]] = [
    ("u_seq", "u_cmd", "cmd (push opcode)", "cmd"),
    ("u_cmd", "u_n", "cmd (pop opcode)", "cmd"),
    ("u_n", "u_resp", "resp (push)", "resp"),
    ("u_resp", "u_seq", "resp (pop)", "resp"),
    ("u_seq", "u_n", "ss_link (opcode)", "link"),
    ("u_n", "u_seq", "ss_link (resp)", "link"),
    ("u_boot", "u_sdram", "DMA flash→poids", "boot"),
    ("u_boot", "u_seq", "latch shifts", "boot"),
]


def build_graph_dict(top: str, g: RTLGraph, max_depth: int = 3) -> dict:
    """Construit le dict JSON servi par /api/graph/{top}."""
    root = build_inst_tree(top, g, max_depth=max_depth)
    nodes = flatten(root)
    node_by_hier = {n.hier: n for n in nodes}
    modules_meta = {
        n.hier: {
            "inst": n.inst,
            "module": n.module,
            "hier": n.hier,
            "parent": n.parent,
            "depth": n.depth,
            "cluster": cluster_of(n.parent, n.inst) if n.parent else (),
            "buses": n.buses,
            "id": css_id("mod", n.hier),
            "phase": module_meta(n.inst, n.module)[0],
            "desc": module_meta(n.inst, n.module)[1],
        }
        for n in nodes
    }
    edgelist = []
    for parent, child, buses in edges(root):
        eid = css_id("net", f"{parent.hier}__{child.hier}")
        edgelist.append({
            "from": parent.hier,
            "to": child.hier,
            "buses": buses,
            "id": eid,
        })
    # Index inst -> hier (premier module portant cet inst name) pour résoudre
    # les COMM_EDGES en arêtes concrètes sur ce top.
    inst_to_hier: Dict[str, str] = {}
    for h, m in modules_meta.items():
        inst_to_hier.setdefault(m["inst"], h)
    comm_edgelist = []
    for from_inst, to_inst, label, kind in COMM_EDGES:
        fh = inst_to_hier.get(from_inst)
        th = inst_to_hier.get(to_inst)
        if not fh or not th:
            continue  # module absent de ce top (ex. u_boot en sim)
        comm_edgelist.append({
            "from": fh,
            "to": th,
            "label": label,
            "kind": kind,
            "id": css_id("comm", f"{fh}__{th}"),
        })
    return {
        "top": top,
        "modules": modules_meta,
        "nodes": [{"hier": n.hier, "inst": n.inst, "module": n.module, "depth": n.depth} for n in nodes],
        "edges": edgelist,
        "comm_edges": comm_edgelist,
        "clusters": sorted({c for n in nodes for c in cluster_of(n.parent, n.inst) if n.parent}),
    }


def default_graph() -> dict:
    g = RTLGraph.build(list(config.RTL_FILES))
    return build_graph_dict(config.TOP_SILICON, g, max_depth=3)


if __name__ == "__main__":
    import json
    g = RTLGraph.build(list(config.RTL_FILES))
    print(f"modules déclarés : {len(g.modules)}")
    for name, mod in sorted(g.modules.items()):
        print(f"  {name:24} ({Path(mod.file).name}:{mod.instances and ''} {len(mod.instances)} inst.)")
    print("\n--- arbre depuis gen_seq_silicon ---")
    d = build_graph_dict(config.TOP_SILICON, g, max_depth=3)
    for h, m in d["modules"].items():
        print(f"  {'  ' * m['depth']}{m['inst']} : {m['module']}  {m['cluster']}  buses={m['buses']}")
    print(f"\n{len(d['edges'])} edges")