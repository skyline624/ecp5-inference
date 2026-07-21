"""rtl/graphviz_svg.py — Phase A : graphe de modules -> SVG via Graphviz `dot`.

Construit un .dot avec clusters Graphviz imbriqués (SEQUENCER/FIFOS/NODE/...),
passe `dot -Tsvg`, et retourne le SVG. Les ids CSS-safe (`mod:...`, `net:...`)
sont injectés via l'attribut `id` natif de Graphviz sur les nœuds et edges ->
le frontend (Phase C) les retrouve par querySelector.

Le layout reste au niveau MODULE (pas gate-level) pour la lisibilité.
"""
from __future__ import annotations
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from .. import config, env


# Couleurs des clusters (identifiant cluster_path join par _).
CLUSTER_COLORS = {
    "SEQUENCER": "#4a90d9",   # bleu
    "FIFOS":     "#9b59b6",   # violet
    "LINK_HOST": "#16a085",   # vert
    "NODE":      "#e67e22",   # orange
    "PLL":       "#f1c40f",   # jaune
    "MEM":       "#c0392b",   # rouge
    "LINK":      "#16a085",   # vert
    "NN":        "#2c3e50",   # bleu foncé
    "ALU":       "#7f8c8d",   # gris
}
CLUSTER_BG = {
    "SEQUENCER": "#eaf2fb",
    "FIFOS":     "#f4ecf7",
    "LINK_HOST": "#e8f8f5",
    "NODE":      "#fdf2e9",
    "PLL":       "#fef9e7",
    "MEM":       "#fadbd8",
    "LINK":      "#e8f8f5",
    "NN":        "#eaecee",
    "ALU":       "#f4f6f6",
}

# Cluster tree
class _CTreeNode:
    def __init__(self, path: Tuple[str, ...]) -> None:
        self.path = path
        self.children: Dict[str, "_CTreeNode"] = {}
        self.nodes: List[dict] = []  # nœuds directement dans ce cluster


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _build_cluster_tree(modules: Dict[str, dict]) -> Tuple[Tuple[str, ...], _CTreeNode, List[dict]]:
    """Construit l'arbre des clusters. Retourne (root_cluster_path, root_node, top_level_nodes)."""
    root = _CTreeNode(())
    top_level: List[dict] = []
    for m in modules.values():
        cp = tuple(m["cluster"])
        if not cp:
            top_level.append(m)
            continue
        node = root
        for part in cp:
            if part not in node.children:
                node.children[part] = _CTreeNode(node.path + (part,))
            node = node.children[part]
        node.nodes.append(m)
    return (), root, top_level


def _node_dot(m: dict, fill: str = "#cfe8ff") -> str:
    nid = m["id"]
    label = f"{_escape(m['inst'])}\\n: {_escape(m['module'])}"
    return f'    "{nid}" [label="{label}", id="{nid}", fillcolor="{fill}"];'


def _emit_cluster(node: _CTreeNode, out: List[str], depth: int = 0) -> None:
    indent = "  " * (depth + 1)
    if node.path:
        cname = "cluster_" + "_".join(node.path)
        color = CLUSTER_COLORS.get(node.path[-1], "#888888")
        bg = CLUSTER_BG.get(node.path[-1], "#fafafa")
        out.append(f'{indent}subgraph {cname} {{')
        out.append(f'{indent}  label="{node.path[-1]}";')
        out.append(f'{indent}  style="rounded,filled"; color="{color}"; fillcolor="{bg}"; fontname="Helvetica";')
    for m in node.nodes:
        out.append(f"  {_node_dot(m)}" if not node.path else f'{indent}  {_node_dot(m, fill="white")}')
    for child in node.children.values():
        _emit_cluster(child, out, depth + (1 if node.path else 0))
    if node.path:
        out.append(f"{indent}}}")


def generate_dot(graph: dict) -> str:
    modules = graph["modules"]
    edges = graph["edges"]
    _, ctree, top_level = _build_cluster_tree(modules)
    lines: List[str] = []
    lines.append("digraph G {")
    lines.append('  rankdir=TB;')
    lines.append('  graph [fontname="Helvetica", splines=true, nodesep=0.35, ranksep=0.5];')
    lines.append('  node [shape=box, style="rounded,filled", fillcolor="#f8f8f8", fontname="Helvetica", fontsize=10];')
    lines.append('  edge [fontname="Helvetica", fontsize=8, color="#666666"];')
    # nœud racine (top-level)
    for m in top_level:
        lines.append("  " + _node_dot(m, fill="#cfe8ff"))
    # clusters
    _emit_cluster(ctree, lines, depth=0)
    # edges
    mod_by_hier = modules
    for e in edges:
        fid = mod_by_hier[e["from"]]["id"]
        tid = mod_by_hier[e["to"]]["id"]
        label = ", ".join(e["buses"]) if e["buses"] else ""
        lbl = f'label="{_escape(label)}"' if label else ""
        lines.append(f'  "{fid}" -> "{tid}" [{lbl} id="{e["id"]}"];')
    lines.append("}")
    return "\n".join(lines)


def render_svg(dot_src: str) -> str:
    """Appelle `dot -Tsvg` et retourne le SVG string."""
    env.setup()
    proc = subprocess.run(
        ["dot", "-Tsvg"],
        input=dot_src,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"dot failed: {proc.stderr}")
    svg = proc.stdout
    # Post-traitement léger : s'assurer que les ids nœuds/edges sont présents.
    # Graphviz met l'attribut `id` sur <g> -- OK pour querySelector.
    return svg


def graph_svg(graph: dict) -> str:
    return render_svg(generate_dot(graph))


if __name__ == "__main__":
    from .extract import default_graph
    env.setup()
    g = default_graph()
    dot = generate_dot(g)
    (config.WORK_DIR / "graph.dot").write_text(dot, encoding="utf-8")
    svg = render_svg(dot)
    (config.WORK_DIR / "graph.svg").write_text(svg, encoding="utf-8")
    print(f"dot -> {config.WORK_DIR / 'graph.dot'}")
    print(f"svg -> {config.WORK_DIR / 'graph.svg'} ({len(svg)} octets)")
    # sanity : compte les id="mod:" et id="net:"
    print(f"mod ids dans SVG : {svg.count('id=\"mod:')}")
    print(f"net ids dans SVG : {svg.count('id=\"net:')}")