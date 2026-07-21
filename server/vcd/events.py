"""vcd/events.py — transforme un VCD (VcdFile) en structures JSON pour le frontend.

Deux sorties principales :
- `timeline(vcd, signals, ...)` : pour chaque signal demandé, la liste des
  transitions (t, value) dans la fenêtre [from_t, to_t], plus la liste des
  fronts montants de l'horloge (= cycles logiques) et l'end_time.
- `signal_summary(vcd, ref)` : width, nb transitions, valeur initiale/finale.

Les values sont normalisées : string binaire VCD brute ('1110', 'x', '1').
Le frontend interprète (hex pour les bus >4 bits, bool pour 1-bit).

Le `clk_ref` sert de référence de cycle : on compte les fronts montants (transitions
'0'->'1' ou première valeur '1'). Si non fourni, on tente <tb>.clk puis <tb>.u_dut.clk.
"""
from __future__ import annotations
from typing import Optional

from .parser import VcdFile


def normalize_ref(ref: str) -> str:
    """Retire le suffixe [N:0] pour le display ('dump_tb.waddr[17:0]' -> 'dump_tb.waddr')."""
    if ref.endswith("]") and "[" in ref:
        return ref.rsplit("[", 1)[0]
    return ref


def detect_clk(vcd: VcdFile, clk_ref: Optional[str] = None) -> Optional[str]:
    if clk_ref and clk_ref in vcd.references:
        return clk_ref
    # heuristique : scope top = partie avant le 1er '.'
    for cand in ("clk", "u_dut.clk"):
        for r in vcd.references:
            if r.endswith("." + cand) or r == cand:
                return r
    return None


def clock_posedges(vcd: VcdFile, clk_ref: str, from_t: int = 0, to_t: Optional[int] = None) -> list:
    """Instants des fronts montants de clk_ref (transitions vers '1')."""
    if to_t is None:
        to_t = vcd.end_time
    tv = vcd.signal_tv(clk_ref, from_t, to_t)
    edges = []
    prev = None
    for t, val in tv:
        if val in ("1",) and prev not in ("1",) and t <= to_t:
            edges.append(t)
        prev = val
    return edges


def timeline(
    vcd: VcdFile,
    signals: list,
    from_t: int = 0,
    to_t: Optional[int] = None,
    clk_ref: Optional[str] = None,
    max_per_signal: int = 20000,
) -> dict:
    if to_t is None:
        to_t = vcd.end_time
    clk = detect_clk(vcd, clk_ref)
    edges = clock_posedges(vcd, clk, from_t, to_t) if clk else []

    sig_out = []
    for ref in signals:
        # accepte ref avec ou sans suffixe ; on essaie d'abord tel quel, puis on
        # cherche la ref complète correspondante dans vcd.references.
        full = ref if ref in vcd.references else None
        if full is None:
            # match par suffixe : 'dump_tb.waddr' -> 'dump_tb.waddr[17:0]'
            cand = [r for r in vcd.references if normalize_ref(r) == ref]
            if cand:
                full = cand[0]
        if full is None:
            sig_out.append({"ref": ref, "found": False, "transitions": []})
            continue
        tv = vcd.signal_tv(full, from_t, to_t)
        truncated = False
        if len(tv) > max_per_signal:
            tv = tv[:max_per_signal]
            truncated = True
        sig_out.append({
            "ref": full,
            "name": normalize_ref(full),
            "found": True,
            "width": vcd.signal_width(full),
            "truncated": truncated,
            "transitions": [{"t": t, "v": val} for t, val in tv],
        })

    return {
        "from": from_t,
        "to": to_t,
        "end_time": vcd.end_time,
        "clk_ref": clk,
        "cycles": edges,
        "n_cycles": len(edges),
        "signals": sig_out,
    }


def signal_summary(vcd: VcdFile, ref: str) -> dict:
    full = ref if ref in vcd.references else None
    if full is None:
        cand = [r for r in vcd.references if normalize_ref(r) == ref]
        full = cand[0] if cand else None
    if full is None:
        return {"ref": ref, "found": False}
    tv = vcd.signal_tv(full)
    return {
        "ref": full,
        "name": normalize_ref(full),
        "found": True,
        "width": vcd.signal_width(full),
        "n_transitions": len(tv),
        "initial": tv[0][1] if tv else None,
        "final": tv[-1][1] if tv else None,
    }


def list_signals_grouped(vcd: VcdFile) -> dict:
    """Group les refs par scope (chaque niveau du chemin hiérarchique)."""
    scopes = {}
    for r in vcd.references:
        name = normalize_ref(r)
        scope = name.rsplit(".", 1)[0] if "." in name else ""
        scopes.setdefault(scope, []).append({"ref": r, "name": name, "width": vcd.signal_width(r)})
    return {"scopes": scopes}