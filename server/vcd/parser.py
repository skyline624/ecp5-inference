"""vcd/parser.py — wrapper vcdvcd pour parser un VCD et extraire les valeurs.

vcdvcd 2.6.0 (pur Python) charge le VCD en RAM. API réelle :
  v.endtime            -> int (dernier t ps)
  v.signals           -> list[str] de refs complètes (avec suffixe [N:0] pour bus)
  v[ref]              -> Signal (ref = entrée exacte de v.signals, suffixe compris)
  sig.tv              -> list[(t:int, value:str)]  value = binaire '1110' / 'x' / 'z'
  sig.size            -> largeur
  sig.var_type        -> 'wire'/'reg'/...
Pas de value_at sur Signal -> on le calcule par bissection de tv.

Pour les VCD géants (> 50 Mo) on bascule sur pyDigitalWaveTools en streaming
(Phase C, sampling.py). Pour la Phase B (tests courts) vcdvcd suffit.

Le VCD produit expose des refs hiérarchiques : `dump_tb.clk`, `dump_tb.we`,
`dump_tb.waddr[17:0]`, ... ; sur gen_seq_top : `gen_seq_top.u_n.state[8:0]`, etc.
"""
from __future__ import annotations
from bisect import bisect_right
from pathlib import Path
from typing import List, Optional, Tuple

from vcdvcd import VCDVCD


class VcdFile:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.vcd = VCDVCD(self.path)
        # refs complètes (avec suffixe [N:0] pour les bus) — clés d'accès vcd[ref].
        self._refs: List[str] = list(self.vcd.signals)
        # index t pour value_at (lazy, par signal)
        self._times: dict = {}

    @property
    def end_time(self) -> int:
        return int(self.vcd.endtime)

    @property
    def references(self) -> List[str]:
        """Refs complètes (= clés d'accès vcd[ref]), suffixe [N:0] compris."""
        return self._refs

    @property
    def timescale(self) -> str:
        try:
            return self.vcd.get_timescale()
        except Exception:
            return "1ps"

    def signal_width(self, ref: str) -> int:
        try:
            return self.vcd[ref].size
        except Exception:
            return 1

    def signal_tv(self, ref: str, from_t: int = 0, to_t: Optional[int] = None) -> List[Tuple[int, str]]:
        """Transitions (t, value) du signal `ref` dans [from_t, to_t].
        value est la string VCD brute ('xxxxxxxx', '1110', '1', 'x')."""
        if to_t is None:
            to_t = self.end_time
        try:
            sig = self.vcd[ref]
        except KeyError:
            return []
        out = []
        for t, val in sig.tv:
            if t < from_t:
                continue
            if t > to_t:
                break
            out.append((t, val))
        return out

    def signal_value_at(self, ref: str, t: int) -> Optional[str]:
        """Valeur du signal à l'instant t (dernière transition <= t). None si
        signal absent ou t avant la 1re transition."""
        try:
            sig = self.vcd[ref]
        except KeyError:
            return None
        tv = sig.tv
        if not tv:
            return None
        if ref not in self._times:
            self._times[ref] = [tt for tt, _ in tv]
        times = self._times[ref]
        # bisect_right donne l'index de la 1re transition > t -> on prend la précédente
        idx = bisect_right(times, t)
        if idx == 0:
            # t avant la 1re transition : valeur initiale = tv[0][1] à t=0 typiquement
            return tv[0][1] if tv[0][0] == 0 else None
        return tv[idx - 1][1]

    def list_signals(self, scope_prefix: str = "") -> List[str]:
        """Refs commençant par scope_prefix (ex 'gen_seq_top.u_n')."""
        if not scope_prefix:
            return self._refs
        return [r for r in self._refs if r.startswith(scope_prefix)]


def open_vcd(path: str | Path) -> VcdFile:
    return VcdFile(path)