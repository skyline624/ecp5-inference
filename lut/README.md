# lut/ — Tables de recherche (LUT) au format `$readmemh`

Six fichiers `.hex` au format Verilog `$readmemh` (texte ASCII, un mot hexadécimal
16-bit par ligne, en-têtes en commentaire `//`). Consommés par les opérateurs
arithmétiques et le séquenceur de génération via `$readmemh`.

## Piège `$readmemh` (important)

`$readmemh` charge les LUT par chemin **relatif au répertoire de travail (cwd)**
de la simulation/synthèse, **pas** relatif au fichier source Verilog. Si les
`.hex` ne sont pas dans le cwd, les opérateurs sortent `X` silencieusement.

- **Simulation** : `sim/Makefile` (cible `hex:`) et `sim/run_cocotb.py` copient
  `lut/*.hex` dans `sim/sim_build/`. En manuel : `cp lut/*.hex sim/sim_build/`.
- **Synthèse** : copier avant yosys — `cp lut/*.hex build/` (non automatisé dans
  `build/build.ps1`, à faire avant `.\build.ps1`).

## Régénération

Tous les `.hex` sont **régénérables** depuis `../scripts/` (numpy requis) :

```bash
cd ecp5-inference
python scripts/gen_exp_lut.py        # -> exp_lut.hex
python scripts/gen_inv_lut.py         # -> inv_lut.hex
python scripts/gen_rsqrt_lut.py       # -> rsqrt_lut.hex
python scripts/gen_silu_lut.py        # -> silu_lut.hex
python scripts/gen_freq_cis_lut.py    # -> freq_cis_cos.hex + freq_cis_sin.hex
# ou tout en un coup :
python scripts/regen_luts.py
```

Paramètres RoPE modifiables par env : `NPOS=17 HS=8 THETA=10000 python scripts/gen_freq_cis_lut.py`.

## Tableau des LUT

| Fichier | Entrées | Largeur | Format | Domaine / échelle | Fonction | Consommé par | Script |
|---|---|---|---|---|---|---|---|
| `rsqrt_lut.hex` | 256 | 16-bit unsigned | Q1.15 | x ∈ [1, 2), pas 1/256 ; `v = round(1/√x · 32768)` | 1/√x (rmsnorm) | `src/rmsnorm_op.v:58` | `gen_rsqrt_lut.py` |
| `silu_lut.hex` | 256 | 16-bit **signed** | Q4.11 (scale 2⁻¹¹) | x ∈ [-8, 8), pas 1/16 ; `v = round(silu(x) · 2048)` | SiLU `x·σ(x)` (FFN) | `src/silu_op.v:43` | `gen_silu_lut.py` |
| `exp_lut.hex` | 256 | 16-bit unsigned | Q15 | x ∈ [-8, 0), pas 1/32 ; `v = round(exp(x) · 32768)` | exp(x) (softmax / attention) | `src/attention_head_op.v:71`, `src/softmax_op.v:52` | `gen_exp_lut.py` |
| `inv_lut.hex` | 256 | 16-bit unsigned | Q15 | x ∈ [1, 2), pas 1/256 ; `v = round(1/x · 32768)` | 1/x (division softmax/attn) | `src/attention_head_op.v:72`, `src/softmax_op.v:53` | `gen_inv_lut.py` |
| `freq_cis_cos.hex` | 68 | 16-bit signed | Q15 | NPOS=17 × 4 paires (HS/2=4) ; `v = round(cos(pos·freqs[k]) · 32768)` | RoPE cosinus (partie réelle) | `src/gen_seq.v:227` (`cos_rom`) | `gen_freq_cis_lut.py` |
| `freq_cis_sin.hex` | 68 | 16-bit signed | Q15 | idem ; `v = round(sin(pos·freqs[k]) · 32768)` | RoPE sinus (partie imaginaire) | `src/gen_seq.v:228` (`sin_rom`) | `gen_freq_cis_lut.py` |

## Notes techniques

### rsqrt (rmsnorm) — parité impaire

La LUT ne couvre que x ∈ [1, 2). Pour un `shift_amt` impair, `rmsnorm_op.v`
multiplie le résultat par √2 pour couvrir la moitié d'intervalle impaire. La
constante `SQRT2_Q15 = 0xb505` (= `round(√2 · 32768)` = 46341) est **hardcodée
dans `src/rmsnorm_op.v`** et calculée par `gen_rsqrt_lut.py:9`. Toute modification
de la LUT doit garder cette constante cohérente.

### freq_cis (RoPE) — Llama2 stories260K

Fréquences RoPE : `freqs[k] = 1 / θ^(2k/HS)` avec θ = 10000, HS = 8.
- `freqs = [1, 0.1, 0.01, 0.001]` (car `10000^[0, 0.25, 0.5, 0.75] = [1, 10, 100, 1000]`).
- `angles[pos, k] = pos · freqs[k]`.
- 4 paires × 17 positions = 68 entrées/fichier. `pos=0` → cos = 0x7fff (1.0), sin = 0x0000 (0.0).

Ces ROMs remplacent les anciens ports `cos_q15`/`sin_q15` (bus 1088-bit) du
design Gowin d'origine : passage host-série → BSRAM ROM interne pour éviter le
mux 68:1 + bus high-fanout qui congestionnait le routage du GW2AR-18.

### Précision

- Q15 (2¹⁵) pour rsqrt, exp, inv, freq_cis — précision ~3e-5 relatif.
- Q4.11 (2¹¹) pour silu — précision ~5e-4 (plage ±8).
- `silu_lut.hex` est la seule LUT **signée** (deux-complément 16-bit) ; les 5
  autres sont non-signées Q15.

## Provenance

Les scripts `gen_{exp,inv,rsqrt,silu}_lut.py` sont issus de l'original Gowin
`dsp_inference/host/` (chemin de sortie ajusté `../src/` → `../lut/`).
`gen_freq_cis_lut.py` reprend la formule `make_freq_cis` de
`dsp_inference/host/test_attn_pos.py:15-21` (aucun script générateur n'existait
dans l'original — les `.hex` RoPE y étaient produits hors-arbre).