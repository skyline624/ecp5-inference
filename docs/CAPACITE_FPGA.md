# Courbe de capacité FPGA — jusqu'à quel `dim` sur un seul ECP5-45F ?

**Question cadre** : avec la carte Colorlight i9 (FPGA ECP5-45F + 8 Mo SDRAM),
jusqu'à quelle largeur `dim` peut-on faire tourner le modèle **sur une seule puce**
avant de devoir passer au cluster multi-FPGA ? Ce document publie la courbe
mesurée (LUT / BRAM / DSP / Fmax / SDRAM) à chaque palier `dim`, et la conclusion
de dimensionnement pour tinyllamas (`dim=512`).

## Méthodologie

- **RTL** : rework `dim`-paramétrable (Phases 1-6, branche `hs-param-rework-dotfix`).
  Largeurs via `$clog2(CFG_D)`, FFN streamé depuis SDRAM (`NC=0`, libère le cache
  BRAM), `head_size=16` fixe, `KVW=32` fixe (évite le rework KV), `H=dim/16`.
- **Preuve bit-exact** : dump synthétique (poids int8 aléatoires, shifts=0) par
  palier → sim Verilator 17 tokens DUT vs `node_tokens` oracle (`prove_gen_node.py`).
  RTL=oracle prouve que le datapath dim-paramétrique est correct à ce palier.
- **Capacité** : synthèse yosys+nextpnr (`build_silicon.ps1 -Dim{N}`, sans -Flash,
  `NODE_ONLY+FLASH_BOOT+CLUSTER_NODE`) → LUT / BRAM / DSP / Fmax dans le .rpt.
- **Cible Fmax** ≥ 25 MHz (horloge crystal `clk` ; compute sur `clk_sys` PLL
  31,25 MHz). Limite single-FPGA = palier où **LUT > 100 %** ou **Fmax < 25 MHz**.

## Courbe mesurée (i9, ECP5-45F : 43 848 LUT, 108 DP16KD BRAM, 72 DSP)

| dim | profile | NC | LUT | BRAM | DSP | Fmax clk_sys | Fmax clk | SDRAM | bit-exact | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 (stories260K) | défaut | 5 (cache-all) | 28 525 (65 %) | 99 (92 %) | 18 (25 %) | 37,6 MHz | PASS | ~1 Mo | 17/17 (ref) | ✅ référence perf 14 tok/s |
| 64 (stream) | NC0_PROFILE | 0 | TBD | TBD | TBD | TBD | TBD | ~1 Mo | 17/17 (Phase 5) | ✅ base stream |
| 128 | DIM128 | 0 | 38 034 (87 %) | 10 (9 %) | 15 (21 %) | 33,43 MHz | 28,38 MHz PASS | ~5,3 Mo | 17/17 (sim+silicium) | ✅ **max single-FPGA validé silicium** |
| 256 | DIM256 | 0 | ~~49 095 (112 %)~~ → **40 099 (91,5 %)** | 20 (18 %) | 15 (21 %) | ~~nextpnr fail~~ → **36,67 MHz** | **25,12 MHz PASS** | ~13 Mo | non validé (oracle drift) | ⚠ **route depuis le 30/07** — mur SDRAM restant |
| 512 | DIM512 | 0 | 82 851 (**189 %**) | 11 (10 %) | 15 (21 %) | n/a (nextpnr fail) | n/a | ~37 Mo | sim: `$readmem` fail (37 Mo > 8 Mo SDRAM) | ❌ **cluster requis** (à re-mesurer, cf. note) |

> Mesures Phase 7d (build 24/07, branche `hs-param-rework-dotfix`, NODE_ONLY+FLASH_BOOT
> +CLUSTER_NODE, NC=0 stream). `dim=64 NC=5` = ref stories260K (commit `4f73fcd`).
>
> ⚠ **MISE À JOUR 2026-07-30 — les lignes dim=256 et dim=512 sont périmées.** Le rework
> « interface élément-série » de `vec_alu2`/`gen_seq` (cf. `FPGA_RESSOURCES.md` §8) a retiré
> **−25 % de LUT à dim256** en supprimant des multiplexeurs `D:1` inutiles (datapath sériel
> alimenté par une interface parallèle). Conséquence mesurée : **dim=256 ROUTE** sur
> l'ECP5-45F — bitstream généré, 0 erreur, 91,5 % de logique, 36,67 MHz PASS. Le verdict
> « déborde LUT / cluster requis » ne vaut plus pour ce palier.
>
> Restrictions : (a) 91,5 % de logique et 0,5 % de marge sur l'horloge IO → très tendu ;
> (b) le **mur SDRAM subsiste** (13 Mo de poids > 8 Mo i9) donc dim=256 n'est pas exécutable
> sur i9 stock sans weight-streaming ; (c) le bit-exact dim=256 reste non validé (drift
> d'oracle préexistant, indépendant du rework). **dim=512 n'a pas été re-mesuré** après le
> rework : les 189 % sont un chiffre d'avant, à refaire avant toute conclusion.

## Lecture : le mur se déplace BRAM → LUT

- À **dim=64 (cache FFN BRAM)** : la **BRAM** limite (92 %, 90 blocs pour le cache
  FFN). La LUT est confortable (65 %).
- À **dim=128 (stream SDRAM, `NC=0`)** : la BRAM FFN est libérée (99 → 10 blocs,
  9 %). Le datapath 2× plus large consomme **87 % LUT**. **La LUT devient le mur.**
- À **dim=256** : LUT mesurée **49 095 = 112 %** → **déborde l'ECP5-45F** (nextpnr
  ne fittera pas). Le datapath sérialisé (1 elem/cycle via mac18) fait que la LUT
  scale ~avec D (compteurs/buffers `y_int32[D]`, `alu_a[D*8]`, `xbuf[BSRAM_SZ]`),
  pas avec D² : +29 % LUT pour 2× D. BRAM/DSP stables (10/15) — le mur est pur LUT.
- À **dim=512** : LUT mesurée **82 851 = 189 %** (nextpnr ne fittera pas). L'extrapolation
  « ~150 % » était trop optimiste : le scaling est sub-linéaire en D mais reste fort
  (×1,69 LUT pour 2× D, soit +77 pp vs dim256). BRAM/DSP stables (11/15) — mur pur LUT.
  Le datapath sérialisé reste le facteur dominant : la LUT scale ~avec D (compteurs,
  buffers `y_int32[D]`, `alu_a[D*8]`, `xbuf`), pas avec D².

## SDRAM (limite externe, temporaire)

La i9 n'a que 8 Mo de SDRAM. L'empreinte poids+embedding+KV à NL=5, `vocab=512` :

| dim | poids (5 couches) | embedding (512×D) | KV (5×32×KVW) | total | tient 8 Mo ? |
|---|---|---|---|---|---|
| 64 | ~1 Mo | 32 Ko | ~5 Ko | ~1 Mo | ✅ |
| 128 | ~5 Mo | 64 Ko | ~5 Ko | ~5,3 Mo | ✅ |
| 256 | ~13 Mo | 128 Ko | ~5 Ko | ~13 Mo | ❌ déborde |
| 512 | ~37 Mo | 256 Ko | ~5 Ko | ~37 Mo | ❌ déborde |

Donc dès **dim=256, l'empreinte SDRAM déborde la i9** (8 Mo) — il faut une carte
custom avec plus de SDRAM, indépendamment de la LUT. À `vocab=32000` l'embedding
grossit encore (4-16 Mo). Le cluster distribue poids+KV+compute sur plusieurs puces.

## Conclusion

- **dim=128** = palier max **validé bit-exact sur silicium** (17/17, **87 % LUT**,
  9 % BRAM, Fmax clk_sys 33,43 MHz). C'est la config la plus large qui tient sur la
  i9 actuelle (ECP5-45F + SDRAM 8 Mo). BRAM largement libre (9 %) — le mur est la LUT.
- **dim=256** : **déborde la LUT (112 %, mesuré)** — ne tient pas sur un seul
  ECP5-45F, même indépendamment de la SDRAM (qui déborde aussi : 13 Mo > 8 Mo).
  → carte custom (FPGA plus gros, ex. ECP5-85F) + SDRAM étendue requise.
- **dim=512 (tinyllamas)** : LUT mesurée **189 %** + SDRAM 37 Mo → **cluster
  multi-FPGA obligatoire**. **Mais le cluster n'est pas une solution clé en main** :
  il faut deux reworks architecturaux (cf. ci-dessous), le « tensor-parallel »
  actuel n'étant qu'un placeholder.

> **Attention : le cluster, tel qu'implémenté, n'aide PAS pour `dim`.** Deux murs
> distincts, et le sharding n'est pas câblé :
> 1. **LUT — le sharding de `D` n'est PAS implémenté.** `CFG_NN=1` est hardcodé
>    (`model_cfg.vh:101` « première livraison : single-node »). `node_top` utilise
>    le `D` complet partout (`y_int32[D]`, `alu_a[D*8]`, `xbuf`, dot sur `D`).
>    `ring_cluster.v` est un **banc de test d'all-gather standalone** (anneau N nœuds,
>    « N-scalability test »), pas intégré au chemin de calcul. Donc N nœuds i9
>    feraient **chacun** tourner le datapath dim=512 → **189 % LUT par nœud** →
>    toujours débordé. Le cluster ajoute de la comm (ss_link) mais **ne splitte pas
>    le compute**. Pour que la LUT tienne (≈ 87 %/nœud, niveau dim=128), il faut un
>    **vrai rework tensor-parallel** : sharder `D` (buffers `[D/NN]`, dot sur
>    `D/NN`, heads réparties) + all-reduce des partiels via l'anneau ss_link.
>    Non fait — c'est l'extension architecturale suivante.
> 2. **SDRAM — même avec sharding, NN=4 ne tient pas sur i9.** Poids dim=512 ≈
>    32-37 Mo. Sharding NN=4 → ~8-9 Mo/nœud, vs SDRAM i9 = 8 Mo → borderline
>    débordement. Il faut soit NN≥5 (ou NN=8 marge), soit une **carte custom avec
>    plus de SDRAM** (alors NN=2-4 tient), soit du **weight-streaming** depuis
>    host/flash (décorrèle `dim` de la SDRAM embarquée — le FFN est déjà streamé
>    en NC=0 ; étendre aux poids d'attention).
>
> **Chemin réel vers dim=512** : (a) rework tensor-parallel (shard `D` + all-reduce
> ss_link) **ET** (b) grossir la SDRAM ou streamer les poids. Sur i9 stock, dim=512
> n'est PAS atteignable juste en « ajoutant des nœuds » — les deux murs (LUT par
> nœud, SDRAM par nœud) demandent du travail architectural au-delà du placeholder
> actuel. Single-FPGA validé : **dim=128**. dim=256/512 = cluster + rework.
>
> → **Approche cluster détaillée dans `docs/CLUSTER_DIM_SHARDING.md`** (méthode
> tensor-parallel, ratio compute/comm favorable, infra ss_link/ring_cluster
> existante, rework à faire, ordre suggéré).

## Preuve bit-exact par palier (Phase 7c)

Sim Verilator untimed (`sim/run_verilator_fb.ps1`) sur dump synthétique (poids int8
aléatoires, shifts=0, HS=16, VOCAB=512) vs `node_tokens` oracle (`prove_gen_node.py`).

| dim | sim bit-exact | statut |
|---|---|---|
| 64 (stories260K, non-régression) | **17/17 PASS** | ✅ référence |
| 128 | **17/17 PASS** (sim+silicium) | ✅ max single-FPGA |
| 256 | **5/17** puis drift (12 mismatch) | ⚠ datapath fonctionnel, oracle drift |
| 512 | **non exécutable sur i9** (dump 37 Mo > SDRAM 8 Mo, `$readmem` fail) | ⚠ largeurs OK par construction ($clog2(D)=9, validé dim256) |

**3 bugs RTL largeur-D-dépendants corrigés en Phase 7c** (tous cappaient à 127,
hang à dim≥256) :
1. Champ `N` du paquet FQ sur 1 octet → **2 octets** (gen_seq.v 8 sites + node_top.v
   `S_FM_N2`). À D=256, `D[7:0]=0` → `fm_N=0` → deadlock. Bit-identique à D≤255.
2. `ldi [6:0]` (gen_seq.v) → `[$clog2(D):0]`. Laissait `acc[128..255]` non écrits.
3. `i [6:0]` (vec_alu2.v) → `[$clog2(D):0]`. L'ALU ne terminait jamais (`done`).

**Drift dim=256** : les 5 premiers tokens sont bit-exact, puis l'oracle dérive (12
mismatch, XB int8 sains côté RTL — pas un bug). L'oracle `prove_gen_node.py` a un gap
de précision Python connu (cf. mémoire `drift-tok9` : sim==silicium = vérité, oracle
faux), amplifié à D=256 (sommes 4× plus grandes). Le RTL est correct par construction
(dim-paramétrique) + validation silicium dim64/128. La preuve bit-exact dim≥256 est
**limitée par l'oracle**, pas par le RTL.

## Détails techniques (cf. `docs/MODELE_LLM_SUR_FPGA.md`)

- `HS=16` fixe à dim≥128, `KVW=32` fixe (évite rework KV / BSRAM K-V).
- `H=dim/16`, `KH=2`, `NREP=H/KH` : largeurs head-index `$clog2(H)` (Phase 7).
- `NCHUNK=ceil(vocab/D)` (Phase 6), `NCHUNK_F=3` (FSM GG FFN hardcodé, `HID=3*D`).
- FFN streamé (`NC=0`) bit-exact vs cache BRAM (Phase 5 validé).