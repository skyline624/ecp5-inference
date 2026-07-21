# ecp5-inference — Accélérateur LLM int8 sur Lattice ECP5 (Colorlight i9 v7.2)

Accélérateur d'inférence LLM quantifié int8 (modèle `stories260K`,
D=64, H=8, NL=5, VOCAB=512), architecture **tensor-parallel en anneau de
nœuds FPGA**, toolchain **100 % open-source** (yosys / nextpnr-ecp5 / ecppack /
openFPGALoader / iverilog / cocotb). Cible : **Colorlight i9 v7.2**
(`LFE5U-45F-6BG381C`, ECP5-45).

## Provenance

Ce dépôt est rendu **indépendant** (propre dépôt GitHub) à partir d'une copie
extraite du sous-répertoire `ecp5/` du projet Gowin `dsp_inference`, branche
`ecp5-g4-nomem2reg-silicon`. La lignée cluster tensor-parallel provient de la
branche `origin/refactor/unified-node-ss-link` du même dépôt d'origine.

Héritage du portage :
- **Portage fidèle** du datapath Gowin → ECP5 : les opérateurs arithmétiques
  (`rmsnorm_op`, `silu_op`, `rope_op`, `softmax_op`, `attention_head_op`,
  UART rx/tx) sont byte-à-byte identiques à l'original ; `top.v` est porté
  (FSM GG strictement identique, seules les primitives vendor changent :
  `Gowin_rPLL`→`EHXPLLL`, `MULTALU18X18`→`MULT18X18D` inféré,
  `syn_ramstyle`→`ram_style`, 27→25 MHz, SDRAM DQM hardwired).
- **Extensions nouvelles** (créées pour ECP5, pas dans l'original Gowin) :
  la lignée cluster (`tp_node`/`tp_ar_node`/`tp_mm_node`, `ring_*`, `ss_link`,
  `link_*`, `async_fifo`, `vec_alu2`, `ffn_tp_seq2`, `attn_tp_seq`), le
  séquenceur autonome (`gen_seq`/`gen_seq_top`/`node_top`/`uart_bridge`),
  `ecp5_ecp5pll.v`, les wrappers silicium et le flow build OSS.
- **Scripts de génération des LUT** (`scripts/gen_*_lut.py`) rapatriérés depuis
  `dsp_inference/host/` (sinon les `.hex` n'étaient plus régénérables) ;
  `gen_freq_cis_lut.py` reprend la formule `make_freq_cis` de l'original
  (aucun script générateur RoPE n'existait auparavant). Voir `lut/README.md`.

Plan complet : `docs/PLAN_ECP5_BRINGUP.md` (cluster), `docs/PLAN_GEN_SEQUENCER.md`
(séquenceur mono-carte), `docs/REFACTOR_UNIFIED_NODE.md` (garde-fous G1-G7).

## Pourquoi

La lignée monolithique (`src/top.v` + FSM `GG`) ne route pas sur GW2AR-18
(congestion-bound, 372–789 nets non routés). La branche cluster a bâti une
lignée tensor-parallel portable prouvée en sim (`gen_seq` 17/17 bit-exact).
Verdict : valider sur un cluster de FPGA ECP5. Ce dossier porte cette lignée
+ les primitives Gowin réécrites pour Lattice.

## État des gates

| Gate | Étape | Statut |
|---|---|---|
| 0 | Toolchain install (oss-cad-suite + cocotb) | ✅ yosys 0.67 / nextpnr-ecp5 0.10 / ecppack 1.4 / openFPGALoader v1.1.1 / iverilog 14.0 / cocotb 2.0.1 |
| 1 | Hello-ECP5 blink sur i9 | ✅ LED D2 clignote ~1.5 Hz (commit 27f292a) |
| 2a | PLL EHXPLLL (lock + clk_sys) | ✅ lock=1 + clk_sys route (compteur clignote) — attributs litex ICP_CURRENT/LPF_RESISTOR étaient le fix |
| 2b | Port SDRAM (8 Mo R/W byte-exact) | ✅ self-test 1024 mots × 2 patterns passe sur i9 (DQM hardwired → accès 32-bit word-aligned) |
| 3 | Port du nœud (matmul bit-exact) — DSP mac18 | ✅ matmul W@x (int8×int8→int32, 8 lignes) bit-exact vs réf Python sur i9 — mac18 RTL → MULT18X18D (sim + silicium) |
| 4 | Lien 2-cartes PMOD (ss_link, skew = inconnue n°1) | à faire (pinout PMOD à assigner — `Colorlight-FPGA-Projects/src/amaranth_boards/colorlight_i9_r7_2.py` lignes 85-89, pmod 2/3 sur P3 recommandés, éviter pmod 0 = L2/LED) |
| 5 | Tensor-parallel NN=2 | à faire (sim vert NN=2 et NN=4 ; silicium non testé) |
| 6 | Génération zéro-PC (boot flash SPI) | boot flash SPI à faire |
| 6' | Génération autonome mono-carte (UART-loading) | ✅ `board/gen_seq_silicon.v` — le PC charge les poids int8 via UART (`LL`/`SH`/`GO`), la FPGA génère 17 tokens sans commande PC pendant l'inférence. Bitstream bâti, route sur ECP5-45. **Autonomie d'inférence atteinte** ; reste l'autonomie de *boot* (flash SPI, Gate 6). |

> **Note** : la Gate 6' (`gen_seq_silicon.v`) est une dérivation parallèle au
> chemin cluster — la même FSM `gen_seq` (NN=1 figé) orchestrée par un wrapper
> silicium. Elle réalise l'objectif « zéro-PC pendant l'inférence » par chargement
> UART des poids plutôt que par boot flash. Cohérent avec l'invariant
> NN=1 mono = NN=N cluster (G7).

> `make` n'est pas installé (nécessaire pour le harness sim cocotb en étape 3+,
> pas pour le build/flash). À installer via MSYS2 (`pacman -S make`) ou
> `choco install make` avant l'étape 3.

## Toolchain (Gate 0)

1. **oss-cad-suite** : télécharger `oss-cad-suite-windows-x64-*.exe` depuis
   https://github.com/YosysHQ/oss-cad-suite-build/releases/latest, extraire vers
   `C:\oss-cad-suite`. Fournit yosys, nextpnr-ecp5, ecppack, openFPGALoader,
   **iverilog**. Ajouter `bin` **et** `lib` en tête de PATH (les DLL comme
   libreadline8.dll sont dans `lib\`) :
   ```powershell
   $env:Path = "C:\oss-cad-suite\bin;C:\oss-cad-suite\lib;" + $env:Path
   [Environment]::SetEnvironmentVariable("Path","C:\oss-cad-suite\bin;C:\oss-cad-suite\lib;" + [Environment]::GetEnvironmentVariable("Path","User"),"User")
   [Environment]::SetEnvironmentVariable("OPENFPGALOADER_SOJ_DIR","C:\oss-cad-suite\share\openFPGALoader","User")
   ```
   (Ou lancez `ecp5\build\install_toolchain.ps1` qui fait extraction + PATH + env.)
2. **cocotb** : `pip install cocotb` puis `python -c "import cocotb"`.
   ⚠ **Python Microsoft Store incompatible** : son `python312.dll` est sous
   `C:\Program Files\WindowsApps\...` (ACL verrouillées) → cocotb/iverilog VPI
   échoue avec « Accès refusé » au `LoadLibrary`. Installer **Python.org 3.12**
   à la place : `winget install --id Python.Python.3.12 --scope user` (désinstalle
   d'abord le Python Store + désactive les alias d'app `python.exe`/`python3.exe`
   dans Paramètres → Alias d'exécution d'application), puis `pip install cocotb numpy`
   dans le nouveau Python. Depuis Git Bash, préfixer le PATH :
   `export PATH="/c/oss-cad-suite/bin:/c/oss-cad-suite/lib:/c/Users/User/AppData/Local/Programs/Python/Python312/Scripts:/c/Users/User/AppData/Local/Programs/Python/Python312:$PATH"`
   et `export PYTHONPATH="$(pwd)"` (depuis `sim/`) pour que les modules de test
   soient trouvables.
3. **make** (oss-cad-suite ne le fournit pas) : MSYS2 `pacman -S make`
   (ajouter `C:\msys64\usr\bin` au PATH) ou `choco install make`.
4. Vérifier : `yosys -V ; nextpnr-ecp5 --version ; ecppack --version ;
   openFPGALoader --version ; iverilog -V ; make --version ; python -c "import cocotb"`.

Pièges : `C:\oss-cad-suite\bin` **avant** mingw/MSYS2 sur PATH (conflit DLL
yosys) ; pilote DAPLink (WinUSB via Zadig si « Unknown Device » — le DAPLink
composite expose déjà CMSIS-DAP) ; chemins bash en `/`.

## Gate 1 — Blink

```powershell
cd ecp5\build
.\build.ps1 -Top blink            # yosys + nextpnr + ecppack
.\build.ps1 -Top blink -Flash     # + openFPGALoader -b colorlight-i9
```
Gate 1 = LED D2 clignote ~1.5 Hz. Si `colorlight-i9` n'est pas listé :
`openFPGALoader --list-boards`, fallback `--device cmsis-dap --chip LFE5U-45F`.

Le flow build (`build.ps1` / `build.sh`) effectue automatiquement :
- copie de `lut/*.hex` vers `build/` (requis par `$readmemh`, cf. piège ci-dessous) ;
- passage de `--lpf-allow-unconstrained` à nextpnr (requis par `top.lpf` /
  `gen_seq_silicon.lpf` qui laissent `led[4:0]` non contraints ; inoffensif pour
  les LPF entièrement contraints). Pour le désactiver (LPF strict) :
  `.\build.ps1 -Top <t> -NoAllowUnconstrained` (ou `STRICT=1` côté bash).

Sous Git-Bash, parité avec `-Extra` :
```bash
./build.sh -e ../src/ecp5_ecp5pll.v pll_test flash
./build.sh -e ../src/ecp5_ecp5pll.v,../src/sdram.v sdram_test flash
```

## Layout

```
src/      RTL adapté ECP5 (ops copiées + lignée cluster + mac18.v/ecp5_ecp5pll.v à écrire)
board/    blink.v, blink.lpf, colorlight_i9.lpf (pinout complet), colorlight_i9_v7.2.md (source wuxx)
          + gen_seq_silicon.{v,lpf} (wrapper silicium Gate 6'), tb_wrap.v
build/    synth.ys, build.ps1, build.sh, scaffold.ps1, install_toolchain.ps1
sim/      Makefile natif cocotb+iverilog (sans Docker), modèles + testbenches + test_*.py
scripts/  gen_{exp,inv,rsqrt,silu,freq_cis}_lut.py + regen_luts.py (source des .hex)
lut/      6 .hex (rsqrt/silu/exp/inv/freq_cis_cos/freq_cis_sin) + README.md
docs/     PLAN_ECP5_BRINGUP, PLAN_GEN_SEQUENCER, REFACTOR_UNIFIED_NODE, SESSION_PROMPTS
```

## Portage des primitives Gowin → ECP5 (étape 3, le point dur)

| Gowin | ECP5 | Fichier |
|---|---|---|
| `Gowin_rPLL` (rPLL) | `EHXPLLL` | `src/ecp5_ecp5pll.v` (✅ Gate 2a verte — lock + clk_sys sur silicium) |
| `MULTALU18X18` (mac18) | RTL pur → `MULT18X18D` (inféré) | `src/mac18.v` (✅ sim bit-exact vs Gowin sur 4000 échantillons ; yosys infère MULT18X18D) |
| `(* syn_ramstyle="block_ram" *)` | `(* ram_style="block" *)` | top.v:240,250,258,267,274,282 ; gen_seq.v:220-221 |
| `(* syn_ramstyle="registers" *)` | `(* ram_style="registers" *)` | top.v:866 |
| `.cst` (Gowin) | `.lpf` (Lattice) | `board/colorlight_i9.lpf` |

`mac18` doit préserver exactement les ports `(clk, rst, ce, load, a[17:0], b[17:0], result[53:0])`
pour que `top.v` soit inchangé. Bug du drain : forcer `a=b=0` quand `ce=0`.

## Piège `$readmemh`

`rmsnorm/silu/softmax/attention` chargent leurs LUT par chemin **relatif au cwd**,
pas au fichier source. Le Makefile sim copie auto les `.hex` (`make hex`).
**`build/build.ps1` et `build/build.sh` copient désormais auto `lut/*.hex` vers
`build/` avant yosys** — plus besoin de le faire à la main. En manuel :
`cp lut/*.hex build/` avant yosys (ou `attrmap` dans synth.ys).

## Régénération des LUT

Les 6 `.hex` de `lut/` sont **régénérables** depuis `scripts/` (numpy requis) :
`python scripts/regen_luts.py` (ou chaque `gen_*_lut.py` individuellement).
Paramètres RoPE modifiables par env : `NPOS=17 HS=8 THETA=10000 python scripts/gen_freq_cis_lut.py`.
Détails (domaine, échelle, fonction, consommateur) dans `lut/README.md`.

## Garde-fous (REFACTOR_UNIFIED_NODE.md)

G1 « ça route ≠ ça marche » (sim vert avant P&R) ; G2 prouver en Python avant
RTL ; G3 isoler chaque brique (2-4 bugs/brique) ; G4 non-régression NN=1 et NN=2 ;
G5 construire sur la lignée qui route ; G6 commit sur vert seulement ; G7 un
seul code NN-paramétré (NN=1 mono = NN=N cluster).

## Rescaffold

```powershell
cd ecp5\build ; .\scaffold.ps1    # recopie depuis master src/ + branche cluster
```