# Analyse des ressources FPGA — accélérateur LLM int8 (Lattice ECP5)

> **Document révisé le 2026-07-27 par analyse du code RTL** (`src/*.v`, `board/gen_seq_silicon.v`, `build/*.ps1`) via 4 agents parallèles. La version précédente s'appuyait sur des rapports de synthèse obsolètes ; elle a été corrigée.
> - **Structure RTL, hiérarchie, paramètres, pipeline, DSP/BRAM par module** : vérifiés sur le code (cette version).
> - **Totaux placés-routés (LUT4/DFF/Fmax)** : issus d'un build antérieur, **à régénérer** (les rapports `build/*.log` sont stale). Conservés à titre indicatif.

Cible : **LFE5U-45F** (Colorlight i9 v7.2) — 43 848 LUT4 / 43 848 DFF / 108 BRAM DP16KD / 72 DSP MULT18X18D / 4 PLL EHXPLLL. Package CABGA381, speed 6 (`build.ps1:63`).
Top synthèse : `gen_seq_silicon` (`board/gen_seq_silicon.v:29`).

---

## 1. Hiérarchie d'instanciation (vérifiée sur le code)

**1 instance de chaque module** dans le build silicium. Les chiffres « ×7 / ×3 / ×5 / ×4 » de l'ancien doc étaient des artefacts du rapport Yosys `stat` (comptes de sous-blocs/ROM internes), pas des instances de module.

```
gen_seq_silicon                          [board/gen_seq_silicon.v:29]   (FSM hôte, 2 FIFO async, UART)
├── gen_seq u_seq                        [gen_seq_silicon.v:168]        (séquenceur génération, FSM 35 états)
│   ├── vec_alu2 u_alu                   [gen_seq.v:178]                (ALU sérialisée : résiduels + MUL SwiGLU)
│   │   └── mac18 u_mac                  [vec_alu2.v:35]               (1 DSP)
│   ├── tx8_link u_tx / rx8_link u_rx    [gen_seq.v:167,169]
│   └── (logits, cos_rom, sin_rom en BRAM interne)
├── async_fifo u_cmd, u_resp             [gen_seq_silicon.v:191,194]   (×2, LUT-RAM)
├── top u_node                           [gen_seq_silicon.v:485]        (module de src/node_top.v)
│   ├── ecp5_pll u_pll                   [node_top.v:155]              (EHXPLLL : clk_sys 31.25 + clk_sdram 180°)
│   ├── flash_boot u_fb                  [node_top.v:227]              (boot SPI W25Q64 → SDRAM)
│   ├── wcache_ffn u_wc                  [node_top.v:348]              (cache poids FFN, ~90 BRAM si CFG_NC>0)
│   ├── sdram u_sdram                    [node_top.v:392]              (contrôleur SDRAM 32-bit, FREQ=31.25 MHz)
│   ├── rx8_link u_rx / tx8_link u_tx   [node_top.v:414,417]          (framing UART côté node)
│   ├── rmsnorm_op u_rms                 [node_top.v:640]              (RMSNorm, 1 BRAM rsqrt_lut)
│   ├── silu_op u_silu                   [node_top.v:662]              (SiLU, 1 BRAM silu_lut)
│   ├── rope_op u_rope                   [node_top.v:681]              (RoPE, 4 DSP, 0 BRAM)
│   ├── softmax_op u_soft                [node_top.v:705]              (Softmax, K_MAX=32 hardcodé, 2 BRAM via shared_lut)
│   ├── attention_head_op u_attn         [node_top.v:728]              (Attention mono-tête, 2 BRAM via shared_lut)
│   └── mac18 u_fm_mac                   [node_top.v:755]              (1 DSP, dédié lm_head, non partagé avec vec_alu2)
├── kv_cdc u_kvcdc                       [gen_seq_silicon.v:520]       (CDC clk↔clk_sys pour KV SDRAM)
├── uart_rx_8n1 / uart_tx_8n1            [gen_seq_silicon.v:70,74]    (DIV=25, ~1 Mbaud)
└── USRMCLK                              (primitive MCLK flash)
```

**Modules tensor-parallel présents mais NON activés** (NN=1, single-node) : `ffn_tp_seq2`, `attn_tp_seq`, `tp_node`, `tp_ar_node`, `tp_mm_node`, `ring_cluster`, `ss_link`, `link_send/recv`. Ils ne sont pas dans la liste des sources du build silicium (`build_silicon.ps1:20-38`).
**`src/top.v`** déclare aussi `module top` (legacy dim64 hardcoded) mais n'est **pas compilé** dans le build silicium (`build_silicon.ps1:21` ne liste que `node_top.v`). Doublon à nettoyer.

---

## 2. Paramètres DIM (source unique : `src/model_cfg.vh`)

Sélection par define Verilog `-DDIM128/256/512` (défaut dim64), propagé via `CFG_*` → parameters des modules ops.

| Paramètre | dim64 | dim128 | dim256 | dim512 | Rôle |
|---|---|---|---|---|---|
| `CFG_D` (DIM) | 64 | 128 | 256 | 512 | dimension modèle |
| `CFG_HS` (head size) | 8 | 16 | 16 | 16 | |
| `CFG_H` (NH) | 8 | 8 | 16 | 32 | n_heads (D=H·HS) |
| `CFG_KH` (GQA n_kv_heads) | 4 | 2 | 2 | 2 | |
| `CFG_NREP` = H/KH | 2 | 4 | 8 | 16 | |
| `CFG_NL` (couches) | 5 | 5 | 5 | 5 | |
| `CFG_NPOS` (gen length) | 17 | 17 | 17 | 17 | |
| `CFG_TMAX` (T_MAX) | 32 | 32 | 32 | 32 | |
| `CFG_VOCAB` | 512 | 512 | 512 | 512 (32000 via `VOCAB32000`) | |
| **`CFG_NC` (FFN layers cached)** | **5 (cache-all)** | **0 (stream)** | 0 | 0 | **← détermine la taille du wcache_ffn** |
| `CFG_HID` (FFN hidden) | 172 | 384 | 768 | 1536 | 3·D sauf dim64 |
| `CFG_HIDPAD` | 192 | 384 | 768 | 1536 | |
| `CFG_NCHUNK` (lm_head chunks) | 8 | 4 | 2 | 1 | ceil(VOCAB/D) |
| `CFG_NFFN` (wcache size) | NC·WCACHE_LAYER_SZ ≈ 184320 | 1 | 1 | 1 | **octets de cache FFN on-chip** |
| `CFG_NN` (tensor-parallel) | 1 | 1 | 1 | 1 | single-node (première livraison) |
| `CFG_SCORE_SH` | 0 | 1 | 1 | 1 | calibrage HS |

**Note** : `softmax_op` est instancié avec `K_MAX(32)` **hardcodé** (`node_top.v:705`), non paramétré via `CFG_TMAX`. Couplage implicite (CFG_TMAX=32 partout, mais à surveiller si T_MAX change).

---

## 3. Pipeline de calcul (FSM `gen_seq`, `src/gen_seq.v`)

Phases LLM (`PH_*`, `gen_seq.v:222-224`) ; FSM 35 états (`st`, `gen_seq.v:229-234`). Boucle sur NL=5 couches puis final :

1. **PH_EE** — embedding token entrée (SDRAM `A_EMB + tok·D`)
2. **Boucle NL couches** :
   a. **PH_FN** — RMSNorm pré-attention (`rmsnorm_op u_rms`)
   b. **PH_WQ** — matmul Q=x·WQ + RoPE sur Q (`rope_op u_rope`, state ROPEQ)
   c. **PH_WK** — matmul K=x·WK + RoPE sur K (state ROPEK)
   d. **PH_WV** — matmul V=x·WV
   e. **PH_MM** — attention scan : Q·K, softmax (`softmax_op u_soft`), ·V (`attention_head_op u_attn`) ; écriture KV-cache SDRAM via 4ᵉ maître (kv_cdc, fenêtre KVWR_*)
   f. **PH_WO** — matmul out=A·WO
   g. **RESID** — x += attn_out (via `vec_alu2 u_alu`)
   h. **PH_FN2** — RMSNorm pré-FFN (`rmsnorm_op u_rms`)
   i. **PH_FW1** — matmul h1=x·W1
   j. **PH_FW3** — matmul h3=x·W3
   k. **PH_SS** — SiLU h=h1·σ(h3) (`silu_op u_silu`, MUL via `vec_alu2`)
   l. **PH_FW2** — matmul out=h·W2
   m. **RESID** — x += ffn_out
3. **PH_FNF** — RMSNorm final (`rmsnorm_op u_rms`)
4. **PH_LM** — lm_head : matmul logits=x·W_emb^T chunké en NCHUNK blocs (`mac18 u_fm_mac`)
5. **AMAX** — argmax sur VOCAB logits → `best_idx` = token émis
6. **DONE_ST** — émission token via `tx8_link` ; reboucle PH_EE jusqu'à NPOS=17 tokens

**4 maîtres SDRAM, fenêtres disjointes** : node_top.u_sdram (poids+embedding) + gen_seq (KV-cache via kv_cdc). Refresh SDRAM auto ~15 µs pendant les fenêtres KV (`E_MREFRESH/E_MREF_BUSY/E_MREF_DONE`).

---

## 4. Modules de calcul — DSP & BRAM (comptés sur le RTL)

DSP = opérateurs `*` arithmétiques + instanciations `mac18`. BRAM = `(* ram_style="block" *)` + `shared_lut`. **1 instance de chaque** dans le build silicium.

| Module | DSP (RTL brut) | BRAM DP16KD | Tables non-linéaires | Notes |
|---|---|---|---|---|
| `attention_head_op` | **2** (qk_prod `:201`, attn_v_prod `:210`) | 2 (exp+inv via shared_lut `:76,79`) | exp, inv | doc disait 3 DSP (post-synthèse Yosys décompose) |
| `rmsnorm_op` | **3** (x_sq `:113` + 2 dans prod_reg `:234`) | 1 (rsqrt_lut `:59`) | rsqrt | doc disait 4 DSP |
| `rope_op` | **4** (xr·cos, xi·sin, xr·sin, xi·cos `:80-83`) | 0 | — | cos/sin en ports (ROM dans gen_seq) |
| `silu_op` | **0** | 1 (silu_lut `:44`) | silu | aucun `*` arithmétique |
| `softmax_op` | **0–1** (requantif PH_NORM) | 2 (exp+inv via shared_lut `:53,56`) | exp, inv | K_MAX=32 hardcodé |
| `vec_alu2` | 1 (via mac18 `:35`) | **1** (acc 64×32 `:38`, inféré block RAM) | — | mesuré 2026-07-30 : 1 DP16KD / 0 DPR16X4 — le commentaire « LUT-RAM » du code est trompeur (aucun `ram_style` posé) |
| `mac18` | 1 (`:54`) | 0 | — | pipeline 3 stages, ×2 instances (node_top + vec_alu2) |
| `shared_lut` | 0 | 1 chacun (`:34`) | exp/inv/rsqrt/silu selon HEX | **4 instances** (2 attention + 2 softmax), pas 6 |

**DSP total RTL brut** : 2 (attn) + 3 (rms) + 4 (rope) + 1 (softmax) + 1 (vec_alu2/mac18) + 1 (node_top mac18) ≈ **12 DSP RTL**. Le total placé-routé (17/72) est supérieur car Yosys `SHARE` fusionne les mults mutuellement exclusifs **et décompose les mults larges** (ex. produits 16×16→32 peuvent devenir 2 DSP). L'écart RTL→post-synthèse est attendu et documenté.

---

## 5. Infrastructure & mémoire — BRAM détaillée

**Mesuré** le 2026-07-30 par `.\synth_stat.ps1 -Tag dim64 -Dim 64` (yosys `synth_ecp5
-noflatten` + `stat -top`, log `build/stat_dim64.log`). Colonnes = **compte local**,
hors sous-modules. Le total design (102) dépasse la somme des comptes locaux (100)
car les 2 modules `shared_lut` (exp, inv) sont chacun instanciés 2× (attention +
softmax) → 4 instances physiques.

| Module | BRAM DP16KD | LUT4 (local) | Détail | Instances |
|---|---|---|---|---|
| **`wcache_ffn`** | **90 (dim64) / 1 (dim128+)** | 454 | `mem [0:NFFN-1]` int8 (`wcache_ffn.v:31`), NFFN=184320 octets → 90 blocs. **Activé si CFG_NC>0** (cache-all dim64) ; réduit à 1 si CFG_NC=0 (stream). | 1 |
| `gen_seq` | 3 | 6 087 | logits 512×8 (`:155`), cos_rom + sin_rom RoPE (`:305,306`) | 1 |
| `top`/node_top | **2** | 6 582 | xbuf+wbuf (1024×8) + les 6 arrays `ram_style="block"` (`:535-577`) — yosys les **packe dans 2 DP16KD** : 5 arrays int8 192×8 + `tmp_prod` int32 192×32 ≈ 13,8 kbit + xbuf/wbuf 8,2 kbit = 22 kbit ≤ 2×16 kbit | 1 |
| `shared_lut` | 1 par module → **4 instances** | 0 | exp_lut + inv_lut dans attention ; exp_lut + inv_lut dans softmax (2 modules paramétrés, 2 instances chacun) | 4 |
| **`vec_alu2`** | **1** | 3 143 | `acc [0:D-1]` 64×32 (`vec_alu2.v:38`) — **inféré en BRAM**, cf. avertissement ci-dessous | 1 |
| `rmsnorm_op` | 1 | 841 | rsqrt_lut 256×16 | 1 |
| `silu_op` | 1 | 532 | silu_lut 256×16 | 1 |
| `attention_head_op` | 0 | 4 539 | ses tables exp/inv sont dans `shared_lut` ; stockage `ev_packed` en logique | 1 |
| `gen_seq_silicon` | 0 | 1 373 | FSM hôte | 1 |
| `flash_boot` | 0 | 838 | | 1 |
| `softmax_op` | 0 | 722 | tables dans `shared_lut` ; 8 DPR16X4 | 1 |
| `sdram` | 0 | 440 | | 1 |
| `rope_op`, `mac18`, `uart_*`, `async_fifo`, `kv_cdc`, `ecp5_pll`, `rx8/tx8_link` | 0 | < 100 chacun | async_fifo = LUT-RAM (2 DPR16X4) | — |
| **Total design** | **102** | **26 011** (+1 452 CCU2C) | 20 MULT18X18D, 10 916 FF, 266 DPR16X4 | — |

> ⚠ **`vec_alu2.acc` est en BRAM, pas en LUT-RAM.** Le commentaire du code
> (`vec_alu2.v:38` « products / aligned sums (LUT-RAM) ») et la révision précédente de
> ce document affirmaient l'inverse. Mesure : `vec_alu2` = **1 DP16KD et 0 DPR16X4**.
> L'array n'a **aucun attribut `ram_style`** — yosys choisit donc librement, et il
> choisit le block RAM. Pour le forcer en LUT-RAM (et rendre 1 bloc au budget), il
> faudrait annoter explicitement `(* ram_style = "distributed" *)`.

**LUT-RAM distribué réel** (TRELLIS_DPR16X4, 266 au total) : `gen_seq` 238 (ksh/vsh/csh
shift re-align, pkt, vfile), `top` 16, `softmax_op` 8, `async_fifo` 2 chacun.
`node_top.y_int32` est en registres (volontaire, `ram_style="registers"`).

---

## 6. Synthèse ressources & interprétation

### Build dim64 (validé silicium, 14 tok/s) — chiffres régénérés le 2026-07-27 (build 16:56)
| Ressource | Placé-routé (régénéré) | Vérifié sur code |
|---|---|---|
| LUT4 | **26 874 / 43 848 (61%)** (logic 22 630 + carry 2 696 + RAM 1 032 + RAMW 516) | — |
| DFF | 9 860 (22%) | — |
| **BRAM** | **102 / 108 (94%)** | **~90 wcache_ffn + ~12 tables/buffers = 102** ✅ |
| DSP | 17 / 72 (23%) | ~12 RTL brut → 17 post-synthèse (SHARE + décomposition) ✅ |
| Fmax | **41.46 MHz** (PASS @31.25) ; clk UART 28.72 MHz (PASS @25) | — |

Autres blocs : TRELLIS_IO 61/245 (24%), DCCA 2/56 (3%).

### 🔑 Le mur BRAM est le cache FFN, pas les tables
La saturation BRAM (94%) en dim64 est **dominée par `wcache_ffn` (90 blocs)**, le cache on-chip des poids FFN (W1/W3/W2 des 5 couches, CFG_NC=5 cache-all). Tout le reste ne fait que **12 blocs**, ventilation exacte (mesure §5) : 6 tables non-linéaires (shared_lut ×4 + rsqrt + silu) + 3 `gen_seq` (logits, cos_rom, sin_rom) + 2 `node_top` (buffers packés) + 1 `vec_alu2` (acc) = 12. Soit 90 + 12 = **102/108** ✅.

**Conséquence pour scaler** : passer à dim128/256 ne nécessite pas de toucher aux tables (déjà en BRAM, ~6 blocs fixes), mais de **désactiver le cache FFN on-chip** (CFG_NC=0, stream depuis SDRAM) — c'est exactement ce que fait le profil dim128 (CFG_NC=0 → wcache_ffn=1 bloc). Le coût est en **performance** (stream SDRAM plus lent que cache BRAM), pas en BRAM. Le mur BRAM est donc un choix architectural cache-vs-stream, pas un mur physique des tables.

### DSP sous-utilisé (23%)
~12 DSP RTL / 17 post-synthèse sur 72. Le levier de scale (tensor-parallel NN=2, cf. [[big-fpga-comparison-27b]]) multiplierait l'usage DSP. Les modules TP (`ffn_tp_seq2`, `attn_tp_seq`) sont présents mais NN=1 désactivé.

---

## 7. Builds — correction des écarts

| Build | Invocation | DIM | Cache FFN (CFG_NC) | Tables | Reproductible ? |
|---|---|---|---|---|---|
| **dim64** | `.\build_silicon.ps1` | 64 | 5 (cache-all, ~90 BRAM) | BRAM | ✅ |
| **dim128** | `.\build_silicon.ps1 -Dim128` | 128 | 0 (stream, 1 BRAM) | BRAM | ✅ |
| **dim256/512** | `.\build_silicon.ps1 -Dim256/-Dim512` | 256/512 | 0 | BRAM | ✅ |
| ~~baseline~~ | git stash des 4 modules pré-migration | 64 | 5 | LUT-RAM | ❌ manipulation git manuelle, pas un flag |

**Corrections vs ancien doc** :
- ❌ Ancien : « build flash DIM128 = tables en LUT-RAM (13 BRAM, 81% LUT) ». **Faux** : l'arbre courant (commit `fb8c451`) a migré les tables en BRAM **inconditionnellement** (pas de flag de bascule). Le build dim128 actuel a les tables en BRAM ; sa faible consommation BRAM vient de CFG_NC=0 (wcache désactivé), pas des tables en LUT.
- ❌ Ancien : « baseline = 3ᵉ configuration sélectionnable ». **Faux** : baseline = git stash manuel, pas une cible de build.
- ❌ Ancien : instances « gen_seq ×7, attention ×3, rmsnorm ×5, rope ×4, mac18 ×2, shared_lut ×6 ». **Faux** : 1 instance de chaque module (sauf mac18 ×2 et shared_lut ×4, confirmés). Les chiffres anciens étaient des artefacts du rapport Yosys `stat`.
- ✅ Ancien : « vec_alu2 = 1 BRAM ». **Correct** — la révision du 2026-07-27 avait corrigé ce point à tort (en se fiant au commentaire du code) ; la mesure `-noflatten` du 2026-07-30 rétablit **1 DP16KD**. Cf. §5.
- ❌ Ancien : « wcache_ffn = 1 BRAM ». **Faux** : ~90 BRAM en dim64 (le composant dominant du mur BRAM).

---

## 8. Le mur LUT était en partie artificiel — interface élément-série (2026-07-30)

Le scaling de la LUT avec `dim` ne venait **ni de la taille du die, ni du calcul**, mais d'un
anti-pattern d'interface. Diagnostic obtenu en comparant les LUT par module entre deux
paliers (`synth_stat.ps1 -Dim 64` vs `-Dim 128`) :

| module | LUT dim64 | LUT dim128 | ratio |
|---|---|---|---|
| `top`/node_top | 6 582 | 10 914 | 1,66 |
| `gen_seq` | 6 087 | 9 473 | 1,56 |
| `vec_alu2` | 3 143 | 5 514 | **1,75** |
| `attention_head_op` | 4 539 | 4 619 | **1,02** |
| `softmax_op`/`silu_op`/`rmsnorm_op` | 2 095 | 2 085 | **1,00** |

Les opérateurs arithmétiques ne scalent **pas** (ils accèdent aux données par BRAM ou en
flux). Trois modules portaient tout le scaling. Cause dans `vec_alu2` : le calcul est
sérialisé (1 élément/cycle) mais les opérandes arrivaient en **ports plats `[W*D-1:0]`** —
chaque `a[i*W +: W]` avec `i` variable est un **multiplexeur D:1**. Soit 6 structures `D:1`
(2 mux + 1 démux dans l'ALU, 2 démux de remplissage + 1 mux de relecture dans `gen_seq`),
3·D·8 fils, et 3 tampons plats `alu_a`/`alu_b`/`alu_out` qui ne servaient **qu'à recopier**
`vfile` → ALU → `vfile` élément par élément (états `ALD_A`/`ALD_B`/`AST_A`, 3·D cycles).
Les deux extrémités étaient déjà sérielles : l'interface parallèle était pure perte.

**Correctif** : l'ALU pilote `rd_idx`/`rd_sel` et reçoit l'octet depuis `vfile` ; elle
pilote `wr_idx`/`wr_el`/`wr_we` routés vers le port d'écriture `vfile` existant.
L'**adressage remplace le multiplexage**, et les 3·D cycles de copie disparaissent.

### Gains mesurés (yosys `-noflatten`, LUT4+carry sur 43 848)

| profil | avant | après | gain | BRAM | bascules |
|---|---|---|---|---|---|
| dim64 | 27 463 (62,6 %) | **23 704 (54,1 %)** | −13,7 % | 102 → **101** | −1 533 |
| dim128 | 37 073 (84,5 %) | **29 255 (66,7 %)** | −21,1 % | 13 → 20 | −3 040 |
| dim256 | 55 571 (**126,7 %**) | **41 615 (94,9 %)** | −25,1 % | 13 → 20 | −6 110 |

`vec_alu2` à dim256 : 8 736 → **1 076** LUT (−88 %) — il atteint enfin la cible « a few
hundred LUT » annoncée par son propre en-tête. `gen_seq` : 16 176 → 9 541.
Le gain croît avec `D`, signature attendue puisqu'on a supprimé des structures en `O(D)`.

**Validation** : sim Verilator **17/17 bit-exact** à dim64 (`tok[16]=264`, `best_v=81`, XB
identique bit à bit à la baseline) **et** à dim128 (`tok[16]=60`, `best_v=127`).
Gain en cycles : −76 160 sur 28,9 M, soit **−0,26 % seulement** (le decode est dominé par
les matmuls SDRAM — ce refactor est un gain de surface, pas de vitesse).

### ✅ Confirmé sur silicium (Colorlight i9, 2026-07-30)
Bitstream dim64 refactoré flashé, capture `infer_g5autonomous.py` :
`[403,407,261,378,432,383,286,261,376,268,414,421,303,428,415,412,264]` — **identique bit
pour bit** à la capture de référence, 9/10 vs oracle, `[PASS]`. Le gain se matérialise après
placement-routage réel :

| P&R dim64 | avant | après | delta |
|---|---|---|---|
| Total LUT4 | 26 874 (61 %) | **23 699 (54 %)** | **−3 175** |
| dont logique | 22 630 (51 %) | 18 803 (42 %) | −3 827 |
| DP16KD | 102 (94 %) | **101 (93 %)** | **−1** |
| `clk_sys` | 41,46 MHz | 39,87 MHz | PASS @31,25 |
| `clk` (IO/SDRAM) | 28,72 MHz | **31,96 MHz** | **améliorée** |

L'horloge IO **gagne** 3,2 MHz : supprimer les multiplexeurs `D:1` raccourcit les chemins
combinatoires critiques. `clk_sys` recule de 1,6 MHz mais garde 27 % de marge.

⚠ **Portée des chiffres** : `-noflatten` préserve les frontières de modules et surestime
le flow de build réel (ce doc donnait 49 095 à dim256 en flatten, contre 55 571 mesuré ici).
Le **delta** avant/après est valide (même flow, même code) ; le **% absolu** exigeait un P&R
nextpnr pour être affirmé (garde-fou G1 « ça route ≠ ça marche ») — c'est fait, ci-dessous.

### ✅ P&R dim256 réussi — le palier « cluster requis » tombe (2026-07-30)
`.\build_silicon.ps1 -Dim256` produit un **bitstream** avec **0 erreur** et le timing tenu,
là où l'état antérieur documenté était `nextpnr fail` (LUT 112 %). Rapport `gen_seq_silicon.rpt` :

| ressource | utilisé | % |
|---|---|---|
| TRELLIS_COMB (logique) | 40 099 / 43 848 | **91,5 %** |
| TRELLIS_FF | 17 607 / 43 848 | 40,2 % |
| TRELLIS_RAMW | 612 / 5 481 | 11,2 % |
| **DP16KD** | **20 / 108** | **18,5 %** |
| MULT18X18D | 15 / 72 | 20,8 % |
| `clk_sys` | **36,67 MHz** | PASS @ 31,25 |
| `clk` (IO/SDRAM) | 25,12 MHz | PASS @ 25,00 — **marge 0,5 % seulement** |

**Réserves à garder en tête** :
1. **91,5 % de logique et 0,5 % de marge sur l'horloge IO** : ça route, mais c'est tendu.
   Une variation de process/température, ou tout ajout de logique, peut faire échouer le
   timing. À ne pas considérer comme un budget confortable.
2. ~~**Le modèle dim256 ne tient pas en SDRAM** : ~13 Mo contre 8 Mo~~ — **FAUX, corrigé le
   2026-07-30.** Les « ~13 Mo » de `docs/CAPACITE_FPGA.md` valent pour **vocab=32000**
   (tinyllamas), alors que ce projet construit **`CFG_VOCAB = 512`** par défaut (32000 n'est
   activé que par le define `VOCAB32000`). Empreinte réelle des poids, calculée sur les
   formules `CFG_OFF_*` de `model_cfg.vh` :

   | profil | vocab=512 (ce projet) | vocab=32000 |
   |---|---|---|
   | dim64 | 0,27 Mo | 2,19 Mo |
   | dim128 | 0,96 Mo | 4,81 Mo |
   | **dim256** | **3,64 Mo → tient** | 11,33 Mo → non |
   | dim512 | 14,16 Mo → non | 29,54 Mo → non |

   **dim256 tient donc largement dans les 8 Mo de la i9.** Le mur SDRAM commence à
   **dim512**, pas à dim256.
3. **Le bit-exact dim256 n'est pas validé** (drift d'oracle préexistant, cf. `CAPACITE_FPGA.md`).
   Le refactor est validé bit-exact à dim64 et dim128, pas à dim256.
4. **BRAM à 18,5 %** : 88 blocs libres — large marge pour les migrations suivantes (node_top).

Conséquence stratégique : le cluster tensor-parallel était requis dès dim256 pour **deux**
murs supposés (LUT et SDRAM). Le mur LUT est levé par ce rework, et le mur SDRAM n'existait
pas à ce palier (3,64 Mo pour 8 Mo disponibles). **dim256 est donc atteignable sur une seule
i9.** Le premier palier réellement bloquant devient **dim512** (14,16 Mo de poids), où le
cluster ou le weight-streaming redeviennent nécessaires.

### Piège : fusionner des états supprime des délais implicites
Fusionner `AST_A` dans `ALU_WAIT` a d'abord cassé le bit-exact (8 mismatch/17, divergence
progressive) : `AST_A` intercalait D cycles entre `sov<=alu_out_sh` et la relance
`alu_sb<=sov`, donc `sov` y était déjà à jour. Dans un cycle unique, le non-bloquant fait
capturer l'**ancien** `sov` → shift de requantification périmé sur le résiduel FFN.
Correctif : `alu_sb<=alu_out_sh`. Localisé par banc différentiel (ancienne vs nouvelle ALU
sur les mêmes vecteurs, 8 cas) qui a innocenté l'ALU et pointé l'intégration.

### Suite appliquée à node_top (même journée)
Le même anti-pattern a ensuite été traité dans `node_top`, en deux lots, chacun validé
**sim 17/17 + silicium bit-exact** :

1. **`x_save_packed` / `Out_packed` → arrays indexés** (`x_save`, `out_bytes`). Tous leurs
   accès étaient déjà élément-par-élément : le part-select ne produisait qu'un démux `D:1`
   en écriture et un mux `D:1` en lecture. Sûr par construction — une lecture asynchrone ne
   peut pas aller en DP16KD, donc yosys produit de la LUT-RAM, sans latence (confirmé : même
   nombre de cycles à l'unité près). dim256 : node_top 19 702 → 14 835 LUT (**−24,7 %**).
2. **`y_int32` : retrait de `ram_style="registers"`** — une ligne, le plus gros gain.
   L'attribut imposait `D·32` bascules **plus un mux `D:1` par lecture** sans nécessité (les
   trois lectures sont combinatoires). dim256 : node_top 14 835 → **6 650** LUT (**−55 %**),
   −8 167 bascules, soit exactement les 8 192 bits de l'array.

> ⚠ **Vérification qui a failli manquer.** Sur `y_int32`, yosys mappe via `$__PDPW16KD_`,
> donc en **bloc BRAM** — et non en LUT-RAM comme attendu (`DPR16X4` inchangé, `DP16KD` +1).
> La justification « une lecture asynchrone ne peut pas aller en DP16KD » était **fausse**.
> Or ce projet a déjà été mordu par un mapping BRAM à adresse enregistrée (donnée périmée
> d'un cycle, cf. migration des tables), et **Verilator simule le RTL source, pas la
> netlist** : la sim ne pouvait pas trancher. Le silicium a validé. **Règle générale : un
> changement dont le risque est dans la synthèse, et non dans le RTL, exige une validation
> silicium — la sim RTL n'est pas un arbitre valable.**

> ⚠ **Contrepartie à dim64** : `y_int32` en BRAM consomme 1 DP16KD (101 → 102, soit 94 % du
> budget) pour libérer de la LUT, qui n'est pas le mur à ce profil. C'est favorable à
> dim128+ (BRAM à 18 %, LUT était le mur) et neutre-à-défavorable à dim64. Il reste 6 blocs,
> donc ça passe ; si le budget BRAM dim64 redevient critique,
> `(* ram_style = "distributed" *)` rend le bloc contre ~2 200 LUT.

**`Q_packed` volontairement non migré** : `Q_flat = Q_packed[mh_h*HS*8 +: HS*8]` lit `HS`
octets simultanément pour alimenter l'attention — besoin parallèle réel, contrairement à
`vec_alu2`. En array il faudrait `HS` ports de lecture asynchrone, donc une duplication ×16
de la mémoire à dim256. Le coût est ici justifié.

### Bilan cumulé
| profil | origine | final | gain |
|---|---|---|---|
| dim64 (P&R, Total LUT4) | 26 874 (61 %) | **20 325 (46 %)** | −24,4 % |
| dim256 (`-noflatten`) | 55 571 (**126,7 %**) | **28 535 (65,1 %)** | −48,7 % |
| **dim256 (P&R réel)** | 112 % (`nextpnr fail`) | **26 239 LUT4 — 59 %**, 43,91 MHz PASS | — |
| **dim512** (`-noflatten`) | 82 851 (**189 %**) | **39 116 (89,2 %)** | **−52,8 %** |

Deux paliers déclarés hors d'atteinte le sont redevenus :
- **dim256 route avec une marge confortable** (59 % du die après P&R, BRAM 20 %, DSP 20 %) et
  **tient en mémoire** (3,64 Mo pour 8 Mo). Il est atteignable sur une seule i9.
- **dim512 rentre désormais en logique** (89,2 % en synthèse, ~80 % attendu après P&R, contre
  189 % avant). Il reste bloqué par la **seule** taille du modèle : 14,16 Mo contre 8 Mo de
  SDRAM *et* 8 Mo de flash — donc le weight-streaming ne le lève pas. Leviers : `NL=2`
  (5,81 Mo, simple paramètre) ou `HID=1·D` (6,66 Mo, mais `NCHUNK_F=3` est hardcodé dans la
  FSM). Baisser le vocabulaire est inutile à ce palier (0,25 Mo, soit 1,8 % du total).

Le bit-exact a été vérifié à chaque étape : sim sur deux paliers et trois validations silicium.

**Prochain gisement, si l'on vise dim512** : la hiérarchie s'est inversée. À dim512 c'est
désormais **`gen_seq` qui domine** (15 826 LUT contre 10 975 pour node_top), via son `vfile`
(`NSLOTS*D` = 8×512 octets) et `pkt`.

### Prochain gisement (non traité)
À dim256, `top`/node_top pèse **19 702 LUT sur 40 197** (~49 % du design) et conserve le
même anti-pattern : `y_int32` (`ram_style="registers"`, 8 192 bits à dim256, commentaire
encore marqué « FQ uniquement, N <= 64 » — hypothèse périmée depuis dim128),
`x_save_packed`/`Q_packed`/`Out_packed` (3·D octets en regs plats + part-select), puis
`gen_seq.vfile`/`pkt`. Hors mur `dim` : les 3 regs plats `T_MAX` de `attention_head_op`
(~4 600 LUT constants, justifiés par un « Force **Gowin** à NE PAS inférer BSRAM » sans
objet sur ECP5). **Le mur SDRAM reste physique** (13 Mo à dim256, 37 Mo à dim512 vs 8 Mo) :
ce refactor lève le mur LUT, pas l'autre.

---

## 9. Limites & à faire

- **Totaux LUT4/DFF/Fmax régénérés le 2026-07-27** (build dim64 16:56) : **26 874 LUT4 / 102 BRAM / 17 DSP / 41.46 MHz** — confirmés, identiques au build antérieur (le code n'avait pas bougé sur les totaux ; seule l'interprétation instances/BRAM/build-flash était fausse). Log : `build/build_silicon_viewer.log`. JSON viewer : `build/gen_seq_silicon.pnr.json` (90 MB) + `build/gen_seq_silicon.rpt`.
- ~~**Compte BRAM exact par module**~~ : **fait le 2026-07-30** (`.\synth_stat.ps1 -Tag dim64 -Dim 64`, log `build/stat_dim64.log`) — cf. §5. Résultats notables : `wcache_ffn` = 90 blocs confirmé ; les 6 arrays de node_top sont packés en **2** blocs (et non ~8) ; `vec_alu2` consomme **1** bloc (et non 0). Le script accepte désormais `-Dim 64|128|256|512` (il était figé sur `-DDIM128`, ce qui mesurait le mauvais profil pour le mur BRAM).
- ~~**`src/top.v`** (legacy, doublon de `module top`)~~ : **fait le 2026-07-30** — déplacé en `src/legacy/top_gowin_monolith.v` (aucun flow ne le compilait). Motifs et références documentés dans `src/legacy/README.md`.
- **`vec_alu2.acc`** : 1 DP16KD récupérable en annotant `(* ram_style = "distributed" *)` — à évaluer si le budget BRAM redevient critique (coût : ~64 DPR16X4 supplémentaires).
- **DSP** : 20 MULT18X18D en `-noflatten` (§5) vs 17 après P&R flatten (§6). L'écart vient du flow (le flatten permet à yosys de partager davantage), les deux chiffres sont corrects dans leur contexte.
- **`softmax_op K_MAX=32` hardcodé** : à paramétrer via `CFG_TMAX` si T_MAX doit évoluer.