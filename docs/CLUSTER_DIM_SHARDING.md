# Cluster tensor-parallel — sharder `dim` sur N FPGA

**Statut** : extension architecturale **suivante** (non livrée). Doc de référence
pour la garder en tête et la concevoir quand on attaquera `dim=512` (tinyllamas).
Single-FPGA validé silicium aujourd'hui : `dim=128` (cf. `docs/CAPACITE_FPGA.md`).

## Objectif

Répartir la dimension `dim` (`CFG_D`) sur N nœuds FPGA pour franchir les deux murs
qui bloquent `dim≥256` sur un seul ECP5-45F :

- **Mur LUT** : 87 % à dim=128 → 112 % (dim=256) → 189 % (dim=512). Le datapath
  sérialisé (1 elem/cycle via mac18) fait que la LUT scale ~avec D (buffers
  `y_int32[D]`, `alu_a[D*8]`, `xbuf`), pas avec D². Un seul nœud déborde dès dim=256.
- **Mur SDRAM** : poids ~D² → 5 Mo (dim=128) / 13 Mo (dim=256) / 37 Mo (dim=512).
  La i9 n'a que 8 Mo.

Le cluster tensor-parallel adresse **les deux** (LUT via sharding du compute,
SDRAM via sharding des poids) — à condition d'être réellement câblé.

> **Rappel** (cf. `docs/CAPACITE_FPGA.md`) : le « tensor-parallel » actuel n'est
> qu'un placeholder. `CFG_NN=1` hardcodé (`model_cfg.vh:101` « première livraison :
> single-node »), `node_top` utilise le `D` complet partout, `ring_cluster.v` est
> un banc de test d'all-gather standalone non intégré au datapath. Ce doc décrit
> le rework à faire pour le rendre réel.

## La méthode : tensor-parallel (Megatron-style)

On shard `D` sur N nœuds. Schéma standard pour un transformer :

### 1. Attention — sharde les têtes (split propre)
- `H = D/HS` têtes **indépendantes** (HS=16 fixe). Chaque nœud gère `H/NN` têtes.
- Pas de réduction intra-tête (les têtes ne se mélangent pas avant la concat finale).
- Le projet a déjà `H = D/16` paramétrique → découpe naturelle.
- Sortie par nœud : `D/NN` (ses `H/NN` têtes × HS). All-gather → full D pour la
  projection de sortie WO.

### 2. Projections linéaires (WQ/WO/W1/W2…) — shard les colonnes
- **Entrée `X` (full D) répliquée** sur tous les nœuds (standard Megatron :
  input replicé, poids + output sharded).
- Chaque nœud charge `W[:, d*D/NN +: D/NN]` (sa tranche de colonnes) et calcule
  sa tranche de sortie `D/NN`.
- Pour les poids où l'**entrée** est la dimension sharded (ex. W2 `[HID, D]`,
  W_lm `[VOCAB, D]`), les partiels doivent être **all-reduce** (somme) pour
  reconstituer le full D.

### 3. Réductions — RMSNorm, résiduel, argmax
- RMSNorm / résiduel : ont besoin du full D → all-gather après le shard, ou
  garder le résiduel **répliqué** (full D sur chaque nœud, seuls les I/O matmul
  sont sharded — le résiduel D int8 = D octets, négligeable).
- lm_head/argmax : shard along `VOCAB` (chaque nœud calcule `VOCAB/NN` logits)
  → all-gather → argmax global ; ou shard along `D` (input) → all-reduce sum
  → full logits → argmax.

## Pourquoi ce design est **favorable** au cluster

Le ratio **compute/comm** est excellent — c'est le critère qui rend le
tensor-parallel efficace (vs. comm-bound) :

| | par couche | ordre de grandeur à dim=512 |
|---|---|---|
| **Compute** | ~D² mac (matmul sérialisé 1 elem/cycle) | millions de cycles |
| **Comm** | ~D octets (un vecteur D à all-reduire) | ~1000-1500 cyc (à 2-3 cyc/byte, cf. `gen_seq.v` « PH_MM ss_link-bound ~2-3 cyc/byte ») |

→ Comm **< 1 %** du compute. Le cluster est **compute-bound**, pas comm-bound.
Scaling attendu ~×NN en throughput, ~1 % d'overhead. Beaucoup de designs
parallèles sont limités par la comm — pas ici, parce que le compute est lourd
(D²) et la comm légère (D).

## Infra existante (prototype, non intégré)

- `src/ss_link.v` : lien source-synchrone parallèle inter-FPGA (1 direction),
  CDC-ready (async FIFO). Commentaire `gen_seq.v` : ~2-3 cyc/byte.
- `src/link_send.v` / `link_recv.v` : stream d'un vecteur L-elements sur ss_link.
- `src/ring_cluster.v` : **N nœuds en anneau, all-gather scalable** (« after the
  all-gather every node holds the full vector (all N slices) »). Topologie :
  edge g : nœud g → nœud (g+1)%N, 1 in + 1 out par nœud (ajouter des nœuds
  n'ajoute pas de liens/nœud). Banc de test standalone (horloge unique,
  « N-scalability test ») — **non branché** au chemin de calcul.
- `board/colorlight_i9.lpf:84` : TODO pins PMOD pour le ss_link 2-cartes
  (8 data + 1 frame + 1 clk + GND) — pas encore contraint.
- `CFG_NN` (`model_cfg.vh:101`) : placeholder à `1`.

## Rework à faire (le travail réel)

### A. Côté RTL (`node_top.v`)
- Sharder les buffers/FSM dot : `y_int32[D]` → `y_int32[D/NN]`, `alu_a[D*8]` →
  `alu_a[D/NN * 8]`, `dot_k` jusqu'à `D/NN` (pas `D`), têtes réparties.
  → per-node LUT ≈ niveau dim=128 (87 %) → tient.
- **Bonne nouvelle** : le rework dim-paramétrable déjà fait rend ça net — les
  largeurs sont déjà en `$clog2(D)`, il suffit de les recaler sur `D/NN` (un
  `localparam DN = D/CFG_NN` et remplacer les largeurs datapath par `DN`).
- Intégrer l'all-gather/all-reduce `ring_cluster` dans le FSM : après chaque
  projection sharded, réduire/reconstituer le full D avant la norme/résiduel.

### B. Côté host (preload)
- `gen_sdram_preload_hex.py` / `gen_flash_model.py` : each node ne charge que
  **sa tranche de poids** (`W[:, d*D/NN +: D/NN]`). Réduit SDRAM/nœud de ~NN×
  pour les poids. Le dump modèle reste full, le partitionnement se fait au
  moment du preload (un dump par nœud, ou un dump + offset par nœud).
- L'oracle `prove_gen_node.py` doit aussi sharder pour cross-check bit-exact
  (sinon on compare un nœud seul à l'oracle full-D).

### C. Câblage de l'anneau
- Brancher `ring_cluster` au datapath `node_top` (aujourd'hui standalone).
- Gérer l'all-reduce (somme des partiels FFN) : ring reduce-scatter + all-gather,
  ou all-reduce direct sur l'anneau.
- Pins PMOD ss_link à contraindre dans le LPF (`colorlight_i9.lpf:84`).

### D. Matériel / mémoire
- Stock i9 (8 Mo SDRAM) à NN=4 pour dim=512 = ~8-9 Mo poids/nœud → **borderline**.
  Options : NN≥5 (ou NN=8 marge), **ou** carte custom avec plus de SDRAM
  (alors NN=2-4 tient), **ou** weight-streaming.

## Les deux murs et leurs leviers (rappel)

| Mur | Levier sharding (cluster) | Levier streaming |
|---|---|---|
| **LUT** (189 % à dim=512, buffers `y_int32[D]` etc.) | ✅ shard D → buffers `[D/NN]` → 87 %/nœud | ❌ streaming ne rétrécit pas les buffers d'activation → reste 189 % |
| **SDRAM** (37 Mo poids) | ✅ ~NN× moins de poids/nœud | ✅ streamer les poids (FFN déjà streamé en NC=0, étendre aux poids d'attention) décorrèle `dim` de la SDRAM |

→ Pour `dim=512` : le mur LUT exige le sharding (cluster), le mur SDRAM admet
sharding **ou** streaming. Donc **sharding obligatoire**, streaming optionnel
(mais permet un NN plus petit / une i9 stock). Les deux murs ne sont pas levés
par le même levier — c'est pourquoi « juste ajouter des nœuds » (sharding seul)
tombe sur le mur SDRAM, et « juste streamer » (sans cluster) tombe sur le mur LUT.

## Estimation de gain

- Compute sérialisé : matmul `X·W` avec `W [D, D/NN]`/nœud = `D²/NN` mac/nœud,
  N nœuds en parallèle → throughput `D²/NN` wall-clock → **~×NN speedup**
  (comm ~1 %). dim=512 à NN=4 ≈ 4× plus rapide qu'un nœud seul (qui ne tient
  pas de toute façon).
- LUT par nœud : ~87 % (niveau dim=128). SDRAM/nœud : ~poids/NN (+embedding/KV
  gérables).

## Ouvertures / alternatives

- **Pipeline-parallel** (sharder les couches, pas D) : chaque nœud garde le
  full-D datapath → **ne lève pas le mur LUT** (189 %/nœud). À éviter pour
  `dim` ; pertinent si on voulait distribuer la profondeur (NL), pas la largeur.
- **Weight-streaming seul (sans cluster)** : lève le mur SDRAM mais pas le mur
  LUT → un seul nœud reste 189 % LUT à dim=512. Insuffisant.
- **Carte custom (ECP5-85F + SDRAM étendue)** : repousse les deux murs, mais
  ne rend pas le rework dim-paramétrique inutile — juste déplace les paliers.

## Ordre suggéré pour livrer le cluster

1. **Shard les poids côté host** + oracle sharded (validate bit-exact 1 nœud seul
   avec sa tranche vs oracle tranche — avant tout câblage inter-FPGA).
2. **Rework `node_top` datapath** en `DN = D/NN` (buffers/FSM dot). Valide
   sim single-node sharded bit-exact vs oracle sharded.
3. **Intégre `ring_cluster`** all-reduce au datapath. Valide sim multi-nœud
   (N instances + ss_link) bit-exact vs oracle full-D.
4. **Contraintes LPF ss_link** + silicium 2-cartes, puis N-cartes.
5. Monter à NN=4 (ou 8) + weight-streaming pour tinyllamas dim=512.

## Liens

- `docs/CAPACITE_FPGA.md` : courbe de capacité mesurée (le mur actuel).
- `src/ring_cluster.v`, `src/ss_link.v`, `src/link_send.v`, `src/link_recv.v` :
  proto existant.
- Mémoire `dim-rework-progress` : état du rework dim + caveat cluster.