# Comprendre les réglages d'un modèle LLM et leur impact sur le FPGA

Document de vulgarisation — explique, sans jargon technique, ce que sont les
principaux réglages qui définissent un modèle de langage (LLM) et ce qu'ils
coûtent à faire tourner sur un FPGA (ici l'ECP5-45F de la carte Colorlight i9).

Cible : comprendre **pourquoi tinyllamas ne tient pas sur la puce actuelle**
alors que stories260K tourne à 14 tokens/s, et **quel réglage est le mur**.

---

## 1. Les 5 réglages d'un modèle LLM

Un modèle de langage se décrit par 5 nombres indépendants. Pour bien les
visualiser, imagine le modèle comme **un immeuble avec un annuaire** :

| Réglage | Nom complet | Analogie | stories260K | tinyllamas (15M) |
|---|---|---|---|---|
| `vocab` | vocabulaire | taille de l'annuaire (nb de mots connus) | 512 | 32 000 |
| `dim` | dimension | largeur de chaque étage (richesse de la pensée) | 64 | 512 |
| `NL` | n_layers (couches) | hauteur de l'immeuble (profondeur du raisonnement) | 5 | 12 à 32 |
| `HS` | head_size | largeur d'une "salle de lecture" dans l'attention | 8 | 48 à 64 |
| `T_MAX` | seq_len | mémoire à court terme (mots gardés en tête) | 32 | 256 à 1024 |

> **Important** : `dim` n'est **pas** le nombre de couches. C'est la taille du
> vecteur qui représente un mot. Le nombre de couches, c'est `NL`. Les deux sont
> indépendants : on peut avoir un modèle "large mais peu profond" (grand `dim`,
> petit `NL`) ou "étroit mais profond" (petit `dim`, grand `NL`).

### Schéma : l'immeuble

```
                  ┌───────────────────────┐
   toit (sortie)  │   lm_head → mot choisi │  <- produit 1 score par mot du vocab
                  └───────────────────────┘
                          ▲  vecteur de dim nombres
                          │
                ┌─────────┴─────────┐      ◄── NL = hauteur (nombre d'étages)
   étage 5      │  attention + FFN  │           empilés. Chaque étage affine
                └───────────────────┘           le vecteur du précédent.
                ┌───────────────────┐
   étage 4      │  attention + FFN  │
                └───────────────────┘
                ┌───────────────────┐
   étage 3      │  attention + FFN  │      ◄── dim = largeur de l'étage
                └───────────────────┘           (nombre de "bureaux" côte à côte).
                ┌───────────────────┐           Le vecteur garde cette largeur
   étage 2      │  attention + FFN  │           du rez-de-chaussée au toit.
                └───────────────────┘
                ┌───────────────────┐
   étage 1      │  attention + FFN  │
                └───────────────────┘
                          ▲  vecteur de dim nombres
                          │
                  ┌───────────────────────┐
  rez-de-chaussée │  embedding : mot → vecteur
                  └───────────────────────┘
                          ▲
                          │
                  annuaire de vocab mots  ◄── vocab = nombre d'entrées
```

---

## 2. `vocab` — le dictionnaire

`vocab` = **le nombre de mots différents** que le modèle connaît et peut
produire. Il intervient **deux fois** dans le pipeline, pas une.

### Schéma : la table d'embedding

```
   annuaire (table d'embedding) : une ligne par mot, dim nombres par ligne

        col 0   col 1   col 2   ...   col (dim-1)
       ┌───────┬───────┬───────┬─────────────────┐
mot 0  │  ...  │  ...  │  ...  │      ...        │  ──┐
mot 1  │  ...  │  ...  │  ...  │      ...        │    │  lookup pendant
mot 2  │  ...  │  ...  │  ...  │      ...        │    │  le prefill :
  ...  │  ...  │  ...  │  ...  │      ...        │    │  mot → vecteur
mot 511│  ...  │  ...  │  ...  │      ...        │  ──┘  (stories260K)
mot 512│  ...  │  ...  │  ...  │      ...        │  ◄── tinyllamas continue
  ...  │  ...  │  ...  │  ...  │      ...        │      jusqu'à 31 999
mot 31999 ...                                              │
       └───────┴───────┴───────┴─────────────────┘
        ▲
        │
   vocab = nombre de lignes

   taille totale de la table = vocab × dim nombres
```

### Où `vocab` frappe

1. **Au départ (prefill)** : chaque mot du prompt est cherché dans la table
   → donne le vecteur d'entrée. Si `vocab` est trop petit, le modèle ne connaît
   pas les mots au-delà.
2. **À la fin (lm_head)** : le vecteur final est converti en **un score pour
   chacun** des `vocab` mots, pour choisir le suivant. Plus le dictionnaire est
   gros, plus cette "tête de sortie" est large et lente.

---

## 3. `dim` — la richesse de la pensée

`dim` = **la taille du vecteur** qui représente un mot à l'intérieur du modèle.
Pendant l'embedding, chaque mot devient un vecteur de `dim` nombres. Ce vecteur
traverse tous les étages en gardant la même largeur, puis sert à prédire le mot
suivant.

### Schéma : le vecteur qui traverse le modèle

```
   mot "chat"
       │
       ▼  embedding
   ┌──────────────────────────────────────────┐
   │ [0.2, -1.3, 0.8, ..., 0.5]               │  ◄── dim nombres
   └──────────────────────────────────────────┘      (stories260K : 64 nombres
       │                                              tinyllamas : 512 nombres)
       ▼
   ┌──────────────────────────────────────────┐
   │           étage 1 (attention + FFN)      │  ◄── même largeur dim
   └──────────────────────────────────────────┘
       ▼
   ┌──────────────────────────────────────────┐
   │           étage 2 (attention + FFN)      │  ◄── même largeur dim
   └──────────────────────────────────────────┘
       ...
       ▼
   ┌──────────────────────────────────────────┐
   │           étage NL (attention + FFN)     │  ◄── même largeur dim
   └──────────────────────────────────────────┘
       │  vecteur final, toujours dim nombres
       ▼
   lm_head → score pour chacun des vocab mots → argmax → mot suivant
```

> Plus `dim` est grand, plus le modèle capture de nuances — mais plus tous les
> "tuyaux" internes doivent être larges, et plus les matrices de poids
> (`[dim × dim]`) grossissent en **dim × dim**.

---

## 4. `T_MAX` — la mémoire de conversation

`T_MAX` = **combien de mots le modèle peut garder en tête** d'un seul coup
(la fenêtre de contexte). À chaque mot généré, le modèle compare le mot courant
avec **tous** les mots précédents qu'il a mémorisés (le "cache KV").

### Schéma : le cache KV grandit avec la conversation

```
   Position dans la conversation (T_MAX = nb max de mots mémorisés)

   stories260K (T_MAX = 32) :
   ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
   │M1│M2│M3│M4│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│M32│
   └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
    ◄────────────────── 32 mots en mémoire ──────────────────►

   tinyllamas (T_MAX = 1024) :
   ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬ ... ┐
   │M1│M2│M3│M4│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│..│M32│ ... │
   └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴ ... ┘
    ◄──────────────────────── 1024 mots en mémoire ─────────────────────────────────────────────►

   À chaque nouveau mot, le modèle compare avec TOUS les mots précédents :
   - T_MAX=32   →  32 comparaisons par token
   - T_MAX=1024 → 1024 comparaisons par token  → ~32× plus de travail
```

> Plus la conversation s'allonge, plus chaque token est lent, parce qu'il faut
> comparer le mot courant à tout l'historique mémorisé.

---

## 5. Les 3 ressources du FPGA

Sur la puce et la carte, trois ressources sont en jeu — et chaque réglage ne
les sollicite pas pareil.

```
   ┌──────────────────────────────────────────────────────────────┐
   │  FPGA  (ECP5-45F)                                            │
   │                                                              │
   │   BRAM  = mémoire INTERNE à la puce                          │
   │           • très rapide (accès 1 cycle)                      │
   │           • très limitée (~90 petits blocs)                  │
   │           • déjà ~utilisés à 90 (cache FFN de stories260K)   │
   │                                                              │
   │   LUT   = cellules de logique (le "cerveau" calculatoire)    │
   │           • ~61 % utilisés aujourd'hui                      │
   └───────────────────────────┬──────────────────────────────────┘
                               │  bus (bande passante limitée)
                               ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  SDRAM  (mémoire EXTERNE sur la carte)                       │
   │           • grosse (Mo)                                      │
   │           • lente (aller-retour via le bus)                  │
   │           • stocke : poids du modèle, embedding, cache KV    │
   └──────────────────────────────────────────────────────────────┘

   Vitesse = tokens/seconde produites (dépend des 3 ci-dessus)
```

---

## 6. Impact de chaque réglage sur le FPGA

### `dim` (64 → 512, ×8) — **le plus lourd, frappe partout**

C'est la largeur de tous les tuyaux. Multipliée par 8, **tout** s'élargit :

- **Vecteurs internes** (xbuf, wbuf, Q_packed) : 8× plus gros → **8× plus de BRAM**.
- **Poids des couches** : matrices `[dim × dim]` → taille en **dim × dim**, donc **64× plus de place**.
- **Calcul** : un dot product passe de 64 à 512 multiplications → **8× plus de cycles**.

```
   dim = 64                        dim = 512
   ┌────────┐                      ┌────────────────────────────────────────┐
   │ vecteur│ 64 octets            │ vecteur                                │ 512 octets
   └────────┘                      └────────────────────────────────────────┘
   matrice [64×64]                 matrice [512×512]
   = 4 Ko                          = 256 Ko   (×64)
   BRAM : tient                    BRAM : une couche dépasse toute la puce
```

> **Verdict** : à dim=512, la BRAM de l'ECP5-45F ne peut plus rien cacher.
> Obligation de tout streamer depuis la SDRAM → les 14 tok/s s'effondrent.
> **C'est le mur.**

### `vocab` (512 → 32 000, ×62) — **frappe la SDRAM et la vitesse, peu la BRAM**

- **Table d'embedding** : `vocab × dim` nombres.
  stories260K = 32 Ko → tinyllamas = **16 Mo** (vit en SDRAM).
- **Tête de sortie** : 512 scores (8 morceaux) → 32 000 scores (62 morceaux)
  → **~8× plus de temps** pour choisir le mot suivant.
- **Argmax** : scanner 32 000 entrées au lieu de 512.

> **Verdict** : la BRAM s'en sort, mais la SDRAM grossit et la génération
> ralentit sur la fin de chaque token. Gérable mais lent.

### `T_MAX` (32 → 1024, ×32) — **frappe la SDRAM (cache KV) et la vitesse**

- **Cache KV** : `couches × T_MAX × largeur_KV`.
  stories260K ≈ 5 Ko → tinyllamas ≈ **800 Ko** (vit en SDRAM).
- **Attention** : 32 → 1024 comparaisons par token → **~32× plus de travail**
  quand le contexte se remplit.
- **Tables de shifts** : `couches × T_MAX` entrées → 32× plus grosses.

> **Verdict** : la BRAM tient (cache KV en SDRAM), mais la SDRAM grossit et
> **plus la conversation s'allonge, plus chaque token est lent**.

---

## 7. Récapitulatif

| Réglage | BRAM (puce) | SDRAM (carte) | Vitesse | Verdict sur l'ECP5-45F |
|---|---|---|---|---|
| `dim` ×8 | **explose** (×8 vecteurs, ×64 poids) | ×64 poids | ×8 plus lent | **bloquant** — puce trop petite |
| `vocab` ×62 | faible | table ×500 | ×8 sur la sortie | gérable mais lent |
| `T_MAX` ×32 | faible (KV en SDRAM) | cache KV ×150 | ×32 sur l'attention | gérable mais ralentit |

### L'effet combiné : le vrai problème

Tinyllamas est plus gros sur **les trois axes en même temps**. Ce n'est pas
×8 ou ×32 isolément, c'est la combinaison qui fait exploser les ressources :

```
                           stories260K        tinyllamas 15M
   poids du modèle         ~1 Mo              ~60 Mo      ◄── SDRAM
   table d'embedding       ~32 Ko             ~16 Mo      ◄── SDRAM
   cache KV                ~5 Ko              ~800 Ko     ◄── SDRAM
   BRAM nécessaire         ~90 blocs          >> 90 blocs ◄── NE TIENT PAS
```

```
   stories260K sur ECP5-45F :        tinyllamas sur ECP5-45F :

   ┌─────────────────┐               ┌─────────────────┐
   │     FPGA        │               │     FPGA        │
   │  ┌───────────┐  │               │  ┌───────────┐  │
   │  │ BRAM ~90% │  │               │  │ BRAM ???? │  │  ◄── déborde
   │  │  LUT ~61% │  │               │  │  LUT ???? │  │      (dim=512)
   │  └───────────┘  │               │  └───────────┘  │
   │  14 tok/s ✅    │               │  ? tok/s   ❓   │
   └────────┬────────┘               └────────┬────────┘
            │                                  │
            ▼                                  ▼
   ┌─────────────────┐               ┌─────────────────┐
   │     SDRAM       │               │     SDRAM       │
   │  ~1 Mo utilisé  │               │  ~77 Mo requis  │  ◄── peut déborder
   └─────────────────┘               └─────────────────┘
```

---

## 8. Ce qu'on a mesuré sur la puce (dim=128)

Le rework `dim`-paramétrable (jusqu'à 512) est en cours. Le premier palier,
**dim=128**, est **validé bit-exact sur silicium** (carte i9) : 17/17 tokens
identiques à l'oracle Python. La puce exécute bien un modèle deux fois plus
large que stories260K.

L'astuce qui rend dim=128 possible : **libérer la BRAM du cache FFN**. En mode
*stream SDRAM* (`NC=0`), on ne cache plus les poids FFN dans la puce — on les
relit depuis la SDRAM à chaque token (plus de cycles, zéro BRAM de cache). Les
90 blocs BRAM qu'occupait le cache FFN de stories260K retombent à 10.

Résultat mesuré (synthèse yosys+nextpnr + test silicium, NC=0 stream) :

| Ressource | stories260K (dim=64, cache FFN) | dim=128 (stream SDRAM) |
|---|---|---|
| BRAM (108 blocs) | 99 blocs (92 %) — **limitant** | 10 blocs (9 %) — libre |
| LUT (43 848) | 28 525 (65 %) | 38 067 (**86 %**) — **limitant** |
| DSP (72) | 18 (25 %) | 16 (22 %) |
| Fmax (compute) | 37,6 MHz | 33,8 MHz (cible 31,25 ✅) |
| SDRAM (8 Mo) | ~1 Mo | ~5,3 Mo |
| Vitesse (decode) | 14 tok/s (cache) | ~3-5 tok/s (stream) |

**Le mur a changé de nature** : à dim=64 c'est la **BRAM** qui limitait (90 %
pleine). À dim=128, la BRAM est libérée et c'est la **LUT** qui devient le mur
(86 % du datapath large). On a gagné ~90 blocs BRAM contre ~10 000 LUT.

```
   stories260K (dim=64) :         dim=128 (stream SDRAM) :

   ┌─────────────────┐            ┌─────────────────┐
   │     FPGA        │            │     FPGA        │
   │  ┌───────────┐  │            │  ┌───────────┐  │
   │  │ BRAM 92 % │  │            │  │ BRAM  9 % │  │  ◄── libérée
   │  │ LUT  65 % │  │            │  │ LUT  86 % │  │  ◄── nouveau mur
   │  └───────────┘  │            │  └───────────┘  │
   │  14 tok/s  ✅   │            │  ~4 tok/s ✅     │  ◄── bit-exact
   └────────┬────────┘            └────────┬────────┘
            │                              │
            ▼                              ▼
   ┌─────────────────┐            ┌─────────────────┐
   │     SDRAM       │            │     SDRAM       │
   │  ~1 Mo utilisé  │            │  ~5,3 Mo utilisé │
   └─────────────────┘            └─────────────────┘
```

À dim=256 (~13 Mo) puis dim=512 (~37 Mo), l'empreinte SDRAM déborde les 8 Mo de
la i9 — il faudra soit une carte avec plus de SDRAM, soit un cluster de FPGAs.
Le datapath LUT continuerait de grossir aussi. La suite du rework synthétise
ces paliers pour tracer la courbe de capacité exacte (voir §9).

> Note de debug : la validation dim=128 a révélé un **vrai bug RTL** (chargement
> du ROM RoPE : le fichier HS=8 était utilisé pour un config HS=16) que la
> simulation ne voyait pas — le ROM est « baked » dans le bitstream, identique
> en sim et sur silicium. Seul l'oracle Python, qui lit le bon fichier de
> référence, avait raison. Leçon : pour les bugs de données d'initialisation,
> l'oracle est l'arbitre, pas la cohérence sim/silicium.

---

## 9. Conclusion

- **`dim` est le mur**. À dim=512, la BRAM de l'ECP5-45F ne peut plus rien
  cacher — mais on sait désormais la **libérer** (stream SDRAM, NC=0). Le mur
  se déplace alors vers la **LUT** (86 % à dim=128), qui devient la vraie
  limite du single-FPGA.
- **dim=128 est validé** (bit-exact silicium, 86 % LUT, 9 % BRAM). dim=256/512
  restent à mesurer (courbe de capacité en cours) — l'empreinte SDRAM déborde
  la i9 dès dim=256, donc le cluster devient nécessaire vers dim=512.
- **`vocab` et `T_MAX`** sont surtout des questions de **SDRAM** et de
  **vitesse** : surmontables, mais la génération ralentit.
- Le rework **`head_size`** déjà livré n'élargit qu'**un seul tuyau** (celui de
  l'attention). Le rework **`dim`** (en cours) élargit le tuyau principal ;
  `vocab` et `T_MAX` restent à faire.

> L'étude de faisabilité est en cours : on mesure chaque palier (dim=128 fait,
> 256/512 à venir) pour tracer la courbe LUT/BRAM/Fmax/SDRAM et décider où le
> single-FPGA s'arrête et où le cluster commence. dim=128 tient ; dim=512
> demandera très probablement plusieurs puces.