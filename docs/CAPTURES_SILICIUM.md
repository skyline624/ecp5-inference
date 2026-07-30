# Captures silicium — séquences de référence & signatures d'échec

> Synthèse des captures UART réalisées sur Colorlight i9 v7.2 (LFE5U-45F) lors des
> sessions de validation des 26–27/07/2026. Ce document remplace les `*.log` qui
> traînaient à la racine du dépôt (non versionnés car couverts par `.gitignore`,
> donc perdus au premier nettoyage) : seule l'information réutilisable est conservée.
>
> Procédure de flash et de capture : voir `docs/FLASH_BOOT.md` + la section
> « Profil requis » ci-dessous. Port UART sur la machine de dev : **COM9**
> (`$env:G5_PORT="COM9"`).

## 1. Séquence de référence — profil dim64 (`stories260K`, D=64, H=8/4, NL=5, vocab=512)

Capture nominale, mode 100 % autonome (`RR` + `GO` seuls, modèle et shifts
auto-chargés depuis la flash) via `python host\infer_g5autonomous.py` :

```
[403, 407, 261, 378, 432, 383, 286, 261, 376, 268, 414, 421, 303, 428, 415, 412, 264]
 Once  upon   a    time   ,   there  was   a   little  b     o     l    an     g     h     a    nd
```

- Oracle float de référence : `[1, 403, 407, 261, 378, 432, 383, 286, 261, 376, 298]`
- Alignement sur `oracle[1:]` (le BOS=1 n'est pas réémis après `RR`).
- **Match leading : 9/10** — divergence assumée au token 9 (`268` vs `298`), due au
  softmax simplifié. C'est le critère `PASS` du harness (`match >= 9`).
- Texte rendu : `Once upon a time, there was a little bolanghand`.
- Débit observé : ~0,06 s/token en lecture UART (le decode lui-même est à 14 tok/s).

Cette séquence est **stable et bit-exact identique** avant/après la migration des
tables non-linéaires en BRAM DP16KD (commit `fb8c451`) : les 17 tokens sont
inchangés, la migration n'ajoute que de la latence. Elle sert donc de témoin de
non-régression pour toute modification du datapath.

> Profil **dim128** : validé séparément **17/17 bit-exact** vs oracle
> (`[385,385,385,79,…,79,60]`) via `host\infer_silicon_dim128.py` (commit `e21d94b`).

## 2. Shifts de requantification (profil dim64)

Latchés par l'opcode `SH` en mode non-autonome ; en mode autonome ils sont lus
depuis le trailer flash (`@ base+0x61000`). 47 valeurs :

```
[-5,-5,-5,-5,-5,-6,-6,-6,-6,-6,-6,-6,-7,-6,-6,-8,-9,-8,-8,-8,-7,-8,-8,-8,-7,-7,
 -6,-6,-6,-6,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-4,-6]
```

## 3. Matrice des signatures d'échec (diagnostic rapide)

Toutes ces signatures ont été observées réellement sur silicium. Le token émis
constant est un discriminant fiable de la cause racine :

| Signature capture | Cause racine | Vérification / correctif |
|---|---|---|
| `[403, 407, 261, …]`, match 9/10 | **Nominal** (profil dim64) | — |
| **`214`** constant (parfois `213`) | Modèle absent de la SDRAM, **ou** mismatch dim bitstream ↔ modèle (ex. bitstream `-Dim128` avec un modèle D=64) | Rebuild avec le bon profil ; `diag_g5_readback_flash.py` |
| **`327`** constant (`'owowow…'`) | **Offsets SDRAM erronés dans l'image flash** — `gen_flash_model.py` écrivait des offsets paddés au lieu de cumulatifs (régression du rework dim `e21d94b`, corrigée le 27/07) | Régénérer `host/model_flash.bin` puis reflasher @ `0x200000` |
| **`0`** constant (`<unk>`) | SDRAM lue à zéro / argmax sans logits valides | `diag_g5_readback_flash.py` ; vérifier le boot flash |
| SDRAM = `0xFF` partout | Boot flash a lu une **zone flash vide** → mauvais `FLASH_BASE` ou modèle flashé à la mauvaise adresse | Modèle @ **`0x200000`** (pas `0x120000`, cf. `board/gen_seq_silicon.v` `.FLASH_BASE(24'h200000)`) |
| Tokens décalés / répétés (`"upon upon upon"`) | Octet parasite à l'ouverture du port série | Envoyer `RR` d'abord (déjà fait par `infer_g5autonomous.py`) |
| `--verify` échoue dès le 1ᵉʳ octet | Flash protégée en écriture (blocs `0x120000+` protégés d'usine) | `openFPGALoader -b colorlight-i9 --unprotect-flash` puis `--bulk-erase` |

## 3 bis. ⚠ Vérifier la flash par dump JTAG — le seul diagnostic neutre

**Leçon du 2026-07-30.** Après un reflash de bitstream, la capture sortait `0` constants.
`diag_g5_readback_flash.py` a répondu « AUCUNE REPONSE (CC readback casse) » — inutilisable,
car son opcode `CC` **passe par le RTL**, précisément ce qu'on cherchait à mettre hors de
cause. Le diagnostic décisif ne traverse aucune partie du design :

```powershell
openFPGALoader -b colorlight-i9 --dump-flash -o 0x200000 --file-size 512 --file-type bin dump.bin
# puis comparer les 32 premiers octets a host/model_flash.bin
```

Résultat ce jour-là : `ffffffff…` au lieu de `ed240b16f90f0102…` → **la zone modèle était vide**.
Le modèle avait disparu de la flash entre deux sessions (bulk-erase antérieur, ou effacement
par l'écriture du bitstream au-delà de sa propre taille). Conclusion opérationnelle :

- **Ne jamais supposer que le modèle est encore en flash** parce qu'il y a été écrit une fois.
  Le vérifier par dump avant d'interpréter une capture.
- Un `[FAIL]` de capture n'incrimine le RTL **qu'après** avoir prouvé que les poids sont là.
- L'ordre correct est : dump flash → si vide, reflasher le modèle `@0x200000` → recapturer.

## 4. Diagnostic du DMA flash → SDRAM

`python host\diag_g5_readback_flash.py` compare trois zones de la SDRAM à
`host/model_flash.bin`. Il lit la **SDRAM** (pas la flash), donc il valide le DMA
quelle que soit l'adresse flash source.

Capture nominale :
```
  tok_emb[0..31]    @ 0x000000 : 32/32 octets  [OK]
  wq layer0[0..31]  @ 0x010100 : 32/32 octets  [OK]
  rms_final[0..31]  @ 0x060000 : 32/32 octets  [OK]
[PASS] boot flash valide : SDRAM peuplee bit-exact depuis la flash.
```

Capture en échec (zone flash vide — modèle jamais écrit, ou écrit à la mauvaise
adresse) : les trois zones renvoient `ffff…ff` alors que `exp` attend les poids.

## 5. Trace de flash nominale (openFPGALoader)

Le DAPLink de la i9 est un CMSIS-DAP **v1** ; les deux premières lignes d'erreur
sont normales et non bloquantes :

```
cmsisDAP: failed to find compatible CMSIS-DAP v2 device
cmsisDAP v2: try USB bulk init but failed
Found 1 compatible device:
	0x0d28 0x0204 0x3 CMSIS-DAP-v1-MuseLab
USB bulk init successful
JEDEC ID: 0xef4017
Detected: Winbond W25Q64 128 sectors size: 64Mb
unprotect_flash: Done
```

## 6. Génération de l'image flash

`python host\gen_flash_model.py` → `host/model_flash.bin` (458 752 o = `0x70000`) :
mirror SDRAM `[0x0 … 0x60040)` + trailer des 47 shifts `@ 0x61000`. Le fichier est
absent du dépôt (`.gitignore`) — il faut le régénérer avant tout flash.

```
Chargement modele : host\models\stories260K.bin
  dim=64 hidden=172 layers=5 heads=8/4 vocab=512
Construction de l'image flash (mirror SDRAM + trailer 47 shifts)...
```
