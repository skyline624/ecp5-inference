# Boot flash autonome (W25Q64 → SDRAM)

Objectif : faire du Colorlight i9 (ECP5-45) une « boîte noire » qui, au power-on,
charge elle-même le modèle int8 depuis la flash SPI vers la SDRAM, puis génère
les tokens sans PC. La brique « prompt entrant » (prefill) est reportée à une
phase ultérieure ; ce document couvre le **boot flash** et la **génération
autonome**.

## État

| Étape | Statut | Description |
|-------|--------|-------------|
| 1 | **Validée silicium (2026-07-21)** | Modèle auto-chargé depuis flash ; `SH` (47 shifts) + `GO` → 9 tokens bit-exact |
| 2 | **Validée silicium (2026-07-21)** | Shifts aussi en flash trailer ; `RR` + `GO` seul (100 % autonome) → 9 tokens bit-exact |

## Layout flash (W25Q64, 8 Mo, pins R2/V2/W2 + pad MCLK U3)

```
0x000000   bitstream gen_seq_silicon.bit (~1.02 Mo, écrit par openFPGALoader -f)
0x120000   model_flash.bin : mirror SDRAM [0x000000..0x060040) + trailer @ 0x061000
```

`model_flash.bin` (448 KiB = 0x070000) est produit par `host/gen_flash_model.py` :
mirror byte-for-byte du layout SDRAM que `host/infer_fpga.py` chargerait par `LL`,
plus un trailer de 47 octets à l'offset image 0x061000 (= flash 0x181000) contenant
les shifts dans l'ordre `sh_packet` (étape 2).

Layout SDRAM mirroré (`src/gen_seq.v` OFF_*, `host/infer_fpga.py`) :
- `0x000000` tok_emb [512,64] chunked [64,64]
- `0x010000 + l*0x10000` couche l : rms_att+0, wq+0x100, wk+0x1100, wv+0x1900,
  wo+0x2100, rms_ffn+0x3100, w1+0x3200, w3+0x6200, w2+0x9200
- `0x060000` rms_final [64]
- Dernier octet utile `0x060040` (~384 KiB)

## Architecture RTL

- **`src/flash_boot.v`** — reader SPI flash (commande 0x03, 4-fils, mode 0) + DMA
  octet-par-octet vers `sdram.v`. FSM `F_IDLE → F_CMD → F_CMD_TX → F_DATA → F_WR →
  F_DONE` (+ `F_RECS`/`F_SHIFT` pour l'étape 2). `div_cnt` gelé pendant `F_WR`
  (SCK stable, la W25Q64 tient le bit). `drained` laisse `bref_cnt` glisser un
  refresh SDRAM pendant les stalls. Validé en sim G1 (`build/tb_flash_boot.v` +
  mocks `w25q64_model.v`/`sdram_sink.v`) : 393 280 octets bit-exact, boot ~519 ms
  à `SPI_HALF=1` (SCK ~6.25 MHz).
- **`src/node_top.v`** — ports `flash_*` + instance `u_boot` + mux `eff_*` (route
  le DMA vers la SDRAM pendant `booting`) + refresh cohabitation `bref_cnt`.
  `boot_start` pulsé après l'init SDRAM (`!sd_busy`) ; `booting = ~boot_done` ;
  UART nœud gaté par `rx_valid && !booting`. Paramètres : `FLASH_BASE=0x120000`,
  `FLASH_LEN=0x060040`, `SHIFT_BASE=0x181000`, `SHIFT_LOAD_EN=0` (étape 1),
  `SPI_HALF_DIV=1`.
- **`board/gen_seq_silicon.v`** — ports flash GPIO (CS/MOSI/MISO) + instance
  `USRMCLK` pour SCK (voir ci-dessous) + déclaration `shifts_from_flash` (étape 2).
- **`board/gen_seq_silicon.lpf`** — CS=R2, MISO=V2, MOSI=W2 (GPIO). **Pas de
  contrainte pour SCK** (pad dédié via USRMCLK).
- **Build** : `build/build_silicon.ps1`/.sh passent `-D NODE_ONLY,FLASH_BOOT` et
  incluent `src/flash_boot.v`. Build étape 1 : **48 % LUT** (21356/43848),
  DFF 20 %, Fmax 36 MHz, 0 erreur.

## La pin SCK = pad MCLK dédié (USRMCLK)

Sur CABGA381, la pin `U3` (flash SCK) **n'est pas une IO normale** : c'est le pad
MCLK/CCLK dédié (tile `SPICB0`) utilisé par le hard SPI config controller au
power-on. nextpnr la rejette :
```
ERROR: IO pin 'flash_sck$tr_io' constrained to pin 'U3', which does not exist for package 'CABGA381'.
```
La seule façon de piloter SCK depuis le fabric est la primitive **`USRMCLK`**
(utilisée par litex/neorv32/orangecrab). Le hard controller relâche le pad après
configuration → pas de conflit temporel avec USRMCLK.

**`USRMCLKTS` doit être un tri-state dynamique, pas une constante** :
- `0` = drive SCK (uniquement pendant que `flash_boot` lit),
- `1` = HiZ (sinon) pour laisser le hard config controller / JTAG `spiOverJag`
  reprogrammer la flash sans contention.

Dans `node_top.v` : `assign flash_sck_tsen = ~(boot_started & ~boot_done);`
(HiZ au reset → protège la config FPGA ; drive pendant le boot ; HiZ après).
Le wrapper instancie `USRMCLK` avec `.USRMCLKI(flash_sck_int), .USRMCLKTS(sck_tsen)`.

## Procédure silicium (étape 1)

Depuis `C:\Users\User\Downloads\ecp5-inference`, carte sur COM3,
Python = `C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe` :

```powershell
. C:\oss-cad-suite\environment.ps1

# Flasher (une seule fois, ou après changement bitstream/modèle)
openFPGALoader -b colorlight-i9 --unprotect-flash            # 1. déprotéger la flash
openFPGALoader -b colorlight-i9 --bulk-erase                 # 2. effacer (avant 1er flash modèle)
openFPGALoader -b colorlight-i9 -f build\gen_seq_silicon.bit # 3. bitstream @ 0
Start-Sleep 3                                                # attendre fin boot -> USRMCLK HiZ
openFPGALoader -b colorlight-i9 -f -o 0x120000 --file-type bin --verify host\model_flash.bin

# Rejouer une inférence (le wrapper reste en RUN après GO ; reconfigure pour repartir) :
openFPGALoader -b colorlight-i9 -m build\gen_seq_silicon.bit  # reconfigure -> boot flash redémarre
$env:G5_PORT="COM3"
& "C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe" host\diag_g5_readback_flash.py
& "C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe" host\infer_g5silicon_flash.py
```

- `host/diag_g5_readback_flash.py` — readback SDRAM sans `LL`, compare 3 zones
  (tok_emb@0, wq@0x010100, rms_final@0x060000) à `model_flash.bin`. Valide le DMA.
- `host/infer_g5silicon_flash.py` — `RR` (reset gen_seq) + `SH` + `GO` + lit 17
  tokens. Critère PASS : 9 tokens leading bit-exact (alignement `oracle[1:]` car
  le BOS `1` n'est pas émis après `RR`).

## Pièges silicium résolus (étape 1)

1. **Pin SCK U3 = pad MCLK dédié** → `USRMCLK`, pas de contrainte LPF.
2. **`USRMCLKTS` tri-state dynamique** — une constante 0 fait contentionner
   `spiOverJag` → `--verify` échoue au 1er octet du modèle.
3. **Block protection flash** — la W25Q64 a des blocs protégés d'usine. Offset 0
   (bitstream) non protégé, mais 0x120000 protégé → écriture silencieusement
   ignorée. `--unprotect-flash` + `--bulk-erase` obligatoires. Les écritures
   répétées sans effacement corrompent aussi (flash SPI = 1→0 seulement).
4. **gen_seq parasite pendant le boot** — le wrapper host FSM n'est PAS gelé par
   `booting` (seul le nœud l'est via `rx_valid && !booting`). Un byte parasite à
   l'ouverture du port peut ressembler à GO → gen_seq s'avance → tokens
   décalés/répétés (« upon upon upon… »). Fix host : envoyer `RR` avant `SH`.

## Résultat étape 1

```
Texte FPGA : 'Once upon a time, there was a little bolanghand'
Tokens     : [403, 407, 261, 378, 432, 383, 286, 261, 376, ...]
Oracle     : [1, 403, 407, 261, 378, 432, 383, 286, 261, 376, 298]
Alignement oracle[1:] ; match leading : 9/10   [PASS]
```
« Once upon a time, there was a little » (9 tokens bit-exact), divergence @9
acceptée (identique au run silicium avec chargement `LL`). Reproductible.

## Étape 2 — shifts en flash trailer (100 % autonome) — VALIDÉE

Implémentée et validée silicium (2026-07-21). Le FPGA charge maintenant le modèle
ET les 47 shifts depuis la flash au power-on ; le PC n'envoie que `RR` + `GO`
(4 octets) et lit les tokens. Résultat identique à l'étape 1 : « Once upon a
time, there was a little » (9 tokens bit-exact).

- `SHIFT_LOAD_EN=1` dans l'instance `u_n` du wrapper et défaut `node_top.v`.
- `flash_boot.v` lit le trailer via `F_RECS → F_SHIFT → shifts_out` +
  `shifts_valid` sticky. Sim G1 (`build/tb_flash_boot.v` DUT_SHIFT) :
  `0/47 shift mismatch, shifts_valid=1`.
- Latch `sw_*` dans `board/gen_seq_silicon.v` depuis `shifts_from_flash`, **dans
  le même `always` que `H_SH`** (un `reg` = un seul `always`), via un flag
  `shifts_latched` (latch une fois quand `shifts_valid_from_flash`, reset par
  `rst_n`). Ordre `sh_packet` :
  `sw_rms[5],swq[5],swk[5],swv[5],swo[5],sw_rmsf[5],sw1[5],sw3[5],sw2[5],
  sw_rmsfinal,sw_emb`.
- Le décodage `H_SH` est conservé : un `SH` UART écrase les valeurs flash pour
  debug (itérer sans reflasher).
- `host/infer_g5autonomous.py` — `RR` + `GO` seul, lit 17 tokens, compare à
  `oracle[1:]` (BOS non émis après RR).

Procédure étape 2 (modèle + trailer déjà flashés à 0x120000 depuis l'étape 1) :
```powershell
. C:\oss-cad-suite\environment.ps1
openFPGALoader -b colorlight-i9 -f build\gen_seq_silicon.bit   # bitstream étape 2
$env:G5_PORT="COM3"
& "C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe" host\infer_g5autonomous.py
```

## Suite — prompt entrant (prefill)

La génération est désormais 100 % autonome (power-on → modèle+shifts auto-chargés
→ `GO` → 17 tokens). La brique restante pour un accélérateur « boîte noire »
complet est le **prefill** : recevoir un prompt en entrée et générer la réponse,
au lieu du prompt actuel codé en dur (BOS). Hors périmètre de ce document.