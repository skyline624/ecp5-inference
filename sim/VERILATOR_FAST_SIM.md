# Simulation rapide Verilator (untimed) pour gen_seq_top

Sim **Verilator untimed** pour `gen_seq_top` (path FLASH_BOOT mock). **~35s pour 17
tokens, 17/17 bit-exact** — contre ~15-20 min en cocotb (**~25-30x plus rapide**).

Le TB iverilog pur (`tb_gen_seq_top_fb.v`, ~17 min) reste un flot de secours/référence.
Cocotb (`test_gen_F_fb.py`) reste pour le debug Python (`_peek_dbg`) mais est lent.

## Lancer

```powershell
# depuis sim/
.\run_verilator_fb.ps1              # HS=8 stories260K, 17 tokens (~35s)
.\run_verilator_fb.ps1 -Hs16         # HS=16 Strat A (HS=16 H=4 KH=2, ~31s)
.\run_verilator_fb.ps1 -Ntok 3       # 3 tokens (~6s, iteration rapide)
.\run_verilator_fb.ps1 -SkipBuild    # relancer l'exe existant (sans recompiler)
```

Le wrapper fait : `gen_tb_hex.py` + `gen_sdram_preload_hex.py` (depuis le Dump) ->
`verilator --cc` (untimed) -> `build_verilator_fb.bat` (MSVC cl.exe) -> run + chrono.

## Pourquoi untimed et pas `--timing`

`--timing` (coroutines C++20) patine sur ce gros design (process brûlait du CPU, 0
sortie après 50s) — overhead scheduler/coroutine par always-block. Le mode rapide =
**untimed** : le C++ drive clk (`top->clk=0;eval;clk=1;eval`), pas de scheduler Verilator.

## Setup non-trivial (4 pièges)

### 1. Perl Pod::Usage manquant
Le wrapper `verilator` est un script Perl qui `use Pod::Usage;` (ligne 17,
inconditionnel). Le perl MSYS 5.38 (`/usr/bin/perl`) est ecorché (pas de module Pod).
- Fix : stub `sim/vlib/Pod/Usage.pm` (pod2usage = print + exit) + `$env:PERL5LIB = "$PSScriptRoot\vlib"`.
- Le binaire natif `verilator_bin.exe` marche seul mais ne fait que générer le C++ (pas le link) — le wrapper Perl orchestre le build, donc on a besoin du stub.
- (`verilator --version` sans le stub echoue avec "Can't locate Pod/Usage.pm".)
- **Invocation** : le wrapper `verilator` n'a **pas d'extension** → `cmd /c "verilator …"`
  échoue ("n'est pas reconnu") car cmd ne cherche que `.com/.exe/.bat/.cmd`. On invoque
  donc **perl explicitement** sur le wrapper :
  `& "C:\Program Files\Git\usr\bin\perl.exe" "<verilator>" --cc …` (Git perl = le
  `/usr/bin/perl` MSYS, le seul disponible ; oss-cad-suite ne bundle pas de perl).

### 1bis. VERILATOR_ROOT : passer $0 en path MSYS (/c/…), pas Windows (C:\…)
Le wrapper calcule `VERILATOR_ROOT = realpath("$RealBin/../share/verilator")` via
FindBin/Cwd (ligne 92). **Si `$0` est un path Windows** (`C:\oss-cad-suite\bin\verilator`),
`realpath()` échoue (mélange backslash/forward-slash) → `$verilator_root` undef →
fallback C++ sur le prefix **hardcodé au build** `/yosyshq/share/verilator` (inexistant
sous Git perl) → `%Error: Cannot find verilated_std.sv`.
- Fix : passer `$0` en **path MSYS** (`/c/oss-cad-suite/bin/verilator`) → `realpath()` OK
  → `VERILATOR_ROOT=/c/oss-cad-suite/share/verilator` (correct).
- `run_verilator_fb.ps1` dérive ce path MSYS depuis `$env:YOSYSHQ_ROOT` (positionné par
  `environment.ps1`) : `TrimEnd('\')` → replace `\` par `/` → préfixe le drive `C:` →
  `/c/oss-cad-suite`. Ne pas hardcoder le drive.
- Diagnostique : `perl <verilator> --debug --getenv VERILATOR_ROOT` doit afficher
  `/c/oss-cad-suite/share/verilator` (et non `''` + warning `uninitialized $verilator_root`).

### 2. MSVC cl.exe (pas de g++/gcc/clang sur cette machine)
VS 2022 BuildTools installe : `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`.
- `build_verilator_fb.bat` source `vcvars64.bat` puis `cl /std:c++20 /permissive- /O2 /EHsc`.
  - C++20 requis : `verilated_timing.h` utilise `std::coroutine_handle`.
  - Include paths : `-I vobj -I <verilator>/include -I <verilator>/include/vltstd`
    (vltstd = `svdpi.h`, inclus par le pch genere).
- Runtimes a linker : `verilated.cpp + verilated_timing.cpp + verilated_threads.cpp`.
- `sc_time_stamp()` doit etre defini dans `sim_main.cpp` (requis par le runtime).

### 3. TB en C++ (pas Verilog) pour untimed
Untimed Verilator ne supporte pas `#delay` / `@(posedge)` / `wait` dans un TB Verilog.
- Top = `gen_seq_top` (le DUT direct, pas de wrapper TB Verilog).
- TB = `sim/sim_main.cpp` : drive clk, drive shifts, poll `sim_fill_done`, pulse start,
  capture fronts `token_valid`, lit expected/shifts en C++ depuis les hex.
- **Sonde `sim_fill_done`** ajoutee a `src/gen_seq_top.v` (port output sim-only) =
  `wcache_fill_done_w` sous FLASH_BOOT (else 1). Permet d'attendre le remplissage cache
  SANS ref hierarchique `dut.u_n.wcache_fill_done` (qui exigerait `--public-flat-rw`).
  Le TB iverilog `tb_gen_seq_top_fb.v` l'utilise aussi.

## Fichiers

| Fichier | Role |
|---|---|
| `sim_main.cpp` | TB C++ Verilator untimed (le rapide) |
| `tb_gen_seq_top_fb.v` | TB Verilog pur pour iverilog (flot secours, ~26k cyc/s) |
| `gen_tb_hex.py` | dump JSON -> `tb_meta.hex` (NTOK) + `tb_shifts.hex` (11 ports sw_*) + `tb_expected.hex` (node_tokens) |
| `build_verilator_fb.bat` | cl.exe compile `vobj/*.cpp + sim_main.cpp + runtimes` -> `vobj/tb_verilator.exe` |
| `run_verilator_fb.ps1` | wrapper complet (regen hex+preload -> verilator --cc -> build MSVC -> run+chrono) |
| `run_tb_fb.ps1` | wrapper iverilog TB pur (flot secours) |
| `vlib/Pod/Usage.pm` | stub Perl pour le wrapper verilator |

## Piege : swap du SDRAM preload

`sdram_chip.v` lit `"sdram_preload.hex"` (basename hardcodé, ligne 78). HS=8 et HS=16
partagent ce nom. `run_verilator_fb.ps1` et `run_tb_fb.ps1` **régénèrent
`sdram_preload.hex` depuis le Dump choisi à chaque run** (sinon un run HS=16 laisse le
preload HS=16 et casse le run HS=8 — bug latent qui a fait echouer la 1re tentative).

## Resultats valides (2026-07-23)

- HS=8 stories260K NTOK=17 : 35.6s, 17/17 bit-exact = `[403,407,261,378,432,383,286,261,376,268,414,421,303,428,415,412,264]`
- HS=16 Strat A NTOK=17 : 30.8s, 17/17 bit-exact = `[164x5, 173x12]`
- Cadence ~730k cyc/s (cocotb ~6k, iverilog TB ~26k).

## Recompile HS=8 <-> HS=16

HS=16 change le RTL (`-DHS16_PROFILE` -> CFG_HS=16/H=4/KH=2/SCORE_SH=1), donc Verilator
+ MSVC recompilent (~2-3 min) a chaque switch. Le run lui-meme est ~30s. Pour debugger
rapidement sans recompiler entre les deux, faire tous les tests d'un profile avant de
switcher.