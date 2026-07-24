# =============================================================================
# run_verilator_fb.ps1 -- Sim RAPIDE Verilator untimed pour gen_seq_top (FLASH_BOOT).
# ~25-30x plus rapide que cocotb (35s vs ~15-20min pour 17 tokens), bit-exact.
#
# Flow : gen_tb_hex.py + gen_sdram_preload_hex.py -> verilator --cc (untimed,
# top=gen_seq_top) -> build_verilator_fb.bat (MSVC cl.exe) -> tb_verilator.exe.
# Le TB est en C++ (sim_main.cpp) : drive clk, poll sim_fill_done, capture token_valid.
#
# Usage (depuis sim/) :
#   .\run_verilator_fb.ps1                  # HS=8 stories260K, 17 tokens
#   .\run_verilator_fb.ps1 -Hs16            # HS=16 Strat A (HS=16 H=4 KH=2)
#   .\run_verilator_fb.ps1 -StreamFFN       # NC=0 stream SDRAM (Phase 5 : path FFN stream bit-exact)
#   .\run_verilator_fb.ps1 -Ntok 3          # 3 tokens (iteration rapide ~6s)
#   .\run_verilator_fb.ps1 -SkipBuild       # juste relancer l'exe existant
#
# Prerequis : oss-cad-suite (verilator) + VS 2022 BuildTools (cl.exe) + Python 3.12.
# Verilator (script Perl) a besoin du stub Pod::Usage (vlib/Pod/Usage.pm) car le
# perl MSYS 5.38 n'embarque pas le module Pod. MSVC cl.exe active via vcvars64.bat.
# =============================================================================
param(
    [switch]$Hs16,
    [switch]$StreamFFN,
    [switch]$Dim128,
    [int]$Ntok = 17,
    [string]$Dump = "",
    [switch]$SkipBuild
)

. C:\oss-cad-suite\environment.ps1
$py = "C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe"
# stub Perl Pod::Usage pour le wrapper verilator (perl MSYS ecorche)
$env:PERL5LIB = "$PSScriptRoot\vlib"
Set-Location $PSScriptRoot

if ($Dump -eq "") {
    if ($Dim128)  { $Dump = "..\host\gen_model_dump_dim128.json" }
    elseif ($Hs16) { $Dump = "..\host\gen_model_dump_hs16_rand.json" }
    else           { $Dump = "..\host\gen_model_dump.json" }
}
$env:DUMP = $Dump

# ---- hex (shifts/expected) + preload SDRAM depuis le Dump ----
Write-Host "==> gen_tb_hex.py + gen_sdram_preload_hex.py (Dump=$Dump, NTOK=$Ntok)" -ForegroundColor Cyan
& $py gen_tb_hex.py --dump $Dump --ntok $Ntok | Select-Object -Last 1
& $py ..\host\gen_sdram_preload_hex.py | Select-Object -Last 1

if (-not $SkipBuild) {
    # ---- Verilator --cc (untimed, top=gen_seq_top) ----
    $srcList = "models/ECP5_PLL_sim.v models/MULTALU18X18.v models/sdram_chip.v " +
               "models/flash_boot_sim.v " +
               "../src/node_top.v ../src/wcache_ffn.v ../src/sdram.v ../src/mac18.v " +
               "../src/rmsnorm_op.v ../src/silu_op.v ../src/rope_op.v ../src/softmax_op.v " +
               "../src/attention_head_op.v ../src/vec_alu2.v ../src/async_fifo.v " +
               "../src/uart_bridge.v ../src/gen_seq.v ../src/gen_seq_top.v"
    $defs = "-DLINK_SS -DSDRAM_PRELOAD -DSIM -DFLASH_BOOT"
    if ($Hs16) { $defs += " -DHS16_PROFILE" }
    if ($StreamFFN) { $defs += " -DNC0_PROFILE" }
    if ($Dim128) { $defs += " -DDIM128" }
    # C++ TB : nb de mots uint32 du port result (W*D/32). 16 dim=64, 32 dim=128.
    # Positionné AVANT le build pour que build_verilator_fb.bat (%CXXDEFS%) le voie.
    if ($Dim128) { $env:CXXDEFS = "/DXB_WORDS=32" } else { $env:CXXDEFS = "" }
    $wflags = "-Wno-fatal -Wno-UNOPTFLAT -Wno-UNUSED -Wno-WIDTH -Wno-UNDRIVEN " +
              "-Wno-CASEINCOMPLETE -Wno-INITIALDLY -Wno-PINMISSING -Wno-IMPLICIT " +
              "-Wno-LITENDIAN -Wno-BLKANDNBLK -Wno-CASEX -Wno-UNOPT"
    Write-Host "==> verilator --cc (untimed, top=gen_seq_top, HS16=$Hs16)" -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    # Verilator = wrapper Perl SANS extension ; cmd /c "verilator ..." échoue ("n'est
    # pas reconnu") car cmd ne cherche que .com/.exe/.bat/.cmd. On invoque donc perl
    # (Git MSYS, C:\Program Files\Git\usr\bin\perl.exe) explicitement sur le wrapper
    # (PERL5LIB=stub Pod::Usage, positionné plus haut). Le wrapper exec verilator_bin.exe
    # (trouvé via PATH oss-cad-suite hérité du processus PowerShell).
    $perlExe    = "C:\Program Files\Git\usr\bin\perl.exe"
    # Le wrapper verilator calcule VERILATOR_ROOT = realpath("$RealBin/../share/verilator")
    # via FindBin/Cwd. Si $0 est un path Windows (C:\...\verilator), realpath() ÉCHOUE
    # (backslashes) -> $verilator_root undef -> fallback C++ sur le prefix HARDCODED au
    # build "/yosyshq/share/verilator" (inexistant sous Git perl) -> "Cannot find
    # verilated_std.sv". FIX : passer $0 en path MSYS (/c/...) -> realpath OK ->
    # VERILATOR_ROOT=/c/oss-cad-suite/share/verilator (correct). Dérivé de YOSYSHQ_ROOT
    # (positionné par environment.ps1) pour ne pas hardcoder le drive.
    $yq = $env:YOSYSHQ_ROOT.TrimEnd('\') -replace '\\','/'          # C:/oss-cad-suite
    $drv = ($yq -replace ':.*','').ToLower()                       # c
    $yqMsys = '/' + $drv + ($yq -replace '^.:','')                 # /c/oss-cad-suite
    $verilator  = "$yqMsys/bin/verilator"
    $vargs = @('--cc') +
             ($wflags -split ' ' | Where-Object { $_ }) +
             ($defs    -split ' ' | Where-Object { $_ }) +
             @('-I../src','--top-module','gen_seq_top','-Mdir','vobj') +
             ($srcList -split ' ' | Where-Object { $_ })
    & $perlExe $verilator @vargs *> vobj_vlog.txt
    $vrc = $LASTEXITCODE
    if ($vrc -ne 0) { Get-Content vobj_vlog.txt | Select-Object -Last 20; throw "verilator --cc failed (exit $vrc)" }
    Get-Content vobj_vlog.txt | Select-Object -Last 2
    $sw.Stop(); Write-Host ("  verilator: {0:N1}s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor DarkGray

    # ---- build MSVC (cl.exe via vcvars64) ----
    Write-Host "==> build MSVC (cl.exe /O2)" -ForegroundColor Cyan
    $sw2 = [System.Diagnostics.Stopwatch]::StartNew()
    & ".\build_verilator_fb.bat" | Select-Object -Last 3
    if (-not (Test-Path vobj\tb_verilator.exe)) { throw "build failed (pas d'exe)" }
    $sw2.Stop(); Write-Host ("  build MSVC: {0:N1}s" -f $sw2.Elapsed.TotalSeconds) -ForegroundColor DarkGray
}

# ---- run + chrono ----
# NC=0 stream SDRAM est ~3x plus lent (FFN streamé a chaque token) -> le cap cycles
# anti-hang du TB (40M, calibré NC=5) déborde avant 17 tokens. On lève le cap via env.
# Dim128 = profile DIM128 -> CFG_NC=0 automatique (stream) -> cap élevé aussi.
if ($StreamFFN -or $Dim128) { $env:SIM_CYCLE_CAP = "260000000" }  # dim128 NC=0 stream ~10M cyc/tok -> 17 tokens ~170M+margin
else            { $env:SIM_CYCLE_CAP = "" }   # défaut TB 40M
Write-Host "==> run tb_verilator.exe (NTOK=$Ntok, HS16=$Hs16, StreamFFN=$StreamFFN, Dim128=$Dim128)" -ForegroundColor Cyan
$sw3 = [System.Diagnostics.Stopwatch]::StartNew()
$out = & ".\vobj\tb_verilator.exe" 2>&1
$rc = $LASTEXITCODE
$sw3.Stop()
$out | Select-Object -Last 6
Write-Host ("==> wall-clock sim: {0:N2}s (exit=$rc)" -f $sw3.Elapsed.TotalSeconds) -ForegroundColor $(if ($rc -eq 0) {"Green"} else {"Yellow"})