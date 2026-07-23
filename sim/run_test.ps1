# =============================================================================
# run_test.ps1 -- Lance un test cocotb/icarus SANS make (make non installé ici).
# Encapsule la procédure découverte pour valider la sim gen_seq_top après la
# réparation top.v -> node_top.v (mémoire sim-top-uart-direct-stale).
#
# Usage (depuis sim/) :
#   .\run_test.ps1                       # test_gen_F (17 tokens, ~2.4h icarus)
#   .\run_test.ps1 -Test test_gen_F2     # un autre module de test
#   .\run_test.ps1 -LogFile $true        # redirige vvp vers <Test>_run.log (temps réel)
#
# Prérequis : C:\oss-cad-suite (iverilog/vvp) + Python 3.12 (cocotb 2.0) +
#   ../lut/*.hex + sdram_preload.hex dans sim/ (voir Makefile cible `hex` /
#   host/gen_sdram_preload_hex.py). Les .hex sont copiés par le Makefile ; sans
#   make, copier ../lut/*.hex manuellement une fois (cp ../lut/*.hex .).
# =============================================================================
param(
    [string]$Test   = "test_gen_F",
    [string]$Top    = "gen_seq_top",
    [switch]$LogFile,
    [switch]$FlashBoot,
    [switch]$SimDebug,
    [switch]$Hs16,
    [string]$DumpFile = ""        # override du dump (ex: Strat A rand) ; sinon default -Hs16
)

$ErrorActionPreference = "Continue"

# ---- 1. env : oss-cad-suite (iverilog/vvp + DLLs) + Python 3.12 (cocotb) ----
. C:\oss-cad-suite\environment.ps1
$pyDir = "C:\Users\User\AppData\Local\Programs\Python\Python312"
$env:PATH = "$pyDir;$pyDir\Scripts;" + $env:PATH

# ---- 2. cocotb 2.0 vars (vvp lance cocotb via le module VPI) ----
$env:PYGPI_PYTHON_BIN    = "$pyDir\python.exe"
$env:COCOTB_TEST_MODULES  = $Test
$env:TOPLEVEL             = $Top
$env:TOPLEVEL_LANG        = "verilog"
$env:PYTHONPATH           = $PWD.Path
$env:COCOTB_LOG_LEVEL     = "INFO"

$libdir = "$pyDir\Lib\site-packages\cocotb\libs"

# ---- 3. compile (cmd /c pour passer -o correctement ; PowerShell 5.1 wrap mal -o) ----
# -s $Top force le top-level (sinon iverilog choisit Gowin_rPLL, non instancié,
# comme racine -> cocotb dut=Gowin_rPLL, AttributeError dut.clk). Voir mémoire
# sim-top-uart-direct-stale.
$srcList = "models/ECP5_PLL_sim.v models/MULTALU18X18.v models/sdram_chip.v "
if ($FlashBoot) {
    $srcList += "models/flash_boot_sim.v "              # mock drop-in de flash_boot
    Write-Host "==> FLASH_BOOT ON (-DFLASH_BOOT + flash_boot_sim.v mock)" -ForegroundColor Magenta
}
$srcList += "../src/node_top.v ../src/wcache_ffn.v ../src/sdram.v ../src/mac18.v " +
            "../src/rmsnorm_op.v ../src/silu_op.v ../src/rope_op.v " +
            "../src/softmax_op.v ../src/attention_head_op.v " +
            "../src/vec_alu2.v ../src/async_fifo.v ../src/uart_bridge.v " +
            "../src/gen_seq.v ../src/gen_seq_top.v"
# -DSIM : active en sim seulement (jamais en synthèse build_silicon). Permet à
# node_top.v d'initialiser les BSRAM xbuf/wbuf/obuf et les FF x_raddr_fm/w_raddr_fm
# à 0 pour matcher l'init matérielle ECP5 (BRAM/FF default 0) ; sans ça icarus les
# laisse à X -> stale read X dans le dot FQ -> acc=X -> token=0 (cf flash-boot-mock-sim).
$defs = "-DLINK_SS -DSDRAM_PRELOAD -DSIM"
if ($FlashBoot) { $defs += " -DFLASH_BOOT" }
if ($SimDebug) { $defs += " -DSIM_DEBUG" }
if ($Hs16) {
    $defs += " -DHS16_PROFILE"
    # DUMP : -DumpFile override (ex Strat A rand) sinon default Strat B (stories reshape).
    if ($DumpFile -ne "") {
        $env:DUMP = (Resolve-Path $DumpFile -ErrorAction SilentlyContinue).Path
        if (-not $env:DUMP) { $env:DUMP = $DumpFile }
    } else {
        $env:DUMP = Join-Path $PWD.Path "..\host\gen_model_dump_hs16.json"
    }
    Write-Host "==> HS16_PROFILE ON (DUMP=$env:DUMP, HS=16 H=4 KH=2 score_sh=1)" -ForegroundColor Magenta
}
Write-Host "==> iverilog compile ($Top -> $Top.vvp)" -ForegroundColor Cyan
cmd /c "iverilog -o $Top.vvp -s $Top -g2012 $defs -I../src $srcList 2>&1"
if ($LASTEXITCODE -ne 0) { throw "iverilog compile failed (exit $LASTEXITCODE)" }

# ---- 4. run vvp + cocotb VPI ----
Write-Host "==> vvp + cocotb ($Test)" -ForegroundColor Cyan
if ($LogFile) {
    cmd /c "vvp -M $libdir -m cocotbvpi_icarus $Top.vvp > $Test`_run.log 2>&1"
    Write-Host "==> output dans $Test`_run.log (tail -f pour progression)" -ForegroundColor DarkGray
} else {
    vvp -M $libdir -m cocotbvpi_icarus $Top.vvp
}
Write-Host "==> vvp exit=$LASTEXITCODE" -ForegroundColor $(if ($LASTEXITCODE -eq 0) {"Green"} else {"Yellow"})