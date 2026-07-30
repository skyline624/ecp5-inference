# =============================================================================
# build.ps1 -- Build + (optional) flash for the Colorlight i9 v7.2 (ECP5-45).
# Pipeline: yosys (synth) -> nextpnr-ecp5 (P&R) -> ecppack (bitstream) -> openFPGALoader.
#
# Usage:
#   .\build.ps1 -Top blink                         # synth + P&R + pack only
#   .\build.ps1 -Top blink -Flash                  # + flash via CMSIS-DAP
#   .\build.ps1 -Top pll_test -Extra ../src/ecp5_ecp5pll.v -Flash
#   .\build.ps1 -Top sdram_test -Extra ../src/ecp5_ecp5pll.v,../src/sdram.v -Flash
#
# -Extra : comma-separated list of extra Verilog files (e.g. PLL/SDRAM wrappers
#          in ../src/) to read alongside the board top. Defaults to none.
# -NoAllowUnconstrained : do NOT pass --lpf-allow-unconstrained to nextpnr
#          (by default it is passed, harmless for fully-constrained LPFs but
#          required by top.lpf / gen_seq_silicon.lpf which leave led[4:0] unused).
#
# Requires oss-cad-suite on PATH (yosys, nextpnr-ecp5, ecppack, openFPGALoader).
# Package CABGA381 = BG381, speed grade 6 (LFE5U-45F-6BG381C).
# =============================================================================
param(
    [string]$Top   = "blink",
    [string]$Board = "colorlight-i9",
    [string]$Lpf   = "../board/$Top.lpf",
    [string]$Extra = "",
    [string]$Defines = "",   # CSV list of Verilog defines passed to each read_verilog (e.g. "NODE_ONLY")
    [switch]$Flash,
    [switch]$NoAllowUnconstrained
)
# Note: "Continue" (not "Stop") because nextpnr/yosys write progress to stderr,
# and PowerShell 5.1 wraps each stderr line in an ErrorRecord -- "Stop" would
# abort on the first Info line. Real failures are caught via $LASTEXITCODE below.
$ErrorActionPreference = "Continue"

# --- Copy LUT .hex into build/ (operators load them by cwd-relative $readmemh).
# Without this, rmsnorm/silu/softmax/attention silently output X. Regenerable
# via scripts/gen_*_lut.py if ../lut/*.hex are missing.
if (Test-Path "../lut/*.hex") {
    Copy-Item -Path "../lut/*.hex" -Destination "." -Force
    Write-Host "==> copied LUT .hex -> build/" -ForegroundColor DarkGray
} else {
    Write-Warning "../lut/*.hex not found -- operator LUTs will be missing. Run scripts/regen_luts.py first."
}

# Build optional -D<define> args for read_verilog (e.g. -DNODE_ONLY). Inoffensive on
# files that don't use the define. Needed for gen_seq_silicon (NODE_ONLY drops the
# redundant internal GG FSM of the `top` node, else synth blows to 139% LUT and
# nextpnr cannot route -- see memory note gen-silicon-nodeonly-fix).
$DefFlags = ""
if ($Defines -ne "") {
    $DefFlags = (($Defines -split ",") | ForEach-Object { "-D$_" }) -join " "
}

$readCmd = "read_verilog $DefFlags -I../src -I../board ../board/$Top.v"
if ($Extra -ne "") {
    foreach ($f in ($Extra -split ",")) { $readCmd += "; read_verilog $DefFlags -I../src -I../board $f" }
}

Write-Host "==> yosys synth ($Top)" -ForegroundColor Cyan
yosys -l "$Top.ys.log" -p "$readCmd; synth_ecp5 -top $Top -json $Top.json"
if ($LASTEXITCODE -ne 0) { throw "yosys failed" }

Write-Host "==> nextpnr-ecp5 (P&R)" -ForegroundColor Cyan
# --write : full place-and-route JSON (cell placement + routing) for nextpnr-viewer
#   (https://edacation.github.io/nextpnr-viewer/). Load $Top.pnr.json + $Top.rpt into
#   the viewer to render the design on the ECP5 die (slices/BRAM/DSP placement).
# --report : timing/utilization report JSON (also consumed by the viewer).
$npArgs = @("--package","CABGA381","--speed","6","--json","$Top.json",
            "--lpf","$Lpf","--textcfg","$Top.config",
            "--write","$Top.pnr.json","--report","$Top.rpt")
if (-not $NoAllowUnconstrained) { $npArgs += "--lpf-allow-unconstrained" }
& nextpnr-ecp5 @npArgs
if ($LASTEXITCODE -ne 0) { throw "nextpnr failed" }

Write-Host "==> ecppack" -ForegroundColor Cyan
ecppack "$Top.config" "$Top.bit"
if ($LASTEXITCODE -ne 0) { throw "ecppack failed" }

Write-Host "Done: $Top.bit" -ForegroundColor Green

if ($Flash) {
    Write-Host "==> openFPGALoader (flash via $Board)" -ForegroundColor Cyan
    openFPGALoader -b $Board "$Top.bit"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Board id '$Board' not found? Try: openFPGALoader --list-boards"
        Write-Warning "Fallback: openFPGALoader --device cmsis-dap --chip LFE5U-45F $Top.bit"
        throw "openFPGALoader failed"
    }
    Write-Host "Flashed. Gate 1 = LED D2 should blink ~1.5 Hz." -ForegroundColor Green
}