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
#
# Requires oss-cad-suite on PATH (yosys, nextpnr-ecp5, ecppack, openFPGALoader).
# Package CABGA381 = BG381, speed grade 6 (LFE5U-45F-6BG381C).
# =============================================================================
param(
    [string]$Top   = "blink",
    [string]$Board = "colorlight-i9",
    [string]$Lpf   = "../board/$Top.lpf",
    [string]$Extra = "",
    [switch]$Flash
)
# Note: "Continue" (not "Stop") because nextpnr/yosys write progress to stderr,
# and PowerShell 5.1 wraps each stderr line in an ErrorRecord -- "Stop" would
# abort on the first Info line. Real failures are caught via $LASTEXITCODE below.
$ErrorActionPreference = "Continue"

$readCmd = "read_verilog ../board/$Top.v"
if ($Extra -ne "") {
    foreach ($f in ($Extra -split ",")) { $readCmd += "; read_verilog $f" }
}

Write-Host "==> yosys synth ($Top)" -ForegroundColor Cyan
yosys -l "$Top.ys.log" -p "$readCmd; synth_ecp5 -top $Top -json $Top.json"
if ($LASTEXITCODE -ne 0) { throw "yosys failed" }

Write-Host "==> nextpnr-ecp5 (P&R)" -ForegroundColor Cyan
nextpnr-ecp5 --package CABGA381 --speed 6 --json "$Top.json" `
             --lpf "$Lpf" --textcfg "$Top.config" --report "$Top.rpt"
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