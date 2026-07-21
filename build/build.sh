#!/usr/bin/env bash
# =============================================================================
# build.sh -- Git-Bash wrapper, same pipeline as build.ps1.
#   ./build.sh blink          # synth + P&R + pack
#   ./build.sh blink flash    # + flash via openFPGALoader
# Requires oss-cad-suite on PATH. Use forward slashes for any absolute paths.
# =============================================================================
set -euo pipefail
Top="${1:-blink}"
DoFlash="${2:-}"
Lpf="../board/${Top}.lpf"

echo "==> yosys synth ($Top)"
yosys -l "${Top}.ys.log" -p "read_verilog ../board/${Top}.v; synth_ecp5 -top ${Top} -json ${Top}.json"

echo "==> nextpnr-ecp5 (P&R)"
nextpnr-ecp5 --package CABGA381 --speed 6 --json "${Top}.json" \
             --lpf "$Lpf" --textcfg "${Top}.config" --report "${Top}.rpt"

echo "==> ecppack"
ecppack "${Top}.config" "${Top}.bit"
echo "Done: ${Top}.bit"

if [ "$DoFlash" = "flash" ]; then
    echo "==> openFPGALoader (flash via colorlight-i9)"
    openFPGALoader -b colorlight-i9 "${Top}.bit" || {
        echo "WARN: board id 'colorlight-i9' not found? Try: openFPGALoader --list-boards"
        echo "Fallback: openFPGALoader --device cmsis-dap --chip LFE5U-45F ${Top}.bit"
        exit 1
    }
    echo "Flashed. Gate 1 = LED D2 should blink ~1.5 Hz."
fi