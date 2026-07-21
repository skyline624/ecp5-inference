#!/usr/bin/env bash
# =============================================================================
# build.sh -- Git-Bash wrapper, same pipeline as build.ps1.
#   ./build.sh blink                              # synth + P&R + pack
#   ./build.sh blink flash                        # + flash via openFPGALoader
#   ./build.sh -e ../src/ecp5_ecp5pll.v blink flash        # extra Verilog files
#   ./build.sh -e ../src/ecp5_ecp5pll.v,../src/sdram.v sdram_test flash
# Env:  EXTRA=../src/x.v,../src/y.v  ./build.sh top   # alt way to pass extras
#       STRICT=1 ./build.sh top                     # omit --lpf-allow-unconstrained
# Requires oss-cad-suite on PATH. Use forward slashes for any absolute paths.
# =============================================================================
set -euo pipefail

Extra="${EXTRA:-}"
DoFlash=""
Top=""
while [ $# -gt 0 ]; do
    case "$1" in
        -e|--extra) Extra="$2"; shift 2;;
        -e=*) Extra="${1#-e=}"; shift;;
        flash) DoFlash="flash"; shift;;
        *) Top="$1"; shift;;
    esac
done
Top="${Top:-blink}"
Lpf="../board/${Top}.lpf"
Board="colorlight-i9"

# --- Copy LUT .hex into build/ (operators load them by cwd-relative $readmemh).
# Without this, rmsnorm/silu/softmax/attention silently output X.
if ls ../lut/*.hex >/dev/null 2>&1; then
    cp -f ../lut/*.hex .
    echo "==> copied LUT .hex -> build/"
else
    echo "WARN: ../lut/*.hex not found -- operator LUTs will be missing. Run scripts/regen_luts.py first." >&2
fi

# --- Build the read_verilog command with optional extra sources.
read_cmd="read_verilog ../board/${Top}.v"
if [ -n "$Extra" ]; then
    IFS=',' read -ra _extras <<< "$Extra"
    for f in "${_extras[@]}"; do read_cmd="${read_cmd}; read_verilog ${f}"; done
fi

echo "==> yosys synth (${Top})"
yosys -l "${Top}.ys.log" -p "${read_cmd}; synth_ecp5 -top ${Top} -json ${Top}.json"

echo "==> nextpnr-ecp5 (P&R)"
np_args=(--package CABGA381 --speed 6 --json "${Top}.json" \
         --lpf "$Lpf" --textcfg "${Top}.config" --report "${Top}.rpt")
if [ -z "${STRICT:-}" ]; then np_args+=(--lpf-allow-unconstrained); fi
nextpnr-ecp5 "${np_args[@]}"

echo "==> ecppack"
ecppack "${Top}.config" "${Top}.bit"
echo "Done: ${Top}.bit"

if [ "$DoFlash" = "flash" ]; then
    echo "==> openFPGALoader (flash via ${Board})"
    openFPGALoader -b "$Board" "${Top}.bit" || {
        echo "WARN: board id '${Board}' not found? Try: openFPGALoader --list-boards"
        echo "Fallback: openFPGALoader --device cmsis-dap --chip LFE5U-45F ${Top}.bit"
        exit 1
    }
    echo "Flashed."
fi