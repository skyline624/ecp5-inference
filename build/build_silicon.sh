#!/usr/bin/env bash
# =============================================================================
# build_silicon.sh -- One-command build of the gen_seq_silicon (Gate 6') bitstream.
# Wraps build.sh with the right source list + -D NODE_ONLY.
#
#   ./build_silicon.sh           # synth + P&R + pack -> gen_seq_silicon.bit
#   ./build_silicon.sh flash      # + flash via openFPGALoader -b colorlight-i9
#
# Why NODE_ONLY: see build_silicon.ps1 header. Without it the `top` node's
# redundant internal GG FSM blows synth to 139% LUT -> nextpnr cannot route.
# =============================================================================
set -euo pipefail

EXTRA="../src/ecp5_ecp5pll.v,../src/node_top.v,../src/flash_boot.v,../src/gen_seq.v,\
../src/async_fifo.v,../src/uart_bridge.v,../src/uart_rx_8n1.v,../src/uart_tx_8n1.v,\
../src/vec_alu2.v,../src/sdram.v,../src/mac18.v,../src/rmsnorm_op.v,../src/silu_op.v,\
../src/rope_op.v,../src/softmax_op.v,../src/attention_head_op.v"

exec "$(dirname "$0")/build.sh" -d NODE_ONLY,FLASH_BOOT -e "$EXTRA" gen_seq_silicon "$@"