# =============================================================================
# build_silicon.ps1 -- One-command build of the gen_seq_silicon (Gate 6') bitstream.
# Wraps build.ps1 with the right source list + -D NODE_ONLY.
#
#   .\build_silicon.ps1            # synth + P&R + pack -> gen_seq_silicon.bit
#   .\build_silicon.ps1 -Flash     # + flash via openFPGALoader -b colorlight-i9
#
# Why NODE_ONLY: the `top` node (src/node_top.v) embeds a generation FSM `GG`
# (~343 states, guarded by `ifndef NODE_ONLY) that is REDUNDANT with the
# `gen_seq` sequencer which gen_seq_silicon.v drives. Without NODE_ONLY the synth
# blows to 139% LUT and nextpnr cannot route on ECP5-45. With it: 57% LUT, Fmax
# 33.56 MHz, routes cleanly. See memory note gen-silicon-nodeonly-fix.
# =============================================================================
[CmdletBinding()] param([switch]$Flash)

# Source list for gen_seq_silicon (the wrapper + the `top` node with NODE_ONLY +
# the gen_seq sequencer + async_fifo + the 3 UART PHYs + vec_alu2 + sdram + mac18
# + the 5 operators + the PLL wrapper). Order-independent for yosys.
$Extra = (@(
    "../src/ecp5_ecp5pll.v",
    "../src/node_top.v",
    "../src/flash_boot.v",      # autonomous W25Q64 flash->SDRAM boot (ifdef FLASH_BOOT)
    "../src/gen_seq.v",
    "../src/async_fifo.v",
    "../src/uart_bridge.v",      # rx8_link / tx8_link (used by `top` under LINK_SS)
    "../src/uart_rx_8n1.v",       # UART PHY used by gen_seq_silicon.v host UART
    "../src/uart_tx_8n1.v",
    "../src/vec_alu2.v",
    "../src/sdram.v",
    "../src/mac18.v",
    "../src/rmsnorm_op.v",
    "../src/silu_op.v",
    "../src/rope_op.v",
    "../src/softmax_op.v",
    "../src/attention_head_op.v"
) -join ",")

$args = @("-Top","gen_seq_silicon","-Extra",$Extra,"-Defines","NODE_ONLY,FLASH_BOOT")
if ($Flash) { $args += "-Flash" }
& "$PSScriptRoot\build.ps1" @args