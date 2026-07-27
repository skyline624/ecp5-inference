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
[CmdletBinding()] param([switch]$Flash, [switch]$Dim128, [switch]$Dim256, [switch]$Dim512)

# Source list for gen_seq_silicon (the wrapper + the `top` node with NODE_ONLY +
# the gen_seq sequencer + async_fifo + the 3 UART PHYs + vec_alu2 + sdram + mac18
# + the 5 operators + the PLL wrapper). Order-independent for yosys.
$Extra = (@(
    "../src/ecp5_ecp5pll.v",
    "../src/node_top.v",
    "../src/flash_boot.v",      # autonomous W25Q64 flash->SDRAM boot (ifdef FLASH_BOOT)
    "../src/wcache_ffn.v",       # cache FFN BRAM (instancie u_wc sous FLASH_BOOT, etape 3)
    "../src/gen_seq.v",
    "../src/kv_cdc.v",     # CDC clk<->clk_sys pour le maître KV SDRAM (étape 1)
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
    "../src/shared_lut.v",      # BRAM ROM 256x16 parametrable (exp_lut/inv_lut, migres depuis LUT-RAM)
    "../src/attention_head_op.v"
) -join ",")

# CLUSTER_NODE expose clk_sys_out (PLL bump etape 1, 25->31.25 MHz) -- requis depuis
# l'etape 1 perf : gen_seq_silicon.u_n reference le port clk_sys_out qui n'existe qu'en
# CLUSTER_NODE. Sans ce define : yosys ERROR "Module top ... does not have a port
# named 'clk_sys_out'".
# Appel build.ps1 avec params nommés explicites (le splatting @args est non fiable
# en PS 5.1 : -Top est traité comme valeur positionnelle -> $Top="-Top", synth cassée).
# -Dim128/-Dim256/-Dim512 : ajoute le profile DIM correspondant aux defines
#   (CFG_D=128/256/512 HS=16 NC=0 stream, rework dim Phase 7). Sert de points de la
#   courbe de capacité (LUT/BRAM/Fmax à chaque palier).
$defines = "NODE_ONLY,FLASH_BOOT,CLUSTER_NODE"
if ($Dim512)     { $defines += ",DIM512" }
elseif ($Dim256) { $defines += ",DIM256" }
elseif ($Dim128) { $defines += ",DIM128" }
if ($Flash) {
    & "$PSScriptRoot\build.ps1" -Top gen_seq_silicon -Extra $Extra -Defines $defines -Flash
} else {
    & "$PSScriptRoot\build.ps1" -Top gen_seq_silicon -Extra $Extra -Defines $defines
}