# =============================================================================
# synth_stat.ps1 -- Synthese yosys -noflatten pour mesurer le LUT par module.
#
# Genere stat_<Tag>.log avec le rapport synth_ecp5 -noflatten (preserve les
# frontieres de modules -> LUT decompose par rmsnorm_op / silu_op / softmax_op /
# attention_head_op / shared_lut) + stat hierarchique.
#
# Usage :
#   .\synth_stat.ps1 -Tag baseline    # avant migration (git stash des 4 modules)
#   .\synth_stat.ps1 -Tag after       # apres migration (working tree)
#
# Comparaison : le total LUT (LUT4) doit baisser de ~1-2k, et DP16KD (BRAM)
# doit augmenter de 6 (silu + rsqrt + 2x exp + 2x inv) entre baseline et after.
# =============================================================================
param(
    [Parameter(Mandatory=$true)][string]$Tag,
    [string]$Top = "gen_seq_silicon"
)
$ErrorActionPreference = "Continue"

# Copie les LUT .hex dans build/ (operators les chargent par cwd-relative $readmemh).
if (Test-Path "../lut/*.hex") {
    Copy-Item -Path "../lut/*.hex" -Destination "." -Force
}

# Liste des fichiers RTL (meme liste que stat_noflatten.log existant + shared_lut.v)
$Files = @(
    "../board/gen_seq_silicon.v",
    "../src/ecp5_ecp5pll.v",
    "../src/node_top.v",
    "../src/flash_boot.v",
    "../src/wcache_ffn.v",
    "../src/gen_seq.v",
    "../src/kv_cdc.v",
    "../src/async_fifo.v",
    "../src/uart_bridge.v",
    "../src/uart_rx_8n1.v",
    "../src/uart_tx_8n1.v",
    "../src/vec_alu2.v",
    "../src/sdram.v",
    "../src/mac18.v",
    "../src/rmsnorm_op.v",
    "../src/silu_op.v",
    "../src/rope_op.v",
    "../src/softmax_op.v",
    "../src/shared_lut.v",
    "../src/attention_head_op.v"
)
$Defs = "-DNODE_ONLY -DFLASH_BOOT -DCLUSTER_NODE -DDIM128"
$Inc  = "-I../src -I../board"

# Construit la commande yosys : read_verilog par fichier + hierarchy + proc + opt + synth_ecp5 -noflatten + stat
$cmd = ""
foreach ($f in $Files) { $cmd += "read_verilog $Defs $Inc $f; " }
$cmd += "hierarchy -top $Top; proc; opt; synth_ecp5 -noflatten -top $Top -json ${Tag}_${Top}.json; stat -top $Top"

Write-Host "==> yosys synth -noflatten ($Tag -> stat_$Tag.log)" -ForegroundColor Cyan
yosys -l "stat_$Tag.log" -p $cmd
if ($LASTEXITCODE -ne 0) { Write-Error "yosys failed for tag=$Tag"; exit 1 }
Write-Host "==> stat_$Tag.log ecrit" -ForegroundColor Green