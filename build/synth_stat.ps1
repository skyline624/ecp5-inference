# =============================================================================
# synth_stat.ps1 -- Synthese yosys -noflatten pour mesurer le LUT par module.
#
# Genere stat_<Tag>.log avec le rapport synth_ecp5 -noflatten (preserve les
# frontieres de modules -> LUT decompose par rmsnorm_op / silu_op / softmax_op /
# attention_head_op / shared_lut) + stat hierarchique.
#
# Usage :
#   .\synth_stat.ps1 -Tag dim64             # profil defaut (cache FFN cache-all, mur BRAM)
#   .\synth_stat.ps1 -Tag dim128 -Dim 128   # profil dim128 (CFG_NC=0, stream SDRAM)
#
# -Dim selectionne le profil model_cfg.vh (64 = aucun define, sinon -DDIM<n>). Le
# profil change radicalement le compte BRAM : en dim64 CFG_NC=5 (cache-all) donne
# ~90 DP16KD pour wcache_ffn, en dim128+ CFG_NC=0 (stream) le reduit a 1.
# =============================================================================
param(
    [Parameter(Mandatory=$true)][string]$Tag,
    [ValidateSet("64","128","256","512")][string]$Dim = "64",
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
$Defs = "-DNODE_ONLY -DFLASH_BOOT -DCLUSTER_NODE"
if ($Dim -ne "64") { $Defs += " -DDIM$Dim" }   # dim64 = profil defaut model_cfg.vh (aucun define)
$Inc  = "-I../src -I../board"

# Construit la commande yosys : read_verilog par fichier + hierarchy + proc + opt + synth_ecp5 -noflatten + stat
$cmd = ""
foreach ($f in $Files) { $cmd += "read_verilog $Defs $Inc $f; " }
$cmd += "hierarchy -top $Top; proc; opt; synth_ecp5 -noflatten -top $Top -json ${Tag}_${Top}.json; stat -top $Top"

Write-Host "==> yosys synth -noflatten ($Tag -> stat_$Tag.log)" -ForegroundColor Cyan
yosys -l "stat_$Tag.log" -p $cmd
if ($LASTEXITCODE -ne 0) { Write-Error "yosys failed for tag=$Tag"; exit 1 }
Write-Host "==> stat_$Tag.log ecrit" -ForegroundColor Green