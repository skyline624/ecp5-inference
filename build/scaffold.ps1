# =============================================================================
# scaffold.ps1 -- (Re)populate the ecp5/ folder from master src/ + the cluster
# branch. Idempotent. Run from repo root. Keeps src/ (Gowin) and the branch
# untouched -- ecp5/ is a portable copy.
# =============================================================================
$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot/../.."
Set-Location $repoRoot

$branch = "origin/refactor/unified-node-ss-link"
New-Item -ItemType Directory -Force ecp5/src, ecp5/board, ecp5/build, ecp5/sim/models, ecp5/lut, ecp5/docs | Out-Null

Write-Host "==> portable ops + hex from master src/" -ForegroundColor Cyan
Copy-Item src\rmsnorm_op.v, src\silu_op.v, src\rope_op.v, src\softmax_op.v, `
          src\attention_head_op.v, src\sdram.v, src\uart_rx_8n1.v, src\uart_tx_8n1.v, `
          src\top.v ecp5\src\ -Force
Copy-Item src\*.hex ecp5\lut\ -Force

Write-Host "==> cluster lineage + freq_cis LUT from $branch" -ForegroundColor Cyan
$branchFiles = @(
    "src/freq_cis_cos.hex","src/freq_cis_sin.hex",
    "sim/link/async_fifo.v","sim/link/gen_seq.v","sim/link/gen_seq_top.v",
    "sim/link/ss_link.v","sim/link/link_send.v","sim/link/link_recv.v",
    "sim/link/uart_bridge.v","sim/link/vec_alu2.v",
    "sim/link/ring_node.v","sim/link/ring_cluster.v",
    "sim/link/tp_node.v","sim/link/tp_mm_node.v","sim/link/tp_ar_node.v",
    "sim/ffn_tp_seq2.v","sim/attn_tp_seq.v",
    "docs/PLAN_ECP5_BRINGUP.md","docs/PLAN_GEN_SEQUENCER.md",
    "docs/REFACTOR_UNIFIED_NODE.md","docs/SESSION_PROMPTS.md",
    "sim/models/MULTALU18X18.v","sim/models/sdram_chip.v","sim/models/Gowin_rPLL.v"
)
foreach ($f in $branchFiles) {
    $leaf = Split-Path $f -Leaf
    $dest = switch -Wildcard ($f) {
        "src/*"        { "ecp5/lut/$leaf" }
        "sim/link/*"   { "ecp5/src/$leaf" }
        "sim/*"        { "ecp5/src/$leaf" }
        "docs/*"       { "ecp5/docs/$leaf" }
        "sim/models/*" { "ecp5/sim/models/$leaf" }
        default        { "ecp5/$leaf" }
    }
    git show "${branch}:$f" | Out-File -FilePath $dest -Encoding utf8 -Force
    Write-Host "  $f -> $dest"
}

Write-Host "==> done. ecp5/ scaffolded." -ForegroundColor Green
Write-Host "Next: install oss-cad-suite + cocotb (see ecp5/README.md), then build.ps1 -Top blink."