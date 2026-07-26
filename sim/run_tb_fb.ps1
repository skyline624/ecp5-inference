# =============================================================================
# run_tb_fb.ps1 -- Lance le TB Verilog pur tb_gen_seq_top_fb.v (flot SANS cocotb).
# Remplace run_test.ps1 -FlashBoot pour la sim gen_seq_top : clock + watchdog en
# Verilog/C (pas de callback Python par cycle), ~Nx plus rapide que cocotb.
#
# Usage (depuis sim/) :
#   .\run_tb_fb.ps1                      # HS=8 stories260K, 17 tokens
#   .\run_tb_fb.ps1 -Hs16                # HS=16 Strat A (gen_model_dump_hs16_rand.json)
#   .\run_tb_fb.ps1 -Ntok 3              # 3 tokens seulement (iteration rapide)
#   .\run_tb_fb.ps1 -Dump ..\host\x.json -Ntok 17
#
# Prerequis : source C:\oss-cad-suite\environment.ps1 (iverilog/vvp + DLLs).
# Genere les hex via gen_tb_hex.py si absents ou -RegenHex.
# =============================================================================
param(
    [switch]$Hs16,
    [int]$Ntok = 17,
    [string]$Dump = "",
    [switch]$RegenHex
)

. C:\oss-cad-suite\environment.ps1
$py = "C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe"
Set-Location $PSScriptRoot

if ($Dump -eq "") {
    if ($Hs16) { $Dump = "..\host\gen_model_dump_hs16_rand.json" }
    else       { $Dump = "..\host\gen_model_dump.json" }
}
$env:DUMP = $Dump

# ---- hex shifts/expected (toujours regen : switch HS8<->HS16 change les valeurs) ----
Write-Host "==> gen_tb_hex.py (Dump=$Dump, NTOK=$Ntok)" -ForegroundColor Cyan
& $py gen_tb_hex.py --dump $Dump --ntok $Ntok

# ---- SDRAM preload : regenere depuis le Dump (evite le bug de swap HS8/HS16 :
# sdram_chip.v lit "sdram_preload.hex" en dur -> on l'ecrit tjrs depuis le Dump
# selectionne, sinon un run HS=16 laisse le preload HS=16 et casse le run HS=8) ----
Write-Host "==> gen_sdram_preload_hex.py (Dump=$Dump)" -ForegroundColor Cyan
$env:DUMP = $Dump
& $py ..\host\gen_sdram_preload_hex.py | Select-Object -Last 1

# ---- compile iverilog ----
$srcList = "tb_gen_seq_top_fb.v models/ECP5_PLL_sim.v models/MULTALU18X18.v " +
           "models/sdram_chip.v models/flash_boot_sim.v " +
           "../src/node_top.v ../src/wcache_ffn.v ../src/sdram.v ../src/mac18.v " +
           "../src/rmsnorm_op.v ../src/silu_op.v ../src/rope_op.v ../src/softmax_op.v " +
           "../src/attention_head_op.v ../src/vec_alu2.v ../src/async_fifo.v " +
           "../src/uart_bridge.v ../src/gen_seq.v ../src/gen_seq_top.v"
$defs = "-DLINK_SS -DSDRAM_PRELOAD -DSIM -DFLASH_BOOT"
if ($Hs16) { $defs += " -DHS16_PROFILE" }

Write-Host "==> iverilog compile (tb_gen_seq_top_fb -> tb.vvp)" -ForegroundColor Cyan
$sw = [System.Diagnostics.Stopwatch]::StartNew()
cmd /c "iverilog -o tb.vvp -s tb_gen_seq_top_fb -g2012 $defs -I../src $srcList 2>&1"
if ($LASTEXITCODE -ne 0) { throw "iverilog compile failed (exit $LASTEXITCODE)" }
$sw.Stop(); Write-Host ("  compile: {0:N1}s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor DarkGray

# ---- run vvp (mesure wall-clock) ----
Write-Host "==> vvp tb.vvp (NTOK=$Ntok, HS16=$Hs16)" -ForegroundColor Cyan
$sw2 = [System.Diagnostics.Stopwatch]::StartNew()
cmd /c "vvp tb.vvp > tb_run.log 2>&1"
$rc = $LASTEXITCODE
$sw2.Stop()
Write-Host ("  vvp wall-clock: {0:N1}s (exit=$rc)" -f $sw2.Elapsed.TotalSeconds) -ForegroundColor $(if ($rc -eq 0) {"Green"} else {"Yellow"})
Write-Host "==> output dans tb_run.log" -ForegroundColor DarkGray
# affiche le resultat (PASS/FAIL + tokens)
Get-Content tb_run.log | Select-String -Pattern "PASS|FAIL|MISMATCH|tok\[" | Select-Object -Last 20