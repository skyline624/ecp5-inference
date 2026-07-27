# bisect_test.ps1 -- Script de test pour git bisect sur la sim Verilator dim128.
# Reproduit le flow de run_verilator_fb.ps1 MAIS :
#  - detecte le Python (machine 'pc', pas 'User') en patchant le wrapper recopie,
#  - regenere le dump dim128 (host/gen_synthetic_dim.py --dim 128) a chaque commit,
#  - nettoie vobj/ pour un build propre a chaque commit.
# Exit codes : 0 = GOOD (PASS), 1 = BAD (mismatch), 125 = SKIP (compile/build error).
$ErrorActionPreference = "Continue"

# --- oss-cad-suite env (PATH verilator_bin + cl.exe via wrapper) ---
. C:\oss-cad-suite\environment.ps1

# --- Python : detection (le wrapper original hardcode C:\Users\User\...) ---
$py = $env:PYTHON
if (-not ($py -and (Test-Path $py))) {
    $gc = Get-Command python -ErrorAction SilentlyContinue
    if ($gc) { $py = $gc.Source } else { $py = "C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe" }
}

$root = Split-Path -Parent $PSScriptRoot        # repo root
Set-Location $PSScriptRoot                       # sim/

# --- 1. regen dump dim128 (fidelle au commit courant) ---
$dumpDim = Join-Path $root "host\gen_model_dump_dim128.json"
& $py (Join-Path $root "host\gen_synthetic_dim.py") --dim 128 2>&1 | Out-Null
if (-not (Test-Path $dumpDim)) { Write-Output "SKIP: dump dim128 non genere"; exit 125 }

# --- 2. patcher le wrapper recopie pour forcer $py ---
$origWrapper = Join-Path $PSScriptRoot "run_verilator_fb.ps1"
$tmpWrapper  = Join-Path $PSScriptRoot "_bisect_wrapper.ps1"
$content = Get-Content $origWrapper -Raw
# Remplacer la ligne $py = "C:\Users\User\..." (forme originale) par le python detecte.
# On couvre les variantes : ligne hardcodee OU deja patchee (detection).
$patched = $content -replace '\$py\s*=\s*"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python312\\python\.exe"', "`$py = `"$py`""
if ($patched -eq $content) {
    # deja patche (mon patch) : on force quand meme $py en tete apres le . environment.ps1
    $patched = $content -replace '(\. C:\\oss-cad-suite\\environment\.ps1)', "`$1`r`n`$py = `"$py`"  # bisect override"
}
Set-Content -Path $tmpWrapper -Value $patched -Encoding UTF8

# --- 3. nettoyer vobj/ pour build propre ---
if (Test-Path "vobj") { Remove-Item "vobj" -Recurse -Force -ErrorAction SilentlyContinue }

# --- 4. lancer la sim dim128 -Ntok 3, capturer sortie + exit code ---
Write-Output "=== bisect test : $(git -C $root rev-parse --short HEAD) ==="
$out = & powershell -ExecutionPolicy Bypass -NoProfile -File $tmpWrapper -Dim128 -Ntok 3 2>&1
$rc = $LASTEXITCODE
$out | Select-Object -Last 8
Write-Output "=== wrapper exit=$rc ==="

# menage du wrapper temp
Remove-Item $tmpWrapper -Force -ErrorAction SilentlyContinue

# --- 5. verdict ---
if ($out -match "PASS: \d+/\d+ bit-exact") { Write-Output "BISECT: GOOD"; exit 0 }
if ($out -match "FAIL: \d+ mismatch")      { Write-Output "BISECT: BAD"; exit 1 }
if ($out -match "verilator --cc failed|build failed|cannot open") { Write-Output "BISECT: SKIP"; exit 125 }
# fallback sur exit code du wrapper
if ($rc -eq 0) { Write-Output "BISECT: GOOD (rc)"; exit 0 }
if ($rc -eq 1) { Write-Output "BISECT: BAD (rc)"; exit 1 }
Write-Output "BISECT: SKIP (rc=$rc)"; exit 125