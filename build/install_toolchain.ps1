# =============================================================================
# install_toolchain.ps1 -- Extract oss-cad-suite to C:\oss-cad-suite and add
# it (bin AND lib) to the user PATH. Run AFTER downloading
# oss-cad-suite-windows-x64-*.exe (self-extracting 7z) into C:\Users\User\Downloads\.
#
#   .\install_toolchain.ps1
#
# Then OPEN A NEW SHELL and verify:
#   yosys -V ; nextpnr-ecp5 --version ; ecppack --version
#   openFPGALoader --version ; iverilog -V
#
# NOTE: oss-cad-suite tools need BOTH bin\ (executables) AND lib\ (DLLs like
# libreadline8.dll) on PATH -- lib\ alone is not enough, bin\ alone fails to
# find DLLs. environment.ps1 in the suite root does the same; this script
# persists it so every new shell has it.
# =============================================================================
$ErrorActionPreference = "Stop"
$installer = "C:\Users\User\Downloads\oss-cad-suite-windows-x64-20260720.exe"
$root     = "C:\oss-cad-suite\"

if (-not (Test-Path $installer)) { throw "Installer not found: $installer" }
if (Test-Path "$($root)bin\yosys.exe") {
    Write-Host "oss-cad-suite already extracted at $root" -ForegroundColor Yellow
} else {
    Write-Host "==> Extracting $installer -> $root" -ForegroundColor Cyan
    & $installer -o"$($root.TrimEnd('\'))" -y
    if ($LASTEXITCODE -ne 0) { throw "Extraction failed (exit $LASTEXITCODE)" }
    # The SFX nests an inner oss-cad-suite\ dir; flatten it if so.
    if (Test-Path "$($root)oss-cad-suite\bin\yosys.exe") {
        Write-Host "==> Flattening nested oss-cad-suite\ dir" -ForegroundColor Cyan
        Get-ChildItem "$($root)oss-cad-suite" | ForEach-Object { Move-Item $_.FullName $root -Force }
        Remove-Item "$($root)oss-cad-suite" -Recurse -Force
    }
}
if (-not (Test-Path "$($root)bin\yosys.exe")) { throw "yosys.exe not found after extraction" }
Write-Host "==> yosys present at $($root)bin\yosys.exe" -ForegroundColor Green

# Persist oss-cad-suite (bin + lib) at the FRONT of the user PATH.
$cur = [Environment]::GetEnvironmentVariable("Path","User")
$prefix = "$($root)bin;$($root)lib;"
if ($cur -notlike "*$($root)bin*") {
    [Environment]::SetEnvironmentVariable("Path", $prefix + $cur, "User")
    Write-Host "==> Added $($root)bin + lib to user PATH (front). Open a NEW shell to pick it up." -ForegroundColor Green
} else {
    Write-Host "==> $($root)bin already on user PATH." -ForegroundColor Yellow
}

# openFPGALoader needs its board database dir; yosys/nextpnr SSL certs.
[Environment]::SetEnvironmentVariable("OPENFPGALOADER_SOJ_DIR", "$($root)share\openFPGALoader", "User")
[Environment]::SetEnvironmentVariable("SSL_CERT_FILE", "$($root)etc\cacert.pem", "User")

Write-Host "`nVerify in a NEW shell:" -ForegroundColor Cyan
Write-Host "  yosys -V ; nextpnr-ecp5 --version ; ecppack --version ; openFPGALoader --version ; iverilog -V"
Write-Host "`nThen build the blink:" -ForegroundColor Cyan
Write-Host "  cd ecp5\build ; .\build.ps1 -Top blink -Flash"