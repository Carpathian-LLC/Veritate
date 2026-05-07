# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - one-time toolchain setup. installs LLVM (clang) and NASM via winget.
# ------------------------------------------------------------------------------------

$ErrorActionPreference = "Stop"

Write-Host "veritate setup — installing toolchain"

function Install-IfMissing($id, $cmd) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        Write-Host "  $cmd already installed"
        return
    }
    Write-Host "  installing $id ..."
    winget install --id=$id --silent --accept-source-agreements --accept-package-agreements
}

Install-IfMissing -id "LLVM.LLVM" -cmd "clang"
Install-IfMissing -id "NASM.NASM" -cmd "nasm"

Write-Host ""
Write-Host "done. open a new terminal so PATH refreshes, then run build.bat"
