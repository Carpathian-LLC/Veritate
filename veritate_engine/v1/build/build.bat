@echo off
REM ------------------------------------------------------------------------------------
REM Developed by Carpathian, LLC.
REM ------------------------------------------------------------------------------------
REM Legal Notice: Distribution Not Authorized.
REM ------------------------------------------------------------------------------------
REM Notes:
REM - Windows build. clang from llvm-mingw. signs binary for SAC. writes to
REM   veritate_engine\bin\windows\x86_64\veritate.exe (repo-local).
REM veritate_engine/build/build.bat
REM ------------------------------------------------------------------------------------

setlocal enabledelayedexpansion

set CLANG=
for /f "delims=" %%i in ('where clang 2^>nul') do (
    if "!CLANG!"=="" set CLANG=%%i
)
if "%CLANG%"=="" (
    for /f "delims=" %%d in ('dir /b /a:d "%LOCALAPPDATA%\Microsoft\WinGet\Packages" 2^>nul ^| findstr /i MartinStorsjo') do (
        for /f "delims=" %%e in ('dir /b /a:d "%LOCALAPPDATA%\Microsoft\WinGet\Packages\%%d" 2^>nul ^| findstr /i llvm-mingw') do (
            set CLANG=%LOCALAPPDATA%\Microsoft\WinGet\Packages\%%d\%%e\bin\clang.exe
        )
    )
)
if "%CLANG%"=="" (
    echo no clang found. install via setup.ps1.
    exit /b 1
)

set ROOT=%~dp0..
set OUT=%ROOT%\bin\windows\x86_64
if not exist "%OUT%" mkdir "%OUT%"

set CFLAGS=-O3 -march=native -mavx2 -mavx512f -mavx512bw -mavx512vnni -Wall -Wextra -Wno-unused-parameter -DVERITATE_VERIFY_DECODE -DVERITATE_GELU_ZERO_THRESH=4

echo build: %CLANG%
"%CLANG%" %CFLAGS% ^
    "%ROOT%\src\main.c" ^
    "%ROOT%\src\dispatch.c" ^
    "%ROOT%\src\model.c" ^
    "%ROOT%\src\alloc.c" ^
    "%ROOT%\src\threadpool.c" ^
    "%ROOT%\kernels\scalar\matmul_scalar.c" ^
    "%ROOT%\kernels\scalar\transformer_scalar.c" ^
    "%ROOT%\kernels\x86_64\matmul_avx2.c" ^
    "%ROOT%\kernels\x86_64\matmul_vnni.c" ^
    "%ROOT%\kernels\x86_64\matmul_int4.c" ^
    "%ROOT%\kernels\x86_64\transformer_avx512.c" ^
    -o "%OUT%\veritate.exe"

if errorlevel 1 ( echo build failed & exit /b 1 )

echo sign: %OUT%\veritate.exe
powershell -NoProfile -Command "$c = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert | Where-Object { $_.Subject -eq 'CN=Veritate Dev' } | Select-Object -First 1; if ($c) { Set-AuthenticodeSignature -FilePath '%OUT%\veritate.exe' -Certificate $c -TimestampServer 'http://timestamp.digicert.com' | Out-Null; Write-Output 'signed' } else { Write-Output 'NO CERT FOUND - run setup-cert.ps1' }"

echo done: %OUT%\veritate.exe
