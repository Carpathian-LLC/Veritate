@echo off
REM ------------------------------------------------------------------------------------
REM Developed by Carpathian, LLC.
REM ------------------------------------------------------------------------------------
REM Legal Notice: Distribution Not Authorized.
REM ------------------------------------------------------------------------------------
REM Notes:
REM - Windows build. clang from llvm-mingw. signs binary for SAC. writes to
REM   veritate_engine\bin\windows\x86_64\veritate.exe (repo-local).
REM - Mirrors build.sh: compiles each TU separately so shared code gets a safe
REM   baseline ISA (SSE4.2) and only the specialized kernel TUs get AVX2/AVX512
REM   flags. src\dispatch.c gates each kernel by CPUID at runtime, so a binary
REM   built with the highest kernel ISA still runs on older CPUs without it.
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

set OBJDIR=%~dp0obj
if not exist "%OBJDIR%" mkdir "%OBJDIR%"
del /q "%OBJDIR%\*.o" 2>nul

REM Common flags for every TU. Baseline ISA (SSE4.2) covers any x86_64 CPU the
REM engine targets; per-kernel flags are added only where a TU needs them.
set CFLAGS_COMMON=-O3 -flto=full -Wall -Wextra -Wno-unused-parameter -DVERITATE_VERIFY_DECODE -DVERITATE_GELU_ZERO_THRESH=4 -DV_SEQ=1024
set BASELINE=-msse4.2

set OBJS=

echo build: %CLANG%

REM Pass 1: shared TUs at baseline.
for %%f in (
    src\main.c
    src\dispatch.c
    src\model.c
    src\alloc.c
    src\threadpool.c
    src\fsutil.c
    src\state_cache.c
    src\addons.c
    src\addons\slot_table.c
    kernels\scalar\matmul_scalar.c
    kernels\scalar\matmul_ternary_scalar.c
    kernels\scalar\transformer_scalar.c
) do ( call :cc "%%f" "" || goto :buildfail )

REM Pass 2: hybrid dispatcher + per-ISA kernels. -ffp-contract=off pins mul+add
REM so the fp32 matvec lanes stay bitwise-identical to the scalar reference.
call :cc "src\hybrid.c"                          "-ffp-contract=off"                              || goto :buildfail
call :cc "kernels\x86_64\matvec_f32_avx2.c"      "-mavx2 -mf16c -ffp-contract=off"               || goto :buildfail
call :cc "kernels\x86_64\matmul_prefill_avx2.c"  "-mavx2 -mf16c -ffp-contract=off"               || goto :buildfail
call :cc "kernels\x86_64\matmul_avx2.c"          "-mavx2"                                        || goto :buildfail
call :cc "kernels\x86_64\matmul_vnni.c"          "-mavx2 -mavx512f -mavx512bw -mavx512vl -mavx512vnni" || goto :buildfail
call :cc "kernels\x86_64\matmul_int4.c"          "-mavx2 -mavx512f -mavx512bw -mavx512vl -mavx512vnni" || goto :buildfail
call :cc "kernels\x86_64\matmul_ternary_vnni.c"  "-mavx2 -mavx512f -mavx512bw -mavx512vl -mavx512vnni" || goto :buildfail
call :cc "kernels\x86_64\transformer_avx512.c"   "-mavx512f -mavx512bw -mavx512vl"               || goto :buildfail

REM Pass 3: link.
echo link: %OUT%\veritate.exe
"%CLANG%" %CFLAGS_COMMON% %BASELINE% !OBJS! -o "%OUT%\veritate.exe"
if errorlevel 1 goto :buildfail

echo sign: %OUT%\veritate.exe
powershell -NoProfile -Command "$c = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert | Where-Object { $_.Subject -eq 'CN=Veritate Dev' } | Select-Object -First 1; if ($c) { Set-AuthenticodeSignature -FilePath '%OUT%\veritate.exe' -Certificate $c -TimestampServer 'http://timestamp.digicert.com' | Out-Null; Write-Output 'signed' } else { Write-Output 'NO CERT FOUND - run setup-cert.ps1' }"

echo done: %OUT%\veritate.exe
exit /b 0

REM ------------------------------------------------------------------------------------
REM :cc <src-relative-to-ROOT> <extra-flags>  -> compiles to %OBJDIR%\<basename>.o
REM and appends the object to OBJS. Returns nonzero on compile failure.
REM ------------------------------------------------------------------------------------
:cc
set "SRC=%~1"
set "EXTRA=%~2"
set "OBJ=%OBJDIR%\%~n1.o"
"%CLANG%" %CFLAGS_COMMON% %BASELINE% %EXTRA% -c "%ROOT%\%SRC%" -o "%OBJ%"
if errorlevel 1 ( echo compile failed: %SRC% & exit /b 1 )
set OBJS=!OBJS! "%OBJ%"
goto :eof

:buildfail
echo build failed
exit /b 1
