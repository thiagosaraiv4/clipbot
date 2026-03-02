@echo off
REM ═══════════════════════════════════════════════════════
REM  ClipBot — Script de Build do Instalador (Windows)
REM  Execute este arquivo como Administrador
REM ═══════════════════════════════════════════════════════

echo.
echo  ██████╗██╗     ██╗██████╗ ██████╗  ██████╗ ████████╗
echo  ██╔════╝██║     ██║██╔══██╗██╔══██╗██╔═══██╗╚══██╔══╝
echo  ██║     ██║     ██║██████╔╝██████╔╝██║   ██║   ██║
echo  ██║     ██║     ██║██╔═══╝ ██╔══██╗██║   ██║   ██║
echo  ╚██████╗███████╗██║██║     ██████╔╝╚██████╔╝   ██║
echo   ╚═════╝╚══════╝╚═╝╚═╝     ╚═════╝  ╚═════╝    ╚═╝
echo.
echo  BUILD DO INSTALADOR
echo  ───────────────────────────────────────────────────
echo.

REM Verifica Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado!
    echo  Baixe em: https://www.python.org/downloads/
    pause & exit /b 1
)

REM Verifica ffmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [AVISO] ffmpeg nao encontrado no PATH.
    echo  Baixe em: https://ffmpeg.org/download.html
    echo  e adicione ao PATH do sistema.
    echo.
)

echo [1/5] Atualizando pip...
python -m pip install --upgrade pip --quiet

echo [2/5] Instalando dependencias...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERRO] Falha ao instalar dependencias!
    pause & exit /b 1
)

echo [3/5] Instalando PyInstaller...
pip install pyinstaller --quiet

echo [4/5] Compilando executavel...
pyinstaller clipbot.spec --clean --noconfirm
if errorlevel 1 (
    echo [ERRO] Build falhou!
    pause & exit /b 1
)

echo [5/5] Preparando pasta final...
if exist "ClipBot-Windows" rmdir /s /q "ClipBot-Windows"
mkdir "ClipBot-Windows"
xcopy "dist\ClipBot" "ClipBot-Windows" /e /i /q
copy "README.md" "ClipBot-Windows\" >nul 2>&1

echo.
echo  ✅ BUILD CONCLUIDO!
echo  ───────────────────────────────────────────────────
echo  Executavel em: ClipBot-Windows\ClipBot.exe
echo.
echo  Para usar:
echo   1. Abra ClipBot-Windows\ClipBot.exe
echo   2. Vá na aba "Contas" e faça login
echo   3. Configure as lives na aba "Lives"
echo   4. Clique em INICIAR
echo.
pause
