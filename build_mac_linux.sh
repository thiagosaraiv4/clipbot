#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  ClipBot — Script de Build (macOS / Linux)
# ═══════════════════════════════════════════════════════════

set -e

echo ""
echo "  ◀ CLIPBOT — BUILD"
echo "  ─────────────────────────────────"
echo ""

# Verifica Python
if ! command -v python3 &>/dev/null; then
    echo "[ERRO] Python 3 não encontrado."
    echo "  macOS:  brew install python"
    echo "  Linux:  sudo apt install python3 python3-pip"
    exit 1
fi

# Verifica ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "[AVISO] ffmpeg não encontrado."
    echo "  macOS:  brew install ffmpeg"
    echo "  Linux:  sudo apt install ffmpeg"
fi

echo "[1/5] Atualizando pip..."
python3 -m pip install --upgrade pip -q

echo "[2/5] Instalando dependências..."
pip3 install -r requirements.txt -q

echo "[3/5] Instalando PyInstaller..."
pip3 install pyinstaller -q

echo "[4/5] Compilando..."
pyinstaller clipbot.spec --clean --noconfirm

echo "[5/5] Preparando distribuição..."
rm -rf ClipBot-dist
mkdir -p ClipBot-dist

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: copia .app
    cp -r dist/ClipBot.app ClipBot-dist/ 2>/dev/null || \
    cp -r dist/ClipBot ClipBot-dist/
else
    cp -r dist/ClipBot ClipBot-dist/
fi

echo ""
echo "  ✅ Build concluído!"
echo "  ─────────────────────────────────"
echo "  Executável em: ClipBot-dist/"
echo ""
