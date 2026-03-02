# clipbot.spec — ClipBot v2.4
# Gera ClipBot.exe / ClipBot (Linux/macOS)
# Uso: pyinstaller clipbot.spec
#
# Entrypoint: app.py (pywebview — SEM Flask/SocketIO)

import sys
from pathlib import Path

block_cipher = None

added_files = [
    # UI HTML + assets
    ("ui",                          "ui"),
    # Core Python modules
    ("core/detection",              "detection"),
    ("core/editing",                "editing"),
    ("core/upload",                 "upload"),
    ("core/stream_capture.py",      "."),
    ("core/description_generator.py", "."),
    # Raiz
    ("clipbot_api.py",              "."),
    ("credentials.py",              "."),
]

a = Analysis(
    ["app.py"],
    pathex=[
        str(Path(".")),
        str(Path("core")),
        str(Path("core/detection")),
        str(Path("core/editing")),
        str(Path("core/upload")),
    ],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        # ── GUI ──────────────────────────────────────────────
        "webview",
        "webview.platforms.edgechromium",   # Windows
        "webview.platforms.gtk",            # Linux
        "webview.platforms.cocoa",          # macOS
        "PIL",
        "PIL._tkinter_finder",

        # ── Stream / Captura ─────────────────────────────────
        "streamlink",
        "requests",

        # ── Áudio / Detecção ─────────────────────────────────
        "numpy",

        # ── Whisper ──────────────────────────────────────────
        "whisper",
        "torch",
        "tqdm",

        # ── Upload TikTok (Playwright) ────────────────────────
        "playwright",
        "playwright.sync_api",

        # ── Upload Instagram ──────────────────────────────────
        "instagrapi",

        # ── Upload YouTube ────────────────────────────────────
        "googleapiclient",
        "google.auth",
        "google.oauth2",
        "google_auth_oauthlib",
        "pickle",

        # ── IA / Descrições ───────────────────────────────────
        "anthropic",

        # ── Stdlib usados dinamicamente ───────────────────────
        "http.server",
        "socketserver",
        "mimetypes",
        "urllib.parse",
        "threading",
        "queue",
        "atexit",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Removidos da arquitetura v2.4
        "flask", "flask_socketio", "engineio", "socketio",
        # Não usados
        "customtkinter", "pystray",
        "librosa", "soundfile",
        "moviepy", "ffmpeg",
        "tiktok_uploader",
        "twitchio",
        "aiohttp", "websockets",
        "yaml", "colorlog",
        # Pesados desnecessários
        "matplotlib", "scipy", "pandas", "IPython", "jupyter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ClipBot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,      # sem janela de console no Windows
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="assets/icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ClipBot",
)
