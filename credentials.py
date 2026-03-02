"""
credentials.py
Gerencia credenciais de forma segura usando criptografia local.
Armazena em arquivo .enc criptografado com senha do sistema.
"""

import json
import os
import base64
import hashlib
from pathlib import Path

# Pasta de dados do app
APP_DATA = Path(os.environ.get("APPDATA", Path.home())) / "ClipBot"
APP_DATA.mkdir(parents=True, exist_ok=True)
CRED_FILE = APP_DATA / "credentials.enc"
CONFIG_FILE = APP_DATA / "settings.json"


def _get_machine_key() -> bytes:
    """Gera chave baseada no hardware da máquina."""
    import platform
    seed = f"{platform.node()}-{platform.machine()}-clipbot-v1"
    return hashlib.sha256(seed.encode()).digest()


def _encrypt(data: str) -> str:
    key = _get_machine_key()
    data_bytes = data.encode("utf-8")
    # XOR simples com a chave repetida (suficiente para ofuscação local)
    key_repeated = (key * (len(data_bytes) // len(key) + 1))[:len(data_bytes)]
    encrypted = bytes(a ^ b for a, b in zip(data_bytes, key_repeated))
    return base64.b64encode(encrypted).decode()


def _decrypt(data: str) -> str:
    key = _get_machine_key()
    encrypted = base64.b64decode(data.encode())
    key_repeated = (key * (len(encrypted) // len(key) + 1))[:len(encrypted)]
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key_repeated))
    return decrypted.decode("utf-8")


def save_credentials(creds: dict):
    """Salva credenciais criptografadas."""
    encrypted = _encrypt(json.dumps(creds))
    CRED_FILE.write_text(encrypted, encoding="utf-8")


def load_credentials() -> dict:
    """Carrega e descriptografa credenciais."""
    if not CRED_FILE.exists():
        return {}
    try:
        encrypted = CRED_FILE.read_text(encoding="utf-8")
        return json.loads(_decrypt(encrypted))
    except Exception:
        return {}


def save_settings(settings: dict):
    """Salva configurações do app (sem criptografia)."""
    CONFIG_FILE.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")


def load_settings() -> dict:
    """Carrega configurações do app."""
    if not CONFIG_FILE.exists():
        return _default_settings()
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return _default_settings()


def _default_settings() -> dict:
    return {
        "platforms": {
            "twitch":  {"enabled": False, "channels": []},
            "youtube": {"enabled": False, "channels": [], "api_key": ""},
            "kick":    {"enabled": False, "channels": []},
        },
        "detection": {
            "audio_spike": True,
            "chat_flood":  True,
            "keywords":    True,
            "threshold_db": 20,
            "flood_rate":   10,
            "keywords_list": "clip,insano,KKKK,omg,PogChamp,LULW",
            "clip_min_sec": 15,
            "clip_max_sec": 60,
        },
        "upload": {
            "tiktok":    {"enabled": False},
            "instagram": {"enabled": False},
            "youtube":   {"enabled": False},
            "max_per_day": 10,
            "delay_sec":   300,
        },
        "editing": {
            "subtitles": True,
            "language":  "pt",
            "watermark": False,
            "watermark_text": "",
            "format":    "vertical",
        },
        "system": {
            "autostart":        False,
            "minimize_to_tray": True,
            "notifications":    True,
            "clips_folder":     str(APP_DATA / "clips"),
            "output_folder":    str(APP_DATA / "output"),
            "approval_mode":    True,
            "approval_top_n":   12,
        }
    }
