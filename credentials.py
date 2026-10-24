"""
credentials.py
Gerencia credenciais de forma segura usando criptografia local.
Armazena em arquivo .enc criptografado com senha do sistema.
"""

import json
import logging
import os
import base64
import hashlib
from pathlib import Path
from typing import Optional

import safe_io

logger = logging.getLogger("ClipBot.Credentials")

# Pasta de dados do app
APP_DATA = Path(os.environ.get("APPDATA", Path.home())) / "ClipBot"
APP_DATA.mkdir(parents=True, exist_ok=True)
CRED_FILE = APP_DATA / "credentials.enc"
CONFIG_FILE = APP_DATA / "settings.json"


# ── Saúde do armazenamento ───────────────────────────────────────────────────
#
# Um load que falha devolve {} / defaults para o app conseguir subir. Se o save
# seguinte gravasse esse vazio, ele APAGARIA os dados bons que ainda estão no
# disco — foi o que quase aconteceu em 12/08, quando credentials.enc ficou
# ilegível por permissão e a UI continuou oferecendo "salvar".
#
# Enquanto a última leitura de um arquivo estiver degradada, a escrita dele é
# recusada. Uma leitura bem-sucedida restaura o estado saudável.

class StorageDegraded(RuntimeError):
    """A última leitura falhou; gravar agora sobrescreveria dados bons."""


_degraded: dict = {}      # "credentials" | "settings"  →  motivo


def _mark_degraded(kind: str, reason: str) -> None:
    _degraded[kind] = reason


def _mark_healthy(kind: str) -> None:
    if _degraded.pop(kind, None) is not None:
        logger.info(f"{kind}: leitura voltou a funcionar, escrita reabilitada.")


def storage_degraded(kind: Optional[str] = None) -> bool:
    """True se o arquivo (ou qualquer um, sem argumento) está degradado."""
    return bool(_degraded) if kind is None else kind in _degraded


def degraded_reason(kind: str) -> Optional[str]:
    return _degraded.get(kind)


def reset_storage_health() -> None:
    """Limpa o estado de saúde (boot e testes)."""
    _degraded.clear()


def _guard_write(kind: str, path: Path) -> None:
    reason = _degraded.get(kind)
    if reason is None:
        return
    raise StorageDegraded(
        f"escrita em {path.name} recusada: a última leitura falhou ({reason}). "
        "Gravar agora apagaria dados que ainda estão no disco. "
        "Verifique permissões/antivírus e reinicie o ClipBot."
    )


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
    """
    Salva credenciais criptografadas (atômico, com retry).

    Raises:
        StorageDegraded: se a última leitura falhou (ver _guard_write).
        safe_io.StorageWriteError: se o disco recusou a escrita.
    """
    _guard_write("credentials", CRED_FILE)
    safe_io.atomic_write_text(CRED_FILE, _encrypt(json.dumps(creds)))


def load_credentials() -> dict:
    """
    Carrega e descriptografa credenciais.

    Devolve {} em qualquer falha — mas, ao contrário da versão anterior, marca
    o estado como degradado e registra ERROR quando o arquivo existe e não pôde
    ser lido. "Não existe" e "não consegui ler" param de ser o mesmo evento.
    """
    status, raw, err = safe_io.read_text_classified(CRED_FILE)

    if status == safe_io.MISSING:
        _mark_healthy("credentials")
        return {}

    if status != safe_io.OK:
        _mark_degraded("credentials", err or status)
        logger.error(
            f"❌ {CRED_FILE.name} existe mas não pôde ser lido ({err}). "
            "Tokens e chaves de API indisponíveis nesta sessão — "
            "a escrita está bloqueada para não apagar o arquivo."
        )
        return {}

    try:
        data = json.loads(_decrypt(raw))
        if not isinstance(data, dict):
            raise ValueError(f"esperado objeto JSON, veio {type(data).__name__}")
    except Exception as ex:
        _mark_degraded("credentials", f"conteúdo inválido: {ex}")
        logger.error(
            f"❌ {CRED_FILE.name} corrompido ou gerado em outra máquina ({ex}). "
            "A escrita está bloqueada; apague o arquivo para recomeçar."
        )
        return {}

    _mark_healthy("credentials")
    return data


def save_settings(settings: dict):
    """
    Salva configurações do app (atômico, com retry).

    Raises:
        StorageDegraded: se a última leitura falhou — impede que a UI
            sobrescreva a config real com os defaults.
        safe_io.StorageWriteError: se o disco recusou a escrita.
    """
    _guard_write("settings", CONFIG_FILE)
    safe_io.atomic_write_text(
        CONFIG_FILE, json.dumps(settings, indent=2, ensure_ascii=False)
    )


def load_settings() -> dict:
    """
    Carrega configurações do app.

    Devolve os defaults em falha (o app precisa subir), mas marca o estado como
    degradado: sem isso, um "salvar" na UI trocaria os canais e tokens reais
    pelos defaults vazios.
    """
    status, raw, err = safe_io.read_text_classified(CONFIG_FILE)

    if status == safe_io.MISSING:
        _mark_healthy("settings")
        return _default_settings()

    if status != safe_io.OK:
        _mark_degraded("settings", err or status)
        logger.error(
            f"❌ {CONFIG_FILE.name} existe mas não pôde ser lido ({err}). "
            "Usando defaults APENAS nesta sessão — a escrita está bloqueada "
            "para preservar sua configuração no disco."
        )
        return _default_settings()

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"esperado objeto JSON, veio {type(data).__name__}")
    except Exception as ex:
        _mark_degraded("settings", f"conteúdo inválido: {ex}")
        logger.error(
            f"❌ {CONFIG_FILE.name} corrompido ({ex}). Usando defaults nesta "
            "sessão; a escrita está bloqueada para não perder o original."
        )
        return _default_settings()

    _mark_healthy("settings")
    return data


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
            # "auto" = Whisper detecta idioma por clip (recomendado p/ multi-canal).
            # Force "pt"/"en"/"es" se todos os canais falam o mesmo idioma.
            "language":  "auto",
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
