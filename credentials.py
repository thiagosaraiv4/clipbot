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
import threading
from pathlib import Path
from typing import Optional, Tuple

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
# O estado abaixo registra a ÚLTIMA leitura de cada arquivo, feita por
# qualquer thread — serve ao diagnóstico (logs, painel de saúde). A decisão de
# gravar não pode depender dele: uma leitura de outra thread mudaria a
# resposta para dados que vieram de outro lugar. Por isso:
#   • credentials → update_credentials decide pela leitura feita sob lock;
#   • settings    → ClipBotAPI.save_config decide pela leitura que gerou a
#                   cópia que a UI está devolvendo.

class StorageDegraded(RuntimeError):
    """A última leitura falhou; gravar agora sobrescreveria dados bons."""


_degraded: dict = {}      # "credentials" | "settings"  →  motivo


def _mark_degraded(kind: str, reason: str) -> None:
    _degraded[kind] = reason


def _mark_healthy(kind: str) -> None:
    if _degraded.pop(kind, None) is not None:
        logger.info(f"{kind}: leitura voltou a funcionar.")


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


#: Serializa read-modify-write de credentials.enc (ver update_credentials).
_cred_lock = threading.Lock()


def _read_credentials() -> Tuple[dict, Optional[str]]:
    """
    Lê credentials.enc e devolve (dados, problema DESTA leitura).

    problema None = leitura limpa (inclusive arquivo ausente). Atualiza o
    estado de saúde global, que serve ao diagnóstico — a decisão de gravar
    usa o problema devolvido aqui, não o global (ver update_credentials).
    """
    status, raw, err = safe_io.read_text_classified(CRED_FILE)

    if status == safe_io.MISSING:
        _mark_healthy("credentials")
        return {}, None

    if status != safe_io.OK:
        problem = err or status
        _mark_degraded("credentials", problem)
        logger.error(
            f"❌ {CRED_FILE.name} existe mas não pôde ser lido ({err}). "
            "Tokens e chaves de API indisponíveis nesta leitura — "
            "a escrita está bloqueada para não apagar o arquivo."
        )
        return {}, problem

    try:
        data = json.loads(_decrypt(raw))
        if not isinstance(data, dict):
            raise ValueError(f"esperado objeto JSON, veio {type(data).__name__}")
    except Exception as ex:
        problem = f"conteúdo inválido: {ex}"
        _mark_degraded("credentials", problem)
        logger.error(
            f"❌ {CRED_FILE.name} corrompido ou gerado em outra máquina ({ex}). "
            "A escrita está bloqueada; apague o arquivo para recomeçar."
        )
        return {}, problem

    _mark_healthy("credentials")
    return data, None


def save_credentials(creds: dict):
    """
    Sobrescreve credentials.enc inteiro (atômico, com retry).

    Para alterar algumas chaves, use update_credentials — ele decide pela
    leitura feita dentro do lock, sem janela de corrida.

    Raises:
        StorageDegraded: se a última leitura falhou (ver _guard_write).
        safe_io.StorageWriteError: se o disco recusou a escrita.
    """
    with _cred_lock:
        _guard_write("credentials", CRED_FILE)
        safe_io.atomic_write_text(CRED_FILE, _encrypt(json.dumps(creds)))


def update_credentials(updates: dict) -> dict:
    """
    Read-modify-write atômico: lê o disco AGORA, sob lock, mescla e grava.

    Substitui o padrão `load_credentials() → modificar → save_credentials()`,
    que tinha janela de corrida: entre um load degradado ({}) e o save, outra
    thread podia ler com sucesso e marcar o estado global como saudável — e o
    {} ia ao disco só com a chave nova, apagando as demais.

    Returns:
        O conteúdo gravado.
    Raises:
        StorageDegraded: se a leitura feita aqui falhou.
        safe_io.StorageWriteError: se o disco recusou a escrita.
    """
    with _cred_lock:
        current, problem = _read_credentials()
        if problem is not None:
            raise StorageDegraded(
                f"{CRED_FILE.name} não pôde ser lido agora ({problem}). "
                "Gravar apagaria as credenciais que continuam no disco. "
                "Verifique permissões/antivírus e tente de novo."
            )
        merged = {**current, **updates}
        safe_io.atomic_write_text(CRED_FILE, _encrypt(json.dumps(merged)))
        return merged


def load_credentials() -> dict:
    """
    Carrega e descriptografa credenciais.

    Devolve {} em qualquer falha — mas marca o estado como degradado e registra
    ERROR quando o arquivo existe e não pôde ser lido. "Não existe" e "não
    consegui ler" não são o mesmo evento.
    """
    return _read_credentials()[0]


def save_settings(settings: dict):
    """
    Salva configurações do app (atômico, com retry).

    Não consulta o estado global de leitura: a pergunta certa é se a CÓPIA que
    está sendo gravada veio de uma leitura boa, e só quem a obteve sabe. A UI
    recebe a config uma vez (on_loaded) e devolve o objeto inteiro — o guard
    dela está em ClipBotAPI.save_config, ligado à leitura que gerou a cópia.

    Raises:
        safe_io.StorageWriteError: se o disco recusou a escrita.
    """
    safe_io.atomic_write_text(
        CONFIG_FILE, json.dumps(settings, indent=2, ensure_ascii=False)
    )


def read_settings() -> Tuple[dict, Optional[str]]:
    """
    Lê settings.json e devolve (dados, problema DESTA leitura).

    Em falha devolve os defaults (o app precisa subir) junto com o motivo.
    Quem vai devolver esses dados ao disco (a UI) precisa guardar o motivo:
    gravar defaults trocaria os canais e tokens reais por uma config vazia.
    """
    status, raw, err = safe_io.read_text_classified(CONFIG_FILE)

    if status == safe_io.MISSING:
        _mark_healthy("settings")
        return _default_settings(), None

    if status != safe_io.OK:
        problem = err or status
        _mark_degraded("settings", problem)
        logger.error(
            f"❌ {CONFIG_FILE.name} existe mas não pôde ser lido ({err}). "
            "Usando os valores padrão nesta leitura; sua configuração no "
            "disco não foi alterada."
        )
        return _default_settings(), problem

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"esperado objeto JSON, veio {type(data).__name__}")
    except Exception as ex:
        problem = f"conteúdo inválido: {ex}"
        _mark_degraded("settings", problem)
        logger.error(
            f"❌ {CONFIG_FILE.name} corrompido ({ex}). Usando os valores padrão "
            "nesta leitura; o original no disco não foi alterado."
        )
        return _default_settings(), problem

    _mark_healthy("settings")
    return data, None


def load_settings() -> dict:
    """Carrega configurações do app (defaults em falha — ver read_settings)."""
    return read_settings()[0]


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
