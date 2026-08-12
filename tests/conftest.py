"""
tests/conftest.py — Isolamento de ambiente para a suíte de regressão.

CRÍTICO: pending_store, feedback e metrics resolvem
    APP_DATA = Path(os.environ["APPDATA"]) / "ClipBot"
no MOMENTO DO IMPORT. Se APPDATA não for redirecionado ANTES desses imports,
os testes escrevem/apagam o estado REAL de produção do usuário
(pending.json, feedback_weights.json, channel_history.json).

pytest importa conftest.py antes de qualquer módulo de teste, então setar a
env var aqui, em tempo de import, é o único ponto seguro.

Regras da suíte:
  • Nenhum teste toca rede, GPU, FFmpeg, Whisper ou o disco do usuário.
  • Nenhum teste depende de ordem de execução.
  • Testes que falham hoje descrevem o COMPORTAMENTO DESEJADO (não o atual).
"""

import os
import sys
import tempfile
from pathlib import Path

# ── 1. Redireciona APPDATA ANTES de qualquer import do projeto ───────────────
_TMP_APPDATA = tempfile.mkdtemp(prefix="clipbot_tests_")
os.environ["APPDATA"] = _TMP_APPDATA

# ── 2. sys.path igual ao que app.py/bot_engine montam em runtime ─────────────
_ROOT = Path(__file__).resolve().parent.parent
for _p in (
    _ROOT,
    _ROOT / "core",
    _ROOT / "core" / "detection",
    _ROOT / "core" / "editing",
    _ROOT / "core" / "upload",
):
    sys.path.insert(0, str(_p))

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _assert_isolation():
    """Trava de segurança: aborta a suíte se APPDATA não estiver isolado."""
    import pending_store
    real_appdata = Path.home() / "AppData" / "Roaming" / "ClipBot"
    assert _TMP_APPDATA in str(pending_store.APP_DATA), (
        f"ISOLAMENTO FALHOU: store apontando para {pending_store.APP_DATA}"
    )
    assert Path(pending_store.APP_DATA) != real_appdata
    yield


@pytest.fixture
def store(tmp_path, monkeypatch):
    """
    PendingStore limpo por teste, com arquivo próprio em tmp_path.
    Redireciona PENDING_FILE/APP_DATA do módulo para não compartilhar estado.
    """
    import pending_store
    monkeypatch.setattr(pending_store, "APP_DATA", tmp_path)
    monkeypatch.setattr(pending_store, "PENDING_FILE", tmp_path / "pending.json")
    return pending_store.PendingStore()


@pytest.fixture
def raw_ts(tmp_path):
    """Cria um .ts falso (conteúdo irrelevante — nenhum teste decodifica)."""
    def _make(name="raw_pending_twitch_x_20260101_000000_keyword.ts"):
        p = tmp_path / name
        p.write_bytes(b"\x47" * 4096)   # sync byte de mpegts, só cosmético
        return p
    return _make


def make_moment(start=100.0, end=115.0, reason="keyword", score=0.5, keywords=None):
    """Helper: Moment real (não mock) para exercitar moment_id/hash."""
    from moment_detector import Moment
    return Moment(
        start_sec=start, end_sec=end, reason=reason,
        score=score, keywords=keywords or [],
    )


def detection_config(
    *,
    twitch_enabled=True,
    twitch_channels=("https://www.twitch.tv/baiano", "https://www.twitch.tv/kosky"),
    kick_enabled=False,
    audio_enabled=False,
    buffer_after=20,
    clip_max_sec=60,
):
    """
    Config equivalente ao que BotEngine._build_config produz.
    buffer_after=20 é o valor REAL derivado de clip_min_sec=15
    (max(20, 15 - 15//3)) — é ele que quebra o dedupe hoje.
    """
    return {
        "detection": {
            "audio_spike": {
                "enabled": audio_enabled, "threshold_db": 20,
                "min_clip_duration": 15, "max_clip_duration": clip_max_sec,
                "buffer_before": 10, "buffer_after": buffer_after,
            },
            "chat_flood": {
                "enabled": True, "messages_per_second": 10, "window_seconds": 3,
                "buffer_before": 10, "buffer_after": buffer_after,
            },
            "keywords": {
                "enabled": True, "words": ["kkkk", "omg", "clip"],
                "min_keyword_count": 1,
            },
            "manual_timestamps": {"enabled": False, "file": "manual_clips.txt"},
        },
        "streams": {
            "twitch": {"enabled": twitch_enabled, "channels": list(twitch_channels)},
            "youtube": {"enabled": False, "channels": []},
            "kick": {"enabled": kick_enabled, "channels": ["jonvlogs"]},
        },
    }
