"""
Whisper "small" → "medium" (um modelo só para o app inteiro).

Comparado em 4 clips reais (3 PT-BR do kosky, 1 EN do ishowspeed), com os
parâmetros do editor (auto-idioma, word_timestamps, beam_size=1), GTX 1660 S:

  • small: "superchandão", "esquiso por defender terraplano", "é zero texto"
    (era "zero testosterona"), "na foru" ("For You"), "Línguado?" (inventada
    onde a fala é "tá ligado?"); em EN, "course course course course are
    course are 1846 15".
  • medium acerta esses trechos; 1,8× mais lento (8,2 s x 4,6 s num clip de
    82 s, ainda >9× tempo real); VRAM +1058 MiB x +349 MiB.

O tamanho estava fixo em 3 pontos (legenda no video_editor; título e hook do
LLM no title_generator). Trocar só um deixaria os dois modelos na VRAM
(+1,4 GB) e transcreveria o clip duas vezes. Contrato: os três pedem o mesmo
modelo, o medium, e o processo carrega um só.
"""

from pathlib import Path

import pytest


@pytest.fixture
def whisper(monkeypatch):
    """WhisperService sem modelo de verdade; registra o tamanho pedido."""
    from whisper_service import WhisperService
    monkeypatch.setattr(WhisperService, "_instances", {})
    used = []

    def fake_transcribe(self, audio_path, **kw):
        used.append(self.size)
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "olha isso"}],
                "text": "olha isso", "language": "pt"}

    monkeypatch.setattr(WhisperService, "transcribe", fake_transcribe)
    return WhisperService, used


@pytest.fixture
def config():
    import bot_engine
    import credentials
    return bot_engine.BotEngine._build_config(credentials._default_settings(), {})


def _fake_audio(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"\x00" * 2048)
    return p


def test_subtitles_use_medium(whisper, config, tmp_path):
    from video_editor import VideoEditor
    _, used = whisper
    segs = VideoEditor(config, str(tmp_path))._transcribe(tmp_path / "clip.mp4", "auto")
    assert segs and used == ["medium"]


def test_whisper_title_uses_medium(whisper, config, tmp_path, monkeypatch):
    from title_generator import TitleGenerator
    _, used = whisper
    tg = TitleGenerator(config)
    tg.use_whisper = True
    monkeypatch.setattr(tg, "_extract_short_audio", lambda *a, **k: _fake_audio(tmp_path))
    tg._whisper_micro(tmp_path / "clip.ts")
    assert used == ["medium"]


def test_llm_hook_transcription_uses_medium(whisper, config, tmp_path, monkeypatch):
    import hook_scorer
    from title_generator import TitleGenerator
    _, used = whisper

    class FakeScorer:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def available_providers():
            return ["fake"]

        def __init__(self, language="pt"):
            pass

        def score(self, transcription, reason, **kw):
            return {"title": "OLHA ISSO", "provider": "fake"}

    monkeypatch.setattr(hook_scorer, "HookScorer", FakeScorer)
    raw = tmp_path / "raw_pending_twitch_x_20260101_000000_keyword.ts"
    raw.write_bytes(b"\x00" * 2048)
    tg = TitleGenerator(config)
    tg.use_whisper = True
    monkeypatch.setattr(tg, "_extract_short_audio", lambda *a, **k: _fake_audio(tmp_path))
    assert tg.generate(raw, {"reason": "keyword"}) == "OLHA ISSO"
    assert used == ["medium"]


def test_whole_pipeline_loads_a_single_model(whisper, config, tmp_path, monkeypatch):
    """Legenda + título no mesmo processo → um único singleton na VRAM."""
    from title_generator import TitleGenerator
    from video_editor import VideoEditor
    svc, used = whisper
    VideoEditor(config, str(tmp_path))._transcribe(tmp_path / "clip.mp4", "auto")
    tg = TitleGenerator(config)
    tg.use_whisper = True
    monkeypatch.setattr(tg, "_extract_short_audio", lambda *a, **k: _fake_audio(tmp_path))
    tg._whisper_micro(tmp_path / "clip.ts")
    assert set(used) == {"medium"}
    assert len(svc._instances) == 1
