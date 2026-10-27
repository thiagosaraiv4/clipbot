"""
V3 — título do banner sem filtro, e cortado no meio da frase.

Validação em produção (26/10, 11 clips):

1. O título por Whisper é a primeira frase do clip, sem filtro nenhum:
     "AUGUSTIN CHESTER WHY DID YOU COMMIT SUICIDE IN 19-!"   (ishowspeed)
     "OREI ATÉ QUE QUANDO TÁ BOM TÁ RUIM VIADO!"             (baiano)
   O banner é o primeiro texto que a plataforma e o público leem;
   autoagressão, ofensa e violência sexual ali derrubam alcance ou a conta.
   O título do LLM sai da mesma fala e passa pelo mesmo risco.

2. O banner tem 2 linhas de 18 caracteres; o título pode ter até 50.
   _wrap_title descartava o resto sem aviso:
     "ENEM COM O ULT DO GALIO PRA DAR STUN!" → "ENEM COM O ULT DO / GALIO PRA DAR"
   (conferido no quadro do vídeo final).

Contrato: título com termo sensível não vai para o banner (cai para a
próxima estratégia, até o template); banner que corta palavras termina em "…".
"""

import random

import pytest


@pytest.fixture
def tg():
    import bot_engine
    import credentials
    from title_generator import TitleGenerator
    config = bot_engine.BotEngine._build_config(credentials._default_settings(), {})
    return TitleGenerator(config)


# ── 1. Termos sensíveis ──────────────────────────────────────────────────────

@pytest.mark.parametrize("title", [
    "AUGUSTIN CHESTER WHY DID YOU COMMIT SUICIDE IN 19-!",
    "OREI ATÉ QUE QUANDO TÁ BOM TÁ RUIM VIADO!",
    "ELE PENSOU EM SUICÍDIO!",
    "VOU ME MATAR!",
    "KILL YOURSELF!",
    "QUE RETARDADO!",
    "ESTUPRO NA LIVE?!",
])
def test_unsafe_titles_are_detected(tg, title):
    assert tg._unsafe_term(title)


@pytest.mark.parametrize("title", [
    "WHAT AM I DOING OUT HERE!",
    "ALL RIGHT JEFF!",
    "ENEM COM O ULT DO GALIO PRA DAR STUN!",
    "HUMILHADO POR DEFENDER TERRA PLANA 😱",
    "PASSOU DEBAIXO DO VIADUTO!",       # contém "viad"
    "PEDALADA INSANA!",                 # contém "peda"
    "VAI RETARDAR A JOGADA?",           # verbo, não ofensa
    "O CHAT PEDIU: OMG!",
])
def test_safe_titles_pass(tg, title):
    assert tg._unsafe_term(title) is None


def _entry():
    return {"reason": "keyword", "keywords": ["omg"], "extra": {}, "channel": "x"}


def _raw(tmp_path):
    raw = tmp_path / "raw_pending_twitch_x_20261026_225219_de72661b_keyword.ts"
    raw.write_bytes(b"\x00" * 2048)
    return raw


@pytest.mark.parametrize("spoken", [
    "AUGUSTIN CHESTER WHY DID YOU COMMIT SUICIDE IN 19-!",
    "OREI ATÉ QUE QUANDO TÁ BOM TÁ RUIM VIADO!",
])
def test_unsafe_whisper_title_falls_back_to_template(tg, tmp_path, monkeypatch, spoken):
    import hook_scorer
    monkeypatch.setattr(hook_scorer.HookScorer, "is_available", staticmethod(lambda: False))
    tg.use_whisper = True
    monkeypatch.setattr(tg, "_whisper_micro", lambda p: spoken)
    random.seed(0)
    title = tg.generate(_raw(tmp_path), _entry())
    assert title != spoken
    assert tg._unsafe_term(title) is None
    assert "OMG" in title          # template de keyword


def test_safe_whisper_title_is_kept(tg, tmp_path, monkeypatch):
    import hook_scorer
    monkeypatch.setattr(hook_scorer.HookScorer, "is_available", staticmethod(lambda: False))
    tg.use_whisper = True
    monkeypatch.setattr(tg, "_whisper_micro", lambda p: "WHAT AM I DOING OUT HERE!")
    assert tg.generate(_raw(tmp_path), _entry()) == "WHAT AM I DOING OUT HERE!"


def test_unsafe_llm_title_is_dropped_with_its_metadata(tg, tmp_path, monkeypatch):
    import hook_scorer

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
            return {"title": "ELE FALOU DE SUICÍDIO AO VIVO", "provider": "fake",
                    "hashtags": ["x"], "subtitle_emojis": {"a": "😱"}, "viral_score": 0.9}

    monkeypatch.setattr(hook_scorer, "HookScorer", FakeScorer)
    from whisper_service import WhisperService
    monkeypatch.setattr(WhisperService, "transcribe",
                        lambda self, *a, **k: {"text": "fala", "segments": []})
    tg.use_whisper = True
    monkeypatch.setattr(tg, "_extract_short_audio", lambda *a, **k: _raw(tmp_path))
    monkeypatch.setattr(tg, "_whisper_micro", lambda p: None)
    random.seed(0)
    title = tg.generate(_raw(tmp_path), _entry())
    assert tg._unsafe_term(title) is None
    assert tg.last_provider == "" and tg.last_hashtags == [] and tg.last_subtitle_emojis == {}


# ── 2. Banner não corta em silêncio ──────────────────────────────────────────

@pytest.mark.parametrize("title,expected", [
    ("ENEM COM O ULT DO GALIO PRA DAR STUN!", "ENEM COM O ULT DO\\nGALIO PRA DAR…"),
    ("OREI ATÉ QUE QUANDO TÁ BOM TÁ RUIM", "OREI ATÉ QUE\\nQUANDO TÁ BOM TÁ…"),
])
def test_banner_marks_dropped_words(title, expected):
    from video_editor import VideoEditor
    assert VideoEditor._wrap_title(title, max_chars=18, max_lines=2) == expected


@pytest.mark.parametrize("title", ["WHAT AM I DOING OUT HERE!", "O CHAT PEDIU: OMG!",
                                   "SPAM TOTAL + ELE DISSE 'OMG'!"])
def test_banner_that_fits_is_unchanged(title):
    from video_editor import VideoEditor
    out = VideoEditor._wrap_title(title, max_chars=18, max_lines=2)
    assert "…" not in out
    assert out.replace("\\n", " ") == title


def test_banner_lines_respect_the_width_with_the_ellipsis():
    from video_editor import VideoEditor
    t = "AAAAAAAAA BBBBBBBB CCCCCCCCC DDDDDDDD EEEE"
    lines = VideoEditor._wrap_title(t, max_chars=18, max_lines=2).split("\\n")
    assert len(lines) == 2 and lines[-1].endswith("…")
    assert all(len(l) <= 18 for l in lines)
