"""
V2 — "{kw}" literal no banner do clip.

Validação em produção (26/10): 2 de 6 clips saíram com
  "O CHAT EXPLODIU + ELE DISSE '{KW}'!"  e  "O CHAT EXPLODIU + '{kw}' NA LIVE!"
desenhados no banner (sidecar do clip: keywords=["omg"]).

_template_title só trocava {kw} quando reason == "keyword". O super momento
['chat_flood', 'keyword'] combina um template de cada reason e o de keyword
traz {kw} — nunca preenchido. O mesmo vale para reason == "keyword" sem
keywords: random.choice cai nos templates com {kw}.

Contrato: nenhum título de template contém "{kw}"; com keyword disponível,
ela aparece no lugar.
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


def _all_titles(tg, reason, keywords, extra, n=300):
    out = set()
    for seed in range(n):
        random.seed(seed)
        out.add(tg._template_title(reason, keywords, extra))
    return out


@pytest.mark.parametrize("reasons", [["chat_flood", "keyword"], ["keyword", "chat_flood"],
                                     ["keyword", "audio_spike"]])
def test_super_moment_combo_fills_the_keyword(tg, reasons):
    titles = _all_titles(tg, "super_moment", ["omg"], {"reasons": reasons})
    assert not [t for t in titles if "{kw}" in t.lower()], titles
    # Os templates de keyword que entraram no combo trazem a palavra.
    assert any("OMG" in t for t in titles), titles


@pytest.mark.parametrize("reason,extra", [
    ("keyword", {}),
    ("super_moment", {"reasons": ["chat_flood", "keyword"]}),
    ("super_moment", {"reasons": ["keyword", "chat_flood"]}),
])
def test_no_placeholder_without_keywords(tg, reason, extra):
    titles = _all_titles(tg, reason, [], extra)
    assert not [t for t in titles if "{kw}" in t.lower()], titles


def test_keyword_reason_still_uses_the_keyword(tg):
    titles = _all_titles(tg, "keyword", ["omg"], {})
    assert titles and all("OMG" in t for t in titles), titles


def test_generate_super_moment_from_production(tg, tmp_path):
    """O momento real do clip 73f9cb6a, com LLM e Whisper fora do caminho."""
    tg.use_whisper = False
    raw = tmp_path / "raw_pending_twitch_ishowspeed_20261026_221725_73f9cb6a_super_moment.ts"
    entry = {"reason": "super_moment", "keywords": ["omg"],
             "extra": {"reasons": ["chat_flood", "keyword"]}, "channel": "ishowspeed"}
    for seed in range(100):
        random.seed(seed)
        title = tg.generate(raw, entry)
        assert "{kw}" not in title.lower(), title
