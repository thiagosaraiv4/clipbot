"""
Lote 2 / B6 — "%" apagava o banner de título; emoji virava caixa vazia.

1. O escape trocava "%" por "\\%". Dentro do filter_complex, o texto passa por
   duas camadas de escape antes de chegar ao drawtext, que trata "%" como
   início de expansão (%{...}). A barra é consumida no caminho, o drawtext
   reporta "Stray %" a cada quadro e não desenha NADA — com rc=0, então nada
   vai para o log. Títulos com "%" vêm do Whisper ("100%"), do LLM e do campo
   de título da UI.
   Medido com ffmpeg real: "50\\%" e "É 100\\% REAL" geram quadro idêntico ao
   vazio; com expansion=none, os dois são desenhados.

2. O drawtext não tem fallback de fonte: na Impact, "CADEIRA 😂" sai como
   "CADEIRA □". O prompt do título pede "1 emoji opcional no fim".

Contrato: o texto do banner é literal (expansion=none), sem caracteres que a
fonte não desenha; acentos continuam.
"""

import pytest


@pytest.fixture
def editor(tmp_path):
    import bot_engine
    import credentials
    from video_editor import VideoEditor
    config = bot_engine.BotEngine._build_config(credentials._default_settings(), {})
    return VideoEditor(config, str(tmp_path))


def _drawtexts(steps):
    """Cada invocação de drawtext no filtro, com seus parâmetros."""
    out = []
    for step in steps:
        for part in step.split("drawtext=")[1:]:
            out.append(part)
    return out


@pytest.mark.parametrize("title", ["É 100% REAL", "50% DE DESCONTO NA LIVE", "SÓ 1%"])
def test_every_banner_drawtext_is_literal(editor, title):
    """O que de fato resolveu no ffmpeg real: expansão desligada."""
    dts = _drawtexts(editor._build_title_filter(title))

    assert dts, "nenhum drawtext no filtro do título"
    assert all(":expansion=none" in d for d in dts), (
        "drawtext com expansão ligada: um % no título apaga o banner"
    )


def test_fallback_drawbox_banner_is_literal_too(editor, monkeypatch):
    """Sem o asset do blob, o banner cai na faixa drawbox — mesmo contrato."""
    monkeypatch.setattr(editor, "_blob_asset_path", lambda: None)

    dts = _drawtexts(editor._build_title_filter("É 100% REAL"))

    assert dts and all(":expansion=none" in d for d in dts)


@pytest.mark.parametrize("raw, esperado", [
    ("CAIU DA CADEIRA 😂", "CAIU DA CADEIRA"),
    ("🔥 INSANO 🔥", "INSANO"),
    ("VAI 👀👀", "VAI"),
    ("FAMÍLIA ❤️", "FAMÍLIA"),
])
def test_emoji_is_removed_from_banner_text(editor, raw, esperado):
    assert editor._escape_drawtext(raw) == esperado


def test_accents_and_punctuation_survive(editor):
    assert editor._escape_drawtext("É ÁGUA, NÃO!") == "É ÁGUA, NÃO!"


def test_colon_is_still_escaped(editor):
    """Escape pré-existente que continua necessário (separador de opções)."""
    assert "\\:" in editor._escape_drawtext("PLACAR 2:1")
