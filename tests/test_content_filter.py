"""
Lote 2 / B4 — o filtro de conteúdo casava por pedaço de palavra.

    matched = [kw for kw in blacklist if kw.lower() in full_text]

Com a blacklist padrão (bet, pix, stake, slots, odds…), isso descartava:
    "mano esse pixel art ficou lindo"   → 'pix'
    "minha vó tem diabetes"             → 'bet'
    "aprendi o alfabeto"                → 'bet'
    "foi um mistake feio"               → 'stake'
e process() devolve None — no modo automático o .ts é apagado, no modo
aprovação finish_and_cleanup apaga o momento que o usuário clicou "Gerar".

Contrato: casa PALAVRA ou FRASE inteira (inclusive com acento e espaços
repetidos do Whisper); o que é aposta de verdade continua sendo barrado.

Os testes rodam o _process_inner real até o ponto do filtro: comercial e
transcrição são substituídos, e o passo seguinte (gerar o .ass) levanta um
sentinela — "passou pelo filtro" e "foi barrado" ficam distinguíveis sem
ffmpeg nem Whisper.
"""

import pytest


class _PassedFilter(Exception):
    """O clip passou pelo filtro e seguiu para as legendas."""


@pytest.fixture
def editor(tmp_path, monkeypatch):
    import bot_engine
    import credentials
    from video_editor import VideoEditor

    config = bot_engine.BotEngine._build_config(credentials._default_settings(), {})
    ed = VideoEditor(config, str(tmp_path))
    monkeypatch.setattr(ed, "_detect_commercial_offset", lambda *a, **kw: 0.0)

    def stop_here(*a, **kw):
        raise _PassedFilter()
    monkeypatch.setattr(ed, "_segments_to_ass", stop_here)
    return ed


def _verdict(editor, monkeypatch, tmp_path, text):
    monkeypatch.setattr(editor, "_transcribe",
                        lambda *a, **kw: [{"start": 0.0, "end": 3.0, "text": text, "words": []}])
    try:
        out = editor._process_inner(tmp_path / "raw.ts", tmp_path / "out.mp4", title_text="X")
    except _PassedFilter:
        return "passou"
    return "barrado" if out is None else f"inesperado: {out}"


# ── O bug: pedaço de palavra ─────────────────────────────────────────────────

@pytest.mark.parametrize("fala", [
    "mano esse pixel art ficou lindo",
    "minha vó tem diabetes",
    "aprendi o alfabeto ontem",
    "foi um mistake feio",
    "que mapa betoneira",            # 'bet' em 'betoneira'
    "abetting the crime",            # 'betting' dentro de 'abetting'
])
def test_innocent_words_are_not_filtered(editor, monkeypatch, tmp_path, fala):
    assert _verdict(editor, monkeypatch, tmp_path, fala) == "passou", (
        f"'{fala}' foi descartado por conter pedaço de uma palavra da blacklist"
    )


# ── O que precisa continuar barrado ─────────────────────────────────────────

@pytest.mark.parametrize("fala", [
    "faz o pix pra mim agora",
    "aposta no tigrinho galera",
    "usa meu código promocional na descrição",
    "use my promo code bro",
    "BET agora no site",                    # caixa alta
    "depósito mínimo de dez reais",         # acento
    "código   promocional  na bio",         # espaços repetidos do Whisper
    "link na bio, pix!",                    # pontuação colada
])
def test_real_gambling_promo_is_still_filtered(editor, monkeypatch, tmp_path, fala):
    assert _verdict(editor, monkeypatch, tmp_path, fala) == "barrado", (
        f"'{fala}' deveria ter sido barrado"
    )


def test_filter_disabled_lets_everything_through(editor, monkeypatch, tmp_path):
    editor.cfg["content_filter"]["enabled"] = False
    assert _verdict(editor, monkeypatch, tmp_path, "faz o pix") == "passou"
