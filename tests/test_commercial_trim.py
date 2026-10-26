"""
Lote 2 / B9 — o corte de comercial ignorava o resultado do ffmpeg.

    run_ffmpeg([... "-ss", offset, ... trimmed ...], label="commercial_pretrim", timeout=60)
    if trimmed.exists() and trimmed.stat().st_size > 1000:
        current = trimmed

O retorno era descartado. Num clip longo em 1080p60 com anúncio, sob carga,
o re-encode ultrafast passa de 60 s: o run_ffmpeg mata o processo, o .ts
PARCIAL existe e tem mais de 1 KB — e vira o clip. Medido na revisão: fonte de
40 s, offset 4,5 s → clip final de 25,8 s (esperado ~35,5 s), sem erro algum.
O recheck (commercial_retrim) tinha o mesmo defeito.

Contrato: o arquivo cortado só é usado se o ffmpeg terminou com sucesso; na
falha, vale o fallback que o código já previa para "o trim não gerou
arquivo" (usar o anterior e avisar).

O teste roda o _process_inner real até a transcrição e registra QUAL arquivo
ela recebeu — é ele que segue para legenda e render.
"""

import pytest


class _R:
    def __init__(self, ok=True):
        self.ok = ok
        self.returncode = 0 if ok else -9
        self.stdout = b""
        self.stderr_tail = [] if ok else ["timeout após 60s — processo morto"]


class _Reached(Exception):
    pass


@pytest.fixture
def editor(tmp_path, monkeypatch):
    import bot_engine
    import credentials
    from video_editor import VideoEditor

    config = bot_engine.BotEngine._build_config(credentials._default_settings(), {})
    out = tmp_path / "output"
    out.mkdir()
    ed = VideoEditor(config, str(out))
    src = tmp_path / "raw.ts"
    src.write_bytes(b"\x47" * 4096)
    ed._test_src = src

    seen = {}

    def transcribe(path, *a, **kw):
        seen["path"] = path
        raise _Reached()
    monkeypatch.setattr(ed, "_transcribe", transcribe)
    ed._seen = seen
    return ed


def _setup(editor, monkeypatch, *, pretrim_ok, retrim_ok=True, recheck=0.0):
    """Offset 4,5 s no original; `recheck` no arquivo já cortado."""
    import video_editor

    def offset(path):
        return 4.5 if path == editor._test_src else recheck
    monkeypatch.setattr(editor, "_detect_commercial_offset", offset)

    def fake(cmd, label="", **kw):
        ok = {"commercial_pretrim": pretrim_ok, "commercial_retrim": retrim_ok}.get(label, True)
        if label in ("commercial_pretrim", "commercial_retrim"):
            with open(cmd[-1], "wb") as f:          # parcial também tem > 1 KB
                f.write(b"\x47" * (900_000 if ok else 600_000))
        return _R(ok=ok)

    monkeypatch.setattr(video_editor, "run_ffmpeg", fake)


def _used(editor):
    with pytest.raises(_Reached):
        editor._process_inner(editor._test_src, editor.out_dir / "clip_x.mp4", title_text="X")
    return editor._seen["path"]


def test_failed_pretrim_does_not_become_the_clip(editor, monkeypatch):
    """O caso medido: timeout no pretrim, parcial com > 1 KB."""
    _setup(editor, monkeypatch, pretrim_ok=False)

    used = _used(editor)

    assert used == editor._test_src, f"clip seguiu com o pretrim parcial ({used.name})"


def test_failed_retrim_falls_back_to_the_first_trim(editor, monkeypatch):
    _setup(editor, monkeypatch, pretrim_ok=True, retrim_ok=False, recheck=2.0)

    assert _used(editor).name == "trimmed.ts"


# ── Comportamento preservado ────────────────────────────────────────────────

def test_successful_pretrim_is_used(editor, monkeypatch):
    _setup(editor, monkeypatch, pretrim_ok=True)

    assert _used(editor).name == "trimmed.ts"


def test_successful_retrim_is_used(editor, monkeypatch):
    _setup(editor, monkeypatch, pretrim_ok=True, retrim_ok=True, recheck=2.0)

    assert _used(editor).name == "retrimmed.ts"
