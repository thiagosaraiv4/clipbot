"""
Lote 2 / B8 — render que falhava deixava um clip_*.mp4 corrompido na saída.

O ffmpeg final escrevia direto no caminho definitivo (output_folder/clip_*.mp4)
e, em timeout de 300 s, encoder morrendo no meio ou disco cheio, o
_process_inner levantava RuntimeError sem apagar nada. O mp4 parcial — sem o
átomo moov, ilegível — aparecia na aba de clips (get_clips lista clip_*.mp4),
podia ser enviado pelo upload manual, e o GC só o recolhia em 7 dias.

Contrato: o caminho final só existe quando o render terminou com sucesso.

Sem ffmpeg real (regra da suíte): run_ffmpeg do editor é substituído, e o
render falso escreve no caminho que o comando pediu — como o ffmpeg faz.
"""

import pytest


class _R:
    def __init__(self, ok=True, tail=()):
        self.ok = ok
        self.returncode = 0 if ok else 1
        self.stdout = b""
        self.stderr_tail = list(tail)


@pytest.fixture
def editor(tmp_path, monkeypatch):
    import bot_engine
    import credentials
    import face_tracker
    from video_editor import VideoEditor

    config = bot_engine.BotEngine._build_config(credentials._default_settings(), {})
    out = tmp_path / "output"
    out.mkdir()
    ed = VideoEditor(config, str(out))
    monkeypatch.setattr(ed, "_detect_commercial_offset", lambda *a, **kw: 0.0)
    monkeypatch.setattr(ed, "_transcribe", lambda *a, **kw: [])
    monkeypatch.setattr(face_tracker, "compute_trajectory", lambda *a, **kw: None)
    monkeypatch.setattr(VideoEditor, "_use_nvenc", classmethod(lambda cls: False))
    monkeypatch.setattr(VideoEditor, "_nvenc_runtime_disabled", False)
    src = tmp_path / "raw.ts"
    src.write_bytes(b"\x47" * 4096)
    ed._test_src = src
    return ed


def _fake_render(monkeypatch, outcomes):
    """
    outcomes: lista de (ok, stderr) por chamada de render_final, em ordem.
    Cada render grava bytes no caminho de saída do comando (último argumento),
    inclusive quando falha — o ffmpeg deixa o parcial.
    """
    import video_editor
    calls = []

    def fake(cmd, label="", **kw):
        if label != "render_final":
            return _R()
        ok, tail = outcomes[len(calls)]
        calls.append(cmd[-1])
        with open(cmd[-1], "wb") as f:
            f.write(b"\x00" * (2_000_000 if ok else 1_048_624))
        return _R(ok=ok, tail=tail)

    monkeypatch.setattr(video_editor, "run_ffmpeg", fake)
    return calls


def _clips(editor):
    return sorted(p.name for p in editor.out_dir.glob("clip_*.mp4"))


def _run(editor, name="clip_twitch_kosky_20261026_153045_ab12cd34_keyword"):
    out = editor.out_dir / f"{name}.mp4"
    return editor._process_inner(editor._test_src, out, title_text="X"), out


def test_failed_render_leaves_no_clip_in_the_output_folder(editor, monkeypatch):
    _fake_render(monkeypatch, [(False, ["Conversion failed!"])])

    with pytest.raises(RuntimeError):
        _run(editor)

    assert _clips(editor) == [], f"mp4 parcial ficou na saída: {_clips(editor)}"


def test_successful_render_lands_at_the_final_path(editor, monkeypatch):
    _fake_render(monkeypatch, [(True, [])])

    result, out = _run(editor)

    assert result == out
    assert out.exists() and out.stat().st_size == 2_000_000
    assert [p.name for p in editor.out_dir.iterdir()] == [out.name], "sobrou arquivo temporário"


def test_render_does_not_write_to_the_final_path_while_running(editor, monkeypatch):
    """O ffmpeg escreve num temporário; o caminho final só aparece no fim."""
    calls = _fake_render(monkeypatch, [(True, [])])

    _, out = _run(editor)

    assert calls and calls[0] != str(out)


def test_nvenc_fallback_success_still_lands_at_the_final_path(editor, monkeypatch):
    from video_editor import VideoEditor
    monkeypatch.setattr(VideoEditor, "_use_nvenc", classmethod(lambda cls: True))
    _fake_render(monkeypatch, [(False, ["nvenc: no capable devices found"]), (True, [])])

    result, out = _run(editor)

    assert out.exists() and out.stat().st_size == 2_000_000


def test_nvenc_fallback_failure_leaves_nothing(editor, monkeypatch):
    from video_editor import VideoEditor
    monkeypatch.setattr(VideoEditor, "_use_nvenc", classmethod(lambda cls: True))
    _fake_render(monkeypatch, [(False, ["nvenc: no capable devices found"]),
                               (False, ["Conversion failed!"])])

    with pytest.raises(RuntimeError):
        _run(editor)

    assert _clips(editor) == []
