"""
V6 — a gravação contínua de cada canal nunca era apagada durante a sessão.

Validação em produção (26/10): disco livre 630,9 → 619,6 GB em 35 min
(-11,3 GB). Cada canal grava a live inteira num raw_<plataforma>_*.ts
(~3,7 GB/h em 1080p60). Quando a captura acaba — live encerrada, FROZEN e
reinício, stop do engine — o arquivo fica no disco até a limpeza do boot,
7 dias depois. O link de YouTube que travava a cada ~5 min deixou 6 cópias
de 661 MB em 30 min; o ishowspeed, kick/jonvlogs e kick/fontinnelerj
encerraram a live e deixaram 0,7 + 2,2 + 2,2 GB.

Depois que a captura para, nada mais lê esse arquivo: o detector já parou,
e os momentos pendentes viveram em raw_pending_* próprios (extract_segment).

Contrato: o engine descarta a gravação contínua ao fim de cada attempt
(reinício ou stop); os raw_pending_* ficam. No Windows o ffmpeg recém-morto
ainda segura o handle por instantes → tenta de novo; se não conseguir, não
quebra nada e a limpeza do boot recolhe.
"""

import time
from types import SimpleNamespace

import pytest

from test_channel_resilience import _fakes, _install, _run, engine  # noqa: F401


@pytest.fixture
def capture(tmp_path, monkeypatch):
    import stream_capture
    monkeypatch.setattr(stream_capture, "_DISCARD_BACKOFF_SEC", 0.0)
    return stream_capture.StreamCapture({"streams": {}}, str(tmp_path))


def _recording(capture, tmp_path):
    """Captura 'em andamento' sem ffmpeg: o arquivo e uma sessão falsa."""
    f = tmp_path / "raw_twitch_kosky_20261026_220937835219.ts"
    f.write_bytes(b"\x47" * 188 * 100)
    closed = []
    capture.current_session = SimpleNamespace(close=lambda grace=5.0: closed.append(grace))
    capture.current_process = None
    capture.current_file = f
    capture.start_time = time.time()
    capture.is_recording = True
    return f, closed


# ── StreamCapture ────────────────────────────────────────────────────────────

def test_discard_deletes_the_continuous_recording(capture, tmp_path):
    f, closed = _recording(capture, tmp_path)
    pending = tmp_path / "raw_pending_twitch_kosky_20261026_221203_424b9854_keyword.ts"
    pending.write_bytes(b"\x47" * 188)

    assert capture.stop_recording(discard=True) == f
    assert closed, "parou sem fechar o ffmpeg"
    assert not f.exists(), "gravação contínua ficou no disco"
    assert pending.exists(), "segmento pendente não é da captura — não pode sumir"


def test_default_stop_keeps_the_file(capture, tmp_path):
    """core/main.py (legado) usa o arquivo depois do stop."""
    f, _ = _recording(capture, tmp_path)
    capture.stop_recording()
    assert f.exists()


def test_discard_retries_while_windows_holds_the_handle(capture, tmp_path, monkeypatch):
    from pathlib import Path
    f, _ = _recording(capture, tmp_path)
    real_unlink = Path.unlink
    calls = {"n": 0}

    def busy_twice(self, *a, **k):
        if self == f and calls["n"] < 2:
            calls["n"] += 1
            raise PermissionError(32, "O arquivo já está sendo usado por outro processo")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", busy_twice)
    capture.stop_recording(discard=True)
    assert calls["n"] == 2 and not f.exists()


def test_discard_that_never_succeeds_does_not_break_the_stop(capture, tmp_path, monkeypatch, caplog):
    from pathlib import Path
    f, _ = _recording(capture, tmp_path)

    def always_busy(self, *a, **k):
        raise PermissionError(32, "em uso")

    monkeypatch.setattr(Path, "unlink", always_busy)
    with caplog.at_level("WARNING", logger="ClipBot"):
        assert capture.stop_recording(discard=True) == f
    assert not capture.is_recording
    assert any("não pôde ser apagada" in r.getMessage() for r in caplog.records)


def test_discard_without_recording_is_a_noop(capture):
    assert capture.stop_recording(discard=True) is None


# ── Engine: fim de cada attempt ──────────────────────────────────────────────

def test_restart_discards_the_previous_recording(engine, tmp_path, monkeypatch):  # noqa: F811
    cap, det, st = _fakes(tmp_path, tick_crashes=1)
    _install(monkeypatch, cap, det)

    _run(engine, tmp_path, until=lambda: len(st["detectors"]) >= 2)

    first = st["captures"][0]
    assert first.stopped and first.discarded, "attempt reiniciado deixou a gravação no disco"


def test_engine_stop_discards_the_recording(engine, tmp_path, monkeypatch):  # noqa: F811
    cap, det, st = _fakes(tmp_path)
    _install(monkeypatch, cap, det)

    _run(engine, tmp_path, until=lambda: st["detectors"] and st["detectors"][0].calls >= 2)

    last = st["captures"][-1]
    assert last.stopped and last.discarded, "stop do engine deixou a gravação no disco"
