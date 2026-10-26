"""
Lote 2 / B7 — extract_segment aceitava recorte vazio como sucesso.

Só o código de saída era checado. Com `-ss` além do fim do arquivo, o ffmpeg
sai com rc=0 e grava um .ts só com o cabeçalho (medido na revisão: 2.820
bytes, 0,021 s). Quando a live cai e o chat continua ("caiu", "kkkk"), os
momentos de chat são liberados pelo relógio antes do restart do watchdog — e
a fila de aprovação recebia lixo que o usuário aprovava, ou o modo automático
renderizava e postava.

Contrato: recorte com menos de max(1 s, 50% do pedido) é erro (e o arquivo
inútil é removido); recorte parcial acima disso passa com aviso; sem ffprobe,
um piso de tamanho pega o .ts só-cabeçalho.

Sem ffmpeg real (regra da suíte): run_ffmpeg é substituído pelo que o ffmpeg
devolve em cada caso.
"""

import logging

import pytest


class _R:
    def __init__(self, ok=True, stdout=b"", returncode=0):
        self.ok = ok
        self.stdout = stdout
        self.returncode = returncode
        self.stderr_tail = []


@pytest.fixture
def capture(tmp_path):
    from stream_capture import StreamCapture
    return StreamCapture({"streams": {}}, str(tmp_path))


def _fake_ffmpeg(monkeypatch, *, out_bytes, probed=None, extract_ok=True):
    """
    extract_segment → grava `out_bytes` bytes e devolve rc=0 (ou falha);
    ffprobe → devolve `probed` segundos (None = ffprobe indisponível).
    """
    import stream_capture

    def fake(cmd, label="", **kw):
        if cmd[0] == "ffprobe":
            if probed is None:
                return _R(ok=False, returncode=-127)
            return _R(stdout=f"{probed}\n".encode())
        out = cmd[-1]
        with open(out, "wb") as f:
            f.write(b"\x47" * out_bytes)
        return _R(ok=extract_ok, returncode=0 if extract_ok else 1)

    monkeypatch.setattr(stream_capture, "run_ffmpeg", fake)


# ── O bug ────────────────────────────────────────────────────────────────────

def test_window_past_the_end_is_an_error(capture, tmp_path, monkeypatch):
    """O caso medido: só cabeçalho, 0,021 s de 30 s pedidos."""
    _fake_ffmpeg(monkeypatch, out_bytes=2820, probed=0.021)
    out = tmp_path / "raw_pending_x.ts"

    with pytest.raises(RuntimeError):
        capture.extract_segment(tmp_path / "src.ts", 80.0, 110.0, out)

    assert not out.exists(), "o .ts inútil ficou no disco (vira órfão)"


def test_less_than_half_of_the_window_is_an_error(capture, tmp_path, monkeypatch):
    _fake_ffmpeg(monkeypatch, out_bytes=500_000, probed=5.0)

    with pytest.raises(RuntimeError):
        capture.extract_segment(tmp_path / "src.ts", 15.0, 35.0, tmp_path / "o.ts")


def test_under_one_second_is_an_error_even_for_short_windows(capture, tmp_path, monkeypatch):
    _fake_ffmpeg(monkeypatch, out_bytes=100_000, probed=0.9)

    with pytest.raises(RuntimeError):
        capture.extract_segment(tmp_path / "src.ts", 10.0, 11.5, tmp_path / "o.ts")


def test_without_ffprobe_a_header_only_file_is_still_caught(capture, tmp_path, monkeypatch):
    _fake_ffmpeg(monkeypatch, out_bytes=2820, probed=None)

    with pytest.raises(RuntimeError):
        capture.extract_segment(tmp_path / "src.ts", 80.0, 110.0, tmp_path / "o.ts")


# ── O que continua passando ──────────────────────────────────────────────────

def test_full_window_is_returned(capture, tmp_path, monkeypatch):
    _fake_ffmpeg(monkeypatch, out_bytes=3_000_000, probed=30.4)   # keyframe anterior
    out = tmp_path / "o.ts"

    assert capture.extract_segment(tmp_path / "src.ts", 60.0, 90.0, out) == out
    assert out.exists()


def test_mostly_available_window_passes_with_a_warning(capture, tmp_path, monkeypatch, caplog):
    """Live caiu perto do fim da janela: 15 de 17 s ainda é um clip."""
    _fake_ffmpeg(monkeypatch, out_bytes=1_500_000, probed=15.0)

    with caplog.at_level(logging.WARNING):
        capture.extract_segment(tmp_path / "src.ts", 5.0, 22.0, tmp_path / "o.ts")

    assert any("parcial" in r.getMessage() for r in caplog.records)


def test_without_ffprobe_a_real_sized_file_passes(capture, tmp_path, monkeypatch):
    _fake_ffmpeg(monkeypatch, out_bytes=2_000_000, probed=None)

    out = tmp_path / "o.ts"
    assert capture.extract_segment(tmp_path / "src.ts", 60.0, 90.0, out) == out


def test_ffmpeg_failure_still_raises(capture, tmp_path, monkeypatch):
    """Comportamento pré-existente."""
    _fake_ffmpeg(monkeypatch, out_bytes=0, probed=None, extract_ok=False)

    with pytest.raises(RuntimeError):
        capture.extract_segment(tmp_path / "src.ts", 60.0, 90.0, tmp_path / "o.ts")
