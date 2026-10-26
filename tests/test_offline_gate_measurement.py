"""
Lote 1 / A4 — "não consegui medir o áudio" era tratado como "silêncio".

_audio_rms_and_cv devolvia (0.0, 1.0) em QUALQUER falha: ffmpeg com rc≠0,
timeout de 15s sob carga, clip sem trilha de áudio, menos de 2s de áudio ou
exceção. O Caminho A de is_offline_screen (silêncio absoluto, rms < 0.005) lia
esse 0.0 como silêncio e marcava OFFLINE — e os dois chamadores são
destrutivos:

    modo automático   → raw_clip.unlink()
    modo aprovação    → pending_store.finish_and_cleanup() (apaga o .ts)

Ou seja: o momento em que o usuário clicou "Gerar" sumia porque o ffmpeg
demorou. quality_gate tinha o mesmo defeito (rms 0.0 + motion 0.0 de duas
falhas = "silence+static").

Contrato: portão destrutivo só decide com base no que foi MEDIDO. Falha de
medição não descarta. Silêncio de verdade (medido) continua descartando.
"""

import numpy as np
import pytest


FRAME = 160 * 90
SR = 16000


class _R:
    """Resultado mínimo de run_ffmpeg."""
    def __init__(self, ok=True, stdout=b""):
        self.ok = ok
        self.stdout = stdout


def _moving_frames(n=16):
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, size=n * FRAME, dtype=np.uint8).tobytes()


def _static_frames(n=16):
    return (np.full(n * FRAME, 40, dtype=np.uint8)).tobytes()


def _speech(seconds=8.0):
    """Áudio com variação (fala): rms alto, cv alto."""
    rng = np.random.default_rng(2)
    env = np.repeat(rng.uniform(0.02, 0.4, int(seconds)), SR)
    return (rng.normal(0, 1, env.size) * env).astype(np.float32).tobytes()


def _flat_tone(seconds=8.0, amp=0.1):
    """Música em loop / countdown: rms uniforme, cv ≈ 0."""
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32).tobytes()


def _digital_silence(seconds=8.0):
    return np.zeros(int(seconds * SR), dtype=np.float32).tobytes()


@pytest.fixture
def ffmpeg(monkeypatch):
    """
    Substitui run_ffmpeg por respostas controladas por label.
    Valor pode ser _R, ou uma exceção (será levantada).
    """
    import clip_analyzer
    answers = {}

    def fake(cmd, label="", **kw):
        a = answers.get(label, _R(ok=False))
        if isinstance(a, BaseException):
            raise a
        return a

    monkeypatch.setattr(clip_analyzer, "run_ffmpeg", fake)
    return answers


def _offline(path="x.ts"):
    import clip_analyzer
    return clip_analyzer.is_offline_screen(path, duration_sec=8.0)


# ── O bug: falha de medição virava silêncio ─────────────────────────────────

def test_audio_timeout_is_not_silence(ffmpeg):
    """Reprodução do relatório: mesmo clip normal, só o áudio estourando."""
    ffmpeg["motion_score"] = _R(stdout=_moving_frames())
    ffmpeg["audio_rms_cv"] = _R(ok=False)                 # timeout / rc≠0

    is_off, info = _offline()

    assert is_off is False, f"clip descartado por falha de medição: {info}"


def test_missing_audio_track_is_not_silence(ffmpeg):
    ffmpeg["motion_score"] = _R(stdout=_moving_frames())
    ffmpeg["audio_rms_cv"] = _R(ok=True, stdout=b"")       # sem trilha de áudio

    assert _offline()[0] is False


def test_too_little_audio_is_not_silence(ffmpeg):
    ffmpeg["motion_score"] = _R(stdout=_moving_frames())
    ffmpeg["audio_rms_cv"] = _R(stdout=_speech(seconds=1.0))

    assert _offline()[0] is False


def test_exception_during_measurement_is_not_silence(ffmpeg):
    ffmpeg["motion_score"] = _R(stdout=_moving_frames())
    ffmpeg["audio_rms_cv"] = OSError("pipe quebrado")

    assert _offline()[0] is False


def test_info_says_measurement_failed(ffmpeg):
    """O motivo do 'não descartei' precisa aparecer no log."""
    ffmpeg["motion_score"] = _R(stdout=_moving_frames())
    ffmpeg["audio_rms_cv"] = _R(ok=False)

    _, info = _offline()

    assert "não medido" in info


# ── O que continua funcionando ──────────────────────────────────────────────

def test_measured_digital_silence_is_still_offline(ffmpeg):
    """Caminho A preservado: silêncio MEDIDO é o sinal forte que sempre foi."""
    ffmpeg["motion_score"] = _R(stdout=_moving_frames())
    ffmpeg["audio_rms_cv"] = _R(stdout=_digital_silence())

    assert _offline()[0] is True


def test_measured_silence_decides_even_without_motion(ffmpeg):
    """Caminho A nunca dependeu de movimento."""
    ffmpeg["motion_score"] = _R(ok=False)
    ffmpeg["audio_rms_cv"] = _R(stdout=_digital_silence())

    assert _offline()[0] is True


def test_path_b_still_detects_countdown(ffmpeg):
    """Caminho B preservado: imagem parada + áudio uniforme (countdown)."""
    ffmpeg["motion_score"] = _R(stdout=_static_frames())
    ffmpeg["audio_rms_cv"] = _R(stdout=_flat_tone())

    assert _offline()[0] is True


def test_path_b_needs_a_motion_measurement(ffmpeg):
    """Áudio uniforme com movimento NÃO medido não basta para descartar."""
    ffmpeg["motion_score"] = _R(ok=False)
    ffmpeg["audio_rms_cv"] = _R(stdout=_flat_tone())

    assert _offline()[0] is False


def test_normal_clip_is_not_offline(ffmpeg):
    ffmpeg["motion_score"] = _R(stdout=_moving_frames())
    ffmpeg["audio_rms_cv"] = _R(stdout=_speech())

    assert _offline()[0] is False


# ── quality_gate: mesmo defeito, mesmo contrato ─────────────────────────────

def test_quality_gate_keeps_clip_when_measurements_fail(ffmpeg):
    import clip_analyzer
    ffmpeg["audio_rms"] = _R(ok=False)
    ffmpeg["motion_score"] = _R(ok=False)

    passed, reason = clip_analyzer.quality_gate("x.ts")

    assert passed is True, f"descartado por falha de medição: {reason}"


def test_quality_gate_needs_both_measurements_to_reject(ffmpeg):
    import clip_analyzer
    ffmpeg["audio_rms"] = _R(stdout=_digital_silence(4.0))
    ffmpeg["motion_score"] = _R(ok=False)

    assert clip_analyzer.quality_gate("x.ts")[0] is True


def test_quality_gate_still_rejects_measured_silence_and_static(ffmpeg):
    import clip_analyzer
    ffmpeg["audio_rms"] = _R(stdout=_digital_silence(4.0))
    ffmpeg["motion_score"] = _R(stdout=_static_frames(8))

    passed, reason = clip_analyzer.quality_gate("x.ts")

    assert passed is False and "silence+static" in reason


# ── Contrato dos medidores usados no score (não pode mudar) ─────────────────

def test_scoring_helpers_keep_returning_float_on_failure(ffmpeg):
    """
    compute_reaction_score e find_hook_start fazem conta com o retorno: o
    contrato float (0.0 em falha) deles é preservado.
    """
    import clip_analyzer
    ffmpeg["audio_rms"] = _R(ok=False)
    ffmpeg["motion_score"] = _R(ok=False)

    assert clip_analyzer._extract_audio_rms("x.ts", 0, 4) == 0.0
    assert clip_analyzer._extract_motion_score("x.ts", 0, 4) == 0.0
