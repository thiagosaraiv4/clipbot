"""
Item 3 — pico falso de áudio no início de cada captura.

Evidência de produção (APPDATA/ClipBot/metrics/*.jsonl, obs audio.spike_db):

    09/05   25.7,  26.5    ← picos reais (threshold = 20 dB)
    15/05  132.2           ← impossível
    12/08   71.6,  71.6    ← impossível, e IDÊNTICO nas duas capturas

Um salto de 132 dB não existe em áudio real: a faixa inteira entre silêncio
audível e clipping é ~60 dB. O valor só pode vir do piso epsilon — silêncio
digital (rms exatamente 0) dá 20*log10(0 + 1e-9) = -180.0 dB, que NÃO é uma
medição, é um artefato.

A aritmética fecha exatamente. Com o gate atual (len(baseline) < 5), a média
libera a comparação com 5 amostras; 4 chunks de silêncio digital e o primeiro
chunk real a -14,75 dB dão:

    baseline = (4*(-180) + (-14.75)) / 5 = -146.95
    spike    = -14.75 - (-146.95)        = +132.2   ← o valor registrado em 15/05

O efeito prático: um momento falso por captura. Com auto-restart por canal
(PR7), é um momento falso por reinício — cada um virando .ts no disco, entrada
na fila de aprovação e um clip de ~22s do nada (buffer_before 10 + after 20).

Contrato fixado aqui:
  • silêncio digital não é referência de loudness → fica fora do baseline;
  • nenhuma comparação antes de um aquecimento honesto (BASELINE_MIN);
  • o pico real de 25.7 dB continua disparando (não é uma regressão de
    sensibilidade disfarçada de correção).
"""

from collections import deque

import pytest

from conftest import detection_config


# Níveis de referência, em dBFS de rms (medidos com numpy, ver test_chunk_db_*)
SILENCE_DIGITAL = -180.0   # rms == 0.0 exato → piso epsilon
NOISE_FLOOR     = -80.0    # rms 1e-4: dither/ruído, inaudível
QUIET_REAL      = -55.0    # rms ~0.002: fala baixa, mas é sinal
NORMAL          = -26.0    # rms 0.05: fala normal
LOUD            = -14.75   # rms 0.18: stream alto (o chunk do caso de 15/05)
SCREAM          = -6.0     # rms 0.5: ~20 dB sobre NORMAL — EXATAMENTE o
                           # threshold, e a amostra atual se autodilui em
                           # ~0.4 dB, então fica logo abaixo. Para os casos
                           # "tem de disparar" use BIG_JUMP.
BIG_JUMP        = NORMAL + 30


def _detector(threshold_db=20):
    from moment_detector import AudioSpikeDetector
    cfg = detection_config(audio_enabled=True)
    cfg["detection"]["audio_spike"]["threshold_db"] = threshold_db
    return AudioSpikeDetector(cfg, lambda: 100.0)


def _feed(det, levels, baseline=None):
    """Passa uma sequência de níveis pelo avaliador. Retorna os picos aceitos."""
    b = baseline if baseline is not None else deque(maxlen=det.BASELINE_WIN)
    fired = [r for db in levels if (r := det._consider_chunk(db, b)) is not None]
    return fired, b


def _warm(det, level=NORMAL):
    """
    Baseline em regime permanente (janela cheia) — o estado normal em produção.

    Importa para a aritmética: _consider_chunk inclui a amostra atual no
    próprio baseline (semântica preservada da versão anterior), então o pico
    reportado é diluído em gap/(n+1). Com a janela cheia (60) isso é ~0.4 dB;
    com apenas BASELINE_MIN (15) seria ~1.6 dB.
    """
    _, b = _feed(det, [level] * det.BASELINE_WIN)
    return b


# ── Medição: o piso epsilon é real e o silêncio cai abaixo dele ──────────────

def test_chunk_db_of_digital_silence_is_unusable_as_reference():
    """rms 0 → -180 dB. É o número que envenena a média."""
    det = _detector()
    silence = b"\x00" * det.CHUNK_BYTES

    db = det._chunk_db(silence)

    assert db == pytest.approx(-180.0, abs=0.5)
    assert db <= det.SILENCE_DB, "silêncio digital tem de ficar abaixo do piso"


def test_chunk_db_of_normal_audio_is_above_the_floor():
    """Áudio de verdade — até o baixo — tem de contar como referência."""
    import numpy as np
    det = _detector()

    for rms, nome in [(0.002, "fala baixa"), (0.05, "fala normal"), (0.3, "grito")]:
        samples = np.full(det.CHUNK_BYTES // 4, rms, dtype=np.float32)
        db = det._chunk_db(samples.tobytes())
        assert db > det.SILENCE_DB, f"{nome} (rms={rms}) foi classificado como silêncio"


# ── O bug: silêncio inicial não pode gerar pico ──────────────────────────────

def test_reproduces_the_132db_regression():
    """
    Reprodução literal de 15/05: 4 chunks de silêncio digital seguidos do
    primeiro chunk real. Com o gate antigo (len>=5, média crua) isto produzia
    +132.2 dB — acima de qualquer threshold concebível.
    """
    det = _detector()

    fired, _ = _feed(det, [SILENCE_DIGITAL] * 4 + [LOUD])

    assert fired == [], f"pico fantasma no início da captura: {fired}"


def test_capture_start_with_long_silence_fires_nothing():
    """Caso geral: 10s de silêncio e depois 30s de áudio normal."""
    det = _detector()

    fired, _ = _feed(det, [SILENCE_DIGITAL] * 10 + [NORMAL] * 30)

    assert fired == [], f"{len(fired)} pico(s) falso(s) na entrada do áudio"


def test_silence_never_enters_the_baseline():
    """
    INVARIANTE central: -180 dB não é loudness, é ausência de medição.
    Se entrar no baseline, qualquer som posterior parece uma explosão.
    """
    det = _detector()

    _, b = _feed(det, [SILENCE_DIGITAL] * 20)

    assert len(b) == 0, "silêncio digital contaminou o baseline"


def test_noise_floor_also_stays_out():
    """Ruído inaudível (rms 1e-4) tem o mesmo efeito do silêncio na média."""
    det = _detector()

    _, b = _feed(det, [NOISE_FLOOR] * 20)

    assert len(b) == 0


def test_stream_that_is_silent_forever_never_fires():
    """Tela offline / stream mudo não pode gerar momentos."""
    det = _detector()

    fired, _ = _feed(det, [SILENCE_DIGITAL] * 300)

    assert fired == []


# ── Aquecimento: a comparação precisa de amostras suficientes ───────────────

def test_no_spike_before_warmup_completes():
    """
    Com 5 amostras, a média é o que as 5 amostras disserem. Mesmo sem silêncio,
    um degrau enorme dentro do aquecimento não é informação suficiente.
    """
    det = _detector()
    # BASELINE_MIN-2 válidas + o degrau = BASELINE_MIN-1 amostras: ainda frio.
    levels = [NORMAL] * (det.BASELINE_MIN - 2) + [NORMAL + 40]

    fired, _ = _feed(det, levels)

    assert fired == [], "disparou antes de ter baseline confiável"


def test_comparison_activates_exactly_at_warmup_size():
    """
    Fronteira documentada: a comparação começa quando a amostra atual completa
    BASELINE_MIN amostras válidas (ela conta a si mesma).
    """
    det = _detector()
    levels = [NORMAL] * (det.BASELINE_MIN - 1) + [NORMAL + 40]

    fired, b = _feed(det, levels)

    assert len(fired) == 1, "não ativou na fronteira do aquecimento"
    assert len(b) == det.BASELINE_MIN


def test_warmup_requires_valid_samples_not_just_chunks():
    """
    O aquecimento conta amostras VÁLIDAS. Silêncio não acelera o relógio:
    30 chunks de silêncio + 1 de áudio alto continuam sem baseline.
    """
    det = _detector()

    fired, b = _feed(det, [SILENCE_DIGITAL] * 30 + [SCREAM])

    assert fired == []
    assert len(b) == 1, "só o chunk válido deveria ter entrado"


# ── Não é uma queda de sensibilidade: o pico real continua disparando ───────

def test_real_spike_fires_after_warmup():
    """O caso de 09/05 (+25.7 dB) tem de continuar sendo detectado."""
    det = _detector(threshold_db=20)
    b = _warm(det, NORMAL)          # baseline ≈ -26 dB

    fired = det._consider_chunk(NORMAL + 25.7, b)

    assert fired is not None, "pico real deixou de ser detectado"
    assert fired["spike"] == pytest.approx(25.7, abs=1.0)


def test_spike_is_measured_against_normal_audio_not_silence():
    """
    Mesmo com silêncio no começo da captura, o baseline do pico seguinte tem de
    refletir o áudio normal — não o piso epsilon.
    """
    det = _detector()
    _, b = _feed(det, [SILENCE_DIGITAL] * 10 + [NORMAL] * det.BASELINE_WIN)

    fired = det._consider_chunk(BIG_JUMP, b)

    assert fired is not None
    assert fired["baseline"] == pytest.approx(NORMAL, abs=2.0), (
        f"baseline reportado {fired['baseline']} dB — silêncio ainda contamina"
    )


def test_threshold_is_respected():
    """Um degrau abaixo do threshold não é pico."""
    det = _detector(threshold_db=20)

    assert det._consider_chunk(NORMAL + 19.0, _warm(det, NORMAL)) is None
    assert det._consider_chunk(NORMAL + 21.0, _warm(det, NORMAL)) is not None


def test_threshold_boundary_accounts_for_self_dilution():
    """
    Comportamento preservado (não é novo): a amostra atual entra no próprio
    baseline, então um salto de exatamente threshold_db fica ~0.4 dB abaixo do
    corte com a janela cheia. Documentado para que ninguém "corrija" isso sem
    perceber que mexe na sensibilidade de todo o detector.
    """
    det = _detector(threshold_db=20)

    # Baseline novo por asserção: mesmo um chunk que NÃO dispara entra no
    # baseline e o eleva, então reusar o deque mediria outra coisa.
    assert det._consider_chunk(NORMAL + 20.0, _warm(det, NORMAL)) is None
    assert det._consider_chunk(NORMAL + 20.5, _warm(det, NORMAL)) is not None


def test_quiet_but_real_audio_builds_the_baseline():
    """Stream de sussurro é sinal: tem de aquecer o baseline normalmente."""
    det = _detector()

    _, b = _feed(det, [QUIET_REAL] * det.BASELINE_MIN)

    assert len(b) == det.BASELINE_MIN


# ── Contrato de dados e limites ─────────────────────────────────────────────

def test_spike_payload_keeps_the_sidecar_contract():
    """extra={db, baseline, spike} é consumido pelo sidecar e pela UI."""
    det = _detector()
    b = _warm(det, NORMAL)

    fired = det._consider_chunk(BIG_JUMP, b)

    assert set(fired) == {"db", "baseline", "spike"}
    assert fired["db"] == pytest.approx(BIG_JUMP, abs=0.1)


def test_baseline_stays_bounded():
    """Sessão de 24h não pode crescer memória no baseline."""
    det = _detector()

    _, b = _feed(det, [NORMAL] * 500)

    assert len(b) == det.BASELINE_WIN


def test_warmup_is_large_enough_to_be_meaningful():
    """
    Trava de projeto: o aquecimento antigo (5) era pequeno o bastante para que
    uma única amostra ruim movesse a média 20 dB — exatamente o threshold.
    """
    det = _detector()
    assert det.BASELINE_MIN >= 15, "aquecimento curto demais para uma média estável"
    assert det.BASELINE_MIN <= det.BASELINE_WIN
