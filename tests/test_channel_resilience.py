"""
Lote 2 / B1 — exceção inesperada matava a thread do canal, sem restart.

_monitor_platform tem um retry externo (backoff 30→600 s) e um loop de ticks
interno. Mas o `try` do loop interno só tratava KeyboardInterrupt: qualquer
outra exceção atravessava o `finally`, saía do retry externo e encerrava a
thread. O log dizia "✅ Encerrado." como num stop normal, o traceback ia para
um stderr que não existe no build windowed, e o painel de saúde continuava
mostrando "capturing".

Portas de saída encontradas:
  • tick: `chat_rate` iterava o deque de mensagens com um gerador enquanto a
    thread do IRC fazia append/popleft → "deque mutated during iteration".
    No modo automático isso era lido fora do try por momento, durante flood.
  • criação do detector: MomentDetector(...) e detector.start() ficavam FORA
    do try — se falhassem, a thread morria e o ffmpeg da captura não parava.
  • fora do loop interno (construtor do StreamCapture, backoff): idem.

Com vários canais, um morre em silêncio. Com um só, o engine para.

Contrato: erro inesperado reinicia o canal (com backoff), limpa o attempt e
fica visível; erro no processamento de UM momento pula aquele momento, não o
canal.
"""

import logging
import sys
import threading
import time

import pytest

from conftest import make_moment


LABEL = "TWITCH:kosky"
CHANNEL = "https://www.twitch.tv/kosky"


def _cfg(approval=True):
    return {
        "streams": {"twitch": {"enabled": True, "channels": [CHANNEL]}},
        "system": {"approval_mode": approval, "approval_top_n": 12},
        "detection": {"audio_spike": {"min_clip_duration": 15}},
        "editing": {"clip_analysis": {"enabled": False}},
    }


def _fakes(tmp_path, *, capture_init_fails=0, detector_start_fails=0,
           tick_crashes=0, moments=()):
    """Captura e detector falsos; contadores compartilhados em `state`."""
    state = {"captures": [], "detectors": [], "init_calls": 0}

    class FakeCapture:
        def __init__(self, cfg, clips_dir):
            state["init_calls"] += 1
            if state["init_calls"] <= capture_init_fails:
                raise OSError("streamlink não iniciou")
            self.stopped = False
            state["captures"].append(self)

        def start_recording(self, platform):
            p = tmp_path / f"raw_twitch_kosky_{len(state['captures'])}.ts"
            p.write_bytes(b"\x47" * 4096)
            return p

        def get_elapsed_seconds(self):
            return 10_000.0

        def stop_recording(self):
            self.stopped = True

    class FakeDetector:
        def __init__(self, cfg, elapsed_fn, platform=None, channel=None):
            self.n = len(state["detectors"])
            self.calls = 0
            self.stopped = False
            self.chat = None
            state["detectors"].append(self)

        def start(self, raw):
            if self.n < detector_start_fails:
                raise OSError("detector não subiu")

        def stop(self):
            self.stopped = True

        # Interface de identidade/liberação do MomentDetector real (B3).
        @staticmethod
        def members(m):
            return (m.moment_id,)

        def is_settled(self, m, elapsed):
            return True

        def consume(self, m):
            pass

        @property
        def all_moments(self):
            self.calls += 1
            if self.n < tick_crashes:
                raise RuntimeError("deque mutated during iteration")
            return list(moments)

    return FakeCapture, FakeDetector, state


@pytest.fixture
def engine(store, monkeypatch):
    import bot_engine
    eng = bot_engine.BotEngine(pending_store=store)
    # Tempos curtos: o comportamento é o mesmo, só mais rápido.
    eng._TICK_SEC = 0.02
    eng._RETRY_BACKOFF_INIT = 0.05
    eng._RETRY_BACKOFF_MAX = 0.2
    eng._CRASH_BACKOFF_INIT = 0.05
    eng._CRASH_BACKOFF_MAX = 0.2
    return eng


def _install(monkeypatch, fake_capture, fake_detector):
    import moment_detector
    import stream_capture
    monkeypatch.setattr(stream_capture, "StreamCapture", fake_capture)
    monkeypatch.setattr(moment_detector, "MomentDetector", fake_detector)


def _run(engine, tmp_path, until, cfg=None, timeout=6.0):
    """Roda o monitor numa thread até `until()` ou timeout; depois para."""
    t = threading.Thread(
        target=engine._monitor_platform,
        args=("twitch", CHANNEL, tmp_path, None, None, None, None),
        kwargs={"config": cfg or _cfg()},
        daemon=True,
    )
    t.start()
    deadline = time.time() + timeout
    while time.time() < deadline and not until():
        time.sleep(0.02)
    reached = until()
    alive = t.is_alive()
    engine._stop_event.set()
    t.join(timeout=5)
    return reached, alive, t


# ── O bug: exceção no tick ───────────────────────────────────────────────────

def test_tick_exception_restarts_the_channel(engine, tmp_path, monkeypatch):
    cap, det, st = _fakes(tmp_path, tick_crashes=1)
    _install(monkeypatch, cap, det)

    reached, alive, t = _run(
        engine, tmp_path,
        until=lambda: len(st["detectors"]) >= 2 and st["detectors"][1].calls >= 1,
    )

    assert reached, "o canal não reiniciou depois da exceção — thread morreu"
    assert alive
    assert not t.is_alive(), "stop do engine não encerrou a thread"


def test_crashed_attempt_is_cleaned_up(engine, tmp_path, monkeypatch):
    """O attempt que falhou para captura e detector antes de reiniciar."""
    cap, det, st = _fakes(tmp_path, tick_crashes=1)
    _install(monkeypatch, cap, det)

    _run(engine, tmp_path, until=lambda: len(st["detectors"]) >= 2)

    assert st["captures"][0].stopped, "ffmpeg da captura ficou rodando"
    assert st["detectors"][0].stopped


def test_tick_exception_is_logged_with_traceback(engine, tmp_path, monkeypatch, caplog):
    cap, det, st = _fakes(tmp_path, tick_crashes=1)
    _install(monkeypatch, cap, det)

    with caplog.at_level(logging.ERROR, logger="ClipBot"):
        _run(engine, tmp_path, until=lambda: len(st["detectors"]) >= 2)

    crash = [r for r in caplog.records
             if r.levelno >= logging.ERROR and r.exc_info
             and "deque mutated" in str(r.exc_info[1])]
    assert crash, "a exceção não chegou ao log com traceback"


def test_tick_exception_counts_as_error(engine, tmp_path, monkeypatch):
    cap, det, st = _fakes(tmp_path, tick_crashes=1)
    _install(monkeypatch, cap, det)

    _run(engine, tmp_path, until=lambda: len(st["detectors"]) >= 2)

    assert engine.get_health()[LABEL]["funnel"]["errors"] >= 1


# ── O bug: detector e captura fora do try ───────────────────────────────────

def test_detector_start_failure_stops_capture_and_restarts(engine, tmp_path, monkeypatch):
    cap, det, st = _fakes(tmp_path, detector_start_fails=1)
    _install(monkeypatch, cap, det)

    reached, alive, _ = _run(
        engine, tmp_path, until=lambda: len(st["detectors"]) >= 2
        and st["detectors"][1].calls >= 1,
    )

    assert reached and alive, "falha no detector matou a thread"
    assert st["captures"][0].stopped, "captura do attempt falho não foi parada"


def test_capture_constructor_failure_does_not_kill_thread(engine, tmp_path, monkeypatch):
    """Rede de segurança para o que fica fora do loop interno."""
    cap, det, st = _fakes(tmp_path, capture_init_fails=1)
    _install(monkeypatch, cap, det)

    reached, alive, _ = _run(
        engine, tmp_path, until=lambda: st["detectors"] and st["detectors"][0].calls >= 1,
    )

    assert reached and alive, "erro fora do loop interno matou a thread"


# ── Erro de UM momento não derruba o canal ──────────────────────────────────

def test_scoring_error_skips_the_moment_not_the_channel(engine, tmp_path, monkeypatch):
    """
    Modo automático: o score (que lê chat_rate_ratio) ficava fora do try por
    momento. Erro ali agora pula o momento; o canal segue no mesmo attempt.
    """
    import feedback
    cap, det, st = _fakes(tmp_path, moments=[make_moment(start=100.0, end=115.0)])
    _install(monkeypatch, cap, det)

    def boom(*a, **kw):
        raise RuntimeError("deque mutated during iteration")
    monkeypatch.setattr(feedback, "score_moment_advanced", boom)

    reached, alive, _ = _run(
        engine, tmp_path, cfg=_cfg(approval=False),
        until=lambda: engine.get_health().get(LABEL, {}).get("funnel", {}).get("errors", 0) >= 1
        and st["detectors"][0].calls >= 3,
    )

    assert reached and alive
    assert len(st["captures"]) == 1, "erro de um momento reiniciou a captura"


# ── Invariante: stop normal continua sendo stop ─────────────────────────────

def test_clean_stop_does_not_restart(engine, tmp_path, monkeypatch):
    cap, det, st = _fakes(tmp_path)
    _install(monkeypatch, cap, det)

    reached, alive, t = _run(
        engine, tmp_path, until=lambda: st["detectors"] and st["detectors"][0].calls >= 3,
    )

    assert reached and alive
    assert not t.is_alive()
    assert len(st["captures"]) == 1
    assert st["captures"][0].stopped


# ── A causa concreta: chat_rate sem snapshot ────────────────────────────────

def test_chat_rate_is_safe_while_irc_thread_mutates(monkeypatch):
    """
    O IRC faz append/popleft em _msg_times enquanto o engine lê chat_rate.
    Com intervalo de troca de thread mínimo, a iteração por gerador quebrava.
    """
    from conftest import detection_config
    from moment_detector import TwitchChatDetector
    det = TwitchChatDetector(detection_config(), lambda: 100.0)
    stop = threading.Event()

    def irc():
        while not stop.is_set():
            det._msg_times.append(time.time())
            if len(det._msg_times) > 200:
                det._msg_times.popleft()

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    errors = 0
    t = threading.Thread(target=irc, daemon=True)
    t.start()
    try:
        for _ in range(20_000):
            try:
                det.chat_rate
            except RuntimeError:
                errors += 1
    finally:
        stop.set()
        t.join(timeout=2)
        sys.setswitchinterval(old)

    assert errors == 0, f"{errors} leitura(s) de chat_rate quebraram por mutação concorrente"
