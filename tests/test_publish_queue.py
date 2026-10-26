"""
Lote 2b / Patch 6 — fila de publicação: a captura nunca espera upload.

Antes, no modo automático, o canal gerava as descrições (LLM + Whisper),
postava em cada plataforma e esperava `delay_between_posts` (300 s padrão)
DENTRO do loop de captura. Com 3 plataformas, ~15 min por clip com o canal
parado — e os momentos detectados nesse meio-tempo expiravam (TTL de 300 s).
O upload manual da UI postava numa thread solta, sem retry nem fila.

Contrato da fila:
  • enfileirar é instantâneo; quem publica é uma thread própria;
  • um post por vez, com o intervalo entre posts da UI;
  • falha → nova tentativa em 5 min e 30 min; desiste na 3ª;
  • limite diário → adia para o dia seguinte (não descarta);
  • persiste em disco: reiniciar o app não perde posts na fila;
  • o clip local só é apagado quando TODAS as plataformas confirmaram
    (antes, cleanup_after_upload apagava com qualquer sucesso).
"""

import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from conftest import make_moment


class _Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def qmod(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    import publish_queue
    return publish_queue


def _clip(tmp_path, name="clip_twitch_kosky_x_keyword.mp4"):
    p = tmp_path / name
    p.write_bytes(b"\x00" * 1024)
    return p


def _queue(qmod, publish, *, clock=None, delay=0.0, cleanup=False, results=None, path=None):
    return qmod.PublishQueue(
        publish,
        delay_fn=lambda: delay,
        cleanup_fn=lambda: cleanup,
        on_result=(lambda job, plat, ok, why: results.append((plat, ok, why))) if results is not None else None,
        clock=clock or _Clock(),
        path=path,
    )


# ── Básico ───────────────────────────────────────────────────────────────────

def test_enqueue_does_not_publish(qmod, tmp_path):
    """O ponto do Patch 6: quem enfileira (o canal) não espera o upload."""
    calls = []
    q = _queue(qmod, lambda job, plat: calls.append(plat) or True)

    q.enqueue(_clip(tmp_path), ["tiktok", "youtube"])

    assert calls == []
    assert len(q.pending()) == 1


def test_worker_step_publishes_each_platform_once(qmod, tmp_path):
    calls = []
    q = _queue(qmod, lambda job, plat: calls.append(plat) or True)
    q.enqueue(_clip(tmp_path), ["tiktok", "youtube"])

    while q.run_once():
        pass

    assert calls == ["tiktok", "youtube"]
    assert q.pending() == []


def test_posts_are_spaced_by_the_ui_interval(qmod, tmp_path):
    clock = _Clock()
    calls = []
    q = _queue(qmod, lambda job, plat: calls.append(plat) or True, clock=clock, delay=300.0)
    q.enqueue(_clip(tmp_path), ["tiktok", "youtube"])

    assert q.run_once() is True
    assert q.run_once() is False, "segundo post saiu sem respeitar o intervalo"
    clock.t += 300
    assert q.run_once() is True
    assert calls == ["tiktok", "youtube"]


# ── Falhas ───────────────────────────────────────────────────────────────────

def test_failure_is_retried_with_backoff_then_given_up(qmod, tmp_path):
    clock = _Clock()
    results = []
    calls = []
    q = _queue(qmod, lambda job, plat: calls.append(plat) or False, clock=clock, results=results)
    q.enqueue(_clip(tmp_path), ["tiktok"])

    assert q.run_once()                 # tentativa 1
    assert q.run_once() is False        # espera o backoff
    clock.t += qmod.PublishQueue.RETRY_BACKOFF_SEC[0]
    assert q.run_once()                 # tentativa 2
    clock.t += qmod.PublishQueue.RETRY_BACKOFF_SEC[1]
    assert q.run_once()                 # tentativa 3 → desiste

    assert len(calls) == qmod.PublishQueue.MAX_ATTEMPTS
    assert q.pending() == []
    assert results[-1][:2] == ("tiktok", False)


def test_exception_counts_as_a_failed_attempt(qmod, tmp_path):
    results = []

    def boom(job, plat):
        raise RuntimeError("rede caiu")
    q = _queue(qmod, boom, results=results)
    q.enqueue(_clip(tmp_path), ["tiktok"])

    assert q.run_once() is True
    assert len(q.pending()) == 1, "exceção não pode derrubar nem descartar o job"
    assert "rede caiu" in (q.pending()[0]["last_error"].get("tiktok") or "")


def test_daily_limit_defers_to_the_next_day(qmod, tmp_path):
    """publish devolve None quando o teto diário recusou: adia, não descarta."""
    clock = _Clock()
    answers = [None, True]
    q = _queue(qmod, lambda job, plat: answers.pop(0), clock=clock)
    q.enqueue(_clip(tmp_path), ["tiktok"])

    assert q.run_once() is True
    assert len(q.pending()) == 1
    clock.t += 3600
    assert q.run_once() is False, "tentou de novo no mesmo dia"

    amanha = datetime.fromtimestamp(clock.t).replace(hour=0, minute=0, second=0) + timedelta(days=1, minutes=2)
    clock.t = amanha.timestamp()
    assert q.run_once() is True
    assert q.pending() == []


def test_missing_clip_fails_without_retries(qmod, tmp_path):
    calls = []
    q = _queue(qmod, lambda job, plat: calls.append(plat) or True)
    clip = _clip(tmp_path)
    q.enqueue(clip, ["tiktok"])
    clip.unlink()

    assert q.run_once() is True
    assert calls == []
    assert q.pending() == []


# ── Persistência ─────────────────────────────────────────────────────────────

def test_queue_survives_an_app_restart(qmod, tmp_path):
    _queue(qmod, lambda j, p: True).enqueue(_clip(tmp_path), ["tiktok"], channel="kosky")

    again = _queue(qmod, lambda j, p: True)

    assert [j["channel"] for j in again.pending()] == ["kosky"]


def test_progress_survives_an_app_restart(qmod, tmp_path):
    """Plataforma já postada não é postada de novo depois do restart."""
    calls = []
    q1 = _queue(qmod, lambda j, p: calls.append(p) or True)
    q1.enqueue(_clip(tmp_path), ["tiktok", "youtube"])
    q1.run_once()

    q2 = _queue(qmod, lambda j, p: calls.append(p) or True)
    while q2.run_once():
        pass

    assert calls == ["tiktok", "youtube"]


# ── Limpeza do clip local (o problema do Patch 7) ───────────────────────────

def test_clip_is_deleted_only_when_every_platform_succeeded(qmod, tmp_path):
    clip = _clip(tmp_path)
    q = _queue(qmod, lambda j, p: True, cleanup=True)
    q.enqueue(clip, ["tiktok", "youtube"])
    while q.run_once():
        pass

    assert not clip.exists()


def test_clip_is_kept_when_any_platform_failed(qmod, tmp_path):
    clock = _Clock()
    clip = _clip(tmp_path)
    q = _queue(qmod, lambda j, p: p == "youtube", cleanup=True, clock=clock)
    q.enqueue(clip, ["tiktok", "youtube"])
    for _ in range(10):
        q.run_once()
        clock.t += 3600

    assert q.pending() == []
    assert clip.exists(), "clip apagado com uma plataforma sem post"


# ── Thread do worker ─────────────────────────────────────────────────────────

def test_worker_thread_publishes_in_the_background(qmod, tmp_path):
    done = threading.Event()
    q = _queue(qmod, lambda j, p: done.set() or True)
    q.IDLE_POLL_SEC = 0.02
    q.start()
    try:
        q.enqueue(_clip(tmp_path), ["tiktok"])
        assert done.wait(timeout=3), "worker não publicou"
    finally:
        q.stop()
    assert not q.is_running()


# ── Integração: modo automático enfileira em vez de postar no loop ──────────

def test_auto_mode_enqueues_instead_of_posting_in_the_channel_loop(store, tmp_path, monkeypatch):
    import bot_engine
    import clip_analyzer
    import moment_detector
    import publish_queue
    import stream_capture
    from conftest import detection_config
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(clip_analyzer, "is_offline_screen", lambda *a, **kw: (False, "stub"))
    moment = make_moment(start=90.0, end=120.0)

    class StubbedDetector(moment_detector.MomentDetector):
        def __init__(self, cfg, elapsed_fn, platform=None, channel=None):
            super().__init__(cfg, elapsed_fn, platform=platform, channel=channel)
            self.chat = self.sim = self.manual = None
            self.audio = type("S", (), {"get_moments_snapshot": lambda _s: [moment]})()
            self._detectors = []

        def start(self, stream_file=None):
            pass

        def stop(self):
            pass

    class FakeCapture:
        def __init__(self, cfg, clips_dir):
            pass

        def start_recording(self, platform):
            p = tmp_path / "raw_twitch_kosky_0.ts"
            p.write_bytes(b"\x47" * 4096)
            return p

        def get_elapsed_seconds(self):
            return 400.0

        def extract_segment(self, src, s, e, out):
            Path(out).write_bytes(b"\x47" * 4096)
            return out

        def stop_recording(self):
            pass

    class FakeEditor:
        def process(self, raw_clip, name, **kw):
            p = tmp_path / f"{name}.mp4"
            p.write_bytes(b"\x00" * 1024)
            return p

    monkeypatch.setattr(stream_capture, "StreamCapture", FakeCapture)
    monkeypatch.setattr(moment_detector, "MomentDetector", StubbedDetector)

    published = []
    q = publish_queue.PublishQueue(lambda j, p: published.append(p) or True)
    eng = bot_engine.BotEngine(pending_store=store, publish_queue=q)
    eng._TICK_SEC = 0.01
    cfg = {**detection_config(audio_enabled=False),
           "system": {"approval_mode": False, "cleanup_after_upload": True},
           "editing": {"clip_analysis": {"enabled": False}},
           "upload": {"tiktok": {"enabled": True}, "instagram": {"enabled": False},
                      "youtube": {"enabled": True}, "delay_between_posts": 300,
                      "max_clips_per_day": 10}}

    t = threading.Thread(target=eng._monitor_platform,
                         args=("twitch", "https://www.twitch.tv/kosky", tmp_path,
                               FakeEditor(), None, None, None),
                         kwargs={"config": cfg}, daemon=True)
    t.start()
    deadline = time.time() + 5
    while time.time() < deadline and not q.pending():
        time.sleep(0.01)
    eng._stop_event.set()
    t.join(timeout=5)

    jobs = q.pending()
    assert len(jobs) == 1, "o clip do modo automático não entrou na fila"
    assert jobs[0]["platforms"] == ["tiktok", "youtube"]
    assert jobs[0]["label"] == "TWITCH:kosky"
    assert published == [], "o canal publicou dentro do loop de captura"


# ── Integração: upload manual da UI entra na fila e volta na hora ───────────

def test_manual_upload_returns_before_publishing(tmp_path, monkeypatch):
    from test_publish_limits import _manual_upload_rig
    import uploader
    api, posted = _manual_upload_rig(tmp_path, monkeypatch, max_per_day=5)
    gate = threading.Event()

    class SlowTikTok:
        def __init__(self, config):
            pass

        def upload(self, video, *a, **kw):
            gate.wait(timeout=5)
            posted.append(video)
            return True

    monkeypatch.setattr(uploader, "TikTokUploader", SlowTikTok)
    q = api._publish_queue
    q.IDLE_POLL_SEC = 0.02
    monkeypatch.setattr(q, "start", type(q).start.__get__(q))   # worker de verdade
    clip = tmp_path / "clip_twitch_kosky_x_keyword.mp4"
    clip.write_bytes(b"\x00")

    t0 = time.time()
    r = api.upload_clip(str(clip), ["tiktok"])
    elapsed = time.time() - t0

    try:
        assert r["ok"] is True and r.get("queued")
        assert elapsed < 1.0, f"upload_clip bloqueou {elapsed:.1f}s esperando o post"
        gate.set()
        deadline = time.time() + 5
        while time.time() < deadline and not posted:
            time.sleep(0.02)
        assert len(posted) == 1
    finally:
        gate.set()
        q.stop()


def test_clip_shared_by_another_pending_job_is_not_deleted(qmod, tmp_path):
    """
    O mesmo clip enviado em dois envios separados (TikTok agora, YouTube
    depois): o primeiro job terminar não pode apagar o arquivo do segundo.
    """
    clip = _clip(tmp_path)
    calls = []
    q = _queue(qmod, lambda j, p: calls.append(p) or True, cleanup=True)
    q.enqueue(clip, ["tiktok"])
    q.enqueue(clip, ["youtube"])

    while q.run_once():
        pass

    assert calls == ["tiktok", "youtube"], "o segundo envio perdeu o arquivo"
    assert not clip.exists(), "depois do último job confirmado, o clip deve sair"
