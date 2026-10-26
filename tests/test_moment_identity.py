"""
Lote 2 / B3 — a identidade de um momento fundido mudava entre ticks.

O engine identificava um momento por (label, round(start), reason) e o
liberava em `elapsed >= end + 2` — mas o merge ainda atrai momentos que
começam até `end + MERGE_DIST` (15 s). O grupo crescia DEPOIS de liberado:

  • ganhava chave nova (reason virava super_moment, ou o start mudava com o
    teto de 60 s) → processado de novo → clip DUPLICADO cobrindo o primeiro;
  • ou mantinha a mesma chave → o membro novo era ENGOLIDO, nunca virava clip.

E sem teto de extensão no merge, chat comum ("kkkk" a cada 20 s) encadeava a
live inteira num grupo só: o primeiro clip demorava minutos, e eventos
seguintes saíam atrasados ou nunca.

Contrato:
  • um grupo tem no máximo max_duration de extensão (vira vários, não um);
  • só é liberado quando nenhum momento futuro consegue mais entrar nele;
  • a identidade é o conjunto de MEMBROS; ao ser processado, eles são
    consumidos e não entram em merges futuros.
"""

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import detection_config, make_moment

CHANNEL = "https://www.twitch.tv/kosky"
BUFFER_BEFORE, BUFFER_AFTER = 10.0, 20.0     # os de detection_config


def _raw(t, reason="keyword", score=0.5):
    """Momento como um detector cria no instante t: [t-10, t+20]."""
    return make_moment(start=t - BUFFER_BEFORE, end=t + BUFFER_AFTER,
                       reason=reason, score=score)


class _Source:
    def __init__(self):
        self.moments = []
        self.lock = threading.Lock()

    def add(self, m):
        with self.lock:
            self.moments.append(m)

    def get_moments_snapshot(self):
        with self.lock:
            return list(self.moments)


# ── Unidade: MomentDetector ─────────────────────────────────────────────────

def _detector():
    from moment_detector import MomentDetector
    det = MomentDetector(detection_config(audio_enabled=False), lambda: 0.0,
                         platform="twitch", channel="kosky")
    src = _Source()
    det.audio = src
    det.chat = det.sim = det.manual = None
    return det, src


def test_continuous_chat_is_split_into_bounded_groups():
    """Antes: 16 keywords a cada 20 s viravam UM grupo truncado."""
    from moment_detector import MomentDetector
    chain = [_raw(t) for t in range(100, 420, 20)]

    merged = MomentDetector._merge_moments(chain, max_duration=60.0)

    assert len(merged) >= 4, f"{len(merged)} grupo(s) para 5 min de chat contínuo"
    assert all(m.duration <= 60.0 for m in merged)


def test_group_identity_is_its_members():
    from moment_detector import MomentDetector
    a, b = _raw(100, "audio_spike"), _raw(110, "chat_flood")

    merged = MomentDetector._merge_moments([a, b], max_duration=60.0)

    assert MomentDetector.members(merged[0]) == tuple(sorted([a.moment_id, b.moment_id]))
    assert MomentDetector.members(a) == (a.moment_id,)   # passthrough


def test_consumed_members_leave_future_merges():
    det, src = _detector()
    a = _raw(100)
    src.add(a)
    det.consume(det.all_moments[0])

    b = _raw(112)                       # alcançaria `a` pelo MERGE_DIST
    src.add(b)
    after = det.all_moments

    assert len(after) == 1
    assert det.members(after[0]) == (b.moment_id,), "membro consumido voltou ao merge"


def test_group_is_not_settled_while_a_new_member_can_still_join():
    """
    Um momento criado em t começa em t - buffer_before e entra no grupo se
    começar até end + MERGE_DIST. Antes do fim dessa janela, o grupo está aberto.
    """
    from moment_detector import MomentDetector
    det, src = _detector()
    src.add(_raw(100))                  # [90, 120]
    m = det.all_moments[0]
    fecha = 120 + MomentDetector.MERGE_DIST + BUFFER_BEFORE

    assert not det.is_settled(m, 122.0), "liberado com o grupo ainda aberto (gate antigo)"
    assert not det.is_settled(m, fecha - 1)
    assert det.is_settled(m, fecha + 2)


# ── Integração: engine real em modo aprovação, relógio controlado ───────────

@pytest.fixture
def rig(store, tmp_path, monkeypatch):
    import bot_engine
    import moment_detector
    import stream_capture
    clock = [0.0]
    ticks = [0]
    source = _Source()
    real_all = moment_detector.MomentDetector.all_moments

    class StubbedDetector(moment_detector.MomentDetector):
        """Lógica real de merge/identidade; fontes e IO substituídos."""
        def __init__(self, cfg, elapsed_fn, platform=None, channel=None):
            super().__init__(cfg, elapsed_fn, platform=platform, channel=channel)
            self.audio, self.chat, self.sim, self.manual = source, None, None, None
            self._detectors = []

        def start(self, stream_file=None):
            pass

        def stop(self):
            pass

        @property
        def all_moments(self):
            ticks[0] += 1
            return real_all.fget(self)

    class FakeCapture:
        def __init__(self, cfg, clips_dir):
            pass

        def start_recording(self, platform):
            p = tmp_path / "raw_twitch_kosky_0.ts"
            p.write_bytes(b"\x47" * 4096)
            return p

        def get_elapsed_seconds(self):
            return clock[0]

        def extract_segment(self, src, s, e, out):
            Path(out).write_bytes(b"\x47" * 188)
            return out

        def stop_recording(self):
            pass

    monkeypatch.setattr(stream_capture, "StreamCapture", FakeCapture)
    monkeypatch.setattr(moment_detector, "MomentDetector", StubbedDetector)
    eng = bot_engine.BotEngine(pending_store=store)
    eng._TICK_SEC = 0.002
    added = []
    eng.on_moment_pending = lambda entry: added.append((clock[0], entry))
    return SimpleNamespace(eng=eng, clock=clock, ticks=ticks, source=source,
                           added=added, tmp=tmp_path)


def _play(rig, events, until, step=1):
    """
    Avança o relógio da captura segundo a segundo, injeta os momentos no
    instante em que um detector os criaria, e espera dois ticks do engine a
    cada passo — o engine vê todo instante, inclusive a janela em que o gate
    antigo liberava o grupo cedo demais.
    """
    cfg = {**detection_config(audio_enabled=False),
           "system": {"approval_mode": True, "approval_top_n": 12}}
    first = min(events)
    rig.clock[0] = float(first - 5)
    t = threading.Thread(
        target=rig.eng._monitor_platform,
        args=("twitch", CHANNEL, rig.tmp, None, None, None, None),
        kwargs={"config": cfg}, daemon=True,
    )
    t.start()
    pending_events = sorted(events.items())
    try:
        for now in range(first - 5, until + 1, step):
            rig.clock[0] = float(now)
            # Injeta tudo cujo instante já passou (o passo pode pular o exato).
            while pending_events and pending_events[0][0] <= now:
                for m in pending_events.pop(0)[1]:
                    rig.source.add(m)
            n = rig.ticks[0]
            deadline = time.time() + 2
            while rig.ticks[0] < n + 2 and time.time() < deadline:
                time.sleep(0.0005)
    finally:
        rig.eng._stop_event.set()
        t.join(timeout=5)
    return [e for _, e in rig.added]


def test_second_spike_40s_later_is_not_swallowed(rig):
    """Caso 1 da revisão: antes, 1 pendente para 2 eventos."""
    pend = _play(rig, {100: [_raw(100, "audio_spike")],
                       140: [_raw(140, "audio_spike")]}, until=230)

    assert len(pend) == 2, f"{len(pend)} pendente(s) para 2 picos de áudio"
    janelas = {(e["start_sec"], e["end_sec"]) for e in pend}
    assert len(janelas) == 2


def test_spike_then_chat_is_one_super_moment(rig):
    """Caso 2 da revisão: antes, audio_spike + super_moment sobrepostos."""
    pend = _play(rig, {100: [_raw(100, "audio_spike")],
                       125: [_raw(125, "chat_flood")]}, until=210)

    assert len(pend) == 1, f"{len(pend)} pendentes para um único evento"
    assert pend[0]["reason"] == "super_moment"


def test_continuous_chat_releases_bounded_clips_early(rig):
    """Chat comum a cada 20 s: clips de no máximo 60 s, o primeiro cedo."""
    events = {t: [_raw(t)] for t in range(100, 420, 20)}
    pend = _play(rig, events, until=500, step=2)
    first_at = rig.added[0][0] if rig.added else None

    assert len(pend) >= 4, f"{len(pend)} pendente(s) para 5 min de chat contínuo"
    assert first_at is not None and first_at < 200, f"primeiro clip só em t={first_at}"
    assert all(e["end_sec"] - e["start_sec"] <= 60.0 for e in pend)

    vistos = set()
    for e in pend:
        membros = set(e["extra"].get("members") or [e["moment_id"]])
        assert not (membros & vistos), "o mesmo momento bruto entrou em dois pendentes"
        vistos |= membros
