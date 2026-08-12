"""
Patch 10 — funil de momentos + heartbeat por canal.

Responde à pergunta que hoje não tem instrumento: "o bot está rodando, mas
por que não saiu clip?". Cada porta de descarte do pipeline vira um contador
por canal, e o heartbeat mostra se a thread do canal ainda avança.

Também valida a reconciliação do funil: detected = gates + rendered + em voo.
"""

import threading

import pytest

from conftest import make_moment


class _EditorDiscards:
    def process(self, *a, **kw):
        return None


class _EditorOK:
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def process(self, raw_clip, name, **kw):
        p = self.out_dir / f"{name}.mp4"
        p.write_bytes(b"\x00" * 128)
        return p


@pytest.fixture
def engine(store, tmp_path, monkeypatch):
    import bot_engine
    import clip_analyzer
    monkeypatch.setattr(clip_analyzer, "is_offline_screen",
                        lambda *a, **kw: (False, "stub"))
    eng = bot_engine.BotEngine(pending_store=store)
    out = tmp_path / "output"
    out.mkdir()
    eng._config = {"system": {"output_folder": str(out), "clips_folder": str(tmp_path)}}
    eng._title_gen = None
    eng._output_dir = out
    eng._clips_dir = tmp_path
    return eng


# ── Núcleo: contadores e heartbeat ───────────────────────────────────────────

def test_funnel_counts_by_stage(engine):
    engine._funnel("TWITCH:kosky", "detected")
    engine._funnel("TWITCH:kosky", "detected")
    engine._funnel("TWITCH:kosky", "offline_gate")

    f = engine.get_health()["TWITCH:kosky"]["funnel"]
    assert f["detected"] == 2
    assert f["offline_gate"] == 1
    assert f["rendered"] == 0


def test_funnel_is_isolated_per_channel(engine):
    engine._funnel("TWITCH:kosky", "detected")
    engine._funnel("KICK:jonvlogs", "errors")

    health = engine.get_health()
    assert health["TWITCH:kosky"]["funnel"]["detected"] == 1
    assert health["TWITCH:kosky"]["funnel"]["errors"] == 0
    assert health["KICK:jonvlogs"]["funnel"]["errors"] == 1
    assert health["KICK:jonvlogs"]["funnel"]["detected"] == 0


def test_all_declared_stages_are_present(engine):
    """Todo estágio declarado precisa existir no snapshot (contrato da UI)."""
    engine._funnel("TWITCH:x", "detected")
    f = engine.get_health()["TWITCH:x"]["funnel"]
    for stage in engine.FUNNEL_STAGES:
        assert stage in f, f"estágio '{stage}' ausente do snapshot"


def test_heartbeat_and_state_are_tracked(engine):
    engine._touch_channel("TWITCH:kosky", state="capturing", attempt=1)
    h = engine.get_health()["TWITCH:kosky"]

    assert h["state"] == "capturing"
    assert h["tick_age_sec"] < 1.0
    assert h["uptime_sec"] >= 0.0
    assert h["clip_age_sec"] is None, "sem clip ainda → idade None, não 0"


def test_rendered_updates_last_clip_age(engine):
    engine._funnel("TWITCH:kosky", "rendered")
    h = engine.get_health()["TWITCH:kosky"]
    assert h["clip_age_sec"] is not None and h["clip_age_sec"] < 1.0


def test_moments_active_is_a_gauge_not_a_counter(engine):
    """moments_active reflete o snapshot ATUAL — não acumula entre ticks."""
    engine._touch_channel("TWITCH:kosky", moments_active=12)
    engine._touch_channel("TWITCH:kosky", moments_active=3)
    assert engine.get_health()["TWITCH:kosky"]["moments_active"] == 3


def test_health_snapshot_is_json_serializable(engine):
    """A UI recebe isso via evaluate_js → precisa ser serializável."""
    import json
    engine._funnel("TWITCH:kosky", "detected")
    engine._touch_channel("TWITCH:kosky", state="frozen")
    json.dumps(engine.get_health())     # não pode levantar


def test_snapshot_does_not_leak_internal_state(engine):
    """Mutar o snapshot não pode afetar o estado interno."""
    engine._funnel("TWITCH:kosky", "detected")
    snap = engine.get_health()
    snap["TWITCH:kosky"]["funnel"]["detected"] = 999

    assert engine.get_health()["TWITCH:kosky"]["funnel"]["detected"] == 1


def test_concurrent_updates_are_atomic(engine):
    """N threads de canal incrementam o mesmo funil sem perder contagem."""
    def worker():
        for _ in range(200):
            engine._funnel("TWITCH:kosky", "detected")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert engine.get_health()["TWITCH:kosky"]["funnel"]["detected"] == 800


def test_funnel_never_raises_on_unknown_stage(engine):
    """Observabilidade não pode derrubar produção nem com estágio novo."""
    engine._funnel("TWITCH:kosky", "estagio_inexistente")
    assert engine.get_health()["TWITCH:kosky"]["funnel"]["estagio_inexistente"] == 1


# ── Integração com o pipeline de aprovação ───────────────────────────────────

def _seed(store, raw_ts):
    raw = raw_ts()
    m = make_moment()
    store.add(m, raw, "twitch", "kosky")
    return m


def test_generate_increments_rendered(engine, store, raw_ts):
    m = _seed(store, raw_ts)
    engine._editor = _EditorOK(engine._output_dir)

    engine.generate_clip_from_moment(m.moment_id, title_text="X")

    f = engine.get_health()["TWITCH:kosky"]["funnel"]
    assert f["rendered"] == 1
    assert f["editor_discard"] == 0


def test_generate_increments_editor_discard(engine, store, raw_ts):
    m = _seed(store, raw_ts)
    engine._editor = _EditorDiscards()

    engine.generate_clip_from_moment(m.moment_id, title_text="X")

    f = engine.get_health()["TWITCH:kosky"]["funnel"]
    assert f["editor_discard"] == 1
    assert f["rendered"] == 0


def test_generate_increments_offline_gate(engine, store, raw_ts, monkeypatch):
    import clip_analyzer
    monkeypatch.setattr(clip_analyzer, "is_offline_screen",
                        lambda *a, **kw: (True, "rms=0.00000"))
    m = _seed(store, raw_ts)
    engine._editor = _EditorOK(engine._output_dir)

    result = engine.generate_clip_from_moment(m.moment_id, title_text="X")

    assert result["ok"] is False and result["reason"] == "offline_screen"
    f = engine.get_health()["TWITCH:kosky"]["funnel"]
    assert f["offline_gate"] == 1
    assert f["rendered"] == 0


def test_funnel_reconciles_detected_vs_outcomes(engine, store, raw_ts):
    """
    Invariante de leitura: para os momentos que saíram do pipeline,
    gates + rendered deve bater com o que entrou.
    """
    engine._editor = _EditorDiscards()
    for _ in range(3):
        raw = raw_ts(f"raw_pending_t_{_}.ts")
        m = make_moment(start=100.0 + _, end=115.0 + _)
        store.add(m, raw, "twitch", "kosky")
        engine._funnel("TWITCH:kosky", "detected")
        engine.generate_clip_from_moment(m.moment_id, title_text="X")

    f = engine.get_health()["TWITCH:kosky"]["funnel"]
    outcomes = (f["rendered"] + f["editor_discard"] + f["offline_gate"]
                + f["score_gate"] + f["reaction_gate"] + f["quality_gate"])
    assert f["detected"] == 3
    assert outcomes == 3, f"funil não reconcilia: detected=3, saídas={outcomes}"


# ── Exposição para a UI ──────────────────────────────────────────────────────

def test_api_status_exposes_channel_health(engine, monkeypatch):
    import clipbot_api
    engine._funnel("TWITCH:kosky", "detected")
    engine.is_running = True

    api = clipbot_api.ClipBotAPI()
    api._engine = engine
    status = api.get_status()

    assert status["running"] is True
    assert "TWITCH:kosky" in status["channels"]
    assert status["channels"]["TWITCH:kosky"]["funnel"]["detected"] == 1


def test_api_status_survives_engine_without_health(monkeypatch):
    """get_status jamais pode levantar — a UI depende dele."""
    import clipbot_api

    class Broken:
        is_running = True
        clips_today = 0
        clips_total = 0

        def get_health(self):
            raise RuntimeError("boom")

    api = clipbot_api.ClipBotAPI()
    api._engine = Broken()

    status = api.get_status()
    assert status["channels"] == {}
