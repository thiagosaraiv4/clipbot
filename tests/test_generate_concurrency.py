"""
Lote 2 / B12 — arquivos com nome colidindo e "Gerar" duplicado.

Os nomes levavam o timestamp em SEGUNDOS e o reason:
    clip_{plataforma}_{canal}_{YYYYmmdd_HHMMSS}_{reason}.mp4
    raw_pending_{plataforma}_{canal}_{YYYYmmdd_HHMMSS}_{reason}.ts

  • dois "Gerar" de momentos diferentes, mesmo canal e reason, no mesmo
    segundo → mesmo .mp4 e mesmo sidecar: dois ffmpeg escrevendo no mesmo
    arquivo, os dois retornam ok com o MESMO caminho, um clip some;
  • dois pendentes liberados no mesmo tick (comum depois do B3: vários grupos
    fechados de uma vez) → mesmo raw_pending .ts: a segunda extração
    sobrescreve a primeira, e o primeiro momento passa a apontar para o
    vídeo do segundo.

E o MESMO momento gerado duas vezes (duplo clique, card recriado ao voltar à
aba): _processing é um set, sem checagem; o primeiro a terminar apaga o .ts
que o segundo ainda usa.

Contrato: nome único por momento; um momento só tem uma geração em curso.
"""

import json
import threading
import time
from datetime import datetime as _real_datetime
from pathlib import Path

import pytest

from conftest import make_moment


class _FrozenDatetime(_real_datetime):
    """Relógio parado: todos os nomes caem no mesmo segundo."""
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 26, 15, 30, 45)


class _EditorOK:
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def process(self, raw_clip, name, **kw):
        p = self.out_dir / f"{name}.mp4"
        p.write_bytes(Path(raw_clip).read_bytes())
        return p


@pytest.fixture
def engine(store, tmp_path, monkeypatch):
    import bot_engine
    import clip_analyzer
    monkeypatch.setattr(clip_analyzer, "is_offline_screen", lambda *a, **kw: (False, "stub"))
    monkeypatch.setattr(bot_engine, "datetime", _FrozenDatetime)
    eng = bot_engine.BotEngine(pending_store=store)
    out = tmp_path / "output"
    out.mkdir()
    eng._config = {"system": {"output_folder": str(out), "clips_folder": str(tmp_path)}}
    eng._title_gen = None
    eng._editor = _EditorOK(out)
    return eng


def _pending(store, tmp_path, start, reason="keyword", content=b"A"):
    raw = tmp_path / f"raw_pending_{start:.0f}_{reason}.ts"
    raw.write_bytes(content * 4096)
    m = make_moment(start=start, end=start + 15.0, reason=reason)
    store.add(m, raw, "twitch", "kosky")
    return m


# ── Nomes únicos no "Gerar" ─────────────────────────────────────────────────

def test_two_moments_in_the_same_second_get_distinct_files(engine, store, tmp_path):
    a = _pending(store, tmp_path, 100.0, content=b"A")
    b = _pending(store, tmp_path, 300.0, content=b"B")

    ra = engine.generate_clip_from_moment(a.moment_id, title_text="X")
    rb = engine.generate_clip_from_moment(b.moment_id, title_text="X")

    assert ra["ok"] and rb["ok"]
    assert ra["path"] != rb["path"], "dois momentos geraram o MESMO arquivo"
    assert Path(ra["path"]).read_bytes()[:1] == b"A", "clip do primeiro foi sobrescrito"


def test_each_clip_keeps_its_own_sidecar(engine, store, tmp_path):
    a = _pending(store, tmp_path, 100.0)
    b = _pending(store, tmp_path, 300.0)

    paths = [engine.generate_clip_from_moment(m.moment_id, title_text="X")["path"]
             for m in (a, b)]

    ids = [json.loads(Path(p).with_suffix(".json").read_text(encoding="utf-8"))["moment_id"]
           for p in paths]
    assert ids == [a.moment_id, b.moment_id]


def test_clip_name_still_parses_in_the_clips_tab(engine, store, tmp_path):
    """get_clips lê plataforma, canal e reason por posição no nome."""
    a = _pending(store, tmp_path, 100.0)

    stem = Path(engine.generate_clip_from_moment(a.moment_id, title_text="X")["path"]).stem
    parts = stem.split("_")

    assert (parts[0], parts[1], parts[2], parts[-1]) == ("clip", "twitch", "kosky", "keyword")


# ── Mesmo momento, duas gerações ao mesmo tempo ─────────────────────────────

def test_same_moment_twice_at_once_is_refused(engine, store, tmp_path):
    """
    O segundo "Gerar" do mesmo momento precisa voltar na hora, sem render — e
    sem apagar o .ts que a primeira geração ainda está usando.
    """
    m = _pending(store, tmp_path, 100.0)
    entered, release = threading.Event(), threading.Event()
    renders = []

    class SlowEditor(_EditorOK):
        def process(self, raw_clip, name, **kw):
            renders.append(name)
            entered.set()
            release.wait(timeout=5)
            return super().process(raw_clip, name, **kw)

    engine._editor = SlowEditor(engine._editor.out_dir)
    results = {}
    t1 = threading.Thread(target=lambda: results.update(
        first=engine.generate_clip_from_moment(m.moment_id, title_text="X")))
    t1.start()
    assert entered.wait(timeout=5)

    t2 = threading.Thread(target=lambda: results.update(
        second=engine.generate_clip_from_moment(m.moment_id, title_text="X")))
    t2.start()
    t2.join(timeout=2)
    second_returned = not t2.is_alive()

    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert second_returned, "a segunda geração ficou presa renderizando o mesmo momento"
    assert results["second"]["ok"] is False
    assert results["second"].get("reason") == "already_processing"
    assert results["first"]["ok"] is True, "a primeira geração foi prejudicada"
    assert len(renders) == 1


# ── Pendentes liberados no mesmo tick ───────────────────────────────────────

def test_pendings_released_in_the_same_tick_keep_their_own_raw(store, tmp_path, monkeypatch):
    """
    Dois grupos fechados liberados no mesmo tick (mesmo canal, mesmo reason):
    cada pendente precisa do seu .ts.
    """
    import bot_engine
    import moment_detector
    import stream_capture
    from conftest import detection_config

    monkeypatch.setattr(bot_engine, "datetime", _FrozenDatetime)
    source = [make_moment(start=90.0, end=120.0), make_moment(start=190.0, end=220.0)]

    class StubbedDetector(moment_detector.MomentDetector):
        def __init__(self, cfg, elapsed_fn, platform=None, channel=None):
            super().__init__(cfg, elapsed_fn, platform=platform, channel=channel)
            self.chat = self.sim = self.manual = None
            self.audio = type("S", (), {"get_moments_snapshot": lambda _s: list(source)})()
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
            return 400.0                 # os dois grupos já fechados

        def extract_segment(self, src, s, e, out):
            Path(out).write_bytes(f"{s:.0f}".encode() * 64)
            return out

        def stop_recording(self):
            pass

    monkeypatch.setattr(stream_capture, "StreamCapture", FakeCapture)
    monkeypatch.setattr(moment_detector, "MomentDetector", StubbedDetector)
    eng = bot_engine.BotEngine(pending_store=store)
    eng._TICK_SEC = 0.01
    cfg = {**detection_config(audio_enabled=False),
           "system": {"approval_mode": True, "approval_top_n": 12}}

    t = threading.Thread(target=eng._monitor_platform,
                         args=("twitch", "https://www.twitch.tv/kosky", tmp_path,
                               None, None, None, None),
                         kwargs={"config": cfg}, daemon=True)
    t.start()
    deadline = time.time() + 5
    while time.time() < deadline and len(store.list_all()) < 2:
        time.sleep(0.01)
    eng._stop_event.set()
    t.join(timeout=5)

    entries = store.list_all()
    assert len(entries) == 2
    raws = {e["raw_file"] for e in entries}
    assert len(raws) == 2, "dois pendentes apontam para o MESMO .ts"
    for e in entries:
        conteudo = Path(e["raw_file"]).read_bytes()
        assert conteudo.startswith(f"{e['segment_start_sec']:.0f}".encode()), (
            "o .ts de um pendente contém o segmento de outro"
        )
