"""
Lote 1 / A2 — pending.json ilegível no boot: o furo no item 2.

O item 2 fez o _load distinguir "não existe" de "não consegui ler", e o
comentário dizia: "sobrescrevê-lo os apagaria". Nada impedia isso:

  1. AV segura o pending.json no import → store sobe vazia (UNREADABLE).
  2. Primeiro momento novo → _save_locked grava pending.json só com ele.
     O arquivo com os pendentes da sessão anterior é destruído.
  3. last_save_error volta a None → o sinal de saúde some.
  4. ~10 min depois, _gc_orphan_pending_clips vê os raw_pending_*.ts da
     sessão anterior sem dono na store e os apaga.

O caminho CORRUPT termina igual: a quarentena preserva o JSON, mas o GC apaga
os .ts que ele descrevia.

Contrato:
  • enquanto o arquivo do boot estiver ilegível, ele não é sobrescrito;
  • a cada save a store tenta ler de novo e, se conseguir, mescla e volta ao
    normal (o bloqueio de AV costuma ser transitório);
  • enquanto a store não souber todos os donos, nenhum GC apaga
    raw_pending_*.ts anterior ao boot;
  • o sinal de problema não some no primeiro save bem-sucedido.
"""

import json
import os
import time

import pytest

from conftest import make_moment

OLD = time.time() - 2 * 3600


# ── helpers ─────────────────────────────────────────────────────────────────

def _break_reads(monkeypatch, exc=PermissionError("acesso negado (AV)")):
    """Leituras de texto falham até heal() — como um AV segurando o arquivo."""
    from pathlib import Path
    orig = Path.read_text
    broken = {"on": True}

    def maybe(self, *a, **kw):
        if broken["on"]:
            raise exc
        return orig(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", maybe)
    return lambda: broken.update(on=False)


def _previous_session(tmp_path, n=2):
    """pending.json válido + .ts antigos, como a sessão anterior deixou."""
    entries, raws = [], []
    for i in range(n):
        raw = tmp_path / f"raw_pending_twitch_kosky_2026010{i}_000000_keyword.ts"
        raw.write_bytes(b"\x47" * 4096)
        os.utime(raw, (OLD, OLD))
        m = make_moment(start=100.0 + i * 100, end=115.0 + i * 100)
        entries.append({
            "moment_id": m.moment_id, "start_sec": m.start_sec, "end_sec": m.end_sec,
            "segment_start_sec": m.start_sec, "segment_end_sec": m.end_sec,
            "reason": "keyword", "score": 0.5, "keywords": [], "extra": {},
            "raw_file": str(raw), "platform": "twitch", "channel": "kosky",
            "added_at": "2026-01-01T00:00:00",
        })
        raws.append(raw)
    pf = tmp_path / "pending.json"
    pf.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return pf, raws, [e["moment_id"] for e in entries]


def _new_raw(tmp_path, name="raw_pending_twitch_kosky_20261025_120000_keyword.ts"):
    p = tmp_path / name
    p.write_bytes(b"\x47" * 4096)
    return p


def _gc(store, clips_dir):
    import bot_engine
    return bot_engine.BotEngine(pending_store=store)._gc_orphan_pending_clips(clips_dir)


# ── O furo: sobrescrita do arquivo ilegível ─────────────────────────────────

def test_unreadable_pending_is_not_overwritten(store, tmp_path, monkeypatch):
    import pending_store
    pf, _, _ = _previous_session(tmp_path)
    antes = pf.read_bytes()
    _break_reads(monkeypatch)

    s = pending_store.PendingStore()
    s.add(make_moment(start=900.0, end=915.0), _new_raw(tmp_path), "twitch", "kosky")

    assert pf.read_bytes() == antes, "pending.json da sessão anterior foi sobrescrito"


def test_error_signal_survives_while_writes_are_suspended(store, tmp_path, monkeypatch):
    """Antes: o primeiro save bem-sucedido zerava last_save_error."""
    import pending_store
    _previous_session(tmp_path)
    _break_reads(monkeypatch)

    s = pending_store.PendingStore()
    s.add(make_moment(start=900.0, end=915.0), _new_raw(tmp_path), "twitch", "kosky")

    assert s.last_save_error is not None


def test_new_moment_still_usable_while_writes_are_suspended(store, tmp_path, monkeypatch):
    """INVARIANTE: disco ruim não tira o momento da UI de aprovação."""
    import pending_store
    _previous_session(tmp_path)
    _break_reads(monkeypatch)
    m = make_moment(start=900.0, end=915.0)

    s = pending_store.PendingStore()
    s.add(m, _new_raw(tmp_path), "twitch", "kosky")

    assert s.get(m.moment_id) is not None


# ── O furo: GC apaga os .ts que o arquivo ilegível descrevia ─────────────────

def test_orphan_gc_spares_previous_session_when_unreadable(store, tmp_path, monkeypatch):
    import pending_store
    _, raws, _ = _previous_session(tmp_path)
    _break_reads(monkeypatch)

    s = pending_store.PendingStore()
    removed = _gc(s, tmp_path)

    assert removed == 0
    assert all(r.exists() for r in raws), "GC apagou .ts da sessão anterior"


def test_orphan_gc_spares_previous_session_after_corrupt_quarantine(store, tmp_path):
    import pending_store
    _, raws, _ = _previous_session(tmp_path)
    (tmp_path / "pending.json").write_text("{lixo", encoding="utf-8")

    s = pending_store.PendingStore()
    removed = _gc(s, tmp_path)

    assert removed == 0
    assert all(r.exists() for r in raws)
    assert list(tmp_path.glob("pending.json.corrupt*")), "quarentena não aconteceu"


def test_orphan_gc_still_collects_this_sessions_orphans(store, tmp_path, monkeypatch):
    """
    A proteção vale só para o que existia no boot. Órfãos criados nesta sessão
    (ex.: .ts preservado durante um processamento) continuam sendo recolhidos.
    """
    import pending_store
    _, raws, _ = _previous_session(tmp_path)
    _break_reads(monkeypatch)
    s = pending_store.PendingStore()

    # Simula uma sessão longa: boot há 1h, órfão desta sessão com 30 min.
    s._ownership_cutoff = time.time() - 3600
    orphan = _new_raw(tmp_path, "raw_pending_twitch_kosky_20261025_130000_audio_spike.ts")
    t = time.time() - 1800
    os.utime(orphan, (t, t))

    _gc(s, tmp_path)

    assert all(r.exists() for r in raws), "apagou .ts anterior ao boot"
    assert not orphan.exists(), "deixou de recolher órfão desta sessão"


def test_saves_resume_after_corrupt_quarantine(store, tmp_path):
    """Corrompido + quarentenado: o caminho está livre, gravar não destrói nada."""
    import pending_store
    _previous_session(tmp_path)
    pf = tmp_path / "pending.json"
    pf.write_text("{lixo", encoding="utf-8")
    m = make_moment(start=900.0, end=915.0)

    s = pending_store.PendingStore()
    s.add(m, _new_raw(tmp_path), "twitch", "kosky")

    ids = [e["moment_id"] for e in json.loads(pf.read_text(encoding="utf-8"))]
    assert ids == [m.moment_id]
    assert s.load_warning is not None, "a quarentena precisa continuar visível"


# ── Recuperação: o bloqueio de AV é transitório ─────────────────────────────

def test_recovers_and_merges_when_reads_come_back(store, tmp_path, monkeypatch):
    import pending_store
    pf, raws, old_ids = _previous_session(tmp_path)
    heal = _break_reads(monkeypatch)

    s = pending_store.PendingStore()
    m1 = make_moment(start=900.0, end=915.0)
    s.add(m1, _new_raw(tmp_path), "twitch", "kosky")      # suspenso

    heal()
    m2 = make_moment(start=1000.0, end=1015.0)
    s.add(m2, _new_raw(tmp_path, "raw_pending_twitch_kosky_20261025_121000_keyword.ts"),
          "twitch", "kosky")                               # recupera e grava

    ids = {e["moment_id"] for e in json.loads(pf.read_text(encoding="utf-8"))}
    assert ids == set(old_ids) | {m1.moment_id, m2.moment_id}
    assert s.last_save_error is None
    assert s.load_warning is None
    assert s.ownership_cutoff is None, "após recuperar, a store conhece todos os donos"
    assert len(s.list_all()) == 4


def test_healthy_boot_has_no_cutoff(store):
    """Baseline: boot normal não protege nada a mais."""
    assert store.ownership_cutoff is None
    assert store.load_warning is None


# ── GC de startup (o outro limpador) ────────────────────────────────────────

def test_startup_gc_spares_pending_raws_when_ownership_unknown(store, tmp_path, monkeypatch):
    import bot_engine
    import pending_store
    _, raws, _ = _previous_session(tmp_path)
    muito_velho = time.time() - 8 * 24 * 3600
    for r in raws:
        os.utime(r, (muito_velho, muito_velho))
    captura = tmp_path / "raw_twitch_kosky_20260101_000000.ts"     # não é pendente
    captura.write_bytes(b"\x47")
    os.utime(captura, (muito_velho, muito_velho))
    _break_reads(monkeypatch)

    s = pending_store.PendingStore()
    bot_engine.BotEngine(pending_store=s)._garbage_collect_disk(tmp_path, tmp_path)

    assert all(r.exists() for r in raws), "GC de startup apagou pendentes de dono desconhecido"
    assert not captura.exists(), "captura antiga (nunca é pendente) deve continuar sendo recolhida"


# ── O sinal chega à API ──────────────────────────────────────────────────────

def test_api_reports_boot_problem_before_engine_starts(store, tmp_path, monkeypatch):
    """Antes do START, _storage_health não olhava a store do boot."""
    import clipbot_api
    import pending_store
    _previous_session(tmp_path)
    _break_reads(monkeypatch)
    s = pending_store.PendingStore()
    monkeypatch.setattr(clipbot_api, "_STORE", s)

    api = clipbot_api.ClipBotAPI()
    api._engine = None
    health = api._storage_health()

    assert health["ok"] is False
    assert any(p["what"] == "pending" for p in health["problems"])
