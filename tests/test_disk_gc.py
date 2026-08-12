"""
Patch 1b — GC periódico de raw pendentes órfãos.

Fecha o ciclo do Patch 1: quando um .ts está em uso no instante da remoção,
ele é preservado; este GC o recolhe depois. Também limpa restos de crashes.

Invariantes:
  • nunca apaga .ts referenciado pela PendingStore
  • nunca apaga o .ts de CAPTURA ATIVA (raw_{platform}_{canal}_{ts}.ts)
  • respeita grace period (arquivo recém-extraído pode não estar na store ainda)
"""

import os
import time

from conftest import make_moment


def _engine(store):
    import bot_engine
    return bot_engine.BotEngine(pending_store=store)


def _aged(path, seconds):
    """Envelhece o mtime do arquivo para driblar o grace period."""
    old = time.time() - seconds
    os.utime(path, (old, old))
    return path


def test_removes_orphan_pending_clip(store, tmp_path, raw_ts):
    orphan = _aged(raw_ts("raw_pending_twitch_x_20260101_000000_keyword.ts"), 3600)

    removed = _engine(store)._gc_orphan_pending_clips(tmp_path)

    assert removed == 1
    assert not orphan.exists()


def test_keeps_clip_referenced_by_store(store, tmp_path, raw_ts):
    """INVARIANTE: pendente legítimo aguardando aprovação não pode sumir."""
    raw = _aged(raw_ts("raw_pending_twitch_y_20260101_000000_keyword.ts"), 3600)
    store.add(make_moment(), raw, "twitch", "baiano")

    removed = _engine(store)._gc_orphan_pending_clips(tmp_path)

    assert removed == 0
    assert raw.exists()


def test_keeps_active_capture_file(store, tmp_path, raw_ts):
    """
    INVARIANTE CRÍTICA: o arquivo de gravação contínua NÃO casa com
    raw_pending_* e jamais pode ser apagado (mataria a captura em andamento).
    """
    active = _aged(raw_ts("raw_twitch_baiano_20260101_000000123456.ts"), 3600)

    removed = _engine(store)._gc_orphan_pending_clips(tmp_path)

    assert removed == 0
    assert active.exists()


def test_respects_grace_period_for_fresh_files(store, tmp_path, raw_ts):
    """
    INVARIANTE: arquivo recém-extraído (ainda não adicionado à store)
    sobrevive — evita corrida entre extract_segment e store.add.
    """
    fresh = raw_ts("raw_pending_twitch_z_20260101_000000_keyword.ts")

    removed = _engine(store)._gc_orphan_pending_clips(tmp_path)

    assert removed == 0
    assert fresh.exists()


def test_orphan_after_finish_is_collected(store, tmp_path, raw_ts):
    """
    Integração Patch 1 + 1b: arquivo preservado por estar em uso durante um
    reject externo é recolhido pelo GC depois.
    """
    raw = raw_ts("raw_pending_twitch_w_20260101_000000_keyword.ts")
    m = make_moment()
    store.add(m, raw, "twitch", "baiano")

    store.set_processing(m.moment_id)
    store.remove_and_cleanup(m.moment_id)     # reject externo: preserva arquivo
    store.clear_processing(m.moment_id)
    assert raw.exists()

    _aged(raw, 3600)
    removed = _engine(store)._gc_orphan_pending_clips(tmp_path)

    assert removed == 1
    assert not raw.exists()
