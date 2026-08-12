"""
Regressão P0-1 — PendingStore: set_processing() bloqueia remove_and_cleanup().

Bug: `remove_and_cleanup` tem guard `if moment_id in self._processing: return`.
generate_clip_from_moment() marca o momento como processing ANTES e só limpa
no finally, DEPOIS — então os 3 call sites de remove dentro do try são no-op.

Consequência: momentos aprovados nunca saem da store nem do disco.
Evidência em produção: "remove_and_cleanup: bbfe0534 em processamento, ignorando."
                      "📋 Restaurados 12 momento(s) pendente(s)"
"""

from conftest import make_moment


def test_add_then_remove_deletes_entry_and_file(store, raw_ts):
    """Baseline (passa hoje): fluxo sem processing funciona."""
    raw = raw_ts()
    m = make_moment()
    assert store.add(m, raw, "twitch", "baiano") is True

    store.remove_and_cleanup(m.moment_id)

    assert store.get(m.moment_id) is None
    assert not raw.exists()


def test_remove_after_set_processing_removes_entry(store, raw_ts):
    """
    P0-1 (FALHA HOJE): entry DEVE sair da store mesmo marcada como processing.

    É exatamente a sequência de generate_clip_from_moment():
        set_processing → (trabalho) → remove_and_cleanup → clear_processing
    """
    raw = raw_ts()
    m = make_moment()
    store.add(m, raw, "twitch", "baiano")

    store.set_processing(m.moment_id)
    store.remove_and_cleanup(m.moment_id)
    store.clear_processing(m.moment_id)

    assert store.get(m.moment_id) is None, (
        "momento continua na store após remove_and_cleanup — "
        "UI mostra momento fantasma e permite gerar o mesmo clip N vezes"
    )
    assert len(store) == 0


def test_finish_and_cleanup_deletes_entry_and_raw_file(store, raw_ts):
    """
    P0-1 — fluxo do DONO do processamento (generate_clip_from_moment):
    ao terminar, entry e .ts precisam sair. Senão o disco enche.

    Nota de especificação: este caso e
    `test_reject_during_processing_must_not_delete_file_in_use` executam a
    MESMA sequência (set_processing → remover), mas exigem resultados opostos
    para o arquivo. A diferença é semântica — quem chama:
      • dono terminou   → finish_and_cleanup() → pode apagar
      • reject externo  → remove_and_cleanup() → preserva o arquivo em uso
    """
    raw = raw_ts()
    m = make_moment()
    store.add(m, raw, "twitch", "baiano")

    store.set_processing(m.moment_id)
    store.finish_and_cleanup(m.moment_id)

    assert store.get(m.moment_id) is None
    assert not raw.exists(), "raw .ts ficou órfão no disco"
    assert m.moment_id not in store._processing, "flag de processing não liberada"


def test_finish_and_cleanup_keeps_raw_shared_with_other_pending(store, raw_ts):
    """INVARIANTE I2 vale também no fluxo do dono."""
    raw = raw_ts()
    m1 = make_moment(start=100.0, end=115.0, reason="keyword")
    m2 = make_moment(start=200.0, end=215.0, reason="chat_flood")
    store.add(m1, raw, "twitch", "baiano")
    store.add(m2, raw, "twitch", "baiano")

    store.set_processing(m1.moment_id)
    store.finish_and_cleanup(m1.moment_id)

    assert store.get(m1.moment_id) is None
    assert raw.exists(), "apagou .ts ainda referenciado por outro pendente"


def test_raw_kept_when_shared_by_another_pending(store, raw_ts):
    """
    INVARIANTE I2 (deve continuar passando após o patch):
    dois momentos apontando para o mesmo .ts → arquivo só sai com o último.
    """
    raw = raw_ts()
    m1 = make_moment(start=100.0, end=115.0, reason="keyword")
    m2 = make_moment(start=200.0, end=215.0, reason="chat_flood")
    store.add(m1, raw, "twitch", "baiano")
    store.add(m2, raw, "twitch", "baiano")

    store.remove_and_cleanup(m1.moment_id)
    assert raw.exists(), "apagou .ts ainda referenciado por outro momento"

    store.remove_and_cleanup(m2.moment_id)
    assert not raw.exists()


def test_reject_during_processing_must_not_delete_file_in_use(store, raw_ts):
    """
    INVARIANTE I1 (razão de existir do _processing):
    um reject vindo da UI ENQUANTO o editor lê o arquivo não pode apagá-lo.

    Após o patch, a entry pode sair da store — mas o ARQUIVO deve sobreviver
    enquanto estiver marcado como em processamento.
    """
    raw = raw_ts()
    m = make_moment()
    store.add(m, raw, "twitch", "baiano")

    store.set_processing(m.moment_id)
    store.remove_and_cleanup(m.moment_id)      # "reject" concorrente

    assert raw.exists(), (
        "arquivo apagado enquanto o editor ainda o utiliza — corrupção de render"
    )


def test_persistence_roundtrip_does_not_resurrect_removed(store, raw_ts, tmp_path, monkeypatch):
    """
    P0-1 (FALHA HOJE): após remover, um reload do disco não pode ressuscitar.
    Simula fechar e reabrir o app.
    """
    import pending_store
    raw = raw_ts()
    m = make_moment()
    store.add(m, raw, "twitch", "baiano")
    store.set_processing(m.moment_id)
    store.finish_and_cleanup(m.moment_id)

    reloaded = pending_store.PendingStore()
    assert reloaded.get(m.moment_id) is None, (
        "momento ressuscitou após restart — é o 'Restaurados N momentos' crescente"
    )


def test_load_discards_entries_with_missing_raw(store, raw_ts, tmp_path):
    """Baseline (passa hoje): entries sem .ts no disco são descartadas no load."""
    import pending_store
    raw = raw_ts()
    m = make_moment()
    store.add(m, raw, "twitch", "baiano")
    raw.unlink()

    reloaded = pending_store.PendingStore()
    assert reloaded.get(m.moment_id) is None
