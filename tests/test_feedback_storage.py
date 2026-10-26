"""
Lote 1 / A5 — feedback.py repetia o incidente de 12/08.

Pesos aprendidos (feedback_weights.json) e histórico por canal
(channel_history.json) faziam read-modify-write sem distinguir "não existe" de
"não consegui ler":

  load_weights()         → exists() (mente no Windows em PermissionError) ou
                           exceção na leitura → DEFAULTS
  apply_feedback()       → defaults + 1 ajuste → save → pesos aprendidos apagados

  _load_channel_history()→ exceção → {} (e ainda cacheava o {} por 5s)
  update_channel_history → {} + 1 canal → save → histórico de todos os canais apagado

Sem lock, dois cliques rápidos de feedback ou dois "Gerar" simultâneos também
perdiam um dos updates. E o cache devolvia cópia rasa: o update mutava os
dicts internos do próprio cache antes de gravar.

Contrato:
  • leitura ilegível → nada é gravado por cima (o ajuste é registrado no log
    imutável para não perder o sinal do usuário);
  • corrompido → quarentena e recomeça dos defaults;
  • read-modify-write serializado e decidido pela leitura feita ali dentro.
"""

import json
import threading

import pytest


LEARNED = {"keyword": 1.42, "chat_flood": 0.61}
HISTORY = {
    "baiano": {"total_clips": 7, "total_score": 4.2, "reason_counts": {"keyword": 7},
               "last_clip_ts": None, "best_reasons": ["keyword"], "avg_score": 0.6},
    "kosky":  {"total_clips": 3, "total_score": 1.5, "reason_counts": {"chat_flood": 3},
               "last_clip_ts": None, "best_reasons": ["chat_flood"], "avg_score": 0.5},
}


def _break_reads(monkeypatch, exc=PermissionError("acesso negado (AV)")):
    from pathlib import Path
    orig = Path.read_text
    broken = {"on": True}

    def maybe(self, *a, **kw):
        if broken["on"]:
            raise exc
        return orig(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", maybe)
    return lambda: broken.update(on=False)


@pytest.fixture
def fb(tmp_path, monkeypatch):
    import feedback as f
    monkeypatch.setattr(f, "APP_DATA", tmp_path)
    monkeypatch.setattr(f, "WEIGHTS_FILE", tmp_path / "feedback_weights.json")
    monkeypatch.setattr(f, "FEEDBACK_LOG", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(f, "CHANNEL_HISTORY", tmp_path / "channel_history.json")
    monkeypatch.setattr(f, "_channel_cache", {})
    monkeypatch.setattr(f, "_channel_cache_ts", 0.0)
    return f


def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ── Pesos ────────────────────────────────────────────────────────────────────

def test_unreadable_weights_are_not_overwritten(fb, monkeypatch):
    """O cenário do relatório: 1.42/0.61 viravam 1.0/1.0 no disco."""
    _write(fb.WEIGHTS_FILE, LEARNED)
    heal = _break_reads(monkeypatch)

    fb.apply_feedback("keyword", "top", 0.5)

    heal()
    disk = _read(fb.WEIGHTS_FILE)
    assert disk["keyword"] == 1.42 and disk["chat_flood"] == 0.61


def test_exists_lying_does_not_reset_weights(fb, monkeypatch):
    """No Windows, Path.exists() devolve False em PermissionError."""
    from pathlib import Path
    _write(fb.WEIGHTS_FILE, LEARNED)
    heal = _break_reads(monkeypatch)
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: False)

    fb.apply_feedback("keyword", "top", 0.5)

    heal()
    monkeypatch.setattr(Path, "exists", real_exists)
    assert _read(fb.WEIGHTS_FILE)["chat_flood"] == 0.61


def test_feedback_signal_is_logged_even_when_weights_unreadable(fb, monkeypatch):
    """
    O clique do usuário é informação: vai para o log imutável (sem pesos),
    para poder ser reaplicado depois. Só o arquivo de pesos não é tocado.
    """
    _write(fb.WEIGHTS_FILE, LEARNED)
    heal = _break_reads(monkeypatch)

    fb.apply_feedback("keyword", "reject", 0.7)

    heal()
    line = json.loads(fb.FEEDBACK_LOG.read_text(encoding="utf-8").splitlines()[-1])
    assert line["label"] == "reject" and line["reason"] == "keyword"
    assert line["new_weight"] is None


def test_corrupt_weights_are_quarantined_and_feedback_applies(fb):
    fb.WEIGHTS_FILE.write_text("{lixo", encoding="utf-8")

    fb.apply_feedback("keyword", "top", 0.5)

    assert _read(fb.WEIGHTS_FILE)["keyword"] > 1.0
    assert list(fb.WEIGHTS_FILE.parent.glob("feedback_weights.json.corrupt*"))


def test_healthy_feedback_still_updates(fb):
    """Baseline: o caminho normal continua aprendendo."""
    _write(fb.WEIGHTS_FILE, LEARNED)

    fb.apply_feedback("keyword", "top", 0.5)

    disk = _read(fb.WEIGHTS_FILE)
    assert disk["keyword"] == pytest.approx(1.42 + 0.05 * 0.5)
    assert disk["chat_flood"] == 0.61


def test_concurrent_feedback_does_not_lose_updates(fb):
    """Sem lock, dois read-modify-write simultâneos perdem um dos ajustes."""
    def worker(prefix):
        for i in range(20):
            fb.apply_feedback(f"{prefix}_{i}", "top", 0.5)

    ts = [threading.Thread(target=worker, args=(p,)) for p in ("a", "b")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    disk = _read(fb.WEIGHTS_FILE)
    faltando = [f"{p}_{i}" for p in "ab" for i in range(20) if f"{p}_{i}" not in disk]
    assert faltando == [], f"{len(faltando)} ajuste(s) perdido(s)"


# ── Histórico por canal ──────────────────────────────────────────────────────

def test_unreadable_history_is_not_overwritten(fb, monkeypatch):
    """O cenário do relatório: ficava só ['novo_canal']."""
    _write(fb.CHANNEL_HISTORY, HISTORY)
    heal = _break_reads(monkeypatch)

    fb.update_channel_history("novo_canal", "keyword", 0.5)

    heal()
    assert set(_read(fb.CHANNEL_HISTORY)) == {"baiano", "kosky"}


def test_failed_history_read_is_not_cached_as_empty(fb, monkeypatch):
    """Antes, o {} da falha ficava no cache por 5s e alimentava mais saves."""
    _write(fb.CHANNEL_HISTORY, HISTORY)
    heal = _break_reads(monkeypatch)
    assert fb._load_channel_history() == {}

    heal()
    assert set(fb._load_channel_history()) == {"baiano", "kosky"}


def test_corrupt_history_is_quarantined_and_update_applies(fb):
    fb.CHANNEL_HISTORY.write_text("{lixo", encoding="utf-8")

    fb.update_channel_history("baiano", "keyword", 0.5)

    assert _read(fb.CHANNEL_HISTORY)["baiano"]["total_clips"] == 1
    assert list(fb.CHANNEL_HISTORY.parent.glob("channel_history.json.corrupt*"))


def test_concurrent_history_updates_do_not_lose_counts(fb):
    def worker():
        for _ in range(25):
            fb.update_channel_history("baiano", "keyword", 0.5)

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert _read(fb.CHANNEL_HISTORY)["baiano"]["total_clips"] == 100


def test_stats_snapshot_is_isolated_from_cache(fb):
    """Mutar o que get_channel_stats devolve não pode alterar o cache."""
    _write(fb.CHANNEL_HISTORY, HISTORY)

    stats = fb.get_channel_stats("baiano")
    stats["total_clips"] = 999
    stats["reason_counts"]["keyword"] = 999

    again = fb.get_channel_stats("baiano")
    assert again["total_clips"] == 7
    assert again["reason_counts"]["keyword"] == 7


def test_healthy_history_update_keeps_other_channels(fb):
    _write(fb.CHANNEL_HISTORY, HISTORY)

    fb.update_channel_history("baiano", "keyword", 0.5)

    disk = _read(fb.CHANNEL_HISTORY)
    assert disk["baiano"]["total_clips"] == 8
    assert disk["kosky"] == HISTORY["kosky"]
