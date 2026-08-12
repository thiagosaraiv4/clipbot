"""
Regressão P1-5 — dedupe de chat_flood morto por incompatibilidade aritmética.

Bug: _add_moment compara `abs(m.end_sec - csec) < 15`, mas
     end_sec = csec_anterior + buffer_after, e _build_config gera
     buffer_after = 20  →  |20| > 15  →  NUNCA deduplica.

Medido: 60 mensagens em 1s → 38 momentos com IDs distintos.
Efeito: inflação da lista de momentos, spam de log que satura a fila da UI,
        e múltiplos pendings/clips do MESMO evento.
"""

import pytest

from conftest import detection_config


def _detector(buffer_after=20, elapsed_ref=None):
    from moment_detector import TwitchChatDetector
    cfg = detection_config(buffer_after=buffer_after)
    ref = elapsed_ref if elapsed_ref is not None else [100.0]
    return TwitchChatDetector(cfg, lambda: ref[0]), ref


def _burst(det, ref, n_msgs=60, wall_seconds=1.0, msg="kkkkkkkk"):
    """Simula rajada de chat: n mensagens distribuídas em wall_seconds."""
    for _ in range(n_msgs):
        ref[0] += wall_seconds / n_msgs
        det._process_message(msg)


def test_flood_burst_does_not_explode_moment_count():
    """
    P1-5 (FALHA HOJE): 1 segundo de flood é 1 evento, não dezenas.
    Tolerância generosa (<=3) para não travar a implementação do patch.
    """
    det, ref = _detector()
    _burst(det, ref)

    moments = det.get_moments_snapshot()
    flood = [m for m in moments if m.reason == "chat_flood"]

    assert len(flood) <= 3, (
        f"{len(flood)} momentos chat_flood para 1s de rajada — "
        "dedupe inoperante (buffer_after=20 > janela hardcoded de 15s)"
    )


def test_keyword_burst_does_not_explode_moment_count():
    """P1-5: mesma classe de bug atinge keyword (janela dos últimos 5 estoura)."""
    det, ref = _detector()
    _burst(det, ref)

    kw = [m for m in det.get_moments_snapshot() if m.reason == "keyword"]
    assert len(kw) <= 3, f"{len(kw)} momentos keyword para 1s de rajada"


@pytest.mark.parametrize("buffer_after", [5, 10, 20, 30, 45])
def test_dedupe_holds_for_any_buffer_after(buffer_after):
    """
    Pega a CLASSE do bug, não a instância: o dedupe não pode depender de
    buffer_after ser menor que uma constante hardcoded.
    """
    det, ref = _detector(buffer_after=buffer_after)
    _burst(det, ref)

    flood = [m for m in det.get_moments_snapshot() if m.reason == "chat_flood"]
    assert len(flood) <= 3, (
        f"buffer_after={buffer_after} → {len(flood)} momentos "
        "(dedupe sensível ao valor do buffer)"
    )


def test_distinct_events_are_not_merged_away():
    """
    INVARIANTE: dois floods separados por bastante tempo continuam sendo
    DOIS momentos. O patch não pode deduplicar demais.
    """
    det, ref = _detector()
    _burst(det, ref, n_msgs=40, wall_seconds=1.0)

    ref[0] += 600.0                       # 10 minutos depois
    _burst(det, ref, n_msgs=40, wall_seconds=1.0)

    flood = [m for m in det.get_moments_snapshot() if m.reason == "chat_flood"]
    assert len(flood) >= 2, "eventos distantes foram colapsados — perda de clips"
