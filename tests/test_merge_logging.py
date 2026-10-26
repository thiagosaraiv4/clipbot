"""
Item 4 — `⚡ Super moment` reaparecendo a cada tick.

Mecanismo: `all_moments` é uma *property* que recomputa o merge a cada chamada,
e `bot_engine._monitor_platform` a chama uma vez por segundo
(`self._stop_event.wait(timeout=1)`). Os momentos, porém, vivem até
MOMENT_TTL_SEC = 300s nos detectores. Resultado: o mesmo evento é anunciado
~300 vezes, porque `_collapse` — uma função ESTÁTICA e PURA — escreve log.

É a confusão de camadas que importa aqui: log é registro de um evento que
aconteceu; `_collapse` apenas deriva uma visão a partir de estado existente.
Recomputar uma visão não é um evento. O mesmo vale para a linha
`✂️ Momento truncado`, que repetia pelo mesmo motivo.

Custo real: satura a fila de log da UI (o painel do app mostra DEBUG) e torna
o log inútil para diagnóstico — foi por isso que o funil do Patch 10 teve de
ser inventado para responder "por que não saiu clip?".

Contrato fixado: as funções de merge são silenciosas; o anúncio acontece uma
vez por evento distinto, e o RESULTADO do merge não muda por causa disso.
"""

import logging

import pytest

from conftest import detection_config, make_moment


SUPER = "Super moment"
TRUNC = "Momento truncado"


class _Source:
    """Detector dublê: devolve sempre o mesmo snapshot, como na vida real."""

    def __init__(self, moments):
        self._moments = list(moments)

    def get_moments_snapshot(self):
        return list(self._moments)


def _detector(moments, max_duration=None):
    """MomentDetector com as fontes substituídas por dublês (nada é iniciado)."""
    from moment_detector import MomentDetector
    det = MomentDetector(detection_config(audio_enabled=False), lambda: 500.0,
                         platform="twitch", channel="kosky")
    det.audio = _Source(moments)
    det.chat = det.sim = det.manual = None
    if max_duration is not None:
        det._max_duration = max_duration
    return det


def _super_event():
    """Dois reasons próximos → grupo heterogêneo → super_moment."""
    return [
        make_moment(start=100.0, end=115.0, reason="audio_spike", score=0.6),
        make_moment(start=110.0, end=125.0, reason="chat_flood", score=0.5),
    ]


def _count(caplog, needle):
    return sum(1 for r in caplog.records if needle in r.getMessage())


# ── O bug ────────────────────────────────────────────────────────────────────

def test_super_moment_is_announced_once_across_many_ticks(caplog):
    """300 ticks do mesmo evento = 1 anúncio, não 300."""
    det = _detector(_super_event())

    with caplog.at_level(logging.DEBUG):
        for _ in range(300):
            det.all_moments

    assert _count(caplog, SUPER) == 1, (
        f"{_count(caplog, SUPER)} linhas para um único evento"
    )


def test_truncation_is_announced_once_across_many_ticks(caplog):
    """`✂️ Momento truncado` repetia pelo mesmo motivo."""
    det = _detector(
        [
            make_moment(start=100.0, end=160.0, reason="audio_spike", score=0.9),
            make_moment(start=170.0, end=400.0, reason="chat_flood", score=0.4),
        ],
        max_duration=60.0,
    )

    with caplog.at_level(logging.DEBUG):
        for _ in range(50):
            det.all_moments

    assert _count(caplog, TRUNC) == 1


def test_merge_functions_are_silent(caplog):
    """
    `_merge_moments`/`_collapse` são estáticas e puras: chamá-las diretamente
    não pode emitir log algum. É aqui que a camada errada foi corrigida.
    """
    from moment_detector import MomentDetector

    with caplog.at_level(logging.DEBUG):
        merged = MomentDetector._merge_moments(_super_event(), max_duration=60.0)

    assert merged[0].reason == "super_moment"
    assert caplog.records == [], (
        f"função pura emitiu log: {[r.getMessage() for r in caplog.records]}"
    )


# ── Invariantes: o dedupe é só de LOG ────────────────────────────────────────

def test_merge_result_is_identical_across_ticks():
    """
    INVARIANTE: silenciar o log não pode mudar o que o merge devolve — o engine
    depende desta lista para gerar clips.
    """
    det = _detector(_super_event())

    primeiro = det.all_moments
    for _ in range(20):
        atual = det.all_moments

    assert len(atual) == len(primeiro) == 1
    assert atual[0].moment_id == primeiro[0].moment_id
    assert atual[0].start_sec == primeiro[0].start_sec
    assert atual[0].end_sec == primeiro[0].end_sec
    assert atual[0].extra == primeiro[0].extra


def test_distinct_events_are_each_announced(caplog):
    """Dois eventos separados são dois anúncios — não deduplicar demais."""
    moments = _super_event() + [
        make_moment(start=900.0, end=915.0, reason="audio_spike", score=0.6),
        make_moment(start=910.0, end=925.0, reason="keyword", score=0.5),
    ]
    det = _detector(moments)

    with caplog.at_level(logging.DEBUG):
        for _ in range(10):
            det.all_moments

    assert _count(caplog, SUPER) == 2


def test_homogeneous_group_is_not_announced_as_super(caplog):
    """Grupo de um só reason não é super moment."""
    det = _detector([
        make_moment(start=100.0, end=115.0, reason="chat_flood"),
        make_moment(start=110.0, end=125.0, reason="chat_flood"),
    ])

    with caplog.at_level(logging.DEBUG):
        for _ in range(10):
            det.all_moments

    assert _count(caplog, SUPER) == 0


def test_single_moment_passthrough_is_not_announced(caplog):
    det = _detector([make_moment(start=100.0, end=115.0, reason="keyword")])

    with caplog.at_level(logging.DEBUG):
        for _ in range(10):
            det.all_moments

    assert _count(caplog, SUPER) == 0
    assert _count(caplog, TRUNC) == 0


def test_growing_event_is_not_re_announced(caplog):
    """
    Caso realista: durante um flood o grupo cresce pelo fim a cada tick, mudando
    o end_sec (e o moment_id) do momento fundido. Isso não pode virar uma linha
    nova por tick — é o MESMO evento crescendo.
    """
    src = _Source(_super_event())
    det = _detector([])
    det.audio = src

    with caplog.at_level(logging.DEBUG):
        for i in range(30):
            src._moments.append(
                make_moment(start=120.0 + i, end=135.0 + i,
                            reason="chat_flood", score=0.1)
            )
            det.all_moments

    assert _count(caplog, SUPER) == 1, (
        f"{_count(caplog, SUPER)} anúncios para um evento em crescimento"
    )


def test_growing_truncation_is_not_re_announced(caplog):
    """
    Mesma classe do teste acima, para o truncamento: num flood o span cresce a
    cada tick, então uma chave que incluísse o span renderia uma linha nova por
    crescimento. É a mesma decisão sobre o mesmo evento.
    """
    src = _Source([
        make_moment(start=100.0, end=160.0, reason="audio_spike", score=0.9),
        make_moment(start=170.0, end=400.0, reason="chat_flood", score=0.4),
    ])
    det = _detector([], max_duration=60.0)
    det.audio = src

    with caplog.at_level(logging.DEBUG):
        for i in range(30):
            src._moments.append(
                make_moment(start=400.0 + i * 5, end=415.0 + i * 5,
                            reason="chat_flood", score=0.1)
            )
            det.all_moments

    assert _count(caplog, TRUNC) == 1, (
        f"{_count(caplog, TRUNC)} anúncios de truncamento para um evento"
    )


# ── Memória ──────────────────────────────────────────────────────────────────

def test_announce_cache_stays_bounded():
    """O cache de anúncios não pode crescer sem limite em sessões de 24h."""
    det = _detector([])
    src = _Source([])
    det.audio = src

    for i in range(600):
        base = i * 1000.0
        src._moments = [
            make_moment(start=base, end=base + 15.0, reason="audio_spike"),
            make_moment(start=base + 10.0, end=base + 25.0, reason="chat_flood"),
        ]
        det.all_moments

    assert len(det._announced) <= 256, f"cache com {len(det._announced)} entradas"
