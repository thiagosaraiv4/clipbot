"""
Regressão P1-6 — max_clip_duration nunca é aplicado.

Bug: "max_clip_duration" existe no config (bot_engine.py:166) e NUNCA é lido.
     _merge_moments encadeia momentos próximos (MERGE_DIST=15s); com chat
     contínuo o grupo cresce indefinidamente.

Evidência em produção: [super_moment] 0.0s→81.9s  e  202.9s→338.9s (136s).
Efeito: render 4-10× mais caro, Reels rejeita >90s, retenção péssima.
"""

import pytest

from conftest import make_moment

MAX_DUR = 60.0   # = clip_max_sec default (credentials.py:91)


def _chain(n, step=10.0, span=30.0, reason="chat_flood"):
    """n momentos encadeados a cada `step`s — simula flood contínuo."""
    return [
        make_moment(start=100.0 + i * step, end=100.0 + i * step + span, reason=reason)
        for i in range(n)
    ]


def test_merged_moment_respects_max_duration():
    """P1-6 (FALHA HOJE): o merge não pode produzir clip maior que o teto."""
    from moment_detector import MomentDetector

    merged = MomentDetector._merge_moments(_chain(20))

    longest = max(m.duration for m in merged)
    assert longest <= MAX_DUR, (
        f"momento de {longest:.0f}s (teto={MAX_DUR:.0f}s) — "
        "flood contínuo faz o grupo crescer sem limite"
    )


def test_long_flood_is_split_not_dropped():
    """
    INVARIANTE: aplicar o teto não pode DESCARTAR o evento —
    deve resultar em pelo menos um momento utilizável.
    """
    from moment_detector import MomentDetector

    merged = MomentDetector._merge_moments(_chain(20))

    assert len(merged) >= 1
    assert all(m.duration > 0 for m in merged)


def test_super_moment_still_created_for_heterogeneous_group():
    """INVARIANTE: reasons distintas próximas continuam virando super_moment."""
    from moment_detector import MomentDetector

    ms = [
        make_moment(start=100.0, end=120.0, reason="audio_spike"),
        make_moment(start=105.0, end=125.0, reason="chat_flood"),
    ]
    merged = MomentDetector._merge_moments(ms)

    assert len(merged) == 1
    assert merged[0].reason == "super_moment"
    assert merged[0].extra.get("reasons") == ["audio_spike", "chat_flood"]


def test_distant_moments_are_not_merged():
    """INVARIANTE: momentos além de MERGE_DIST continuam separados."""
    from moment_detector import MomentDetector

    ms = [
        make_moment(start=100.0, end=115.0, reason="keyword"),
        make_moment(start=400.0, end=415.0, reason="keyword"),
    ]
    merged = MomentDetector._merge_moments(ms)

    assert len(merged) == 2


def test_single_moment_passthrough():
    """INVARIANTE: 1 momento entra, o MESMO objeto sai (sem re-hash de id)."""
    from moment_detector import MomentDetector

    m = make_moment()
    merged = MomentDetector._merge_moments([m])

    assert len(merged) == 1
    assert merged[0].moment_id == m.moment_id
