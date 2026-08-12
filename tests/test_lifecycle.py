"""
Regressão P1-10 — leak de atexit no AudioSpikeDetector.

Bug: `atexit.register(self._atexit_cleanup)` dentro de __init__ mantém uma
     referência forte à instância para sempre. O auto-restart do PR7 cria um
     detector NOVO a cada reconexão de canal → cada reconexão vaza um detector
     (com sua lista de momentos, session e proc).

Medido: 50 instâncias criadas e descartadas → 50 ainda vivas após gc.collect().
"""

import gc
import weakref

from conftest import detection_config


def _make_detector():
    from moment_detector import AudioSpikeDetector
    cfg = detection_config(audio_enabled=True)
    return AudioSpikeDetector(cfg, lambda: 0.0)


def test_detector_is_garbage_collected_after_release():
    """P1-10 (FALHA HOJE): detector descartado deve ser coletável."""
    det = _make_detector()
    ref = weakref.ref(det)

    del det
    gc.collect()

    assert ref() is None, (
        "AudioSpikeDetector retido após descarte — "
        "atexit.register(self.…) segura a instância; cada auto-restart vaza uma"
    )


def test_repeated_restarts_do_not_accumulate_instances():
    """P1-10 (FALHA HOJE): simula 30 reconexões de canal."""
    from moment_detector import AudioSpikeDetector

    for _ in range(30):
        d = _make_detector()
        del d
    gc.collect()

    alive = [o for o in gc.get_objects() if isinstance(o, AudioSpikeDetector)]
    assert len(alive) <= 2, (
        f"{len(alive)} detectores vivos após 30 restarts — "
        "crescimento ilimitado em sessão 24h com rede instável"
    )
