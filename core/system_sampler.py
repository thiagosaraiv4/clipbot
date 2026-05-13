"""
core/system_sampler.py — Coleta gauges de sistema em background.

A cada SAMPLE_INTERVAL segundos coleta:
  • sys.cpu_percent      (% do processo)
  • sys.rss_mb           (RAM resident)
  • sys.vms_mb           (RAM virtual)
  • sys.threads          (threads ativas)
  • sys.handles          (Windows only — detecta leak)
  • sys.disk_write_mb_s  (taxa I/O write desde último sample)
  • sys.disk_read_mb_s   (taxa I/O read)
  • sys.disk_free_gb     (livre no clips_dir — antecipa disk fill)

Custo: ~1ms por sample. Runtime <0.01% CPU para interval=10s.
Falhas silenciosas — telemetria nunca derruba o app.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from metrics import gauge

logger = logging.getLogger("ClipBot.SystemSampler")


class SystemSampler(threading.Thread):
    SAMPLE_INTERVAL = 10.0

    def __init__(self, clips_dir: Optional[Path] = None):
        super().__init__(name="system-sampler", daemon=True)
        self.clips_dir = clips_dir
        self._stop     = threading.Event()
        # psutil import lazy: instalação opcional, falha graceful
        try:
            import psutil  # type: ignore
            self._proc = psutil.Process()
            self._psutil = psutil
            self._has_io = hasattr(self._proc, "io_counters")
            self._has_handles = hasattr(self._proc, "num_handles")
            self._last_io = self._proc.io_counters() if self._has_io else None
            self._last_t  = time.monotonic()
        except (ImportError, Exception) as e:
            logger.warning(f"psutil indisponível, sampler desativado: {e}")
            self._proc = None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        if self._proc is None:
            return
        # Primeira chamada de cpu_percent retorna 0 sempre — descartar
        try:
            self._proc.cpu_percent(interval=None)
        except Exception:
            pass

        while not self._stop.wait(self.SAMPLE_INTERVAL):
            try:
                self._sample_once()
            except self._psutil.NoSuchProcess:
                break  # processo morreu, sair limpo
            except Exception as e:
                logger.debug(f"sample failed: {e}")

    def _sample_once(self) -> None:
        try:
            gauge("sys.cpu_percent", self._proc.cpu_percent(interval=None))
        except Exception: pass

        try:
            mi = self._proc.memory_info()
            gauge("sys.rss_mb", round(mi.rss / 1e6, 1))
            gauge("sys.vms_mb", round(mi.vms / 1e6, 1))
        except Exception: pass

        try:
            gauge("sys.threads", self._proc.num_threads())
        except Exception: pass

        if self._has_handles:
            try:
                gauge("sys.handles", self._proc.num_handles())
            except Exception: pass

        # Disk I/O delta
        if self._has_io and self._last_io is not None:
            try:
                cur_io = self._proc.io_counters()
                now    = time.monotonic()
                dt     = now - self._last_t
                if dt > 0:
                    gauge("sys.disk_write_mb_s",
                          round((cur_io.write_bytes - self._last_io.write_bytes) / 1e6 / dt, 2))
                    gauge("sys.disk_read_mb_s",
                          round((cur_io.read_bytes - self._last_io.read_bytes) / 1e6 / dt, 2))
                self._last_io = cur_io
                self._last_t  = now
            except Exception: pass

        # Disco livre no clips_dir
        if self.clips_dir is not None:
            try:
                import shutil
                free = shutil.disk_usage(str(self.clips_dir)).free / 1e9
                gauge("sys.disk_free_gb", round(free, 2))
            except Exception: pass


_sampler: Optional[SystemSampler] = None


def start_sampler(clips_dir: Optional[Path] = None) -> None:
    """Bootstrap: chamar 1× quando engine startar (com clips_dir conhecido)."""
    global _sampler
    if _sampler is not None and _sampler.is_alive():
        return
    _sampler = SystemSampler(clips_dir=clips_dir)
    _sampler.start()
    logger.debug(f"system sampler started (interval={SystemSampler.SAMPLE_INTERVAL}s)")


def stop_sampler() -> None:
    """Shutdown limpo. Idempotente."""
    global _sampler
    if _sampler is not None:
        _sampler.stop()
        _sampler = None
