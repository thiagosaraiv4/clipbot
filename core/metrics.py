"""
core/metrics.py — Telemetria leve persistente para validação quantitativa de PRs.

Sinks:
  • APP_DATA/metrics/metrics-YYYY-MM-DD.jsonl   (eventos individuais, append-only)
  • APP_DATA/metrics/metrics_summary.jsonl      (snapshot agregado a cada 60s)

Tipos:
  • Timer(name, **tags) — context manager. Mede ms.
  • observe(name, value, **tags) — sample arbitrário (ms, count, ratio).
  • incr(name, n=1, **tags) — counter.
  • gauge(name, value, **tags) — valor pontual (RSS, queue depth).

Custo runtime: ~50-100µs por observação. <1% CPU para workload típico.
Falhas de I/O do sink são silenciosas — telemetria nunca quebra produção.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ClipBot.Metrics")

_APP_DATA    = Path(os.environ.get("APPDATA", Path.home())) / "ClipBot"
_METRICS_DIR = _APP_DATA / "metrics"

# ── Module-level state (single-process, lock-protected) ──────────────────────
_lock = threading.Lock()
# Histograma per-name: ring buffer 2048 valores p/ p50/p95/p99 sem dep externa
_hist:    dict     = defaultdict(lambda: deque(maxlen=2048))
_counters: dict    = defaultdict(int)
_gauges:   dict    = {}


def _events_file() -> Path:
    return _METRICS_DIR / f"metrics-{datetime.now().strftime('%Y-%m-%d')}.jsonl"


def _summary_file() -> Path:
    return _METRICS_DIR / "metrics_summary.jsonl"


def _emit(record: dict) -> None:
    """Append linha JSON ao arquivo do dia. Tolerante a I/O errors."""
    try:
        _METRICS_DIR.mkdir(parents=True, exist_ok=True)
        with _lock, open(_events_file(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def Timer(name: str, **tags):
    """
    Context manager que mede tempo entre enter/exit em ms.
    Falhas dentro do bloco AINDA são medidas (mas re-raised).

    Uso:
        with Timer("editor.render_final", platform="twitch"):
            run_ffmpeg(cmd, ...)
    """
    t0 = time.monotonic()
    try:
        yield
    finally:
        dt_ms = (time.monotonic() - t0) * 1000.0
        observe(name, dt_ms, **tags)


def observe(name: str, value: float, **tags) -> None:
    """Registra um sample (latência, throughput, ratio)."""
    with _lock:
        _hist[name].append(value)
    _emit({
        "ts":    datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "kind":  "obs",
        "name":  name,
        "value": value,
        **({"tags": tags} if tags else {}),
    })


def incr(name: str, n: int = 1, **tags) -> None:
    """Incrementa contador. Persiste só se tags ou n != 1 (reduz volume)."""
    with _lock:
        _counters[name] += n
    if tags or n != 1:
        _emit({
            "ts":    datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "kind":  "incr",
            "name":  name,
            "value": n,
            **({"tags": tags} if tags else {}),
        })


def gauge(name: str, value: float, **tags) -> None:
    """Set valor pontual (RSS, queue depth, disk free)."""
    with _lock:
        _gauges[name] = value
    _emit({
        "ts":    datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "kind":  "gauge",
        "name":  name,
        "value": value,
        **({"tags": tags} if tags else {}),
    })


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot agregado (chamado pelo Reporter periódico)
# ──────────────────────────────────────────────────────────────────────────────

def snapshot() -> dict:
    """
    Snapshot atual: percentis de cada histograma + contadores + gauges.
    NÃO reseta estado — contadores são acumulativos pela vida da sessão.
    """
    out = {
        "ts":       datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind":     "summary",
        "hist":     {},
        "counters": {},
        "gauges":   {},
    }
    with _lock:
        for name, vals in _hist.items():
            if not vals:
                continue
            sorted_vals = sorted(vals)
            n = len(sorted_vals)
            out["hist"][name] = {
                "n":    n,
                "p50":  round(sorted_vals[n // 2], 3),
                "p95":  round(sorted_vals[min(n - 1, int(n * 0.95))], 3),
                "p99":  round(sorted_vals[min(n - 1, int(n * 0.99))], 3),
                "max":  round(sorted_vals[-1], 3),
                "mean": round(sum(sorted_vals) / n, 3),
            }
        out["counters"] = dict(_counters)
        out["gauges"]   = dict(_gauges)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Reporter background
# ──────────────────────────────────────────────────────────────────────────────

class _Reporter(threading.Thread):
    def __init__(self, interval: float = 60.0):
        super().__init__(name="metrics-reporter", daemon=True)
        self.interval = interval
        self._stop    = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                snap = snapshot()
                _METRICS_DIR.mkdir(parents=True, exist_ok=True)
                with _lock, open(_summary_file(), "a", encoding="utf-8") as f:
                    f.write(json.dumps(snap, ensure_ascii=False) + "\n")
            except Exception:
                pass


_reporter: Optional[_Reporter] = None


def start_reporter(interval: float = 60.0) -> None:
    """Bootstrap: chamar 1× em app.py:main()."""
    global _reporter
    if _reporter is not None:
        return
    _reporter = _Reporter(interval)
    _reporter.start()
    logger.debug(f"metrics reporter started (interval={interval}s)")


def stop_reporter() -> None:
    """Shutdown clean. Idempotente."""
    global _reporter
    if _reporter is not None:
        _reporter.stop()
        _reporter = None
