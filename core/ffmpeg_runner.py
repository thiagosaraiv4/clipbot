"""
core/ffmpeg_runner.py — ClipBot Iter1 PR1
Wrapper central para execução de FFmpeg.

Responsabilidades (escopo deste PR):
  • Único ponto que chama ffmpeg: instrumentação consistente.
  • stderr SEMPRE capturado em ring buffer + escrito em events.jsonl em erro.
  • stdout opcional via PIPE (curto) ou exposto via FFmpegSession.proc.stdout (longo).
  • Timeout mata process tree (psutil → taskkill /T → proc.kill).
  • Cleanup garantido em exceção; FFmpegSession.close() idempotente.
  • Suprime janela console no Windows (CREATE_NO_WINDOW).

Não-objetivos (intencional, fora deste PR):
  • Sem retry. Caller decide.
  • Sem cache. Caller decide.
  • Sem asyncio. Threading puro.
  • Sem auto-detecção de path do ffmpeg. PATH ou cmd[0] absoluto.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger("ClipBot.FFmpeg")


# ── Locations ─────────────────────────────────────────────────────────────────
_APP_DATA   = Path(os.environ.get("APPDATA", Path.home())) / "ClipBot"
_EVENTS_DIR = _APP_DATA / "events"

# ── Constants ─────────────────────────────────────────────────────────────────
_STDERR_RING_LINES = 200       # tamanho do ring buffer de stderr
_STDERR_TAIL_LOG   = 30        # quantas linhas finais ir para o logger em erro
_KILL_GRACE_SEC    = 3.0       # quanto esperar entre terminate() e kill tree
_DRAIN_JOIN_SEC    = 2.0       # quanto esperar threads drainer encerrarem
_DEFAULT_ENC       = "utf-8"

# ── Module-level lock para events.jsonl (single-process I/O lock) ─────────────
_events_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FFmpegResult:
    """Resultado de uma execução curta de FFmpeg (run_ffmpeg)."""
    label:        str
    cmd:          List[str]
    returncode:   int
    stdout:       bytes        = b""
    stderr_tail:  List[str]    = field(default_factory=list)
    elapsed_sec:  float        = 0.0
    timed_out:    bool         = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


# ──────────────────────────────────────────────────────────────────────────────
# Process tree kill (Windows-aware, fallbacks em cascata)
# ──────────────────────────────────────────────────────────────────────────────

def kill_tree(proc: Optional[subprocess.Popen],
              grace: float = _KILL_GRACE_SEC) -> None:
    """
    Mata processo + descendentes. Nunca lança. Idempotente.

    Sequência:
      1. terminate() + wait(grace)            — gracioso
      2. psutil.kill() em filhos + pai        — preferencial
      3. taskkill /F /T (Windows) ou proc.kill() (POSIX)  — fallback
      4. wait(2s) final para confirmar morte  — evita zombie
    """
    if proc is None or proc.poll() is not None:
        return

    # 1) terminate gracioso
    try:
        proc.terminate()
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            pass
    except Exception:
        pass

    # 2) psutil tree kill (preferencial)
    killed_via_psutil = False
    try:
        import psutil  # type: ignore
        try:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
            try:
                parent.kill()
                killed_via_psutil = True
            except Exception:
                pass
        except psutil.NoSuchProcess:
            killed_via_psutil = True   # já morto, OK
    except ImportError:
        pass

    # 3) Fallback OS-específico
    if not killed_via_psutil:
        if sys.platform == "win32":
            try:
                # /F = force, /T = tree
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True, timeout=5,
                )
            except Exception:
                try: proc.kill()
                except Exception: pass
        else:
            try: proc.kill()
            except Exception: pass

    # 4) Confirmação final
    try:
        proc.wait(timeout=2.0)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Stream draining (thread daemon)
# ──────────────────────────────────────────────────────────────────────────────

class _StderrDrainer(threading.Thread):
    """Drena stderr linha-a-linha para um deque (ring buffer)."""

    def __init__(self, stream, *, name: str, ring: deque):
        super().__init__(name=name, daemon=True)
        self._stream = stream
        self._ring   = ring

    def run(self) -> None:
        try:
            # readline com sentinel b"" sai em EOF
            for line in iter(self._stream.readline, b""):
                if not line:
                    break
                try:
                    text = line.decode(_DEFAULT_ENC, errors="replace").rstrip()
                except Exception:
                    continue
                if text:
                    self._ring.append(text)
        except Exception:
            # stream pode ter sido fechado por kill_tree; não escala exceção
            pass
        finally:
            try: self._stream.close()
            except Exception: pass


class _StdoutSinkDrainer(threading.Thread):
    """Drena stdout em chunks para um bytearray (binary)."""

    def __init__(self, stream, *, name: str, sink: bytearray,
                 max_bytes: int = 256 * 1024 * 1024):
        super().__init__(name=name, daemon=True)
        self._stream    = stream
        self._sink      = sink
        self._max_bytes = max_bytes

    def run(self) -> None:
        try:
            while True:
                chunk = self._stream.read(65536)
                if not chunk:
                    break
                # Cap de segurança: ffmpeg que produz stdout sem fim mata processo eventualmente
                if len(self._sink) + len(chunk) > self._max_bytes:
                    self._sink.extend(chunk[: self._max_bytes - len(self._sink)])
                    break
                self._sink.extend(chunk)
        except Exception:
            pass
        finally:
            try: self._stream.close()
            except Exception: pass


# ──────────────────────────────────────────────────────────────────────────────
# events.jsonl (rotated daily, append-only, fault-tolerant)
# ──────────────────────────────────────────────────────────────────────────────

def _events_file_for_today() -> Path:
    return _EVENTS_DIR / f"events-{datetime.now().strftime('%Y-%m-%d')}.jsonl"


def _emit_event(level: str, label: str, msg: str, **extra) -> None:
    """Append linha JSON em events.jsonl. Tolerante a I/O errors."""
    event = {
        "ts":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "level":  level,
        "source": "ffmpeg",
        "label":  label,
        "msg":    msg,
    }
    if extra:
        event.update(extra)

    try:
        _EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        with _events_lock:
            with open(_events_file_for_today(), "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        # Logging do logging não pode quebrar produção
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Popen flags compartilhados
# ──────────────────────────────────────────────────────────────────────────────

def _popen_kwargs(*, capture_stdout: bool) -> dict:
    """Defaults consistentes para qualquer Popen FFmpeg deste módulo."""
    kw = {
        "stdin":    subprocess.DEVNULL,
        "stdout":   subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        "stderr":   subprocess.PIPE,
        "close_fds": True,           # evita herança de handles inesperada
    }
    if sys.platform == "win32":
        # Sem janela console pop-up (importante em GUI app pywebview)
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kw


# ──────────────────────────────────────────────────────────────────────────────
# Public API: run_ffmpeg (curto, síncrono, com timeout)
# ──────────────────────────────────────────────────────────────────────────────

def run_ffmpeg(
    cmd:            Sequence[str],
    *,
    label:          str,
    timeout:        Optional[float] = None,
    capture_stdout: bool = False,
    log_command:    bool = False,
    quiet:          bool = False,
) -> FFmpegResult:
    """
    Executa ffmpeg sincronamente com observabilidade.

    Args:
        cmd:            lista de args. cmd[0] tipicamente "ffmpeg".
        label:          identificador para logs ("render_final", "motion_score", ...).
        timeout:        segundos antes de matar tree. None = sem timeout.
        capture_stdout: True → stdout em result.stdout. False → DEVNULL.
        log_command:    True → log do cmd completo (verbose).

    Returns:
        FFmpegResult (não levanta em rc != 0; caller checa result.ok).

    Failure modes cobertos:
        • ffmpeg não encontrado          → returncode=-127, log + event ERROR.
        • timeout                        → kill tree, timed_out=True.
        • stderr flood (>200 linhas)     → ring buffer mantém últimas 200.
        • stdout grande sem capture      → DEVNULL, sem alocação.
    """
    cmd_list = list(cmd)
    started  = time.monotonic()

    if log_command:
        logger.debug(f"[FFmpeg][{label}] cmd: {' '.join(cmd_list)}")
    if not quiet:
        logger.info(f"[FFmpeg][{label}] started")
    # quiet=True: silêncio total no start; só erros aparecem

    # ── Spawn ─────────────────────────────────────────────────────────────────
    spawn_t0 = time.monotonic()
    try:
        proc = subprocess.Popen(cmd_list, **_popen_kwargs(capture_stdout=capture_stdout))
    except FileNotFoundError as e:
        logger.error(f"[FFmpeg][{label}] FATAL: executável não encontrado: {e}")
        _emit_event("ERROR", label, "ffmpeg executable not found",
                    cmd_head=cmd_list[:1])
        return FFmpegResult(
            label=label, cmd=cmd_list, returncode=-127,
            elapsed_sec=0.0, timed_out=False,
        )
    except OSError as e:
        logger.error(f"[FFmpeg][{label}] FATAL: spawn falhou: {e}")
        _emit_event("ERROR", label, f"spawn failed: {e}",
                    cmd_head=cmd_list[:8])
        return FFmpegResult(
            label=label, cmd=cmd_list, returncode=-1,
            elapsed_sec=0.0, timed_out=False,
        )

    # ── Métrica de spawn latency (Windows tipico ~80-150ms) ───────────────────
    spawn_ms = (time.monotonic() - spawn_t0) * 1000.0
    try:
        from metrics import observe as _metrics_observe
        _metrics_observe("ffmpeg.spawn_ms", spawn_ms, label=label)
    except Exception:
        pass

    # ── Drainers ──────────────────────────────────────────────────────────────
    stderr_ring: deque = deque(maxlen=_STDERR_RING_LINES)
    err_drainer = _StderrDrainer(
        proc.stderr, name=f"ffmpeg-stderr-{label}", ring=stderr_ring,
    )
    err_drainer.start()

    stdout_sink: Optional[bytearray] = None
    out_drainer: Optional[_StdoutSinkDrainer] = None
    if capture_stdout:
        stdout_sink = bytearray()
        out_drainer = _StdoutSinkDrainer(
            proc.stdout, name=f"ffmpeg-stdout-{label}", sink=stdout_sink,
        )
        out_drainer.start()

    # ── Wait ──────────────────────────────────────────────────────────────────
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        logger.warning(f"[FFmpeg][{label}] TIMEOUT após {timeout}s — matando tree")
        kill_tree(proc)

    # Aguarda drainers fecharem (proc encerrou, streams EOF)
    err_drainer.join(timeout=_DRAIN_JOIN_SEC)
    if out_drainer is not None:
        out_drainer.join(timeout=_DRAIN_JOIN_SEC)

    elapsed = time.monotonic() - started
    rc      = proc.returncode if proc.returncode is not None else -1

    # Métrica: tempo total da execução (incluindo wait + drain)
    try:
        from metrics import observe as _metrics_observe, incr as _metrics_incr
        _metrics_observe("ffmpeg.elapsed_ms", elapsed * 1000.0,
                         label=label, ok=(rc == 0 and not timed_out))
        if rc != 0 or timed_out:
            _metrics_incr("ffmpeg.failures", label=label,
                          reason="timeout" if timed_out else f"rc{rc}")
    except Exception:
        pass

    result = FFmpegResult(
        label       = label,
        cmd         = cmd_list,
        returncode  = rc,
        stdout      = bytes(stdout_sink) if stdout_sink is not None else b"",
        stderr_tail = list(stderr_ring),
        elapsed_sec = elapsed,
        timed_out   = timed_out,
    )

    # ── Logging final ─────────────────────────────────────────────────────────
    if result.ok:
        if not quiet:
            logger.info(f"[FFmpeg][{label}] exited rc=0 elapsed={elapsed:.1f}s")
        # quiet=True: silêncio total em sucesso
    else:
        kind = "TIMEOUT" if timed_out else "FAILED"
        # quiet=True silencia logger; events.jsonl ainda registra tudo
        if not quiet:
            logger.error(f"[FFmpeg][{label}] {kind} rc={rc} elapsed={elapsed:.1f}s")
            tail = result.stderr_tail[-_STDERR_TAIL_LOG:]
            if tail:
                logger.error(f"[FFmpeg][{label}] last stderr lines:")
                for line in tail:
                    logger.error(f"  {line}")
        _emit_event(
            "ERROR" if not timed_out else "WARNING",
            label,
            f"ffmpeg {'timeout' if timed_out else 'failed'} rc={rc}",
            returncode  = rc,
            timed_out   = timed_out,
            elapsed_sec = round(elapsed, 2),
            stderr_tail = result.stderr_tail,   # tail completo (≤200) no event
            cmd_head    = cmd_list[:8],
        )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Public API: FFmpegSession (long-running, capture/streaming)
# ──────────────────────────────────────────────────────────────────────────────

class FFmpegSession:
    """
    FFmpeg de longa duração (stream capture, audio pipe, etc.).

    Uso típico (capture):
        with FFmpegSession(cmd, label="capture_twitch", capture_stdout=False) as s:
            ...                   # processo roda em background; stderr drenado
        # __exit__ chama close()

    Uso típico (audio spike pipe):
        s = FFmpegSession(cmd, label="audio_spike", capture_stdout=True)
        s.start()
        while ...:
            chunk = s.proc.stdout.read(N)   # caller consome stdout
        s.close()

    Garantias:
        • stderr SEMPRE drenado em thread daemon — sem deadlock por buffer cheio.
        • close() é idempotente. Chame quantas vezes quiser.
        • close() mata process tree se ainda vivo.
    """

    def __init__(self, cmd: Sequence[str], *,
                 label:          str,
                 capture_stdout: bool = True,
                 quiet:          bool = False):
        """
        quiet=True: logs de start/end com rc=0 ficam em DEBUG (não INFO).
        Use para sessions recicladas com alta frequência (ex: audio_spike a
        cada ~3s) que poluiriam o log do usuário. Erros/warnings continuam INFO+.
        """
        self.cmd            = list(cmd)
        self.label          = label
        self.capture_stdout = capture_stdout
        self.quiet          = quiet
        self.proc:          Optional[subprocess.Popen]   = None
        self.stderr_ring:   deque                        = deque(maxlen=_STDERR_RING_LINES)
        self._err_thread:   Optional[_StderrDrainer]     = None
        self._started_at:   float                        = 0.0
        self._closed:       bool                         = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> "FFmpegSession":
        if self.proc is not None:
            return self    # idempotent start

        if not self.quiet:
            logger.info(f"[FFmpeg][{self.label}] session starting")
        # quiet=True: silêncio total no start; só erros aparecem
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                **_popen_kwargs(capture_stdout=self.capture_stdout),
            )
        except (FileNotFoundError, OSError) as e:
            logger.error(f"[FFmpeg][{self.label}] FATAL spawn: {e}")
            _emit_event("ERROR", self.label, f"session spawn failed: {e}",
                        cmd_head=self.cmd[:8])
            raise

        self._started_at = time.monotonic()
        self._err_thread = _StderrDrainer(
            self.proc.stderr,
            name=f"ffmpeg-stderr-{self.label}",
            ring=self.stderr_ring,
        )
        self._err_thread.start()
        return self

    def __enter__(self) -> "FFmpegSession":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self, *, grace: float = _KILL_GRACE_SEC) -> int:
        """
        Encerra sessão. Idempotente.
        Se processo já saiu por conta própria, apenas drena estado.
        Se ainda vivo, kill_tree.
        """
        if self._closed:
            return self.proc.returncode if self.proc else -1
        self._closed = True

        if self.proc is None:
            return -1

        elapsed = time.monotonic() - self._started_at

        # Se ainda rodando, kill
        if self.proc.poll() is None:
            kill_tree(self.proc, grace=grace)

        # Aguarda drainer drenar streams já fechadas
        if self._err_thread is not None:
            self._err_thread.join(timeout=_DRAIN_JOIN_SEC)

        rc = self.proc.returncode if self.proc.returncode is not None else -1

        if rc == 0:
            if not self.quiet:
                logger.info(f"[FFmpeg][{self.label}] session ended rc=0 elapsed={elapsed:.1f}s")
            # quiet=True: silêncio total em sucesso
        else:
            tail = list(self.stderr_ring)[-_STDERR_TAIL_LOG:]
            # quiet=True silencia logger; events.jsonl ainda registra tudo
            if not self.quiet:
                logger.warning(f"[FFmpeg][{self.label}] session ended rc={rc} elapsed={elapsed:.1f}s")
                if tail:
                    logger.warning(f"[FFmpeg][{self.label}] last stderr lines:")
                    for line in tail:
                        logger.warning(f"  {line}")
            _emit_event(
                "WARNING", self.label, f"session exited rc={rc}",
                returncode  = rc,
                elapsed_sec = round(elapsed, 1),
                stderr_tail = list(self.stderr_ring),
                cmd_head    = self.cmd[:8],
            )
        return rc

    # ── Status ────────────────────────────────────────────────────────────────

    @property
    def returncode(self) -> Optional[int]:
        return self.proc.returncode if self.proc else None

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def stdout(self):
        """stdout pipe (se capture_stdout=True). None caso contrário."""
        return self.proc.stdout if self.proc else None
