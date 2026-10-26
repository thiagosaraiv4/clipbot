"""
detection/moment_detector.py  — ClipBot v2.4

Garantias de concorrência:
  • self._lock (privado) em cada detector; nunca expõe self.moments diretamente.
  • get_moments_snapshot() — única forma de ler moments de fora: cópia sob lock.
  • AudioSpikeDetector: thread leitora + queue.Queue (cross-platform, incl. Windows).
    Se nenhum chunk chegar em >IO_STALL_SEC → mata FFmpeg e reinicia.
    Antes de relançar a thread leitora: join(2s) + drain da fila.
  • _merge_super_moments: sweep-line por overlap OU <=MERGE_DIST s de distância.
    Grupos heterogêneos → reason="super_moment"; homogêneos → reason original.
    score = min(1.0, soma). extra contém reasons=[] e sources=[].
"""

import time
import queue
import random
import socket
import ssl
import struct
import atexit
import hashlib
import logging
import threading
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from pathlib import Path

from ffmpeg_runner import FFmpegSession

# Telemetria: import lazy/tolerante
try:
    from metrics import incr as _metrics_incr, observe as _metrics_observe
except ImportError:
    def _metrics_incr(*_a, **_k): pass
    def _metrics_observe(*_a, **_k): pass

logger = logging.getLogger("ClipBot.Detection")


# ──────────────────────────────────────────────────────────────────────────────
# Utilitário global: kill robusto  (terminate → wait → kill)
# ──────────────────────────────────────────────────────────────────────────────

def _kill_proc(proc, timeout: float = 3.0) -> None:
    """Encerra subprocess.Popen sem lançar exceção. Seguro para atexit/finally."""
    if proc is None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
            return
        except Exception:
            pass
        proc.kill()
        try:
            proc.wait(timeout=2.0)
        except Exception:
            pass
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Cleanup global de detectores de áudio (registro ÚNICO no import do módulo)
#
# P1-10: antes, cada AudioSpikeDetector fazia atexit.register(self._atexit_cleanup)
# no __init__. atexit guarda referência FORTE → a instância nunca era coletada.
# Com o auto-restart por canal (PR7), cada reconexão vazava um detector com sua
# lista de momentos, session e proc. Medido: 30 restarts → 30 instâncias vivas.
#
# WeakSet permite que o GC recolha detectores descartados, mantendo a garantia
# de matar processos FFmpeg pendentes no encerramento do app.
# ──────────────────────────────────────────────────────────────────────────────

_active_audio_detectors: "weakref.WeakSet" = weakref.WeakSet()


def _cleanup_all_audio_detectors() -> None:
    """Encerra sessions/procs de todos os detectores de áudio ainda vivos."""
    for det in list(_active_audio_detectors):
        try:
            det._atexit_cleanup()
        except Exception:
            pass


atexit.register(_cleanup_all_audio_detectors)


# ──────────────────────────────────────────────────────────────────────────────
# Data class
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Moment:
    """Um momento detectado ou fundido (super_moment)."""
    start_sec:  float
    end_sec:    float
    reason:     str          # audio_spike | chat_flood | keyword | crash | spin |
                             # near_miss | fastest_lap | manual | super_moment
    score:      float = 1.0  # 0..1
    keywords:   List[str] = field(default_factory=list)
    extra:      dict       = field(default_factory=dict)
    # moment_id: hash MD5 estável (não entra no construtor, calculado em __post_init__)
    moment_id:  str        = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        """Gera hash MD5 estável baseado em start/end/reason/keywords."""
        raw = (
            f"{self.start_sec:.3f}:{self.end_sec:.3f}"
            f":{self.reason}:{'|'.join(sorted(self.keywords))}"
        )
        self.moment_id = hashlib.md5(raw.encode()).hexdigest()[:12]

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


# ──────────────────────────────────────────────────────────────────────────────
# AudioSpikeDetector
# ──────────────────────────────────────────────────────────────────────────────

# Sentinel: sinaliza que a thread leitora encerrou sem dados
_PIPE_EOF = object()


class AudioSpikeDetector:
    """
    Leitura de áudio via pipe FFmpeg usando thread dedicada + queue.Queue.

    Cross-platform (Windows/Linux/macOS): não usa select() em file descriptors.
    Se nenhum chunk chegar em IO_STALL_SEC → mata FFmpeg, drena fila,
    aguarda join da thread leitora e reinicia tudo do seek_sec atual.

    Bound de memória: get_moments_snapshot poda momentos com end_sec antigo
    (default: 5 min antes do elapsed atual). Lista cresceria sem bound em
    sessões longas (~1 spike/min × 24h = 1440 entries).
    """

    SAMPLE_RATE    = 22050
    CHUNK_SECS     = 1.0
    CHUNK_BYTES    = int(SAMPLE_RATE * CHUNK_SECS) * 4   # float32
    BASELINE_WIN   = 60
    BASELINE_MIN   = 15    # amostras VÁLIDAS antes de comparar (era 5)
    SILENCE_DB     = -60.0 # abaixo disto não é referência de loudness
    IO_STALL_SEC   = 5.0   # tempo máximo sem chunk → reinicia
    QUEUE_POLL_SEC = 1.0   # timeout de cada get() na fila
    MOMENT_TTL_SEC = 300.0 # 5min: poda momentos com end_sec antes desse limiar

    def __init__(self, config: dict, elapsed_fn: Callable[[], float]):
        self.cfg        = config["detection"]["audio_spike"]
        self.elapsed_fn = elapsed_fn
        self._moments:  List[Moment] = []   # privado — acesse via get_moments_snapshot()
        self._lock      = threading.Lock()
        self._running   = False
        self._thread:   Optional[threading.Thread] = None
        self._proc      = None              # alias do session.proc — usado por _pipe_reader
        self._session: Optional[FFmpegSession] = None

        # Cleanup no encerramento via WeakSet de módulo (não atexit.register
        # por instância — isso retinha o objeto para sempre, ver P1-10).
        _active_audio_detectors.add(self)

    def _atexit_cleanup(self) -> None:
        """Chamado via atexit: encerra session/proc atual (se houver)."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                _kill_proc(self._proc)
        else:
            _kill_proc(self._proc)

    # ── API pública ───────────────────────────────────────────────────────────

    def get_moments_snapshot(self) -> List[Moment]:
        """
        Retorna cópia thread-safe da lista de momentos.
        Aplica poda lazy: descarta momentos com end_sec < (elapsed - TTL).
        """
        try:
            cutoff = self.elapsed_fn() - self.MOMENT_TTL_SEC
        except Exception:
            cutoff = -1.0
        with self._lock:
            if cutoff > 0 and self._moments:
                # Limpa em-place se houver muitos antigos (>10 ou >25% do total)
                old_count = sum(1 for m in self._moments if m.end_sec < cutoff)
                if old_count > 10 or (old_count > 0 and old_count * 4 > len(self._moments)):
                    self._moments = [m for m in self._moments if m.end_sec >= cutoff]
            return list(self._moments)

    # ── Análise de um chunk (puro, testável) ─────────────────────────────────

    @staticmethod
    def _chunk_db(raw: bytes) -> float:
        """Nível rms do chunk em dBFS. Silêncio digital devolve o piso (-180)."""
        import numpy as np
        audio = np.frombuffer(raw, dtype=np.float32)
        if audio.size == 0:
            return -180.0
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        return float(20.0 * np.log10(rms + 1e-9))

    def _consider_chunk(self, db: float, baseline: deque) -> Optional[dict]:
        """
        Decide se o chunk é um pico, atualizando o baseline.

        Returns:
            None quando não há pico (silêncio, aquecimento ou abaixo do
            threshold); senão o dict que vira Moment.extra.

        Dois gates nasceram de picos falsos medidos em produção
        (audio.spike_db = 132.2 em 15/05 e 71.6 em 12/08, idêntico nas duas
        capturas do dia — ver tests/test_audio_baseline.py):

          1. Silêncio digital não entra no baseline. rms exatamente 0 produz
             20*log10(0 + 1e-9) = -180 dB, que é artefato do epsilon e não
             medição. Na média, ele fazia o primeiro chunk audível parecer um
             salto de 70 a 130 dB — a faixa inteira entre silêncio audível e
             clipping é ~60 dB, então nenhum desses valores é fisicamente real.

          2. O aquecimento conta amostras VÁLIDAS. Com as 5 anteriores, uma
             única amostra ruim deslocava a média em ~20 dB, o threshold
             inteiro. Silêncio não adianta esse relógio.
        """
        if db <= self.SILENCE_DB:
            return None

        baseline.append(db)

        if len(baseline) < self.BASELINE_MIN:
            return None

        baseline_db = sum(baseline) / len(baseline)
        spike_db    = db - baseline_db
        if spike_db < self.cfg["threshold_db"]:
            return None

        return {
            "db":       round(db, 1),
            "baseline": round(baseline_db, 1),
            "spike":    round(spike_db, 1),
        }

    def start(self, stream_file: Path) -> None:
        self._running = True
        self._thread  = threading.Thread(
            target=self._monitor_loop, args=(stream_file,), daemon=True
        )
        self._thread.start()
        logger.info("🔊 AudioSpikeDetector iniciado (queue cross-platform)")

    def stop(self) -> None:
        self._running = False
        _kill_proc(self._proc)   # desbloqueia read() pendente na thread leitora

    # ── Thread leitora do pipe ────────────────────────────────────────────────

    @staticmethod
    def _pipe_reader(proc, chunk_queue: queue.Queue,
                     chunk_bytes: int, stop_flag: threading.Event) -> None:
        """
        Roda em daemon thread. Lê CHUNK_BYTES do stdout do FFmpeg e empurra
        na fila. Ao encerrar (EOF, erro ou stop_flag), empurra _PIPE_EOF.

        Com maxsize na queue: put pode bloquear → loop com timeout check stop_flag,
        evita deadlock em shutdown enquanto consumer está travado.
        """
        try:
            while not stop_flag.is_set():
                try:
                    data = proc.stdout.read(chunk_bytes)
                except Exception:
                    break
                if not data:
                    break
                # put com timeout para checar stop_flag periodicamente
                while not stop_flag.is_set():
                    try:
                        chunk_queue.put(data, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except Exception:
            pass
        finally:
            # Sentinel EOF — best effort sem bloquear
            try:
                chunk_queue.put_nowait(_PIPE_EOF)
            except queue.Full:
                # Queue cheia: drena 1 entry para abrir espaço para sentinel
                try:
                    chunk_queue.get_nowait()
                    chunk_queue.put_nowait(_PIPE_EOF)
                except (queue.Empty, queue.Full):
                    pass

    # ── Loop principal ────────────────────────────────────────────────────────

    def _monitor_loop(self, stream_file: Path) -> None:
        # Valida a dependência AQUI, fora do try do loop: sem isto, um numpy
        # ausente viraria um logger.debug por chunk, reiniciando para sempre.
        import numpy as np  # noqa: F401  (usado via _chunk_db)

        # Aguarda o arquivo ter dados suficientes
        logger.info("🔊 Aguardando stream acumular dados...")
        waited = 0
        while self._running:
            if stream_file.exists() and stream_file.stat().st_size > 500_000:
                break
            time.sleep(2)
            waited += 2
            if waited > 60:
                logger.warning("⏳ Stream não cresceu — áudio desativado.")
                return

        baseline:       deque  = deque(maxlen=self.BASELINE_WIN)
        last_spike_sec: float  = -999.0
        seek_sec:       float  = 0.0

        # maxsize=120: ~2min de buffer (1 chunk = 1s). Reader bloqueia em put()
        # quando cheio → backpressure correto. Sem maxsize, OOM em 30min de stall.
        chunk_queue:    queue.Queue             = queue.Queue(maxsize=120)
        reader_thread:  Optional[threading.Thread] = None
        reader_stop:    Optional[threading.Event]  = None

        logger.info("🔊 Iniciando captura de áudio por fila...")

        while self._running:

            # ── 1. Limpeza da iteração anterior ──────────────────────────────
            if reader_thread is not None:
                # Sinaliza parada e drena a fila antes do join
                if reader_stop:
                    reader_stop.set()
                # Drena para desbloquear possível put() bloqueado (fila cheia)
                while not chunk_queue.empty():
                    try:
                        chunk_queue.get_nowait()
                    except queue.Empty:
                        break
                reader_thread.join(timeout=2)
                reader_thread = None

            # ── 2. Inicia novo processo FFmpeg + thread leitora ───────────────
            session = None
            proc = None
            reader_stop = threading.Event()
            try:
                cmd = [
                    "ffmpeg",
                    "-ss", str(seek_sec),
                    "-i",  str(stream_file),
                    "-vn",
                    "-ar", str(self.SAMPLE_RATE),
                    "-ac", "1",
                    "-f",  "f32le",
                    "pipe:1",
                ]
                # Long-running session reciclada a cada ~3s: quiet=True para
                # evitar inundar o log do usuário (start/end OK ficam em DEBUG).
                # Erros/warnings continuam visíveis.
                session = FFmpegSession(
                    cmd, label="audio_spike", capture_stdout=True, quiet=True,
                )
                session.start()
                proc        = session.proc
                self._session = session
                self._proc  = proc

                reader_thread = threading.Thread(
                    target=AudioSpikeDetector._pipe_reader,
                    args=(proc, chunk_queue, self.CHUNK_BYTES, reader_stop),
                    daemon=True,
                )
                reader_thread.start()

                last_chunk_t = time.time()

                # ── 3. Loop de consumo da fila ────────────────────────────────
                while self._running:
                    try:
                        item = chunk_queue.get(timeout=self.QUEUE_POLL_SEC)
                    except queue.Empty:
                        # Nenhum chunk — verifica stall
                        if time.time() - last_chunk_t > self.IO_STALL_SEC:
                            logger.warning(
                                "⚠️  Áudio: nenhum chunk em "
                                f"{self.IO_STALL_SEC:.0f}s — reiniciando FFmpeg..."
                            )
                            break   # sai do inner loop → finally mata proc
                        continue

                    last_chunk_t = time.time()

                    # EOF ou sentinel
                    if item is _PIPE_EOF or not item:
                        break
                    if len(item) < self.CHUNK_BYTES // 4:
                        break

                    # ── Análise RMS ───────────────────────────────────────────
                    seek_sec += self.CHUNK_SECS
                    was_warm  = len(baseline) >= self.BASELINE_MIN
                    extra     = self._consider_chunk(self._chunk_db(item), baseline)

                    if not was_warm and len(baseline) >= self.BASELINE_MIN:
                        logger.info(
                            f"🔊 Baseline de áudio pronto "
                            f"({len(baseline)} amostras, {sum(baseline)/len(baseline):.1f} dB) "
                            "— detecção de picos ativa."
                        )

                    if extra is None:
                        continue

                    current_sec = self.elapsed_fn()
                    if current_sec - last_spike_sec < self.cfg["buffer_before"]:
                        continue
                    last_spike_sec = current_sec

                    m = Moment(
                        start_sec = max(0.0, current_sec - self.cfg["buffer_before"]),
                        end_sec   = current_sec + self.cfg["buffer_after"],
                        reason    = "audio_spike",
                        score     = min(1.0, extra["spike"] / 40.0),
                        extra     = extra,
                    )
                    with self._lock:
                        self._moments.append(m)
                    _metrics_incr("audio.spike_detected")
                    _metrics_observe("audio.spike_db", extra["spike"])
                    logger.info(
                        f"🎯 Pico de áudio em {current_sec:.1f}s "
                        f"({extra['spike']:+.1f} dB acima da média de "
                        f"{extra['baseline']:.1f} dB)"
                    )

            except Exception as e:
                logger.debug(f"Audio loop erro: {e}")
            finally:
                # Sinaliza leitora e encerra session; join da reader_thread no
                # início do próximo ciclo. session.close() drena stderr ring + log.
                if reader_stop:
                    reader_stop.set()
                if session is not None:
                    try:
                        session.close()
                    except Exception:
                        _kill_proc(proc)
                else:
                    _kill_proc(proc)
                self._session = None
                self._proc = None

            if self._running:
                time.sleep(3)

        # Cleanup final quando stop() foi chamado
        if reader_thread is not None:
            if reader_stop:
                reader_stop.set()
            while not chunk_queue.empty():
                try:
                    chunk_queue.get_nowait()
                except queue.Empty:
                    break
            reader_thread.join(timeout=2)


# ──────────────────────────────────────────────────────────────────────────────
# TwitchChatDetector
# ──────────────────────────────────────────────────────────────────────────────

class TwitchChatDetector:
    """
    Chat Twitch via IRC — flood e keywords. Thread-safe via _lock privado.

    Reconnect:
      • Loop iterativo (sem recursão — stack constante, seguro 24/7).
      • Backoff exponencial 5→10→20→40→…→300s, com jitter ±25%.
      • Counter reseta após sessão estável (>60s conectado).
      • Shutdown imediato via threading.Event + sock.shutdown() para acordar recv.
    """

    # Servers, em ordem de preferência. (host, port, use_ssl)
    _SERVERS = (
        ("irc.chat.twitch.tv", 6697, True),
        ("irc.chat.twitch.tv", 6667, False),
    )

    # Backoff
    _BACKOFF_BASE_SEC      = 5.0
    _BACKOFF_CAP_SEC       = 300.0     # 5 min
    _BACKOFF_JITTER_FRAC   = 0.25      # ±25%
    _STABLE_SESSION_SEC    = 60.0      # connection ≥ 60s reseta backoff
    _MSG_TIMES_MAXLEN      = 2000      # bound contra burst absurdo
    _MOMENT_TTL_SEC        = 300.0     # poda momentos com end_sec < elapsed-300s
    _DEDUPE_WINDOW_SEC     = 15.0      # gatilhos da mesma reason dentro disso = 1 evento

    # Socket timeouts
    _CONNECT_TIMEOUT_SEC   = 10.0
    _RECV_TIMEOUT_SEC      = 5.0       # também é granularidade de shutdown check

    @staticmethod
    def _sanitize_channel(raw: str) -> str:
        """
        Aceita 'baiano', 'https://www.twitch.tv/baiano', '@baiano', etc.
        Retorna nome limpo: 'baiano'.
        Crítico: IRC JOIN com URL completa silenciosamente é ignorado pela Twitch.
        """
        s = (raw or "").strip().rstrip("/")
        for prefix in (
            "https://www.twitch.tv/", "http://www.twitch.tv/",
            "https://twitch.tv/",     "http://twitch.tv/",
            "www.twitch.tv/",         "twitch.tv/",
        ):
            if s.lower().startswith(prefix):
                s = s[len(prefix):]
                break
        s = s.lstrip("@").lstrip("#")
        # Remove query/fragment se houver
        for sep in ("?", "#", "/"):
            if sep in s:
                s = s.split(sep, 1)[0]
        return s.strip()

    def __init__(self, config: dict, elapsed_fn: Callable[[], float],
                 channel: str = None):
        """
        channel: canal desta thread (explícito). Se None, resolve do config —
        primeiro `streams.twitch.channel` (setado por _monitor_platform para a
        plataforma monitorada) e só então o fallback `channels[0]`.
        Passar explicitamente é o caminho correto: o fallback para channels[0]
        era o que fazia threads de outras plataformas ouvirem o canal errado.
        """
        self.cfg_flood  = config["detection"]["chat_flood"]
        self.cfg_kw     = config["detection"]["keywords"]
        self.keywords   = [k.lower() for k in self.cfg_kw.get("words", [])]
        self.elapsed_fn = elapsed_fn
        raw_channel    = (
            channel or
            config["streams"]["twitch"].get("channel") or
            (config["streams"]["twitch"].get("channels") or [""])[0]
        )
        self.channel = self._sanitize_channel(raw_channel)
        self._moments:   List[Moment] = []
        self._lock       = threading.Lock()
        # bound contra burst extremo (cleanup janela ainda roda em _process_message)
        self._msg_times: deque = deque(maxlen=self._MSG_TIMES_MAXLEN)
        self._kw_times:  deque = deque()
        # reason → csec do último gatilho aceito (base do dedupe, ver _add_moment)
        self._last_trigger: dict = {}
        self._baseline_rate: float = 0.0
        self._rate_samples: deque = deque(maxlen=60)

        # Lifecycle
        self._stop_event: threading.Event = threading.Event()
        self._thread:     Optional[threading.Thread] = None
        # Socket atual (para shutdown externo acordar recv). Acesso sob _sock_lock.
        self._sock:       Optional[socket.socket] = None
        self._sock_lock:  threading.Lock = threading.Lock()

    @property
    def chat_rate(self) -> float:
        """Taxa atual de mensagens por segundo."""
        if not self._msg_times:
            return 0.0
        win = self.cfg_flood["window_seconds"]
        now = time.time()
        count = sum(1 for t in self._msg_times if now - t <= win)
        return count / win

    @property
    def chat_rate_ratio(self) -> float:
        """Ratio da taxa atual vs baseline (>2 = chat 2x mais ativo que o normal)."""
        if self._baseline_rate <= 0.01:
            return 1.0
        return self.chat_rate / self._baseline_rate

    def get_moments_snapshot(self) -> List[Moment]:
        """Snapshot com poda lazy de momentos antigos (TTL=5min)."""
        try:
            cutoff = self.elapsed_fn() - self._MOMENT_TTL_SEC
        except Exception:
            cutoff = -1.0
        with self._lock:
            if cutoff > 0 and self._moments:
                old_count = sum(1 for m in self._moments if m.end_sec < cutoff)
                if old_count > 10 or (old_count > 0 and old_count * 4 > len(self._moments)):
                    self._moments = [m for m in self._moments if m.end_sec >= cutoff]
            return list(self._moments)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._connect_loop, name="TwitchIRC", daemon=True,
        )
        self._thread.start()
        logger.info("[IRC] detector started")

    def stop(self) -> None:
        """
        Sinaliza shutdown e desbloqueia recv via socket.shutdown.
        Não faz join — caller pode aguardar self._thread.join() se quiser.
        """
        self._stop_event.set()
        # Acorda recv pendente fechando socket assistido
        with self._sock_lock:
            sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                # SSL socket pode levantar em shutdown; fechar como fallback
                try: sock.close()
                except Exception: pass

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff capped + jitter ±25%."""
        base = self._BACKOFF_BASE_SEC * (2 ** max(0, attempt - 1))
        base = min(base, self._BACKOFF_CAP_SEC)
        jitter = base * self._BACKOFF_JITTER_FRAC
        return max(0.5, base + random.uniform(-jitter, jitter))

    def _close_sock(self, sock: Optional[socket.socket]) -> None:
        """Close idempotente. Limpa self._sock se for o atual."""
        if sock is None:
            return
        try:
            sock.close()
        except Exception:
            pass
        with self._sock_lock:
            if self._sock is sock:
                self._sock = None

    def _try_connect(self) -> Optional[socket.socket]:
        """Tenta cada servidor em ordem. Retorna socket conectado e JOINed, ou None."""
        nick    = "justinfan12345"
        channel = f"#{self.channel.lower().strip()}"

        for server, port, use_ssl in self._SERVERS:
            if self._stop_event.is_set():
                return None
            raw: Optional[socket.socket] = None
            sock: Optional[socket.socket] = None
            t0 = time.monotonic()
            try:
                raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                raw.settimeout(self._CONNECT_TIMEOUT_SEC)
                if use_ssl:
                    ctx = ssl.create_default_context()
                    sock = ctx.wrap_socket(raw, server_hostname=server)
                else:
                    sock = raw
                sock.connect((server, port))
                # IRC handshake
                sock.send(f"NICK {nick}\r\n".encode())
                sock.send(f"USER {nick} 8 * :{nick}\r\n".encode())
                sock.send(f"JOIN {channel}\r\n".encode())
                sock.settimeout(self._RECV_TIMEOUT_SEC)
                elapsed = time.monotonic() - t0
                logger.info(
                    f"[IRC] connected to {server}:{port} ssl={use_ssl} "
                    f"channel={channel} ({elapsed:.2f}s)"
                )
                # Publica para shutdown externo
                with self._sock_lock:
                    self._sock = sock
                return sock
            except Exception as e:
                logger.warning(
                    f"[IRC] connect {server}:{port} failed: "
                    f"{type(e).__name__}: {e}"
                )
                # Cleanup explícito de raw e sock — evita leak de socket pré-wrap
                if sock is not None and sock is not raw:
                    try: sock.close()
                    except Exception: pass
                if raw is not None:
                    try: raw.close()
                    except Exception: pass
        return None

    def _listen(self, sock: socket.socket) -> str:
        """
        Loop de leitura. Retorna razão de saída ('eof', 'shutdown', 'recv_error', 'send_error').
        Buffer interno garante que linhas cortadas entre recvs sejam processadas inteiras.
        """
        channel = f"#{self.channel.lower().strip()}"
        buf = bytearray()
        msg_count = 0

        while not self._stop_event.is_set():
            try:
                data = sock.recv(4096)
            except socket.timeout:
                continue
            except (OSError, ssl.SSLError) as e:
                logger.debug(f"[IRC] recv error: {type(e).__name__}: {e}")
                return "recv_error"

            if not data:
                return "eof"

            buf.extend(data)
            # Processa apenas linhas completas (\r\n). Resto fica no buffer.
            while True:
                idx = buf.find(b"\r\n")
                if idx < 0:
                    break
                line_bytes = bytes(buf[:idx])
                del buf[:idx + 2]
                try:
                    line = line_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                if not line:
                    continue

                # PING/PONG correto: linha começa com "PING " (não substring em payload)
                if line.startswith("PING "):
                    try:
                        sock.send(b"PONG :tmi.twitch.tv\r\n")
                    except Exception as e:
                        logger.debug(f"[IRC] PONG send failed: {e}")
                        return "send_error"
                    continue

                # Numeric reply 366 (RPL_ENDOFNAMES) — deve vir após o nick, não substring
                if " 366 " in line:
                    logger.info(f"[IRC] joined {channel}")
                    continue

                if "PRIVMSG" in line:
                    msg_count += 1
                    msg = line.split("PRIVMSG", 1)[-1].split(":", 1)[-1].strip()
                    if msg_count <= 3 or msg_count % 50 == 0:
                        logger.debug(f"[IRC] msg#{msg_count}: {msg[:60]}")
                    self._process_message(msg.lower())

            # Bound do buffer contra payload patológico sem \r\n
            if len(buf) > 64 * 1024:
                logger.warning("[IRC] recv buffer >64KB without delimiter — discarding")
                buf.clear()

        return "shutdown"

    def _connect_loop(self) -> None:
        """
        Loop iterativo. Stack constante.
        Backoff entre conexões; reseta após sessão estável (>60s).
        """
        attempt = 0
        while not self._stop_event.is_set():
            sock = self._try_connect()
            if sock is None:
                if self._stop_event.is_set():
                    break
                attempt += 1
                _metrics_incr("irc.connect_failed")
                delay = self._backoff_delay(attempt)
                logger.warning(
                    f"[IRC] all servers failed (attempt {attempt}), "
                    f"retry in {delay:.1f}s"
                )
                if self._stop_event.wait(timeout=delay):
                    break
                continue

            # Sessão estabelecida
            session_start = time.monotonic()
            reason = "unknown"
            try:
                reason = self._listen(sock)
            except Exception as e:
                logger.error(
                    f"[IRC] listen crashed: {type(e).__name__}: {e}"
                )
                reason = "exception"
            finally:
                self._close_sock(sock)

            elapsed = time.monotonic() - session_start
            _metrics_observe("irc.session_duration_sec", round(elapsed, 1))
            _metrics_incr("irc.disconnected", reason=reason)
            if elapsed >= self._STABLE_SESSION_SEC:
                # Sessão considerada estável → reset backoff
                if attempt > 0:
                    logger.info(f"[IRC] resetting backoff after stable session ({elapsed:.0f}s)")
                attempt = 0
            else:
                attempt += 1

            if self._stop_event.is_set():
                break

            delay = self._backoff_delay(attempt)
            logger.info(
                f"[IRC] disconnected reason={reason} alive={elapsed:.1f}s "
                f"reconnect in {delay:.1f}s (attempt {attempt})"
            )
            if self._stop_event.wait(timeout=delay):
                break

        logger.info("[IRC] connect loop exited")

    def _process_message(self, msg: str) -> None:
        now   = time.time()
        csec  = self.elapsed_fn()
        win_f = self.cfg_flood["window_seconds"]
        win_k = 5
        _metrics_incr("chat.msg_received")

        self._msg_times.append(now)
        while self._msg_times and now - self._msg_times[0] > win_f:
            self._msg_times.popleft()
        rate = len(self._msg_times) / win_f

        # Atualiza baseline para chat_rate_ratio
        self._rate_samples.append(rate)
        if len(self._rate_samples) >= 5:
            self._baseline_rate = sum(self._rate_samples) / len(self._rate_samples)
        if rate >= self.cfg_flood["messages_per_second"]:
            _metrics_incr("chat.flood_detected")
            self._add_moment("chat_flood", csec,
                             self.cfg_flood["buffer_before"],
                             self.cfg_flood["buffer_after"],
                             score=min(1.0, rate / 20))

        found = [k for k in self.keywords if k in msg]
        if found:
            _metrics_incr("chat.keyword_detected")
            logger.info(f"🔑 Keyword: {found} | {msg[:60]}")
            self._kw_times.append((now, found))
            while self._kw_times and now - self._kw_times[0][0] > win_k:
                self._kw_times.popleft()
            total_kw  = sum(len(e[1]) for e in self._kw_times)
            threshold = max(1, self.cfg_kw.get("min_keyword_count", 1))
            if total_kw >= threshold:
                self._add_moment("keyword", csec, 5, 10,
                                 keywords=found,
                                 score=min(1.0, total_kw / 10))

    def _add_moment(self, reason: str, csec: float, buf_before: float,
                    buf_after: float, keywords=None, score: float = 1.0) -> None:
        """
        Dedupe por instante do GATILHO (P1-5).

        Bug anterior: comparava `abs(m.end_sec - csec) < 15`, misturando
        referenciais — `end_sec` já inclui `buffer_after`, então a diferença era
        ~buffer_after (20s por padrão) e o teste 20 < 15 nunca era satisfeito:
        o dedupe ficava permanentemente desligado. Medido: 60 mensagens em 1s
        geravam 31 momentos chat_flood + 7 keyword.

        Agora rastreamos o csec do último gatilho por reason — invariante ao
        valor dos buffers. A varredura em `_moments[-5:]` também saía pela
        janela sob flood misto; `_last_trigger` não tem esse limite.
        """
        with self._lock:
            last = self._last_trigger.get(reason)
            if last is not None and (csec - last) < self._DEDUPE_WINDOW_SEC:
                return
            self._last_trigger[reason] = csec
            m = Moment(
                start_sec = max(0.0, csec - buf_before),
                end_sec   = csec + buf_after,
                reason    = reason,
                score     = score,
                keywords  = keywords or [],
            )
            self._moments.append(m)
        logger.info(f"🎯 {reason} em {csec:.1f}s")


# ──────────────────────────────────────────────────────────────────────────────
# SimRacingDetector
# ──────────────────────────────────────────────────────────────────────────────

class SimRacingDetector:
    """Telemetria iRacing + Assetto Corsa. Thread-safe via _lock privado."""

    _AC_RT_CAR_INFO = struct.Struct("<fffffffffifiiii")

    def __init__(self, config: dict, elapsed_fn: Callable[[], float]):
        self.cfg        = config.get("detection", {}).get("sim_racing", {})
        self.elapsed_fn = elapsed_fn
        self._moments:  List[Moment] = []
        self._lock      = threading.Lock()
        self._running   = False

        self._crash_g        = float(self.cfg.get("crash_g_force",    4.0))
        self._spin_speed_pct = float(self.cfg.get("spin_speed_drop",  0.4))
        self._damage_thr     = float(self.cfg.get("damage_threshold", 5.0))
        self._udp_port       = int(self.cfg.get("udp_port",           9996))

        self._prev_speed    = 0.0
        self._prev_speed_t  = 0.0
        self._best_lap_ms   = 999_999_999
        self._last_moment_t = -999.0
        self._udp_sock      = None

    def get_moments_snapshot(self) -> List[Moment]:
        with self._lock:
            return list(self._moments)

    def start(self) -> None:
        self._running = True
        if self._try_iracing():
            logger.info("🏎️  SimRacing: iRacing (shared memory)")
        else:
            logger.info(f"🏎️  SimRacing: AC UDP porta {self._udp_port}")
            threading.Thread(target=self._listen_udp, daemon=True).start()

    def stop(self) -> None:
        self._running = False
        if self._udp_sock:
            try:
                self._udp_sock.close()
            except Exception:
                pass

    def _try_iracing(self) -> bool:
        import sys as _sys
        if _sys.platform != "win32":
            return False
        try:
            import mmap
            mm = mmap.mmap(-1, 1, tagname="Local\\IRSDKMemMapFileName",
                           access=mmap.ACCESS_READ)
            mm.close()
            threading.Thread(target=self._iracing_loop, daemon=True).start()
            return True
        except Exception:
            return False

    def _iracing_loop(self) -> None:
        try:
            import mmap
            MM_SIZE = 1 * 1024 * 1024
            mm = mmap.mmap(-1, MM_SIZE, tagname="Local\\IRSDKMemMapFileName",
                           access=mmap.ACCESS_READ)
        except Exception as e:
            logger.warning(f"🏎️  iRacing mmap: {e} — fallback UDP")
            self._listen_udp()
            return

        logger.info("🏎️  iRacing shared memory conectado")
        prev_damage = 0.0
        prev_time   = time.time()

        while self._running:
            try:
                time.sleep(0.1)
                now = time.time()
                self._evaluate_telemetry(
                    speed_kmh=0.0, lat_g=0.0, lon_g=0.0, g_total=0.0,
                    damage=0.0, lap_ms=0, dt=now - prev_time,
                )
                prev_damage = 0.0
                prev_time   = now
            except Exception as e:
                logger.debug(f"iRacing loop: {e}")

        mm.close()

    def _listen_udp(self) -> None:
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", self._udp_port))
            sock.settimeout(1.0)
            self._udp_sock = sock
            atexit.register(sock.close)
            logger.info(f"🏎️  AC UDP: 127.0.0.1:{self._udp_port}")
        except Exception as e:
            logger.warning(f"🏎️  AC UDP bind falhou: {e}")
            return

        prev_time = time.time()
        while self._running:
            try:
                data, _ = sock.recvfrom(1024)
                if len(data) < self._AC_RT_CAR_INFO.size:
                    continue
                f   = self._AC_RT_CAR_INFO.unpack_from(data)
                now = time.time()
                self._evaluate_telemetry(
                    speed_kmh=f[0], lat_g=f[9], lon_g=f[10],
                    g_total=(f[9]**2 + f[10]**2)**0.5,
                    damage=0.0, lap_ms=f[16], dt=now - prev_time,
                )
                prev_time = now
            except socket.timeout:
                continue
            except Exception as e:
                logger.debug(f"AC UDP: {e}")
        sock.close()

    def _evaluate_telemetry(self, speed_kmh: float, lat_g: float, lon_g: float,
                             g_total: float, damage: float, lap_ms: int,
                             dt: float) -> None:
        now     = time.time()
        elapsed = self.elapsed_fn()

        if now - self._last_moment_t < 10.0:
            self._update_speed(speed_kmh, now)
            return

        reason = score = extra = None

        if g_total >= self._crash_g:
            reason = "crash"
            score  = min(1.0, g_total / (self._crash_g * 2))
            extra  = {"g_force": round(g_total, 2), "speed_kmh": round(speed_kmh, 1)}
            logger.info(f"💥 Crash! G={g_total:.2f}g")

        elif damage >= self._damage_thr:
            reason = "crash"
            score  = min(1.0, damage / 20.0)
            extra  = {"damage": round(damage, 1)}
            logger.info(f"💥 Dano! {damage:.1f}")

        elif (self._prev_speed > 60 and dt > 0
              and speed_kmh < self._prev_speed * (1 - self._spin_speed_pct)
              and (now - self._prev_speed_t) < 2.0):
            drop   = (self._prev_speed - speed_kmh) / max(self._prev_speed, 1)
            reason = "spin"
            score  = min(1.0, drop)
            extra  = {"drop_pct": round(drop * 100, 1)}
            logger.info(f"🌀 Spin! {self._prev_speed:.0f}→{speed_kmh:.0f} km/h")

        elif g_total >= self._crash_g * 0.7 and damage < 1:
            reason = "near_miss"
            score  = g_total / self._crash_g
            extra  = {"g_force": round(g_total, 2)}
            logger.info(f"😱 Near-miss! G={g_total:.2f}g")

        elif 0 < lap_ms < self._best_lap_ms and lap_ms > 5000:
            imp    = self._best_lap_ms - lap_ms
            self._best_lap_ms = lap_ms
            reason = "fastest_lap"
            score  = min(1.0, imp / 5000.0)
            extra  = {"lap_ms": lap_ms, "improvement_ms": imp}
            logger.info(f"🏆 Melhor volta! {lap_ms/1000:.3f}s")

        if reason:
            buf_b = self.cfg.get("crash_buffer_before", 15) \
                    if reason in ("crash", "spin") else self.cfg.get("buffer_before", 10)
            buf_a = self.cfg.get("crash_buffer_after", 25) \
                    if reason in ("crash", "spin") else self.cfg.get("buffer_after", 20)

            with self._lock:
                self._moments.append(Moment(
                    start_sec=max(0.0, elapsed - buf_b),
                    end_sec=elapsed + buf_a,
                    reason=reason,
                    score=score,
                    extra=extra or {},
                ))
            self._last_moment_t = now

        self._update_speed(speed_kmh, now)

    def _update_speed(self, speed_kmh: float, now: float) -> None:
        if speed_kmh > self._prev_speed * 0.9:
            self._prev_speed   = speed_kmh
            self._prev_speed_t = now


# ──────────────────────────────────────────────────────────────────────────────
# ManualTimestampDetector  (somente leitura após load, sem lock necessário)
# ──────────────────────────────────────────────────────────────────────────────

class ManualTimestampDetector:
    """Lê manual_clips.txt no startup. Imutável após load() — sem lock."""

    def __init__(self, config: dict):
        self.file     = Path(config["detection"]["manual_timestamps"]["file"])
        self._moments: List[Moment] = []

    def get_moments_snapshot(self) -> List[Moment]:
        return list(self._moments)   # imutável após load, mas consistente com API

    def load(self) -> None:
        if not self.file.exists():
            return
        with open(self.file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    s, e = line.split("-", 1)
                    self._moments.append(Moment(
                        start_sec=self._to_sec(s.strip()),
                        end_sec=self._to_sec(e.strip()),
                        reason="manual",
                        score=1.0,
                    ))
                    logger.info(f"📝 Manual: {s.strip()} → {e.strip()}")
                except Exception as ex:
                    logger.warning(f"Timestamp inválido '{line}': {ex}")

    @staticmethod
    def _to_sec(t: str) -> float:
        parts = list(map(float, t.split(":")))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0]


# ──────────────────────────────────────────────────────────────────────────────
# MomentDetector — orquestrador
# ──────────────────────────────────────────────────────────────────────────────

class MomentDetector:
    """
    Orquestra detectores e retorna lista unificada com merge avançado.

    Merge (sweep-line):
      • Dois momentos se fundem se: overlap OU start_next <= end_group + MERGE_DIST
      • Grupo heterogêneo (>1 reason) → reason="super_moment"
      • Grupo homogêneo → mantém reason original
      • score = min(1.0, soma_scores)
      • extra: reasons=[...], sources=[...]
    """

    MERGE_DIST = 15.0   # segundos após o fim do grupo para atrair o próximo

    # Teto de duração de um momento fundido. Default = clip_max_sec padrão (60s).
    # P1-6: "max_clip_duration" existia no config e nunca era lido; flood
    # contínuo produzia momentos de 82-340s em produção.
    MAX_MOMENT_DURATION = 60.0

    def __init__(self, config: dict, elapsed_fn: Callable[[], float],
                 platform: str = None, channel: str = None):
        """
        platform/channel: identidade da thread que criou este detector.
        Passá-los é o caminho correto (bot_engine._monitor_platform faz isso).

        P1-4: antes, o chat da Twitch era criado sempre que `streams.twitch`
        estivesse habilitado, sem olhar a plataforma monitorada — threads de
        Kick/YouTube abriam IRC no primeiro canal Twitch da lista e atribuíam
        os momentos daquele chat ao canal errado.
        """
        self.config     = config
        self.elapsed_fn = elapsed_fn
        self.platform   = platform
        self.channel    = channel
        self._detectors = []

        det = config["detection"]

        self.audio = None
        if det["audio_spike"]["enabled"]:
            self.audio = AudioSpikeDetector(config, elapsed_fn)
            self._detectors.append(self.audio)

        self.chat = None
        if self._twitch_chat_applies(config, platform):
            self.chat = TwitchChatDetector(config, elapsed_fn, channel=channel)
            self._detectors.append(self.chat)

        # Teto de duração: usa o config quando disponível
        try:
            self._max_duration = float(
                det["audio_spike"].get("max_clip_duration")
                or self.MAX_MOMENT_DURATION
            )
        except Exception:
            self._max_duration = self.MAX_MOMENT_DURATION

        self.sim = None
        if det.get("sim_racing", {}).get("enabled", False):
            self.sim = SimRacingDetector(config, elapsed_fn)
            self._detectors.append(self.sim)
            logger.info("🏎️  SimRacing habilitado")

        self.manual = None
        if det["manual_timestamps"]["enabled"]:
            self.manual = ManualTimestampDetector(config)
            self.manual.load()

    @staticmethod
    def _twitch_chat_applies(config: dict, platform: Optional[str]) -> bool:
        """
        O chat IRC da Twitch só vale para uma thread que monitora a Twitch.

        • platform explícito  → só "twitch" habilita (caminho do bot_engine).
        • platform None       → compat: exige que ESTA config aponte um canal
          Twitch concreto (`streams.twitch.channel`), que é o que
          _monitor_platform seta apenas para a plataforma monitorada.
          O fallback antigo para `channels[0]` não habilita mais nada sozinho.
        """
        det = config["detection"]
        if not (det["chat_flood"]["enabled"] or det["keywords"]["enabled"]):
            return False
        tw = config["streams"].get("twitch", {})
        if not tw.get("enabled"):
            return False
        if platform is not None:
            return platform == "twitch"
        return bool((tw.get("channel") or "").strip())

    def start(self, stream_file: Path = None) -> None:
        if self.audio and stream_file:
            self.audio.start(stream_file)
        if self.chat:
            self.chat.start()
        if self.sim:
            self.sim.start()

    def stop(self) -> None:
        for d in self._detectors:
            d.stop()

    @property
    def all_moments(self) -> List[Moment]:
        """
        Coleta snapshots thread-safe de todos os detectores,
        ordena e aplica merge por overlap/proximidade.
        """
        moments: List[Moment] = []

        # Usa sempre get_moments_snapshot() — nunca acessa _moments diretamente
        for src in (self.audio, self.chat, self.sim, self.manual):
            if src is not None:
                moments.extend(src.get_moments_snapshot())

        moments.sort(key=lambda m: m.start_sec)
        return self._merge_moments(
            moments, max_duration=getattr(self, "_max_duration", None)
        )

    # ── Algoritmo de merge (sweep-line) ──────────────────────────────────────

    @staticmethod
    def _merge_moments(moments: List[Moment],
                       max_duration: float = None) -> List[Moment]:
        """
        Sweep-line: acumula grupo enquanto próximo momento tiver overlap
        OU start <= end_do_grupo + MERGE_DIST.

        max_duration (P1-6): teto do momento resultante. Default = 60s
        (clip_max_sec padrão). Sem ele, flood contínuo encadeava grupos sem
        limite — produção registrou 82s e 136s.
        """
        if not moments:
            return []

        DIST    = MomentDetector.MERGE_DIST
        max_dur = float(max_duration or MomentDetector.MAX_MOMENT_DURATION)
        result: List[Moment] = []

        group:     List[Moment] = [moments[0]]
        group_end: float        = moments[0].end_sec

        for m in moments[1:]:
            # Overlap ou dentro da janela de atração
            if m.start_sec <= group_end + DIST:
                group.append(m)
                group_end = max(group_end, m.end_sec)
            else:
                result.append(MomentDetector._collapse(group, max_dur))
                group     = [m]
                group_end = m.end_sec

        result.append(MomentDetector._collapse(group, max_dur))
        return result

    @staticmethod
    def _collapse(group: List[Moment], max_dur: float = None) -> Moment:
        """
        Reduz um grupo de momentos a um único Moment, respeitando o teto.

        Quando o grupo excede max_dur, a janela é centrada no momento de MAIOR
        score do grupo (o núcleo do evento) — preserva o melhor conteúdo em vez
        de truncar cegamente o início ou o fim.
        """
        max_dur = float(max_dur or MomentDetector.MAX_MOMENT_DURATION)

        if len(group) == 1:
            m = group[0]
            if m.duration <= max_dur:
                return m          # passthrough: mantém o moment_id original
            # Momento único longo demais (ex.: buffers largos): corta do início
            return Moment(
                start_sec=m.start_sec, end_sec=m.start_sec + max_dur,
                reason=m.reason, score=m.score,
                keywords=list(m.keywords), extra=dict(m.extra),
            )

        reasons   = {m.reason for m in group}
        g_start   = min(m.start_sec for m in group)
        g_end     = max(m.end_sec   for m in group)
        start, end = g_start, g_end

        if (g_end - g_start) > max_dur:
            best   = max(group, key=lambda m: m.score)
            center = (best.start_sec + best.end_sec) / 2.0
            start  = max(g_start, center - max_dur / 2.0)
            end    = start + max_dur
            if end > g_end:                      # colou no fim do grupo
                end   = g_end
                start = max(g_start, end - max_dur)
            logger.debug(
                f"✂️  Momento truncado {g_end - g_start:.0f}s → {end - start:.0f}s "
                f"(teto={max_dur:.0f}s, centrado no pico score={best.score:.2f})"
            )

        merged = Moment(
            start_sec = start,
            end_sec   = end,
            reason    = "super_moment" if len(reasons) > 1 else group[0].reason,
            score     = min(1.0, sum(m.score for m in group)),
            keywords  = list({kw for m in group for kw in m.keywords}),
            extra     = {
                "reasons": sorted(reasons),
                "sources": sorted({m.reason for m in group}),
                "count":   len(group),
                "span_sec": round(g_end - g_start, 1),   # duração antes do teto
            },
        )
        if len(reasons) > 1:
            logger.info(
                f"⚡ Super moment: {sorted(reasons)}  "
                f"{merged.start_sec:.1f}s–{merged.end_sec:.1f}s  "
                f"score={merged.score:.2f}"
            )
        return merged
