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
import struct
import atexit
import hashlib
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from pathlib import Path

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
    """

    SAMPLE_RATE    = 22050
    CHUNK_SECS     = 1.0
    CHUNK_BYTES    = int(SAMPLE_RATE * CHUNK_SECS) * 4   # float32
    BASELINE_WIN   = 60
    IO_STALL_SEC   = 5.0   # tempo máximo sem chunk → reinicia
    QUEUE_POLL_SEC = 1.0   # timeout de cada get() na fila

    def __init__(self, config: dict, elapsed_fn: Callable[[], float]):
        self.cfg        = config["detection"]["audio_spike"]
        self.elapsed_fn = elapsed_fn
        self._moments:  List[Moment] = []   # privado — acesse via get_moments_snapshot()
        self._lock      = threading.Lock()
        self._running   = False
        self._thread:   Optional[threading.Thread] = None
        self._proc      = None

        # Registra atexit UMA única vez, apontando para self._proc
        atexit.register(self._atexit_cleanup)

    def _atexit_cleanup(self) -> None:
        """Chamado via atexit: mata o proc atual (se houver)."""
        _kill_proc(self._proc)

    # ── API pública ───────────────────────────────────────────────────────────

    def get_moments_snapshot(self) -> List[Moment]:
        """Retorna cópia thread-safe da lista de momentos."""
        with self._lock:
            return list(self._moments)

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
        """
        try:
            while not stop_flag.is_set():
                try:
                    data = proc.stdout.read(chunk_bytes)
                except Exception:
                    break
                chunk_queue.put(data if data else _PIPE_EOF)
                if not data:
                    break
        except Exception:
            pass
        finally:
            chunk_queue.put(_PIPE_EOF)   # garante que o loop principal desbloqueia

    # ── Loop principal ────────────────────────────────────────────────────────

    def _monitor_loop(self, stream_file: Path) -> None:
        import numpy as np
        import subprocess as sp

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

        chunk_queue:    queue.Queue             = queue.Queue()
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
                proc        = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.DEVNULL)
                self._proc  = proc
                # atexit já registrado no __init__ — usa self._proc atualizado

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
                    audio     = np.frombuffer(item, dtype=np.float32)
                    rms       = float(np.sqrt(np.mean(audio ** 2)))
                    db        = 20.0 * np.log10(rms + 1e-9)
                    baseline.append(db)

                    if len(baseline) < 5:
                        continue

                    baseline_db = sum(baseline) / len(baseline)
                    spike_db    = db - baseline_db

                    if spike_db >= self.cfg["threshold_db"]:
                        current_sec = self.elapsed_fn()
                        if current_sec - last_spike_sec < self.cfg["buffer_before"]:
                            continue
                        last_spike_sec = current_sec

                        m = Moment(
                            start_sec = max(0.0, current_sec - self.cfg["buffer_before"]),
                            end_sec   = current_sec + self.cfg["buffer_after"],
                            reason    = "audio_spike",
                            score     = min(1.0, spike_db / 40.0),
                            extra     = {
                                "db":       round(db, 1),
                                "baseline": round(baseline_db, 1),
                                "spike":    round(spike_db, 1),
                            },
                        )
                        with self._lock:
                            self._moments.append(m)
                        logger.info(
                            f"🎯 Pico de áudio em {current_sec:.1f}s "
                            f"({spike_db:+.1f} dB acima da média)"
                        )

            except Exception as e:
                logger.debug(f"Audio loop erro: {e}")
            finally:
                # Sinaliza leitora e mata o proc; join feito no início do próximo ciclo
                if reader_stop:
                    reader_stop.set()
                _kill_proc(proc)
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
    """Chat Twitch via IRC — flood e keywords. Thread-safe via _lock privado."""

    def __init__(self, config: dict, elapsed_fn: Callable[[], float]):
        self.cfg_flood  = config["detection"]["chat_flood"]
        self.cfg_kw     = config["detection"]["keywords"]
        self.keywords   = [k.lower() for k in self.cfg_kw.get("words", [])]
        self.elapsed_fn = elapsed_fn
        self.channel    = (
            config["streams"]["twitch"].get("channel") or
            (config["streams"]["twitch"].get("channels") or [""])[0]
        )
        self._moments:   List[Moment] = []
        self._lock       = threading.Lock()
        self._msg_times: deque = deque()
        self._kw_times:  deque = deque()
        self._running    = False

    def get_moments_snapshot(self) -> List[Moment]:
        with self._lock:
            return list(self._moments)

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._connect_loop, daemon=True).start()
        logger.info("💬 TwitchChatDetector iniciado")

    def stop(self) -> None:
        self._running = False

    def _connect_loop(self) -> None:
        import socket, ssl

        nick    = "justinfan12345"
        channel = f"#{self.channel.lower().strip()}"

        connections = [
            ("irc.chat.twitch.tv", 6697, True),
            ("irc.chat.twitch.tv", 6667, False),
        ]

        sock = None
        for server, port, use_ssl in connections:
            try:
                raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                raw.settimeout(10)
                if use_ssl:
                    ctx = ssl.create_default_context()
                    sock = ctx.wrap_socket(raw, server_hostname=server)
                else:
                    sock = raw
                sock.connect((server, port))
                logger.info(f"💬 Chat conectado: {server}:{port} (SSL={use_ssl})")
                break
            except Exception as e:
                logger.debug(f"Chat {server}:{port} falhou: {e}")
                sock = None

        if not sock:
            logger.error("❌ Chat IRC: sem conexão. Keywords desativadas.")
            return

        try:
            sock.send(f"NICK {nick}\r\n".encode())
            sock.send(f"USER {nick} 8 * :{nick}\r\n".encode())
            sock.send(f"JOIN {channel}\r\n".encode())
            sock.settimeout(5)
            logger.info(f"💬 Entrando: {channel}")

            msg_count = 0
            while self._running:
                try:
                    data = sock.recv(4096).decode("utf-8", errors="ignore")
                    if not data:
                        break
                    if "PING" in data:
                        sock.send("PONG :tmi.twitch.tv\r\n".encode())
                    for line in data.split("\r\n"):
                        if "PRIVMSG" in line:
                            msg_count += 1
                            msg = line.split("PRIVMSG", 1)[-1].split(":", 1)[-1].strip()
                            if msg_count <= 3 or msg_count % 50 == 0:
                                logger.debug(f"💬 [{msg_count}]: {msg[:60]}")
                            self._process_message(msg.lower())
                        elif "366" in line:
                            logger.info(f"✅ Chat: canal {channel} conectado!")
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.debug(f"Chat recv: {e}")
                    break
        except Exception as e:
            logger.error(f"Chat IRC erro: {e}")
        finally:
            try:
                sock.close()
            except Exception:
                pass
            if self._running:
                logger.warning("💬 Chat desconectado. Reconectando em 10s...")
                time.sleep(10)
                self._connect_loop()

    def _process_message(self, msg: str) -> None:
        now   = time.time()
        csec  = self.elapsed_fn()
        win_f = self.cfg_flood["window_seconds"]
        win_k = 5

        self._msg_times.append(now)
        while self._msg_times and now - self._msg_times[0] > win_f:
            self._msg_times.popleft()
        rate = len(self._msg_times) / win_f
        if rate >= self.cfg_flood["messages_per_second"]:
            self._add_moment("chat_flood", csec,
                             self.cfg_flood["buffer_before"],
                             self.cfg_flood["buffer_after"],
                             score=min(1.0, rate / 20))

        found = [k for k in self.keywords if k in msg]
        if found:
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
        with self._lock:
            # Dedup: ignora duplicatas da mesma razão em <15s
            for m in self._moments[-5:]:
                if m.reason == reason and abs(m.end_sec - csec) < 15:
                    return
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

    def __init__(self, config: dict, elapsed_fn: Callable[[], float]):
        self.config     = config
        self.elapsed_fn = elapsed_fn
        self._detectors = []

        det = config["detection"]

        self.audio = None
        if det["audio_spike"]["enabled"]:
            self.audio = AudioSpikeDetector(config, elapsed_fn)
            self._detectors.append(self.audio)

        self.chat = None
        if config["streams"]["twitch"]["enabled"] and (
            det["chat_flood"]["enabled"] or det["keywords"]["enabled"]
        ):
            self.chat = TwitchChatDetector(config, elapsed_fn)
            self._detectors.append(self.chat)

        self.sim = None
        if det.get("sim_racing", {}).get("enabled", False):
            self.sim = SimRacingDetector(config, elapsed_fn)
            self._detectors.append(self.sim)
            logger.info("🏎️  SimRacing habilitado")

        self.manual = None
        if det["manual_timestamps"]["enabled"]:
            self.manual = ManualTimestampDetector(config)
            self.manual.load()

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
        return self._merge_moments(moments)

    # ── Algoritmo de merge (sweep-line) ──────────────────────────────────────

    @staticmethod
    def _merge_moments(moments: List[Moment]) -> List[Moment]:
        """
        Sweep-line: acumula grupo enquanto próximo momento tiver overlap
        OU start <= end_do_grupo + MERGE_DIST.
        """
        if not moments:
            return []

        DIST   = MomentDetector.MERGE_DIST
        result: List[Moment] = []

        group:     List[Moment] = [moments[0]]
        group_end: float        = moments[0].end_sec

        for m in moments[1:]:
            # Overlap ou dentro da janela de atração
            if m.start_sec <= group_end + DIST:
                group.append(m)
                group_end = max(group_end, m.end_sec)
            else:
                result.append(MomentDetector._collapse(group))
                group     = [m]
                group_end = m.end_sec

        result.append(MomentDetector._collapse(group))
        return result

    @staticmethod
    def _collapse(group: List[Moment]) -> Moment:
        """Reduz um grupo de momentos a um único Moment."""
        if len(group) == 1:
            return group[0]

        reasons = {m.reason for m in group}
        merged  = Moment(
            start_sec = min(m.start_sec for m in group),
            end_sec   = max(m.end_sec   for m in group),
            reason    = "super_moment" if len(reasons) > 1 else group[0].reason,
            score     = min(1.0, sum(m.score for m in group)),
            keywords  = list({kw for m in group for kw in m.keywords}),
            extra     = {
                "reasons": sorted(reasons),
                "sources": sorted({m.reason for m in group}),
                "count":   len(group),
            },
        )
        if len(reasons) > 1:
            logger.info(
                f"⚡ Super moment: {sorted(reasons)}  "
                f"{merged.start_sec:.1f}s–{merged.end_sec:.1f}s  "
                f"score={merged.score:.2f}"
            )
        return merged
