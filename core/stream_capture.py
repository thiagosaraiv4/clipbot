"""
capture/stream_capture.py  — ClipBot v2.4
Captura streams de Twitch, YouTube, Kick e RTMP em tempo real.

Mudanças v2.4:
  • start_recording: try/finally interno — se Popen falhar após atexit.register,
    limpa o estado corretamente.
  • stop_recording: usa _kill_proc(terminate→kill) e trata KeyboardInterrupt.
  • Sem outras mudanças estruturais — a base já estava correta em v2.3.
"""

import os
import time
import atexit
import threading
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ffmpeg_runner import FFmpegSession, run_ffmpeg, kill_tree

logger = logging.getLogger("ClipBot.Capture")


def _kill_proc(proc, timeout: float = 3.0) -> None:
    """
    Encerra subprocess.Popen: terminate() → wait(timeout) → kill().
    Nunca lança exceção. Seguro para atexit e finally.
    """
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


_active_captures: list = []
_captures_lock = threading.Lock()


def _cleanup_all_captures() -> None:
    with _captures_lock:
        captures = list(_active_captures)
    for cap in captures:
        # Preferir close() da session (drena stderr buffer + log estruturado)
        if getattr(cap, "current_session", None) is not None:
            try:
                cap.current_session.close()
            except Exception:
                kill_tree(cap.current_process)
        else:
            _kill_proc(cap.current_process)


atexit.register(_cleanup_all_captures)


def _probe_duration(path) -> Optional[float]:
    """Duração do arquivo em segundos via ffprobe; None se não deu para medir."""
    try:
        r = run_ffmpeg(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            label="probe_segment", timeout=15, capture_stdout=True, quiet=True,
        )
        if not r.ok:
            return None
        txt = (r.stdout or b"").decode("utf-8", "replace").strip()
        return float(txt) if txt and txt != "N/A" else None
    except Exception as e:
        logger.debug(f"ffprobe falhou: {e}")
        return None


# Windows: o ffmpeg recém-encerrado ainda segura o handle do .ts por instantes.
_DISCARD_RETRIES     = 5
_DISCARD_BACKOFF_SEC = 0.2


def _discard_recording(path: Path) -> bool:
    """
    Apaga a gravação contínua de uma captura encerrada.

    Antes ela ficava até a limpeza do boot (7 dias): na validação de 26/10
    foram -11,3 GB em 35 min, com o mesmo VOD do YouTube regravado 6 vezes.
    Se o arquivo não sair, a limpeza do boot recolhe — nunca levanta.
    """
    try:
        size_gb = path.stat().st_size / 1e9
    except OSError:
        size_gb = 0.0
    last_err: Optional[OSError] = None
    for attempt in range(1, _DISCARD_RETRIES + 1):
        try:
            path.unlink(missing_ok=True)
            logger.info(f"🗑️  Gravação contínua apagada: {path.name} ({size_gb:.2f} GB)")
            return True
        except PermissionError as e:
            last_err = e
            time.sleep(_DISCARD_BACKOFF_SEC * attempt)
        except OSError as e:
            last_err = e
            break
    logger.warning(
        f"⚠️  Gravação contínua não pôde ser apagada ({last_err}); "
        f"fica para a limpeza do boot: {path.name}"
    )
    return False


class StreamCapture:

    def __init__(self, config: dict, clips_folder: str):
        self.config       = config
        self.clips_folder = Path(clips_folder)
        self.clips_folder.mkdir(parents=True, exist_ok=True)

        self.is_recording    = False
        self.current_session = None    # FFmpegSession ativo (ou None)
        self.current_process = None    # alias do session.proc — mantido p/ legacy callers
        self.current_file    = None
        self.start_time      = None
        self._lock           = threading.Lock()

        with _captures_lock:
            _active_captures.append(self)

    def _unregister(self) -> None:
        with _captures_lock:
            try:
                _active_captures.remove(self)
            except ValueError:
                pass

    # ──────────────────────────────────────────────────────────────────────────
    # URL Resolvers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_twitch_url(self) -> str:
        ch = self.config["streams"]["twitch"]["channel"].strip()
        return ch if ch.startswith("http") else f"https://www.twitch.tv/{ch}"

    def _get_youtube_url(self) -> str:
        """
        Resolve URL do stream YouTube. Aceita 4 formatos no `channel_id`:

          1. Watch URL completa: https://www.youtube.com/watch?v=XYZ
             → usa direto (streamlink resolve se a live estiver no ar).
          2. youtu.be/XYZ → idem.
          3. @handle (ou youtube.com/@handle) → usa `youtube.com/@handle/live`
             (URL "atalho" que YouTube redireciona pra live ativa). Funciona
             sem API key.
          4. Channel ID puro (UCxxxxxxxxxxxxxxxxxxxxxx, 24 chars começando UC):
             usa YouTube Data API v3 search (requer api_key configurada).

        Sem live ativa → RuntimeError("Nenhuma live ativa no YouTube").
        """
        raw = self.config["streams"]["youtube"]["channel_id"].strip()
        api_key = (self.config["streams"]["youtube"].get("api_key") or "").strip()

        # ── 1+2: URL direta de video ─────────────────────────────────────────
        if "watch?v=" in raw or "youtu.be/" in raw:
            if not raw.startswith("http"):
                raw = "https://www.youtube.com/" + raw.lstrip("/")
            return raw

        # ── 3: handle @user ─ usa /live shortcut (sem API key) ──────────────
        # Aceita "@nome", "youtube.com/@nome", ou URL completa com @
        m_handle = None
        for prefix in (
            "https://www.youtube.com/", "https://youtube.com/",
            "www.youtube.com/", "youtube.com/",
        ):
            if raw.lower().startswith(prefix):
                rest = raw[len(prefix):].rstrip("/")
                if rest.startswith("@"):
                    m_handle = rest
                break
        if m_handle is None and raw.startswith("@"):
            m_handle = raw
        if m_handle:
            return f"https://www.youtube.com/{m_handle}/live"

        # ── 4: channel ID UCxxxx (24 chars) — requer api_key ────────────────
        ch_id_clean = raw
        # Strip prefixos comuns se passou URL completa do channel
        for prefix in (
            "https://www.youtube.com/channel/", "https://youtube.com/channel/",
        ):
            if ch_id_clean.lower().startswith(prefix):
                ch_id_clean = ch_id_clean[len(prefix):].rstrip("/")
                break

        if not (len(ch_id_clean) == 24 and ch_id_clean.startswith("UC")):
            raise RuntimeError(
                f"YouTube channel_id formato desconhecido: {raw!r}. "
                f"Use 'https://www.youtube.com/watch?v=ID' (vídeo direto), "
                f"'@handle' (sem API key), ou 'UCxxxxxxxxxxxxxxxxxxxxxx' "
                f"(channel ID + api_key configurada)."
            )

        if not api_key:
            raise RuntimeError(
                f"Channel ID detectado ({ch_id_clean[:6]}...) mas api_key "
                f"do YouTube Data API não configurada. Configure a key ou "
                f"use formato @handle / watch?v= URL."
            )

        import requests
        url = (
            "https://www.googleapis.com/youtube/v3/search"
            f"?part=id&channelId={ch_id_clean}&eventType=live"
            f"&type=video&key={api_key}"
        )
        resp  = requests.get(url, timeout=10)
        items = resp.json().get("items", [])
        if not items:
            raise RuntimeError("Nenhuma live ativa no YouTube.")
        return f"https://www.youtube.com/watch?v={items[0]['id']['videoId']}"

    def _get_kick_url(self) -> str:
        ch = self.config["streams"]["kick"]["channel"].strip()
        return ch if ch.startswith("http") else f"https://kick.com/{ch}"

    def _resolve_stream_url(self, raw_url: str) -> str:
        import shutil
        sl = shutil.which("streamlink") or shutil.which("streamlink.exe")
        if not sl:
            raise RuntimeError(
                "streamlink não encontrado! Instale com:  pip install streamlink"
            )
        result = subprocess.run(
            [sl, "--stream-url", raw_url, "best"],
            capture_output=True, text=True, timeout=60,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0 or not stdout:
            if "No playable streams" in stderr or "error: No streams" in stderr:
                raise RuntimeError(
                    f"Live não está ativa no momento.\nURL: {raw_url}\n"
                    "Aguarde a live começar e tente novamente."
                )
            elif "Unable to open URL" in stderr or "404" in stderr:
                raise RuntimeError(
                    f"Canal não encontrado: {raw_url}\n"
                    "Verifique se o nome do canal está correto."
                )
            else:
                raise RuntimeError(
                    f"Streamlink falhou para: {raw_url}\n"
                    f"Detalhe: {stderr or 'sem resposta'}"
                )

        logger.info("🔗 Stream URL resolvida com sucesso")
        return stdout

    # ──────────────────────────────────────────────────────────────────────────
    # Recording
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_name(s: str) -> str:
        s = s.strip().rstrip("/")
        for prefix in [
            "https://www.twitch.tv/", "https://twitch.tv/",
            "https://kick.com/",       "https://www.kick.com/",
            "https://www.youtube.com/","https://youtube.com/",
        ]:
            if s.lower().startswith(prefix):
                s = s[len(prefix):]
        s = s.lstrip("@")
        for c in r'\/:*?"<>| ':
            s = s.replace(c, "_")
        return s[:30] or "canal"

    def _output_path(self, platform: str, channel: str = "") -> Path:
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        clean = self._clean_name(channel) if channel else platform
        return self.clips_folder / f"raw_{platform}_{clean}_{ts}.ts"

    def start_recording(self, platform: str) -> Path:
        """
        Inicia gravação contínua. try/finally interno garante que um
        Popen criado mas com erro posterior não fique orphan.
        """
        with self._lock:
            if self.is_recording:
                logger.warning("Gravação já em andamento.")
                return self.current_file

            logger.info(f"🎬 Iniciando captura: {platform}")

            if platform == "twitch":
                ch       = self.config["streams"]["twitch"]["channel"].strip()
                if not ch:
                    raise ValueError("Nome do canal Twitch não preenchido!")
                page_url = self._get_twitch_url()
            elif platform == "youtube":
                ch = (self.config["streams"]["youtube"].get("channel_id", "") or
                      self.config["streams"]["youtube"].get("channel", "")).strip()
                if not ch:
                    raise ValueError("Channel ID do YouTube não preenchido!")
                page_url = self._get_youtube_url()
            elif platform == "kick":
                ch       = self.config["streams"]["kick"]["channel"].strip()
                if not ch:
                    raise ValueError("Nome do canal Kick não preenchido!")
                page_url = self._get_kick_url()
            elif platform == "custom_rtmp":
                ch       = ""
                page_url = self.config["streams"]["custom_rtmp"]["url"]
            else:
                raise ValueError(f"Plataforma desconhecida: {platform}")

            stream_url = self._resolve_stream_url(page_url)
            out_file   = self._output_path(platform, ch)

            cmd = [
                "ffmpeg", "-y",
                "-i",   stream_url,
                "-c:v", "copy",
                "-c:a", "aac",
                "-f",   "mpegts",
                str(out_file),
            ]

            # Long-running session: stderr drenado em thread, sem deadlock por
            # buffer cheio. capture_stdout=False (ffmpeg escreve direto no arquivo).
            # quiet=True: capture sempre é killed no shutdown (rc!=0 esperado);
            # events.jsonl ainda preserva stderr_tail para diagnóstico real.
            session = FFmpegSession(
                cmd,
                label=f"capture_{platform}_{self._clean_name(ch)[:24]}",
                capture_stdout=False,
                quiet=True,
            )
            session.start()

            # Só atualiza o estado após start() bem-sucedido
            self.current_session = session
            self.current_process = session.proc   # mantido para compat com kill_tree legado
            self.current_file    = out_file
            self.start_time      = time.time()
            self.is_recording    = True
            logger.info(f"✅ Gravando em: {out_file}")
            return out_file

    def stop_recording(self, discard: bool = False) -> Path:
        """
        Para a gravação. _kill_proc faz terminate → wait(5s) → kill.
        Seguro para KeyboardInterrupt no meio do wait.

        discard=True apaga a gravação contínua depois de parar (o engine, ao
        fim de cada attempt). Os segmentos já recortados (raw_pending_*) são
        arquivos próprios e ficam. Sem discard o arquivo fica (core/main.py).
        """
        with self._lock:
            if not self.is_recording:
                return None

            logger.info("⏹ Parando gravação...")
            try:
                if self.current_session is not None:
                    self.current_session.close(grace=5.0)
                else:
                    _kill_proc(self.current_process, timeout=5.0)
            except KeyboardInterrupt:
                # Ctrl+C durante o wait → força kill imediato via kill_tree
                kill_tree(self.current_process, grace=0.5)

            self.is_recording    = False
            file                 = self.current_file
            self.current_file    = None
            self.current_session = None
            self.current_process = None
            self.start_time      = None
            self._unregister()
            logger.info("✅ Gravação encerrada.")
            if discard and file is not None:
                _discard_recording(Path(file))
            return file

    def get_elapsed_seconds(self) -> float:
        if not self.is_recording or self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    # ──────────────────────────────────────────────────────────────────────────
    # Extração de segmento
    # ──────────────────────────────────────────────────────────────────────────

    def extract_segment(self, source_file: Path, start_sec: float,
                        end_sec: float, output_path: Path) -> Path:
        duration = end_sec - start_sec
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i",  str(source_file),
            "-t",  str(duration),
            "-c:v", "copy",
            "-c:a", "copy",
            str(output_path),
        ]
        result = run_ffmpeg(cmd, label="extract_segment", timeout=120)
        if not result.ok:
            tail = "\n".join(result.stderr_tail[-30:])
            raise RuntimeError(
                f"ffmpeg recorte falhou rc={result.returncode}:\n{tail}"
            )
        self._validate_segment(Path(output_path), duration)
        logger.info(f"✂️  Segmento salvo: {output_path}")
        return output_path

    # Recorte mínimo aceitável. Com -ss além do fim do arquivo o ffmpeg sai
    # com rc=0 e grava um .ts só com o cabeçalho (~3 KB, 0,02 s): a live caiu,
    # o chat continuou gerando momentos, e o lixo ia para a fila de aprovação.
    _MIN_SEGMENT_SEC   = 1.0
    _MIN_SEGMENT_RATIO = 0.5
    _MIN_SEGMENT_BYTES = 64 * 1024     # piso sem ffprobe: cabeçalho de .ts ~3 KB

    def _validate_segment(self, path: Path, requested_sec: float) -> None:
        """
        Confere a duração real do recorte. Remove o arquivo e levanta
        RuntimeError se ele não serve; recorte parcial passa com aviso.
        """
        got = _probe_duration(path)
        if got is None:
            size = path.stat().st_size if path.exists() else 0
            if size < self._MIN_SEGMENT_BYTES:
                path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"recorte vazio: {path.name} tem {size} bytes "
                    "(sem ffprobe para medir a duração)"
                )
            return

        need = max(self._MIN_SEGMENT_SEC, self._MIN_SEGMENT_RATIO * requested_sec)
        if got < need:
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"recorte curto demais: {got:.1f}s de {requested_sec:.1f}s pedidos "
                "— a janela passou do fim do material capturado"
            )
        if got < 0.95 * requested_sec:
            logger.warning(
                f"⚠️  Recorte parcial: {got:.1f}s de {requested_sec:.1f}s pedidos "
                f"({path.name})"
            )
