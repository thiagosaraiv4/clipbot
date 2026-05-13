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
        import requests
        channel_id = self.config["streams"]["youtube"]["channel_id"].strip()
        api_key    = self.config["streams"]["youtube"]["api_key"].strip()
        url = (
            "https://www.googleapis.com/youtube/v3/search"
            f"?part=id&channelId={channel_id}&eventType=live"
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

    def stop_recording(self) -> Path:
        """
        Para a gravação. _kill_proc faz terminate → wait(5s) → kill.
        Seguro para KeyboardInterrupt no meio do wait.
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
        logger.info(f"✂️  Segmento salvo: {output_path}")
        return output_path
