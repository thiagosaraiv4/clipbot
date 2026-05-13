"""
clipbot_api.py — ClipBot v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bridge Python ↔ JavaScript via pywebview.js_api

Todos os métodos públicos (sem _) ficam disponíveis no frontend como:
    await window.pywebview.api.nomeDoMetodo(args)

Design:
  • Thread-safe: callbacks do engine chegam de threads — evaluate_js é thread-safe no pywebview
  • Poll de 250ms enquanto engine ativo — consome fila sem busy-loop
  • Métodos retornam dict {"ok": bool, ...} para tratamento uniforme no JS
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from credentials import load_settings, save_settings, load_credentials, save_credentials
from bot_engine import BotEngine

# ── Pending store compartilhado entre sessões do engine ──────────────────────
# bot_engine já inseriu core/ no sys.path quando foi importado acima.
from pending_store import PendingStore
from feedback import load_weights, score_moment, apply_feedback as _apply_fb

_STORE = PendingStore()   # singleton — carrega pending.json do disco na inicialização

logger = logging.getLogger("ClipBot.API")


# ─── Servidor local para streaming de vídeo ──────────────────────────────────
import http.server
import socketserver
import urllib.parse
import mimetypes

_VIDEO_SERVER_PORT: int = 0


class _AbsolutePathVideoHandler(http.server.BaseHTTPRequestHandler):
    """Serve qualquer arquivo via path absoluto URL-encoded. CORS liberado."""

    def do_GET(self):
        try:
            # Path vem como /?path=C%3A%5Cclips%5Cvideo.mp4
            qs   = urllib.parse.urlparse(self.path).query
            params = dict(urllib.parse.parse_qsl(qs))
            fpath  = Path(params.get("f", ""))

            if not fpath.exists() or not fpath.is_file():
                self.send_error(404, "File not found")
                return

            size  = fpath.stat().st_size
            mime, _ = mimetypes.guess_type(str(fpath))
            mime  = mime or "video/mp4"

            # Suporte a Range requests (necessário para scrubbing no <video>)
            range_header = self.headers.get("Range", "")
            start, end   = 0, size - 1

            if range_header:
                import re
                m = re.match(r"bytes=(\d+)-(\d*)", range_header)
                if m:
                    start = int(m.group(1))
                    end   = int(m.group(2)) if m.group(2) else size - 1

            length = end - start + 1
            status = 206 if range_header else 200

            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            with open(fpath, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            logger.debug(f"VideoServer: {e}")

    def log_message(self, *_):
        pass  # silencia logs


def _ensure_video_server() -> int:
    """Inicia o servidor uma única vez. Retorna a porta."""
    global _VIDEO_SERVER_PORT
    if _VIDEO_SERVER_PORT:
        return _VIDEO_SERVER_PORT

    class _Reuse(socketserver.TCPServer):
        allow_reuse_address = True

    srv = _Reuse(("127.0.0.1", 0), _AbsolutePathVideoHandler)
    _VIDEO_SERVER_PORT = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    logger.debug(f"VideoServer iniciado na porta {_VIDEO_SERVER_PORT}")
    return _VIDEO_SERVER_PORT


class ClipBotAPI:

    def __init__(self):
        self._window   = None
        self._engine: Optional[BotEngine] = None
        self._poll_timer: Optional[threading.Timer] = None

    # ─── Janela ───────────────────────────────────────────────────────────────

    def _set_window(self, window):
        self._window = window

    def _js(self, code: str):
        """Executa JS no frontend — thread-safe."""
        if self._window:
            try:
                self._window.evaluate_js(code)
            except Exception as e:
                logger.debug(f"evaluate_js: {e}")

    # ─── Config ───────────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        return load_settings()

    def save_config(self, config: dict) -> dict:
        try:
            save_settings(config)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ─── Engine ───────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        running = bool(self._engine and self._engine.is_running)
        return {
            "running":     running,
            "clips_today": self._engine.clips_today if self._engine else 0,
            "clips_total": self._engine.clips_total if self._engine else 0,
        }

    def start_engine(self) -> dict:
        if self._engine and self._engine.is_running:
            return {"ok": False, "error": "Engine já está rodando"}

        settings    = load_settings()
        credentials = load_credentials()

        self._engine = BotEngine(
            on_status         = self._on_status,
            on_log            = self._on_log,
            on_clip_posted    = self._on_clip,
            on_moment_pending = self._on_moment_pending,
            pending_store     = _STORE,
        )
        self._engine.start(settings, credentials)
        self._start_poll()
        return {"ok": True}

    def stop_engine(self) -> dict:
        if self._engine:
            self._engine.stop()
        # NÃO cancela o poll aqui — o engine ainda está rodando (estado "stopping")
        # e gerando logs de shutdown. O poll para sozinho quando _on_status("stopped") chegar.
        return {"ok": True}

    # ─── Clips ────────────────────────────────────────────────────────────────

    def get_clips(self) -> list:
        settings = load_settings()
        folder   = Path(settings.get("system", {}).get("output_folder", ""))
        if not folder.exists():
            return []

        clips = []
        try:
            for f in sorted(
                folder.glob("clip_*.mp4"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )[:80]:
                stat  = f.stat()
                parts = f.stem.split("_")
                clips.append({
                    "name":     f.name,
                    "path":     str(f),
                    "size_mb":  round(stat.st_size / 1_048_576, 1),
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%H:%M  %d/%m"),
                    "platform": parts[1] if len(parts) > 1 else "unknown",
                    "channel":  parts[2] if len(parts) > 2 else "unknown",
                    "reason":   parts[-1] if parts else "unknown",
                })
        except Exception as e:
            logger.error(f"get_clips: {e}")
        return clips

    def upload_clip(self, path: str, platforms: list) -> dict:
        """Upload manual — dispara thread daemon e retorna imediatamente."""
        threading.Thread(
            target=self._do_upload,
            args=(path, platforms),
            daemon=True,
        ).start()
        return {"ok": True}

    def delete_clip(self, path: str) -> dict:
        try:
            Path(path).unlink(missing_ok=True)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def read_video_base64(self, path: str) -> dict:
        """Inicia servidor local e retorna URL para streaming direto do vídeo."""
        try:
            video = Path(path)
            if not video.exists():
                return {"ok": False, "error": f"Arquivo não encontrado: {path}"}

            port = _ensure_video_server()
            # Passa o path absoluto como query param URL-encoded
            encoded = urllib.parse.urlencode({"f": str(video)})
            url     = f"http://127.0.0.1:{port}/?{encoded}"
            return {"ok": True, "url": url}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def open_in_explorer(self, path: str) -> dict:
        """
        Abre o arquivo no explorador nativo do OS — fire-and-forget.

        Em vez de context manager (que bloquearia em proc.wait), descartamos a
        ref do Popen explicitamente. GC limpa o handle quando o objeto sair
        do escopo. stdin/stdout/stderr=DEVNULL evita herança de FDs do parent.
        Em Windows, DETACHED_PROCESS desacopla totalmente do Job Object para
        que o explorer sobreviva ao shutdown do ClipBot (file manager é UX
        independente — não queremos que feche junto).
        """
        import subprocess, sys as _sys
        try:
            p = Path(path)
            common_kw = dict(
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, close_fds=True,
            )
            if _sys.platform == "win32":
                # DETACHED_PROCESS (0x8) + CREATE_BREAKAWAY_FROM_JOB (0x1000000):
                # explorer escapa do Job Object e sobrevive ao ClipBot fechar.
                creationflags = 0x00000008 | 0x01000000
                _ = subprocess.Popen(
                    ["explorer", "/select,", str(p)],
                    creationflags=creationflags, **common_kw,
                )
            elif _sys.platform == "darwin":
                _ = subprocess.Popen(["open", "-R", str(p)], **common_kw)
            else:
                _ = subprocess.Popen(["xdg-open", str(p.parent)], **common_kw)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ─── Callbacks do engine → JS ─────────────────────────────────────────────

    def _on_status(self, status: str):
        self._js(f"onEngineStatus({json.dumps(status)})")
        if status == "stopped":
            # Drena TODOS os logs restantes antes de parar o poll
            # (o engine ainda tem logs na fila do shutdown)
            if self._engine:
                remaining = self._engine.poll_logs()
                for entry in remaining:
                    self._on_log(entry.get("level", "INFO"), entry.get("msg", ""))
            self._stop_poll()

    def _on_log(self, level: str, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._js(f"addLog({json.dumps(level)},{json.dumps(msg)},{json.dumps(ts)})")

    def _on_clip(self, info: dict):
        self._js(f"onNewClip({json.dumps(info)})")

    def _on_moment_pending(self, entry: dict):
        """Chamado da thread do engine quando um momento é enfileirado para aprovação."""
        self._js(f"onNewMoment({json.dumps(entry)})")

    # ─── Pending Moments ─────────────────────────────────────────────────────

    def list_moments(self) -> list:
        """
        Retorna todos os momentos pendentes com final_score calculado.
        Funciona mesmo com engine parado (lê do _STORE compartilhado).
        """
        try:
            weights = load_weights()
            moments = _STORE.list_all()
            for m in moments:
                m["final_score"] = score_moment(m, weights)
            return moments
        except Exception as e:
            logger.error(f"list_moments: {e}")
            return []

    def generate_clip(self, moment_id: str, title_text: str = None) -> dict:
        """
        Dispara geração do clip em thread daemon.
        Retorna {"ok": True, "started": True} imediatamente.
        Quando pronto, chama onClipGenerated() no JS.
        """
        if not (self._engine and self._engine.is_running):
            return {"ok": False, "error": "Engine não está rodando"}
        threading.Thread(
            target=self._do_generate_clip,
            args=(moment_id, title_text),
            daemon=True,
        ).start()
        return {"ok": True, "started": True}

    def _do_generate_clip(self, moment_id: str, title_text: str = None) -> None:
        result = self._engine.generate_clip_from_moment(moment_id, title_text=title_text)
        payload = {**result, "moment_id": moment_id}
        self._js(f"onClipGenerated({json.dumps(payload)})")

    def feedback_clip(self, moment_id_or_path: str, label: str) -> dict:
        """
        Registra feedback (approve / top / reject) sobre um momento ou clip gerado.

        • label='reject' antes de gerar  → remove da PendingStore + apaga raw_clip
        • Qualquer label  → atualiza pesos via apply_feedback() e grava em feedback.jsonl

        moment_id_or_path: moment_id (12 hex chars) ou path de um clip gerado.
        """
        try:
            label = label.lower().strip()

            # Tenta resolver como moment_id na store
            entry = _STORE.get(moment_id_or_path)

            # Fallback: tenta ler sidecar .json de um clip gerado
            if not entry:
                sidecar = Path(moment_id_or_path).with_suffix(".json")
                if sidecar.exists():
                    try:
                        entry = json.loads(sidecar.read_text(encoding="utf-8"))
                    except Exception:
                        pass

            if not entry:
                return {"ok": False, "error": "Momento/clip não encontrado"}

            reason     = entry.get("reason", "audio_spike")
            base_score = float(entry.get("base_score") or entry.get("score") or 1.0)

            # Rejeitar antes de gerar → limpa o raw_clip do disco
            if label == "reject":
                mid = entry.get("moment_id", moment_id_or_path)
                _STORE.remove_and_cleanup(mid)
                logger.info(f"🗑️  Momento {mid[:8]} rejeitado e removido.")

            _apply_fb(reason, label, base_score)
            return {"ok": True, "label": label, "reason": reason}

        except Exception as e:
            logger.error(f"feedback_clip: {e}")
            return {"ok": False, "error": str(e)}

    # ─── Poll de logs (fila do engine, 250ms) ────────────────────────────────

    def _start_poll(self):
        self._stop_poll()
        self._poll_once()

    def _stop_poll(self):
        if self._poll_timer:
            self._poll_timer.cancel()
            self._poll_timer = None

    def _poll_once(self):
        if self._engine:
            for entry in self._engine.poll_logs():
                self._on_log(entry.get("level", "INFO"), entry.get("msg", ""))

        # Continua enquanto engine está running OU em estado "stopping"
        # (is_running só vira False depois do on_status("stopped"))
        if self._engine and self._engine.is_running:
            self._poll_timer = threading.Timer(0.25, self._poll_once)
            self._poll_timer.daemon = True
            self._poll_timer.start()

    # ─── Upload interno (thread) ──────────────────────────────────────────────

    def _do_upload(self, path: str, platforms: list):
        video = Path(path)
        settings    = load_settings()
        credentials = load_credentials()

        # Uploaders esperam o config transformado por _build_config, NÃO load_settings() cru.
        # _build_config é @staticmethod — chama direto na classe, sem instanciar BotEngine
        config = BotEngine._build_config(settings, credentials)

        for plat in platforms:
            self._on_log("INFO", f"📤 Upload manual → {plat}: {video.name}")
            try:
                ok = False
                if plat == "tiktok":
                    from uploader import TikTokUploader  # type: ignore
                    ok = TikTokUploader(config).upload(video)
                elif plat == "youtube":
                    from uploader import YouTubeUploader  # type: ignore
                    ok = YouTubeUploader(config).upload(video)
                elif plat == "instagram":
                    from uploader import InstagramUploader  # type: ignore
                    ok = InstagramUploader(config).upload(video)

                icon = "✅" if ok else "❌"
                self._on_log(
                    "INFO" if ok else "ERROR",
                    f"{icon} {plat}: {'concluído' if ok else 'falhou'}",
                )
                self._js(f"onUploadResult({json.dumps({'platform': plat, 'ok': ok, 'clip': video.name})})")

            except Exception as e:
                self._on_log("ERROR", f"❌ {plat}: {e}")
                self._js(f"onUploadResult({json.dumps({'platform': plat, 'ok': False, 'clip': video.name})})")
