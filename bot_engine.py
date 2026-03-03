"""
bot_engine.py  — ClipBot v2.4
Motor do bot — roda em thread separada, reporta status via callbacks.

Mudanças v2.4:
  • _stats_lock (era _counters_lock): protege clips_today/clips_total.
  • _run: trata KeyboardInterrupt explicitamente.
  • _run_bot: try/finally próprio que chama cleanup imediato de todas as threads
    filhas (não depende apenas de atexit).
  • _monitor_platform: try/finally já presente — mantido; logging melhorado.
"""

import sys
import json
import time
import atexit
import threading
import logging
import queue
from pathlib import Path
from datetime import datetime

_BASE = Path(__file__).parent
_CORE = _BASE / "core"
sys.path.insert(0, str(_CORE))
# Note: stream_capture.py lives directly in core/ (already on sys.path above)
sys.path.insert(0, str(_CORE / "detection"))
sys.path.insert(0, str(_CORE / "editing"))
sys.path.insert(0, str(_CORE / "upload"))

logger = logging.getLogger("ClipBot.Engine")


class QueueHandler(logging.Handler):
    """Redireciona logs para fila consumida pela GUI."""
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.q = log_queue

    def emit(self, record):
        try:
            self.q.put_nowait({
                "type":  "log",
                "level": record.levelname,
                "msg":   self.format(record),
            })
        except queue.Full:
            pass   # backpressure: descarta log se fila cheia


class BotEngine:

    def __init__(self, on_status=None, on_log=None, on_clip_posted=None,
                 on_moment_pending=None, pending_store=None):
        self.on_status         = on_status         or (lambda s: None)
        self.on_log            = on_log            or (lambda lvl, msg: None)
        self.on_clip_posted    = on_clip_posted    or (lambda info: None)
        self.on_moment_pending = on_moment_pending or (lambda entry: None)

        self._thread:     threading.Thread = None
        self._stop_event  = threading.Event()
        self._log_queue   = queue.Queue(maxsize=5000)
        self.is_running   = False

        # ── Contadores ────────────────────────────────────────────────────────
        # _stats_lock: protege incrementos concorrentes de N threads de canais
        self.clips_today  = 0
        self.clips_total  = 0
        self.start_time   = None
        self._stats_lock  = threading.Lock()   # ← renomeado de _counters_lock

        # ── Pending store (compartilhado com a API) ───────────────────────────
        if pending_store is not None:
            self.pending_store = pending_store
        else:
            from pending_store import PendingStore
            self.pending_store = PendingStore()

        # ── Estado do ciclo de vida do engine ─────────────────────────────────
        # Definidos em _run_bot; usados por generate_clip_from_moment
        self._config:     dict = None
        self._editor            = None
        self._title_gen         = None
        self._output_dir: Path = None
        self._clips_dir:  Path = None

        # ── Log handler ───────────────────────────────────────────────────────
        handler = QueueHandler(self._log_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        logging.getLogger("ClipBot").addHandler(handler)
        logging.getLogger("ClipBot").setLevel(logging.DEBUG)

    # ──────────────────────────────────────────────────────────────────────────
    # Start / Stop
    # ──────────────────────────────────────────────────────────────────────────

    def start(self, settings: dict, credentials: dict) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(settings, credentials),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if not self.is_running:
            return
        self._stop_event.set()
        self.on_status("stopping")

    def poll_logs(self):
        messages = []
        while not self._log_queue.empty():
            messages.append(self._log_queue.get_nowait())
        return messages

    # ──────────────────────────────────────────────────────────────────────────
    # Config builder
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_config(settings: dict, credentials: dict) -> dict:
        s     = settings
        det   = s["detection"]
        upl   = s["upload"]
        ed    = s["editing"]
        sys_  = s["system"]
        creds = credentials

        keywords = [k.strip() for k in det.get("keywords_list", "").split(",") if k.strip()]

        return {
            "streams": {
                "twitch": {
                    "enabled":       s["platforms"]["twitch"]["enabled"],
                    "channels":      s["platforms"]["twitch"].get("channels", []),
                    "client_id":     creds.get("twitch_client_id", ""),
                    "client_secret": creds.get("twitch_client_secret", ""),
                },
                "youtube": {
                    "enabled":  s["platforms"]["youtube"]["enabled"],
                    "channels": s["platforms"]["youtube"].get("channels", []),
                    "api_key":  s["platforms"]["youtube"].get("api_key", ""),
                },
                "kick": {
                    "enabled":  s["platforms"]["kick"]["enabled"],
                    "channels": s["platforms"]["kick"].get("channels", []),
                },
                "custom_rtmp": {"enabled": False, "url": ""},
            },
            "detection": {
                "audio_spike": {
                    "enabled":           det["audio_spike"],
                    "threshold_db":      det["threshold_db"],
                    "min_clip_duration": det["clip_min_sec"],
                    "max_clip_duration": det["clip_max_sec"],
                    "buffer_before":     max(10, det["clip_min_sec"] // 3),
                    "buffer_after":      max(20, det["clip_min_sec"] - (det["clip_min_sec"] // 3)),
                },
                "chat_flood": {
                    "enabled":             det["chat_flood"],
                    "messages_per_second": det["flood_rate"],
                    "window_seconds":      3,
                    "buffer_before":       max(10, det["clip_min_sec"] // 3),
                    "buffer_after":        max(20, det["clip_min_sec"] - (det["clip_min_sec"] // 3)),
                },
                "keywords": {
                    "enabled":           det["keywords"],
                    "words":             keywords,
                    "min_keyword_count": 1,
                },
                "manual_timestamps": {
                    "enabled": False,
                    "file":    "manual_clips.txt",
                },
            },
            "editing": {
                "output_format": ed["format"],
                "resolution":    [1080, 1920],
                "fps":           30,
                "subtitles": {
                    "enabled":       ed["subtitles"],
                    "language":      ed["language"],
                    "font":          "Arial",
                    "font_size":     16,
                    "color":         "white",
                    "outline_color": "black",
                    "outline_width": 2,
                    "position":      "bottom",
                    "margin_v":      None,
                },
                "layout_preset": "tiktok",
                "watermark": {"enabled": False, "image_path": "", "position": "top-right", "opacity": 0.7},
                "intro":     {"enabled": False, "video_path": "", "duration": 3},
                "outro":     {"enabled": False, "video_path": "", "duration": 3},
                "title_banner": {
                    "enabled":    True,
                    "bg_color":   "0xCC0000",
                    "text_color": "white",
                    "bg_opacity": 0.92,
                    "border_w":   18,
                    "max_chars":  50,
                    "banner_duration": 5,
                    "use_whisper_title": True,
                },
            },
            "upload": {
                "delay_between_posts": upl["delay_sec"],
                "max_clips_per_day":   upl["max_per_day"],
                "tiktok": {
                    "enabled":  upl["tiktok"]["enabled"],
                    # UI salva em settings["upload"]["tiktok"]["session_id"];
                    # credenciais cifradas usam "tiktok_session_id" — lemos dos dois.
                    "session_id": (
                        creds.get("tiktok_session_id") or
                        upl.get("tiktok", {}).get("session_id", "")
                    ),
                    "default_caption": "🔥 Momento insano da live! #live #gaming #clips",
                    "hashtags":        ["#live", "#gaming", "#clips", "#fypシ"],
                    "schedule_hour":   None,
                },
                "instagram": {
                    "enabled":  upl["instagram"]["enabled"],
                    # UI salva em settings["upload"]["instagram"]["username/password"].
                    "username": (
                        creds.get("instagram_user") or
                        upl.get("instagram", {}).get("username") or
                        upl.get("instagram", {}).get("user", "")
                    ),
                    "password": (
                        creds.get("instagram_pass") or
                        upl.get("instagram", {}).get("password", "")
                    ),
                    "default_caption": "🔥 Melhor momento da live!\n\n#live #gaming #reels",
                    "post_as":         "reel",
                },
                "youtube": {
                    "enabled":              upl["youtube"]["enabled"],
                    "credentials_file":     (
                        creds.get("youtube_credentials_path") or
                        upl.get("youtube", {}).get("credentials_path") or
                        str(Path.home() / "AppData" / "Roaming" / "ClipBot" / "youtube_credentials.json")
                    ),
                    "default_title":        "🔥 CLIP INSANO da Live!",
                    "default_description":  "Melhor momento da live!",
                    "category_id":          "20",
                    "privacy":              "public",
                    "made_for_kids":        False,
                },
            },
            "notifications": {
                "discord":  {"enabled": False, "webhook_url": ""},
                "telegram": {"enabled": False, "bot_token": "", "chat_id": ""},
            },
            "system": {
                "clips_folder":         sys_["clips_folder"],
                "output_folder":        sys_["output_folder"],
                "log_level":            "INFO",
                "cleanup_after_upload": True,
                "check_interval":       5,
                "approval_mode":        bool(sys_.get("approval_mode", False)),
                "approval_top_n":       int(sys_.get("approval_top_n", 12)),
            },
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Garbage collection de disco (raw_*.ts e clips antigos)
    # ──────────────────────────────────────────────────────────────────────────

    def _garbage_collect_disk(self, clips_dir: Path, output_dir: Path) -> None:
        """
        Varre clips_dir e output_dir, deletando arquivos antigos (>7 dias)
        que não estejam mapeados no PendingStore.
        Chamado uma vez no início do engine.
        """
        import os
        MAX_AGE_SECS = 7 * 24 * 3600  # 7 dias
        now = time.time()

        # Coleta raw_files atualmente referenciados pela PendingStore
        pending_files = set()
        for entry in self.pending_store.list_all():
            rf = entry.get("raw_file", "")
            if rf:
                pending_files.add(os.path.normpath(rf))

        removed = 0
        for folder in (clips_dir, output_dir):
            if not folder.exists():
                continue
            for pattern in ("raw_*.ts", "*.mp4", "*.json"):
                for f in folder.glob(pattern):
                    try:
                        age = now - f.stat().st_mtime
                        if age < MAX_AGE_SECS:
                            continue
                        if os.path.normpath(str(f)) in pending_files:
                            continue
                        f.unlink(missing_ok=True)
                        removed += 1
                    except Exception:
                        pass

        if removed:
            logger.info(f"🧹 Garbage collect: {removed} arquivo(s) antigo(s) removido(s)")

    # ──────────────────────────────────────────────────────────────────────────
    # Execução principal
    # ──────────────────────────────────────────────────────────────────────────

    def _run(self, settings: dict, credentials: dict) -> None:
        """Entry-point da thread principal do bot."""
        self.is_running = True
        self.start_time = datetime.now()
        self.on_status("running")
        logger.info("🚀 ClipBot iniciado!")

        try:
            config = self._build_config(settings, credentials)
            self._run_bot(config)
        except KeyboardInterrupt:
            logger.info("⌨️  Interrupção manual (Ctrl+C) recebida.")
        except Exception as e:
            logger.error(f"Erro crítico: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        finally:
            self.is_running = False
            self.on_status("stopped")
            logger.info("👋 ClipBot encerrado.")

    def _run_bot(self, config: dict) -> None:
        """
        Instancia workers, lança threads por canal e aguarda stop_event.
        try/finally garante cleanup imediato mesmo em exceção — não depende só de atexit.
        """
        from stream_capture        import StreamCapture
        from moment_detector       import MomentDetector
        from video_editor          import VideoEditor
        from uploader              import MultiPlatformUploader, Notifier
        from description_generator import DescriptionGenerator
        from title_generator       import TitleGenerator

        clips_dir  = Path(config["system"]["clips_folder"])
        output_dir = Path(config["system"]["output_folder"])
        clips_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Limpeza automática de arquivos antigos (>7 dias)
        try:
            self._garbage_collect_disk(clips_dir, output_dir)
        except Exception as e:
            logger.debug(f"Garbage collect: {e}")

        editor   = VideoEditor(config, str(output_dir))
        uploader = MultiPlatformUploader(config)
        desc_gen  = DescriptionGenerator(config)
        title_gen = TitleGenerator(config)
        Notifier(config)   # side-effect: instancia notificadores

        # Expõe para generate_clip_from_moment (thread externa)
        self._config     = config
        self._editor     = editor
        self._title_gen  = title_gen
        self._output_dir = output_dir
        self._clips_dir  = clips_dir

        # Monta lista de (plataforma, canal)
        active = []
        for plat in ("twitch", "youtube", "kick"):
            cfg_p = config["streams"].get(plat, {})
            if not cfg_p.get("enabled"):
                continue
            for ch in cfg_p.get("channels", []):
                if ch.strip():
                    active.append((plat, ch.strip()))

        if not active:
            logger.warning("⚠️  Nenhum canal ativo! Configure em 'Lives' e salve.")
            return

        logger.info(
            f"📡 Monitorando {len(active)} canal(is): "
            + ", ".join(f"{p}:{c}" for p, c in active)
        )

        threads = []
        for plat, channel in active:
            t = threading.Thread(
                target=self._monitor_platform,
                args=(plat, channel, clips_dir, editor, uploader, desc_gen, title_gen),
                kwargs={"config": config},
                daemon=True,
            )
            t.start()
            threads.append(t)

        try:
            # Bloqueia até stop() ser chamado OU todas as threads morrerem
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=5)
                if self._stop_event.is_set():
                    break
                # Watchdog: se TODAS as threads filhas morreram, não há mais nada
                alive = [t for t in threads if t.is_alive()]
                if not alive:
                    logger.warning(
                        "⚠️  Todas as threads de canal encerraram. "
                        "Verifique se as lives estão ativas."
                    )
                    self._stop_event.set()
                    break
        finally:
            # Cleanup imediato: aguarda threads filhas encerrarem (max 15s cada)
            logger.info("⏹ Aguardando threads de canal encerrarem...")
            for t in threads:
                t.join(timeout=15)
            logger.info("✅ Todas as threads encerradas.")
            # Libera referências para GC e evita uso acidental após stop
            self._config     = None
            self._editor     = None
            self._output_dir = None
            self._clips_dir  = None

    # ──────────────────────────────────────────────────────────────────────────
    # Monitor por canal (uma thread por canal)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_channel_name(channel: str) -> str:
        ch = channel.strip().rstrip("/")
        for prefix in [
            "https://www.twitch.tv/", "https://twitch.tv/",
            "https://kick.com/",       "https://www.kick.com/",
            "https://www.youtube.com/","https://youtube.com/",
        ]:
            if ch.lower().startswith(prefix):
                ch = ch[len(prefix):]
        ch = ch.lstrip("@")
        for c in r'\/:*?"<>|':
            ch = ch.replace(c, "_")
        return ch or "canal"

    def _monitor_platform(self, platform: str, channel: str,
                           clips_dir: Path, editor, uploader, desc_gen,
                           title_gen=None, config: dict = None) -> None:
        """Monitora UM canal em sua própria thread. try/finally garante cleanup."""
        from stream_capture  import StreamCapture
        from moment_detector import MomentDetector
        import copy

        cfg = copy.deepcopy(config)
        cfg["streams"][platform]["channel"]    = channel
        cfg["streams"][platform]["channel_id"] = channel

        clean    = self._clean_channel_name(channel)
        label    = f"{platform.upper()}:{clean}"
        capture  = StreamCapture(cfg, str(clips_dir))
        processed: set = set()

        try:
            raw_file = capture.start_recording(platform)
        except Exception as e:
            logger.error(f"❌ [{label}] Captura falhou: {e}")
            return

        detector = MomentDetector(cfg, elapsed_fn=capture.get_elapsed_seconds)
        detector.start(raw_file)

        approval_mode  = config.get("system", {}).get("approval_mode", False)
        approval_top_n = int(config.get("system", {}).get("approval_top_n", 12))

        try:
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=1)
                if self._stop_event.is_set():
                    break

                moments = detector.all_moments

                # ── Modo aprovação: enfileira para revisão da UI ───────────────
                if approval_mode:
                    candidates = []
                    for m in moments:
                        key = (label, round(m.start_sec, 1), m.reason)
                        if key in processed:
                            continue
                        if capture.get_elapsed_seconds() < m.end_sec + 2:
                            continue
                        candidates.append((key, m))

                    # Ordena por score desc e toma top N por ciclo
                    candidates.sort(key=lambda x: x[1].score, reverse=True)
                    for key, m in candidates[:approval_top_n]:
                        processed.add(key)
                        logger.info(
                            f"📋 [{label}] [{m.reason}] "
                            f"{m.start_sec:.1f}s→{m.end_sec:.1f}s  "
                            f"score={m.score:.2f} → pendente aprovação"
                        )
                        try:
                            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                            raw_name = f"pending_{platform}_{clean}_{ts}_{m.reason}"
                            raw_clip = clips_dir / f"raw_{raw_name}.ts"

                            min_dur = config["detection"]["audio_spike"]["min_clip_duration"]
                            seg_start, seg_end = m.start_sec, m.end_sec
                            if (seg_end - seg_start) < min_dur:
                                pad       = (min_dur - (seg_end - seg_start)) / 2
                                seg_start = max(0.0, seg_start - pad)
                                seg_end   = seg_end + pad

                            capture.extract_segment(raw_file, seg_start, seg_end, raw_clip)

                            self.pending_store.add(m, raw_clip, platform, clean)
                            entry = self.pending_store.get(m.moment_id)
                            if entry:
                                self.on_moment_pending(entry)
                        except Exception as e:
                            logger.error(f"❌ [{label}] Pending {m.reason}: {e}")
                    continue   # ← não cai no bloco de extração automática

                # ── Modo automático: extrai e posta sem aprovação ─────────────
                for m in moments:
                    key = (label, round(m.start_sec, 1), m.reason)
                    if key in processed:
                        continue
                    if capture.get_elapsed_seconds() < m.end_sec + 2:
                        continue

                    processed.add(key)
                    logger.info(
                        f"🎯 [{label}] [{m.reason}] "
                        f"{m.start_sec:.1f}s → {m.end_sec:.1f}s  score={m.score:.2f}"
                    )

                    try:
                        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                        name     = f"clip_{platform}_{clean}_{ts}_{m.reason}"
                        raw_clip = clips_dir / f"raw_{name}.ts"

                        min_dur = config["detection"]["audio_spike"]["min_clip_duration"]
                        start, end = m.start_sec, m.end_sec
                        if (end - start) < min_dur:
                            pad   = (min_dur - (end - start)) / 2
                            start = max(0.0, start - pad)
                            end   = end + pad

                        capture.extract_segment(raw_file, start, end, raw_clip)

                        # Título context-aware via TitleGenerator
                        moment_entry = {
                            "reason":   m.reason,
                            "keywords": m.keywords,
                            "score":    m.score,
                            "extra":    m.extra,
                        }
                        if title_gen:
                            auto_title = title_gen.generate(raw_clip, moment_entry)
                        else:
                            auto_title = "CLIP ÉPICO DA LIVE!"

                        edited = editor.process(raw_clip, name, title_text=auto_title)
                        raw_clip.unlink(missing_ok=True)

                        logger.info("✨ Gerando descrições com IA...")
                        all_results = {}

                        active_upload = {
                            "tiktok":    config["upload"]["tiktok"]["enabled"],
                            "instagram": config["upload"]["instagram"]["enabled"],
                            "youtube":   config["upload"]["youtube"]["enabled"],
                        }

                        for upload_plat, enabled in active_upload.items():
                            if not enabled:
                                continue
                            try:
                                desc    = desc_gen.generate(
                                    video_path=edited, platform=upload_plat,
                                    channel=clean, reason=m.reason,
                                )
                                title   = desc["title"]
                                caption = desc["caption"]
                                tags    = desc["hashtags"]
                                logger.info(f"📝 [{upload_plat}] {title}")

                                if upload_plat == "tiktok" and uploader.tiktok:
                                    all_results["tiktok"] = uploader.tiktok.upload(
                                        edited, caption, tags)
                                elif upload_plat == "instagram" and uploader.instagram:
                                    all_results["instagram"] = uploader.instagram.upload(
                                        edited, caption)
                                elif upload_plat == "youtube" and uploader.youtube:
                                    all_results["youtube"] = uploader.youtube.upload(
                                        edited, title, caption)

                                delay = config["upload"]["delay_between_posts"]
                                if all_results.get(upload_plat) and delay > 0:
                                    logger.info(f"⏳ Delay {delay}s entre uploads...")
                                    self._stop_event.wait(timeout=delay)

                            except Exception as e:
                                logger.error(f"❌ Upload {upload_plat}: {e}")
                                all_results[upload_plat] = False

                        ok = [p for p, v in all_results.items() if v]

                        # ── Contadores com lock ───────────────────────────────
                        with self._stats_lock:
                            self.clips_today += 1
                            self.clips_total += 1

                        if ok:
                            logger.info(f"✅ [{label}] Postado em: {', '.join(ok)}")
                        else:
                            logger.info(f"💾 [{label}] Clip salvo: {edited}")

                        self.on_clip_posted({
                            "name":      name,
                            "reason":    m.reason,
                            "platforms": ok,
                            "time":      datetime.now().strftime("%H:%M"),
                        })

                        if config["system"]["cleanup_after_upload"] and ok:
                            edited.unlink(missing_ok=True)

                    except Exception as e:
                        logger.error(f"❌ [{label}] Clip {m.reason}: {e}")

        except KeyboardInterrupt:
            logger.info(f"⌨️  [{label}] Ctrl+C na thread do canal.")
        finally:
            # Cleanup imediato — não depende de atexit
            logger.info(f"⏹ [{label}] Encerrando detector e captura...")
            try:
                detector.stop()
            except Exception as e:
                logger.debug(f"detector.stop: {e}")
            try:
                capture.stop_recording()
            except Exception as e:
                logger.debug(f"capture.stop_recording: {e}")
            logger.info(f"✅ [{label}] Encerrado.")

    # ──────────────────────────────────────────────────────────────────────────
    # Geração de clip a partir de momento pendente (chamado pela API)
    # ──────────────────────────────────────────────────────────────────────────

    def generate_clip_from_moment(self, moment_id: str,
                                   title_text: str = None) -> dict:
        """
        Processa um momento da PendingStore:
          1. Busca o raw_clip pré-extraído
          2. Aplica VideoEditor (formato, legendas…)
          3. Grava sidecar .json com metadados (scores, reason, etc.)
          4. Remove da PendingStore e limpa raw_clip do disco

        Thread-safe — pode ser chamado de qualquer thread.
        Retorna {"ok": bool, "path": str, "name": str} ou {"ok": False, "error": str}.
        """
        entry = self.pending_store.get(moment_id)
        if not entry:
            return {"ok": False, "error": f"Momento {moment_id[:8]} não encontrado"}

        raw_clip = Path(entry["raw_file"])
        if not raw_clip.exists():
            self.pending_store.remove_and_cleanup(moment_id)
            return {"ok": False, "error": f"Raw clip ausente: {raw_clip.name}"}

        config = self._config
        editor = self._editor
        if config is None or editor is None:
            return {"ok": False, "error": "Engine não está rodando — inicie o engine primeiro"}

        # Marca como "processing" para impedir reject concorrente de deletar o raw_file
        self.pending_store.set_processing(moment_id)
        try:
            from feedback import load_weights, score_moment_advanced, update_channel_history
            weights     = load_weights()
            duration    = entry.get("end_sec", 0) - entry.get("start_sec", 0)
            final_score = score_moment_advanced(
                entry, weights,
                duration=duration,
                channel=entry.get("channel"),
            )

            ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
            name        = (
                f"clip_{entry['platform']}_{entry['channel']}"
                f"_{ts}_{entry['reason']}"
            )

            output_dir = Path(config["system"]["output_folder"])
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"🎬 Gerando clip do momento {moment_id[:8]} [{entry['reason']}]…")

            # Título para o banner: custom (da UI) ou automático por reason
            if not title_text:
                moment_entry = {
                    "reason":   entry["reason"],
                    "keywords": entry.get("keywords", []),
                    "score":    entry.get("score", 1.0),
                    "extra":    entry.get("extra", {}),
                }
                if self._title_gen:
                    title_text = self._title_gen.generate(raw_clip, moment_entry)
                else:
                    title_text = "CLIP ÉPICO DA LIVE!"

            edited = editor.process(raw_clip, name, title_text=title_text)

            # ── Sidecar JSON ──────────────────────────────────────────────────
            sidecar = edited.with_suffix(".json")
            reasons  = entry.get("extra", {}).get("reasons", [entry["reason"]])
            sidecar.write_text(json.dumps({
                "moment_id":    moment_id,
                "reason":       entry["reason"],
                "reasons":      reasons,
                "platform":     entry["platform"],
                "channel":      entry["channel"],
                "start_sec":    entry["start_sec"],
                "end_sec":      entry["end_sec"],
                "base_score":   round(entry["score"], 4),
                "final_score":  round(final_score, 4),
                "keywords":     entry.get("keywords", []),
                "generated_at": datetime.now().isoformat(),
            }, indent=2, ensure_ascii=False), encoding="utf-8")

            # Remove da store (apaga raw_clip se não referenciado por outros)
            self.pending_store.remove_and_cleanup(moment_id)

            with self._stats_lock:
                self.clips_today += 1
                self.clips_total += 1

            # Registra no histórico do canal
            try:
                update_channel_history(
                    entry.get("channel", "unknown"),
                    entry["reason"],
                    final_score,
                )
            except Exception:
                pass

            info = {
                "name":      name,
                "path":      str(edited),
                "reason":    entry["reason"],
                "platforms": [],
                "channel":   entry.get("channel", ""),
                "time":      datetime.now().strftime("%H:%M"),
                "moment_id": moment_id,
            }
            self.on_clip_posted(info)

            logger.info(
                f"✅ Clip gerado: {edited.name}  "
                f"(score base={entry['score']:.2f} → final={final_score:.2f})"
            )
            return {"ok": True, "path": str(edited), "name": edited.name}

        except Exception as e:
            logger.error(f"❌ generate_clip_from_moment: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {"ok": False, "error": str(e)}
        finally:
            self.pending_store.clear_processing(moment_id)
