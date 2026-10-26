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

import os
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

# Telemetria — import lazy via try/except para tolerância em PR1.5
try:
    from metrics import incr as _metrics_incr
except ImportError:
    def _metrics_incr(*_args, **_kw): pass
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
                 on_moment_pending=None, pending_store=None, publish_queue=None):
        self.on_status         = on_status         or (lambda s: None)
        self.on_log            = on_log            or (lambda lvl, msg: None)
        self.on_clip_posted    = on_clip_posted    or (lambda info: None)
        self.on_moment_pending = on_moment_pending or (lambda entry: None)
        # Fila de publicação (Patch 6): o modo automático só enfileira.
        self.publish_queue     = publish_queue

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

        # ── Observabilidade: funil + heartbeat por canal (Patch 10) ───────────
        # Responde "o bot está vivo mas não gera clips?" sem depender de log.
        self._channel_health: dict = {}
        self._health_lock = threading.Lock()

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
    # Observabilidade: funil de momentos + heartbeat por canal (Patch 10)
    # ──────────────────────────────────────────────────────────────────────────

    # Estágios do funil, em ordem de pipeline. Um momento que chega ao fim
    # passou por: detected → extracted → rendered → published.
    # Qualquer "gate" é uma saída (descarte) — a soma dos gates + rendered
    # deve reconciliar com detected (a menos dos que ainda estão em voo).
    FUNNEL_STAGES = (
        "detected",        # momento único, maduro, entrou no pipeline
        "score_gate",      # descartado: score avançado < 0.15
        "reaction_gate",   # descartado: score efetivo (reaction) < 0.10
        "extracted",       # segmento .ts extraído com sucesso
        "offline_gate",    # descartado: tela offline/placeholder
        "quality_gate",    # descartado: silêncio + estático
        "editor_discard",  # descartado pelo editor (comercial / blacklist)
        "rendered",        # clip final gerado
        "published",       # upload confirmado em >=1 plataforma
        "pending_queued",  # (approval-mode) enfileirado para aprovação
        "errors",          # exceção no processamento do momento
    )

    def _new_health(self) -> dict:
        now = time.time()
        return {
            "state":          "starting",   # starting|capturing|frozen|reconnecting|stopped
            "started_at":     now,
            "last_tick":      now,          # loop do canal vivo
            "last_moment":    None,         # último momento detectado
            "last_clip":      None,         # último clip renderizado
            "attempt":        1,            # nº da tentativa de captura
            "funnel":         {s: 0 for s in self.FUNNEL_STAGES},
            "moments_active": 0,            # tamanho do snapshot atual (gauge)
        }

    def _funnel(self, label: str, stage: str, n: int = 1,
                platform: str = "", channel: str = "") -> None:
        """
        Incrementa um estágio do funil do canal e emite a métrica.
        Nunca lança — observabilidade jamais derruba produção.
        """
        try:
            with self._health_lock:
                h = self._channel_health.setdefault(label, self._new_health())
                h["funnel"][stage] = h["funnel"].get(stage, 0) + n
                now = time.time()
                h["last_tick"] = now
                if stage == "detected":
                    h["last_moment"] = now
                elif stage == "rendered":
                    h["last_clip"] = now
            _metrics_incr(f"funnel.{stage}", n=n,
                          platform=platform or "?", channel=channel or label)
        except Exception:
            pass

    def _touch_channel(self, label: str, **fields) -> None:
        """Atualiza heartbeat/estado do canal (state, attempt, moments_active)."""
        try:
            with self._health_lock:
                h = self._channel_health.setdefault(label, self._new_health())
                h["last_tick"] = time.time()
                h.update(fields)
        except Exception:
            pass

    def get_health(self) -> dict:
        """
        Snapshot serializável do funil + heartbeat de todos os canais.
        Consumido pela UI (get_status) e pelo resumo periódico de log.

        Campos derivados (idades em segundos) evitam que o consumidor
        precise saber o relógio do engine.
        """
        now = time.time()
        out = {}
        with self._health_lock:
            for label, h in self._channel_health.items():
                out[label] = {
                    "state":            h["state"],
                    "attempt":          h["attempt"],
                    "uptime_sec":       round(now - h["started_at"], 1),
                    "tick_age_sec":     round(now - h["last_tick"], 1),
                    "moment_age_sec":   (round(now - h["last_moment"], 1)
                                         if h["last_moment"] else None),
                    "clip_age_sec":     (round(now - h["last_clip"], 1)
                                         if h["last_clip"] else None),
                    "moments_active":   h["moments_active"],
                    "funnel":           dict(h["funnel"]),
                }
        return out

    def _log_health_summary(self) -> None:
        """
        Resumo periódico no log — a resposta rápida para
        "está rodando, mas por que não saiu clip?".
        """
        # Disco de estado: se a persistência está quebrada, isto precisa
        # aparecer ANTES do funil — momentos "ativos" que não estão no
        # pending.json desaparecem quando o app fecha.
        save_err = getattr(self.pending_store, "last_save_error", None)
        if save_err:
            logger.error(
                f"❌ [storage] pending.json não está sendo gravado ({save_err}). "
                "Momentos aprovados existem só em memória."
            )
        load_warn = getattr(self.pending_store, "load_warning", None)
        if load_warn:
            logger.warning(
                f"⚠️  [storage] {load_warn} — .ts da sessão anterior preservados "
                "até o arquivo voltar a ser lido."
            )
        try:
            import credentials as _c
            for _kind in ("credentials", "settings"):
                _why = _c.degraded_reason(_kind)
                if _why:
                    logger.error(
                        f"❌ [storage] {_kind} ilegível ({_why}) — escrita bloqueada."
                    )
        except Exception:
            pass

        health = self.get_health()
        if not health:
            return
        for label, h in health.items():
            f = h["funnel"]
            gates = (
                f"score={f['score_gate']} reaction={f['reaction_gate']} "
                f"offline={f['offline_gate']} quality={f['quality_gate']} "
                f"editor={f['editor_discard']}"
            )
            last_clip = (f"{h['clip_age_sec']:.0f}s atrás"
                         if h["clip_age_sec"] is not None else "nenhum")
            logger.info(
                f"📊 [{label}] {h['state']} up={h['uptime_sec']/60:.0f}min | "
                f"detected={f['detected']} → extracted={f['extracted']} → "
                f"rendered={f['rendered']} → published={f['published']} | "
                f"descartes: {gates} | erros={f['errors']} | "
                f"ativos={h['moments_active']} | último clip: {last_clip}"
            )

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
                    "safe_margin_right": 60,
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
                "content_filter": {
                    "enabled": True,
                    "blacklist_keywords": [
                        "aposta", "aposte", "apostar", "apostas",
                        "bet", "bets", "betting", "odds", "stake",
                        "cassino", "casino", "roleta", "slots",
                        "pix", "deposite", "depósito",
                        "código promocional", "promo code",
                        "patrocinado", "sponsor",
                    ],
                    "action": "skip",
                },
                "clip_analysis": {
                    "enabled": True,
                    "reaction_weight": 0.5,
                    "hook_pre_roll": 3.0,
                    "audio_norm": 0.06,
                    "motion_norm": 12.0,
                    "quality_gate": {
                        "enabled": True,
                        "min_rms": 0.01,
                        "min_motion": 5.0,
                    },
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
                        str(Path(os.environ.get("APPDATA", Path.home())) / "ClipBot" / "youtube_credentials.json")
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

    # ──────────────────────────────────────────────────────────────────────────
    # Disk monitor (warn em <5GB, halt em <1GB)
    # ──────────────────────────────────────────────────────────────────────────

    _DISK_WARN_GB = 5.0
    _DISK_HALT_GB = 1.0
    _DISK_CHECK_INTERVAL_SEC = 60.0

    # GC de órfãos: pendentes extraídos que não estão mais na store.
    # Grace de 10 min evita corrida com extract_segment → pending_store.add().
    _GC_ORPHAN_INTERVAL_SEC = 600.0
    _GC_ORPHAN_MIN_AGE_SEC  = 600.0
    # Resumo do funil no log (Patch 10). 5min: legível sem virar spam.
    _HEALTH_LOG_INTERVAL_SEC = 300.0

    def _gc_orphan_pending_clips(self, clips_dir: Path) -> int:
        """
        Apaga raw_pending_*.ts que não são referenciados pela PendingStore.

        Complementa o Patch 1: quando um .ts está em uso no instante da
        remoção, ele é preservado — e recolhido aqui depois. Também limpa
        pendentes deixados por crashes/restarts anteriores.

        NÃO toca no .ts de captura ativo: o padrão raw_pending_* exclui os
        arquivos de gravação contínua (raw_{platform}_{canal}_{ts}.ts).
        """
        try:
            referenced = {
                os.path.normpath(e.get("raw_file", ""))
                for e in self.pending_store.list_all()
                if e.get("raw_file")
            }
        except Exception as e:
            logger.debug(f"gc_orphan: store indisponível ({e})")
            return 0

        # pending.json do boot ilegível/corrompido: a store não sabe quem é
        # dono dos .ts que já existiam. Apagá-los aqui era a segunda metade da
        # perda de dados do boot degradado (o arquivo era sobrescrito e os .ts
        # que ele descrevia iam embora 10 min depois).
        cutoff = getattr(self.pending_store, "ownership_cutoff", None)

        now = time.time()
        removed = 0
        spared = 0
        freed_mb = 0.0
        for f in clips_dir.glob("raw_pending_*.ts"):
            try:
                if os.path.normpath(str(f)) in referenced:
                    continue
                mtime = f.stat().st_mtime
                if (now - mtime) < self._GC_ORPHAN_MIN_AGE_SEC:
                    continue          # recém-extraído: pode ainda não estar na store
                if cutoff is not None and mtime < cutoff:
                    spared += 1
                    continue          # anterior ao boot, dono desconhecido
                size_mb = f.stat().st_size / 1e6
                f.unlink(missing_ok=True)
                removed += 1
                freed_mb += size_mb
            except Exception:
                pass

        if removed:
            logger.info(
                f"🧹 GC órfãos: {removed} raw pendente(s) removido(s) "
                f"({freed_mb:.0f} MB liberados)"
            )
            _metrics_incr("disk.orphan_gc_removed", n=removed)
        if spared:
            logger.debug(
                f"GC órfãos: {spared} .ts anterior(es) ao boot preservado(s) — "
                "pending.json do boot não foi lido, dono desconhecido"
            )
        return removed

    def _disk_monitor_loop(self, clips_dir: Path) -> None:
        """
        Roda em thread daemon paralela. A cada 60s checa o disco do clips_dir.
        Em <5GB livre: warn (1× por estado).
        Em <1GB livre: warn final + set _stop_event para parar engine.
        A cada 10min: GC de raw pendentes órfãos (Patch 1b).
        """
        import shutil
        last_state = "ok"  # ok | warn | halt
        last_gc_t  = time.time()
        last_health_t = time.time()
        while not self._stop_event.is_set():
            try:
                # ── GC periódico de órfãos ───────────────────────────────────
                if time.time() - last_gc_t >= self._GC_ORPHAN_INTERVAL_SEC:
                    last_gc_t = time.time()
                    try:
                        self._gc_orphan_pending_clips(clips_dir)
                    except Exception as e:
                        logger.debug(f"gc_orphan falhou: {e}")

                # ── Resumo do funil por canal (Patch 10) ─────────────────────
                if time.time() - last_health_t >= self._HEALTH_LOG_INTERVAL_SEC:
                    last_health_t = time.time()
                    try:
                        self._log_health_summary()
                    except Exception as e:
                        logger.debug(f"health summary falhou: {e}")

                free_gb = shutil.disk_usage(str(clips_dir)).free / 1e9
                if free_gb < self._DISK_HALT_GB:
                    if last_state != "halt":
                        logger.error(
                            f"🛑 Disk crítico: {free_gb:.2f}GB livres em {clips_dir} "
                            "— parando engine."
                        )
                        last_state = "halt"
                    self._stop_event.set()
                    return
                elif free_gb < self._DISK_WARN_GB:
                    if last_state != "warn":
                        logger.warning(
                            f"⚠️  Disk baixo: {free_gb:.2f}GB livres em {clips_dir} "
                            f"(threshold {self._DISK_WARN_GB:.0f}GB). "
                            "Pretende reduzir tempo de live ou limpar clips."
                        )
                        last_state = "warn"
                else:
                    if last_state != "ok":
                        logger.info(
                            f"✅ Disk OK: {free_gb:.2f}GB livres (recuperado)"
                        )
                        last_state = "ok"
            except Exception as e:
                logger.debug(f"disk_monitor: {e}")
            # Sleep interruptível (acorda em stop)
            if self._stop_event.wait(timeout=self._DISK_CHECK_INTERVAL_SEC):
                break

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

        # SÓ nomes que o ClipBot cria. clips_folder e output_folder são campos
        # livres na UI: com "*.mp4"/"*.json", quem apontava a saída para Vídeos
        # ou para o Desktop perdia todo vídeo e JSON pessoal com mais de 7 dias.
        #   raw_*.ts   → captura contínua e segmentos (raw_pending_/raw_clip_)
        #   clip_*.mp4 → clip renderizado;  clip_*.json → sidecar
        patterns = ("raw_*.ts", "clip_*.mp4", "clip_*.json")

        cutoff = getattr(self.pending_store, "ownership_cutoff", None)

        removed = 0
        seen: set = set()                 # clips_dir == output_dir é comum
        for folder in (clips_dir, output_dir):
            if not folder.exists():
                continue
            for pattern in patterns:
                for f in folder.glob(pattern):
                    key = os.path.normcase(os.path.normpath(str(f)))
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        mtime = f.stat().st_mtime
                        if now - mtime < MAX_AGE_SECS:
                            continue
                        if os.path.normpath(str(f)) in pending_files:
                            continue
                        # Dono desconhecido (pending.json do boot ilegível):
                        # pendentes anteriores ao boot ficam. A captura contínua
                        # (raw_{plataforma}_...) nunca é pendente e segue o GC.
                        if (cutoff is not None and f.name.startswith("raw_pending_")
                                and mtime < cutoff):
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

        # Disk monitor — para detectar disk-fill em sessão longa
        disk_thread = threading.Thread(
            target=self._disk_monitor_loop,
            args=(clips_dir,),
            name="disk-monitor",
            daemon=True,
        )
        disk_thread.start()

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

    # Tempos do monitor por canal. Atributos (não locais) para os testes
    # encurtarem — o comportamento é o mesmo, só mais rápido.
    _TICK_SEC           = 1.0
    _RETRY_BACKOFF_INIT = 30.0
    _RETRY_BACKOFF_MAX  = 600.0    # 10min teto
    _CRASH_BACKOFF_INIT = 30.0
    _CRASH_BACKOFF_MAX  = 600.0

    def _monitor_platform(self, platform: str, channel: str,
                           clips_dir: Path, editor, uploader, desc_gen,
                           title_gen=None, config: dict = None) -> None:
        """
        Thread de um canal: roda o monitor e o reinicia se ele cair.

        Rede de segurança para o que fica fora do loop interno de
        _monitor_platform_attempts (construtor da captura, backoff, cleanup).
        Antes, uma exceção ali encerrava a thread em silêncio — com vários
        canais, um deles simplesmente parava; com um só, o engine parava.
        """
        clean = self._clean_channel_name(channel)
        label = f"{platform.upper()}:{clean}"
        delay = self._CRASH_BACKOFF_INIT
        while not self._stop_event.is_set():
            try:
                self._monitor_platform_attempts(
                    platform, channel, clips_dir, editor, uploader, desc_gen,
                    title_gen=title_gen, config=config,
                )
                return
            except Exception:
                logger.exception(
                    f"❌ [{label}] Monitor do canal caiu com erro inesperado; "
                    f"reiniciando em {delay:.0f}s"
                )
                _metrics_incr("capture.monitor_crash", platform=platform)
                self._funnel(label, "errors", platform=platform, channel=clean)
                self._touch_channel(label, state="reconnecting")
                if self._stop_event.wait(timeout=delay):
                    return
                delay = min(self._CRASH_BACKOFF_MAX, delay * 2)

    def _monitor_platform_attempts(self, platform: str, channel: str,
                                   clips_dir: Path, editor, uploader, desc_gen,
                                   title_gen=None, config: dict = None) -> None:
        """
        Monitora UM canal em sua própria thread, com **auto-restart**.

        Loop externo de retry com backoff exponencial:
          • Falha de start_recording (offline/network) → backoff 30s→60s→120s→...→600s
          • Watchdog detecta stream FROZEN >120s → restart isolado do canal
          • Captura rodando >5min com sucesso → reseta backoff pra 30s
          • Engine stop_event → sai imediatamente do loop

        Cada attempt cria novo StreamCapture + MomentDetector. O `processed` dict
        é local de attempt — só serve pra evitar processar 2× o mesmo momento
        durante uma única gravação contínua. Após reconexão, novo .ts file =
        novos timestamps relativos.
        """
        from stream_capture  import StreamCapture
        from moment_detector import MomentDetector
        import copy

        cfg = copy.deepcopy(config)
        cfg["streams"][platform]["channel"]    = channel
        cfg["streams"][platform]["channel_id"] = channel

        clean    = self._clean_channel_name(channel)
        label    = f"{platform.upper()}:{clean}"

        # ── Estado do loop de retry (escopo da thread) ────────────────────────
        _RETRY_BACKOFF_INIT = self._RETRY_BACKOFF_INIT
        _RETRY_BACKOFF_MAX  = self._RETRY_BACKOFF_MAX
        _RETRY_BACKOFF_MULT = 2.0
        _RETRY_RESET_AFTER  = 300.0   # 5min de uptime resetam backoff
        backoff_s = _RETRY_BACKOFF_INIT
        attempt   = 0

        while not self._stop_event.is_set():
            attempt += 1
            attempt_start_ts = time.time()
            if attempt > 1:
                logger.info(
                    f"🔄 [{label}] Reconectando (attempt #{attempt})..."
                )
                _metrics_incr("capture.reconnect_attempt", platform=platform)

            # Per-attempt state
            capture       = StreamCapture(cfg, str(clips_dir))
            processed: dict = {}
            last_clip_time: float = 0.0
            _PRUNE_AGE    = 3600.0
            _restart_now  = False   # flag setada pelo watchdog em frozen prolongado

            try:
                raw_file = capture.start_recording(platform)
            except Exception as e:
                logger.error(f"❌ [{label}] Captura falhou (attempt #{attempt}): {e}")
                self._touch_channel(label, state="reconnecting", attempt=attempt)
                # Backoff antes de tentar de novo. Se stop_event veio durante
                # o wait, sai do loop externo.
                if self._stop_event.wait(timeout=backoff_s):
                    return
                backoff_s = min(_RETRY_BACKOFF_MAX, backoff_s * _RETRY_BACKOFF_MULT)
                continue

            detector = None   # criado dentro do try (ver abaixo)

            approval_mode  = config.get("system", {}).get("approval_mode", False)
            approval_top_n = int(config.get("system", {}).get("approval_top_n", 12))

            # ── Watchdog: detecta live caída (raw .ts congelado) ──────────────────
            # Stream Twitch best ~6 Mbps = ~45 MB/min crescimento esperado.
            # Se raw_file não cresce em FROZEN_THRESHOLD segundos = anomalia
            # (live caiu, network glitch, ffmpeg travou em reconnect HLS).
            _WD_CHECK_INTERVAL = 15.0     # entre samples
            _WD_FROZEN_THRESHOLD = 60.0   # sem crescimento → frozen
            _WD_REWARN_INTERVAL = 120.0   # re-emite warn a cada 2min se permanecer frozen
            _wd_last_size = 0
            _wd_last_growth_t = time.time()
            _wd_last_check_t = time.time()
            _wd_last_warn_t = 0.0
            _wd_state = "ok"               # "ok" | "frozen"

            try:
                # Dentro do try: se o detector não subir, o finally ainda para
                # a captura. Antes ficava fora — a thread morria com o ffmpeg
                # rodando.
                # platform/channel explícitos: o detector de chat da Twitch só é
                # criado para threads da Twitch, no canal DESTA thread (P1-4).
                detector = MomentDetector(
                    cfg, elapsed_fn=capture.get_elapsed_seconds,
                    platform=platform, channel=channel,
                )
                detector.start(raw_file)
                self._touch_channel(label, state="capturing", attempt=attempt)

                while not self._stop_event.is_set():
                    self._stop_event.wait(timeout=self._TICK_SEC)
                    if self._stop_event.is_set():
                        break

                    moments = detector.all_moments
                    # Heartbeat do loop do canal + gauge de momentos ativos.
                    # tick_age_sec alto = thread travada (ex.: presa em upload).
                    self._touch_channel(label, moments_active=len(moments))

                    # Poda periódica do dict processed (>1h)
                    now_ts = time.time()
                    if len(processed) > 500:
                        processed = {k: t for k, t in processed.items()
                                     if now_ts - t < _PRUNE_AGE}

                    # ── Watchdog: live caída? ─────────────────────────────────────
                    if now_ts - _wd_last_check_t >= _WD_CHECK_INTERVAL:
                        _wd_last_check_t = now_ts
                        try:
                            cur_size = raw_file.stat().st_size
                        except Exception:
                            cur_size = _wd_last_size

                        if cur_size > _wd_last_size:
                            # Cresceu: live OK. Resetar timer e estado se estava frozen.
                            bytes_per_s = (cur_size - _wd_last_size) / _WD_CHECK_INTERVAL
                            _wd_last_size = cur_size
                            _wd_last_growth_t = now_ts
                            if _wd_state == "frozen":
                                logger.info(f"✅ [{label}] Stream voltou (live recuperou)")
                                _metrics_incr("capture.recovered", platform=platform)
                                _wd_state = "ok"
                                _wd_last_warn_t = 0.0
                                self._touch_channel(label, state="capturing")
                            # Gauge informativo (uma observação por canal)
                            try:
                                from metrics import observe as _m_obs
                                _m_obs("capture.bytes_per_sec", bytes_per_s,
                                       platform=platform)
                            except Exception:
                                pass
                        else:
                            # Sem crescimento. Quanto tempo está parado?
                            frozen_for = now_ts - _wd_last_growth_t
                            if frozen_for >= _WD_FROZEN_THRESHOLD:
                                should_warn = (
                                    _wd_state == "ok"   # primeira vez
                                    or (now_ts - _wd_last_warn_t >= _WD_REWARN_INTERVAL)
                                )
                                if should_warn:
                                    logger.warning(
                                        f"⚠️  [{label}] Stream FROZEN — raw .ts sem "
                                        f"crescimento há {frozen_for:.0f}s "
                                        f"(live caiu / network glitch / ffmpeg travou)"
                                    )
                                    _metrics_incr("capture.frozen",
                                                  platform=platform, channel=clean)
                                    _wd_state = "frozen"
                                    _wd_last_warn_t = now_ts
                                    self._touch_channel(label, state="frozen")

                            # Trigger AUTO-RESTART do canal: frozen >120s.
                            # Damos 60s pro warn aparecer + 60s pra streamlink tentar
                            # se recuperar. Se nada, desiste e reconecta com backoff.
                            if frozen_for >= 120.0:
                                logger.warning(
                                    f"🔄 [{label}] FROZEN >120s — auto-restart do canal"
                                )
                                _metrics_incr("capture.auto_restart",
                                              platform=platform, channel=clean)
                                _restart_now = True
                                break   # sai do inner while → cleanup → outer retry

                    # ── Modo aprovação: enfileira para revisão da UI ───────────────
                    if approval_mode:
                        candidates = []
                        for m in moments:
                            # Identidade = membros (não start/reason, que mudavam
                            # quando o grupo crescia) e só grupo FECHADO sai —
                            # o gate antigo, end + 2 s, liberava grupo aberto.
                            key = (label, detector.members(m))
                            if key in processed:
                                continue
                            if not detector.is_settled(m, capture.get_elapsed_seconds()):
                                continue
                            candidates.append((key, m))

                        # Ordena por score desc e toma top N por ciclo
                        candidates.sort(key=lambda x: x[1].score, reverse=True)
                        for key, m in candidates[:approval_top_n]:
                            processed[key] = now_ts
                            detector.consume(m)
                            logger.info(
                                f"📋 [{label}] [{m.reason}] "
                                f"{m.start_sec:.1f}s→{m.end_sec:.1f}s  "
                                f"score={m.score:.2f} → pendente aprovação"
                            )
                            try:
                                ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                                # moment_id no nome: vários grupos fechados podem
                                # sair no mesmo tick, e o mesmo segundo + reason
                                # fazia a 2ª extração sobrescrever o .ts da 1ª.
                                raw_name = f"pending_{platform}_{clean}_{ts}_{m.moment_id[:8]}_{m.reason}"
                                raw_clip = clips_dir / f"raw_{raw_name}.ts"

                                min_dur = config["detection"]["audio_spike"]["min_clip_duration"]
                                seg_start, seg_end = self._segment_bounds(
                                    m.start_sec, m.end_sec, min_dur,
                                    available=capture.get_elapsed_seconds(),
                                )

                                capture.extract_segment(raw_file, seg_start, seg_end, raw_clip)
                                self._funnel(label, "detected", platform=platform, channel=clean)
                                self._funnel(label, "extracted", platform=platform, channel=clean)

                                self.pending_store.add(m, raw_clip, platform, clean,
                                                       segment=(seg_start, seg_end))
                                self._funnel(label, "pending_queued",
                                             platform=platform, channel=clean)
                                entry = self.pending_store.get(m.moment_id)
                                if entry:
                                    self.on_moment_pending(entry)
                            except Exception as e:
                                logger.error(f"❌ [{label}] Pending {m.reason}: {e}")
                                self._funnel(label, "errors", platform=platform, channel=clean)
                        continue   # ← não cai no bloco de extração automática

                    # ── Modo automático: extrai e posta sem aprovação ─────────────
                    for m in moments:
                        key = (label, detector.members(m))
                        if key in processed:
                            continue
                        if not detector.is_settled(m, capture.get_elapsed_seconds()):
                            continue

                        processed[key] = now_ts
                        detector.consume(m)
                        self._funnel(label, "detected", platform=platform, channel=clean)
                        logger.info(
                            f"🎯 [{label}] [{m.reason}] "
                            f"{m.start_sec:.1f}s → {m.end_sec:.1f}s  score={m.score:.2f}"
                        )

                        try:
                            # Score avançado com recency e chat. Dentro do try:
                            # chat_rate_ratio lê estado da thread do IRC, e um
                            # erro aqui deve pular ESTE momento, não o canal.
                            from feedback import score_moment_advanced
                            secs_since = (time.time() - last_clip_time) if last_clip_time > 0 else None
                            chat_ratio = None
                            if hasattr(detector, 'chat') and detector.chat and hasattr(detector.chat, 'chat_rate_ratio'):
                                chat_ratio = detector.chat.chat_rate_ratio
                            adv_score = score_moment_advanced(
                                {"reason": m.reason, "score": m.score},
                                duration=m.end_sec - m.start_sec,
                                seconds_since_last_clip=secs_since,
                                chat_rate_ratio=chat_ratio,
                                channel=channel,
                            )
                            if adv_score < 0.15:
                                logger.info(f"⏭️ [{label}] Score avançado muito baixo ({adv_score:.3f}), pulando")
                                self._funnel(label, "score_gate", platform=platform, channel=clean)
                                continue

                            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                            name     = f"clip_{platform}_{clean}_{ts}_{m.moment_id[:8]}_{m.reason}"
                            raw_clip = clips_dir / f"raw_{name}.ts"

                            min_dur = config["detection"]["audio_spike"]["min_clip_duration"]
                            start, end = self._segment_bounds(
                                m.start_sec, m.end_sec, min_dur,
                                available=capture.get_elapsed_seconds(),
                            )

                            # ── ClipBot v3: Reaction + Hook + QualityGate ───
                            ca_cfg = config["editing"].get("clip_analysis", {})
                            if ca_cfg.get("enabled", False):
                                from clip_analyzer import compute_reaction_score, find_hook_start, quality_gate

                                # (1) ReactionDetector
                                reaction = compute_reaction_score(
                                    raw_file, m.start_sec, m.end_sec,
                                    audio_norm=ca_cfg.get("audio_norm", 0.06),
                                    motion_norm=ca_cfg.get("motion_norm", 12.0),
                                )
                                rw = ca_cfg.get("reaction_weight", 0.5)
                                effective_score = adv_score * (1 - rw + rw * reaction)
                                logger.info(
                                    f"\U0001f3ad [{label}] Reaction: {reaction:.2f}  "
                                    f"base={adv_score:.2f}  effective={effective_score:.2f}"
                                )
                                if effective_score < 0.10:
                                    logger.info(f"\u23ed\ufe0f [{label}] Score efetivo baixo ({effective_score:.2f}), pulando")
                                    self._funnel(label, "reaction_gate",
                                                 platform=platform, channel=clean)
                                    continue

                                # (2) HookStartFinder
                                hook_start = find_hook_start(
                                    raw_file, start,
                                    pre_roll=ca_cfg.get("hook_pre_roll", 3.0),
                                    audio_norm=ca_cfg.get("audio_norm", 0.06),
                                    motion_norm=ca_cfg.get("motion_norm", 12.0),
                                )
                                if hook_start < start:
                                    logger.info(f"\U0001f3af [{label}] Hook start: {start:.1f}s \u2192 {hook_start:.1f}s")
                                    start = hook_start

                            capture.extract_segment(raw_file, start, end, raw_clip)
                            self._funnel(label, "extracted", platform=platform, channel=clean)

                            # (3a) OfflineScreenGate \u2014 pega countdown/placeholder
                            # ANTES do QualityGate (mais barato + mais decisivo).
                            try:
                                from clip_analyzer import is_offline_screen
                                is_off, off_info = is_offline_screen(raw_clip, duration_sec=8.0)
                                if is_off:
                                    logger.warning(
                                        f"\U0001f6ab [{label}] OfflineScreen: {off_info} \u2014 SKIP"
                                    )
                                    _metrics_incr("engine.skipped_offline",
                                                  platform=platform, channel=clean)
                                    self._funnel(label, "offline_gate",
                                                 platform=platform, channel=clean)
                                    raw_clip.unlink(missing_ok=True)
                                    continue
                            except Exception as e:
                                logger.debug(f"offline_screen check falhou: {e}")

                            # (3b) QualityGate
                            if ca_cfg.get("enabled", False):
                                qg = ca_cfg.get("quality_gate", {})
                                if qg.get("enabled", True):
                                    passed, qg_reason = quality_gate(
                                        raw_clip,
                                        min_rms=qg.get("min_rms", 0.01),
                                        min_motion=qg.get("min_motion", 5.0),
                                    )
                                    if not passed:
                                        logger.warning(f"\U0001f6ab [{label}] QualityGate: {qg_reason} \u2014 SKIP")
                                        _metrics_incr("engine.quality_gate_failed", platform=platform)
                                        self._funnel(label, "quality_gate",
                                                     platform=platform, channel=clean)
                                        raw_clip.unlink(missing_ok=True)
                                        continue

                            # Título context-aware via TitleGenerator
                            moment_entry = {
                                "reason":   m.reason,
                                "keywords": m.keywords,
                                "score":    m.score,
                                "extra":    m.extra,
                            }
                            if title_gen:
                                auto_title = title_gen.generate(raw_clip, moment_entry)
                                # Iter2 PR4: emojis inline nas legendas (LLM Real Oficial style).
                                # Se cascata caiu p/ template, last_subtitle_emojis = {} (no-op).
                                sub_emojis = getattr(title_gen, "last_subtitle_emojis", {}) or {}
                            else:
                                auto_title = "CLIP ÉPICO DA LIVE!"
                                sub_emojis = {}

                            edited = editor.process(
                                raw_clip, name,
                                title_text=auto_title,
                                subtitle_emojis=sub_emojis,
                            )
                            raw_clip.unlink(missing_ok=True)

                            # process() retorna None se comercial detectado
                            if edited is None:
                                logger.warning(f"⏭️ [{label}] Clip descartado (comercial ou filtro)")
                                self._funnel(label, "editor_discard",
                                             platform=platform, channel=clean)
                                continue

                            self._funnel(label, "rendered", platform=platform, channel=clean)
                            last_clip_time = time.time()

                            # ── Publicação: o canal só ENFILEIRA (Patch 6) ────────────
                            # Antes: descrições (Whisper + LLM), upload em cada
                            # plataforma e delay_between_posts DENTRO do loop de
                            # captura — ~15 min parado por clip com 3 plataformas,
                            # e os momentos detectados nesse meio-tempo expiravam.
                            # A fila publica numa thread própria, com intervalo,
                            # retry, teto diário e limpeza só com tudo confirmado.
                            targets = [p for p in ("tiktok", "instagram", "youtube")
                                       if (config["upload"].get(p) or {}).get("enabled")]
                            queued = None
                            if targets and self.publish_queue is not None:
                                queued = self.publish_queue.enqueue(
                                    edited, targets, channel=clean, reason=m.reason,
                                    label=label, title=auto_title,
                                )
                            elif targets:
                                logger.warning(
                                    f"⚠️  [{label}] sem fila de publicação — clip salvo sem postar"
                                )

                            # ── Contadores com lock ───────────────────────────────
                            with self._stats_lock:
                                self.clips_today += 1
                                self.clips_total += 1

                            if queued:
                                logger.info(f"📬 [{label}] Clip na fila de publicação: {', '.join(targets)}")
                            else:
                                logger.info(f"💾 [{label}] Clip salvo: {edited}")

                            self.on_clip_posted({
                                "name":      name,
                                "reason":    m.reason,
                                "platforms": [],
                                "queued":    targets if queued else [],
                                "time":      datetime.now().strftime("%H:%M"),
                            })

                        except Exception as e:
                            logger.error(f"❌ [{label}] Clip {m.reason}: {e}")
                            self._funnel(label, "errors", platform=platform, channel=clean)

            except KeyboardInterrupt:
                logger.info(f"⌨️  [{label}] Ctrl+C na thread do canal.")
            except Exception:
                # Qualquer outro erro reinicia o canal pelo retry externo, com
                # backoff. Antes ele atravessava o finally e encerrava a thread
                # — o log dizia "Encerrado." como num stop normal e o painel
                # de saúde continuava em "capturing".
                logger.exception(
                    f"❌ [{label}] Erro inesperado no monitor (attempt #{attempt}); "
                    "reiniciando o canal"
                )
                _metrics_incr("capture.loop_crash", platform=platform)
                self._funnel(label, "errors", platform=platform, channel=clean)
                self._touch_channel(label, state="reconnecting", attempt=attempt)
                _restart_now = True
            finally:
                # Cleanup imediato do attempt — não depende de atexit
                if _restart_now:
                    logger.info(f"⏹ [{label}] Cleanup pré-restart (attempt #{attempt})")
                else:
                    logger.info(f"⏹ [{label}] Encerrando detector e captura...")
                try:
                    if detector is not None:
                        detector.stop()
                except Exception as e:
                    logger.debug(f"detector.stop: {e}")
                try:
                    capture.stop_recording()
                except Exception as e:
                    logger.debug(f"capture.stop_recording: {e}")
                if not _restart_now:
                    logger.info(f"✅ [{label}] Encerrado.")

            # ── Pós-attempt: decide se faz restart ou encerra de vez ─────────
            # Sai do outer loop se engine pediu stop (KeyboardInterrupt ou
            # _stop_event.set()). Senão, faz backoff + retry.
            if self._stop_event.is_set():
                return

            if not _restart_now:
                # Inner loop saiu sem flag de restart = stop_event externo.
                # Acabou natural; não retentar.
                return

            # Reset backoff se attempt durou bastante (canal estava OK até cair)
            attempt_duration = time.time() - attempt_start_ts
            if attempt_duration > _RETRY_RESET_AFTER:
                logger.info(
                    f"🔁 [{label}] Attempt rodou {attempt_duration:.0f}s OK; "
                    f"resetando backoff para {_RETRY_BACKOFF_INIT:.0f}s"
                )
                backoff_s = _RETRY_BACKOFF_INIT
                attempt = 0   # resetar contador também

            logger.warning(
                f"⏸ [{label}] Aguardando {backoff_s:.0f}s antes de reconectar..."
            )
            if self._stop_event.wait(timeout=backoff_s):
                return   # stop_event chegou durante o wait
            backoff_s = min(_RETRY_BACKOFF_MAX, backoff_s * _RETRY_BACKOFF_MULT)
            # Loop continua → novo attempt

        # outer while exit (stop_event)
        logger.info(f"👋 [{label}] Monitor encerrado (engine stop).")

    # ──────────────────────────────────────────────────────────────────────────
    # Geração de clip a partir de momento pendente (chamado pela API)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _segment_bounds(start_sec: float, end_sec: float, min_dur: float,
                        available: float = None) -> tuple:
        """
        Janela a extrair para o .ts, com padding até min_clip_duration.

        O padding é centrado no momento, com duas correções que vêm do clip de
        12/08 (15s declarados no sidecar, ~36s de vídeo entregue):

          • Cabeça que não cabe é compensada na cauda. `max(0.0, start - pad)`
            descartava duração em silêncio, e o min_clip_duration configurado
            não era respeitado nos primeiros minutos de stream.

          • Cauda não pede além do que já foi capturado (`available`). Antes, a
            janela ultrapassava a borda da live e o FFmpeg entregava só o que
            existia — por isso o arquivo tinha 36s. A duração perdida é
            recuperada puxando o início para trás, onde há material.

        INVARIANTE: o retorno sempre contém [start_sec, end_sec]. Quando nem
        assim a duração mínima cabe, o segmento devolvido é menor — e é essa
        janela, a real, que vai para o sidecar.
        """
        dur = end_sec - start_sec
        seg_start, seg_end = start_sec, end_sec

        if dur < min_dur:
            pad = (min_dur - dur) / 2.0
            seg_start = start_sec - pad
            seg_end   = end_sec + pad

        if available is not None and seg_end > available:
            # Nunca recorta o momento: se nem ele foi capturado por inteiro, o
            # portão de candidatos (elapsed >= end_sec + 2) já teria barrado.
            seg_end   = max(end_sec, available)
            deficit   = min_dur - (seg_end - seg_start)
            if deficit > 0:
                seg_start -= deficit

        if seg_start < 0.0:
            # Compensa na cauda o que não couber antes de zero, sem voltar a
            # passar do material disponível.
            seg_end = max(seg_end, min(seg_end - seg_start,
                                       available if available is not None
                                       else seg_end - seg_start))
            seg_start = 0.0

        return seg_start, seg_end

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

        # Marca como "processing" para impedir reject concorrente de deletar o
        # raw_file — e recusa uma segunda geração do mesmo momento em curso.
        if not self.pending_store.try_set_processing(moment_id):
            return {
                "ok": False,
                "reason": "already_processing",
                "error": "Este momento já está sendo gerado",
            }
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
            # moment_id no nome: o timestamp é em segundos, e dois momentos do
            # mesmo canal e reason gerados no mesmo segundo escreviam no MESMO
            # .mp4/.json. O reason continua por último (get_clips o lê ali).
            name        = (
                f"clip_{entry['platform']}_{entry['channel']}"
                f"_{ts}_{moment_id[:8]}_{entry['reason']}"
            )

            output_dir = Path(config["system"]["output_folder"])
            output_dir.mkdir(parents=True, exist_ok=True)

            # Label do funil igual ao usado pelas threads de canal, para que
            # approval-mode e auto-mode somem no MESMO canal.
            _flabel = f"{entry['platform'].upper()}:{entry['channel']}"

            logger.info(f"🎬 Gerando clip do momento {moment_id[:8]} [{entry['reason']}]…")

            # OfflineScreenGate: pega countdown / "stream starting soon" / placeholder
            # ANTES de gastar Whisper title + LLM + editor.process (~20s economizados).
            try:
                from clip_analyzer import is_offline_screen
                is_off, off_info = is_offline_screen(raw_clip, duration_sec=8.0)
                if is_off:
                    logger.warning(
                        f"\U0001f6ab Clip {moment_id[:8]} é OFFLINE SCREEN: {off_info} — SKIP"
                    )
                    _metrics_incr("engine.skipped_offline",
                                  channel=entry.get("channel", "?"),
                                  reason=entry.get("reason", "?"))
                    self._funnel(_flabel, "offline_gate",
                                 platform=entry.get("platform", ""),
                                 channel=entry.get("channel", ""))
                    self.pending_store.finish_and_cleanup(moment_id)
                    return {
                        "ok": False,
                        "reason": "offline_screen",
                        "error": f"Offline screen detectada: {off_info}",
                    }
            except Exception as e:
                logger.debug(f"offline_screen check falhou: {e}")

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

            # editor.process() retorna None se clip foi descartado (comercial,
            # content_filter blacklist, ou outro motivo). Não cria sidecar —
            # só limpa o pending e sai.
            #
            # CONTRATO (P0-3): esta função retorna SEMPRE dict com chave "ok".
            # Retornar None quebrava clipbot_api._do_generate_clip ({**result}
            # → TypeError → thread morta → UI presa em "gerando...").
            if edited is None:
                logger.warning(
                    f"⏭️ Clip {moment_id[:8]} descartado pelo editor (comercial/filtro/SKIP)"
                )
                self._funnel(_flabel, "editor_discard",
                             platform=entry.get("platform", ""),
                             channel=entry.get("channel", ""))
                try:
                    self.pending_store.finish_and_cleanup(moment_id)
                except Exception as e:
                    logger.debug(f"cleanup falhou: {e}")
                return {
                    "ok": False,
                    "reason": "editor_discard",
                    "error": "Clip descartado pelo editor (comercial, blacklist ou filtro)",
                }

            # ── Sidecar JSON ──────────────────────────────────────────────────
            sidecar = edited.with_suffix(".json")
            reasons  = entry.get("extra", {}).get("reasons", [entry["reason"]])
            seg_start = entry.get("segment_start_sec", entry["start_sec"])
            seg_end   = entry.get("segment_end_sec", entry["end_sec"])
            sidecar.write_text(json.dumps({
                "moment_id":    moment_id,
                "reason":       entry["reason"],
                "reasons":      reasons,
                "platform":     entry["platform"],
                "channel":      entry["channel"],
                # Momento: ONDE está o evento. Segmento: o que o arquivo cobre.
                # Eram a mesma coisa aqui, e com padding para min_clip_duration
                # o número declarado não batia com o vídeo entregue (15s
                # declarados, ~36s no arquivo, em 12/08).
                "start_sec":    entry["start_sec"],
                "end_sec":      entry["end_sec"],
                "segment_start_sec":    seg_start,
                "segment_end_sec":      seg_end,
                "segment_duration_sec": round(seg_end - seg_start, 1),
                "base_score":   round(entry["score"], 4),
                "final_score":  round(final_score, 4),
                "keywords":     entry.get("keywords", []),
                "generated_at": datetime.now().isoformat(),
            }, indent=2, ensure_ascii=False), encoding="utf-8")

            self._funnel(_flabel, "rendered",
                         platform=entry.get("platform", ""),
                         channel=entry.get("channel", ""))

            # Terminamos de usar o raw_clip: libera processing + remove entry +
            # apaga o .ts (se nenhum outro pendente o referenciar).
            self.pending_store.finish_and_cleanup(moment_id)

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
