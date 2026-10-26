"""
publish_queue.py — fila de publicação com thread própria (Patch 6).

Antes, o modo automático gerava as descrições, postava em cada plataforma e
esperava `delay_between_posts` (300 s padrão) DENTRO do loop de captura do
canal: ~15 min parado por clip com 3 plataformas, e os momentos detectados
nesse meio-tempo expiravam (TTL de 300 s). O upload manual da UI postava numa
thread solta, sem retry nem fila.

Agora o canal (ou a UI) só enfileira; uma thread publica um post por vez:

  • intervalo entre posts = o da UI (delay_fn, lido a cada post);
  • falha ou exceção → nova tentativa em 5 min e 30 min; desiste na 3ª;
  • publish_fn devolvendo None = teto diário recusou → adia para o dia
    seguinte (não descarta);
  • estado em APPDATA/ClipBot/publish_queue.json (safe_io): reiniciar o app
    não perde posts nem repete plataformas já postadas;
  • o clip local só é apagado (cleanup_fn) quando TODAS as plataformas do
    job confirmaram — antes, qualquer sucesso apagava o arquivo.

Quem publica de fato é publish_fn(job, platform) -> True | False | None,
injetado. Trocar o destino (uploaders atuais, API oficial da Meta,
upload-post) muda só ele.
"""

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional

import safe_io

logger = logging.getLogger("ClipBot.PublishQueue")


def _default_path() -> Path:
    return Path(os.environ.get("APPDATA", Path.home())) / "ClipBot" / "publish_queue.json"


class PublishQueue:
    MAX_ATTEMPTS      = 3
    RETRY_BACKOFF_SEC = (300.0, 1800.0)     # 5 min, 30 min
    IDLE_POLL_SEC     = 5.0

    def __init__(self, publish_fn: Callable[[dict, str], Optional[bool]], *,
                 delay_fn: Callable[[], float] = lambda: 0.0,
                 cleanup_fn: Callable[[], bool] = lambda: False,
                 on_result: Optional[Callable[[dict, str, Optional[bool], Optional[str]], None]] = None,
                 clock: Callable[[], float] = time.time,
                 path: Optional[Path] = None) -> None:
        self._publish_fn = publish_fn
        self._delay_fn   = delay_fn
        self._cleanup_fn = cleanup_fn
        self._on_result  = on_result
        self._clock      = clock
        self._path       = Path(path) if path else _default_path()

        self._lock = threading.Lock()
        self._jobs: List[dict] = []
        self._next_post_at = 0.0
        self._load_blocked: Optional[str] = None

        self._stop   = threading.Event()
        self._wake   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._load()

    # ── API ───────────────────────────────────────────────────────────────────

    def enqueue(self, clip, platforms, *, channel: str = "", reason: str = "",
                label: str = "", title: Optional[str] = None) -> Optional[str]:
        """Põe o clip na fila e volta na hora (não publica)."""
        platforms = [p for p in (platforms or []) if p]
        if not platforms:
            return None
        job = {
            "id":         uuid.uuid4().hex[:12],
            "clip":       str(clip),
            "platforms":  list(platforms),
            "results":    {},     # plataforma → True (postado) / False (desistiu)
            "attempts":   {},
            "not_before": {},     # plataforma → epoch da próxima tentativa
            "last_error": {},
            "channel":    channel,
            "reason":     reason,
            "label":      label,
            "title":      title,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with self._lock:
            self._jobs.append(job)
            self._save_locked()
        logger.info(f"📬 Na fila de publicação: {Path(clip).name} → {', '.join(platforms)}")
        self._wake.set()
        return job["id"]

    def pending(self) -> List[dict]:
        with self._lock:
            return json.loads(json.dumps(self._jobs))

    def run_once(self) -> bool:
        """
        Publica UM (job, plataforma) que esteja na vez. True se fez algo.
        A thread do worker chama em loop; os testes chamam direto.
        """
        now = self._clock()
        with self._lock:
            due = self._due_locked(now)
            if due is None:
                return False
            job_id, plat = due
            job = self._find_locked(job_id)
            snapshot = json.loads(json.dumps(job))

        clip = Path(snapshot["clip"])
        if not clip.exists():
            self._settle(job_id, plat, False, "arquivo do clip não existe mais")
            return True

        try:
            res, err = self._publish_fn(snapshot, plat), None
        except Exception as e:
            res, err = False, f"{type(e).__name__}: {e}"

        if res is None:
            self._defer_to_tomorrow(job_id, plat)
        elif res:
            self._next_post_at = self._clock() + max(0.0, float(self._delay_fn() or 0.0))
            self._settle(job_id, plat, True, None)
        else:
            self._retry_or_give_up(job_id, plat, err or "upload devolveu falha")
        return True

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="publish-queue")
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Worker ────────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                did = self.run_once()
            except Exception:
                logger.exception("❌ Fila de publicação: erro inesperado no worker")
                did = False
            if not did:
                self._wake.wait(timeout=self.IDLE_POLL_SEC)
                self._wake.clear()

    def _due_locked(self, now: float):
        if now < self._next_post_at:
            return None
        for job in self._jobs:
            for plat in job["platforms"]:
                if plat in job["results"]:
                    continue
                if job["not_before"].get(plat, 0.0) <= now:
                    return job["id"], plat
        return None

    def _find_locked(self, job_id: str) -> Optional[dict]:
        return next((j for j in self._jobs if j["id"] == job_id), None)

    def _settle(self, job_id: str, plat: str, ok: bool, why: Optional[str]) -> None:
        """Resultado final de uma plataforma; fecha o job quando todas fecharem."""
        delete_clip = None
        with self._lock:
            job = self._find_locked(job_id)
            if job is None:
                return
            job["results"][plat] = ok
            if why:
                job["last_error"][plat] = why
            snapshot = json.loads(json.dumps(job))
            if all(p in job["results"] for p in job["platforms"]):
                self._jobs.remove(job)
                # Só apaga se nenhum outro job ainda aponta para o mesmo clip
                # (o mesmo arquivo enviado em dois envios separados).
                shared = any(j["clip"] == job["clip"] for j in self._jobs)
                if all(job["results"].values()) and not shared and self._cleanup_fn():
                    delete_clip = Path(job["clip"])
            self._save_locked()

        if ok:
            logger.info(f"✅ Postado em {plat}: {Path(snapshot['clip']).name}")
        else:
            logger.error(f"❌ {plat}: post desistido ({why}) — {Path(snapshot['clip']).name}")
        if delete_clip is not None:
            try:
                delete_clip.unlink(missing_ok=True)
            except OSError as e:
                logger.warning(f"⚠️  Clip postado não pôde ser apagado: {e}")
        self._notify(snapshot, plat, ok, why)

    def _retry_or_give_up(self, job_id: str, plat: str, why: str) -> None:
        with self._lock:
            job = self._find_locked(job_id)
            if job is None:
                return
            n = job["attempts"].get(plat, 0) + 1
            job["attempts"][plat] = n
            job["last_error"][plat] = why
            if n < self.MAX_ATTEMPTS:
                wait = self.RETRY_BACKOFF_SEC[min(n - 1, len(self.RETRY_BACKOFF_SEC) - 1)]
                job["not_before"][plat] = self._clock() + wait
                self._save_locked()
                logger.warning(
                    f"⚠️  {plat}: falha no post ({why}); nova tentativa em "
                    f"{wait / 60:.0f} min ({n}/{self.MAX_ATTEMPTS})"
                )
                return
        self._settle(job_id, plat, False, why)

    def _defer_to_tomorrow(self, job_id: str, plat: str) -> None:
        now = datetime.fromtimestamp(self._clock())
        tomorrow = (now.replace(hour=0, minute=0, second=0, microsecond=0)
                    + timedelta(days=1, minutes=1)).timestamp()
        with self._lock:
            job = self._find_locked(job_id)
            if job is None:
                return
            job["not_before"][plat] = tomorrow
            snapshot = json.loads(json.dumps(job))
            self._save_locked()
        logger.info(f"⏸ {plat}: limite diário atingido — post adiado para amanhã")
        self._notify(snapshot, plat, None, "limite diário: adiado para amanhã")

    def _notify(self, job: dict, plat: str, ok: Optional[bool], why: Optional[str]) -> None:
        if self._on_result is None:
            return
        try:
            self._on_result(job, plat, ok, why)
        except Exception as e:
            logger.debug(f"on_result falhou: {e}")

    # ── Persistência ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        status, raw, err = safe_io.read_text_classified(self._path)
        if status == safe_io.MISSING:
            return
        if status != safe_io.OK:
            # Ilegível: não sobrescreve (perderia os posts na fila); a cada
            # save tenta ler de novo e mescla. Mesmo padrão do PendingStore.
            self._load_blocked = err
            logger.error(f"❌ Fila de publicação ilegível ({err}); gravação suspensa")
            return
        jobs = self._parse(raw)
        if jobs is not None:
            self._jobs = jobs

    def _parse(self, raw: str) -> Optional[List[dict]]:
        try:
            data = json.loads(raw)
            jobs = data.get("jobs", []) if isinstance(data, dict) else []
            return [j for j in jobs if isinstance(j, dict) and j.get("id") and j.get("clip")]
        except Exception as e:
            safe_io.quarantine(self._path)
            logger.error(f"❌ Fila de publicação corrompida ({e}); cópia em quarentena")
            return None

    def _save_locked(self) -> None:
        if self._load_blocked is not None:
            status, raw, err = safe_io.read_text_classified(self._path)
            if status == safe_io.UNREADABLE:
                logger.error(f"❌ Fila de publicação não gravada: arquivo ainda ilegível ({err})")
                return
            disk = self._parse(raw) if status == safe_io.OK else []
            known = {j["id"] for j in self._jobs}
            self._jobs = [j for j in (disk or []) if j["id"] not in known] + self._jobs
            self._load_blocked = None
        try:
            safe_io.atomic_write_text(self._path, json.dumps({"jobs": self._jobs}, indent=2,
                                                             ensure_ascii=False))
        except OSError as e:
            logger.error(f"❌ Fila de publicação não gravada ({e}); posts só em memória")
