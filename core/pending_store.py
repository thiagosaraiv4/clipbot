"""
pending_store.py — ClipBot v2.4
Thread-safe store for moments awaiting user approval.

Persistência: APPDATA/ClipBot/pending.json
  • Momentos sobrevivem ao fechar e reabrir o app.
  • Ao restaurar, valida que o raw_file ainda existe no disco.

Limpeza de disco:
  • remove_and_cleanup(id) → remove da store e, se o raw_file
    não for referenciado por nenhum outro momento pendente, apaga-o.
"""

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("ClipBot.PendingStore")

APP_DATA    = Path(os.environ.get("APPDATA", Path.home())) / "ClipBot"
PENDING_FILE = APP_DATA / "pending.json"


class PendingStore:
    """
    Dicionário thread-safe de moment_id → entry dict.

    entry keys:
        moment_id, start_sec, end_sec, reason, score,
        keywords, extra, raw_file (path para o .ts pré-extraído),
        platform, channel, added_at
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._data: Dict[str, dict] = {}
        self._processing: set = set()   # moment_ids currently being processed
        self._load()

    # ── Leitura ───────────────────────────────────────────────────────────────

    def get(self, moment_id: str) -> Optional[dict]:
        """Retorna entry ou None (cópia segura)."""
        with self._lock:
            entry = self._data.get(moment_id)
            return dict(entry) if entry else None

    def list_all(self) -> List[dict]:
        """Snapshot de todos os momentos, ordenados por score DESC."""
        with self._lock:
            return sorted(
                (dict(e) for e in self._data.values()),
                key=lambda m: m.get("score", 0),
                reverse=True,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def set_processing(self, moment_id: str) -> None:
        """Marca um momento como 'em processamento'. Impede deleção do raw_file."""
        with self._lock:
            self._processing.add(moment_id)

    def clear_processing(self, moment_id: str) -> None:
        """Remove a marca de processamento."""
        with self._lock:
            self._processing.discard(moment_id)

    # ── Escrita ───────────────────────────────────────────────────────────────

    def add(self, moment, raw_file: Path, platform: str, channel: str) -> bool:
        """
        Adiciona momento se ainda não presente.
        Retorna True se inserido, False se já existia.
        """
        mid = moment.moment_id
        with self._lock:
            if mid in self._data:
                return False
            self._data[mid] = {
                "moment_id": mid,
                "start_sec": moment.start_sec,
                "end_sec":   moment.end_sec,
                "reason":    moment.reason,
                "score":     moment.score,
                "keywords":  list(moment.keywords),
                "extra":     dict(moment.extra),
                "raw_file":  str(raw_file),
                "platform":  platform,
                "channel":   channel,
                "added_at":  datetime.now().isoformat(),
            }
            self._save_locked()

        logger.info(
            f"📋 Pending: {mid[:8]} [{moment.reason}] "
            f"{moment.start_sec:.1f}s→{moment.end_sec:.1f}s  score={moment.score:.2f}"
        )
        return True

    def remove_and_cleanup(self, moment_id: str) -> None:
        """
        Remove o momento. Se o raw_file não for referenciado por
        nenhum outro momento pendente, apaga o arquivo do disco.
        """
        with self._lock:
            # Não remove se está sendo processado por generate_clip
            if moment_id in self._processing:
                logger.debug(f"remove_and_cleanup: {moment_id[:8]} em processamento, ignorando.")
                return
            entry = self._data.pop(moment_id, None)
            if not entry:
                return
            raw_file = Path(entry.get("raw_file", ""))
            # Verifica se algum outro momento usa o mesmo raw_file
            still_used = any(
                Path(e.get("raw_file", "")) == raw_file
                for e in self._data.values()
            )
            # Também verifica se algum outro momento em processamento usa o arquivo
            processing_uses = any(
                Path(self._data[mid].get("raw_file", "")) == raw_file
                for mid in self._processing
                if mid in self._data
            )
            self._save_locked()

        # Apaga fora do lock para não bloquear outros threads
        if not still_used and not processing_uses and raw_file.exists():
            try:
                raw_file.unlink()
                logger.info(f"🗑️  Raw clip removido: {raw_file.name}")
            except Exception as ex:
                logger.warning(f"Falha ao remover raw clip: {ex}")

    # ── Persistência ──────────────────────────────────────────────────────────

    def _save_locked(self) -> None:
        """Salva no disco. DEVE ser chamado DENTRO do lock."""
        try:
            APP_DATA.mkdir(parents=True, exist_ok=True)
            PENDING_FILE.write_text(
                json.dumps(list(self._data.values()), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as ex:
            logger.warning(f"pending.json save falhou: {ex}")

    def _load(self) -> None:
        """Carrega do disco na inicialização. Descarta entradas sem raw_file."""
        if not PENDING_FILE.exists():
            return
        try:
            entries = json.loads(PENDING_FILE.read_text(encoding="utf-8"))
            recovered = 0
            for e in entries:
                mid = e.get("moment_id")
                if not mid:
                    continue
                raw = Path(e.get("raw_file", ""))
                if not raw.exists():
                    logger.debug(f"pending: raw_file ausente, descartando {mid[:8]}")
                    continue
                self._data[mid] = e
                recovered += 1
            if recovered:
                logger.info(f"📋 Restaurados {recovered} momento(s) pendente(s) do disco.")
        except Exception as ex:
            logger.warning(f"pending.json load falhou: {ex}")
