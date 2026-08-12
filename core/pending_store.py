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
        Remoção EXTERNA (reject da UI, descarte, limpeza).

        A entry sai SEMPRE da store. O arquivo .ts só é apagado se ninguém
        mais precisar dele:
          • nenhum outro momento pendente aponta para o mesmo raw_file, E
          • nenhum momento em processamento (inclusive este) o está lendo.

        Se o arquivo estiver em uso, ele é preservado e recolhido depois pelo
        GC de órfãos do disk monitor — nunca fica órfão permanente.

        Antes (bug P0-1): o guard `if moment_id in self._processing: return`
        abortava a função inteira, então generate_clip_from_moment() — que
        marca processing ANTES e limpa DEPOIS — nunca conseguia remover nada.
        """
        self._remove(moment_id, released=False)

    def finish_and_cleanup(self, moment_id: str) -> None:
        """
        Remoção pelo DONO do processamento: "terminei de usar este momento".

        Libera a marca de processing e remove entry + arquivo (se não for
        compartilhado com outro pendente). É o que generate_clip_from_moment()
        deve chamar em todos os seus desfechos (sucesso, descarte, offline).
        """
        with self._lock:
            self._processing.discard(moment_id)
        self._remove(moment_id, released=True)

    # ── Implementação compartilhada ──────────────────────────────────────────

    def _remove(self, moment_id: str, *, released: bool) -> None:
        """
        released=True  → chamador detinha o processing e já terminou.
        released=False → remoção externa; respeita quem estiver processando.
        Nunca lança. Idempotente.
        """
        with self._lock:
            entry = self._data.pop(moment_id, None)
            if not entry:
                return
            raw_file = Path(entry.get("raw_file", ""))

            # Outro momento pendente referencia o mesmo .ts?
            still_used = any(
                Path(e.get("raw_file", "")) == raw_file
                for e in self._data.values()
            )
            # Alguém está processando este .ts agora?
            #   • o próprio momento (só conta se a remoção NÃO veio do dono)
            #   • qualquer outro momento em processamento com o mesmo arquivo
            in_use_by_processing = (
                (not released and moment_id in self._processing)
                or any(
                    Path(self._data[mid].get("raw_file", "")) == raw_file
                    for mid in self._processing
                    if mid in self._data
                )
            )
            self._save_locked()

        # Apaga fora do lock para não bloquear outras threads
        if still_used or in_use_by_processing:
            logger.debug(
                f"raw clip preservado ({raw_file.name}): "
                f"still_used={still_used} in_use={in_use_by_processing}"
            )
            return
        if raw_file.exists():
            try:
                raw_file.unlink()
                logger.info(f"🗑️  Raw clip removido: {raw_file.name}")
            except Exception as ex:
                logger.warning(f"Falha ao remover raw clip: {ex}")

    # ── Persistência ──────────────────────────────────────────────────────────

    def _save_locked(self) -> None:
        """
        Salva no disco atomicamente. DEVE ser chamado DENTRO do lock.

        Atomicidade: escreve em arquivo .tmp e faz os.replace (rename atômico no
        Windows desde Python 3.3+). Crash no meio do write não corrompe o
        pending.json original — perde-se apenas a última mutação.
        """
        try:
            APP_DATA.mkdir(parents=True, exist_ok=True)
            tmp_path = PENDING_FILE.with_suffix(".json.tmp")
            payload = json.dumps(
                list(self._data.values()), indent=2, ensure_ascii=False
            )
            # Escreve em .tmp + flush/fsync antes de renomear
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    # Alguns FS (network drives, FAT32) não suportam fsync
                    pass
            # Rename atômico (substitui pending.json antigo)
            os.replace(tmp_path, PENDING_FILE)
        except Exception as ex:
            logger.warning(f"pending.json save falhou: {ex}")
            # Cleanup do .tmp em caso de falha
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

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
