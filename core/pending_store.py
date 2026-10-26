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

import safe_io

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
        #: Última falha de persistência (None = disco saudável). Exposto no
        #: heartbeat para que "momento sumiu" deixe de ser invisível.
        self.last_save_error: Optional[str] = None
        self._save_fail_count: int = 0
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

    def add(self, moment, raw_file: Path, platform: str, channel: str,
            segment: tuple = None) -> bool:
        """
        Adiciona momento se ainda não presente.
        Retorna True se inserido, False se já existia.

        segment: (start, end) do que foi REALMENTE extraído para o .ts. Difere
            do momento quando houve padding para atingir min_clip_duration —
            e é essa janela que descreve o arquivo no disco. Omitido = sem
            padding (segmento == momento).
        """
        mid = moment.moment_id
        seg_start, seg_end = segment if segment else (moment.start_sec, moment.end_sec)
        with self._lock:
            if mid in self._data:
                return False
            self._data[mid] = {
                "moment_id": mid,
                "start_sec": moment.start_sec,
                "end_sec":   moment.end_sec,
                # Janela do arquivo (≠ momento quando houve padding)
                "segment_start_sec": seg_start,
                "segment_end_sec":   seg_end,
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
        Salva no disco atomicamente (safe_io: tmp + fsync + rename + retry).
        DEVE ser chamado DENTRO do lock.

        A falha NÃO propaga: o momento já está em memória e a UI de aprovação
        tem de continuar funcionando com o disco quebrado. Mas ela deixa de ser
        silenciosa — em 12/08, três pendentes (128 MB de .ts) não apareceram no
        pending.json e os logs não permitiam distinguir "a escrita falhou" de
        "a escrita nunca foi tentada". Agora a falha vira ERROR + last_save_error,
        que o engine expõe no heartbeat do canal.
        """
        try:
            safe_io.atomic_write_text(
                PENDING_FILE,
                json.dumps(list(self._data.values()), indent=2, ensure_ascii=False),
            )
        except OSError as ex:
            self.last_save_error = str(ex)
            self._save_fail_count += 1
            logger.error(
                f"❌ pending.json NÃO foi gravado ({ex}). "
                f"{len(self._data)} momento(s) existem só em memória e serão "
                f"perdidos ao fechar o app (falha nº {self._save_fail_count}). "
                "Verifique permissões/antivírus em "
                f"{PENDING_FILE.parent}"
            )
            return

        if self.last_save_error is not None:
            logger.info("✅ pending.json voltou a ser gravado.")
        self.last_save_error = None

    def _load(self) -> None:
        """Carrega do disco na inicialização. Descarta entradas sem raw_file."""
        status, raw, err = safe_io.read_text_classified(PENDING_FILE)

        if status == safe_io.MISSING:
            return

        if status != safe_io.OK:
            # Existe e não abriu: NÃO é "primeira execução". Pendentes reais
            # podem estar nesse arquivo, e sobrescrevê-lo os apagaria.
            self.last_save_error = err
            logger.error(
                f"❌ {PENDING_FILE.name} existe mas não pôde ser lido ({err}). "
                "Momentos pendentes da sessão anterior não foram restaurados."
            )
            return

        try:
            entries = json.loads(raw)
            if not isinstance(entries, list):
                raise ValueError(f"esperado lista, veio {type(entries).__name__}")
        except Exception as ex:
            self._quarantine_corrupt(ex)
            return

        recovered = 0
        for e in entries:
            if not isinstance(e, dict):
                continue
            mid = e.get("moment_id")
            if not mid:
                continue
            raw_path = Path(e.get("raw_file", ""))
            if not raw_path.exists():
                logger.debug(f"pending: raw_file ausente, descartando {mid[:8]}")
                continue
            # Entradas de versões anteriores não têm a janela do segmento.
            # Sem padding conhecido, o palpite honesto é o próprio momento —
            # o mesmo valor que aquelas versões gravavam no sidecar.
            e.setdefault("segment_start_sec", e.get("start_sec"))
            e.setdefault("segment_end_sec", e.get("end_sec"))
            self._data[mid] = e
            recovered += 1
        if recovered:
            logger.info(f"📋 Restaurados {recovered} momento(s) pendente(s) do disco.")

    def _quarantine_corrupt(self, ex: Exception) -> None:
        """
        Move um pending.json ilegível para .corrupt.<timestamp> em vez de
        deixá-lo ser sobrescrito pelo próximo save. O arquivo é a única pista
        de quais .ts no disco tinham dono — descartá-lo em silêncio transforma
        um bug diagnosticável em GB de órfãos sem explicação.
        """
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = PENDING_FILE.with_name(f"{PENDING_FILE.name}.corrupt.{stamp}")
        try:
            os.replace(PENDING_FILE, backup)
            where = f"cópia preservada em {backup.name}"
        except OSError as mv_ex:
            where = f"cópia NÃO pôde ser preservada ({mv_ex})"
        logger.error(f"❌ {PENDING_FILE.name} corrompido ({ex}). {where}")
