"""
feedback.py — ClipBot v2.5
Sistema de pesos aprendidos por feedback + ranking multi-dimensional.

Arquivos gerados em APPDATA/ClipBot/:
  feedback_weights.json  — pesos atuais por reason (persistente)
  feedback.jsonl         — log imutável de cada feedback recebido
  channel_history.json   — histórico de sucesso por canal

API:
  load_weights()           → dict[reason, float]
  save_weights(weights)
  score_moment(entry, w)   → float  (base_score * weight, clampado 0..1)
  score_moment_advanced()  → float  (multi-dimensional: dur + recency + chat)
  apply_feedback(...)      → dict atualizado
  update_channel_history() — registra clip gerado
  get_channel_stats()      → dict com stats do canal

Labels reconhecidos:
  "top"     → target 1.0  (peso sobe)
  "approve" → target 0.8  (peso sobe levemente)
  "reject"  → target 0.0  (peso desce)

Regra de atualização:
  w_new = clamp(w + LR * (target - base_score), W_MIN, W_MAX)
  LR = 0.05, range [0.5, 1.5]
"""

import copy
import json
import logging

import safe_io
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger("ClipBot.Feedback")

APP_DATA         = Path(os.environ.get("APPDATA", Path.home())) / "ClipBot"
WEIGHTS_FILE     = APP_DATA / "feedback_weights.json"
FEEDBACK_LOG     = APP_DATA / "feedback.jsonl"
CHANNEL_HISTORY  = APP_DATA / "channel_history.json"

LR    = 0.05
W_MIN = 0.5
W_MAX = 1.5

_LABEL_TARGET: Dict[str, float] = {
    "top":     1.0,
    "approve": 0.8,
    "reject":  0.0,
}

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "audio_spike":  1.0,
    "chat_flood":   1.0,
    "keyword":      1.0,
    "crash":        1.0,
    "spin":         1.0,
    "near_miss":    1.0,
    "fastest_lap":  1.0,
    "manual":       1.0,
    "super_moment": 1.1,
}


# ─── Load / Save ─────────────────────────────────────────────────────────────
#
# Pesos e histórico são read-modify-write. Uma leitura que falha devolve
# defaults/{} para o scoring seguir, mas o write que vem depois APAGARIA os
# dados bons que continuam no disco — o mesmo mecanismo do incidente de 12/08
# com credentials.enc. Por isso:
#   • _read_* devolvem (dados, problema DESTA leitura); problema != None
#     bloqueia a gravação;
#   • corrompido → quarentena (.corrupt.<ts>) e recomeça dos defaults;
#   • read-modify-write sob lock, decidido pela leitura feita ali dentro;
#   • caminho de leitura (scoring, ~1×/s) loga em DEBUG; quem recusa gravar
#     loga ERROR.

_weights_lock = threading.Lock()
_history_lock = threading.Lock()


def _read_weights() -> Tuple[Dict[str, float], Optional[str]]:
    """(pesos mesclados com defaults, problema que impede gravar)."""
    status, raw, err = safe_io.read_text_classified(WEIGHTS_FILE)
    if status == safe_io.MISSING:
        return dict(_DEFAULT_WEIGHTS), None
    if status != safe_io.OK:
        return dict(_DEFAULT_WEIGHTS), (err or status)
    try:
        saved = json.loads(raw)
        if not isinstance(saved, dict):
            raise ValueError(f"esperado objeto JSON, veio {type(saved).__name__}")
        merged = dict(_DEFAULT_WEIGHTS)
        merged.update({k: float(v) for k, v in saved.items()})
        return merged, None
    except Exception as ex:
        moved = safe_io.quarantine(WEIGHTS_FILE)
        logger.error(
            f"❌ {WEIGHTS_FILE.name} corrompido ({ex}); "
            + (f"cópia em {moved.name}, recomeçando dos pesos padrão."
               if moved else "quarentena falhou — gravação bloqueada.")
        )
        return dict(_DEFAULT_WEIGHTS), (None if moved else f"corrompido: {ex}")


def load_weights() -> Dict[str, float]:
    """Carrega pesos do disco. Mescla com defaults para reasons novos."""
    weights, problem = _read_weights()
    if problem is not None:
        logger.debug(f"{WEIGHTS_FILE.name} ilegível ({problem}); usando pesos padrão")
    return weights


def save_weights(weights: Dict[str, float]) -> None:
    """
    Salva pesos no disco (atômico, com retry).

    Perder os pesos degrada o ranking, não derruba o bot — por isso a falha é
    engolida. Mas vira WARNING com o caminho, para não repetir o 12/08, em que
    um arquivo de estado simplesmente não apareceu e nada no log explicava.
    """
    try:
        safe_io.atomic_write_text(
            WEIGHTS_FILE, json.dumps(weights, indent=2, ensure_ascii=False)
        )
    except OSError as ex:
        logger.warning(f"⚠️  {WEIGHTS_FILE.name} não foi gravado: {ex}")


# ─── Scoring ──────────────────────────────────────────────────────────────────

def score_moment(entry: dict, weights: Optional[Dict[str, float]] = None) -> float:
    """
    Retorna final_score = base_score * weight, clampado em [0, 1].
    Se weights não fornecido, carrega do disco.
    """
    if weights is None:
        weights = load_weights()
    reason     = entry.get("reason", "audio_spike")
    w          = weights.get(reason, 1.0)
    base_score = float(entry.get("score", 1.0))
    return round(min(1.0, max(0.0, base_score * w)), 4)


# ─── Multi-dimensional scoring (v2.5) ────────────────────────────────────────

def _duration_bonus(duration: float) -> float:
    """
    Bonifica clips no sweet spot (8-15s).
    <8s = penalidade leve, 15-20s = leve, >20s = moderada, >30s = forte.
    """
    if 8.0 <= duration <= 15.0:
        return 1.0
    elif duration < 8.0:
        return 0.85
    elif duration <= 20.0:
        return 0.9
    elif duration <= 30.0:
        return 0.8
    else:
        return 0.7


def _recency_penalty(seconds_since_last_clip: Optional[float]) -> float:
    """
    Penaliza clips muito próximos do anterior.
    >60s = sem penalidade, <30s = penalidade forte.
    """
    if seconds_since_last_clip is None:
        return 1.0
    if seconds_since_last_clip >= 60.0:
        return 1.0
    elif seconds_since_last_clip >= 30.0:
        # Interpolação linear 0.75 → 1.0
        t = (seconds_since_last_clip - 30.0) / 30.0
        return 0.75 + 0.25 * t
    else:
        return 0.5


def _chat_boost(chat_rate_ratio: Optional[float]) -> float:
    """
    Boost quando chat sobe muito (ratio = msgs_agora / msgs_baseline).
    >2x = boost de 1.2, >3x = 1.3, else 1.0.
    """
    if chat_rate_ratio is None:
        return 1.0
    if chat_rate_ratio >= 3.0:
        return 1.3
    elif chat_rate_ratio >= 2.0:
        return 1.2
    elif chat_rate_ratio >= 1.5:
        return 1.1
    return 1.0


def score_moment_advanced(
    entry: dict,
    weights: Optional[Dict[str, float]] = None,
    duration: Optional[float] = None,
    seconds_since_last_clip: Optional[float] = None,
    chat_rate_ratio: Optional[float] = None,
    channel: Optional[str] = None,
) -> float:
    """
    Score composto multi-dimensional:
      final = base × weight × dur_bonus × recency × chat × channel_bonus
    Clampado em [0, 1].
    """
    base = score_moment(entry, weights)

    dur_bonus  = _duration_bonus(duration) if duration else 1.0
    recency    = _recency_penalty(seconds_since_last_clip)
    chat       = _chat_boost(chat_rate_ratio)

    # Channel bonus: reasons que historicamente performam bem
    ch_bonus = 1.0
    if channel:
        stats = get_channel_stats(channel)
        reason = entry.get("reason", "")
        best = stats.get("best_reasons", [])
        if reason in best[:2]:
            ch_bonus = 1.1

    final = base * dur_bonus * recency * chat * ch_bonus
    final = round(min(1.0, max(0.0, final)), 4)

    logger.debug(
        f"🧠 Score avançado: base={base:.3f} × dur={dur_bonus:.2f} "
        f"× rec={recency:.2f} × chat={chat:.2f} × ch={ch_bonus:.2f} "
        f"= {final:.4f}"
    )
    return final


# ─── Channel History ──────────────────────────────────────────────────────────
# Cache TTL para evitar disk read em hot path:
# score_moment_advanced é chamado ~1×/s/momento no monitor; sem cache vira N reads/s.

_CHANNEL_HISTORY_TTL_SEC = 5.0
_channel_cache: dict = {}
_channel_cache_ts: float = 0.0
_channel_cache_lock = threading.Lock()


def _read_channel_history() -> Tuple[dict, Optional[str]]:
    """(histórico do disco, problema que impede gravar). Sem cache."""
    status, raw, err = safe_io.read_text_classified(CHANNEL_HISTORY)
    if status == safe_io.MISSING:
        return {}, None
    if status != safe_io.OK:
        return {}, (err or status)
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"esperado objeto JSON, veio {type(data).__name__}")
        return data, None
    except Exception as ex:
        moved = safe_io.quarantine(CHANNEL_HISTORY)
        logger.error(
            f"❌ {CHANNEL_HISTORY.name} corrompido ({ex}); "
            + (f"cópia em {moved.name}, recomeçando o histórico."
               if moved else "quarentena falhou — gravação bloqueada.")
        )
        return {}, (None if moved else f"corrompido: {ex}")


def _load_channel_history() -> dict:
    """
    Carrega histórico do disco com cache TTL=5s (caminho de leitura/scoring).
    Reduz de N reads/s para 1 read/5s em monitor com muitos canais.

    Devolve cópia PROFUNDA: o histórico tem dicts aninhados por canal, e a
    cópia rasa de antes deixava o caller mutar o próprio cache.
    """
    global _channel_cache, _channel_cache_ts
    now = time.monotonic()
    with _channel_cache_lock:
        if _channel_cache_ts and (now - _channel_cache_ts) < _CHANNEL_HISTORY_TTL_SEC:
            return copy.deepcopy(_channel_cache)

    data, problem = _read_channel_history()
    if problem is not None:
        # NÃO cacheia a falha: antes o {} ficava 5s no cache e alimentava saves.
        logger.debug(f"{CHANNEL_HISTORY.name} ilegível ({problem}); histórico vazio nesta leitura")
        return {}

    with _channel_cache_lock:
        _channel_cache = data
        _channel_cache_ts = now
        return copy.deepcopy(data)


def _save_channel_history(history: dict) -> None:
    """Salva histórico no disco e invalida cache (próxima leitura recarrega)."""
    global _channel_cache, _channel_cache_ts
    try:
        safe_io.atomic_write_text(
            CHANNEL_HISTORY, json.dumps(history, indent=2, ensure_ascii=False)
        )
        # Atualiza cache em vez de invalidar — evita roundtrip desnecessário
        with _channel_cache_lock:
            _channel_cache = copy.deepcopy(history)
            _channel_cache_ts = time.monotonic()
    except OSError as ex:
        logger.warning(f"⚠️  {CHANNEL_HISTORY.name} não foi gravado: {ex}")


def update_channel_history(channel: str, reason: str, score: float) -> None:
    """
    Registra um clip gerado no histórico do canal.

    Read-modify-write sob lock, a partir do DISCO (não do cache: um leitor
    concorrente pode ter reposto no cache uma versão anterior ao último save).
    """
    with _history_lock:
        _update_channel_history_locked(channel, reason, score)


def _update_channel_history_locked(channel: str, reason: str, score: float) -> None:
    history, problem = _read_channel_history()
    if problem is not None:
        logger.error(
            f"❌ Histórico de [{channel}] NÃO atualizado: {CHANNEL_HISTORY.name} "
            f"ilegível ({problem}). Gravar agora apagaria o histórico dos outros canais."
        )
        return

    if channel not in history:
        history[channel] = {
            "total_clips": 0,
            "total_score": 0.0,
            "reason_counts": {},
            "last_clip_ts": None,
        }

    ch = history[channel]
    ch["total_clips"] += 1
    ch["total_score"]  = round(ch.get("total_score", 0.0) + score, 4)

    # Conta reasons
    rc = ch.get("reason_counts", {})
    rc[reason] = rc.get(reason, 0) + 1
    ch["reason_counts"] = rc

    ch["last_clip_ts"] = datetime.now().isoformat()

    # Calcula best_reasons (top 3 por contagem)
    sorted_reasons = sorted(rc.items(), key=lambda x: x[1], reverse=True)
    ch["best_reasons"] = [r for r, _ in sorted_reasons[:3]]
    ch["avg_score"] = round(ch["total_score"] / max(ch["total_clips"], 1), 4)

    history[channel] = ch
    _save_channel_history(history)

    logger.debug(
        f"📊 Channel history [{channel}]: "
        f"clips={ch['total_clips']} avg={ch['avg_score']:.3f} "
        f"best={ch['best_reasons']}"
    )


def get_channel_stats(channel: str) -> dict:
    """Retorna stats de um canal (vazio se nunca usado)."""
    history = _load_channel_history()
    return history.get(channel, {
        "total_clips": 0,
        "avg_score": 0.0,
        "best_reasons": [],
        "last_clip_ts": None,
    })


# ─── Apply feedback ───────────────────────────────────────────────────────────

def apply_feedback(
    reason: str,
    label: str,
    base_score: float,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Atualiza o peso do reason com base no label recebido.
    Persiste e também anexa a feedback.jsonl.
    Retorna os pesos atualizados.

    Fórmula: w += LR * (target - base_score), clamp [W_MIN, W_MAX]

    Com weights=None (o caminho de produção), lê, ajusta e grava sob lock,
    decidido pela leitura feita ali dentro. Se o arquivo de pesos estiver
    ilegível, NADA é gravado por cima — gravar defaults+1 ajuste apagaria os
    pesos aprendidos. O clique continua registrado em feedback.jsonl (sem
    pesos), para poder ser reaplicado depois.
    """
    target = _LABEL_TARGET.get(label, 0.5)

    with _weights_lock:
        problem = None
        if weights is None:
            weights, problem = _read_weights()

        w      = float(weights.get(reason, 1.0))
        w_new  = w + LR * (target - base_score)
        w_new  = round(max(W_MIN, min(W_MAX, w_new)), 4)

        if problem is None:
            weights[reason] = w_new
            save_weights(weights)
        else:
            logger.error(
                f"❌ Feedback [{label}] {reason} NÃO aplicado aos pesos: "
                f"{WEIGHTS_FILE.name} ilegível ({problem}). Registrado em "
                f"{FEEDBACK_LOG.name} para reaplicar depois."
            )

        # ── Log imutável (sob o mesmo lock: append concorrente intercala) ──
        try:
            APP_DATA.mkdir(parents=True, exist_ok=True)
            line = json.dumps({
                "ts":         datetime.now().isoformat(),
                "reason":     reason,
                "label":      label,
                "base_score": round(base_score, 4),
                "old_weight": round(w, 4) if problem is None else None,
                "new_weight": w_new if problem is None else None,
            }, ensure_ascii=False)
            with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as ex:
            logger.warning(f"feedback.jsonl append falhou: {ex}")

    if problem is None:
        logger.info(
            f"🧠 Feedback [{label}] reason={reason}  "
            f"w: {w:.3f} → {w_new:.3f}  (target={target}, base={base_score:.2f})"
        )
    return weights

