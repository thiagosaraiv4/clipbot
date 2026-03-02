"""
feedback.py — ClipBot v2.4
Sistema de pesos aprendidos por feedback de aprovação de momentos.

Arquivos gerados em APPDATA/ClipBot/:
  feedback_weights.json  — pesos atuais por reason (persistente)
  feedback.jsonl         — log imutável de cada feedback recebido

API:
  load_weights()           → dict[reason, float]
  save_weights(weights)
  score_moment(entry, w)   → float  (base_score * weight, clampado 0..1)
  apply_feedback(reason, label, base_score, weights) → dict atualizado

Labels reconhecidos:
  "top"     → target 1.0  (peso sobe)
  "approve" → target 0.8  (peso sobe levemente)
  "reject"  → target 0.0  (peso desce)

Regra de atualização:
  w_new = clamp(w + LR * (target - base_score), W_MIN, W_MAX)
  LR = 0.05, range [0.5, 1.5]
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("ClipBot.Feedback")

APP_DATA      = Path(os.environ.get("APPDATA", Path.home())) / "ClipBot"
WEIGHTS_FILE  = APP_DATA / "feedback_weights.json"
FEEDBACK_LOG  = APP_DATA / "feedback.jsonl"

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

def load_weights() -> Dict[str, float]:
    """Carrega pesos do disco. Mescla com defaults para reasons novos."""
    if not WEIGHTS_FILE.exists():
        return dict(_DEFAULT_WEIGHTS)
    try:
        saved = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
        merged = dict(_DEFAULT_WEIGHTS)
        merged.update({k: float(v) for k, v in saved.items()})
        return merged
    except Exception as ex:
        logger.warning(f"feedback_weights.json load falhou: {ex}")
        return dict(_DEFAULT_WEIGHTS)


def save_weights(weights: Dict[str, float]) -> None:
    """Salva pesos no disco."""
    try:
        APP_DATA.mkdir(parents=True, exist_ok=True)
        WEIGHTS_FILE.write_text(
            json.dumps(weights, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as ex:
        logger.warning(f"feedback_weights.json save falhou: {ex}")


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
    """
    if weights is None:
        weights = load_weights()

    target = _LABEL_TARGET.get(label, 0.5)
    w      = float(weights.get(reason, 1.0))
    w_new  = w + LR * (target - base_score)
    w_new  = round(max(W_MIN, min(W_MAX, w_new)), 4)

    weights[reason] = w_new
    save_weights(weights)

    # ── Log imutável ──────────────────────────────────────────────────────────
    try:
        APP_DATA.mkdir(parents=True, exist_ok=True)
        line = json.dumps({
            "ts":         datetime.now().isoformat(),
            "reason":     reason,
            "label":      label,
            "base_score": round(base_score, 4),
            "old_weight": round(w, 4),
            "new_weight": w_new,
        }, ensure_ascii=False)
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as ex:
        logger.warning(f"feedback.jsonl append falhou: {ex}")

    logger.info(
        f"🧠 Feedback [{label}] reason={reason}  "
        f"w: {w:.3f} → {w_new:.3f}  (target={target}, base={base_score:.2f})"
    )
    return weights
