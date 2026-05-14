"""
core/hook_scorer.py — Iter2 PR3: Re-ranking + título via Claude Haiku.

Por que existe:
  Hoje títulos vêm de templates estáticos (`title_generator._TEMPLATES`) ou de
  micro-transcrição Whisper sem entendimento editorial. Resultado: títulos
  genéricos ("OLHA ISSO: KKKK!") que não capturam o que torna o momento clipável.

  Hook scoring usa LLM (Claude Haiku 4.5) para:
    • Analisar transcrição + motivo de detecção (chat_flood, keyword, etc.)
    • Sugerir título com tom editorial (não-template)
    • Sugerir hashtags contextualmente relevantes
    • Retornar viral_score 0..1 — opcional gate para descartar candidatos fracos

Custo:
  ~500 input + 150 output tokens por call.
  Claude Haiku 4.5: $0.80/1M input + $4/1M output ≈ $0.001/clip.
  Em 30 clips/dia × 30 dias = ~$0.90/mês. Negligível.

Fallback automático:
  • Sem API key (env ou config) → is_available()=False → caller usa template.
  • Erro de API/network/parse → score() retorna None → caller usa template.
  • Nunca lança exceção pro caller. Telemetria conta falhas.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger("ClipBot.HookScorer")

# Modelo Haiku rápido + barato. Mesma versão usada em description_generator.
_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 400
_TRANSCRIPT_CHAR_CAP = 600   # transcrição de clip 30s tipica < 400 chars; cap defensivo
_TIMEOUT_SEC = 8.0           # request timeout — não trava engine em network glitch


_REASON_LABELS = {
    "audio_spike":  "pico de áudio (grito/reação intensa)",
    "chat_flood":   "chat lotou de mensagens (momento que viralizou no chat)",
    "keyword":      "chat pediu clip explicitamente",
    "super_moment": "múltiplos triggers simultâneos (provável momento épico)",
    "crash":        "batida/crash em sim racing",
    "spin":         "perda de controle em sim racing",
    "near_miss":    "quase-acidente",
    "fastest_lap":  "volta mais rápida",
    "manual":       "selecionado manualmente",
}

_LANG_LABELS = {"pt": "português brasileiro", "en": "inglês", "es": "espanhol"}


def _load_api_key() -> Optional[str]:
    """
    Cascata de lookup para ANTHROPIC_API_KEY:
      1. ENV var (padrão do SDK anthropic)
      2. credentials.enc do ClipBot (anthropic_api_key)
      3. settings.json (system.anthropic_api_key — não cifrado, fallback)

    Retorna None se nenhum tem valor não-vazio.
    """
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if k:
        return k
    # ClipBot credentials (cifrado)
    try:
        from credentials import load_credentials
        creds = load_credentials() or {}
        k = (creds.get("anthropic_api_key") or "").strip()
        if k:
            return k
    except Exception:
        pass
    # settings.json (não cifrado — apenas conveniência dev)
    try:
        from credentials import load_settings
        s = load_settings() or {}
        k = (s.get("system", {}).get("anthropic_api_key") or "").strip()
        if k:
            return k
    except Exception:
        pass
    return None


class HookScorer:
    """
    Scorer LLM single-shot por clip. Stateless; pode ser singleton ou instância
    descartável. Reutiliza o client Anthropic se chamado múltiplas vezes
    (warmup de conexão TCP/HTTP2).
    """

    _shared_client = None
    _availability_cached: Optional[bool] = None

    def __init__(self, language: str = "pt"):
        self.language = language

    @classmethod
    def is_available(cls) -> bool:
        """
        Cached: anthropic SDK importável + API key presente.
        Re-checa apenas se _availability_cached é None (1× por processo).
        """
        if cls._availability_cached is not None:
            return cls._availability_cached
        try:
            import anthropic  # noqa: F401
        except ImportError:
            logger.debug("anthropic SDK indisponível — HookScorer desativado")
            cls._availability_cached = False
            return False
        key = _load_api_key()
        if not key:
            logger.debug("ANTHROPIC_API_KEY ausente — HookScorer desativado")
            cls._availability_cached = False
            return False
        cls._availability_cached = True
        return True

    @classmethod
    def _client(cls):
        """Lazy + reusable client. Anthropic SDK gerencia HTTP keep-alive."""
        if cls._shared_client is not None:
            return cls._shared_client
        import anthropic
        key = _load_api_key()
        cls._shared_client = anthropic.Anthropic(
            api_key=key, timeout=_TIMEOUT_SEC,
        )
        return cls._shared_client

    # ── API pública ──────────────────────────────────────────────────────────

    def score(
        self,
        transcription: str,
        reason: str,
        channel: str = "",
        extra_reasons: Optional[list] = None,
    ) -> Optional[dict]:
        """
        Retorna:
            {
              "viral_score": float [0..1],
              "title":       str   (max ~50 chars, uppercase),
              "hashtags":    list[str],
              "hook":        str   (curta justificativa — útil em logs/UI)
            }
        ou None em qualquer falha (caller faz fallback).
        """
        if not self.is_available():
            return None

        try:
            prompt = self._build_prompt(transcription, reason, channel, extra_reasons)
            client = self._client()
            msg = client.messages.create(
                model=_MODEL,
                max_tokens=_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            # Tira possível markdown wrap
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            data = json.loads(raw)

            # Sanitiza/clamp campos
            result = self._sanitize(data)
            logger.info(
                f"✨ HookScorer: score={result['viral_score']:.2f} "
                f'title="{result["title"]}"'
            )
            return result

        except Exception as e:
            logger.debug(f"HookScorer falhou ({type(e).__name__}: {e}) — fallback")
            return None

    # ── Internals ────────────────────────────────────────────────────────────

    def _build_prompt(self, transcription: str, reason: str,
                      channel: str, extra_reasons: Optional[list]) -> str:
        # Corta transcrição muito longa (clips curtos raramente passam de 400 chars)
        trans = (transcription or "").strip()[:_TRANSCRIPT_CHAR_CAP]
        if not trans:
            trans = "(sem áudio transcrito)"
        reason_label = _REASON_LABELS.get(reason, "momento detectado pelo bot")
        extras = ""
        if extra_reasons and len(extra_reasons) > 1:
            extras = " + ".join(extra_reasons[:3])
            extras = f"\nTriggers combinados: {extras}"
        lang_label = _LANG_LABELS.get(self.language, "português")
        channel_line = f"Canal: {channel}\n" if channel else ""

        return f"""Você é um editor experiente de clips de live para TikTok/Reels/Shorts.

Analise este clip e retorne avaliação editorial em JSON.

{channel_line}Motivo de detecção: {reason_label}{extras}
Transcrição do áudio: "{trans}"

Retorne EXCLUSIVAMENTE um JSON neste formato (sem markdown, sem texto extra):
{{
  "viral_score": <float 0.0-1.0>,
  "title": "<TÍTULO EM CAIXA ALTA, MAX 40 CHARS, com 1 emoji opcional no fim>",
  "hashtags": ["3", "a", "5", "hashtags", "sem#"],
  "hook": "<frase curta explicando por que vale clipar>"
}}

Diretrizes:
- viral_score: julgue se vale postar. 0.0=lixo, 0.5=mediano, 0.9=potencial viral.
- title: chamativo, sem clickbait barato, em {lang_label}. NÃO invente contexto.
  Se transcrição vazia, infira do motivo de detecção.
- hashtags: 3-5 itens em {lang_label} + 1-2 globais (#fyp #viral). SEM emojis.
- hook: 1 frase, em {lang_label}, explicando o appeal editorial."""

    @staticmethod
    def _sanitize(data: dict) -> dict:
        vs = data.get("viral_score", 0.5)
        try:    vs = float(vs)
        except: vs = 0.5
        vs = max(0.0, min(1.0, vs))

        title = (data.get("title") or "").strip().strip('"').strip("'")
        # Cap 50 chars para combinar com banner drawtext do video_editor
        if len(title) > 50:
            title = title[:49].rstrip() + "…"

        tags = data.get("hashtags") or []
        if not isinstance(tags, list):
            tags = []
        # Remove '#' se LLM teimosamente adicionar, e dedupe preservando ordem
        seen = set()
        clean_tags = []
        for t in tags:
            t = str(t).lstrip("#").strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            clean_tags.append(t)

        hook = (data.get("hook") or "").strip().strip('"')

        return {
            "viral_score": round(vs, 3),
            "title":       title or "MOMENTO ÉPICO!",
            "hashtags":    clean_tags[:6],
            "hook":        hook,
        }
