"""
core/hook_scorer.py — Iter2 PR3: Re-ranking + título via LLM cascata.

Cascata de 3 providers (ordem padrão):
  1. Anthropic Claude Haiku 4.5      (melhor qualidade editorial)
  2. OpenAI gpt-4o-mini              (fallback se Anthropic sem créditos)
  3. Google Gemini 2.0 Flash         (fallback final, free tier 1500 req/dia)

Por que existe:
  Hoje títulos vêm de templates estáticos (`title_generator._TEMPLATES`) ou de
  micro-transcrição Whisper sem entendimento editorial. Resultado: títulos
  genéricos ("OLHA ISSO: KKKK!") que não capturam o que torna o momento clipável.

  Hook scoring usa LLM para:
    • Analisar transcrição + motivo de detecção (chat_flood, keyword, etc.)
    • Sugerir título com tom editorial (não-template)
    • Sugerir hashtags contextualmente relevantes
    • Retornar viral_score 0..1

Cascata com fallback transparente:
  • Cada provider tem is_available() cacheado (SDK + key)
  • Se provider 1 falha (rate limit, sem créditos, network) → tenta próximo
  • Se TODOS falham → retorna None → caller usa template
  • Nunca lança exceção pro caller. Logs em DEBUG.

Custo estimado (30 clips/dia):
  • Anthropic Haiku 4.5: $0.001/clip = ~$0.90/mês
  • OpenAI gpt-4o-mini:  $0.0003/clip = ~$0.27/mês
  • Gemini 2.0 Flash:    grátis até 1500 req/dia
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional, List

logger = logging.getLogger("ClipBot.HookScorer")

# ──────────────────────────────────────────────────────────────────────────────
# Constants — modelos rápidos e baratos por provider
# ──────────────────────────────────────────────────────────────────────────────

_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_OPENAI_MODEL    = "gpt-4o-mini"
# 2.5-flash-lite: free tier ativo em novas keys (2.0-flash retorna quota=0).
# Qualidade suficiente pra título editorial curto.
_GEMINI_MODEL    = "gemini-2.5-flash-lite"

_MAX_TOKENS = 400
_TRANSCRIPT_CHAR_CAP = 600
_TIMEOUT_SEC = 8.0
# A API do Gemini recusa deadline < 10 s (400 INVALID_ARGUMENT, medido em
# produção em 26/10 com os 8 s acima: todas as chamadas falhavam).
_GEMINI_MIN_TIMEOUT_SEC = 10.0

_DEFAULT_ORDER = ["anthropic", "openai", "gemini"]


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


# ──────────────────────────────────────────────────────────────────────────────
# Key lookup — ENV → credentials.enc → settings.json
# ──────────────────────────────────────────────────────────────────────────────

def _load_key(env_name: str, cred_keys: List[str], settings_keys: List[str]) -> Optional[str]:
    """
    Cascata genérica de lookup de API key.
    Diagnóstico granular: loga onde encontra (ou onde falha).
    """
    # 1. ENV
    k = (os.environ.get(env_name) or "").strip()
    if k:
        logger.debug(f"  ✓ {env_name}: encontrada via ENV (len={len(k)})")
        return k

    # 2. credentials.enc
    try:
        from credentials import load_credentials, degraded_reason, CRED_FILE
        creds = load_credentials() or {}
        # NÃO use CRED_FILE.exists() aqui: no Windows ele devolve False quando a
        # abertura falha por permissão, e foi assim que o diagnóstico de 12/08
        # reportou "file_exists=False" para um arquivo que existia e funcionava.
        # load_credentials já classifica o estado real.
        degraded = degraded_reason("credentials")
        logger.debug(
            f"  credentials.enc: path={CRED_FILE} "
            f"state={'DEGRADADO: ' + degraded if degraded else 'ok'} "
            f"loaded_keys={sorted(creds.keys()) if creds else '[]'}"
        )
        for ck in cred_keys:
            v = (creds.get(ck) or "").strip()
            if v:
                logger.debug(f"  ✓ {env_name}: encontrada via credentials.enc[{ck}] (len={len(v)})")
                return v
    except Exception as e:
        logger.warning(f"  ✗ credentials.enc load falhou: {type(e).__name__}: {e}")

    # 3. settings.json
    try:
        from credentials import load_settings
        s = load_settings() or {}
        node = s
        for sk in settings_keys[:-1]:
            node = (node or {}).get(sk, {})
        v = ((node or {}).get(settings_keys[-1]) or "").strip()
        if v:
            logger.debug(f"  ✓ {env_name}: encontrada via settings.json (len={len(v)})")
            return v
    except Exception as e:
        logger.warning(f"  ✗ settings.json load falhou: {type(e).__name__}: {e}")

    logger.debug(f"  ✗ {env_name}: NÃO encontrada em ENV/credentials.enc/settings.json")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Providers — cada um isolado, com is_available() + _call(prompt) -> str
# ──────────────────────────────────────────────────────────────────────────────

class _BaseProvider:
    """Interface mínima de provider LLM."""
    name: str = "base"
    _client = None
    _available_cached: Optional[bool] = None

    @classmethod
    def is_available(cls) -> bool:
        if cls._available_cached is not None:
            return cls._available_cached
        # Diagnóstico explícito: SDK ok? key ok? Importante porque em contextos
        # diferentes (app.py vs cli isolado) o resultado pode divergir e cache
        # silencioso esconde o motivo.
        sdk_ok = False
        sdk_err = ""
        try:
            sdk_ok = cls._check_sdk()
        except Exception as e:
            sdk_err = f"{type(e).__name__}: {e}"
        key_ok = False
        key_err = ""
        try:
            key_ok = cls._check_key()
        except Exception as e:
            key_err = f"{type(e).__name__}: {e}"
        ok = sdk_ok and key_ok
        cls._available_cached = ok
        if ok:
            logger.info(f"✅ HookScorer[{cls.name}]: disponível")
        else:
            logger.warning(
                f"⚠️  HookScorer[{cls.name}]: indisponível "
                f"(sdk_ok={sdk_ok}{' err='+sdk_err if sdk_err else ''}, "
                f"key_ok={key_ok}{' err='+key_err if key_err else ''})"
            )
        return ok

    @classmethod
    def _check_sdk(cls) -> bool:
        """Sobrescreva em subclasses pra retornar True se o SDK importa OK."""
        raise NotImplementedError

    @classmethod
    def _check_key(cls) -> bool:
        """Sobrescreva em subclasses pra retornar True se a key está acessível."""
        raise NotImplementedError

    @classmethod
    def _check_available(cls) -> bool:
        raise NotImplementedError

    @classmethod
    def call(cls, prompt: str) -> Optional[str]:
        """Retorna texto bruto do LLM ou None em falha."""
        raise NotImplementedError

    @classmethod
    def reset_cache(cls):
        """Re-checa disponibilidade na próxima call (após user adicionar key)."""
        cls._available_cached = None
        cls._client = None


class _AnthropicProvider(_BaseProvider):
    name = "anthropic"

    @classmethod
    def _check_sdk(cls) -> bool:
        import anthropic  # noqa: F401
        return True

    @classmethod
    def _check_key(cls) -> bool:
        return bool(_load_key(
            "ANTHROPIC_API_KEY",
            ["anthropic_api_key"],
            ["system", "anthropic_api_key"],
        ))

    @classmethod
    def call(cls, prompt: str) -> Optional[str]:
        try:
            if cls._client is None:
                import anthropic
                key = _load_key(
                    "ANTHROPIC_API_KEY",
                    ["anthropic_api_key"],
                    ["system", "anthropic_api_key"],
                )
                cls._client = anthropic.Anthropic(api_key=key, timeout=_TIMEOUT_SEC)
            msg = cls._client.messages.create(
                model=_ANTHROPIC_MODEL,
                max_tokens=_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return (msg.content[0].text or "").strip()
        except Exception as e:
            logger.debug(f"anthropic: {type(e).__name__}: {e}")
            return None


class _OpenAIProvider(_BaseProvider):
    name = "openai"

    @classmethod
    def _check_sdk(cls) -> bool:
        import openai  # noqa: F401
        return True

    @classmethod
    def _check_key(cls) -> bool:
        return bool(_load_key(
            "OPENAI_API_KEY",
            ["openai_api_key"],
            ["system", "openai_api_key"],
        ))

    @classmethod
    def call(cls, prompt: str) -> Optional[str]:
        try:
            if cls._client is None:
                from openai import OpenAI
                key = _load_key(
                    "OPENAI_API_KEY",
                    ["openai_api_key"],
                    ["system", "openai_api_key"],
                )
                cls._client = OpenAI(api_key=key, timeout=_TIMEOUT_SEC)
            resp = cls._client.chat.completions.create(
                model=_OPENAI_MODEL,
                max_tokens=_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
                # JSON mode garante parse limpo (sem markdown wrap)
                response_format={"type": "json_object"},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.debug(f"openai: {type(e).__name__}: {e}")
            return None


class _GeminiProvider(_BaseProvider):
    name = "gemini"
    # SDK novo google-genai (2024+). Substitui google-generativeai (depreciado).

    @classmethod
    def _check_sdk(cls) -> bool:
        from google import genai  # noqa: F401
        return True

    @classmethod
    def _check_key(cls) -> bool:
        return bool(_load_key(
            "GOOGLE_API_KEY",
            ["google_api_key", "gemini_api_key"],
            ["system", "google_api_key"],
        ))

    @classmethod
    def call(cls, prompt: str) -> Optional[str]:
        try:
            if cls._client is None:
                from google import genai
                key = _load_key(
                    "GOOGLE_API_KEY",
                    ["google_api_key", "gemini_api_key"],
                    ["system", "google_api_key"],
                )
                cls._client = genai.Client(api_key=key, http_options=_gemini_http_options())
            from google.genai import types as genai_types
            resp = cls._client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=_MAX_TOKENS,
                    response_mime_type="application/json",
                ),
            )
            return (resp.text or "").strip()
        except Exception as e:
            logger.debug(f"gemini: {type(e).__name__}: {e}")
            return None


def _gemini_http_options():
    """
    Timeout do cliente Gemini: o dos outros providers, mas nunca abaixo do
    mínimo que a API aceita (o SDK o envia como deadline do servidor).

    No google-genai, sem http_options o SDK passa timeout=None ao httpx: nem
    connect nem read têm limite, e uma conexão que cai sem RST bloqueia o recv
    para sempre (canal parado no modo automático; "Gerar" eterno no modo
    aprovação). Atenção à unidade: aqui o timeout é em MILISSEGUNDOS.
    """
    from google.genai import types as genai_types
    secs = max(_TIMEOUT_SEC, _GEMINI_MIN_TIMEOUT_SEC)
    return genai_types.HttpOptions(timeout=int(secs * 1000))


_PROVIDERS = {
    "anthropic": _AnthropicProvider,
    "openai":    _OpenAIProvider,
    "gemini":    _GeminiProvider,
}


# ──────────────────────────────────────────────────────────────────────────────
# HookScorer — orquestra cascata
# ──────────────────────────────────────────────────────────────────────────────

class HookScorer:
    """
    Scorer LLM com cascata de providers. Stateless externamente.

    Ordem configurável via settings.json:
      editing.hook_scorer.providers: ["anthropic", "openai", "gemini"]

    Cada provider tem cache de disponibilidade (1× check por processo).
    """

    def __init__(self, language: str = "pt"):
        self.language = language

    # ── API pública ──────────────────────────────────────────────────────────

    @classmethod
    def is_available(cls) -> bool:
        """True se PELO MENOS UM provider está disponível."""
        return any(p.is_available() for p in _PROVIDERS.values())

    @classmethod
    def available_providers(cls) -> List[str]:
        """Lista nomes dos providers prontos pra usar (útil pra UI/diagnóstico)."""
        return [name for name, p in _PROVIDERS.items() if p.is_available()]

    @classmethod
    def reset_cache(cls):
        """Re-checa disponibilidade de todos os providers (chame após salvar key na UI)."""
        for p in _PROVIDERS.values():
            p.reset_cache()

    def score(
        self,
        transcription: str,
        reason: str,
        channel: str = "",
        extra_reasons: Optional[list] = None,
    ) -> Optional[dict]:
        """
        Tenta cada provider em ordem; retorna 1º resultado válido.

        Returns:
            {
              "viral_score": float [0..1],
              "title":       str   (max ~50 chars, uppercase),
              "hashtags":    list[str],
              "hook":        str,
              "provider":    str   ("anthropic"|"openai"|"gemini")
            }
            ou None se TODOS providers falharem (caller faz fallback).
        """
        order = self._provider_order()
        if not order:
            return None

        prompt = self._build_prompt(transcription, reason, channel, extra_reasons)

        for name in order:
            provider = _PROVIDERS.get(name)
            if not provider or not provider.is_available():
                continue

            raw = provider.call(prompt)
            if not raw:
                continue

            data = self._parse_json(raw)
            if not data:
                logger.debug(f"{name}: JSON inválido, próximo provider")
                continue

            result = self._sanitize(data)
            result["provider"] = name
            logger.info(
                f"✨ HookScorer[{name}]: score={result['viral_score']:.2f} "
                f'title="{result["title"]}"'
            )
            return result

        # Todos falharam
        logger.debug(f"HookScorer: todos {len(order)} providers falharam → fallback template")
        return None

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _provider_order() -> List[str]:
        """Lê ordem do settings.json; cai no padrão se ausente/inválido."""
        try:
            from credentials import load_settings
            s = load_settings() or {}
            order = (
                s.get("editing", {})
                 .get("hook_scorer", {})
                 .get("providers")
            )
            if isinstance(order, list) and order:
                # Filtra entradas inválidas e dedupe preservando ordem
                seen = set()
                clean = []
                for p in order:
                    p = str(p).lower().strip()
                    if p in _PROVIDERS and p not in seen:
                        seen.add(p)
                        clean.append(p)
                if clean:
                    return clean
        except Exception:
            pass
        return list(_DEFAULT_ORDER)

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        """Parse robusto: aceita markdown wrap, JSON puro, ou texto + JSON inline."""
        # Remove fence ```json ... ``` se presente
        raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        # Tentativa direta
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        # Procura primeiro objeto JSON na string (Gemini às vezes adiciona prefácio)
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return None

    def _build_prompt(self, transcription: str, reason: str,
                      channel: str, extra_reasons: Optional[list]) -> str:
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
  "hook": "<frase curta explicando por que vale clipar>",
  "subtitle_emojis": {{"palavra_chave": "🔥", "outra_palavra": "😱"}}
}}

Diretrizes:
- viral_score: julgue se vale postar. 0.0=lixo, 0.5=mediano, 0.9=potencial viral.
- title: chamativo, sem clickbait barato, em {lang_label}. NÃO invente contexto.
  Se transcrição vazia, infira do motivo de detecção.
- hashtags: 3-5 itens em {lang_label} + 1-2 globais (#fyp #viral). SEM emojis.
- hook: 1 frase, em {lang_label}, explicando o appeal editorial.
- subtitle_emojis: 2-4 pares "palavra (lowercase, sem acento)" → "emoji unicode único".
  As palavras DEVEM aparecer na transcrição (não invente). Escolha palavras de impacto
  (substantivos, verbos fortes). Emoji deve ser temático à palavra. Ex:
  "crash" → "💥", "rir" → "😂", "porta" → "🚪", "carro" → "🚗".
  Se a transcrição é vazia, retorne {{}}."""

    @staticmethod
    def _sanitize(data: dict) -> dict:
        vs = data.get("viral_score", 0.5)
        try:    vs = float(vs)
        except: vs = 0.5
        vs = max(0.0, min(1.0, vs))

        title = (data.get("title") or "").strip().strip('"').strip("'")
        if len(title) > 50:
            title = title[:49].rstrip() + "…"

        tags = data.get("hashtags") or []
        if not isinstance(tags, list):
            tags = []
        seen = set()
        clean_tags = []
        for t in tags:
            t = str(t).lstrip("#").strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            clean_tags.append(t)

        hook = (data.get("hook") or "").strip().strip('"')

        # subtitle_emojis: dict palavra_lowercase → emoji_unicode (1 char esperado)
        raw_emojis = data.get("subtitle_emojis") or {}
        sub_emojis: dict = {}
        if isinstance(raw_emojis, dict):
            for k, v in raw_emojis.items():
                k = str(k).strip().lower()
                v = str(v).strip()
                if not k or not v:
                    continue
                # Sanitiza emoji: pega só os primeiros chars (max 2 code points
                # — alguns emojis modernos são 2 code points: 🤦‍♂️ etc).
                # Aceita ZWJ sequences truncando em 8 chars.
                v = v[:8]
                sub_emojis[k] = v
                if len(sub_emojis) >= 6:   # cap defensivo
                    break

        return {
            "viral_score":     round(vs, 3),
            "title":           title or "MOMENTO ÉPICO!",
            "hashtags":        clean_tags[:6],
            "hook":            hook,
            "subtitle_emojis": sub_emojis,
        }
