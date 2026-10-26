"""
description_generator.py
Gera título, descrição e hashtags automaticamente usando a transcrição
do Whisper + Claude API (anthropic).
Fallback inteligente se a API não estiver disponível.
"""

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ClipBot.Description")

# Teto da chamada de descrição. O padrão do SDK é read de 600 s com 2
# retentativas — até ~30 min travando quem chamou (thread do canal no modo
# automático). Com max_retries=1, o pior caso fica em 2 × 15 s.
_DESC_TIMEOUT_SEC = 15.0
_DESC_MAX_RETRIES = 1


# ------------------------------------------------------------------
# Gerador principal
# ------------------------------------------------------------------

class DescriptionGenerator:

    def __init__(self, config: dict):
        self.config   = config
        self.language = config.get("editing", {}).get("subtitles", {}).get("language", "pt")

    # ------------------------------------------------------------------
    # Extrai transcrição do vídeo
    # ------------------------------------------------------------------

    def _transcribe_video(self, video_path: Path) -> str:
        """Usa Whisper para extrair o texto falado no clip."""
        try:
            import whisper
            model  = whisper.load_model("tiny")   # tiny = rápido para descrição
            result = model.transcribe(str(video_path), language=self.language)
            text   = result.get("text", "").strip()
            logger.info(f"🎤 Transcrição para descrição: {text[:80]}...")
            return text
        except Exception as e:
            logger.debug(f"Transcrição falhou: {e}")
            return ""

    # ------------------------------------------------------------------
    # Geração via Claude API
    # ------------------------------------------------------------------

    def _generate_with_ai(
        self,
        transcription: str,
        platform: str,
        channel: str,
        reason: str
    ) -> dict:
        """Chama a API do Claude para gerar título + descrição + hashtags."""
        try:
            import anthropic

            reason_map = {
                "audio_spike": "momento com reação/grito intenso",
                "chat_flood":  "momento que lotou o chat de mensagens",
                "keyword":     "momento que o chat pediu clip",
                "manual":      "momento selecionado manualmente",
            }
            reason_label = reason_map.get(reason, "momento épico")

            lang_map = {
                "pt": "português brasileiro",
                "en": "inglês",
                "es": "espanhol",
            }
            lang_label = lang_map.get(self.language, "português")

            platform_map = {
                "tiktok":    {"chars": 150, "style": "curto, divertido, com emojis e trend"},
                "instagram": {"chars": 200, "style": "engajador, com emojis e call-to-action"},
                "youtube":   {"chars": 300, "style": "descritivo com palavras-chave para SEO"},
            }
            plat_cfg = platform_map.get(platform, platform_map["tiktok"])

            prompt = f"""Você é um especialista em criação de conteúdo para redes sociais.

Analise este clip de live e gere conteúdo para {platform.upper()}:

Canal: {channel}
Motivo do clip: {reason_label}
Transcrição do áudio: "{transcription or 'sem transcrição disponível'}"

Gere em {lang_label} um JSON com exatamente este formato (sem markdown, só o JSON):
{{
  "title": "título chamativo de até 80 caracteres",
  "description": "descrição de até {plat_cfg['chars']} caracteres, estilo {plat_cfg['style']}",
  "hashtags": ["hashtag1", "hashtag2", "hashtag3", "hashtag4", "hashtag5"]
}}

Regras:
- Use emojis relevantes
- Seja autêntico ao contexto do clip
- Hashtags sem o # no JSON (será adicionado depois)
- Hashtags em {lang_label} misturado com inglês popular (#fyp, #viral)
- NÃO invente contexto que não está na transcrição"""

            client = anthropic.Anthropic(timeout=_DESC_TIMEOUT_SEC, max_retries=_DESC_MAX_RETRIES)
            msg    = client.messages.create(
                model="claude-haiku-4-5-20251001",   # rápido e barato
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}]
            )

            raw  = msg.content[0].text.strip()
            # Remove possível markdown ```json ... ```
            raw  = re.sub(r"```json|```", "", raw).strip()
            import json
            data = json.loads(raw)

            logger.info(f"✨ Descrição gerada por IA para {platform}: {data['title']}")
            return data

        except ImportError:
            logger.debug("anthropic não instalado — usando fallback")
            return {}
        except Exception as e:
            logger.debug(f"IA falhou ({e}) — usando fallback")
            return {}

    # ------------------------------------------------------------------
    # Fallback inteligente sem IA
    # ------------------------------------------------------------------

    def _fallback(
        self,
        platform: str,
        channel: str,
        reason: str,
        transcription: str
    ) -> dict:
        """Gera descrição sem IA baseada no contexto disponível."""

        reason_emoji = {
            "audio_spike": "🔊",
            "chat_flood":  "💬",
            "keyword":     "🔥",
            "manual":      "🎯",
        }
        emoji = reason_emoji.get(reason, "🎬")

        # Pega primeiras palavras da transcrição como contexto
        context = ""
        if transcription:
            words = transcription.strip().split()[:8]
            context = " ".join(words)
            if len(transcription.split()) > 8:
                context += "..."

        channel_clean = channel.split("/")[-1].strip("@") if "/" in channel else channel

        titles = {
            "audio_spike": f"{emoji} REAÇÃO INSANA na live do {channel_clean}!",
            "chat_flood":  f"{emoji} O chat EXPLODIU na live do {channel_clean}!",
            "keyword":     f"{emoji} O chat pediu CLIP! Live do {channel_clean}",
            "manual":      f"{emoji} Melhor momento da live do {channel_clean}!",
        }
        title = titles.get(reason, f"{emoji} Clip épico - {channel_clean}")

        hashtags_pt  = ["live", "clip", "gaming", "viral", "foryou"]
        hashtags_en  = ["fyp", "foryoupage", "clips", "streamer", "twitch"]
        hashtags_map = {
            "twitch":  ["twitch", "twitchbrasil", "clip"] + hashtags_en,
            "kick":    ["kick", "kickstreaming", "clip"] + hashtags_en,
            "youtube": ["youtube", "youtubelive", "clip"] + hashtags_pt,
        }
        # Detecta plataforma de origem pelo channel
        origin = "twitch"
        if "kick" in channel.lower():
            origin = "kick"
        elif "youtube" in channel.lower() or channel.startswith("UC"):
            origin = "youtube"

        hashtags = hashtags_map.get(origin, hashtags_pt)[:5]

        descs = {
            "tiktok":    f"{title}\n{'  '.join(context.split()[:5]) if context else ''}",
            "instagram": f"{title}\n\n{context}\n\nSiga para mais clips! 👇",
            "youtube":   f"{title}\n\n{context}\n\nClip gerado automaticamente da live de {channel_clean}.\nInscreva-se para mais conteúdo!",
        }

        return {
            "title":       title,
            "description": descs.get(platform, title),
            "hashtags":    hashtags,
        }

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def generate(
        self,
        video_path: Path,
        platform: str,
        channel: str,
        reason: str,
        transcription: str = None
    ) -> dict:
        """
        Gera título, descrição e hashtags para um clip.
        Retorna: {"title": str, "description": str, "hashtags": list}
        """
        # Usa transcrição passada ou extrai do vídeo
        if not transcription:
            transcription = self._transcribe_video(video_path)

        # Tenta IA primeiro
        result = self._generate_with_ai(transcription, platform, channel, reason)

        # Fallback se IA falhar
        if not result or not result.get("title"):
            result = self._fallback(platform, channel, reason, transcription)

        # Formata hashtags com #
        hashtags = result.get("hashtags", [])
        result["hashtags_str"] = " ".join(
            f"#{h.strip('#').strip()}" for h in hashtags
        )

        # Caption final = descrição + hashtags
        result["caption"] = f"{result['description']}\n\n{result['hashtags_str']}"

        logger.info(f"📝 [{platform}] Título: {result['title']}")
        return result
