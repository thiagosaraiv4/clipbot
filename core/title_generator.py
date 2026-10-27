"""
title_generator.py — ClipBot v2.5
Gerador de títulos context-aware para o banner do clip.

3 estratégias em cascata:
  1. Whisper micro-transcrição (3-6s do início) → frase do áudio
  2. Templates variados por reason + keywords do Moment
  3. Fallback estático (mapa fixo)

API:
  TitleGenerator(config).generate(raw_clip, moment_entry) → str
"""

import logging
import random
import tempfile
from pathlib import Path
from typing import Optional

from ffmpeg_runner import run_ffmpeg

logger = logging.getLogger("ClipBot.Title")


# ──────────────────────────────────────────────────────────────────────────────
# Templates por reason (variação aleatória)
# ──────────────────────────────────────────────────────────────────────────────

_TEMPLATES = {
    "audio_spike": [
        "DO NADA…",
        "QUE ISSO?!",
        "EU NÃO ACREDITO!",
        "OLHA O GRITO!",
        "REAÇÃO INSANA!",
        "SEM PALAVRAS!",
        "CHOCADO!",
    ],
    "chat_flood": [
        "CHAT ENLOUQUECEU!",
        "SPAM TOTAL!",
        "TODO MUNDO GRITOU!",
        "O CHAT EXPLODIU!",
        "FLOOD INSANO!",
    ],
    "keyword": [
        "'{kw}' NA LIVE!",
        "ELE DISSE '{kw}'!",
        "OLHA ISSO: {kw}!",
        "O CHAT PEDIU: {kw}!",
    ],
    "crash": [
        "BATIDA ABSURDA!",
        "DESTRUIÇÃO TOTAL!",
        "CAPOTOU!",
        "BATIDA INSANA!",
    ],
    "spin": [
        "RODOU FEIO!",
        "SAVE IMPOSSÍVEL!",
        "QUE GIRO!",
        "RODOU TUDO!",
    ],
    "near_miss": [
        "POR UM TRIZ!",
        "QUASE!",
        "ESCAPOU POR POUCO!",
        "POR MUITO POUCO!",
    ],
    "fastest_lap": [
        "VOLTA MAIS RÁPIDA!",
        "RECORD!",
        "VOOU NA PISTA!",
    ],
    "super_moment": [
        "MOMENTO ÉPICO!",
        "LOUCURA TOTAL!",
        "INACREDITÁVEL!",
    ],
    "manual": [
        "MELHOR MOMENTO!",
        "CLIP OBRIGATÓRIO!",
        "PRECISAVA SER CLIPPADO!",
    ],
}

_FALLBACK_TITLES = {
    "audio_spike":   "REAÇÃO INSANA NA LIVE!",
    "chat_flood":    "O CHAT EXPLODIU!",
    "keyword":       "O CHAT PEDIU CLIP!",
    "super_moment":  "MOMENTO ÉPICO!",
    "manual":        "MELHOR MOMENTO DA LIVE!",
    "crash":         "BATIDA INSANA!",
    "near_miss":     "POR MUITO POUCO!",
    "fastest_lap":   "VOLTA MAIS RÁPIDA!",
}


class TitleGenerator:
    """Gera títulos de banner context-aware."""

    def __init__(self, config: dict):
        self.config   = config
        self.language = (
            config.get("editing", {})
            .get("subtitles", {})
            .get("language", "pt")
        )
        tb_cfg = config.get("editing", {}).get("title_banner", {})
        self.use_whisper = tb_cfg.get("use_whisper_title", True)
        self.max_title_len = tb_cfg.get("max_chars", 50)

        # Metadata da última geração (set por generate() quando LLM responde).
        # Caller (bot_engine) pode ler estes pra passar p/ video_editor:
        #   • last_subtitle_emojis: dict palavra_lower → emoji unicode
        #   • last_hashtags:        list[str]
        #   • last_viral_score:     float [0..1]
        # Reset a cada generate(). Vazio se cascata caiu pra template.
        self.last_subtitle_emojis: dict = {}
        self.last_hashtags:        list = []
        self.last_viral_score:     float = 0.0
        self.last_hook:            str  = ""
        self.last_provider:        str  = ""

    # ──────────────────────────────────────────────────────────────────────
    # Estratégia A: Whisper micro-transcrição (primeiros 5s)
    # ──────────────────────────────────────────────────────────────────────

    def _extract_short_audio(self, video_path: Path, duration: float = 5.0) -> Optional[Path]:
        """Extrai os primeiros N segundos de áudio como WAV temporário."""
        tmp_path: Optional[Path] = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            tmp_path = Path(tmp.name)
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-t", str(duration),
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                tmp.name,
            ]
            result = run_ffmpeg(cmd, label="title_extract_audio", timeout=15)
            if not result.ok:
                # Cleanup tempfile em falha (era leak no código antigo)
                try: tmp_path.unlink(missing_ok=True)
                except Exception: pass
                return None
            if tmp_path.exists() and tmp_path.stat().st_size > 1000:
                return tmp_path
            return None
        except Exception as e:
            logger.debug(f"Extração de áudio curto falhou: {e}")
            if tmp_path is not None:
                try: tmp_path.unlink(missing_ok=True)
                except Exception: pass
            return None

    def _whisper_micro(self, video_path: Path) -> Optional[str]:
        """
        Transcreve primeiros 5s do clip via Whisper singleton.
        Reutiliza o MESMO modelo carregado pelo video_editor
        (WhisperService.DEFAULT_SIZE) — em vez de carregar um separado, que
        dobraria a memória.
        """
        if not self.use_whisper:
            return None

        audio_file = self._extract_short_audio(video_path, duration=5.0)
        if not audio_file:
            return None

        try:
            from whisper_service import WhisperService
            # Reusa singleton — mesmo modelo do video_editor (CUDA se disponível)
            service = WhisperService.get(device="auto")
            result  = service.transcribe(
                audio_file, language=self.language,
                word_timestamps=False, beam_size=1,
            )
            text = result.get("text", "").strip()

            if not text or len(text) < 3:
                return None

            # Formata: pega as primeiras palavras significativas
            title = self._clean_whisper_text(text)
            if title and len(title) >= 4:
                logger.info(f"🎤 Título via Whisper: \"{title}\"")
                return title

            return None

        except ImportError:
            logger.debug("Whisper não disponível para título")
            return None
        except Exception as e:
            logger.debug(f"Whisper micro falhou: {e}")
            return None
        finally:
            if audio_file and audio_file.exists():
                try: audio_file.unlink(missing_ok=True)
                except Exception: pass

    def _clean_whisper_text(self, raw_text: str) -> Optional[str]:
        """
        Limpa transcrição do Whisper para virar título de banner.
        Ex: "que isso mano olha essa jogada" → "QUE ISSO MANO!"
        """
        text = raw_text.strip()
        if not text:
            return None

        # Remove pontuação do Whisper e normaliza
        import re
        text = re.sub(r'[,;]', '', text)

        # Pega as primeiras palavras (até max_title_len ou primeiro ponto/!)
        sentences = re.split(r'[.!?]', text)
        best = ""
        for s in sentences:
            s = s.strip()
            if len(s) > len(best) and len(s) <= self.max_title_len:
                best = s

        if not best:
            # Se todas as frases são longas, pega as primeiras palavras
            words = text.split()[:6]
            best = " ".join(words)

        # Trunca se necessário
        if len(best) > self.max_title_len:
            words = best.split()
            best = ""
            for w in words:
                nxt = (best + " " + w).strip()
                if len(nxt) > self.max_title_len:
                    break
                best = nxt

        if not best or len(best) < 4:
            return None

        # Uppercase + ! no final
        best = best.upper().rstrip("!?. ")
        best += "!"

        return best

    # ──────────────────────────────────────────────────────────────────────
    # Estratégia B: Templates variados por reason
    # ──────────────────────────────────────────────────────────────────────

    def _template_title(self, reason: str, keywords: list = None,
                        extra: dict = None) -> str:
        """Gera título usando templates variados por reason."""
        templates = _TEMPLATES.get(reason, _TEMPLATES.get("super_moment", ["CLIP ÉPICO!"]))

        # Keyword injection para reason=keyword
        if reason == "keyword" and keywords:
            kw = keywords[0].upper() if keywords else "CLIP"
            # Trunca keyword se muito longa
            if len(kw) > 15:
                kw = kw[:14] + "…"
            template = random.choice(templates)
            return template.replace("{kw}", kw)

        # Super moment: combina reasons se disponível
        if reason == "super_moment" and extra:
            reasons = extra.get("reasons", [])
            if len(reasons) >= 2:
                r1_templates = _TEMPLATES.get(reasons[0], ["LOUCURA"])
                r2_templates = _TEMPLATES.get(reasons[1], ["TOTAL"])
                t1 = random.choice(r1_templates).rstrip("!")
                t2 = random.choice(r2_templates).rstrip("!")
                combo = f"{t1} + {t2}!"
                if len(combo) <= self.max_title_len:
                    return combo

        return random.choice(templates)

    # ──────────────────────────────────────────────────────────────────────
    # Estratégia C: Fallback estático
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_title(reason: str) -> str:
        """Mapa fixo de títulos por reason (última opção)."""
        return _FALLBACK_TITLES.get(reason, "CLIP ÉPICO DA LIVE!")

    # ──────────────────────────────────────────────────────────────────────
    # API principal
    # ──────────────────────────────────────────────────────────────────────

    def generate(self, raw_clip: Path, moment_entry: dict) -> str:
        """
        Gera título para o banner do clip.
        Cascata: LLM Hook Scorer → Whisper → Templates → Fallback.

        Args:
            raw_clip: Path do arquivo de vídeo raw
            moment_entry: dict com reason, keywords, score, extra

        Returns:
            str: título pronto para o banner (UPPERCASE, ≤max_title_len chars)
        """
        reason   = moment_entry.get("reason", "audio_spike")
        keywords = moment_entry.get("keywords", [])
        extra    = moment_entry.get("extra", {})
        channel  = moment_entry.get("channel", "")

        # Reset metadata da geração anterior (LLM-only fields).
        self.last_subtitle_emojis = {}
        self.last_hashtags = []
        self.last_viral_score = 0.0
        self.last_hook = ""
        self.last_provider = ""

        # ── Estratégia 0: LLM Hook Scorer (Iter2 PR3) ─────────────────
        # Acima do Whisper micro: usa transcrição completa + contexto editorial.
        # Sem API key configurada → is_available()=False, cai para próximo.
        try:
            from hook_scorer import HookScorer
            _llm_ok    = HookScorer.is_available()
            _whisp_ok  = self.use_whisper
            _clip_ok   = raw_clip.exists()
            logger.info(
                f"🔎 LLM Strategy 0 gate: llm_available={_llm_ok} "
                f"use_whisper={_whisp_ok} clip_exists={_clip_ok} "
                f"providers={HookScorer.available_providers()}"
            )
            if _llm_ok and _whisp_ok and _clip_ok:
                # Reusa Whisper singleton para obter transcrição (já carregado por
                # video_editor neste ponto do pipeline; warm ~2s).
                from whisper_service import WhisperService
                svc = WhisperService.get(device="auto")
                # Transcreve só os primeiros 8s (suficiente p/ hook editorial)
                tmp_audio = self._extract_short_audio(raw_clip, duration=8.0)
                if tmp_audio:
                    try:
                        result = svc.transcribe(
                            tmp_audio, language=self.language,
                            word_timestamps=False, beam_size=1,
                        )
                        transcription = (result.get("text") or "").strip()
                    finally:
                        try: tmp_audio.unlink(missing_ok=True)
                        except Exception: pass
                    scorer = HookScorer(language=self.language)
                    extra_reasons = (extra or {}).get("reasons", [])
                    score_result = scorer.score(
                        transcription, reason, channel=channel,
                        extra_reasons=extra_reasons,
                    )
                    if score_result and score_result.get("title"):
                        llm_title = score_result["title"]
                        # Captura metadata p/ video_editor consumir
                        self.last_subtitle_emojis = score_result.get("subtitle_emojis") or {}
                        self.last_hashtags        = score_result.get("hashtags") or []
                        self.last_viral_score     = score_result.get("viral_score", 0.0)
                        self.last_hook            = score_result.get("hook") or ""
                        self.last_provider        = score_result.get("provider", "")
                        logger.info(
                            f"✨ Título via LLM[{self.last_provider}]: \"{llm_title}\" "
                            f"(viral={self.last_viral_score:.2f}, "
                            f"emojis={len(self.last_subtitle_emojis)})"
                        )
                        return llm_title
        except Exception as e:
            logger.debug(f"LLM hook scorer falhou: {e}")

        # ── Estratégia A: Whisper micro ────────────────────────────────
        if self.use_whisper and raw_clip.exists():
            try:
                whisper_title = self._whisper_micro(raw_clip)
                if whisper_title:
                    return whisper_title
            except Exception as e:
                logger.debug(f"Whisper title falhou: {e}")

        # ── Estratégia B: Templates ────────────────────────────────────
        try:
            template_title = self._template_title(reason, keywords, extra)
            if template_title:
                logger.info(f"🎯 Título via template: \"{template_title}\"")
                return template_title
        except Exception as e:
            logger.debug(f"Template title falhou: {e}")

        # ── Estratégia C: Fallback ─────────────────────────────────────
        fallback = self._fallback_title(reason)
        logger.info(f"📌 Título via fallback: \"{fallback}\"")
        return fallback
