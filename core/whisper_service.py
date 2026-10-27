"""
core/whisper_service.py — Singleton lazy-loaded para Whisper.

Backend preference (em ordem):
  1. faster-whisper CUDA INT8_FP16 (1660 Super Turing tem Tensor Cores)
     ~600MB VRAM, 5-10× mais rápido que openai-whisper CPU.
  2. faster-whisper CPU INT8
     ~200MB RAM, 2× mais rápido que openai-whisper CPU.
  3. openai-whisper CPU (fallback final, ~460MB RAM, mais lento).

Por que existe:
  • video_editor._transcribe e title_generator._whisper_micro carregavam
    whisper.load_model() por chamada → 460MB I/O por clip + JIT warm-up.
  • Singleton mantém modelo em RAM por toda sessão. Lazy-load.

API pública:
  WhisperService.get(device=...).transcribe(audio_path, ...) → {"segments": [...]}
  Sem `size`, usa DEFAULT_SIZE — o mesmo modelo para legenda, título e hook
  (um singleton só na VRAM).

Saída de transcribe() segue formato openai-whisper (dict com segments) para
compatibilidade total com callers existentes.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger("ClipBot.Whisper")


class WhisperService:
    """
    Singleton por (size, device_preference). Modelo carregado uma vez, reutilizado.

    device_preference:
      "auto"  — tenta CUDA INT8_FP16 → CPU INT8 → openai-whisper CPU (default)
      "cuda"  — força CUDA (erro se indisponível)
      "cpu"   — força CPU
    """

    # medium: PT-BR bem mais fiel que small em clips reais ("zero testosterona"
    # vs "zero texto", "For You" vs "foru"); 1,8× mais lento (~8 s num clip
    # de 82 s na 1660 S) e +1 GB de VRAM. Ver tests/test_whisper_model_size.py.
    DEFAULT_SIZE = "medium"

    _instances: Dict[tuple, "WhisperService"] = {}
    _class_lock = threading.Lock()

    @classmethod
    def get(cls, size: Optional[str] = None, device: str = "auto") -> "WhisperService":
        size = size or cls.DEFAULT_SIZE
        # Aceita "cpu" antigo dos callers existentes; mapeamos para "auto" por padrão
        # mas mantemos diferenciação se caller pediu explicitamente "cpu".
        key = (size, device)
        with cls._class_lock:
            inst = cls._instances.get(key)
            if inst is None:
                inst = cls(size, device)
                cls._instances[key] = inst
            return inst

    def __init__(self, size: str, device: str):
        self.size       = size
        self.device     = device
        self._model     = None
        self._load_lock = threading.Lock()
        # CRITICAL: faster-whisper model.transcribe NÃO é thread-safe — duas
        # chamadas paralelas no MESMO modelo CUDA corrompem state (segments
        # iterator vazio, segfault em ctranslate2). Lock serializa transcribe.
        # GPU single-stream é gargalo anyway; perda < 5% throughput.
        self._transcribe_lock = threading.Lock()
        self._backend:  Optional[str] = None

    # ── Carga lazy com cascata de backends ────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            logger.info(f"📥 Whisper loading: size={self.size} device_pref={self.device}...")

            # Estratégia de carga conforme preferência
            if self.device == "cpu":
                attempts = [("faster-cpu",), ("openai-cpu",)]
            elif self.device == "cuda":
                attempts = [("faster-cuda",), ("faster-cpu",), ("openai-cpu",)]
            else:  # "auto"
                attempts = [("faster-cuda",), ("faster-cpu",), ("openai-cpu",)]

            last_err: Optional[Exception] = None
            for (backend,) in attempts:
                try:
                    if backend == "faster-cuda":
                        self._load_faster_cuda()
                    elif backend == "faster-cpu":
                        self._load_faster_cpu()
                    elif backend == "openai-cpu":
                        self._load_openai_cpu()
                    logger.info(f"✅ Whisper loaded: backend={self._backend} size={self.size}")
                    return
                except Exception as e:
                    last_err = e
                    logger.debug(f"   {backend} falhou: {type(e).__name__}: {e}")

            # Tudo falhou
            logger.error(f"❌ Nenhum backend Whisper disponível. Último erro: {last_err}")
            raise RuntimeError(f"Whisper não disponível: {last_err}")

    @staticmethod
    def _ensure_nvidia_dll_path() -> None:
        """
        ctranslate2 nativo usa LoadLibrary legacy no Windows — esse mecanismo
        ignora os.add_dll_directory(). Solução: prepend ao PATH para que
        LoadLibrary resolva via PATH.

        Também faz add_dll_directory (cinto+suspensórios) para callers Python.
        Idempotente — checa se já está no PATH antes de prepend.
        """
        import os, sysconfig
        site_pkgs = sysconfig.get_paths()["purelib"]
        candidates = [
            os.path.join(site_pkgs, "nvidia", "cuda_runtime", "bin"),
            os.path.join(site_pkgs, "nvidia", "cublas", "bin"),
            os.path.join(site_pkgs, "nvidia", "cudnn", "bin"),
            os.path.join(site_pkgs, "nvidia", "cuda_nvrtc", "bin"),
        ]
        existing_path = os.environ.get("PATH", "")
        path_dirs = existing_path.split(os.pathsep)
        path_lower = {p.lower() for p in path_dirs}
        for path in candidates:
            if not os.path.isdir(path):
                continue
            # Prepend ao PATH se ainda não está lá
            if path.lower() not in path_lower:
                os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                path_lower.add(path.lower())
            # add_dll_directory também (Python-level lookup; falha silenciosa em <3.8)
            try:
                os.add_dll_directory(path)
            except (OSError, AttributeError):
                pass

    def _load_faster_cuda(self) -> None:
        # Adiciona paths de DLL CUDA ANTES de carregar ctranslate2
        self._ensure_nvidia_dll_path()
        from faster_whisper import WhisperModel
        import ctranslate2
        if ctranslate2.get_cuda_device_count() == 0:
            raise RuntimeError("Sem CUDA device disponível")
        self._model = WhisperModel(
            self.size,
            device="cuda",
            compute_type="int8_float16",   # Turing-friendly
        )
        self._backend = "faster-whisper-cuda-int8_fp16"

    def _load_faster_cpu(self) -> None:
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            self.size,
            device="cpu",
            compute_type="int8",
        )
        self._backend = "faster-whisper-cpu-int8"

    def _load_openai_cpu(self) -> None:
        import whisper
        self._model = whisper.load_model(self.size, device="cpu")
        self._backend = "openai-whisper-cpu"

    # ── API pública (formato openai-whisper) ─────────────────────────────────

    def transcribe(
        self,
        audio_path,
        *,
        language: str = "pt",
        word_timestamps: bool = False,
        beam_size: int = 1,
    ) -> dict:
        """
        Transcreve audio_path. Retorna formato openai-whisper:
            {"segments": [...], "text": "..."}

        language="auto" (ou None/"") → faster-whisper auto-detecta idioma
        do clip. Útil pra canais multilíngues (ex: streamer EN num clip,
        outro PT em outro clip). Custa ~1s extra na primeira inferência
        pra detection; depois é cached.

        Defaults otimizados para clips curtos:
          • beam_size=1 (~3.5× mais rápido que default 5, queda <1% WER)
          • condition_on_previous_text=False (evita loop autorregressivo)
          • temperature=0.0 (determinístico)
        """
        self._ensure_loaded()

        # Auto-detect: faster-whisper aceita language=None
        lang = None if language in ("auto", None, "") else language

        # Serializa transcribe (faster-whisper não é thread-safe)
        with self._transcribe_lock:
            if self._backend.startswith("faster-whisper"):
                return self._transcribe_faster(
                    audio_path, language=lang,
                    word_timestamps=word_timestamps, beam_size=beam_size,
                )
            else:
                return self._transcribe_openai(
                    audio_path, language=lang,
                    word_timestamps=word_timestamps, beam_size=beam_size,
                )

    def _transcribe_faster(self, audio_path, *, language, word_timestamps, beam_size) -> dict:
        """
        faster-whisper retorna iterator; convertemos para dict format.

        Anti-hallucination (clipbot v2.5):
          • compression_ratio_threshold=1.8: dropa segments com texto altamente
            repetitivo (caso clássico "do sinal do sinal..." tem compression ratio
            ~3.5). Default 2.4 deixa passar; 1.8 catch agressivo.
          • no_speech_threshold=0.7: dropa segments onde prob "sem fala" > 0.7
            (default 0.6 — 0.7 é mais agressivo, melhor pra background music).
          • log_prob_threshold=-0.8: dropa segments com confiança baixa
            (default -1.0 = 36% prob; -0.8 = 45% prob, mais exigente).
          • vad_parameters: min_silence 500ms (filtra silêncios pequenos antes
            do Whisper sequer ver).
          • Post-process: detecta loop "AAA AAA AAA" e dropa segment.
        """
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            best_of=1,
            condition_on_previous_text=False,
            temperature=0.0,
            word_timestamps=word_timestamps,
            vad_filter=True,
            compression_ratio_threshold=1.8,
            no_speech_threshold=0.7,
            log_prob_threshold=-0.8,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        # Log idioma detectado (útil em modo language=None/auto)
        try:
            logger.debug(
                f"   lang={info.language} prob={info.language_probability:.2f} "
                f"duration={info.duration:.1f}s"
            )
        except Exception:
            pass

        segments_list = []
        dropped_loop  = 0
        for s in segments_iter:
            text = (s.text or "").strip()
            if not text:
                continue
            # Post-process: detecta loop autorregressivo ("do sinal do sinal...")
            if self._is_hallucination_loop(text):
                dropped_loop += 1
                logger.debug(f"   ⚠️  hallucination loop dropped: {text[:60]!r}...")
                continue
            seg = {
                "id":    getattr(s, "id", len(segments_list)),
                "start": float(s.start),
                "end":   float(s.end),
                "text":  s.text,
            }
            if word_timestamps and s.words:
                seg["words"] = [
                    {"word": w.word, "start": float(w.start), "end": float(w.end)}
                    for w in s.words
                ]
            segments_list.append(seg)
        if dropped_loop:
            logger.info(f"🚫 Whisper hallucination filter: dropped {dropped_loop} loop segment(s)")
        full_text = " ".join(s["text"].strip() for s in segments_list)
        return {"segments": segments_list, "text": full_text, "language": info.language}

    @staticmethod
    def _is_hallucination_loop(text: str) -> bool:
        """
        Heurística de detecção de loop autorregressivo.
        Whisper, com áudio ambíguo (música, silêncio, ruído), entra em loops
        do tipo "do sinal do sinal do sinal..." ou "obrigado obrigado obrigado".
        Critérios cumulativos (qualquer um TRUE → é loop):

        1. Top-1 palavra repete ≥40% do total das palavras
           (5+ palavras + ratio alto)
        2. Top bigram (par de palavras) repete ≥30% do total dos bigramas
           ("do sinal" × 10 num texto de 20 palavras = 50% → loop)
        """
        words = text.lower().split()
        n = len(words)
        if n < 5:
            return False
        # Regra 1: monopalavra
        from collections import Counter
        cnt = Counter(words)
        top_word, top_count = cnt.most_common(1)[0]
        if top_count / n >= 0.40:
            return True
        # Regra 2: bigrama dominante
        if n >= 6:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(n - 1)]
            bc = Counter(bigrams)
            top_bg, top_bg_count = bc.most_common(1)[0]
            if top_bg_count / len(bigrams) >= 0.30:
                return True
        return False

    def _transcribe_openai(self, audio_path, *, language, word_timestamps, beam_size) -> dict:
        return self._model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=word_timestamps,
            beam_size=beam_size,
            best_of=1,
            condition_on_previous_text=False,
            temperature=0.0,
        )

    # ── Introspeção ──────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    # ── Liberação manual (raro; só para shutdown) ────────────────────────────

    def unload(self) -> None:
        """Libera modelo. Próxima transcribe recarrega."""
        with self._load_lock:
            self._model = None
            self._backend = None
            logger.info(f"📤 Whisper unloaded: size={self.size}")
