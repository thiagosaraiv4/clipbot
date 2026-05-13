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
  WhisperService.get(size, device).transcribe(audio_path, ...) → {"segments": [...]}

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

    _instances: Dict[tuple, "WhisperService"] = {}
    _class_lock = threading.Lock()

    @classmethod
    def get(cls, size: str = "small", device: str = "auto") -> "WhisperService":
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

        Implementação interna decide entre faster-whisper e openai-whisper.
        Adapter de formato garante compat com callers existentes.

        Defaults otimizados para clips curtos:
          • beam_size=1 (~3.5× mais rápido que default 5, queda <1% WER)
          • condition_on_previous_text=False
          • temperature=0.0 (determinístico)
          • word_timestamps=False (custo extra ~30%)
        """
        self._ensure_loaded()

        if self._backend.startswith("faster-whisper"):
            return self._transcribe_faster(
                audio_path, language=language,
                word_timestamps=word_timestamps, beam_size=beam_size,
            )
        else:
            return self._transcribe_openai(
                audio_path, language=language,
                word_timestamps=word_timestamps, beam_size=beam_size,
            )

    def _transcribe_faster(self, audio_path, *, language, word_timestamps, beam_size) -> dict:
        """faster-whisper retorna iterator; convertemos para dict format."""
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            best_of=1,
            condition_on_previous_text=False,
            temperature=0.0,
            word_timestamps=word_timestamps,
            vad_filter=True,    # pula silêncio gratuito; reduz alucinações
        )
        segments_list = []
        for s in segments_iter:
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
        full_text = " ".join(s["text"].strip() for s in segments_list)
        return {"segments": segments_list, "text": full_text, "language": info.language}

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
