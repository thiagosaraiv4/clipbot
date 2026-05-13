"""
ClipBot v3 — Clip Analyzer
Funções leves de análise de vídeo/áudio usando FFmpeg + numpy vetorizado.

PR3d: motion + audio RMS migrados de loop Python para numpy.
       ~50× speedup mensurável.
"""
import logging
from pathlib import Path

import numpy as np

from ffmpeg_runner import run_ffmpeg
from metrics import Timer as _Timer

logger = logging.getLogger("clipbot.clip_analyzer")


# ──────────────────────────────────────────────────────────────────────
# Helpers internos
# ──────────────────────────────────────────────────────────────────────

def _extract_audio_rms(video_path, start_sec: float, duration: float) -> float:
    """Extrai áudio mono PCM f32le e calcula RMS via numpy vetorizado."""
    with _Timer("analyzer.audio_rms"):
        return _extract_audio_rms_impl(video_path, start_sec, duration)


def _extract_audio_rms_impl(video_path, start_sec: float, duration: float) -> float:
    cmd = [
        "ffmpeg", "-ss", str(start_sec), "-i", str(video_path),
        "-t", str(duration),
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "f32le", "-acodec", "pcm_f32le",
        "pipe:1"
    ]
    try:
        # audio_rms chamado várias vezes por clip; quiet evita poluir log
        result = run_ffmpeg(cmd, label="audio_rms",
                            timeout=15, capture_stdout=True, quiet=True)
        if not result.ok or len(result.stdout) < 4:
            return 0.0

        # Trunca para múltiplo de 4 bytes (sample float32) para evitar erro do frombuffer
        raw_len = (len(result.stdout) // 4) * 4
        if raw_len == 0:
            return 0.0
        audio = np.frombuffer(result.stdout[:raw_len], dtype=np.float32)
        if audio.size == 0:
            return 0.0
        # RMS = sqrt(mean(x^2)) — totalmente vetorizado
        return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))

    except Exception as e:
        logger.debug(f"Erro RMS: {e}")
        return 0.0


def _extract_motion_score(video_path, start_sec: float, duration: float) -> float:
    """
    Extrai frames a 2fps em grayscale 160x90 e calcula motion como
    média das diferenças absolutas entre frames consecutivos.
    Implementação numpy vetorizada (~50× mais rápida que loop Python).
    """
    with _Timer("analyzer.motion"):
        return _extract_motion_score_impl(video_path, start_sec, duration)


def _extract_motion_score_impl(video_path, start_sec: float, duration: float) -> float:
    FRAME_W = 160
    FRAME_H = 90
    FRAME_SIZE = FRAME_W * FRAME_H

    cmd = [
        "ffmpeg", "-ss", str(start_sec), "-i", str(video_path),
        "-t", str(duration),
        "-vf", f"fps=2,scale={FRAME_W}:{FRAME_H},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "pipe:1"
    ]
    try:
        # motion_score chamado várias vezes por clip; quiet evita poluir log
        result = run_ffmpeg(cmd, label="motion_score",
                            timeout=20, capture_stdout=True, quiet=True)
        if not result.ok or len(result.stdout) < FRAME_SIZE * 2:
            return 0.0

        # Trunca para múltiplo de FRAME_SIZE
        raw = result.stdout
        n_frames = len(raw) // FRAME_SIZE
        if n_frames < 2:
            return 0.0

        usable = raw[:n_frames * FRAME_SIZE]
        # uint8 → int16 para diff sem overflow; reshape (n_frames, FRAME_SIZE)
        frames = np.frombuffer(usable, dtype=np.uint8).reshape(n_frames, FRAME_SIZE).astype(np.int16)
        # diff entre frames consecutivos, abs, média sobre tudo
        diffs = np.abs(np.diff(frames, axis=0))
        return float(diffs.mean())

    except Exception as e:
        logger.debug(f"Erro motion: {e}")
        return 0.0


# ──────────────────────────────────────────────────────────────────────
# (A) ReactionDetector
# ──────────────────────────────────────────────────────────────────────

def compute_reaction_score(video_path, start_sec: float, end_sec: float,
                           audio_norm: float = 0.06,
                           motion_norm: float = 12.0) -> float:
    """
    Analisa trecho do vídeo e retorna score de reação (0..1).
    Combina motion (40%) + áudio RMS (60%).
    """
    video_path = Path(video_path)
    duration = end_sec - start_sec
    if duration <= 0:
        return 0.0

    rms_raw = _extract_audio_rms(video_path, start_sec, duration)
    motion_raw = _extract_motion_score(video_path, start_sec, duration)

    # Normalizar com clamp
    audio = min(1.0, rms_raw / audio_norm) if audio_norm > 0 else 0.0
    motion = min(1.0, motion_raw / motion_norm) if motion_norm > 0 else 0.0

    score = 0.6 * audio + 0.4 * motion

    logger.debug(
        f"🎭 Reaction raw: rms={rms_raw:.4f} motion={motion_raw:.2f} | "
        f"norm: audio={audio:.2f} motion={motion:.2f} | score={score:.2f}"
    )

    return score


# ──────────────────────────────────────────────────────────────────────
# (B) HookStartFinder
# ──────────────────────────────────────────────────────────────────────

def find_hook_start(video_path, original_start: float,
                    pre_roll: float = 3.0,
                    audio_norm: float = 0.06,
                    motion_norm: float = 12.0) -> float:
    """
    Procura o melhor ponto de início do clip analisando a janela
    [original_start - pre_roll, original_start] em steps de 0.5s.
    Retorna o primeiro timestamp onde a 'energia' ultrapassa 60% do pico.
    """
    with _Timer("analyzer.hook_find", pre_roll=pre_roll):
        return _find_hook_start_impl(video_path, original_start, pre_roll, audio_norm, motion_norm)


def _find_hook_start_impl(video_path, original_start: float, pre_roll: float,
                          audio_norm: float, motion_norm: float) -> float:
    video_path = Path(video_path)
    window_start = max(0.0, original_start - pre_roll)
    step = 0.5
    energies = []

    t = window_start
    while t < original_start:
        rms = _extract_audio_rms(video_path, t, step)
        mot = _extract_motion_score(video_path, t, step)

        audio = min(1.0, rms / audio_norm) if audio_norm > 0 else 0.0
        motion = min(1.0, mot / motion_norm) if motion_norm > 0 else 0.0
        energy = 0.7 * audio + 0.3 * motion

        energies.append((t, energy))
        t += step

    if not energies:
        return original_start

    peak = max(e for _, e in energies)
    if peak < 0.05:
        # Sem atividade significativa antes, manter original
        return original_start

    threshold = peak * 0.6

    # Encontrar primeiro ponto onde energia >= threshold
    for t, e in energies:
        if e >= threshold:
            hook_t = max(window_start, t - 0.5)  # 0.5s antes do onset
            logger.info(f"🎯 Hook start: {original_start:.1f}s → {hook_t:.1f}s (energy={e:.2f}, peak={peak:.2f})")
            return hook_t

    return original_start


# ──────────────────────────────────────────────────────────────────────
# (C) QualityGate
# ──────────────────────────────────────────────────────────────────────

def quality_gate(raw_clip_path, min_rms: float = 0.01,
                 min_motion: float = 5.0) -> tuple:
    """
    Verifica se o clip raw tem conteúdo mínimo.
    Analisa primeiros 4s (ou clip todo se menor).
    Retorna (True, "") se passou, (False, "reason") se falhou.
    """
    raw_clip_path = Path(raw_clip_path)

    # Medir nos primeiros 4 segundos
    check_dur = 4.0

    rms = _extract_audio_rms(raw_clip_path, 0, check_dur)
    motion = _extract_motion_score(raw_clip_path, 0, check_dur)

    logger.debug(f"🔍 QualityGate: rms={rms:.4f} (min={min_rms}) motion={motion:.2f} (min={min_motion})")

    if rms < min_rms and motion < min_motion:
        return (False, f"silence+static (rms={rms:.4f}, motion={motion:.1f})")

    return (True, "")
