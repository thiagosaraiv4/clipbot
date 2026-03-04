"""
ClipBot v3 — Clip Analyzer
Funções leves de análise de vídeo/áudio usando apenas FFmpeg + stdlib.
Sem dependências externas (sem OpenCV, sem numpy).
"""
import subprocess
import struct
import math
import logging
from pathlib import Path

logger = logging.getLogger("clipbot.clip_analyzer")


# ──────────────────────────────────────────────────────────────────────
# Helpers internos
# ──────────────────────────────────────────────────────────────────────

def _extract_audio_rms(video_path, start_sec: float, duration: float) -> float:
    """Extrai áudio mono PCM f32le e calcula RMS."""
    cmd = [
        "ffmpeg", "-ss", str(start_sec), "-i", str(video_path),
        "-t", str(duration),
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "f32le", "-acodec", "pcm_f32le",
        "pipe:1"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        if result.returncode != 0 or len(result.stdout) < 4:
            return 0.0

        raw = result.stdout
        n_samples = len(raw) // 4
        if n_samples == 0:
            return 0.0

        sum_sq = 0.0
        for i in range(n_samples):
            sample = struct.unpack_from('<f', raw, i * 4)[0]
            sum_sq += sample * sample

        return math.sqrt(sum_sq / n_samples)

    except Exception as e:
        logger.debug(f"Erro RMS: {e}")
        return 0.0


def _extract_motion_score(video_path, start_sec: float, duration: float) -> float:
    """
    Extrai frames a 2fps em grayscale 160x90, calcula motion como
    média das diferenças absolutas entre frames consecutivos.
    """
    cmd = [
        "ffmpeg", "-ss", str(start_sec), "-i", str(video_path),
        "-t", str(duration),
        "-vf", "fps=2,scale=160:90,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "pipe:1"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=20)
        if result.returncode != 0 or len(result.stdout) < 160 * 90 * 2:
            return 0.0

        raw = result.stdout
        frame_size = 160 * 90
        n_frames = len(raw) // frame_size

        if n_frames < 2:
            return 0.0

        total_diff = 0.0
        n_diffs = 0

        for i in range(1, n_frames):
            offset_prev = (i - 1) * frame_size
            offset_curr = i * frame_size
            frame_diff = 0
            for p in range(frame_size):
                frame_diff += abs(raw[offset_curr + p] - raw[offset_prev + p])
            total_diff += frame_diff / frame_size
            n_diffs += 1

        return total_diff / n_diffs if n_diffs > 0 else 0.0

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
