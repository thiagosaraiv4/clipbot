"""
ClipBot v3 — Clip Analyzer
Funções leves de análise de vídeo/áudio usando FFmpeg + numpy vetorizado.

PR3d: motion + audio RMS migrados de loop Python para numpy.
       ~50× speedup mensurável.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ffmpeg_runner import run_ffmpeg
from metrics import Timer as _Timer

logger = logging.getLogger("clipbot.clip_analyzer")


# ──────────────────────────────────────────────────────────────────────
# Helpers internos
# ──────────────────────────────────────────────────────────────────────

# Medição × contrato de score
# ───────────────────────────
# _measure_* devolvem None quando NÃO conseguiram medir (ffmpeg com rc≠0,
# timeout, sem trilha, poucos dados, exceção) — diferente de "mediu zero".
# Os portões destrutivos (is_offline_screen, quality_gate) usam essas versões:
# falha de medição não pode descartar um clip e apagar o .ts dele.
# _extract_* mantêm o contrato float (0.0 quando não mediu) dos scorers
# (compute_reaction_score, find_hook_start), que fazem conta com o retorno.

def _measure_audio_rms(video_path, start_sec: float, duration: float) -> Optional[float]:
    """RMS do áudio mono, ou None se não foi possível medir."""
    with _Timer("analyzer.audio_rms"):
        return _measure_audio_rms_impl(video_path, start_sec, duration)


def _extract_audio_rms(video_path, start_sec: float, duration: float) -> float:
    """Extrai áudio mono PCM f32le e calcula RMS (0.0 quando não mediu)."""
    v = _measure_audio_rms(video_path, start_sec, duration)
    return 0.0 if v is None else v


def _measure_audio_rms_impl(video_path, start_sec: float, duration: float) -> Optional[float]:
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
            return None

        # Trunca para múltiplo de 4 bytes (sample float32) para evitar erro do frombuffer
        raw_len = (len(result.stdout) // 4) * 4
        if raw_len == 0:
            return None
        audio = np.frombuffer(result.stdout[:raw_len], dtype=np.float32)
        if audio.size == 0:
            return None
        # RMS = sqrt(mean(x^2)) — totalmente vetorizado
        return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))

    except Exception as e:
        logger.debug(f"Erro RMS: {e}")
        return None


def _measure_motion(video_path, start_sec: float, duration: float) -> Optional[float]:
    """
    Extrai frames a 2fps em grayscale 160x90 e calcula motion como
    média das diferenças absolutas entre frames consecutivos.
    Implementação numpy vetorizada (~50× mais rápida que loop Python).
    None se não foi possível medir.
    """
    with _Timer("analyzer.motion"):
        return _measure_motion_impl(video_path, start_sec, duration)


def _extract_motion_score(video_path, start_sec: float, duration: float) -> float:
    """Motion score com contrato float (0.0 quando não mediu)."""
    v = _measure_motion(video_path, start_sec, duration)
    return 0.0 if v is None else v


def _measure_motion_impl(video_path, start_sec: float, duration: float) -> Optional[float]:
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
            return None

        # Trunca para múltiplo de FRAME_SIZE
        raw = result.stdout
        n_frames = len(raw) // FRAME_SIZE
        if n_frames < 2:
            return None

        usable = raw[:n_frames * FRAME_SIZE]
        # uint8 → int16 para diff sem overflow; reshape (n_frames, FRAME_SIZE)
        frames = np.frombuffer(usable, dtype=np.uint8).reshape(n_frames, FRAME_SIZE).astype(np.int16)
        # diff entre frames consecutivos, abs, média sobre tudo
        diffs = np.abs(np.diff(frames, axis=0))
        return float(diffs.mean())

    except Exception as e:
        logger.debug(f"Erro motion: {e}")
        return None


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

# ──────────────────────────────────────────────────────────────────────
# (D) OfflineScreenDetector
# ──────────────────────────────────────────────────────────────────────

def _audio_coeff_variation(video_path, start_sec: float, duration: float) -> float:
    """
    Coeficient of variation (std/mean) do RMS audio por chunks de 1s.

    Música em loop / silêncio constante / "stream starting soon" tem RMS
    uniforme → CV baixo (~0.05-0.15).
    Fala humana com efeitos/reações tem RMS variando → CV alto (~0.4-1.5).

    Métrica robusta a volume absoluto (CV é normalizado pela média).
    Retorna 1.0 se não há áudio suficiente pra avaliar.
    """
    video_path = Path(video_path)
    cmd = [
        "ffmpeg", "-ss", str(start_sec), "-i", str(video_path),
        "-t", str(duration),
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "f32le", "-acodec", "pcm_f32le",
        "pipe:1"
    ]
    try:
        result = run_ffmpeg(cmd, label="audio_cv",
                            timeout=15, capture_stdout=True, quiet=True)
        if not result.ok or len(result.stdout) < 16000 * 4 * 2:
            return 1.0   # < 2s de áudio → não avalia

        raw_len = (len(result.stdout) // 4) * 4
        audio = np.frombuffer(result.stdout[:raw_len], dtype=np.float32)

        # Chunks de 1s (16000 samples a 16kHz)
        chunk_size = 16000
        n_chunks = audio.size // chunk_size
        if n_chunks < 2:
            return 1.0

        chunks = audio[: n_chunks * chunk_size].reshape(n_chunks, chunk_size).astype(np.float64)
        rms_per_chunk = np.sqrt(np.mean(np.square(chunks), axis=1))
        mean = float(np.mean(rms_per_chunk))
        if mean < 1e-6:
            return 0.0   # silêncio absoluto → CV degenera

        return float(np.std(rms_per_chunk) / mean)

    except Exception as e:
        logger.debug(f"Erro audio_cv: {e}")
        return 1.0


def is_offline_screen(video_path, duration_sec: float = 8.0,
                      motion_max: float = 3.0,
                      audio_cv_max: float = 0.18,
                      audio_silence_max: float = 0.005) -> tuple:
    """
    Detecta tela "OFFLINE PLACEHOLDER" (countdown, "Stream Will Be On...").

    Dispara OFFLINE em 2 caminhos independentes:

      Caminho A (alta confiança — silêncio absoluto):
        • Audio RMS médio < audio_silence_max (~ -46dBFS)
        Streams humanas reais NUNCA têm 8s sem nenhum som (mesmo dramatic
        pause carrega room noise ~ -40dB). RMS=0 → áudio "morto" → offline.

      Caminho B (motion estática + áudio uniforme):
        • Motion mean < motion_max E
        • Audio CV (std/mean) < audio_cv_max
        Frames quase idênticos + RMS uniforme = música em loop / countdown.

    Falsos positivos protegidos:
      • Fala calma com cena escura: motion baixo, audio_cv ALTO → não offline
      • Música ambiente em scene change: audio_cv baixo, motion ALTO → não offline

    Returns: (is_offline: bool, info: str).
    Custo: ~1.5s (1 motion + 1 audio extract, ambos quiet).
    """
    video_path = Path(video_path)

    motion_mean = _measure_motion(video_path, 0.0, duration_sec)
    m_txt = "não medido" if motion_mean is None else f"{motion_mean:.2f}"

    # Audio: extrai 1× e calcula mean + cv num passo só (evita 2× ffmpeg)
    rms_mean, audio_cv = _audio_rms_and_cv(video_path, 0.0, duration_sec)

    # Os dois chamadores são destrutivos (apagam o .ts). Falha de medição —
    # timeout sob carga, clip sem trilha de áudio — não é evidência de nada.
    # Antes, ela virava rms=0.0 e o Caminho A a lia como silêncio absoluto.
    if rms_mean is None:
        return (False, f"motion={m_txt} áudio não medido → gate não aplicado")

    # Caminho A: silêncio absoluto MEDIDO (super forte; quase 0 falsos positivos)
    if rms_mean < audio_silence_max:
        info = (
            f"motion={m_txt} rms_mean={rms_mean:.5f} "
            f"(silêncio < {audio_silence_max}) → OFFLINE SCREEN"
        )
        return (True, info)

    if motion_mean is None:
        return (False, f"motion não medido rms={rms_mean:.4f} "
                       f"audio_cv={audio_cv:.3f} → gate não aplicado")

    # Caminho B: motion estática + audio loop (música/efeito constante)
    motion_static = motion_mean < motion_max
    audio_flat    = audio_cv    < audio_cv_max
    is_offline    = motion_static and audio_flat

    info = (
        f"motion={motion_mean:.2f} (max={motion_max}) "
        f"rms={rms_mean:.4f} audio_cv={audio_cv:.3f} (max={audio_cv_max})"
    )
    if is_offline:
        info += " → OFFLINE SCREEN"
    return (is_offline, info)


def _audio_rms_and_cv(video_path, start_sec: float, duration: float) -> tuple:
    """
    Extrai áudio 1× e retorna (rms_mean, rms_cv) — evita 2× ffmpeg.
    rms_cv = coefficient of variation (std/mean) por chunks de 1s.

    (None, None) quando não foi possível medir; (0.0, 0.0) é silêncio
    digital MEDIDO. Os dois não podem ser confundidos (ver is_offline_screen).
    """
    video_path = Path(video_path)
    cmd = [
        "ffmpeg", "-ss", str(start_sec), "-i", str(video_path),
        "-t", str(duration),
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "f32le", "-acodec", "pcm_f32le",
        "pipe:1"
    ]
    try:
        result = run_ffmpeg(cmd, label="audio_rms_cv",
                            timeout=15, capture_stdout=True, quiet=True)
        if not result.ok or len(result.stdout) < 16000 * 4 * 2:
            return (None, None)

        raw_len = (len(result.stdout) // 4) * 4
        audio = np.frombuffer(result.stdout[:raw_len], dtype=np.float32)

        chunk_size = 16000
        n_chunks = audio.size // chunk_size
        if n_chunks < 2:
            return (None, None)

        chunks = audio[: n_chunks * chunk_size].reshape(n_chunks, chunk_size).astype(np.float64)
        rms_per_chunk = np.sqrt(np.mean(np.square(chunks), axis=1))
        mean = float(np.mean(rms_per_chunk))
        if mean < 1e-6:
            return (0.0, 0.0)
        return (mean, float(np.std(rms_per_chunk) / mean))

    except Exception as e:
        logger.debug(f"Erro audio_rms_and_cv: {e}")
        return (None, None)


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

    rms = _measure_audio_rms(raw_clip_path, 0, check_dur)
    motion = _measure_motion(raw_clip_path, 0, check_dur)

    # Descarte exige as DUAS medições (o critério é silêncio E imagem parada).
    # Antes, duas falhas de ffmpeg sob carga davam 0.0 + 0.0 = "silence+static".
    if rms is None or motion is None:
        logger.info(
            f"🔍 QualityGate: medição incompleta (rms={'não medido' if rms is None else f'{rms:.4f}'}, "
            f"motion={'não medido' if motion is None else f'{motion:.2f}'}) — não descarta"
        )
        return (True, "")

    logger.debug(f"🔍 QualityGate: rms={rms:.4f} (min={min_rms}) motion={motion:.2f} (min={min_motion})")

    if rms < min_rms and motion < min_motion:
        return (False, f"silence+static (rms={rms:.4f}, motion={motion:.1f})")

    return (True, "")
