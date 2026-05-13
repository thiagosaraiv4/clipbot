"""
core/face_tracker.py — Speaker tracking via OpenCV Haar Cascade.

Por que existe:
  O pipeline atual faz pad+blur para o formato vertical 9:16 (vídeo inteiro
  centralizado dentro do frame, com fundo borrado preenchendo o resto). Isso
  consome ~50% da tela vertical em "borda blur" e reduz retenção em Reels/TikTok.

  Speaker tracking: detecta face do streamer, faz CROP HORIZONTAL dinâmico
  seguindo o rosto, preenche a tela vertical inteira. É o que OpusClip/Cutlabs
  fazem. Diferencial visual real.

Estratégia:
  • Amostra 2fps de frames via ffmpeg pipe (sem tempfiles)
  • Detecta face por Haar Cascade (sem deps GPU, sem download)
  • Filtra: maior face por frame (= streamer principal)
  • Smooth: EMA α=0.3 + interpolação para misses isolados
  • Se hit_ratio < 30% (face raramente visível): retorna None → caller fallback
  • Gera FFmpeg crop expression PWL com `between+lerp` entre keypoints

Limitações honestas:
  • Haar Cascade é frontal-only. Rosto em ângulo (>45°) não detecta.
  • Sensível a iluminação ruim.
  • Iter2.5 pode upgrade para mediapipe tasks API (blaze_face) ou YOLOv8n-face.

Compatibilidade:
  • Sem deps GPU (CPU puro, opencv-contrib).
  • Sem download de modelo (Haar embarcado em cv2.data).
  • Não bloqueia se opencv não estiver disponível (fallback automático).
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("ClipBot.FaceTracker")


# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports (cv2 só é tocado quando FaceTracker for usado de fato)
# ─────────────────────────────────────────────────────────────────────────────

_cv2 = None
_cascade = None


def _ensure_cv2():
    global _cv2, _cascade
    if _cv2 is not None and _cascade is not None:
        return True
    try:
        import cv2
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        cas = cv2.CascadeClassifier(path)
        if cas.empty():
            logger.warning("Haar cascade vazio; speaker tracking desabilitado")
            return False
        _cv2     = cv2
        _cascade = cas
        return True
    except Exception as e:
        logger.warning(f"cv2 indisponível, speaker tracking desabilitado: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceSample:
    """Uma amostra de detecção de face em um timestamp."""
    t:      float    # segundos do início do clip
    cx_n:   float    # centro X normalizado [0..1]
    cy_n:   float    # centro Y normalizado [0..1]
    area_n: float    # área normalizada [0..1] (proxy de confiança/proximidade)


@dataclass
class TrackResult:
    samples:     List[FaceSample]    # já suavizado + interpolado
    hit_ratio:   float               # frames com face / frames totais
    avg_area_n:  float               # tamanho médio da face (informativo)
    in_w:        int                 # resolução original do input
    in_h:        int


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_FPS         = 2.0
_MIN_FACE_SIZE_REL  = 0.06     # face mínima = 6% da menor dimensão (filtra ruído)
# 50% hit_ratio é conservador: bench em 3 clips de gaming (baiano) deu 8-27%
# → fallback automático correto. Streams IRL/just-chatting tipicamente >70%.
# Threshold mais baixo (ex: 30%) ativaria tracking com trajetória "saltitante"
# entre keypoints distantes — pior visualmente que pad+blur estável.
_HIT_RATIO_MIN      = 0.50
_EMA_ALPHA          = 0.30     # smoothing: 30% peso da nova amostra
_OUTLIER_MAX_JUMP_N = 0.20     # > 20% de movimento entre frames consecutivos = outlier


# ─────────────────────────────────────────────────────────────────────────────
# Frame extraction via ffmpeg pipe (raw gray)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_frames_gray(video_path: Path, fps: float, width: int = 480) -> Tuple[bytes, int, int]:
    """
    Extrai frames em grayscale via pipe ffmpeg.
    Retorna (raw_bytes, frame_w, frame_h). Frames concatenados.

    Width fixo em 480 para acelerar Haar (cascade em 480p ~16ms/frame).
    Ratio mantido pelo ffmpeg automaticamente com -1 no height.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps},scale={width}:-2,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    # height calculado por ffmpeg; precisamos descobrir depois
    proc = subprocess.run(cmd, capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extract failed rc={proc.returncode}")
    # Descobrir h via probe rápido
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(video_path)],
        capture_output=True, timeout=10,
    )
    stderr = probe.stderr.decode("utf-8", errors="replace")
    # Parse "Stream #0:0: Video: ... 1920x1080" pegando WxH original
    import re
    m = re.search(r"Stream #\d+:\d+.*?Video.*?(\d{2,5})x(\d{2,5})", stderr)
    if not m:
        raise RuntimeError("não foi possível detectar dimensões do input")
    in_w, in_h = int(m.group(1)), int(m.group(2))
    # Calcular h escalado mantendo ratio (ffmpeg arredonda para par)
    h = (in_h * width // in_w)
    if h % 2: h -= 1
    return proc.stdout, width, h, in_w, in_h


# ─────────────────────────────────────────────────────────────────────────────
# Detection + filtering + smoothing
# ─────────────────────────────────────────────────────────────────────────────

def _detect_largest_face(gray_frame, cascade) -> Optional[Tuple[int, int, int, int]]:
    """Detecta faces, retorna a maior (provavelmente streamer principal)."""
    h, w = gray_frame.shape[:2]
    min_size = max(20, int(min(w, h) * _MIN_FACE_SIZE_REL))
    faces = cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=4,
        minSize=(min_size, min_size),
    )
    if len(faces) == 0:
        return None
    # Maior por área
    return max(faces, key=lambda f: f[2] * f[3])


def _smooth_and_interpolate(samples: List[FaceSample],
                            total_t: float) -> List[FaceSample]:
    """
    Pós-processa amostras detectadas:
      1. Remove outliers (movimento >20% entre frames consecutivos).
      2. EMA suaviza posição (reduz jitter).
      3. Não interpola misses aqui — caller pode usar trajetória parcial.

    Mantém apenas hits (não preenche faltas). FFmpeg expression fará interpolação
    linear entre keypoints próximos via `lerp`.
    """
    if len(samples) < 2:
        return samples
    # Filtra outliers
    filtered = [samples[0]]
    for s in samples[1:]:
        prev = filtered[-1]
        dx = abs(s.cx_n - prev.cx_n)
        dy = abs(s.cy_n - prev.cy_n)
        if dx > _OUTLIER_MAX_JUMP_N or dy > _OUTLIER_MAX_JUMP_N:
            # Salto grande: provavelmente falso positivo. Pula.
            continue
        filtered.append(s)
    # EMA smoothing
    smoothed = [filtered[0]]
    for s in filtered[1:]:
        prev = smoothed[-1]
        cx = _EMA_ALPHA * s.cx_n + (1 - _EMA_ALPHA) * prev.cx_n
        cy = _EMA_ALPHA * s.cy_n + (1 - _EMA_ALPHA) * prev.cy_n
        smoothed.append(FaceSample(s.t, cx, cy, s.area_n))
    return smoothed


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_trajectory(video_path, duration_sec: float) -> Optional[TrackResult]:
    """
    Computa trajetória de face para o clip inteiro.

    Args:
        video_path: caminho do raw clip (.ts/.mp4)
        duration_sec: duração total estimada do clip (para validar hit_ratio)

    Returns:
        TrackResult com samples suavizados, ou None se:
          • opencv não disponível
          • hit_ratio < 30% (face raramente visível, melhor fallback)
          • erro de extração
    """
    if not _ensure_cv2():
        return None

    video_path = Path(video_path)
    try:
        raw, w, h, in_w, in_h = _extract_frames_gray(video_path, _SAMPLE_FPS)
    except Exception as e:
        logger.warning(f"extract frames falhou: {e}")
        return None

    frame_size = w * h
    n_frames = len(raw) // frame_size
    if n_frames < 2:
        return None

    import numpy as np
    arr = np.frombuffer(raw[: n_frames * frame_size], dtype=np.uint8).reshape(n_frames, h, w)

    samples: List[FaceSample] = []
    for i in range(n_frames):
        face = _detect_largest_face(arr[i], _cascade)
        if face is None:
            continue
        x, y, fw, fh = face
        cx_n = (x + fw / 2.0) / w
        cy_n = (y + fh / 2.0) / h
        area_n = (fw * fh) / float(w * h)
        t_sec = i / _SAMPLE_FPS
        samples.append(FaceSample(t_sec, cx_n, cy_n, area_n))

    hit_ratio = len(samples) / n_frames if n_frames else 0.0
    if hit_ratio < _HIT_RATIO_MIN:
        logger.debug(
            f"face tracking hit_ratio={hit_ratio:.0%} (<{int(_HIT_RATIO_MIN*100)}%); "
            "fallback para pad+blur"
        )
        return None

    smoothed = _smooth_and_interpolate(samples, duration_sec)
    if not smoothed:
        return None

    avg_area = sum(s.area_n for s in smoothed) / len(smoothed)
    logger.info(
        f"face tracking: hit={hit_ratio:.0%} ({len(samples)}/{n_frames}) "
        f"samples_smoothed={len(smoothed)} avg_face={avg_area*100:.1f}% input={in_w}x{in_h}"
    )
    return TrackResult(
        samples=smoothed, hit_ratio=hit_ratio, avg_area_n=avg_area,
        in_w=in_w, in_h=in_h,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Crop expression builder
# ─────────────────────────────────────────────────────────────────────────────

def build_crop_expr(track: TrackResult, out_aspect: float = 9 / 16) -> Tuple[str, int, int]:
    """
    Constrói FFmpeg crop expression PWL para trajetória.

    Retorna (x_expr, crop_w, crop_h) onde:
      • crop_w, crop_h: dimensões do crop (mantém altura, ajusta largura
        para aspect 9:16, ou vice-versa).
      • x_expr: string FFmpeg para o filtro crop=W:H:x_expr:0

    Estratégia PWL:
      • Reduz samples a no máx 12 keypoints (decimation uniforme) para evitar
        expression gigante.
      • Entre keypoints consecutivos: lerp linear.
      • Antes do 1º e depois do último: clampa ao keypoint extremo.
      • Clampa X final em [0, in_w - crop_w] para nunca vazar.
    """
    in_w, in_h = track.in_w, track.in_h
    # Target: vertical 9:16 com altura completa → crop W = H * 9/16
    crop_h = in_h
    crop_w = int(in_h * out_aspect)
    if crop_w % 2: crop_w -= 1
    max_x = in_w - crop_w

    samples = track.samples
    # Decimation para no máximo 12 keypoints
    MAX_KP = 12
    if len(samples) > MAX_KP:
        step = len(samples) / MAX_KP
        kps = [samples[min(int(i * step), len(samples) - 1)] for i in range(MAX_KP)]
    else:
        kps = list(samples)

    # Converte cx_n → x_pixel (canto esquerdo do crop, centrado no rosto)
    points: List[Tuple[float, float]] = []   # (t, x_pixel_clamped)
    for s in kps:
        cx_pix = s.cx_n * in_w
        x_left = cx_pix - crop_w / 2.0
        x_left = max(0.0, min(max_x, x_left))
        points.append((s.t, x_left))

    if not points:
        # Fallback degenerado: centraliza
        return (f"(in_w-{crop_w})/2", crop_w, crop_h)

    if len(points) == 1:
        return (f"{points[0][1]:.1f}", crop_w, crop_h)

    # Constrói expression PWL aninhada: if(lt(t, t1), val1, if(lt(t, t2), lerp(...), ...))
    # Para evitar gigantismo: vamos usar nested if + lerp por segmento.
    #
    # Forma: if(lt(t,T1), X0,
    #          if(lt(t,T2), lerp(X1,X2,(t-T1)/(T2-T1)),
    #            ...))
    # Início (t < t0): mantém X0
    # Fim   (t > tN):  mantém XN

    def lerp_seg(t0, t1, x0, x1):
        # FFmpeg lerp(a,b,k): a + (b-a)*k. k = (t-t0)/(t1-t0)
        dt = t1 - t0
        if dt < 1e-6:
            return f"{x1:.1f}"
        return f"lerp({x0:.1f},{x1:.1f},(t-{t0:.2f})/{dt:.2f})"

    # Build de trás para frente
    last_t, last_x = points[-1]
    expr = f"{last_x:.1f}"   # default = último valor (extrapolação constante)
    for i in range(len(points) - 1, 0, -1):
        t0, x0 = points[i - 1]
        t1, x1 = points[i]
        seg_expr = lerp_seg(t0, t1, x0, x1)
        expr = f"if(lt(t,{t1:.2f}),{seg_expr},{expr})"
    # Pre-roll: t < t0 mantém x0
    t0, x0 = points[0]
    expr = f"if(lt(t,{t0:.2f}),{x0:.1f},{expr})"

    # FFmpeg filter_graph usa ',' como separador entre filtros encadeados.
    # Vírgulas dentro de expressions (lerp(a,b,k), if(a,b,c)) precisam ser
    # escapadas como '\,' para que o parser não fragmente o filter.
    expr_escaped = expr.replace(",", "\\,")

    return expr_escaped, crop_w, crop_h
