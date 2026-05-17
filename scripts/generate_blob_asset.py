"""
scripts/generate_blob_asset.py
Gera asset PNG do blob vermelho usado no banner de título (estilo Real Oficial).

Forma: blob irregular (não-retangular), bordas com curvas suaves, leve textura.
Saída: assets/title_banner_blob.png — 1080×280, fundo transparente.

Como usar:
    python scripts/generate_blob_asset.py

Rode 1× e commita o PNG. Não precisa regenerar a cada run.
"""

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

W, H = 1080, 300              # canvas (deixa margem nas bordas pro blob não cortar)
BLOB_COLOR = (213, 36, 36)    # #D52424 — vermelho do Real Oficial (saturação alta)
SHADOW_BLUR = 8               # leve sombra atrás
SEED = 7                      # seed fixo → blob reproducível


def _bezier_blob_points(cx: float, cy: float, rx: float, ry: float,
                        n_points: int = 14, jitter: float = 0.18) -> list:
    """
    Gera N pontos em torno de uma elipse, com perturbação aleatória no raio
    para criar contorno irregular.
    """
    rng = random.Random(SEED)
    pts = []
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        # Jitter no raio: ±jitter*100% — bordas onduladas mas previsíveis
        r_jitter = 1.0 + rng.uniform(-jitter, jitter)
        x = cx + rx * r_jitter * math.cos(angle)
        y = cy + ry * r_jitter * math.sin(angle)
        pts.append((x, y))
    return pts


def _smooth_polygon(pts: list, smooth_iter: int = 3) -> list:
    """
    Chaikin smoothing: a cada iter, troca cada vértice por 2 pontos
    em 1/4 e 3/4 da aresta. Resultado: curva suave sem dependência externa.
    """
    for _ in range(smooth_iter):
        new_pts = []
        n = len(pts)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            q  = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r  = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new_pts.extend([q, r])
        pts = new_pts
    return pts


def generate_blob(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Canvas RGBA transparente
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # ── Sombra (camada offset + blur) ──────────────────────────────────────
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    cx, cy = W / 2, H / 2 + 6        # 6px abaixo p/ efeito drop-shadow
    rx, ry = W * 0.46, H * 0.36
    sh_pts = _smooth_polygon(_bezier_blob_points(cx, cy, rx, ry))
    shadow_draw.polygon(sh_pts, fill=(0, 0, 0, 110))
    shadow = shadow.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))
    img.alpha_composite(shadow)

    # ── Blob principal ─────────────────────────────────────────────────────
    blob = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    blob_draw = ImageDraw.Draw(blob)
    cx, cy = W / 2, H / 2
    pts = _smooth_polygon(_bezier_blob_points(cx, cy, rx, ry))
    blob_draw.polygon(pts, fill=BLOB_COLOR + (255,))
    img.alpha_composite(blob)

    # ── Highlight sutil (camada clara no topo, 25% opacidade) ──────────────
    # Dá sensação de volume/glossy sem virar pop-art
    hl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hl_draw = ImageDraw.Draw(hl)
    hl_pts = _smooth_polygon(
        _bezier_blob_points(cx, cy - H * 0.15, rx * 0.85, ry * 0.45, jitter=0.10)
    )
    hl_draw.polygon(hl_pts, fill=(255, 255, 255, 35))
    hl = hl.filter(ImageFilter.GaussianBlur(4))
    img.alpha_composite(hl)

    img.save(out_path, "PNG")
    print(f"OK: {out_path} ({W}x{H}, {out_path.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    out  = base / "assets" / "title_banner_blob.png"
    generate_blob(out)
