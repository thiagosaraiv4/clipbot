"""
editing/video_editor.py
Edita clips: redimensiona, adiciona legendas automáticas,
watermark, intro/outro e exporta em formato final.
"""

import logging
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

from ffmpeg_runner import run_ffmpeg
from metrics import Timer as _Timer, incr as _metrics_incr

logger = logging.getLogger("ClipBot.Editor")


class VideoEditor:
    # NVENC detection cached at class-level (uma vez por sessão de processo).
    # _runtime_disabled flag permite fallback automático se NVENC quebrar
    # mid-session (driver crash, contention, etc.).
    _nvenc_available:        Optional[bool] = None
    _nvenc_runtime_disabled: bool           = False

    def __init__(self, config: dict, output_folder: str):
        self.cfg    = config["editing"]
        self.out_dir = Path(output_folder)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # NVENC detection + encoder args
    # ------------------------------------------------------------------

    @classmethod
    def _detect_nvenc(cls) -> bool:
        """
        Detecta UMA VEZ se h264_nvenc funciona nesta máquina.
        Faz encode sintético de 0.1s para validar codec + driver + permissões
        (não basta conferir se ffmpeg foi compilado com nvenc).
        Resultado cacheado em class var.
        """
        if cls._nvenc_available is not None:
            return cls._nvenc_available
        try:
            r = run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=size=320x240:duration=0.1:rate=30",
                "-c:v", "h264_nvenc", "-preset:v", "p5",
                "-f", "null", "-",
            ], label="detect_nvenc", timeout=10, quiet=True)
            cls._nvenc_available = r.ok
            if r.ok:
                logger.info("✅ NVENC disponível — usando h264_nvenc no render")
            else:
                logger.info("ℹ️  NVENC indisponível — usando libx264 no render")
        except Exception as e:
            logger.debug(f"nvenc detect: {e}")
            cls._nvenc_available = False
        return cls._nvenc_available

    @classmethod
    def _use_nvenc(cls) -> bool:
        """True se NVENC detectado E não desabilitado em runtime."""
        return cls._detect_nvenc() and not cls._nvenc_runtime_disabled

    def _encoder_args(self, use_nvenc: bool) -> list:
        """
        Retorna args ffmpeg para encoder de vídeo.

        NVENC config (1660 Super Turing 7th gen):
          • preset p5 = quality balanced (NÃO usar slow/medium antigos — deprecated)
          • cq=22 quality target ≈ libx264 CRF 22 (melhor que 23)
          • spatial_aq + temporal_aq habilitados — qualidade visual ↑↑ em scenes complexas
          • bf=3 + refs=3 = compressão melhor sem custo perceptível
          • rc-lookahead=32 = compressão mais inteligente
          • profile high level 4.1 yuv420p = compat máxima TikTok/Instagram/YouTube

        libx264 fallback: preset fast CRF 23 (estado anterior).
        """
        fps = str(self.cfg.get("fps", 30))
        if use_nvenc:
            # NVENC config validada empiricamente em 1660 Super (Turing 7th gen):
            # - level omitido: 1080×1920 60fps excede level 4.1 (max ~245k MB/s).
            #   Auto-level evita rc=-22 sem prejuízo de compat (TikTok/IG aceitam 4.2+).
            # - tune=hq + rc=vbr + cq=22: equivale ~CRF 22 com qualidade espacial.
            # - spatial_aq + temporal_aq: AQ habilitado, qualidade visual ↑↑ em scenes complexas.
            # - bf=3 refs=3: B-frames + reference frames = compressão melhor.
            # - rc-lookahead=32: análise temporal melhor.
            return [
                "-c:v", "h264_nvenc",
                "-preset:v", "p5",
                "-tune:v", "hq",
                "-rc:v", "vbr",
                "-cq:v", "22",
                "-b:v", "0",
                "-maxrate:v", "12M",
                "-bufsize:v", "24M",
                "-profile:v", "high",
                "-spatial_aq", "1",
                "-temporal_aq", "1",
                "-bf", "3",
                "-refs", "3",
                "-rc-lookahead", "32",
                "-pix_fmt", "yuv420p",
                "-r", fps,
            ]
        else:
            return [
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-r", fps,
            ]

    # ------------------------------------------------------------------
    # Transcrição com Whisper
    # ------------------------------------------------------------------

    def _transcribe(self, video_path: Path, language: str) -> list:
        """
        Gera legenda via Whisper (singleton). Retorna lista de segmentos.
        word_timestamps=False: legenda atual usa só seg.start/end → custo extra
        de word-level é desperdício. Ativar quando ASS karaoke entrar (Iter2).
        """
        try:
            from whisper_service import WhisperService
            logger.info("🎤 Transcrevendo com Whisper...")
            # device="auto": tenta CUDA INT8_FP16 → CPU INT8 → openai-whisper CPU
            service = WhisperService.get(size="small", device="auto")
            with _Timer("editor.whisper", backend=service.backend or "unknown"):
                result = service.transcribe(
                    video_path, language=language,
                    word_timestamps=False, beam_size=1,
                )
            logger.info("✅ Transcrição concluída.")
            return result["segments"]
        except ImportError:
            logger.warning("Whisper não instalado. Legendas ignoradas.")
            return []
        except Exception as e:
            logger.error(f"Erro Whisper: {e}")
            return []

    @staticmethod
    def _wrap_subtitle_text(text: str, max_chars: int = 40, max_lines: int = 2) -> str:
        """Quebra texto longo em linhas de ≤max_chars, máximo max_lines."""
        words = text.split()
        if not words:
            return text
        lines = []
        current = words[0]
        for w in words[1:]:
            if len(current) + 1 + len(w) <= max_chars:
                current += " " + w
            else:
                lines.append(current)
                current = w
                if len(lines) >= max_lines - 1:
                    # Última linha: junta tudo restante
                    remaining = [w] + words[words.index(w)+1:] if w in words else [current]
                    current = current  # já tem a palavra atual
                    break
        lines.append(current)
        return "\n".join(lines[:max_lines])

    def _segments_to_srt(self, segments: list, srt_path: Path):
        """Converte segmentos Whisper para arquivo .srt com word wrap."""
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = self._seconds_to_srt_time(seg["start"])
                end   = self._seconds_to_srt_time(seg["end"])
                text  = self._wrap_subtitle_text(seg["text"].strip())
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    @staticmethod
    def _seconds_to_srt_time(s: float) -> str:
        h  = int(s // 3600)
        m  = int((s % 3600) // 60)
        sc = int(s % 60)
        ms = int((s % 1) * 1000)
        return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"

    # ------------------------------------------------------------------
    # Commercial break detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_commercial_frame(video_path, timestamp: float) -> bool:
        """
        Extrai mini-frame 64x36 em `timestamp` e calcula % de pixels
        com padrão roxo/azul da Twitch (R<120, B>140, B-R>40).
        Retorna True se >= 45% dos pixels são comercial.
        """
        try:
            cmd = [
                "ffmpeg", "-ss", str(timestamp), "-i", str(video_path),
                "-vframes", "1", "-vf", "scale=64:36",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "pipe:1"
            ]
            # commercial_frame chamado ×6 por clip; quiet evita poluir log
            result = run_ffmpeg(cmd, label="commercial_frame",
                                timeout=10, capture_stdout=True, quiet=True)
            if not result.ok or len(result.stdout) < 64 * 36 * 3:
                return False

            pixels = result.stdout
            total = 64 * 36
            commercial_px = 0
            for i in range(total):
                r = pixels[i * 3]
                g = pixels[i * 3 + 1]
                b = pixels[i * 3 + 2]
                if r < 120 and b > 140 and (b - r) > 40:
                    commercial_px += 1

            pct = commercial_px / total
            logger.debug(f"\U0001f4fa t={timestamp:.1f}s: {pct:.0%} px comercial")
            return pct >= 0.45

        except Exception as e:
            logger.debug(f"\U0001f4fa Erro frame t={timestamp}: {e}")
            return False

    @classmethod
    def _detect_commercial_offset(cls, video_path, max_total: float = 14.0) -> float:
        """
        Scan robusto: analisa frames nos primeiros 12s para encontrar
        comerçiais. Determina offset para trim.
        Retorna:
          >0  = segundos de comercial para trimar
          0   = sem comercial
          -1  = comercial muito longo (>max_total), SKIP
        """
        check_times = [0.5, 2.5, 4.5, 6.5, 8.5, 10.5]
        commercial_flags = []

        for t in check_times:
            is_comm = cls._is_commercial_frame(video_path, t)
            commercial_flags.append((t, is_comm))

        # Contar quantos são comercial
        comm_count = sum(1 for _, f in commercial_flags if f)

        if comm_count == 0:
            return 0.0

        # Encontrar o ULTIMO frame de comercial para definir offset
        last_commercial_t = 0.0
        for t, is_comm in commercial_flags:
            if is_comm:
                last_commercial_t = t

        # Offset = dopo do ultimo frame comercial + 2s margem
        offset = last_commercial_t + 2.0

        if offset > max_total:
            return -1.0  # comercial muito longo

        logger.info(
            f"\U0001f4fa Comercial: {comm_count}/{len(check_times)} frames, "
            f"\u00faltimo em t={last_commercial_t:.1f}s, offset={offset:.1f}s"
        )
        return offset

    # ------------------------------------------------------------------
    # Title Banner helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_title(text: str, max_chars: int = 50, max_lines: int = 1) -> str:
        """Limpa e trunca título. max_lines=1 pois drawtext não suporta \\n."""
        text = " ".join(text.strip().upper().split())  # remove espaços duplos
        if len(text) <= max_chars:
            return text
        words = text.split()
        lines, cur = [], ""
        for w in words:
            nxt = (cur + " " + w).strip()
            if cur and len(nxt) > max_chars:
                lines.append(cur.strip())
                cur = w
                if len(lines) >= max_lines:
                    break
            else:
                cur = nxt
        if cur and len(lines) < max_lines:
            lines.append(cur.strip())
        # Trunca última linha se ainda for longa
        if lines and len(lines[-1]) > max_chars:
            lines[-1] = lines[-1][:max_chars - 1].rstrip() + "…"
        return "\\n".join(lines) if lines else text[:max_chars - 1].rstrip() + "…"

    @staticmethod
    def _escape_drawtext(text: str) -> str:
        """
        Escapa caracteres especiais para drawtext do FFmpeg.

        Aspas simples são INTRATÁVEIS dentro de text='...': drawtext NÃO aceita
        \\' como escape — a primeira ' termina a string e o filter_complex
        inteiro quebra silenciosamente (label [vout] some, render falha com
        'Output with label vout does not exist'). Dois templates de keyword
        produziam títulos como "ELE DISSE 'CLIP'!" → quebravam render.

        Solução: substituir ' por aspas tipográficas Unicode U+2019 (')
        antes de qualquer escape. Visualmente idêntico, drawtext-safe.
        """
        # Substituir aspas simples por aspas tipográficas Unicode ANTES de escapar
        text = text.replace("'", "’")
        # Aspas duplas idem (paranoia + estilo)
        text = text.replace('"', "”")
        # Ordem importa: \ primeiro
        text = text.replace("\\", "\\\\")
        text = text.replace(":", "\\:")
        text = text.replace("%", "\\%")
        text = text.replace("\n", "\\n")
        return text

    @staticmethod
    def _ffmpeg_fontfile(font_path: str) -> str:
        """
        Converte path de fonte para formato FFmpeg.
        Windows: C:\\Windows\\Fonts\\arial.ttf → C\\:/Windows/Fonts/arial.ttf
        """
        # Troca backslash por forward slash
        fp = font_path.replace("\\", "/")
        # Escapa o ':' do drive letter (ex: C: → C\:)
        if len(fp) >= 2 and fp[1] == ":":
            fp = fp[0] + "\\:" + fp[2:]
        return fp

    def _build_title_filter(self, title_text: str) -> str:
        """Constrói filtro estilo clip (faixa full-width + texto bold grande com sombra)."""
        tb_cfg = self.cfg.get("title_banner", {})
        max_chars = tb_cfg.get("max_chars", 14)
        wrapped = self._wrap_title(title_text, max_chars=max_chars)
        escaped = self._escape_drawtext(wrapped)

        # ── Auto-fit fontsize baseado no comprimento do texto ─────────
        # Impact font em 1080px: cada char ocupa ~fontsize*0.55
        text_len = len(wrapped)
        if text_len > 30:
            fontsize = "h/30"
        elif text_len > 25:
            fontsize = "h/26"
        elif text_len > 20:
            fontsize = "h/24"
        elif text_len > 15:
            fontsize = "h/20"
        else:
            fontsize = "h/16"

        # ── Resolução de fonte (cadeia de fallback) ────────────────────
        # tb_cfg já lido acima
        cfg_font = tb_cfg.get("font_path", "")

        font_candidates = [
            cfg_font,
            "C:\\Windows\\Fonts\\impact.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
        resolved_font = None
        for fp in font_candidates:
            if fp and Path(fp).exists():
                resolved_font = fp
                break

        if resolved_font:
            font_arg = f"fontfile='{self._ffmpeg_fontfile(resolved_font)}'"
        else:
            font_arg = "font='Arial'"

        # ── Cores e dimensões configuráveis ────────────────────────────
        bg_color   = tb_cfg.get("bg_color", "0xCC0000")
        text_color = tb_cfg.get("text_color", "white")
        bg_opacity = tb_cfg.get("bg_opacity", 0.92)
        banner_dur = tb_cfg.get("banner_duration", 5)

        # Sanitiza '#RRGGBB' → '0xRRGGBB' (# é comentário no FFmpeg)
        if bg_color.startswith("#"):
            bg_color = "0x" + bg_color[1:]

        # drawbox: h/w são auto-referência do box; usar ih/iw para input
        # drawtext: h/w referem ao input, então OK
        strip_h_db = "ih/8"       # drawbox: altura da faixa
        y_pos_db   = "ih*0.40"    # drawbox: posição Y
        strip_h_dt = "h/8"        # drawtext: mesma proporção
        y_pos_dt   = "h*0.40"     # drawtext: mesma posição

        # Margem de segurança: clamp x para nunca vazar nas bordas
        x_expr = "min(max(40\\,(w-text_w)/2)\\,w-text_w-40)"

        # 1) drawbox: faixa colorida cobrindo 100% da largura (5s)
        # 2) drawtext: texto grande centralizado na faixa, com sombra (5s)
        return (
            f"drawbox=x=0:y={y_pos_db}:w=iw:h={strip_h_db}"
            f":color={bg_color}@{bg_opacity}:t=fill"
            f":enable='lt(t\\,{banner_dur})',"
            f"drawtext={font_arg}"
            f":text='{escaped}'"
            f":fontsize={fontsize}"
            f":fontcolor={text_color}"
            f":shadowcolor=black@0.75:shadowx=4:shadowy=4"
            f":x={x_expr}"
            f":y={y_pos_dt}+({strip_h_dt}-text_h)/2"
            f":enable='lt(t\\,{banner_dur})'"
        )

    # ------------------------------------------------------------------
    # Filtro ffmpeg
    # ------------------------------------------------------------------

    def _build_ffmpeg_filter(self, srt_path: Optional[Path] = None,
                             title_text: Optional[str] = None) -> str:
        """
        Constrói filtergraph ffmpeg.
        Modo vertical: vídeo inteiro em 9:16 com fundo desfocado (sem corte).
        """
        cfg = self.cfg
        w, h = cfg["resolution"]   # 1080 x 1920

        filters = []

        if cfg["output_format"] == "vertical":
            # ── Fundo borrado (bg) — BLUR LOW-RES ────────────────────────────
            # Bench medido: boxblur em 1080×1920 = ~45s/clip de 30s (75% do render!).
            # Trocando para 270×480 (16× menos pixels) + upscale após blur:
            #   • Visualmente idêntico — blur já é borrado, ninguém vê o downscale.
            #   • boxblur com sigma menor (5:2 em vez de 20:5) compensa resolução
            #     menor mantendo aparência.
            #   • Render 60s → 14s no libx264; com NVENC vira ~10s.
            BG_W, BG_H = 270, 480
            bg = (
                f"scale={BG_W}:{BG_H}:force_original_aspect_ratio=increase,"
                f"crop={BG_W}:{BG_H},"
                f"boxblur=5:2,"
                f"scale={w}:{h}"
            )

            # ── Vídeo principal (fg) ────────────────────────────────────────
            # Reduz para caber inteiramente dentro de 1080x1920 mantendo proporção
            fg = (
                f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black@0"   # centraliza, fundo transparente
            )

            # ── Sobrepõe fg sobre bg ────────────────────────────────────────
            # Resultado: fundo desfocado + vídeo centralizado nítido por cima
            filters.append(f"[0:v]split=2[bg_raw][fg_raw]")
            filters.append(f"[bg_raw]{bg}[bg]")
            filters.append(f"[fg_raw]{fg}[fg]")
            filters.append(f"[bg][fg]overlay=(W-w)/2:(H-h)/2[vout]")
            use_complex = True
        else:
            filters.append(f"scale={w}:{h}")
            use_complex = False

        # ── Legendas (com safe area dinâmica) ─────────────────────────────
        sub_filter = ""
        if srt_path and cfg["subtitles"]["enabled"]:
            font      = cfg["subtitles"]["font"]
            font_size = cfg["subtitles"]["font_size"]
            outline_w = cfg["subtitles"]["outline_width"]
            srt_esc   = str(srt_path).replace("\\", "/").replace(":", "\\:")

            # MarginV dinâmico por preset de plataforma
            layout    = cfg.get("layout_preset", "tiktok")
            margin_v  = cfg["subtitles"].get("margin_v", None)
            if margin_v is None:
                _preset_margins = {"tiktok": 220, "shorts": 200, "reels": 210}
                margin_v = _preset_margins.get(layout, 220)

            # FontSize limitado para não dominar a tela em 1080×1920
            effective_fs = min(font_size, 12)

            sub_filter = (
                f"subtitles='{srt_esc}':"
                f"force_style='FontName={font},FontSize={effective_fs},"
                f"PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,"
                f"Outline={outline_w},Bold=1,Alignment=2,"
                f"MarginV={margin_v},WrapStyle=0'"
            )

        # ── Title banner (drawtext) ──────────────────────────────────────────
        title_filter = ""
        tb_enabled = self.cfg.get("title_banner", {}).get("enabled", True)
        if title_text and title_text.strip() and tb_enabled:
            title_filter = self._build_title_filter(title_text)

        return {
            "use_complex": use_complex,
            "complex_steps": filters if use_complex else [],
            "simple_vf": filters[0] if not use_complex else "",
            "sub_filter": sub_filter,
            "title_filter": title_filter,
        }

    # ------------------------------------------------------------------
    # Processo principal de edição
    # ------------------------------------------------------------------

    def process(self, input_path: Path, output_name: str,
                title_text: Optional[str] = None) -> Path:
        """Processa um clip bruto e retorna o arquivo final editado."""
        output_path = self.out_dir / f"{output_name}.mp4"
        logger.info(f"✂️  Editando: {input_path.name} → {output_path.name}")
        # Timer end-to-end (inclui commercial detect, whisper, render)
        _t = _Timer("editor.process_total")
        _t.__enter__()
        try:
            return self._process_inner(input_path, output_path, title_text)
        finally:
            _t.__exit__(None, None, None)

    def _process_inner(self, input_path: Path, output_path: Path,
                       title_text: Optional[str] = None) -> Path:

        cfg_sub  = self.cfg["subtitles"]
        cfg_wm   = self.cfg["watermark"]
        cfg_in   = self.cfg["intro"]
        cfg_out  = self.cfg["outro"]

        srt_path = None

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            current = input_path

            # --- Intro ---
            if cfg_in["enabled"] and Path(cfg_in["video_path"]).exists():
                intro_out = tmp / "with_intro.mp4"
                concat_list = tmp / "concat_intro.txt"
                with open(concat_list, "w") as f:
                    f.write(f"file '{cfg_in['video_path']}'\nfile '{current}'\n")
                r = run_ffmpeg([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(concat_list), "-c", "copy", str(intro_out),
                ], label="concat_intro", timeout=120)
                if not r.ok:
                    raise RuntimeError(f"intro concat falhou rc={r.returncode}")
                current = intro_out

            # --- Outro ---
            if cfg_out["enabled"] and Path(cfg_out["video_path"]).exists():
                outro_out = tmp / "with_outro.mp4"
                concat_list = tmp / "concat_outro.txt"
                with open(concat_list, "w") as f:
                    f.write(f"file '{current}'\nfile '{cfg_out['video_path']}'\n")
                r = run_ffmpeg([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(concat_list), "-c", "copy", str(outro_out),
                ], label="concat_outro", timeout=120)
                if not r.ok:
                    raise RuntimeError(f"outro concat falhou rc={r.returncode}")
                current = outro_out

            # --- Detecção de comercial por análise de frame (robusto) ---
            with _Timer("editor.commercial_detect"):
                commercial_offset = self._detect_commercial_offset(current)
            if commercial_offset < 0:
                _metrics_incr("editor.skipped", reason="commercial_long")
                logger.warning("\U0001f6ab Comercial muito longo (>14s) \u2014 clip SKIP")
                return None

            if commercial_offset > 0:
                logger.info(f"\U0001f4fa Comercial: trimando {commercial_offset:.1f}s + reset PTS")
                trimmed = tmp / "trimmed.ts"
                # Pre-trim COM reset de PTS (corrige desync)
                run_ffmpeg([
                    "ffmpeg", "-y", "-ss", str(commercial_offset),
                    "-i", str(current),
                    "-vf", "setpts=PTS-STARTPTS",
                    "-af", "asetpts=PTS-STARTPTS",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k",
                    str(trimmed),
                ], label="commercial_pretrim", timeout=60)
                if trimmed.exists() and trimmed.stat().st_size > 1000:
                    # Recheck: verificar se ainda tem comercial no trimado
                    recheck = self._detect_commercial_offset(trimmed)
                    if recheck < 0:
                        logger.warning("\U0001f6ab Comercial persist\u00ea ap\u00f3s trim \u2014 SKIP")
                        _metrics_incr("editor.skipped", reason="commercial_persistent")
                        return None
                    elif recheck > 0:
                        logger.info(f"\U0001f4fa Recheck: mais {recheck:.1f}s de comercial")
                        retrimmed = tmp / "retrimmed.ts"
                        run_ffmpeg([
                            "ffmpeg", "-y", "-ss", str(recheck),
                            "-i", str(trimmed),
                            "-vf", "setpts=PTS-STARTPTS",
                            "-af", "asetpts=PTS-STARTPTS",
                            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                            "-c:a", "aac", "-b:a", "192k",
                            str(retrimmed),
                        ], label="commercial_retrim", timeout=60)
                        if retrimmed.exists() and retrimmed.stat().st_size > 1000:
                            current = retrimmed
                        else:
                            current = trimmed
                    else:
                        current = trimmed
                else:
                    logger.warning("\u26a0\ufe0f Trim de comercial falhou, usando original")

            # --- Transcrição / Legendas ---
            segments = []
            if cfg_sub["enabled"]:
                segments = self._transcribe(current, cfg_sub["language"])
                if segments:
                    # --- Filtro de conteúdo (apostas/propaganda) ---
                    cf = self.cfg.get("content_filter", {})
                    if cf.get("enabled", False):
                        full_text = " ".join(
                            seg.get("text", "") for seg in segments
                        ).lower()
                        blacklist = cf.get("blacklist_keywords", [])
                        matched = [kw for kw in blacklist if kw.lower() in full_text]
                        if matched and cf.get("action", "skip") == "skip":
                            _metrics_incr("editor.skipped", reason="content_filter")
                            logger.warning(
                                f"\U0001f6ab Conteúdo filtrado (blacklist): {matched[:3]} \u2014 clip SKIP"
                            )
                            return None

                    srt_path = tmp / "subtitles.srt"
                    self._segments_to_srt(segments, srt_path)

            # --- Monta filtro ---
            flt = self._build_ffmpeg_filter(srt_path, title_text=title_text)
            cmd = ["ffmpeg", "-y", "-i", str(current)]

            if flt["use_complex"]:
                # Modo vertical: fundo blur + vídeo sobreposto
                steps = flt["complex_steps"].copy()   # split, bg, fg, overlay

                # Pipeline: overlay → title_banner → subtitles
                # O último step atual termina com [vout]

                if flt["title_filter"] and flt["sub_filter"]:
                    # Título + legendas: overlay→[pre_title]→title→[pre_sub]→subs→[vout]
                    steps[-1] = steps[-1].replace("[vout]", "[pre_title]")
                    steps.append(f"[pre_title]{flt['title_filter']}[pre_sub]")
                    steps.append(f"[pre_sub]{flt['sub_filter']}[vout]")
                elif flt["title_filter"]:
                    # Só título: overlay→[pre_title]→title→[vout]
                    steps[-1] = steps[-1].replace("[vout]", "[pre_title]")
                    steps.append(f"[pre_title]{flt['title_filter']}[vout]")
                elif flt["sub_filter"]:
                    # Só legendas (comportamento original)
                    steps[-1] = steps[-1].replace("[vout]", "[pre_sub]")
                    steps.append(f"[pre_sub]{flt['sub_filter']}[vout]")

                filter_complex = ";".join(steps)
                cmd += ["-filter_complex", filter_complex, "-map", "[vout]", "-map", "0:a?"]
            else:
                # Modo horizontal simples
                vf_parts = []
                if flt["simple_vf"]:
                    vf_parts.append(flt["simple_vf"])
                if flt["title_filter"]:
                    vf_parts.append(flt["title_filter"])
                if flt["sub_filter"]:
                    vf_parts.append(flt["sub_filter"])
                vf = ",".join(vf_parts)
                cmd += ["-vf", vf]

            # Encoder: NVENC se disponível, libx264 fallback. Áudio idem em ambos.
            use_nvenc = self._use_nvenc()
            video_args = self._encoder_args(use_nvenc=use_nvenc)
            audio_args = [
                "-c:a", "aac", "-b:a", "128k",
                "-af", "aresample=async=1:first_pts=0",
            ]
            mux_args = ["-movflags", "+faststart"]   # TikTok/IG preview-friendly
            cmd_base = list(cmd)   # snapshot p/ retry caso NVENC falhe runtime
            cmd += video_args + audio_args + mux_args + [str(output_path)]

            encoder_tag = "nvenc" if use_nvenc else "libx264"
            with _Timer("editor.render_final",
                        format=self.cfg.get("output_format", "?"),
                        encoder=encoder_tag):
                result = run_ffmpeg(cmd, label="render_final",
                                    timeout=300, log_command=True)

            # Fallback runtime: NVENC falha → desabilita resto da sessão + retry libx264
            if not result.ok and use_nvenc:
                tail_blob = " ".join(result.stderr_tail[-30:]).lower()
                if "nvenc" in tail_blob or "cuda" in tail_blob or "gpu" in tail_blob:
                    logger.warning(
                        f"⚠️  NVENC runtime falhou (rc={result.returncode}) — "
                        "fallback para libx264 e desabilitando NVENC nesta sessão"
                    )
                    self.__class__._nvenc_runtime_disabled = True
                    _metrics_incr("editor.nvenc_runtime_disabled")
                    # Retry com libx264
                    cmd_retry = cmd_base + self._encoder_args(use_nvenc=False) \
                                + audio_args + mux_args + [str(output_path)]
                    with _Timer("editor.render_final",
                                format=self.cfg.get("output_format", "?"),
                                encoder="libx264_fallback"):
                        result = run_ffmpeg(cmd_retry, label="render_final",
                                            timeout=300, log_command=True)

            if not result.ok:
                _metrics_incr("editor.render_failed",
                              rc=str(result.returncode), encoder=encoder_tag)
                tail = "\n".join(result.stderr_tail[-30:])
                raise RuntimeError(
                    f"ffmpeg render falhou rc={result.returncode}:\n{tail}"
                )

        _metrics_incr("editor.clips_generated", encoder=encoder_tag)
        logger.info(f"✅ Clip editado: {output_path}")
        return output_path
