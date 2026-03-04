"""
editing/video_editor.py
Edita clips: redimensiona, adiciona legendas automáticas,
watermark, intro/outro e exporta em formato final.
"""

import logging
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ClipBot.Editor")


class VideoEditor:
    def __init__(self, config: dict, output_folder: str):
        self.cfg    = config["editing"]
        self.out_dir = Path(output_folder)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Transcrição com Whisper
    # ------------------------------------------------------------------

    def _transcribe(self, video_path: Path, language: str) -> list:
        """Gera legenda via OpenAI Whisper. Retorna lista de segmentos."""
        try:
            import whisper
            logger.info("🎤 Transcrevendo com Whisper...")
            model = whisper.load_model("small")
            result = model.transcribe(str(video_path), language=language, word_timestamps=True)
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
    def _detect_commercial_offset(segments: list, max_skip: float = 8.0) -> float:
        """
        Verifica se os primeiros segmentos contêm 'commercial break'.
        Retorna:
          >0  = segundos para pular (trim do início)
          0   = sem comercial detectado
          -1  = comercial muito longo (>max_skip), clip deve ser SKIP
        """
        if not segments:
            return 0.0
        for seg in segments:
            if seg.get("end", 0) > 6.0:
                break
            text_lower = seg.get("text", "").lower()
            if "commercial break" in text_lower or "commercial" in text_lower:
                end_time = seg.get("end", 0)
                if end_time > max_skip:
                    return -1.0  # comercial muito longo → skip
                return end_time + 0.5  # +0.5s de margem
        return 0.0

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
        """Escapa caracteres especiais para o filtro drawtext do FFmpeg."""
        # Ordem importa: \ primeiro
        text = text.replace("\\", "\\\\")
        text = text.replace(":", "\\:")
        text = text.replace("'", "\\'")
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
            # ── Fundo borrado (bg) ──────────────────────────────────────────
            # Escala para cobrir todo o 9:16 e aplica blur forte
            bg = (
                f"scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h},"
                f"boxblur=20:5"
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
                subprocess.run([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(concat_list), "-c", "copy", str(intro_out)
                ], check=True, capture_output=True)
                current = intro_out

            # --- Outro ---
            if cfg_out["enabled"] and Path(cfg_out["video_path"]).exists():
                outro_out = tmp / "with_outro.mp4"
                concat_list = tmp / "concat_outro.txt"
                with open(concat_list, "w") as f:
                    f.write(f"file '{current}'\nfile '{cfg_out['video_path']}'\n")
                subprocess.run([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(concat_list), "-c", "copy", str(outro_out)
                ], check=True, capture_output=True)
                current = outro_out

            # --- Transcrição / Legendas ---
            segments = []
            commercial_offset = 0.0
            if cfg_sub["enabled"]:
                segments = self._transcribe(current, cfg_sub["language"])
                if segments:
                    # Detectar comercial nos primeiros segundos
                    commercial_offset = self._detect_commercial_offset(segments)
                    if commercial_offset < 0:
                        logger.warning("🚫 Comercial muito longo — clip SKIP")
                        return None

                    if commercial_offset > 0:
                        logger.info(f"📺 Comercial detectado: trimando {commercial_offset:.1f}s do início")
                        # Ajustar timestamps dos segmentos
                        adjusted = []
                        for seg in segments:
                            new_start = seg["start"] - commercial_offset
                            new_end   = seg["end"]   - commercial_offset
                            if new_end > 0:
                                adjusted.append({
                                    **seg,
                                    "start": max(0, new_start),
                                    "end":   new_end,
                                })
                        segments = adjusted

                    # --- Filtro de conteúdo (apostas/propaganda) ---
                    cf = self.cfg.get("content_filter", {})
                    if cf.get("enabled", False) and segments:
                        full_text = " ".join(
                            seg.get("text", "") for seg in segments
                        ).lower()
                        blacklist = cf.get("blacklist_keywords", [])
                        matched = [kw for kw in blacklist if kw.lower() in full_text]
                        if matched and cf.get("action", "skip") == "skip":
                            logger.warning(
                                f"🚫 Conteúdo filtrado (blacklist): {matched[:3]} — clip SKIP"
                            )
                            return None

                    srt_path = tmp / "subtitles.srt"
                    self._segments_to_srt(segments, srt_path)

            # --- Monta filtro ---
            flt = self._build_ffmpeg_filter(srt_path, title_text=title_text)
            cmd = ["ffmpeg", "-y"]
            if commercial_offset > 0:
                cmd += ["-ss", str(commercial_offset)]
            cmd += ["-i", str(current)]

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

            cmd += [
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-r", str(self.cfg["fps"]),
                str(output_path)
            ]

            logger.debug(f"ffmpeg cmd: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg edição falhou:\n{result.stderr.decode()}")

        logger.info(f"✅ Clip editado: {output_path}")
        return output_path
