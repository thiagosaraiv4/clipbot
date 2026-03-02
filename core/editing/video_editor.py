"""
editing/video_editor.py
Edita clips: redimensiona, adiciona legendas automáticas,
watermark, intro/outro e exporta em formato final.
"""

import logging
import subprocess
import tempfile
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

    def _segments_to_srt(self, segments: list, srt_path: Path):
        """Converte segmentos Whisper para arquivo .srt."""
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = self._seconds_to_srt_time(seg["start"])
                end   = self._seconds_to_srt_time(seg["end"])
                text  = seg["text"].strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    @staticmethod
    def _seconds_to_srt_time(s: float) -> str:
        h  = int(s // 3600)
        m  = int((s % 3600) // 60)
        sc = int(s % 60)
        ms = int((s % 1) * 1000)
        return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"

    # ------------------------------------------------------------------
    # Filtro ffmpeg
    # ------------------------------------------------------------------

    def _build_ffmpeg_filter(self, srt_path: Optional[Path] = None) -> str:
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

        # ── Legendas ────────────────────────────────────────────────────────
        sub_filter = ""
        if srt_path and cfg["subtitles"]["enabled"]:
            font      = cfg["subtitles"]["font"]
            font_size = cfg["subtitles"]["font_size"]
            outline_w = cfg["subtitles"]["outline_width"]
            srt_esc   = str(srt_path).replace("\\", "/").replace(":", "\\:")
            sub_filter = (
                f"subtitles='{srt_esc}':"
                f"force_style='FontName={font},FontSize={font_size},"
                f"PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,"
                f"Outline={outline_w},Bold=1,Alignment=2'"
            )

        return {
            "use_complex": use_complex,
            "complex_steps": filters if use_complex else [],
            "simple_vf": filters[0] if not use_complex else "",
            "sub_filter": sub_filter,
        }

    # ------------------------------------------------------------------
    # Processo principal de edição
    # ------------------------------------------------------------------

    def process(self, input_path: Path, output_name: str) -> Path:
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
            if cfg_sub["enabled"]:
                segments = self._transcribe(current, cfg_sub["language"])
                if segments:
                    srt_path = tmp / "subtitles.srt"
                    self._segments_to_srt(segments, srt_path)

            # --- Monta filtro ---
            flt = self._build_ffmpeg_filter(srt_path)
            cmd = ["ffmpeg", "-y", "-i", str(current)]

            if flt["use_complex"]:
                # Modo vertical: fundo blur + vídeo sobreposto
                steps = flt["complex_steps"].copy()   # split, bg, fg, overlay

                if flt["sub_filter"]:
                    # Adiciona legendas sobre o resultado final
                    steps[-1] = steps[-1].replace("[vout]", "[pre_sub]")
                    steps.append(f"[pre_sub]{flt['sub_filter']}[vout]")

                filter_complex = ";".join(steps)
                cmd += ["-filter_complex", filter_complex, "-map", "[vout]", "-map", "0:a?"]
            else:
                # Modo horizontal simples
                vf = flt["simple_vf"]
                if flt["sub_filter"]:
                    vf = vf + "," + flt["sub_filter"] if vf else flt["sub_filter"]
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
