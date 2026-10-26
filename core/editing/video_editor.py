"""
editing/video_editor.py
Edita clips: redimensiona, adiciona legendas automáticas,
watermark, intro/outro e exporta em formato final.
"""

import logging
import re
import shutil
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

from ffmpeg_runner import run_ffmpeg
from metrics import Timer as _Timer, incr as _metrics_incr


def _blacklist_hits(text: str, blacklist) -> list:
    """
    Termos da blacklist presentes no texto como PALAVRA ou FRASE inteira.

    Antes era `kw in texto`, por pedaço: 'pix' casava em "pixel", 'bet' em
    "diabetes" e "alfabeto", 'stake' em "mistake" — e o clip era descartado
    (com o .ts apagado). Fronteira de palavra Unicode (acentos contam como
    letra) e espaços repetidos do Whisper tratados como um só.
    """
    norm = re.sub(r"\s+", " ", (text or "").lower())
    hits = []
    for kw in blacklist or []:
        words = (kw or "").lower().split()
        if not words:
            continue
        pattern = r"(?<!\w)" + r"\s+".join(re.escape(w) for w in words) + r"(?!\w)"
        if re.search(pattern, norm):
            hits.append(kw)
    return hits

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
                # word_timestamps=True: necessário para karaoke ASS (Iter2 PR2).
                # Custo extra ~8× no transcribe mas é único hit por clip.
                # Para SRT plano, defaults seriam beam_size=1, word_timestamps=False.
                result = service.transcribe(
                    video_path, language=language,
                    word_timestamps=True, beam_size=1,
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
    # ASS karaoke subtitles (Iter2 PR2 — estilo TikTok/CapCut)
    # ------------------------------------------------------------------

    @staticmethod
    def _seconds_to_ass_time(s: float) -> str:
        """ASS format: H:MM:SS.cc (centissegundos, single digit hour)."""
        if s < 0: s = 0.0
        h  = int(s // 3600)
        m  = int((s % 3600) // 60)
        sc = s % 60
        return f"{h}:{m:02d}:{sc:05.2f}"

    @staticmethod
    def _escape_ass_text(text: str) -> str:
        """Escape de caracteres especiais ASS — `{`, `}`, `\\`, newline."""
        text = text.replace("\\", "\\\\")
        text = text.replace("{", "\\{").replace("}", "\\}")
        text = text.replace("\n", "\\N").replace("\r", "")
        return text

    def _build_ass_header(self) -> str:
        """
        Header ASS estilo Real Oficial (vertical 9:16 TikTok/Reels).
          • PlayRes 1080×1920.
          • Fonte: Impact (heavy bold) com fallback Arial Black.
          • Tamanho 78 ≈ 4% altura — visível em mobile, não invasivo.
          • Cor base: AMARELO (palavras padrão — futuras E passadas).
          • Cor active (verde): aplicada via eventos overlay alpha-mask por
            palavra em _segments_to_ass (não está no Style).
          • Alignment 8 = top-center. MarginV = distância do TOPO até o topo
            do texto. 620 ≈ 32% do topo, deixa centro do texto em ~700 (37%).
        """
        cfg_sub = self.cfg.get("subtitles", {})
        layout  = self.cfg.get("layout_preset", "tiktok")
        # MarginV calibrado por preset. Estilo Real Oficial usa centro-superior.
        _preset_margins = {"tiktok": 620, "shorts": 600, "reels": 610}
        margin_v = cfg_sub.get("margin_v") or _preset_margins.get(layout, 620)

        font_size = cfg_sub.get("ass_font_size", 78)

        return (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 1080\n"
            "PlayResY: 1920\n"
            "ScaledBorderAndShadow: yes\n"
            # WrapStyle 0 = smart wrapping (libass quebra automaticamente para
            # caber dentro de PlayResX - MarginL - MarginR). Antes (WrapStyle 2)
            # desabilitava quebra → frases longas saíam da tela.
            "WrapStyle: 0\n"
            "\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            # PrimaryColour = amarelo (&H00BBGGRR — yellow é &H0000FFFF).
            # SecondaryColour = amarelo idem (não usamos \k clássico mais).
            # OutlineColour = preto, Outline=3 (mais fino que antes).
            # Alignment=8 = top-center. MarginV = distância do topo.
            f"Style: Default,Impact,{font_size},&H0000FFFF,&H0000FFFF,"
            f"&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,0,8,60,60,{margin_v},1\n"
            "\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
            "MarginV, Effect, Text\n"
        )

    @staticmethod
    def _normalize_word(s: str) -> str:
        """
        Normaliza palavra para matching de subtitle_emojis:
          • lowercase
          • remove pontuação ao redor
          • remove acentos (compatível com keys vindas da LLM sem acento)
        """
        import unicodedata
        s = s.strip().lower().strip(".,!?;:\"'()[]{}…")
        # Remove diacríticos
        s = "".join(
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )
        return s

    def _segments_to_ass(self, segments: list, ass_path: Path,
                         subtitle_emojis: Optional[dict] = None) -> None:
        """
        Converte segmentos Whisper para ASS estilo Real Oficial.

        Estratégia "layered alpha-mask" (verde só na palavra ativa):
          1. Base event (layer 0): linha INTEIRA em amarelo, dura toda a linha.
          2. Para cada palavra: 1 event overlay (layer 1) com:
             • Início/fim alinhado ao timestamp da palavra.
             • Texto = linha completa com TODAS palavras em \\alpha&HFF& (invisível)
               EXCETO a palavra ativa em \\alpha&H00&\\1c&H00FF00& (verde opaca).
             • libass renderiza no MESMO local que a base (alignment idêntico,
               glyphs invisíveis preservam layout) → a palavra ativa fica
               sobreposta exatamente em cima da versão amarela.
          Resultado visual: linha toda amarela, palavra atual flasha verde,
          volta amarela quando próxima começa. Igual screenshot Real Oficial.

        Fallback se segment não tem `words` (openai-whisper sem word_ts):
          • Linha plana só em amarelo (sem flash). Visível ainda assim.
        """
        header = self._build_ass_header()
        events = []

        for seg in segments:
            words = seg.get("words") or []
            text  = (seg.get("text") or "").strip()
            if not text:
                continue

            seg_start = seg.get("start", 0.0)
            seg_end   = seg.get("end",   seg_start + 1.0)
            if seg_end <= seg_start:
                continue

            # Filtra apenas palavras com texto válido
            valid_words = [
                w for w in words if (w.get("word") or "").strip()
            ] if words else []

            if valid_words:
                # Normaliza mapa de emojis uma vez (lowercase, sem acento).
                # LLM já é instruída a retornar keys lowercase sem acento — esse
                # normalize aqui é defensivo (renormaliza pra garantir match).
                emoji_map = {}
                if subtitle_emojis:
                    for k, v in subtitle_emojis.items():
                        nk = self._normalize_word(k)
                        if nk and v:
                            emoji_map[nk] = v

                # Texto escapado por palavra. Se a palavra bate com uma key do
                # emoji_map, anexa o emoji adjacente (mesma posição em base
                # e overlays → layout idêntico, alpha-mask alinha).
                escaped = []
                for w in valid_words:
                    raw = (w.get("word") or "").strip()
                    e = self._escape_ass_text(raw)
                    if emoji_map:
                        norm = self._normalize_word(raw)
                        em = emoji_map.get(norm)
                        if em:
                            e = f"{e} {em}"
                    escaped.append(e)
                line_start = valid_words[0].get("start", seg_start)
                line_end   = valid_words[-1].get("end",   seg_end)

                # ── 1. Base event: linha inteira em amarelo (PrimaryColour default) ──
                base_text = " ".join(escaped)
                base_start = self._seconds_to_ass_time(line_start)
                base_end   = self._seconds_to_ass_time(line_end)
                events.append(
                    f"Dialogue: 0,{base_start},{base_end},Default,,0,0,0,,{base_text}\n"
                )

                # ── 2. Per-word overlay: green flash durante palavra ativa ──
                for idx, w in enumerate(valid_words):
                    w_start = w.get("start", line_start)
                    w_end   = w.get("end",   w_start + 0.2)
                    if w_end <= w_start:
                        continue

                    # Linha mascarada: todas \alpha=FF (totalmente invisíveis,
                    # inclui outline+shadow), exceto idx que recebe:
                    #   • \1a=00 (primary fill visível)
                    #   • \3a=00 (outline preto visível — borda do texto)
                    #   • \1c=&H00FF00 (cor primary verde em BBGGRR)
                    # Resultado: palavra ativa = green fill + black outline,
                    # palavras inativas = 100% transparentes (mas ocupam espaço,
                    # preservando layout idêntico ao base event amarelo).
                    parts = []
                    for j, word_txt in enumerate(escaped):
                        if j == idx:
                            parts.append(
                                f"{{\\alpha&HFF&\\1a&H00&\\3a&H00&"
                                f"\\1c&H00FF00&}}{word_txt}"
                            )
                        else:
                            parts.append(f"{{\\alpha&HFF&}}{word_txt}")
                    masked = " ".join(parts)

                    ws_str = self._seconds_to_ass_time(w_start)
                    we_str = self._seconds_to_ass_time(w_end)
                    events.append(
                        f"Dialogue: 1,{ws_str},{we_str},Default,,0,0,0,,{masked}\n"
                    )

            else:
                # Sem word_timestamps: linha plana amarela (sem flash)
                text_line = self._escape_ass_text(text)
                start_str = self._seconds_to_ass_time(seg_start)
                end_str   = self._seconds_to_ass_time(seg_end)
                events.append(
                    f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text_line}\n"
                )

        ass_path.write_text(header + "".join(events), encoding="utf-8")

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

    @staticmethod
    def _ffmpeg_movie_path(p: Path) -> str:
        """
        Path normalizado para o filtro `movie=` (Windows precisa de escape '\:').
        Ex: C:\\path\\file.png → C\\:/path/file.png
        """
        s = str(p).replace("\\", "/")
        if len(s) >= 2 and s[1] == ":":
            s = s[0] + "\\:" + s[2:]
        return s

    def _resolve_title_font(self) -> str:
        """Resolve fontfile pro FFmpeg drawtext (cadeia de fallback)."""
        tb_cfg = self.cfg.get("title_banner", {})
        cfg_font = tb_cfg.get("font_path", "")
        font_candidates = [
            cfg_font,
            "C:\\Windows\\Fonts\\impact.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
        for fp in font_candidates:
            if fp and Path(fp).exists():
                return f"fontfile='{self._ffmpeg_fontfile(fp)}'"
        return "font='Arial'"

    def _blob_asset_path(self) -> Optional[Path]:
        """
        Retorna path do PNG do blob vermelho (estilo Real Oficial) se existir.
        None → fallback para faixa retangular drawbox.
        """
        # Asset: assets/title_banner_blob.png (gerado por scripts/generate_blob_asset.py)
        # Procura relativo ao módulo (../../../assets) e relativo ao cwd.
        candidates = [
            Path(__file__).resolve().parent.parent.parent / "assets" / "title_banner_blob.png",
            Path.cwd() / "assets" / "title_banner_blob.png",
        ]
        # Permite override por config (editing.title_banner.blob_png)
        override = self.cfg.get("title_banner", {}).get("blob_png", "")
        if override:
            candidates.insert(0, Path(override))
        for p in candidates:
            if p.exists():
                return p
        return None

    def _build_title_filter(self, title_text: str,
                            in_label: str = "pre_title",
                            out_label: str = "pre_sub") -> list:
        """
        Constrói filtros do banner de título.

        Retorna LISTA de steps semicolon-separated (já rotulados).
        Wire-up no _process_inner faz steps.extend(result) — não precisa replace().

        Modo blob (assets/title_banner_blob.png existe):
          movie=blob.png [blob] → overlay no topo → drawtext do título por cima
        Modo legado (sem PNG):
          drawbox + drawtext (faixa retangular, compatibilidade)
        """
        tb_cfg = self.cfg.get("title_banner", {})
        # max_chars antigo (50) era pra banner 1-linha. Blob 2-line precisa ~18/linha.
        # Use `line_max_chars` (nova chave) ou cai pra 18 default. Ignora `max_chars`.
        max_chars  = tb_cfg.get("line_max_chars", 18)
        banner_dur = tb_cfg.get("banner_duration", 5)
        text_color = tb_cfg.get("text_color", "white")

        # Wrap em 2 linhas (Real Oficial usa 2 linhas curtas no blob).
        # _wrap_title separa por `\\n` literal (FFmpeg drawtext escape), não \n real.
        wrapped = self._wrap_title(title_text, max_chars=max_chars, max_lines=2)
        if "\\n" in wrapped:
            raw_lines = wrapped.split("\\n")
        else:
            raw_lines = [wrapped]
        # Sanitiza por linha; cada drawtext fica com 1 linha sem \n
        lines = [self._escape_drawtext(l.strip()) for l in raw_lines if l.strip()][:2]

        # Auto-fit fontsize: 2 linhas precisa mais compacto
        max_line_len = max(len(l) for l in lines) if lines else 1
        if max_line_len > 22:
            fontsize_val = 64
        elif max_line_len > 16:
            fontsize_val = 78
        else:
            fontsize_val = 92

        font_arg = self._resolve_title_font()
        x_expr = "(w-text_w)/2"
        enable = f"lt(t\\,{banner_dur})"

        # Posição vertical do banner: 6% do topo (acima das legendas)
        # 1920 * 0.06 = ~115px do topo. Centro do blob fica ~y=180.
        blob_top_y = 60        # margem do topo do blob ao topo da tela
        blob_h = 280           # altura visível do blob (matches asset)
        # Texto centralizado no centro do blob
        text_y_center = blob_top_y + blob_h // 2     # = 200

        # Y de cada linha (2 linhas empilhadas centradas no centro do blob)
        if len(lines) >= 2:
            y1 = f"({text_y_center} - text_h - 4)"
            y2 = f"({text_y_center} + 4)"
        else:
            y1 = f"({text_y_center} - text_h/2)"
            y2 = None

        # ── Modo blob (PNG asset existe) ─────────────────────────────────────
        blob_path = self._blob_asset_path()
        if blob_path:
            movie_path = self._ffmpeg_movie_path(blob_path)
            steps = [
                # Carrega PNG. ATENÇÃO: `loop=0` em movie filter significa LOOP
                # INFINITO — trava o pipeline ffmpeg pq overlay nunca recebe EOF
                # do blob. Default (loop=1, sem flag) = lê o single-frame uma vez,
                # depois EOF. Overlay default eof_action=repeat → mostra o último
                # frame indefinidamente, que é exatamente o que queremos.
                f"movie='{movie_path}',scale=1080:{blob_h}[__blob]",
                # Overlay do blob sobre o vídeo principal nos primeiros banner_dur seg
                f"[{in_label}][__blob]overlay=x=(W-w)/2:y={blob_top_y}"
                f":enable='{enable}'[__blob_bg]",
            ]
            # Drawtext linha 1
            dt1 = (
                f"[__blob_bg]drawtext={font_arg}:text='{lines[0]}'"
                f":fontsize={fontsize_val}:fontcolor={text_color}"
                f":borderw=3:bordercolor=black"
                f":x={x_expr}:y={y1}"
                f":enable='{enable}'"
            )
            if y2 is None:
                steps.append(f"{dt1}[{out_label}]")
            else:
                steps.append(f"{dt1}[__tline1]")
                steps.append(
                    f"[__tline1]drawtext={font_arg}:text='{lines[1]}'"
                    f":fontsize={fontsize_val}:fontcolor={text_color}"
                    f":borderw=3:bordercolor=black"
                    f":x={x_expr}:y={y2}"
                    f":enable='{enable}'[{out_label}]"
                )
            return steps

        # ── Fallback legado: faixa retangular drawbox ────────────────────────
        # Mantido para builds sem o asset PNG (compatibilidade).
        bg_color   = tb_cfg.get("bg_color", "0xCC0000")
        bg_opacity = tb_cfg.get("bg_opacity", 0.92)
        if bg_color.startswith("#"):
            bg_color = "0x" + bg_color[1:]
        line_join = lines[0] if len(lines) == 1 else (lines[0] + "  " + lines[1])
        chain = (
            f"drawbox=x=0:y=ih*0.06:w=iw:h=ih/8"
            f":color={bg_color}@{bg_opacity}:t=fill"
            f":enable='{enable}',"
            f"drawtext={font_arg}:text='{line_join}'"
            f":fontsize={fontsize_val}:fontcolor={text_color}"
            f":borderw=3:bordercolor=black"
            f":x={x_expr}:y=ih*0.06+(ih/8-text_h)/2"
            f":enable='{enable}'"
        )
        return [f"[{in_label}]{chain}[{out_label}]"]

    # ------------------------------------------------------------------
    # Filtro ffmpeg
    # ------------------------------------------------------------------

    def _build_ffmpeg_filter(self, srt_path: Optional[Path] = None,
                             title_text: Optional[str] = None,
                             face_track=None) -> str:
        """
        Constrói filtergraph ffmpeg.

        Modo vertical:
          • face_track presente   → speaker tracking (crop dinâmico seguindo rosto)
          • face_track None       → fallback pad+blur (compatibilidade)
        """
        cfg = self.cfg
        w, h = cfg["resolution"]   # 1080 x 1920

        filters = []

        if cfg["output_format"] == "vertical" and face_track is not None:
            # ── Speaker tracking: crop horizontal dinâmico + scale ────────────
            # Não há pad nem blur — tela vertical é preenchida 100% pelo crop.
            # Esse é o padrão OpusClip/Cutlabs/TikTok 2024.
            from face_tracker import build_crop_expr
            x_expr, crop_w, crop_h = build_crop_expr(face_track, out_aspect=w / h)
            # Single chain (sem split/overlay) → muito mais rápido + visual cheio.
            filters.append(
                f"[0:v]crop={crop_w}:{crop_h}:{x_expr}:0,scale={w}:{h}[vout]"
            )
            use_complex = True

        elif cfg["output_format"] == "vertical":
            # ── Fallback pad+blur (sem face track ou hit_ratio<30%) ──────────
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
            fg = (
                f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black@0"
            )
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
            srt_esc = str(srt_path).replace("\\", "/").replace(":", "\\:")
            is_ass  = str(srt_path).lower().endswith(".ass")

            if is_ass:
                # Iter2 PR2: arquivo .ass já contém [V4+ Styles] inline.
                # force_style sobrescreveria o karaoke; deixar libass usar
                # o style do próprio arquivo é o que faz \k funcionar.
                sub_filter = f"subtitles='{srt_esc}'"
            else:
                # Fallback SRT legado (force_style estilo SRT plano)
                font      = cfg["subtitles"]["font"]
                font_size = cfg["subtitles"]["font_size"]
                outline_w = cfg["subtitles"]["outline_width"]
                layout    = cfg.get("layout_preset", "tiktok")
                margin_v  = cfg["subtitles"].get("margin_v")
                if margin_v is None:
                    _preset_margins = {"tiktok": 220, "shorts": 200, "reels": 210}
                    margin_v = _preset_margins.get(layout, 220)
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
                title_text: Optional[str] = None,
                subtitle_emojis: Optional[dict] = None) -> Path:
        """
        Processa um clip bruto e retorna o arquivo final editado.

        Args:
            input_path:      arquivo raw (.ts/.mp4)
            output_name:     nome do output sem extensão
            title_text:      texto do banner de título (opcional)
            subtitle_emojis: dict palavra_lower → emoji unicode, vindo da LLM
                             (hook_scorer). Cada palavra do transcript que bater
                             com uma key recebe o emoji adjacente nas legendas.
                             None ou {} = sem emojis injetados.
        """
        output_path = self.out_dir / f"{output_name}.mp4"
        logger.info(f"✂️  Editando: {input_path.name} → {output_path.name}")
        _t = _Timer("editor.process_total")
        _t.__enter__()
        try:
            return self._process_inner(
                input_path, output_path, title_text,
                subtitle_emojis=subtitle_emojis or {},
            )
        finally:
            _t.__exit__(None, None, None)

    def _process_inner(self, input_path: Path, output_path: Path,
                       title_text: Optional[str] = None,
                       subtitle_emojis: Optional[dict] = None) -> Path:

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
                        matched = _blacklist_hits(full_text, blacklist)
                        if matched and cf.get("action", "skip") == "skip":
                            _metrics_incr("editor.skipped", reason="content_filter")
                            logger.warning(
                                f"\U0001f6ab Conteúdo filtrado (blacklist): {matched[:3]} \u2014 clip SKIP"
                            )
                            return None

                    # Iter2 PR2: ASS karaoke (com word_timestamps do Whisper).
                    # Fallback SRT é mantido como variável de mesmo nome (srt_path)
                    # para compat com _build_ffmpeg_filter que decide o filter.
                    srt_path = tmp / "subtitles.ass"
                    self._segments_to_ass(
                        segments, srt_path,
                        subtitle_emojis=subtitle_emojis or {},
                    )
                    logger.info(
                        f"📝 Legendas: {len(segments)} segments → ASS "
                        f"(language={cfg_sub.get('language','pt')})"
                    )
                else:
                    # Segments vazio = Whisper desistiu (áudio ruim / VAD agressivo /
                    # filtro hallucination dropou tudo / idioma errado). Clip vai
                    # renderizar SEM legendas. Avisa no log.
                    logger.warning(
                        f"⚠️  Transcrição vazia — clip SEM legendas "
                        f"(language={cfg_sub.get('language','pt')}). "
                        f"Causas: áudio com música/jogos, idioma errado "
                        f"(tente 'auto'), ou clip silencioso."
                    )

            # --- Speaker tracking (Iter2): detecta rosto e propõe crop dinâmico ---
            # Custo: ~3-12s por clip (extract frames sparse + Haar Cascade CPU).
            # Se hit_ratio < 30% → None → fallback automático para pad+blur.
            face_track = None
            if self.cfg.get("output_format") == "vertical":
                try:
                    from face_tracker import compute_trajectory
                    # Estima duração do current via filesize (heurística leve)
                    # ou simplesmente passa 30s default (face_tracker não usa de fato)
                    with _Timer("editor.face_tracking"):
                        face_track = compute_trajectory(current, duration_sec=30.0)
                    if face_track is not None:
                        _metrics_incr("editor.face_track_hit",
                                      hit_ratio_band=f"{int(face_track.hit_ratio*10)*10}")
                    else:
                        _metrics_incr("editor.face_track_miss")
                except Exception as e:
                    logger.debug(f"face tracking falhou (fallback): {e}")
                    _metrics_incr("editor.face_track_error")

            # --- Monta filtro ---
            flt = self._build_ffmpeg_filter(srt_path, title_text=title_text,
                                            face_track=face_track)
            cmd = ["ffmpeg", "-y", "-i", str(current)]

            if flt["use_complex"]:
                # Modo vertical: fundo blur + vídeo sobreposto
                steps = flt["complex_steps"].copy()   # split, bg, fg, overlay

                # Pipeline: overlay → title_banner (lista de steps) → subtitles
                # title_filter agora é list[str] (steps já rotulados in/out)
                title_steps = flt["title_filter"] or []
                has_title   = bool(title_steps)
                has_sub     = bool(flt["sub_filter"])

                if has_title and has_sub:
                    # overlay→[pre_title]→title_steps→[pre_sub]→subs→[vout]
                    steps[-1] = steps[-1].replace("[vout]", "[pre_title]")
                    steps.extend(title_steps)   # já termina em [pre_sub]
                    steps.append(f"[pre_sub]{flt['sub_filter']}[vout]")
                elif has_title:
                    # title only: precisa fechar em [vout], então rebatiza o último step
                    steps[-1] = steps[-1].replace("[vout]", "[pre_title]")
                    # Substitui [pre_sub] → [vout] no último step do banner
                    title_steps = list(title_steps)
                    title_steps[-1] = title_steps[-1].replace("[pre_sub]", "[vout]")
                    steps.extend(title_steps)
                elif has_sub:
                    steps[-1] = steps[-1].replace("[vout]", "[pre_sub]")
                    steps.append(f"[pre_sub]{flt['sub_filter']}[vout]")

                filter_complex = ";".join(steps)
                cmd += ["-filter_complex", filter_complex, "-map", "[vout]", "-map", "0:a?"]
            else:
                # Modo horizontal simples — modo blob (movie=) só faz sentido em
                # complex graph. Para vf simples, _build_title_filter precisa
                # retornar string única. Solução: usar fallback drawbox.
                vf_parts = []
                if flt["simple_vf"]:
                    vf_parts.append(flt["simple_vf"])
                if flt["title_filter"]:
                    # title_filter é lista; extrai chain "interno" do fallback drawbox.
                    # Steps fallback têm formato "[in]chain[out]" — pega só o chain.
                    for s in flt["title_filter"]:
                        # Remove labels [in]...[out] do step para uso em -vf
                        import re as _re
                        m = _re.match(r"^\[[^\]]+\](.*?)\[[^\]]+\]$", s)
                        if m:
                            vf_parts.append(m.group(1))
                        else:
                            vf_parts.append(s)
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
            # Render num temporário: em timeout, encoder morrendo ou disco cheio
            # o ffmpeg deixa um mp4 sem moov, e escrito direto na saída ele
            # aparecia na aba de clips e podia ir para o upload manual. O
            # caminho final só passa a existir depois de um render com sucesso
            # (e o TemporaryDirectory apaga o parcial nos caminhos de erro).
            render_out = tmp / "render.mp4"
            cmd += video_args + audio_args + mux_args + [str(render_out)]

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
                                + audio_args + mux_args + [str(render_out)]
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

            # shutil.move: o temporário pode estar em outro disco que a saída.
            shutil.move(str(render_out), str(output_path))

        _metrics_incr("editor.clips_generated", encoder=encoder_tag)
        logger.info(f"✅ Clip editado: {output_path}")
        return output_path
