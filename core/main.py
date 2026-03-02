"""
main.py
ClipBot - Orquestrador principal.
Monitora lives, detecta momentos, edita e posta automaticamente.
"""

import os
import sys
import time
import logging
import argparse
import yaml
import colorlog
from pathlib import Path
from datetime import datetime

# Módulos locais
sys.path.insert(0, str(Path(__file__).parent))
from stream_capture            import StreamCapture
from detection.moment_detector import MomentDetector, Moment
from editing.video_editor      import VideoEditor
from upload.uploader           import MultiPlatformUploader, Notifier


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def setup_logging(level: str):
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(name)s] %(levelname)s%(reset)s  %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red",
        }
    ))
    logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger().addHandler(handler)
    # Silencia logs barulhentos
    for lib in ("urllib3", "httpx", "googleapiclient", "instagrapi"):
        logging.getLogger(lib).setLevel(logging.WARNING)


# ------------------------------------------------------------------
# ClipBot
# ------------------------------------------------------------------

class ClipBot:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        setup_logging(self.config["system"]["log_level"])
        self.logger = logging.getLogger("ClipBot")

        self.clips_dir  = Path(self.config["system"]["clips_folder"])
        self.output_dir = Path(self.config["system"]["output_folder"])

        self.capture  = StreamCapture(self.config, str(self.clips_dir))
        self.editor   = VideoEditor(self.config, str(self.output_dir))
        self.uploader = MultiPlatformUploader(self.config)
        self.notifier = Notifier(self.config)

        self._processed_moments: set = set()

    # ------------------------------------------------------------------
    # Processamento de um clip
    # ------------------------------------------------------------------

    def _process_moment(
        self,
        moment: Moment,
        raw_file: Path,
        platform: str
    ):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"clip_{platform}_{ts}_{moment.reason}"

        # 1. Recorta trecho bruto
        raw_clip = self.clips_dir / f"raw_{name}.mp4"
        try:
            self.capture.extract_segment(
                raw_file,
                moment.start_sec,
                moment.end_sec,
                raw_clip
            )
        except Exception as e:
            self.logger.error(f"Erro ao recortar: {e}")
            return

        # 2. Edita
        try:
            edited = self.editor.process(raw_clip, name)
        except Exception as e:
            self.logger.error(f"Erro ao editar: {e}")
            return
        finally:
            if raw_clip.exists():
                raw_clip.unlink()

        # 3. Upload
        title = self._build_title(moment, platform)
        results = self.uploader.upload_all(
            edited,
            title=title,
            description=self.config["upload"]["youtube"]["default_description"]
        )

        # 4. Notificação
        ok_platforms = [p for p, v in results.items() if v]
        msg = (
            f"🎬 Novo clip postado!\n"
            f"Motivo: {moment.reason}\n"
            f"Plataformas: {', '.join(ok_platforms) or 'nenhuma'}\n"
            f"Arquivo: {edited.name}"
        )
        self.notifier.send(msg)
        self.logger.info(msg)

        # 5. Limpeza
        if self.config["system"]["cleanup_after_upload"] and edited.exists():
            edited.unlink()
            self.logger.debug(f"🗑  {edited.name} removido.")

    def _build_title(self, moment: Moment, platform: str) -> str:
        emojis = {"audio_spike": "🔊", "chat_flood": "💬", "keyword": "🔥", "manual": "🎯"}
        emoji  = emojis.get(moment.reason, "🎬")
        if moment.keywords:
            kw = " ".join(moment.keywords[:3])
            return f"{emoji} {kw} | Clip da Live"
        return f"{emoji} Momento épico da Live! | Clip"

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------

    def run(self, platforms: list = None):
        active_platforms = platforms or self._get_active_platforms()

        if not active_platforms:
            self.logger.error("Nenhuma plataforma ativa no config.yaml!")
            return

        self.logger.info(f"🚀 ClipBot iniciado para: {active_platforms}")

        for platform in active_platforms:
            try:
                raw_file = self.capture.start_recording(platform)
            except Exception as e:
                self.logger.error(f"Não foi possível capturar {platform}: {e}")
                continue

            detector = MomentDetector(
                self.config,
                elapsed_fn=self.capture.get_elapsed_seconds
            )
            detector.start(raw_file)
            self.logger.info(f"✅ Monitorando {platform}. Ctrl+C para parar.\n")

            try:
                interval = self.config["system"]["check_interval"]
                while True:
                    time.sleep(interval)
                    moments = detector.all_moments

                    for m in moments:
                        key = (round(m.start_sec, 1), m.reason)
                        if key in self._processed_moments:
                            continue
                        # Espera acumular buffer depois do momento
                        if self.capture.get_elapsed_seconds() < m.end_sec + 2:
                            continue
                        self._processed_moments.add(key)
                        self.logger.info(f"🎯 Processando momento [{m.reason}] "
                                         f"{m.start_sec:.1f}s → {m.end_sec:.1f}s")
                        self._process_moment(m, raw_file, platform)

            except KeyboardInterrupt:
                self.logger.info("\n⏹ Encerrando ClipBot...")
            finally:
                detector.stop()
                self.capture.stop_recording()
                self.logger.info("👋 ClipBot encerrado.")

    def _get_active_platforms(self) -> list:
        return [
            p for p in ("twitch", "youtube", "kick", "custom_rtmp")
            if self.config["streams"].get(p, {}).get("enabled")
        ]

    # ------------------------------------------------------------------
    # Modo manual: processar arquivo local já gravado
    # ------------------------------------------------------------------

    def process_file(self, video_path: str, timestamps_file: str = None):
        """Processa um arquivo de vídeo local com timestamps manuais."""
        video = Path(video_path)
        if not video.exists():
            self.logger.error(f"Arquivo não encontrado: {video_path}")
            return

        if timestamps_file:
            self.config["detection"]["manual_timestamps"]["file"] = timestamps_file

        from detection.moment_detector import ManualTimestampDetector
        det = ManualTimestampDetector(self.config)
        det.load()

        moments = det.get_moments_snapshot()
        if not moments:
            self.logger.warning("Nenhum timestamp encontrado. Criando clip do vídeo inteiro.")
            moments = [Moment(start_sec=0, end_sec=999999, reason="manual")]

        for m in moments:
            self._process_moment(m, video, "local")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ClipBot - Clips automáticos de lives"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Caminho para o config.yaml (padrão: config.yaml)"
    )
    parser.add_argument(
        "--platforms", nargs="+",
        choices=["twitch", "youtube", "kick", "custom_rtmp"],
        help="Plataformas a monitorar (sobrescreve config)"
    )
    parser.add_argument(
        "--file",
        help="Processar arquivo de vídeo local ao invés de capturar live"
    )
    parser.add_argument(
        "--timestamps",
        help="Arquivo de timestamps manuais (usado com --file)"
    )
    args = parser.parse_args()

    bot = ClipBot(args.config)

    if args.file:
        bot.process_file(args.file, args.timestamps)
    else:
        bot.run(args.platforms)


if __name__ == "__main__":
    main()
