"""
app.py — ClipBot v2.0 Desktop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Janela nativa via pywebview.
Zero Flask · Zero sockets · Zero portas abertas.

Arquitetura:
  Python  ←→  window.pywebview.api.*  ←→  JavaScript (ui/index.html)
  ClipBotAPI recebe chamadas do JS e invoca bot_engine callbacks.

Uso:
    pip install pywebview
    python app.py
"""

import sys
import logging
from pathlib import Path

# ── Logging minimalista (só warnings+) em produção ──────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
# ClipBot engine pode logar em DEBUG; GUI vai mostrar via poll
logging.getLogger("ClipBot").setLevel(logging.DEBUG)

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from clipbot_api import ClipBotAPI


def main():
    try:
        import webview
    except ImportError:
        print(
            "\n╔══════════════════════════════════════════════╗\n"
            "║  ClipBot: pywebview não está instalado!      ║\n"
            "║                                              ║\n"
            "║  Instale com:  pip install pywebview         ║\n"
            "╚══════════════════════════════════════════════╝\n"
        )
        sys.exit(1)

    api = ClipBotAPI()

    window = webview.create_window(
        title            = "ClipBot v2.0",
        url              = str(BASE / "ui" / "index.html"),
        js_api           = api,
        width            = 1280,
        height           = 820,
        min_size         = (900, 620),
        resizable        = True,
        background_color = "#09090b",
        # Oculta barra de menu nativa (parece app nativo)
        text_select      = False,
    )

    # ── Eventos do ciclo de vida ─────────────────────────────────────────────

    def on_loaded():
        """Chamado quando o HTML terminou de carregar."""
        import json
        from credentials import load_settings
        api._set_window(window)
        cfg = load_settings()
        # Injeta config inicial no JS sem precisar de fetch
        window.evaluate_js(f"window._initConfig({json.dumps(cfg)})")

    def on_closing():
        """Shutdown limpo: para engine antes de destruir janela."""
        api._stop_poll()
        if api._engine and api._engine.is_running:
            api._engine.stop()

    window.events.loaded  += on_loaded
    window.events.closing += on_closing

    # gui=True necessário no Windows para abrir na main thread
    webview.start(gui="edgechromium" if sys.platform == "win32" else None, debug=False)


if __name__ == "__main__":
    main()
