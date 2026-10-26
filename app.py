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
sys.path.insert(0, str(BASE / "core"))

# ── Job Object (Windows): mata todos os children se Python morrer ───────────
# DEVE rodar ANTES de qualquer subprocess.Popen ser criado. Por isso primeiro.
try:
    from job_object import setup_job_object
    setup_job_object()
except Exception:
    pass

from clipbot_api import ClipBotAPI

# ── Telemetria: bootstrap antes de qualquer engine work ──────────────────────
try:
    from metrics import start_reporter, stop_reporter
    from system_sampler import start_sampler, stop_sampler
    _HAS_METRICS = True
except ImportError:
    _HAS_METRICS = False


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

    # ── Disco de estado: descobrir AGORA que o APPDATA está bloqueado ──────
    # Em 12/08 o app rodou 6 min achando que estava persistindo momentos e
    # credenciais. Uma sonda de escrita no boot custa milissegundos e troca
    # "sumiu misteriosamente" por uma linha explícita no log.
    try:
        import safe_io
        from credentials import APP_DATA as _APP_DATA
        _ok, _err = safe_io.check_writable(_APP_DATA)
        if not _ok:
            logging.getLogger("ClipBot").error(
                f"❌ Pasta de estado NÃO é gravável: {_APP_DATA} ({_err}). "
                "Momentos pendentes, credenciais e configurações não serão "
                "salvos. Verifique antivírus/permissões antes de continuar."
            )
    except Exception as _e:
        logging.getLogger("ClipBot").warning(f"checagem de gravabilidade: {_e}")


    # ── Telemetria: reporter + sampler em background (custo <0.1% CPU) ──────
    if _HAS_METRICS:
        try:
            start_reporter(interval=60.0)
            from credentials import load_settings as _ls
            _cfg = _ls()
            _clips = Path(_cfg.get("system", {}).get("clips_folder", BASE / "clips"))
            start_sampler(clips_dir=_clips)
        except Exception:
            pass

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
        api._set_window(window)
        # config_for_ui registra se esta leitura falhou: é essa cópia que a UI
        # devolve inteira no "Salvar" (ver ClipBotAPI.save_config).
        cfg = api.config_for_ui()
        # Injeta config inicial no JS sem precisar de fetch
        window.evaluate_js(f"window._initConfig({json.dumps(cfg)})")

    def on_closing():
        """Shutdown limpo: para engine antes de destruir janela."""
        api._stop_poll()
        if api._engine and api._engine.is_running:
            api._engine.stop()
        if _HAS_METRICS:
            try:
                stop_sampler()
                stop_reporter()
            except Exception:
                pass

    window.events.loaded  += on_loaded
    window.events.closing += on_closing

    # gui=True necessário no Windows para abrir na main thread
    webview.start(gui="edgechromium" if sys.platform == "win32" else None, debug=False)


if __name__ == "__main__":
    main()
