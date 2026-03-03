"""
upload/uploader.py  — ClipBot v2.4
Faz upload automático para TikTok, Instagram Reels e YouTube Shorts.

Mudanças v2.4 (teardown Playwright):
  • _upload_playwright: usa sync_playwright().start() em vez do context manager,
    guardando a referência `pw` para chamar pw.stop() explicitamente no finally.
  • Dois finally aninhados garantem: (1) browser.close(), (2) pw.stop().
  • _active_browsers + atexit._cleanup_playwright: mantidos como camada extra
    de segurança para crashes antes do finally.
"""

import logging
import time
import atexit
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger("ClipBot.Upload")


# ── Registry global de instâncias Playwright ─────────────────────────────────
# Segurança extra: se o processo crashar antes do finally, atexit ainda fecha.

_active_playwright: list = []   # lista de objetos Playwright (pw)
_active_browsers:   list = []   # lista de browsers abertos
_pw_lock = threading.Lock()


def _cleanup_playwright() -> None:
    """Fecha browsers e para instâncias Playwright. Chamado via atexit."""
    with _pw_lock:
        browsers_copy   = list(_active_browsers)
        playwright_copy = list(_active_playwright)

    for browser in browsers_copy:
        try:
            if browser.is_connected():
                browser.close()
        except Exception:
            pass

    for pw in playwright_copy:
        try:
            pw.stop()
        except Exception:
            pass


atexit.register(_cleanup_playwright)


# ──────────────────────────────────────────────────────────────────────────────
# TikTok
# ──────────────────────────────────────────────────────────────────────────────

class TikTokUploader:

    def __init__(self, config: dict):
        self.cfg = config["upload"]["tiktok"]

    def upload(self, video_path: Path, title: str = None, tags: list = None) -> bool:
        caption      = (title or self.cfg.get("default_caption", ""))[:150]
        hashtags     = list(dict.fromkeys(tags or self.cfg.get("hashtags", [])))
        hashtags_str = " ".join(f"#{t.strip('#')}" for t in hashtags[:5])
        full_caption = f"{caption}\n\n{hashtags_str}".strip()
        session_id   = self.cfg["session_id"].strip()

        logger.info(f"📤 TikTok: enviando {video_path.name}...")
        try:
            return self._upload_playwright(video_path, full_caption, session_id)
        except Exception as e:
            logger.error(f"❌ TikTok upload falhou: {e}")
            return False

    def _upload_playwright(self, video_path: Path, caption: str, session_id: str) -> bool:
        """
        Gerencia o ciclo de vida completo do Playwright com dois finally aninhados:
          • finally interno:  browser.close()
          • finally externo:  pw.stop()
        Mesmo se browser.close() lançar, pw.stop() é garantido.
        """
        from playwright.sync_api import sync_playwright

        # Inicia Playwright de forma explícita (não usa context manager)
        # para podermos chamar pw.stop() no finally independentemente.
        pw = sync_playwright().start()

        with _pw_lock:
            _active_playwright.append(pw)

        browser = None
        try:
            browser = pw.chromium.launch(
                headless=False,   # TikTok detecta headless → bloqueia silenciosamente
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                    "--window-size=1280,800",
                ],
            )

            with _pw_lock:
                _active_browsers.append(browser)

            try:
                return self._run_upload_flow(pw, browser, video_path, caption, session_id)
            finally:
                # Finally interno: fecha o browser
                try:
                    browser.close()
                    logger.debug("🧹 TikTok: browser fechado.")
                except Exception:
                    pass
                with _pw_lock:
                    try:
                        _active_browsers.remove(browser)
                    except ValueError:
                        pass

        finally:
            # Finally externo: para o Playwright (chrome.exe desaparece do SO)
            try:
                pw.stop()
                logger.debug("🧹 TikTok: Playwright parado.")
            except Exception:
                pass
            with _pw_lock:
                try:
                    _active_playwright.remove(pw)
                except ValueError:
                    pass

    def _run_upload_flow(self, pw, browser, video_path: Path,
                         caption: str, session_id: str) -> bool:
        """Lógica de upload desacoplada do ciclo de vida do browser/playwright."""
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            viewport={"width": 1280, "height": 800},
        )

        ctx.add_cookies([
            {"name": "sessionid",    "value": session_id, "domain": ".tiktok.com", "path": "/"},
            {"name": "sessionid_ss", "value": session_id, "domain": ".tiktok.com", "path": "/"},
        ])

        page = ctx.new_page()

        logger.info("📤 TikTok: abrindo página de upload...")
        page.goto("https://www.tiktok.com/upload?lang=en",
                  wait_until="domcontentloaded", timeout=60000)
        time.sleep(3)

        # ── Helper: fecha modais ──────────────────────────────────────────────
        def dismiss_modals():
            for sel in [
                "button:has-text('Got it')",
                "button:has-text('OK')",
                "button:has-text('Close')",
                "button:has-text('Dismiss')",
                "button:has-text('Continue')",
                "button:has-text('Confirm')",
                "[aria-label='Close']",
                ".TUXModal button",
                "[data-e2e='modal-close-inner-button']",
            ]:
                try:
                    btn = page.locator(sel).first
                    if btn.is_visible(timeout=800):
                        btn.click(force=True)
                        time.sleep(0.4)
                except Exception:
                    pass

        dismiss_modals()

        if "login" in page.url.lower():
            logger.error("❌ TikTok: sessão expirada! Atualize o Session ID.")
            return False

        # ── Upload ────────────────────────────────────────────────────────────
        logger.info("📤 TikTok: fazendo upload do arquivo...")
        try:
            fi = page.locator("input[type='file']")
            fi.wait_for(state="attached", timeout=15000)
            fi.set_input_files(str(video_path))
        except Exception as e:
            logger.error(f"❌ TikTok: campo de upload não encontrado: {e}")
            return False

        logger.info("📤 TikTok: aguardando processamento...")
        try:
            page.wait_for_selector(
                ".upload-card--uploading, [class*='uploading'], [class*='loading']",
                timeout=15000,
            )
            page.wait_for_selector(
                ".upload-card--uploading, [class*='uploading'], [class*='loading']",
                state="hidden", timeout=180000,
            )
        except Exception:
            pass
        time.sleep(3)
        dismiss_modals()

        # ── Descrição ─────────────────────────────────────────────────────────
        logger.info("📤 TikTok: preenchendo descrição...")
        desc_filled = False
        for attempt in range(5):
            try:
                time.sleep(2)
                dismiss_modals()
                page.wait_for_selector("div[contenteditable='true']", timeout=10000)
                box = page.locator("div[contenteditable='true']").first
                box.scroll_into_view_if_needed()
                time.sleep(0.5)
                box.click(force=True)
                time.sleep(0.3)
                page.keyboard.press("Control+a")
                time.sleep(0.2)
                page.keyboard.press("Backspace")
                time.sleep(0.2)
                page.keyboard.type(caption[:150], delay=25)
                time.sleep(1)
                if box.inner_text().strip():
                    logger.info("✅ TikTok: descrição preenchida")
                    desc_filled = True
                    break
                logger.warning(f"⚠️ TikTok: campo vazio (tentativa {attempt + 1})")
            except Exception as e:
                logger.warning(f"⚠️ TikTok tentativa {attempt + 1}: {e}")

        if not desc_filled:
            logger.warning("⚠️ TikTok: postando sem descrição")

        dismiss_modals()
        time.sleep(1)

    # ── Post ──────────────────────────────────────────────────────────────
        logger.info("📤 TikTok: clicando em Post...")
        posted = False

        try:
            for sel in [
                "button[data-e2e='post-btn']",
                "button:has-text('Post')",
                "div[role='button']:has-text('Post')",
            ]:
                try:
                    btn = page.locator(sel).last
                    if btn.is_visible(timeout=3000):
                        # Primeiro: scroll_into_view (preferível)
                        try:
                            btn.scroll_into_view_if_needed()
                            time.sleep(0.5)
                        except Exception:
                            # Fallback: scroll global
                            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                            time.sleep(1)
                        btn.click(force=True)
                        posted = True
                        logger.info("✅ TikTok: botão Post clicado")
                        break
                except Exception:
                    continue

            if not posted:
                # Fallback JS: scroll + click
                page.evaluate("""
                    const b = document.querySelector('button[data-e2e="post-btn"]')
                           || [...document.querySelectorAll('button')]
                              .find(x => x.textContent.trim() === 'Post');
                    if (b) { b.scrollIntoView(); b.click(); }
                """)
                posted = True
        except Exception as e:
            logger.warning(f"⚠️ TikTok Post click: {e}")

        if not posted:
            logger.error("❌ TikTok: não conseguiu clicar no botão Post.")
            return False

        # ── Confirmação (URL OU elemento de sucesso) ──────────────────────
        logger.info("📤 TikTok: aguardando confirmação...")
        confirm_timeout = 60   # segundos
        published = False

        # Estratégia 1: aguardar redirecionamento de URL
        try:
            page.wait_for_url(
                lambda url: (
                    "upload" not in url.lower()
                    and ("tiktok.com" in url or "tiktokstudio" in url)
                ),
                timeout=confirm_timeout * 1000,
            )
            published = True
            logger.info(f"✅ TikTok: publicado (URL mudou)! {page.url}")
        except Exception:
            pass

        # Estratégia 2: procurar toasts/elementos de sucesso
        if not published:
            success_selectors = [
                ":text('Upload another')",
                ":text('View post')",
                ":text('Posted')",
                ":text('Uploaded')",
                ":text('Your video is being uploaded')",
                ":text('Your video is being processed')",
                "[class*='success']",
                "[class*='posted']",
                "[data-e2e*='success']",
            ]
            for sel in success_selectors:
                try:
                    el = page.locator(sel).first
                    if el.is_visible(timeout=3000):
                        published = True
                        logger.info(f"✅ TikTok: sucesso detectado ('{sel}')")
                        break
                except Exception:
                    continue

        # Estratégia 3: checar se saiu da página de upload
        if not published:
            current_url = page.url.lower()
            if "upload" not in current_url and "tiktok" in current_url:
                published = True
                logger.info(f"✅ TikTok: redirecionou para {page.url}")

        if published:
            logger.info("✅ TikTok: upload concluído!")
        else:
            logger.error("❌ TikTok: publicação NÃO confirmada. "
                         "Verifique manualmente se o vídeo foi postado.")

        return published


# ──────────────────────────────────────────────────────────────────────────────
# Instagram Reels
# ──────────────────────────────────────────────────────────────────────────────

class InstagramUploader:

    def __init__(self, config: dict):
        self.cfg     = config["upload"]["instagram"]
        self._client = None

    def _get_client(self):
        if self._client is None:
            from instagrapi import Client
            cl = Client()
            cl.login(self.cfg["username"], self.cfg["password"])
            self._client = cl
        return self._client

    def upload(self, video_path: Path, caption: str = None,
               cover_path: Path = None) -> bool:
        try:
            cl   = self._get_client()
            text = caption or self.cfg["default_caption"]
            logger.info(f"📤 Instagram: enviando {video_path.name}...")
            cl.clip_upload(
                path=str(video_path),
                caption=text,
                thumbnail=str(cover_path) if cover_path else None,
            )
            logger.info("✅ Instagram: upload concluído!")
            return True
        except ImportError:
            logger.error("instagrapi não instalado. Execute: pip install instagrapi")
            return False
        except Exception as e:
            logger.error(f"❌ Instagram upload falhou: {e}")
            return False


# ──────────────────────────────────────────────────────────────────────────────
# YouTube Shorts
# ──────────────────────────────────────────────────────────────────────────────

class YouTubeUploader:

    def __init__(self, config: dict):
        self.cfg      = config["upload"]["youtube"]
        self._service = None

    def _get_service(self):
        if self._service is not None:
            return self._service

        from googleapiclient.discovery import build
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        import pickle

        SCOPES    = ["https://www.googleapis.com/auth/youtube.upload"]
        cred_file = self.cfg.get("credentials_file", "")

        if not cred_file or not Path(cred_file).exists():
            fallback = Path.home() / "AppData" / "Roaming" / "ClipBot" / "youtube_credentials.json"
            if fallback.exists():
                cred_file = str(fallback)
            else:
                raise FileNotFoundError(
                    "youtube_credentials.json não encontrado!\n"
                    "Vá em Accounts → YouTube e selecione o arquivo do Google Cloud."
                )

        token_path = Path.home() / "AppData" / "Roaming" / "ClipBot" / "youtube_token.pickle"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        creds = None

        if token_path.exists():
            with open(token_path, "rb") as f:
                creds = pickle.load(f)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow  = InstalledAppFlow.from_client_secrets_file(cred_file, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(token_path, "wb") as f:
                pickle.dump(creds, f)

        self._service = build("youtube", "v3", credentials=creds)
        return self._service

    def upload(self, video_path: Path, title: str = None,
               description: str = None) -> bool:
        try:
            from googleapiclient.http import MediaFileUpload

            service   = self._get_service()
            vid_title = (title or self.cfg["default_title"])[:100]
            vid_desc  = description or self.cfg["default_description"]

            body = {
                "snippet": {
                    "title":       vid_title,
                    "description": vid_desc,
                    "categoryId":  self.cfg["category_id"],
                    "tags":        ["shorts", "clips", "live", "gaming"],
                },
                "status": {
                    "privacyStatus":           self.cfg["privacy"],
                    "madeForKids":             self.cfg["made_for_kids"],
                    "selfDeclaredMadeForKids": self.cfg["made_for_kids"],
                },
            }

            logger.info(f"📤 YouTube: enviando {video_path.name}...")
            media   = MediaFileUpload(
                str(video_path), mimetype="video/mp4",
                resumable=True, chunksize=10 * 1024 * 1024,
            )
            request  = service.videos().insert(
                part=",".join(body.keys()), body=body, media_body=media,
            )
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    logger.debug(f"  YouTube: {int(status.progress() * 100)}%")

            logger.info(f"✅ YouTube: https://youtube.com/shorts/{response.get('id')}")
            return True

        except ImportError:
            logger.error("google-api-python-client não instalado.")
            return False
        except Exception as e:
            logger.error(f"❌ YouTube upload falhou: {e}")
            return False


# ──────────────────────────────────────────────────────────────────────────────
# Coordenador de uploads
# ──────────────────────────────────────────────────────────────────────────────

class MultiPlatformUploader:

    def __init__(self, config: dict):
        self.config = config
        self.up_cfg = config["upload"]

        self.tiktok    = TikTokUploader(config)    if self.up_cfg["tiktok"]["enabled"]    else None
        self.instagram = InstagramUploader(config) if self.up_cfg["instagram"]["enabled"] else None
        self.youtube   = YouTubeUploader(config)   if self.up_cfg["youtube"]["enabled"]   else None

        self._daily_counts = {"tiktok": 0, "instagram": 0, "youtube": 0}
        self._last_date    = datetime.now().date()

    def _reset_daily_counts(self) -> None:
        today = datetime.now().date()
        if today != self._last_date:
            self._daily_counts = {k: 0 for k in self._daily_counts}
            self._last_date    = today

    def upload_all(self, video_path: Path, title: str = None,
                   description: str = None, tags: list = None) -> dict:
        self._reset_daily_counts()
        max_per_day = self.up_cfg["max_clips_per_day"]
        delay       = self.up_cfg["delay_between_posts"]
        results     = {}

        platforms = [
            ("tiktok",    self.tiktok,    lambda: self.tiktok.upload(video_path, title, tags)),
            ("instagram", self.instagram, lambda: self.instagram.upload(video_path, title)),
            ("youtube",   self.youtube,   lambda: self.youtube.upload(video_path, title, description)),
        ]

        for name, upl, fn in platforms:
            if upl is None:
                continue
            if self._daily_counts[name] >= max_per_day:
                logger.warning(f"⚠️  {name}: limite diário ({max_per_day}) atingido.")
                results[name] = False
                continue
            success = fn()
            results[name] = success
            if success:
                self._daily_counts[name] += 1
            if delay > 0:
                logger.info(f"⏳ Aguardando {delay}s...")
                time.sleep(delay)

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Notificações
# ──────────────────────────────────────────────────────────────────────────────

class Notifier:

    def __init__(self, config: dict):
        self.cfg = config.get("notifications", {})

    def send(self, message: str) -> None:
        self._discord(message)
        self._telegram(message)

    def _discord(self, message: str) -> None:
        cfg = self.cfg.get("discord", {})
        if not cfg.get("enabled"):
            return
        try:
            import requests
            requests.post(cfg["webhook_url"], json={"content": message}, timeout=10)
        except Exception as e:
            logger.debug(f"Discord notif: {e}")

    def _telegram(self, message: str) -> None:
        cfg = self.cfg.get("telegram", {})
        if not cfg.get("enabled"):
            return
        try:
            import requests
            url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
            requests.post(url, json={"chat_id": cfg["chat_id"], "text": message}, timeout=10)
        except Exception as e:
            logger.debug(f"Telegram notif: {e}")
