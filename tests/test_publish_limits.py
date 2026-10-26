"""
Lote 2b / B2 — o "Max clips/dia" da UI nunca era aplicado.

O único código que lia max_clips_per_day era MultiPlatformUploader.upload_all,
e nada em produção o chamava: o modo automático chamava cada uploader direto
(bot_engine) e o upload manual instanciava os uploaders na hora
(clipbot_api._do_upload). Com o slider em 10, o modo automático postava
quantos clips a live gerasse — exatamente o volume repetitivo que as regras
antispam do TikTok punem (e acima do teto de ~15 posts/dia por conta da API).
O contador de upload_all, além disso, vivia só em memória.

Contrato:
  • um teto diário POR plataforma, valendo nos dois caminhos que postam;
  • só post bem-sucedido consome vaga (falha devolve);
  • persiste em disco — reiniciar o app não zera o dia;
  • o valor da UI nunca passa do teto duro da plataforma;
  • sem conseguir ler o contador, não posta (risco de passar do teto).
"""

import json
import threading
from datetime import date

import pytest


@pytest.fixture
def limits_mod(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    import uploader
    return uploader


def _limits(mod, max_per_day=3, today=date(2026, 10, 26)):
    return mod.DailyPublishLimit(max_per_day, today=lambda: today)


# ── Unidade ──────────────────────────────────────────────────────────────────

def test_reserve_respects_the_daily_limit(limits_mod):
    lim = _limits(limits_mod, 2)

    assert [lim.try_reserve("tiktok") for _ in range(3)] == [True, True, False]


def test_limit_is_per_platform(limits_mod):
    lim = _limits(limits_mod, 1)

    assert lim.try_reserve("tiktok") is True
    assert lim.try_reserve("youtube") is True
    assert lim.try_reserve("tiktok") is False


def test_failed_upload_gives_the_slot_back(limits_mod):
    lim = _limits(limits_mod, 1)
    assert lim.try_reserve("tiktok")

    lim.release("tiktok")

    assert lim.try_reserve("tiktok") is True


def test_count_survives_an_app_restart(limits_mod):
    """Antes: contador em memória — reiniciar zerava o dia."""
    _limits(limits_mod, 2).try_reserve("tiktok")
    _limits(limits_mod, 2).try_reserve("tiktok")

    assert _limits(limits_mod, 2).try_reserve("tiktok") is False


def test_a_new_day_starts_from_zero(limits_mod):
    ontem = _limits(limits_mod, 1, today=date(2026, 10, 25))
    assert ontem.try_reserve("tiktok")
    assert ontem.try_reserve("tiktok") is False

    hoje = _limits(limits_mod, 1, today=date(2026, 10, 26))
    assert hoje.try_reserve("tiktok") is True


def test_platform_hard_cap_wins_over_the_ui_value(limits_mod):
    lim = _limits(limits_mod, 100)

    assert lim.limit_for("tiktok") == limits_mod.DailyPublishLimit.HARD_CAPS["tiktok"]
    assert lim.limit_for("tiktok") <= 15


def test_unreadable_counter_refuses_to_post(limits_mod, monkeypatch, tmp_path):
    from pathlib import Path
    lim = _limits(limits_mod, 5)
    lim.try_reserve("tiktok")                       # cria o arquivo
    orig = Path.read_text

    def boom(self, *a, **kw):
        if self.name == "publish_counts.json":
            raise PermissionError("AV")
        return orig(self, *a, **kw)
    monkeypatch.setattr(Path, "read_text", boom)

    assert lim.try_reserve("tiktok") is False


def test_concurrent_reserves_never_exceed_the_limit(limits_mod):
    lim = _limits(limits_mod, 3)
    results = []

    def worker():
        results.append(lim.try_reserve("tiktok"))

    ts = [threading.Thread(target=worker) for _ in range(12)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert results.count(True) == 3


# ── publish_with_limit: o ponto único usado pelos dois caminhos ─────────────

def test_publish_with_limit_counts_only_successes(limits_mod):
    lim = _limits(limits_mod, 1)

    assert limits_mod.publish_with_limit(lim, "tiktok", lambda: False) is False
    assert limits_mod.publish_with_limit(lim, "tiktok", lambda: True) is True
    assert limits_mod.publish_with_limit(lim, "tiktok", lambda: True) is None   # recusado


def test_publish_with_limit_gives_back_the_slot_on_exception(limits_mod):
    lim = _limits(limits_mod, 1)

    def boom():
        raise RuntimeError("rede caiu")
    with pytest.raises(RuntimeError):
        limits_mod.publish_with_limit(lim, "tiktok", boom)

    assert lim.try_reserve("tiktok") is True


# ── Integração: upload manual da UI ─────────────────────────────────────────

def test_manual_upload_respects_the_ui_limit(tmp_path, monkeypatch):
    """O caminho de produção que o teste da revisão apontou."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    import clipbot_api
    import credentials as c
    import uploader
    monkeypatch.setattr(c, "CONFIG_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(c, "CRED_FILE", tmp_path / "credentials.enc")
    c.reset_storage_health()
    settings = c._default_settings()
    settings["upload"]["max_per_day"] = 2
    c.save_settings(settings)

    posted = []

    class FakeTikTok:
        def __init__(self, config):
            pass

        def upload(self, video, *a, **kw):
            posted.append(video)
            return True

    monkeypatch.setattr(uploader, "TikTokUploader", FakeTikTok)
    api = clipbot_api.ClipBotAPI()
    clip = tmp_path / "clip_twitch_kosky_x_keyword.mp4"
    clip.write_bytes(b"\x00")

    for _ in range(3):
        api._do_upload(str(clip), ["tiktok"])

    assert len(posted) == 2, f"{len(posted)} posts com limite diário de 2"
    counts = json.loads((tmp_path / "ClipBot" / "publish_counts.json").read_text(encoding="utf-8"))
    assert counts["counts"]["tiktok"] == 2
