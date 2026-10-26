"""
Lote 1 / A3 — o guard de escrita olhava a leitura errada.

O item 2 bloqueava a escrita de settings/credentials enquanto "a última
leitura" estivesse degradada. Mas "a última leitura" é de QUALQUER thread:

  settings
    1. on_loaded lê settings.json durante um lock de AV → a UI recebe os
       DEFAULTS via _initConfig (é a única vez que ela recebe a config).
    2. Uma leitura em segundo plano funciona (get_clips, start_engine,
       hook_scorer a cada score) → o flag global volta a "saudável".
    3. O usuário mexe num slider e clica Salvar → a UI manda os defaults
       inteiros → canais e tokens reais substituídos.
    O inverso também acontecia: uma leitura de fundo degradada bloqueava o
    Salvar de uma UI que tinha a config real.

  credentials
    set_llm_keys fazia load → check do flag → save. Entre o load degradado
    ({}) e o check, outra thread podia marcar "saudável", e o {} ia ao disco
    com só a key nova — apagando tiktok/instagram/youtube.

Contrato:
  • settings: o bloqueio é da CÓPIA que a UI tem. Quem decide é a leitura que
    a gerou, não leituras posteriores de outras threads.
  • credentials: read-modify-write atômico sob lock, decidido pela leitura
    feita ali dentro.
"""

import json
import threading

import pytest


REAL = {"platforms": {"twitch": {"enabled": True,
                                 "channels": ["https://www.twitch.tv/baiano",
                                              "https://www.twitch.tv/kosky"]}}}


def _break_reads(monkeypatch, exc=PermissionError("acesso negado (AV)")):
    from pathlib import Path
    orig = Path.read_text
    broken = {"on": True}

    def maybe(self, *a, **kw):
        if broken["on"]:
            raise exc
        return orig(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", maybe)
    return lambda: broken.update(on=False)


@pytest.fixture
def creds_mod(tmp_path, monkeypatch):
    import credentials as c
    monkeypatch.setattr(c, "APP_DATA", tmp_path)
    monkeypatch.setattr(c, "CRED_FILE", tmp_path / "credentials.enc")
    monkeypatch.setattr(c, "CONFIG_FILE", tmp_path / "settings.json")
    c.reset_storage_health()
    yield c
    c.reset_storage_health()


@pytest.fixture
def api(creds_mod):
    import clipbot_api
    a = clipbot_api.ClipBotAPI()
    a._engine = None
    return a


def _on_disk(creds_mod):
    return json.loads(creds_mod.CONFIG_FILE.read_text(encoding="utf-8"))


# ── settings: o cenário do relatório ────────────────────────────────────────

def test_ui_cannot_save_a_config_that_came_from_defaults(creds_mod, api, monkeypatch):
    creds_mod.save_settings(REAL)
    heal = _break_reads(monkeypatch)
    ui_cfg = api.config_for_ui()                     # on_loaded durante o lock
    heal()

    r = api.save_config(ui_cfg)

    assert r["ok"] is False and r["reason"] == "storage_degraded"
    assert _on_disk(creds_mod) == REAL


def test_background_good_read_does_not_unblock_ui_save(creds_mod, api, monkeypatch):
    """O cenário exato: get_clips/hook_scorer leem com sucesso depois."""
    creds_mod.save_settings(REAL)
    heal = _break_reads(monkeypatch)
    ui_cfg = api.config_for_ui()
    heal()

    creds_mod.load_settings()                        # leitura de fundo OK
    r = api.save_config(ui_cfg)

    assert r["ok"] is False, "leitura de outra thread liberou o save da UI"
    assert _on_disk(creds_mod) == REAL


def test_background_degraded_read_does_not_block_a_legit_ui_save(creds_mod, api, monkeypatch):
    """O inverso: UI com a config real não pode ser travada por leitura alheia."""
    creds_mod.save_settings(REAL)
    ui_cfg = api.config_for_ui()                     # leitura boa

    heal = _break_reads(monkeypatch)
    creds_mod.load_settings()                        # leitura de fundo degradada
    ui_cfg["platforms"]["twitch"]["channels"].append("https://www.twitch.tv/novo")
    r = api.save_config(ui_cfg)
    heal()

    assert r["ok"] is True, r
    assert "https://www.twitch.tv/novo" in _on_disk(creds_mod)["platforms"]["twitch"]["channels"]


def test_ui_reload_with_a_good_read_unblocks(creds_mod, api, monkeypatch):
    creds_mod.save_settings(REAL)
    heal = _break_reads(monkeypatch)
    api.config_for_ui()
    heal()

    ui_cfg = api.config_for_ui()                     # UI recarregada
    r = api.save_config(ui_cfg)

    assert r["ok"] is True


def test_get_config_records_provenance(creds_mod, api, monkeypatch):
    """O caminho exposto ao JS também registra de onde veio a cópia."""
    creds_mod.save_settings(REAL)
    heal = _break_reads(monkeypatch)
    ui_cfg = api.get_config()
    heal()

    assert api.save_config(ui_cfg)["ok"] is False


def test_refusal_explains_itself(creds_mod, api, monkeypatch):
    creds_mod.save_settings(REAL)
    heal = _break_reads(monkeypatch)
    ui_cfg = api.config_for_ui()
    heal()

    msg = api.save_config(ui_cfg)["error"]

    assert "settings.json" in msg and "padr" in msg.lower()


def test_first_run_without_settings_can_save(creds_mod, api):
    """Baseline: ausência não é degradação — primeira execução salva normal."""
    ui_cfg = api.config_for_ui()

    assert api.save_config(ui_cfg)["ok"] is True


# ── credentials: read-modify-write atômico ──────────────────────────────────

def test_update_refuses_unreadable_file_even_if_flag_says_healthy(creds_mod, monkeypatch):
    """
    A janela de corrida do relatório: outra thread marcou "saudável" entre o
    load degradado e o save. A decisão tem de vir da leitura feita AGORA.
    """
    creds_mod.save_credentials({"tiktok_session_id": "SID", "anthropic_api_key": "k"})
    antes = creds_mod.CRED_FILE.read_bytes()
    heal = _break_reads(monkeypatch)
    creds_mod.reset_storage_health()                 # "outra thread" curou o flag

    with pytest.raises(creds_mod.StorageDegraded):
        creds_mod.update_credentials({"google_api_key": "nova"})

    heal()
    assert creds_mod.CRED_FILE.read_bytes() == antes


def test_update_merges_with_what_is_on_disk(creds_mod):
    creds_mod.save_credentials({"tiktok_session_id": "SID", "anthropic_api_key": "k"})

    creds_mod.update_credentials({"google_api_key": "g"})
    creds_mod.update_credentials({"anthropic_api_key": "k2"})

    assert creds_mod.load_credentials() == {
        "tiktok_session_id": "SID", "anthropic_api_key": "k2", "google_api_key": "g",
    }


def test_concurrent_updates_do_not_lose_keys(creds_mod):
    """Sem lock, dois read-modify-write simultâneos perdem um dos updates."""
    errors = []

    def worker(prefix):
        for i in range(20):
            try:
                creds_mod.update_credentials({f"{prefix}_{i}": "v"})
            except Exception as ex:          # pragma: no cover — relatado no assert
                errors.append(ex)

    threads = [threading.Thread(target=worker, args=(p,)) for p in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(creds_mod.load_credentials()) == 40


def test_set_llm_keys_preserves_other_credentials(creds_mod, api):
    creds_mod.save_credentials({"tiktok_session_id": "SID", "instagram_user": "eu"})

    r = api.set_llm_keys({"google_api_key": "AIza-nova"})

    assert r["ok"] is True
    assert set(creds_mod.load_credentials()) == {
        "tiktok_session_id", "instagram_user", "google_api_key",
    }


def test_set_llm_keys_reports_degraded_storage(creds_mod, api, monkeypatch):
    creds_mod.save_credentials({"tiktok_session_id": "SID"})
    antes = creds_mod.CRED_FILE.read_bytes()
    heal = _break_reads(monkeypatch)
    creds_mod.reset_storage_health()

    r = api.set_llm_keys({"google_api_key": "AIza-nova"})

    heal()
    assert r["ok"] is False and r["reason"] == "storage_degraded"
    assert creds_mod.CRED_FILE.read_bytes() == antes
