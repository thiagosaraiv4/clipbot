"""
Item 2 da recuperação — I/O silencioso.

A sessão de produção de 12/08 deixou dois rastros do MESMO defeito de classe:

  1. O diagnóstico de boot reportou "credentials.enc: file_exists=False" para um
     arquivo que existia e era legível (no Windows, Path.exists() devolve False
     quando a abertura falha por permissão — AV/indexador segurando o handle).
     Resultado: a cascata de LLM ficou desligada a sessão inteira sem UM erro.

  2. Três momentos pendentes (128 MB de .ts) não apareciam no pending.json.
     Não é possível saber, pelos logs, se a escrita falhou ou nunca aconteceu:
     o único caminho de falha é um logger.warning sem rastro durável.

A raiz comum: o código não distingue "não existe" de "não consegui ler/escrever".
Ambos viram silêncio — e o silêncio vira PERDA DE DADOS no próximo save:

    clipbot_api.set_llm_keys:   creds = load_credentials() or {}   # {} em falha
                                creds[k] = v
                                save_credentials(creds)
                                → grava só a key nova, apagando as demais.

    UI "salvar configurações":  load_settings() devolve os defaults em falha
                                → save_config() sobrescreve canais e tokens reais.

Esta suíte fixa o contrato: ausente é normal, ilegível é ERRO, e um estado
degradado BLOQUEIA escrita até a leitura voltar a funcionar.
"""

import json

import pytest


# ─────────────────────────────── helpers ────────────────────────────────────

def _break_reads(monkeypatch, exc=PermissionError("acesso negado (AV)")):
    """
    Toda leitura de texto falha — como um antivírus segurando o arquivo.

    Devolve heal(): libera as leituras SEM mexer nos outros patches do teste.
    (monkeypatch.undo() serviria, mas desfaria também o redirecionamento de
    CRED_FILE/CONFIG_FILE para tmp_path, mandando o teste no estado real.)
    """
    from pathlib import Path
    orig = Path.read_text
    broken = {"on": True}

    def maybe(self, *a, **kw):
        if broken["on"]:
            raise exc
        return orig(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", maybe)
    return lambda: broken.update(on=False)


def _break_replace(monkeypatch, exc=PermissionError("arquivo em uso")):
    """O rename atômico falha sempre."""
    import safe_io

    def boom(src, dst):
        raise exc

    monkeypatch.setattr(safe_io, "_replace", boom)


@pytest.fixture
def creds_mod(tmp_path, monkeypatch):
    """credentials.py com os arquivos redirecionados para tmp_path."""
    import credentials as c
    monkeypatch.setattr(c, "APP_DATA", tmp_path)
    monkeypatch.setattr(c, "CRED_FILE", tmp_path / "credentials.enc")
    monkeypatch.setattr(c, "CONFIG_FILE", tmp_path / "settings.json")
    c.reset_storage_health()
    yield c
    c.reset_storage_health()


# ───────────────────────── safe_io: escrita atômica ─────────────────────────

def test_atomic_write_survives_transient_failure(tmp_path, monkeypatch):
    """
    O rename falha 2x e funciona na 3a (AV/indexador soltando o handle).
    Uma falha transitória NÃO pode custar a escrita.
    """
    import safe_io
    target = tmp_path / "x.json"
    calls = {"n": 0}
    real = safe_io._replace

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("arquivo em uso")
        return real(src, dst)

    monkeypatch.setattr(safe_io, "_replace", flaky)
    safe_io.atomic_write_text(target, '{"ok":1}', retries=4, backoff=0.0)

    assert target.read_text(encoding="utf-8") == '{"ok":1}'
    assert calls["n"] == 3, "não houve retry"


def test_atomic_write_raises_after_retries(tmp_path, monkeypatch):
    """Esgotadas as tentativas, o erro PRECISA chegar ao chamador."""
    import safe_io
    _break_replace(monkeypatch)

    with pytest.raises(OSError):
        safe_io.atomic_write_text(tmp_path / "x.json", "dados", retries=2, backoff=0.0)


def test_atomic_write_leaves_no_temp_garbage(tmp_path, monkeypatch):
    import safe_io
    _break_replace(monkeypatch)

    with pytest.raises(OSError):
        safe_io.atomic_write_text(tmp_path / "x.json", "dados", retries=1, backoff=0.0)

    assert list(tmp_path.glob("*.tmp")) == [], "sobrou arquivo .tmp"


def test_atomic_write_preserves_previous_content_on_failure(tmp_path, monkeypatch):
    """INVARIANTE: escrita que falha preserva o conteúdo bom anterior."""
    import safe_io
    target = tmp_path / "x.json"
    target.write_text('{"bom":true}', encoding="utf-8")
    _break_replace(monkeypatch)

    with pytest.raises(OSError):
        safe_io.atomic_write_text(target, '{"novo":true}', retries=1, backoff=0.0)

    assert json.loads(target.read_text(encoding="utf-8")) == {"bom": True}


def test_read_classified_distinguishes_the_three_states(tmp_path, monkeypatch):
    """O coração do item: ausente, legível e ILEGÍVEL são estados distintos."""
    import safe_io

    assert safe_io.read_text_classified(tmp_path / "nao_existe")[0] == safe_io.MISSING

    ok = tmp_path / "ok.json"
    ok.write_text("conteudo", encoding="utf-8")
    assert safe_io.read_text_classified(ok)[0] == safe_io.OK

    _break_reads(monkeypatch)
    status, _, err = safe_io.read_text_classified(ok)
    assert status == safe_io.UNREADABLE, (
        "arquivo ilegível classificado como ausente — é exatamente a raiz do bug"
    )
    assert err, "erro precisa ser propagado para o log"


def test_read_classified_does_not_trust_exists(tmp_path, monkeypatch):
    """
    No Windows, Path.exists() devolve False em PermissionError. Se a checagem
    de existência for a primeira porta, 'ilegível' vira 'ausente' outra vez.
    """
    import safe_io
    from pathlib import Path
    p = tmp_path / "credentials.enc"
    p.write_text("x", encoding="utf-8")

    monkeypatch.setattr(Path, "exists", lambda self: False)
    _break_reads(monkeypatch)

    assert safe_io.read_text_classified(p)[0] == safe_io.UNREADABLE


# ────────────────────── credentials: caminho de perda de dados ──────────────

def test_missing_credentials_is_normal_and_healthy(creds_mod):
    """Baseline: primeira execução não é erro."""
    assert creds_mod.load_credentials() == {}
    assert creds_mod.storage_degraded("credentials") is False


def test_unreadable_credentials_marks_degraded(creds_mod, monkeypatch, caplog):
    """Ilegível != ausente: precisa marcar degradado e gritar no log."""
    creds_mod.save_credentials({"tiktok_session_id": "abc", "anthropic_api_key": "k"})
    _break_reads(monkeypatch)

    with caplog.at_level("ERROR"):
        assert creds_mod.load_credentials() == {}

    assert creds_mod.storage_degraded("credentials") is True
    assert any(r.levelname == "ERROR" for r in caplog.records), "falha ficou silenciosa"


def test_save_credentials_refuses_while_degraded(creds_mod, monkeypatch):
    """
    P0 — o cenário de 12/08 exatamente: load falhou -> {} -> salvar uma key nova
    apagaria tiktok/instagram/youtube. O save tem de RECUSAR.
    """
    creds_mod.save_credentials({"tiktok_session_id": "abc", "anthropic_api_key": "k"})
    raw_antes = creds_mod.CRED_FILE.read_bytes()

    heal = _break_reads(monkeypatch)
    creds = creds_mod.load_credentials() or {}        # {} degradado
    creds["google_api_key"] = "nova"

    with pytest.raises(creds_mod.StorageDegraded):
        creds_mod.save_credentials(creds)

    heal()
    assert creds_mod.CRED_FILE.read_bytes() == raw_antes, "arquivo bom foi sobrescrito"
    assert set(creds_mod.load_credentials()) == {"tiktok_session_id", "anthropic_api_key"}


def test_corrupt_credentials_do_not_look_absent(creds_mod):
    """Conteúdo corrompido também é anomalia — não pode virar {} silencioso."""
    creds_mod.CRED_FILE.write_text("isto-nao-decodifica!!", encoding="utf-8")

    assert creds_mod.load_credentials() == {}
    assert creds_mod.storage_degraded("credentials") is True


def test_unreadable_settings_marks_degraded_but_app_still_boots(creds_mod, monkeypatch):
    """Degradar não pode impedir o app de subir — só de escrever."""
    creds_mod.save_settings({"platforms": {"twitch": {"enabled": True, "channels": ["x"]}}})
    _break_reads(monkeypatch)

    cfg = creds_mod.load_settings()

    assert "platforms" in cfg, "precisa devolver algo utilizável"
    assert creds_mod.storage_degraded("settings") is True


def test_save_settings_refuses_while_degraded(creds_mod, monkeypatch):
    """Impede que a UI sobrescreva a config real com os defaults."""
    real = {"platforms": {"twitch": {"enabled": True, "channels": ["baiano", "kosky"]}}}
    creds_mod.save_settings(real)

    heal = _break_reads(monkeypatch)
    creds_mod.load_settings()                         # degrada
    with pytest.raises(creds_mod.StorageDegraded):
        creds_mod.save_settings({"platforms": {}})

    heal()
    assert creds_mod.load_settings() == real


def test_good_read_clears_degraded_state(creds_mod, monkeypatch):
    """Passado o bloqueio, uma leitura boa restaura o estado saudável."""
    creds_mod.save_credentials({"k": "v"})
    heal = _break_reads(monkeypatch)
    creds_mod.load_credentials()
    assert creds_mod.storage_degraded("credentials") is True

    heal()
    assert creds_mod.load_credentials() == {"k": "v"}
    assert creds_mod.storage_degraded("credentials") is False


def test_settings_write_failure_is_not_swallowed(creds_mod, monkeypatch):
    """save_settings nunca pode retornar como sucesso sem ter gravado."""
    _break_replace(monkeypatch)

    with pytest.raises(OSError):
        creds_mod.save_settings({"platforms": {}})


# ───────────────────────────── pending store ────────────────────────────────

def test_pending_save_failure_is_loud_and_durable(store, raw_ts, monkeypatch, caplog):
    """
    12/08: 3 pendentes (128 MB) não chegaram ao disco e os logs não permitem
    distinguir 'falhou' de 'nunca tentou'. A falha precisa virar ERROR + flag
    consultável no store.
    """
    from conftest import make_moment
    _break_replace(monkeypatch)

    with caplog.at_level("ERROR"):
        store.add(make_moment(), raw_ts(), "twitch", "x")

    assert store.last_save_error is not None, "falha de persistência não sinalizada"
    assert any(r.levelname == "ERROR" for r in caplog.records), "falha ficou silenciosa"


def test_pending_save_failure_keeps_moment_in_memory(store, raw_ts, monkeypatch):
    """
    INVARIANTE: disco quebrado não pode fazer o momento desaparecer da sessão
    atual — a UI de aprovação ainda tem de poder gerar o clip.
    """
    from conftest import make_moment
    m = make_moment()
    _break_replace(monkeypatch)

    store.add(m, raw_ts(), "twitch", "x")

    assert store.get(m.moment_id) is not None


def test_pending_save_success_clears_error_flag(store, raw_ts):
    from conftest import make_moment
    store.add(make_moment(), raw_ts(), "twitch", "x")
    assert store.last_save_error is None


def test_pending_load_backs_up_corrupt_file(store, tmp_path):
    """pending.json corrompido não pode ser descartado sem deixar cópia."""
    import pending_store
    (tmp_path / "pending.json").write_text("{lixo nao json", encoding="utf-8")

    s2 = pending_store.PendingStore()

    assert len(s2.list_all()) == 0
    assert list(tmp_path.glob("pending.json.corrupt*")), "não guardou backup"


def test_pending_load_unreadable_is_logged_as_error(store, tmp_path, monkeypatch, caplog):
    """Store que não consegue LER o pending é anomalia, não 'primeira execução'."""
    import pending_store
    (tmp_path / "pending.json").write_text("[]", encoding="utf-8")
    _break_reads(monkeypatch)

    with caplog.at_level("ERROR"):
        pending_store.PendingStore()

    assert any(r.levelname == "ERROR" for r in caplog.records)


# ───────────────────────────── checagem no boot ─────────────────────────────

def test_check_writable_reports_ok(tmp_path):
    import safe_io
    ok, err = safe_io.check_writable(tmp_path)
    assert ok is True and err is None


def test_check_writable_reports_failure(tmp_path, monkeypatch):
    import safe_io
    _break_replace(monkeypatch)
    ok, err = safe_io.check_writable(tmp_path)
    assert ok is False and err


def test_check_writable_leaves_no_probe_file(tmp_path):
    import safe_io
    before = set(tmp_path.iterdir())
    safe_io.check_writable(tmp_path)
    assert set(tmp_path.iterdir()) == before, "sobrou arquivo de sonda"
