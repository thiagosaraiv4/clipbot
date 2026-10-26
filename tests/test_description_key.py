"""
Lote 2 / B10 — o gerador de descrições ignorava a key salva pela UI.

A UI grava as keys de LLM em credentials.enc (clipbot_api.set_llm_keys), e o
hook_scorer as encontra pela cascata ENV → credentials.enc → settings.json.
O DescriptionGenerator, porém, criava anthropic.Anthropic() sem api_key — o
SDK só olha a variável de ambiente, que nada exporta. Resultado: com a key
configurada pela UI, títulos e scores funcionavam, mas TODA legenda de post
caía no _fallback genérico, e o motivo ficava num log DEBUG.

Contrato: a descrição usa a mesma cascata de keys do hook_scorer; sem key
nenhuma, vai direto ao fallback sem tentar a API.
"""

from types import SimpleNamespace

import pytest


@pytest.fixture
def creds(tmp_path, monkeypatch):
    import credentials as c
    monkeypatch.setattr(c, "APP_DATA", tmp_path)
    monkeypatch.setattr(c, "CRED_FILE", tmp_path / "credentials.enc")
    monkeypatch.setattr(c, "CONFIG_FILE", tmp_path / "settings.json")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c.reset_storage_health()
    yield c
    c.reset_storage_health()


@pytest.fixture
def built(monkeypatch):
    import anthropic
    seen = {}

    class FakeAnthropic:
        def __init__(self, **kw):
            seen.update(kw)
            seen["constructed"] = True
            self.messages = SimpleNamespace(create=lambda **k: SimpleNamespace(
                content=[SimpleNamespace(
                    text='{"title": "t", "description": "d", "hashtags": ["a"]}'
                )]
            ))

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    return seen


def _describe():
    from description_generator import DescriptionGenerator
    gen = DescriptionGenerator({"editing": {"subtitles": {"language": "pt"}}})
    return gen._generate_with_ai("transcrição", "tiktok", "kosky", "keyword")


def test_uses_the_key_saved_in_the_ui(creds, built):
    """O cenário do relatório: key só no credentials.enc."""
    creds.save_credentials({"anthropic_api_key": "sk-ant-da-ui"})

    result = _describe()

    assert built.get("api_key") == "sk-ant-da-ui", "key da UI não chegou ao cliente"
    assert result["title"] == "t"


def test_environment_key_still_wins(creds, built, monkeypatch):
    """Mesma ordem da cascata do hook_scorer: ENV primeiro."""
    creds.save_credentials({"anthropic_api_key": "sk-ant-da-ui"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-do-env")

    _describe()

    assert built.get("api_key") == "sk-ant-do-env"


def test_without_any_key_goes_straight_to_fallback(creds, built):
    """Sem key, nem tenta a API (antes: chamada inútil + erro de auth)."""
    assert _describe() == {}
    assert "constructed" not in built
