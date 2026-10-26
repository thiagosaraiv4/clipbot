"""
Lote 2 / B5 — chamadas de LLM sem limite de tempo.

  • Gemini: genai.Client(api_key=key) sem http_options. No google-genai 2.x,
    sem timeout o SDK passa timeout=None ao httpx — nem connect nem read têm
    limite. Uma conexão que cai sem RST (troca de Wi-Fi, suspensão, NAT)
    deixa o recv bloqueado para sempre. É a última da cascata, e a única para
    quem só configurou a key gratuita do Gemini.
      modo automático → o canal para de processar momentos até reiniciar o app
      modo aprovação  → generate_clip_from_moment nunca retorna; momento preso
                        em "processing", spinner eterno, reject não apaga
  • Descrições: anthropic.Anthropic() com o padrão do SDK (read de 600 s,
    max_retries=2 → até ~30 min por descrição).

Anthropic e OpenAI no hook_scorer já usavam 8 s — o contrato aqui é que
nenhuma chamada de LLM do pipeline fique sem teto.
"""

from types import SimpleNamespace

import pytest


# ── Gemini (hook_scorer) ─────────────────────────────────────────────────────

@pytest.fixture
def gemini(monkeypatch):
    """Cliente falso que registra como foi construído."""
    import hook_scorer
    from google import genai
    built = {}

    class FakeClient:
        def __init__(self, **kw):
            built.update(kw)
            self.models = SimpleNamespace(
                generate_content=lambda **k: SimpleNamespace(text='{"ok": 1}')
            )

    monkeypatch.setattr(genai, "Client", FakeClient)
    monkeypatch.setattr(hook_scorer._GeminiProvider, "_client", None)
    monkeypatch.setattr(hook_scorer, "_load_key", lambda *a, **kw: "chave-teste")
    return built


def test_gemini_client_has_an_http_timeout(gemini):
    import hook_scorer

    assert hook_scorer._GeminiProvider.call("prompt") == '{"ok": 1}'

    opts = gemini.get("http_options")
    assert opts is not None, "cliente Gemini criado sem http_options — sem timeout"
    assert opts.timeout is not None, "http_options sem timeout"


def test_gemini_timeout_matches_the_other_providers(gemini):
    """google-genai mede o timeout em MILISSEGUNDOS; os outros usam segundos."""
    import hook_scorer

    hook_scorer._GeminiProvider.call("prompt")

    assert gemini["http_options"].timeout == int(hook_scorer._TIMEOUT_SEC * 1000)


def test_gemini_http_options_bound_connect_and_read():
    """
    Confere no SDK real que o valor vira limite de verdade no httpx — sem
    rede: só inspeciona o cliente que o SDK monta.
    """
    import hook_scorer
    from google import genai

    client = genai.Client(api_key="k", http_options=hook_scorer._gemini_http_options())
    timeout = client._api_client._http_options.timeout

    assert timeout is not None and timeout > 0


# ── Descrições (description_generator) ──────────────────────────────────────

@pytest.fixture
def anthropic_client(monkeypatch):
    import anthropic
    built = {}
    # Sem key nenhuma o gerador vai direto ao fallback (B10) e nem cria o
    # cliente — aqui o que se testa é o cliente, então há uma key no ENV.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-teste")

    class FakeAnthropic:
        def __init__(self, **kw):
            built.update(kw)
            self.messages = SimpleNamespace(create=lambda **k: SimpleNamespace(
                content=[SimpleNamespace(
                    text='{"title": "t", "description": "d", "hashtags": ["a"]}'
                )]
            ))

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    return built


def _describe():
    from description_generator import DescriptionGenerator
    gen = DescriptionGenerator({"editing": {"subtitles": {"language": "pt"}}})
    return gen._generate_with_ai("transcrição", "tiktok", "kosky", "keyword")


def test_description_client_has_a_timeout(anthropic_client):
    assert _describe()["title"] == "t"

    t = anthropic_client.get("timeout")
    assert t is not None, "cliente de descrição sem timeout (padrão do SDK: 600 s)"
    assert t <= 30


def test_description_client_has_bounded_retries(anthropic_client):
    """Padrão do SDK = 2 retentativas: 3 × o timeout no pior caso."""
    _describe()

    assert anthropic_client.get("max_retries") is not None
    assert anthropic_client["max_retries"] <= 1
