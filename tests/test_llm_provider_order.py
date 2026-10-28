"""
G1 — Gemini primeiro na cascata de LLM.

Decisão do usuário (28/10): sem como pôr crédito na Anthropic e na OpenAI,
fica só o Gemini. A ordem padrão era anthropic → openai → gemini; com as
duas primeiras sem crédito, cada clip pagava ~5 s e 2 erros antes de chegar
ao Gemini (validação de 26/10: 400 "credit balance is too low" e 429
"insufficient_quota" em todo clip). A ordem só é configurável pelo
settings.json, sem campo na UI, e a UI não apaga keys.

Contrato: sem override no settings, o Gemini é tentado primeiro; os outros
continuam como reserva (se um dia tiverem crédito); o override segue valendo.
"""

from types import SimpleNamespace

import pytest

GOOD = '{"title": "OLHA ISSO", "hook": "h", "viral_score": 0.7, "hashtags": [], "subtitle_emojis": {}}'


@pytest.fixture
def providers(monkeypatch):
    import hook_scorer
    calls = []
    out = {"anthropic": GOOD, "openai": GOOD, "gemini": GOOD}

    def fake(name):
        def call(prompt):
            calls.append(name)
            return out[name]
        return SimpleNamespace(name=name, is_available=lambda: True, call=call)

    monkeypatch.setattr(hook_scorer, "_PROVIDERS", {n: fake(n) for n in out})
    return calls, out


def _score():
    from hook_scorer import HookScorer
    return HookScorer(language="pt").score("olha isso mano", "keyword", channel="kosky")


def test_gemini_is_tried_first(providers):
    calls, _ = providers
    r = _score()
    assert r and r["provider"] == "gemini"
    assert calls == ["gemini"], f"chamou {calls} — sem crédito, cada um custa um erro"


def test_others_stay_as_fallback(providers):
    calls, out = providers
    out["gemini"] = None                # Gemini fora do ar / sem quota
    r = _score()
    assert calls[:2] == ["gemini", "anthropic"]
    assert r and r["provider"] == "anthropic"


def test_settings_override_still_wins(providers, monkeypatch):
    import credentials
    calls, _ = providers
    monkeypatch.setattr(credentials, "load_settings",
                        lambda: {"editing": {"hook_scorer": {"providers": ["openai", "gemini"]}}})
    _score()
    assert calls[0] == "openai"
