"""
G2 — descrição dos posts só sabia usar a Anthropic, e sem transcrição.

Decisão do usuário (28/10): fica só o Gemini. Mas o DescriptionGenerator
tinha um cliente próprio, só Anthropic — sem crédito ali, toda legenda e
hashtag de post caía no _fallback genérico, mesmo com o Gemini funcionando
para o título.

E a transcrição que alimenta a descrição usava `import whisper`
(openai-whisper), que não está instalado (nem o torch): falhava em silêncio
e a IA recebia "sem transcrição disponível" — legenda genérica mesmo com
um LLM disponível.

Contrato: a descrição usa a mesma cascata do título (provider, ordem, keys,
timeouts do hook_scorer — Gemini primeiro); sem key nenhuma vai direto ao
fallback (B10); a transcrição usa o WhisperService do app (mesmo modelo).
"""

from types import SimpleNamespace

import pytest

GEMINI_JSON = '{"title": "Gemini T", "description": "Gemini D", "hashtags": ["terraplana", "fyp"]}'


@pytest.fixture
def creds(tmp_path, monkeypatch):
    import credentials as c
    import hook_scorer
    monkeypatch.setattr(c, "APP_DATA", tmp_path)
    monkeypatch.setattr(c, "CRED_FILE", tmp_path / "credentials.enc")
    monkeypatch.setattr(c, "CONFIG_FILE", tmp_path / "settings.json")
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    c.reset_storage_health()
    hook_scorer.HookScorer.reset_cache()
    yield c
    hook_scorer.HookScorer.reset_cache()
    c.reset_storage_health()


@pytest.fixture
def sdks(monkeypatch):
    """Gemini e Anthropic falsos; registra quem foi construído/chamado."""
    import anthropic
    from google import genai
    seen = {"gemini_calls": 0, "anthropic_built": False}

    class FakeGenai:
        def __init__(self, **kw):
            seen["gemini_key"] = kw.get("api_key")

            def gen(**k):
                seen["gemini_calls"] += 1
                seen["gemini_prompt"] = k.get("contents")
                return SimpleNamespace(text=GEMINI_JSON)
            self.models = SimpleNamespace(generate_content=gen)

    class FakeAnthropic:
        def __init__(self, **kw):
            seen["anthropic_built"] = True
            self.messages = SimpleNamespace(create=lambda **k: SimpleNamespace(
                content=[SimpleNamespace(text='{"title": "Claude T", "description": "d", "hashtags": []}')]))

    monkeypatch.setattr(genai, "Client", FakeGenai)
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    return seen


def _gen():
    from description_generator import DescriptionGenerator
    return DescriptionGenerator({"editing": {"subtitles": {"language": "pt"}}})


def test_only_google_key_describes_with_gemini(creds, sdks):
    creds.save_credentials({"google_api_key": "chave-google-teste"})
    r = _gen()._generate_with_ai("quer ser chamado de esquizo por defender terra plana",
                                 "tiktok", "kosky", "keyword")
    assert r.get("title") == "Gemini T", "só com a key do Google a legenda caía no genérico"
    assert sdks["gemini_key"] == "chave-google-teste"
    assert "terra plana" in sdks["gemini_prompt"]
    assert not sdks["anthropic_built"]


def test_gemini_first_when_both_keys_exist(creds, sdks):
    creds.save_credentials({"google_api_key": "chave-google-teste",
                            "anthropic_api_key": "sk-ant-sem-credito"})
    r = _gen()._generate_with_ai("fala", "instagram", "kosky", "keyword")
    assert r.get("title") == "Gemini T"
    assert not sdks["anthropic_built"], "tentou a Anthropic (sem crédito) antes do Gemini"


def test_without_any_key_goes_straight_to_fallback(creds, sdks):
    assert _gen()._generate_with_ai("fala", "tiktok", "kosky", "keyword") == {}
    assert sdks["gemini_calls"] == 0 and not sdks["anthropic_built"]


def test_full_caption_with_gemini(creds, sdks, tmp_path, monkeypatch):
    creds.save_credentials({"google_api_key": "chave-google-teste"})
    gen = _gen()
    monkeypatch.setattr(gen, "_transcribe_video", lambda p: "terra plana")
    r = gen.generate(tmp_path / "clip.mp4", "tiktok", "kosky", "keyword")
    assert r["title"] == "Gemini T"
    assert r["caption"] == "Gemini D\n\n#terraplana #fyp"


def test_transcription_uses_the_app_whisper(tmp_path, monkeypatch):
    """Antes: `import whisper` (não instalado) → "" sempre."""
    from whisper_service import WhisperService
    monkeypatch.setattr(WhisperService, "_instances", {})
    used = []

    def fake_transcribe(self, audio_path, **kw):
        used.append((self.size, kw.get("language")))
        return {"text": " quer ser chamado de esquizo ", "segments": [], "language": "pt"}

    monkeypatch.setattr(WhisperService, "transcribe", fake_transcribe)
    assert _gen()._transcribe_video(tmp_path / "clip.mp4") == "quer ser chamado de esquizo"
    assert used == [("medium", "pt")], "fora do singleton do app (legenda/título já usam medium)"
