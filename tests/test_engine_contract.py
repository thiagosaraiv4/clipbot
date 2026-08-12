"""
Regressão P0-3 — contrato de retorno de generate_clip_from_moment().

Bug: 6 caminhos retornam dict, 1 retorna None (quando o editor descarta o clip).
O consumidor faz `{**result, "moment_id": ...}` → TypeError → a thread daemon
morre → onClipGenerated() nunca é chamado → UI trava em "gerando...".

Invariantes:
  I1 — retorna SEMPRE dict com chave "ok"
  I2 — onClipGenerated é chamado exatamente 1× por invocação, em qualquer desfecho
"""

import json
import pytest

from conftest import make_moment


class _EditorDiscards:
    """Editor que descarta o clip (comercial / content_filter) → retorna None."""
    def process(self, *a, **kw):
        return None


class _EditorOK:
    """Editor que gera arquivo real (para o caminho de sucesso)."""
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def process(self, raw_clip, name, **kw):
        p = self.out_dir / f"{name}.mp4"
        p.write_bytes(b"\x00" * 128)
        return p


@pytest.fixture
def engine(store, tmp_path, monkeypatch):
    """BotEngine com estado mínimo, sem threads, sem FFmpeg, sem LLM."""
    import bot_engine
    import clip_analyzer

    # Nunca chamar FFmpeg no teste: offline gate sempre "não é offline"
    monkeypatch.setattr(clip_analyzer, "is_offline_screen",
                        lambda *a, **kw: (False, "stub"))

    eng = bot_engine.BotEngine(pending_store=store)
    out = tmp_path / "output"
    out.mkdir()
    eng._config = {"system": {"output_folder": str(out), "clips_folder": str(tmp_path)}}
    eng._title_gen = None          # força título fixo, sem Whisper/LLM
    eng._output_dir = out
    eng._clips_dir = tmp_path
    return eng


def _seed(store, raw_ts):
    raw = raw_ts()
    m = make_moment()
    store.add(m, raw, "twitch", "baiano")
    return m, raw


def test_returns_dict_when_editor_discards(engine, store, raw_ts):
    """P0-3 (FALHA HOJE): retorno DEVE ser dict, não None."""
    m, _ = _seed(store, raw_ts)
    engine._editor = _EditorDiscards()

    result = engine.generate_clip_from_moment(m.moment_id, title_text="X")

    assert isinstance(result, dict), (
        "retornou None quando o editor descartou — "
        "o caller faz {**result} e levanta TypeError, matando a thread da UI"
    )
    assert result.get("ok") is False
    assert "error" in result


def test_returns_dict_on_success(engine, store, raw_ts, tmp_path):
    """Baseline: caminho feliz continua retornando dict com ok=True."""
    m, _ = _seed(store, raw_ts)
    engine._editor = _EditorOK(engine._output_dir)

    result = engine.generate_clip_from_moment(m.moment_id, title_text="X")

    assert isinstance(result, dict)
    assert result.get("ok") is True
    assert result.get("path")


def test_all_return_paths_are_dicts(engine, store, raw_ts):
    """
    I1 abrangente: momento inexistente, raw ausente e engine parado
    já retornam dict — trava contra regressão nesses caminhos.
    """
    assert isinstance(engine.generate_clip_from_moment("naoexiste"), dict)

    m, raw = _seed(store, raw_ts)
    raw.unlink()                                    # raw ausente
    assert isinstance(engine.generate_clip_from_moment(m.moment_id), dict)

    m2, _ = _seed(store, raw_ts)
    engine._config = None                           # engine parado
    assert isinstance(engine.generate_clip_from_moment(m2.moment_id), dict)


def test_api_thread_survives_editor_discard(engine, store, raw_ts, monkeypatch):
    """
    I2 (FALHA HOJE): _do_generate_clip não pode levantar e DEVE notificar o JS
    mesmo quando o clip é descartado.
    """
    import clipbot_api

    m, _ = _seed(store, raw_ts)
    engine._editor = _EditorDiscards()

    api = clipbot_api.ClipBotAPI()
    api._engine = engine
    sent = []
    monkeypatch.setattr(api, "_js", lambda code: sent.append(code))

    api._do_generate_clip(m.moment_id, None)        # não pode levantar

    assert len(sent) == 1, "UI nunca recebeu onClipGenerated → spinner infinito"
    assert "onClipGenerated" in sent[0]
    payload = json.loads(sent[0][sent[0].index("(") + 1: sent[0].rindex(")")])
    assert payload["moment_id"] == m.moment_id
    assert payload["ok"] is False


def test_processing_flag_cleared_on_every_path(engine, store, raw_ts):
    """I3: clear_processing sempre roda — senão o momento fica preso pra sempre."""
    m, _ = _seed(store, raw_ts)
    engine._editor = _EditorDiscards()

    engine.generate_clip_from_moment(m.moment_id, title_text="X")

    assert m.moment_id not in store._processing
