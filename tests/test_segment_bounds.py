"""
Item 5 — o sidecar declarava os limites do MOMENTO, não do segmento extraído.

Observado em 12/08: um clip declarava 15s no .json e entregava ~36s de vídeo.

O caminho: o momento detectado tem 15s, mas a extração aplica padding para
atingir `clip_min_sec` (= min_clip_duration = 50 na config do usuário):

    pad = (50 - 15) / 2 = 17.5
    seg = (start - 17.5, end + 17.5)        ← é ISTO que vai para o disco
    pending_store.add(m, ...)               ← gravava m.start_sec / m.end_sec

Quem lê o sidecar para achar o trecho no VOD, ou para conferir o que foi
publicado, recebe uma janela que não corresponde ao arquivo.

Segundo defeito, encontrado ao ler o código do padding: o clamp em zero
descartava duração em silêncio. Para um momento perto do início da captura,
`max(0.0, start - pad)` perdia a cabeça do padding e NÃO compensava na cauda,
entregando um segmento menor que `min_clip_duration` — ou seja, a configuração
do usuário não era respeitada justamente nos primeiros minutos de stream.
"""

import json

import pytest

from conftest import make_moment


# ── A janela realmente extraída ─────────────────────────────────────────────

def _bounds(start, end, min_dur):
    import bot_engine
    return bot_engine.BotEngine._segment_bounds(start, end, min_dur)


def test_long_enough_moment_is_not_padded():
    """Momento que já satisfaz o mínimo passa intacto."""
    assert _bounds(100.0, 160.0, 50.0) == (100.0, 160.0)


def test_short_moment_is_padded_symmetrically():
    """15s com mínimo de 50s → 17.5s de cada lado."""
    s, e = _bounds(100.0, 115.0, 50.0)

    assert (s, e) == pytest.approx((82.5, 132.5))
    assert e - s == pytest.approx(50.0)


def test_clamp_at_zero_compensates_on_the_tail():
    """
    O defeito achado na leitura: momento perto do início do stream.
    Sem compensação, o clamp em zero entregava 32.5s em vez dos 50s pedidos.
    """
    s, e = _bounds(2.0, 17.0, 50.0)

    assert s == 0.0
    assert e - s == pytest.approx(50.0), (
        f"segmento de {e - s:.1f}s — min_clip_duration não foi respeitado"
    )


def test_padding_never_shrinks_the_moment():
    """INVARIANTE: o evento detectado está sempre dentro do segmento."""
    for start, end, min_dur in [
        (0.0, 5.0, 50.0), (2.0, 17.0, 50.0), (100.0, 115.0, 50.0),
        (100.0, 160.0, 50.0), (3.0, 4.0, 15.0),
    ]:
        s, e = _bounds(start, end, min_dur)
        assert s <= start and e >= end, f"segmento {s}-{e} recorta {start}-{end}"


def test_zero_minimum_is_a_noop():
    assert _bounds(100.0, 115.0, 0.0) == (100.0, 115.0)


def test_bounds_are_deterministic():
    """Chamadas repetidas devolvem o mesmo — nada de drift entre tentativas."""
    assert _bounds(2.0, 17.0, 50.0) == _bounds(2.0, 17.0, 50.0)


# ── A store guarda a janela do arquivo ──────────────────────────────────────

def test_store_records_the_segment_window(store, raw_ts):
    m = make_moment(start=100.0, end=115.0)

    store.add(m, raw_ts(), "twitch", "kosky", segment=(82.5, 132.5))
    entry = store.get(m.moment_id)

    assert entry["segment_start_sec"] == 82.5
    assert entry["segment_end_sec"] == 132.5
    # O momento continua registrado: é onde o evento está DENTRO do segmento.
    assert entry["start_sec"] == 100.0
    assert entry["end_sec"] == 115.0


def test_store_defaults_segment_to_the_moment(store, raw_ts):
    """Sem padding informado, segmento == momento (compatibilidade)."""
    m = make_moment(start=100.0, end=115.0)

    store.add(m, raw_ts(), "twitch", "kosky")
    entry = store.get(m.moment_id)

    assert entry["segment_start_sec"] == 100.0
    assert entry["segment_end_sec"] == 115.0


def test_old_pending_entries_without_segment_are_tolerated(store, raw_ts, tmp_path):
    """
    pending.json gravado por versões anteriores não tem as chaves novas.
    Restaurar não pode explodir nem inventar valores.
    """
    import pending_store
    raw = raw_ts()
    (tmp_path / "pending.json").write_text(json.dumps([{
        "moment_id": "abc123def456",
        "start_sec": 100.0, "end_sec": 115.0,
        "reason": "keyword", "score": 0.5, "keywords": [], "extra": {},
        "raw_file": str(raw), "platform": "twitch", "channel": "kosky",
        "added_at": "2026-08-12T18:00:00",
    }]), encoding="utf-8")

    entry = pending_store.PendingStore().get("abc123def456")

    assert entry is not None
    assert entry["segment_start_sec"] == 100.0
    assert entry["segment_end_sec"] == 115.0


# ── O sidecar descreve o arquivo ────────────────────────────────────────────

class _EditorOK:
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def process(self, raw_clip, name, **kw):
        p = self.out_dir / f"{name}.mp4"
        p.write_bytes(b"\x00" * 128)
        return p


@pytest.fixture
def engine(store, tmp_path, monkeypatch):
    import bot_engine
    import clip_analyzer
    monkeypatch.setattr(clip_analyzer, "is_offline_screen", lambda *a, **kw: (False, "stub"))
    eng = bot_engine.BotEngine(pending_store=store)
    out = tmp_path / "output"
    out.mkdir()
    eng._config = {"system": {"output_folder": str(out), "clips_folder": str(tmp_path)}}
    eng._title_gen = None
    eng._output_dir = out
    eng._clips_dir = tmp_path
    eng._editor = _EditorOK(out)
    return eng


def _sidecar_of(engine, store, raw_ts, segment):
    m = make_moment(start=100.0, end=115.0)
    store.add(m, raw_ts(), "twitch", "kosky", segment=segment)

    result = engine.generate_clip_from_moment(m.moment_id, title_text="X")

    assert result["ok"] is True, result
    path = __import__("pathlib").Path(result["path"]).with_suffix(".json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_sidecar_declares_the_extracted_segment(engine, store, raw_ts):
    """O número declarado tem de ser o do arquivo: 50s, não os 15s do momento."""
    data = _sidecar_of(engine, store, raw_ts, (82.5, 132.5))

    assert data["segment_start_sec"] == 82.5
    assert data["segment_end_sec"] == 132.5
    assert data["segment_duration_sec"] == pytest.approx(50.0)


def test_sidecar_keeps_the_moment_window(engine, store, raw_ts):
    """
    Os limites do momento continuam no sidecar — é a informação de ONDE está o
    evento dentro do segmento. Remover quebraria quem já consome start_sec.
    """
    data = _sidecar_of(engine, store, raw_ts, (82.5, 132.5))

    assert data["start_sec"] == 100.0
    assert data["end_sec"] == 115.0


def test_sidecar_segment_contains_the_moment(engine, store, raw_ts):
    """INVARIANTE legível no próprio arquivo."""
    data = _sidecar_of(engine, store, raw_ts, (82.5, 132.5))

    assert data["segment_start_sec"] <= data["start_sec"]
    assert data["segment_end_sec"] >= data["end_sec"]


def test_sidecar_without_padding_matches_the_moment(engine, store, raw_ts):
    data = _sidecar_of(engine, store, raw_ts, None)

    assert data["segment_duration_sec"] == pytest.approx(15.0)
    assert data["segment_start_sec"] == data["start_sec"]


# ── A cauda não pode pedir além do material capturado ──────────────────────

def _bounds_av(start, end, min_dur, available):
    import bot_engine
    return bot_engine.BotEngine._segment_bounds(start, end, min_dur, available)


def test_tail_is_clamped_to_captured_material():
    """
    A explicação real dos 36s de 12/08: o padding pedia além da borda da live
    e o FFmpeg entregava só o que existia. A duração é recuperada na cabeça.
    """
    s, e = _bounds_av(100.0, 115.0, 50.0, available=120.0)

    assert e <= 120.0, "pediu material que ainda não existe"
    assert e - s == pytest.approx(50.0), f"segmento de {e - s:.1f}s"
    assert s <= 100.0 and e >= 115.0


def test_clamp_never_cuts_the_moment():
    """Mesmo com pouquíssimo material após o evento, o evento fica inteiro."""
    s, e = _bounds_av(100.0, 115.0, 50.0, available=116.0)

    assert s <= 100.0 and e >= 115.0
    assert e <= 116.0


def test_clamp_reports_a_shorter_segment_when_nothing_else_fits():
    """
    Quando o mínimo não cabe nem puxando a cabeça, o segmento devolvido é
    menor — e é essa janela, a verdadeira, que vai ao sidecar.
    """
    s, e = _bounds_av(2.0, 17.0, 50.0, available=20.0)

    assert s == 0.0
    assert e == pytest.approx(20.0)
    assert e - s == pytest.approx(20.0)


def test_available_beyond_the_window_changes_nothing():
    com    = _bounds_av(100.0, 115.0, 50.0, available=10_000.0)
    sem    = _bounds(100.0, 115.0, 50.0)

    assert com == sem


def test_clamped_bounds_still_contain_the_moment():
    """INVARIANTE varrendo o espaço de parâmetros."""
    for start, end in [(0.5, 3.0), (2.0, 17.0), (100.0, 115.0), (100.0, 160.0)]:
        for av in (end + 0.1, end + 2.0, end + 10.0, end + 100.0):
            s, e = _bounds_av(start, end, 50.0, available=av)
            assert s <= start and e >= end, f"{s}-{e} recorta {start}-{end} (av={av})"
            assert s >= 0.0
            assert e <= max(end, av) + 1e-9, f"{e} passa do disponível {av}"
