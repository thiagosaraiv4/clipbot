"""
Lote 1 / A1 — o GC de startup apagava arquivos que não são do ClipBot.

`_garbage_collect_disk` roda a cada START e varria clips_folder e output_folder
com os padrões "raw_*.ts", "*.mp4" e "*.json", apagando o que tivesse mais de
7 dias. Os dois caminhos são campos livres na UI: quem aponta a saída para
Vídeos, Desktop ou uma pasta de projeto perde, sem aviso, todo vídeo e todo
JSON pessoal com mais de uma semana.

O ClipBot só cria, nessas pastas:
    clips_folder   raw_{plataforma}_{canal}_{ts}.ts      (captura contínua)
                   raw_pending_*.ts / raw_clip_*.ts       (segmentos)
    output_folder  clip_*.mp4 + clip_*.json               (clip + sidecar)

Contrato: o GC só toca o que segue esses nomes.
"""

import os
import time

import pytest

OLD = time.time() - 8 * 24 * 3600      # além do limite de 7 dias


@pytest.fixture
def engine(store):
    import bot_engine
    return bot_engine.BotEngine(pending_store=store)


@pytest.fixture
def dirs(tmp_path):
    clips, out = tmp_path / "clips", tmp_path / "out"
    clips.mkdir()
    out.mkdir()
    return clips, out


def _touch(path, mtime=OLD, data=b"x"):
    path.write_bytes(data)
    os.utime(path, (mtime, mtime))
    return path


def test_user_files_survive_even_when_old(engine, dirs):
    """O bug: arquivos pessoais com mais de 7 dias eram apagados."""
    clips, out = dirs
    pessoais = [
        _touch(out / "ferias_2025.mp4"),
        _touch(out / "projeto.json"),
        _touch(clips / "video_do_casamento.mp4"),
        _touch(clips / "config_do_obs.json"),
        _touch(clips / "gravacao_minha.ts"),
    ]

    engine._garbage_collect_disk(clips, out)

    sumiram = [p.name for p in pessoais if not p.exists()]
    assert sumiram == [], f"GC apagou arquivos do usuário: {sumiram}"


def test_old_clipbot_files_are_still_collected(engine, dirs):
    """O GC continua fazendo o trabalho dele com os arquivos do app."""
    clips, out = dirs
    do_app = [
        _touch(out / "clip_twitch_kosky_20260101_000000_keyword.mp4"),
        _touch(out / "clip_twitch_kosky_20260101_000000_keyword.json"),
        _touch(clips / "raw_twitch_kosky_20260101_000000.ts"),
        _touch(clips / "raw_pending_twitch_kosky_20260101_000000_keyword.ts"),
        _touch(clips / "raw_clip_twitch_kosky_20260101_000000_keyword.ts"),
    ]

    engine._garbage_collect_disk(clips, out)

    ficaram = [p.name for p in do_app if p.exists()]
    assert ficaram == [], f"GC deixou de recolher arquivos do app: {ficaram}"


def test_recent_clipbot_files_are_kept(engine, dirs):
    clips, out = dirs
    recente = _touch(out / "clip_twitch_kosky_20260101_000000_keyword.mp4",
                     mtime=time.time() - 3600)

    engine._garbage_collect_disk(clips, out)

    assert recente.exists()


def test_raw_referenced_by_pending_is_kept(engine, store, dirs):
    """INVARIANTE pré-existente: .ts de um momento pendente nunca é apagado."""
    from conftest import make_moment
    clips, out = dirs
    raw = _touch(clips / "raw_pending_twitch_kosky_20260101_000000_keyword.ts")
    store.add(make_moment(), raw, "twitch", "kosky")
    os.utime(raw, (OLD, OLD))

    engine._garbage_collect_disk(clips, out)

    assert raw.exists()


def test_same_folder_for_clips_and_output(engine, tmp_path):
    """Configuração comum: as duas pastas apontando para o mesmo lugar."""
    pasta = tmp_path / "Videos"
    pasta.mkdir()
    pessoal = _touch(pasta / "minhas_ferias.mp4")
    do_app  = _touch(pasta / "clip_twitch_kosky_20260101_000000_keyword.mp4")

    engine._garbage_collect_disk(pasta, pasta)

    assert pessoal.exists()
    assert not do_app.exists()
