"""
Regressão — BlazeFace precisa carregar a partir de caminho NÃO-ASCII.

Bug observado em produção (12/08), após mover o projeto para
"C:/Users/thiag/OneDrive/Área de Trabalho/clipbot":

    BlazeFace falhou (FileNotFoundError: Unable to open file at
    .../Área de Trabalho/clipbot/assets/blaze_face_short_range.tflite)
    → fallback Haar → hit_ratio=1% → pad+blur

O arquivo existia e era legível pelo Python; quem não abre caminho com "Á" é o
loader nativo do MediaPipe (model_asset_path vai direto para o C++). Passar os
bytes via model_asset_buffer mantém o caminho fora do loader nativo.

Sem este teste, qualquer pasta com acento/cedilha desliga o PR8 inteiro
silenciosamente — o fallback mascara a falha.
"""

import shutil

import pytest

mediapipe = pytest.importorskip("mediapipe", reason="mediapipe não instalado")

from pathlib import Path  # noqa: E402

MODEL = Path(__file__).resolve().parent.parent / "assets" / "blaze_face_short_range.tflite"

pytestmark = pytest.mark.skipif(
    not MODEL.exists(), reason="modelo blaze_face_short_range.tflite ausente"
)


@pytest.fixture
def fresh_tracker(monkeypatch):
    """face_tracker com estado global de carga zerado."""
    import face_tracker as ft
    monkeypatch.setattr(ft, "_mp_detector", None)
    monkeypatch.setattr(ft, "_mp_load_attempted", False)
    return ft


def test_loads_from_non_ascii_path(fresh_tracker, tmp_path, monkeypatch):
    """REGRESSÃO: caminho com acento (ex.: 'Área de Trabalho') deve funcionar."""
    weird = tmp_path / "Área de Trabalho" / "ção ñ" / "assets"
    weird.mkdir(parents=True)
    shutil.copy2(MODEL, weird / MODEL.name)
    monkeypatch.setenv("BLAZEFACE_MODEL", str(weird / MODEL.name))

    assert fresh_tracker._ensure_blazeface() is True, (
        "BlazeFace não carregou de caminho não-ASCII — tracker cai para Haar "
        "e o speaker tracking do PR8 fica desativado sem erro visível"
    )
    assert fresh_tracker._mp_detector is not None


def test_loads_from_ascii_path(fresh_tracker, tmp_path, monkeypatch):
    """Baseline: caminho comum continua funcionando."""
    plain = tmp_path / "assets"
    plain.mkdir()
    shutil.copy2(MODEL, plain / MODEL.name)
    monkeypatch.setenv("BLAZEFACE_MODEL", str(plain / MODEL.name))

    assert fresh_tracker._ensure_blazeface() is True


def test_missing_model_falls_back_without_raising(fresh_tracker, tmp_path, monkeypatch):
    """INVARIANTE: sem modelo, retorna False (→ Haar) em vez de explodir."""
    monkeypatch.setenv("BLAZEFACE_MODEL", str(tmp_path / "nao_existe.tflite"))
    monkeypatch.setattr(fresh_tracker, "_find_blazeface_model", lambda: None)

    assert fresh_tracker._ensure_blazeface() is False
