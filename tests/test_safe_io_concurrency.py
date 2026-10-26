"""
Lote 1 / A6 — safe_io.atomic_write_text usava um .tmp de nome fixo.

O temporário era sempre "<arquivo>.tmp". Dois escritores do mesmo arquivo
(dois feedbacks, dois "Gerar", update de credenciais) abriam o MESMO tmp com
"w" — o segundo truncava o que o primeiro escrevia — e os dois faziam rename
dele para o destino:

  • o rename de um encontrava o tmp já movido → StorageWriteError espúrio;
  • o destino podia receber o conteúdo intercalado dos dois → JSON inválido
    no disco, que a próxima leitura tratava como corrompido.

A revisão mediu 4 erros de escrita e 1 leitura de JSON inválido em 2×300
escritas. O pending.json escapava porque grava sob o lock da store.

Contrato: cada chamada usa o seu próprio temporário.
"""

import json
import threading

import pytest


def test_concurrent_writers_use_distinct_temp_files(tmp_path, monkeypatch):
    """
    Determinístico: segura os dois escritores no rename e compara os tmps.
    Com o nome fixo, os dois chegam aqui com o mesmo caminho.
    """
    import safe_io
    target = tmp_path / "x.json"
    seen = []
    barrier = threading.Barrier(2, timeout=2)
    real = safe_io._replace

    def spy(src, dst):
        seen.append(str(src))
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return real(src, dst)

    monkeypatch.setattr(safe_io, "_replace", spy)

    def write(payload):
        try:
            safe_io.atomic_write_text(target, payload, retries=2, backoff=0.0)
        except OSError:
            pass

    ts = [threading.Thread(target=write, args=(json.dumps({"w": i}),)) for i in (1, 2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert len(seen) >= 2
    assert seen[0] != seen[1], f"os dois escritores usaram o mesmo tmp: {seen[0]}"


def test_concurrent_writers_never_fail_or_leave_garbage(tmp_path):
    import safe_io
    target = tmp_path / "x.json"
    errors = []

    def writer(tag):
        for i in range(100):
            payload = json.dumps({"tag": tag, "i": i, "pad": "x" * 4000})
            try:
                safe_io.atomic_write_text(target, payload)
            except OSError as ex:
                errors.append(ex)

    ts = [threading.Thread(target=writer, args=(t,)) for t in ("a", "b")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert errors == [], f"{len(errors)} escrita(s) falharam: {errors[:2]}"
    json.loads(target.read_text(encoding="utf-8"))           # destino íntegro
    assert list(tmp_path.glob("*.tmp")) == [], "sobrou arquivo temporário"


def test_readers_never_see_interleaved_content(tmp_path):
    """
    O destino só pode conter um payload inteiro de UM escritor. Leituras que
    falham por lock momentâneo do rename são aceitáveis; JSON quebrado não.
    """
    import safe_io
    target = tmp_path / "x.json"
    safe_io.atomic_write_text(target, json.dumps({"tag": "init"}))
    stop = threading.Event()
    broken = []

    def writer(tag):
        for i in range(100):
            try:
                safe_io.atomic_write_text(
                    target, json.dumps({"tag": tag, "i": i, "pad": tag * 4000})
                )
            except OSError:
                pass

    def reader():
        while not stop.is_set():
            try:
                raw = target.read_text(encoding="utf-8")
            except OSError:
                continue
            try:
                d = json.loads(raw)
                if "pad" in d and set(d["pad"]) != {d["tag"]}:
                    broken.append("misturado")
            except ValueError:
                broken.append(raw[:40])
            stop.wait(0.002)

    r = threading.Thread(target=reader)
    r.start()
    ws = [threading.Thread(target=writer, args=(t,)) for t in ("a", "b")]
    for t in ws:
        t.start()
    for t in ws:
        t.join()
    stop.set()
    r.join()

    assert broken == [], f"{len(broken)} leitura(s) de conteúdo inválido"


def test_single_writer_behaviour_is_unchanged(tmp_path):
    """Baseline: escrita simples continua idêntica (conteúdo e sem lixo)."""
    import safe_io
    target = tmp_path / "x.json"

    safe_io.atomic_write_text(target, '{"ok": 1}')

    assert target.read_text(encoding="utf-8") == '{"ok": 1}'
    assert list(tmp_path.iterdir()) == [target]
