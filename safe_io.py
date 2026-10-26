"""
safe_io.py — I/O de estado que não mente.

Três regras, nascidas do incidente de 12/08 (credentials.enc "inexistente" que
existia, e pendentes que não chegaram ao disco sem um único erro no log):

  1. NUNCA use Path.exists() para decidir se um arquivo pode ser lido.
     No Windows, exists() devolve False quando a abertura falha por permissão
     (antivírus/indexador com o handle aberto). Tente ler e classifique pela
     exceção: ausente e ilegível são estados DIFERENTES.

  2. Escrita é atômica E tem retry. tmp + fsync + rename atômico protege contra
     crash no meio do write; o retry cobre a janela em que o AV ainda segura o
     arquivo de destino. Se as tentativas acabarem, o erro SOBE — quem chamou
     decide, ninguém engole.

  3. Falha de escrita não trunca o arquivo bom anterior. O destino só é
     substituído por um rename bem-sucedido.

Usado por credentials.py, pending_store.py e feedback.py.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("ClipBot.SafeIO")

# Estados possíveis de uma leitura
MISSING    = "missing"      # não existe — situação normal (primeira execução)
OK         = "ok"           # lido com sucesso
UNREADABLE = "unreadable"   # existe mas não abriu/decodificou — ANOMALIA
CORRUPT    = "corrupt"      # leu, mas o conteúdo não faz sentido — ANOMALIA

#: Indireção para os testes injetarem falha de rename sem tocar o módulo `os`.
_replace = os.replace


class StorageWriteError(OSError):
    """Escrita falhou após esgotar as tentativas. É OSError de propósito."""


def read_text_classified(
    path, encoding: str = "utf-8"
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Lê um arquivo de texto e classifica o resultado.

    Returns:
        (status, conteudo, erro) — conteúdo só em OK, erro só em UNREADABLE.

    Não chama exists(): a tentativa de leitura É a checagem (regra 1).
    """
    try:
        return OK, Path(path).read_text(encoding=encoding), None
    except FileNotFoundError:
        return MISSING, None, None
    except NotADirectoryError:
        # Um componente do caminho não é pasta → o arquivo não existe
        return MISSING, None, None
    except (OSError, UnicodeDecodeError) as ex:
        # PermissionError, IsADirectoryError, erro de encoding, I/O do disco…
        return UNREADABLE, None, f"{type(ex).__name__}: {ex}"


def atomic_write_text(
    path,
    text: str,
    *,
    retries: int = 3,
    backoff: float = 0.05,
    encoding: str = "utf-8",
) -> None:
    """
    Escreve texto atomicamente, com retry em falha transitória.

    Raises:
        StorageWriteError: se todas as tentativas falharem. O arquivo de
            destino permanece com o conteúdo anterior intacto.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    last_err: Optional[BaseException] = None

    for attempt in range(1, max(1, retries) + 1):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding=encoding) as f:
                f.write(text)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    # Alguns FS (drives de rede, FAT32) não suportam fsync
                    pass
            _replace(tmp, path)        # rename atômico substitui o antigo
            return
        except OSError as ex:
            last_err = ex
            if attempt < max(1, retries):
                logger.debug(
                    f"escrita de {path.name} falhou (tentativa {attempt}): {ex}"
                )
                time.sleep(backoff * attempt)

    # Esgotou: limpa o lixo e deixa o erro subir (regras 2 e 3)
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    raise StorageWriteError(
        f"não foi possível gravar {path} após {max(1, retries)} tentativa(s): {last_err}"
    ) from last_err


def quarantine(path) -> Optional[Path]:
    """
    Move um arquivo de estado corrompido para <nome>.corrupt.<timestamp>.

    Preserva o conteúdo para diagnóstico e libera o caminho para o app
    recomeçar dos defaults sem destruir nada. None se não conseguiu mover.
    """
    path = Path(path)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.corrupt.{stamp}")
    try:
        _replace(path, backup)
        return backup
    except OSError as ex:
        logger.debug(f"quarentena de {path.name} falhou: {ex}")
        return None


def check_writable(directory) -> Tuple[bool, Optional[str]]:
    """
    Verifica, de fato escrevendo, se a pasta de estado aceita gravação.

    Chamado no boot: descobrir que o APPDATA está bloqueado na inicialização é
    muito melhor que descobrir ao perder o primeiro momento pendente.

    Returns:
        (ok, erro) — não levanta exceção.
    """
    probe = Path(directory) / ".clipbot_write_probe"
    try:
        atomic_write_text(probe, "ok", retries=1, backoff=0.0)
        return True, None
    except OSError as ex:
        return False, str(ex)
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
