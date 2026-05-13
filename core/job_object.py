"""
core/job_object.py — Windows Job Object para kill-on-parent-death.

Por que existe:
  Quando o processo Python morre por SIGKILL (Task Manager force quit, crash
  do PyWebView, BSOD, etc.), atexit handlers NÃO rodam. Children ffmpeg.exe
  ficam zombies consumindo CPU/RAM até o usuário matá-los manualmente.

  Job Object com JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE resolve isso no nível do
  kernel: quando o handle do Job é fechado (sempre acontece quando o processo
  pai morre), Windows mata TODOS os processos no Job, qualquer que tenha sido
  a causa da morte do pai.

Implementação:
  • ctypes direto na kernel32.dll — sem dependência pywin32.
  • Idempotente. Setup falho não derruba app (graceful degradation).
  • No-op em Linux/macOS (ali usar process group / nuclear option diferente).

Nota técnica (Windows nested jobs):
  No Win8+ um processo PODE estar em N jobs aninhados (se cada Job pai permite
  via JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK). Mas em alguns ambientes
  (Docker host process, alguns sandboxes) AssignProcessToJobObject pode falhar
  com ERROR_ACCESS_DENIED. Tratamos com warning, continua sem Job Object.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes
from typing import Optional

logger = logging.getLogger("ClipBot.JobObject")

# ── Constants do Windows SDK ─────────────────────────────────────────────────
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE       = 0x00002000
JOB_OBJECT_LIMIT_BREAKAWAY_OK            = 0x00000800
JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK     = 0x00001000
JobObjectExtendedLimitInformation        = 9

# ── Structs ──────────────────────────────────────────────────────────────────

class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit",     wintypes.LARGE_INTEGER),
        ("LimitFlags",              wintypes.DWORD),
        ("MinimumWorkingSetSize",   ctypes.c_size_t),
        ("MaximumWorkingSetSize",   ctypes.c_size_t),
        ("ActiveProcessLimit",      wintypes.DWORD),
        ("Affinity",                ctypes.c_size_t),
        ("PriorityClass",           wintypes.DWORD),
        ("SchedulingClass",         wintypes.DWORD),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount",  ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount",   ctypes.c_uint64),
        ("WriteTransferCount",  ctypes.c_uint64),
        ("OtherTransferCount",  ctypes.c_uint64),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo",                _IO_COUNTERS),
        ("ProcessMemoryLimit",    ctypes.c_size_t),
        ("JobMemoryLimit",        ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed",     ctypes.c_size_t),
    ]


# ── State global (Job handle mantido vivo enquanto processo viver) ──────────
_job_handle: Optional[int] = None


def setup_job_object() -> bool:
    """
    Cria + configura + atribui processo atual a um Job Object com KILL_ON_CLOSE.

    Idempotente. Em sucesso: True. Em falha (incl. Linux/macOS): False com
    warning logado. Nunca lança exceção pro caller.

    Mantém o handle do Job em variável module-level para não ser GC'd —
    se o handle fechar, Windows mata o Job (e tudo dentro), que é o
    comportamento que queremos no shutdown mas NÃO durante runtime.
    """
    global _job_handle
    if _job_handle is not None:
        return True   # already set up
    if sys.platform != "win32":
        logger.debug("Job Object: non-Windows platform, skipping")
        return False

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        # Type hints para evitar warnings + clareza
        kernel32.CreateJobObjectW.argtypes  = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype   = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype  = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype  = wintypes.BOOL
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype  = wintypes.HANDLE

        # 1) Cria Job (sem nome — anônimo, só para este processo)
        h_job = kernel32.CreateJobObjectW(None, None)
        if not h_job:
            err = ctypes.get_last_error()
            logger.warning(f"Job Object: CreateJobObjectW falhou (errno={err}); seguindo sem")
            return False

        # 2) Configura LimitFlags = KILL_ON_JOB_CLOSE
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            h_job, JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info),
        )
        if not ok:
            err = ctypes.get_last_error()
            logger.warning(f"Job Object: SetInformationJobObject falhou (errno={err}); seguindo sem")
            return False

        # 3) Atribui o processo atual ao Job
        # Caveat: pode falhar com ERROR_ACCESS_DENIED (5) se já estamos em outro
        # Job que não permite breakaway (ex: alguns sandboxes / Docker host).
        h_proc = kernel32.GetCurrentProcess()
        ok = kernel32.AssignProcessToJobObject(h_job, h_proc)
        if not ok:
            err = ctypes.get_last_error()
            if err == 5:  # ERROR_ACCESS_DENIED
                logger.info(
                    "Job Object: processo já em Job pai sem breakaway "
                    "(comum em alguns sandboxes); seguindo sem novo Job"
                )
            else:
                logger.warning(f"Job Object: AssignProcessToJobObject falhou (errno={err})")
            return False

        # Sucesso — manter handle vivo
        _job_handle = h_job
        logger.info("✅ Job Object: ativo (children morrerão se processo pai morrer)")
        return True

    except Exception as e:
        logger.warning(f"Job Object: setup falhou inesperadamente: {e}")
        return False


def is_active() -> bool:
    return _job_handle is not None
