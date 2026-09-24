"""Runtime resource controls for constrained server deployments."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import gc
import os
from threading import Lock
from typing import Iterator


_INFERENCE_LOCK = Lock()
_THREADS_CONFIGURED = False
_TRUTHY = {"1", "true", "yes", "on"}


def low_resource_mode() -> bool:
    """Return whether the opt-in low-memory server policy is enabled."""
    return os.getenv("SIGANUSMORPH_LOW_RESOURCE_MODE", "").strip().lower() in _TRUTHY


def configure_runtime_threads() -> int:
    """Apply conservative native-library thread limits once per process."""
    global _THREADS_CONFIGURED
    raw = os.getenv("SIGANUSMORPH_RUNTIME_THREADS", os.getenv("OMP_NUM_THREADS", "1"))
    try:
        threads = max(1, int(raw))
    except (TypeError, ValueError):
        threads = 1
    if _THREADS_CONFIGURED:
        return threads

    try:
        import cv2

        cv2.setNumThreads(threads)
    except Exception:
        pass
    try:
        import torch

        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch only allows this before inter-op work starts.
            pass
    except Exception:
        pass
    _THREADS_CONFIGURED = True
    return threads


def collect_runtime_memory() -> None:
    """Collect Python objects and return free heap pages on Linux when possible."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if os.name == "posix":
        try:
            malloc_trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
            if malloc_trim is not None:
                malloc_trim(0)
        except Exception:
            pass


@contextmanager
def measurement_inference_guard() -> Iterator[None]:
    """Serialize heavy inference only in low-resource server mode."""
    if low_resource_mode():
        with _INFERENCE_LOCK:
            yield
        return
    yield
