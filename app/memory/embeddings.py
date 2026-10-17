"""
Local text embeddings with fastembed (ONNX on the CPU). No text leaves the machine.

The model (snowflake/snowflake-arctic-embed-s, 384 dimensions, ~130 MB) is
downloaded once into .cache/fastembed (or AGENT_MODEL_CACHE) and loaded lazily.
It was chosen over bge-small and all-MiniLM because it alone matched time-of-day
preferences ("no calls before noon") to requests ("sync at 10am?") in a test set.

onnxruntime's own usage telemetry (Microsoft 1DS, HTTPS uploads on Linux) is switched off twice: the
ORT_DISABLE_TELEMETRY variable is set when this module loads, before fastembed or onnxruntime can be
imported through it (with it set, onnxruntime never starts its telemetry client), and
disable_telemetry_events() runs before the model's session is created, in case something else imported
onnxruntime first. Developer decision 2026-10-17: no data to Microsoft.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Protocol

# Must come before any onnxruntime import (fastembed imports it).
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_MODEL = os.getenv("AGENT_EMBEDDING_MODEL", "snowflake/snowflake-arctic-embed-s")

# Retrieval models embed a search query with an instruction prefix (stored texts without).
QUERY_PREFIXES = {
    "snowflake/snowflake-arctic-embed-s": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}
MODEL_CACHE_DIR = Path(os.getenv("AGENT_MODEL_CACHE") or PROJECT_ROOT / ".cache" / "fastembed")


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def disable_onnx_telemetry() -> None:
    """Turn off onnxruntime's telemetry events before any inference session exists."""

    import onnxruntime

    onnxruntime.disable_telemetry_events()


def _normalized(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


class FastEmbedder:
    """fastembed TextEmbedding, loaded on first use and shared by all threads."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, cache_dir: Path = MODEL_CACHE_DIR):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            disable_onnx_telemetry()
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name, cache_dir=str(self.cache_dir))
        return self._model

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        with self._lock:
            return [_normalized(vector) for vector in self._get_model().embed(texts)]

    def embed_query(self, text: str) -> np.ndarray:
        prefix = QUERY_PREFIXES.get(self.model_name, "")
        with self._lock:
            return _normalized(next(iter(self._get_model().embed([prefix + text]))))

    def warm_up(self) -> None:
        with self._lock:
            self._get_model()


_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = FastEmbedder()
        return _embedder


def set_embedder(embedder: Embedder | None) -> None:
    """Replace the process-wide embedder (tests use a small deterministic one)."""

    global _embedder
    with _embedder_lock:
        _embedder = embedder
