"""Embedding backbones behind one interface, with an append-only float16 disk cache keyed by
sha1(text). Reproduction mode (allow_live=False) never computes a new vector."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

import numpy as np
from loguru import logger

from txcat.utils import model_slug, sha1_text


class CacheMissError(RuntimeError):
    """Raised in reproduce mode when a text is not in the embedding cache."""


class Backend(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class FakeHashBackend:
    """Deterministic pseudo-embeddings for tests."""

    def __init__(self, dim: int = 8):
        self.dim, self.calls, self.n_texts = dim, 0, 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.n_texts += len(texts)
        rows = []
        for t in texts:
            rng = np.random.default_rng(int(sha1_text(t)[:8], 16))
            rows.append(rng.standard_normal(self.dim))
        return np.asarray(rows, dtype=np.float32)


class SentenceTransformerBackend:
    """Local CPU sentence-transformers model."""

    def __init__(self, model_name: str, batch_size: int = 256):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.batch_size = batch_size
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, batch_size=self.batch_size,
                              show_progress_bar=len(texts) > 5000,
                              convert_to_numpy=True, normalize_embeddings=False),
            dtype=np.float32,
        )


class FinBERTMeanPoolBackend:
    """ProsusAI/finbert used as an encoder: mean-pool last hidden state (documented choice)."""

    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        self.torch = torch
        self.batch_size = batch_size
        self.dim = int(self.model.config.hidden_size)

    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                b = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                             max_length=32, return_tensors="pt")
                h = self.model(**b).last_hidden_state
                mask = b["attention_mask"].unsqueeze(-1).float()
                out.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


class OpenAIEmbeddingBackend:
    """OpenAI embeddings API; spend is recorded on the ledger if one is given."""

    def __init__(self, model: str, dim: int = 1536, price_per_1m: float = 0.02, ledger=None):
        from openai import OpenAI

        self.client = OpenAI()
        self.model, self.dim, self.price, self.ledger = model, dim, price_per_1m, ledger

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for i in range(0, len(texts), 512):
            chunk = texts[i:i + 512]
            r = self.client.embeddings.create(model=self.model, input=chunk)
            if self.ledger is not None:
                self.ledger.add("embedding", r.usage.total_tokens / 1e6 * self.price, self.model)
            rows.extend(d.embedding for d in r.data)
        return np.asarray(rows, dtype=np.float32)


def make_backend(model_name: str, batch_size: int = 256, ledger=None) -> Backend:
    """Pick a backend from the model id."""
    if model_name.startswith("text-embedding-3"):
        return OpenAIEmbeddingBackend(model_name, ledger=ledger)
    if "finbert" in model_name.lower():
        return FinBERTMeanPoolBackend(model_name)
    return SentenceTransformerBackend(model_name, batch_size=batch_size)


class Embedder:
    """Cached, L2-normalized embeddings for one model."""

    def __init__(self, model_name: str, cache_dir: str | Path, backend: Backend | None = None,
                 allow_live: bool = True, batch_size: int = 256, ledger=None):
        self.model_name = model_name
        self.dir = Path(cache_dir) / model_slug(model_name)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._backend = backend
        self._batch_size, self._ledger = batch_size, ledger
        self.allow_live = allow_live
        self._keys: list[str] = []
        self._pos: dict[str, int] = {}
        self._vecs = np.zeros((0, 0), dtype=np.float16)
        self._load()

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = make_backend(self.model_name, self._batch_size, self._ledger)
        return self._backend

    def _load(self) -> None:
        kp, vp = self.dir / "keys.json", self.dir / "vectors.npy"
        if kp.exists() and vp.exists():
            self._keys = json.loads(kp.read_text())
            self._vecs = np.load(vp)
            self._pos = {k: i for i, k in enumerate(self._keys)}
            logger.info(f"embedding cache {self.model_name}: {len(self._keys)} vectors")

    def _save(self) -> None:
        tmp_v, tmp_k = self.dir / "vectors.npy.tmp", self.dir / "keys.json.tmp"
        with open(tmp_v, "wb") as fh:  # file handle: np.save must not append ".npy"
            np.save(fh, self._vecs)
        tmp_k.write_text(json.dumps(self._keys))
        os.replace(tmp_v, self.dir / "vectors.npy")
        os.replace(tmp_k, self.dir / "keys.json")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return float32 L2-normalized vectors, computing and caching any misses."""
        hashes = [sha1_text(t) for t in texts]
        missing = sorted({h for h in hashes if h not in self._pos})
        if missing:
            if not self.allow_live:
                raise CacheMissError(
                    f"{len(missing)} texts missing from cache for {self.model_name}"
                )
            h2t = {sha1_text(t): t for t in texts}
            new_texts = [h2t[h] for h in missing]
            new = self.backend.encode(new_texts).astype(np.float32)
            new /= np.maximum(np.linalg.norm(new, axis=1, keepdims=True), 1e-12)
            new16 = new.astype(np.float16)
            self._vecs = new16 if self._vecs.size == 0 else np.vstack([self._vecs, new16])
            for h in missing:
                self._pos[h] = len(self._keys)
                self._keys.append(h)
            self._save()
        out = self._vecs[[self._pos[h] for h in hashes]].astype(np.float32)
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        return out
