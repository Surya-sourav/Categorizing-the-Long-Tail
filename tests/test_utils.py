import random

import numpy as np

from txcat.utils import model_slug, set_seed, sha1_text, sha256_text


def test_set_seed_is_reproducible():
    set_seed(7)
    a = (random.random(), np.random.rand())
    set_seed(7)
    b = (random.random(), np.random.rand())
    assert a == b


def test_hashes_are_stable():
    assert sha1_text("x") == "11f6ad8ec52a2984abaafd7c3b516503785c2072"
    assert len(sha1_text("x")) == 40
    assert len(sha256_text("x")) == 64
    assert sha1_text("a") != sha1_text("b")


def test_model_slug():
    slug = model_slug("sentence-transformers/all-MiniLM-L6-v2")
    assert slug == "sentence-transformers__all-MiniLM-L6-v2"
    assert model_slug("text-embedding-3-small") == "text-embedding-3-small"
