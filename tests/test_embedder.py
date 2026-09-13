import numpy as np
import pytest

from txcat.embedder import CacheMissError, Embedder, FakeHashBackend


def test_embed_is_normalized_and_cached(tmp_path):
    be = FakeHashBackend(dim=8)
    e = Embedder("fake/model", tmp_path, backend=be)
    v = e.embed(["STAPLES", "USPS", "STAPLES"])
    assert v.shape == (3, 8)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(v[0], v[2])
    assert be.calls == 1 and be.n_texts == 2  # deduped before hitting the backend

    e2 = Embedder("fake/model", tmp_path, backend=be)  # fresh instance reloads cache
    v2 = e2.embed(["USPS", "STAPLES"])
    assert be.calls == 1  # no new backend calls
    np.testing.assert_allclose(v2[1], v[0])


def test_reproduce_mode_raises_on_miss(tmp_path):
    e = Embedder("fake/model", tmp_path, backend=FakeHashBackend(8), allow_live=False)
    with pytest.raises(CacheMissError):
        e.embed(["NEVER SEEN"])


def test_cache_dir_is_per_model(tmp_path):
    Embedder("a/x", tmp_path, backend=FakeHashBackend(4)).embed(["Q"])
    Embedder("b/y", tmp_path, backend=FakeHashBackend(4)).embed(["Q"])
    assert (tmp_path / "a__x" / "vectors.npy").exists()
    assert (tmp_path / "b__y" / "vectors.npy").exists()
