import json

from txcat.manifest import build_manifest


def test_manifest_has_required_keys(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    m = build_manifest(
        prompt_paths=[tmp_path / "a.txt"], config={"index": {"M": 32}}, model_versions={"x": "y"}
    )
    for k in (
        "timestamp",
        "git_commit",
        "host",
        "python",
        "packages",
        "prompt_hashes",
        "hnsw_params",
        "model_versions",
    ):
        assert k in m
    assert m["hnsw_params"] == {"M": 32}
    assert "/Users/" not in json.dumps(m)  # no absolute paths in the committed manifest
    assert len(next(iter(m["prompt_hashes"].values()))) == 64
    json.dumps(m)  # serializable
