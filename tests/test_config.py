import pytest
from pydantic import ValidationError

from txcat.config import _ROOT, load_config


def test_default_config_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.tail_k == 3
    assert cfg.datasets["dc"].window.train_start == "2019-01-01"
    assert cfg.llm.models[0].name == "gpt-5-nano"
    assert cfg.budget.max_usd == 40.0


def test_default_only_branch_loads():
    cfg = load_config()
    assert cfg.active_dataset == "dc"
    assert cfg.seed == 42


def test_dc_config_loads():
    cfg = load_config("configs/dc.yaml")
    assert cfg.active_dataset == "dc"
    assert cfg.index.M == 32  # dc.yaml is a near-no-op override


def test_override_merges(tmp_path):
    p = tmp_path / "o.yaml"
    p.write_text("tail_k: 5\nindex:\n  M: 16\n")
    cfg = load_config(str(p))
    assert cfg.tail_k == 5
    assert cfg.index.M == 16
    assert cfg.index.ef_construction == 200  # untouched default survives deep merge


def test_misspelled_key_is_rejected(tmp_path):
    p = tmp_path / "typo.yaml"
    p.write_text("gate:\n  treshold: 0.9\n")
    with pytest.raises(ValidationError):
        load_config(str(p))


def test_paths_are_root_anchored(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    for value in (
        cfg.embed.cache_dir,
        cfg.results_dir,
        cfg.logs_dir,
        cfg.taxonomy_dir,
        cfg.budget.ledger_path,
        cfg.search.cache_dir,
        cfg.llm.cache_dir,
        cfg.llm.prompt_with_web,
        cfg.llm.prompt_no_web,
        cfg.datasets["dc"].raw_dir,
        cfg.datasets["dc"].processed_path,
    ):
        assert value.startswith(str(_ROOT)), value


def test_relative_override_resolves_from_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = load_config("configs/dc.yaml")
    assert cfg.active_dataset == "dc"


def test_config_is_frozen():
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.tail_k = 99


def test_unquoted_dates_are_coerced(tmp_path):
    p = tmp_path / "dates.yaml"
    p.write_text("datasets:\n  dc:\n    window:\n      train_start: 2020-01-01\n")
    cfg = load_config(str(p))
    assert cfg.datasets["dc"].window.train_start == "2020-01-01"


def test_bad_literal_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("gate:\n  mode: nonsense\n")
    with pytest.raises(ValidationError):
        load_config(str(p))


def test_unknown_active_dataset_is_rejected(tmp_path):
    p = tmp_path / "bad_active.yaml"
    p.write_text("active_dataset: nope\n")
    with pytest.raises(ValidationError):
        load_config(str(p))
