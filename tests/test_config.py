from txcat.config import load_config


def test_default_config_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.tail_k == 3
    assert cfg.datasets["dc"].window.train_start == "2019-01-01"
    assert cfg.llm.models[0].name == "gpt-5-nano"
    assert cfg.budget.max_usd == 40.0


def test_override_merges(tmp_path):
    p = tmp_path / "o.yaml"
    p.write_text("tail_k: 5\nindex:\n  M: 16\n")
    cfg = load_config(str(p))
    assert cfg.tail_k == 5
    assert cfg.index.M == 16
    assert cfg.index.ef_construction == 200  # untouched default survives deep merge
