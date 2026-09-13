import gzip
import json

from txcat.cachepack import load_packed, pack


def _llm_rec(model, prompt, merchant, cat):
    return {
        "model": model,
        "prompt_hash": "h",
        "prompt_file": f"prompts/{prompt}.txt",
        "merchant": merchant,
        "timestamp": "t",
        "request": {},
        "raw_response": "{}",
        "result": {
            "category": cat,
            "confidence": 0.5,
            "evidence": "",
            "valid": True,
            "model_id": model,
            "tokens_in": 1,
            "tokens_out": 1,
            "latency_ms": 1.0,
            "cached": False,
        },
    }


def test_pack_groups_by_model_and_prompt_and_reads_back(tmp_path):
    (tmp_path / "k1.json").write_text(
        json.dumps(_llm_rec("gpt-5-nano", "fallback_no_web", "A", "retail"))
    )
    (tmp_path / "k2.json").write_text(
        json.dumps(_llm_rec("gpt-5-nano", "fallback_with_web", "A", "retail"))
    )
    (tmp_path / "k3.json").write_text(
        json.dumps(_llm_rec("org/model-x", "fallback_no_web", "B", "health"))
    )
    counts = pack(tmp_path, "llm", remove=True)
    assert counts == {
        "gpt-5-nano__fallback_no_web": 1,
        "gpt-5-nano__fallback_with_web": 1,
        "org__model-x__fallback_no_web": 1,
    }
    assert not list(tmp_path.glob("*.json"))  # per-call files removed
    packed = load_packed(tmp_path)
    assert set(packed) == {"k1", "k2", "k3"} and packed["k3"]["result"]["category"] == "health"
    # re-pack with a new per-call file: existing packed records survive, new one is added
    (tmp_path / "k4.json").write_text(
        json.dumps(_llm_rec("gpt-5-nano", "fallback_no_web", "C", "lodging"))
    )
    counts = pack(tmp_path, "llm", remove=True)
    assert counts["gpt-5-nano__fallback_no_web"] == 2
    with gzip.open(tmp_path / "packed" / "gpt-5-nano__fallback_no_web.jsonl.gz", "rt") as fh:
        keys = [json.loads(line)["key"] for line in fh]
    assert keys == ["k1", "k4"]


def test_pack_search_groups_by_provider(tmp_path):
    (tmp_path / "s1.json").write_text(
        json.dumps({"query": "Q", "provider": "firecrawl", "timestamp": "t", "results": []})
    )
    assert pack(tmp_path, "search") == {"firecrawl": 1}
    assert load_packed(tmp_path)["s1"]["query"] == "Q"
