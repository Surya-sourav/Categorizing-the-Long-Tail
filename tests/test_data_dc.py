import json
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest
import responses

from txcat.data import dc
from txcat.data.dc import build_query_params, features_to_frame
from txcat.data.schema import PROCESSED_COLUMNS, validate_processed

FAKE_FEATURES = [
    {"attributes": {"OBJECTID": 1, "AGENCY": "Office of Latino Affairs",
                    "TRANSACTION_DATE": 1704067200000,
                    "TRANSACTION_AMOUNT": 16.8, "VENDOR_NAME": "USPS 1050050275    QQQ",
                    "VENDOR_STATE_PROVINCE": "DC",
                    "MCC_DESCRIPTION": "Postage Services-Government Only"}},
    {"attributes": {"OBJECTID": 2, "AGENCY": "DDOT", "TRANSACTION_DATE": 1704153600000,
                    "TRANSACTION_AMOUNT": 229.5, "VENDOR_NAME": "WW GRAINGER 912",
                    "VENDOR_STATE_PROVINCE": "DC",
                    "MCC_DESCRIPTION": "Industrial Supplies, Not Elsewhere Classified"}},
    {"attributes": {"OBJECTID": 3, "AGENCY": "DDOT", "TRANSACTION_DATE": None,
                    "TRANSACTION_AMOUNT": 1.0, "VENDOR_NAME": None, "VENDOR_STATE_PROVINCE": "DC",
                    "MCC_DESCRIPTION": "X"}},
]


def test_features_to_frame_drops_pii_and_bad_rows():
    df = features_to_frame(FAKE_FEATURES)
    assert list(df.columns) == PROCESSED_COLUMNS
    assert len(df) == 2  # row with null date / vendor dropped
    assert df.loc[0, "txn_id"] == "dc:1"
    assert df.loc[0, "raw_merchant"] == "USPS 1050050275    QQQ"
    assert df.loc[0, "source"] == "dc"
    assert str(df.loc[0, "date"].date()) == "2024-01-01"
    for forbidden in ("AGENCY", "TRANSACTION_AMOUNT", "VENDOR_STATE_PROVINCE"):
        assert forbidden not in df.columns
    validate_processed(df)  # does not raise


def test_validate_processed_rejects_extra_columns():
    df = features_to_frame(FAKE_FEATURES)
    df["amount"] = 1.0
    with pytest.raises(ValueError):
        validate_processed(df)


def test_build_query_params_pages_and_filters():
    p = build_query_params(start_date="2019-01-01", offset=3000, page_size=1000)
    assert p["where"] == "TRANSACTION_DATE >= DATE '2019-01-01'"
    assert p["resultOffset"] == 3000 and p["resultRecordCount"] == 1000
    assert p["outFields"] == "OBJECTID,TRANSACTION_DATE,VENDOR_NAME,MCC_DESCRIPTION"
    assert p["orderByFields"] == "OBJECTID ASC"


# --- download/resume layer (mocked HTTP) ------------------------------------------------------


def _feature(oid: int) -> dict:
    return {
        "attributes": {
            "OBJECTID": oid,
            "TRANSACTION_DATE": 1704067200000,
            "VENDOR_NAME": f"VENDOR {oid}",
            "MCC_DESCRIPTION": "X",
        }
    }


def _page(oids: list[int], exceeded: bool) -> dict:
    page = {"features": [_feature(o) for o in oids]}
    if exceeded:
        page["exceededTransferLimit"] = True
    return page


def _register_layer(pages: dict[int, dict], count: int | None, seen: list) -> None:
    """Serve ``pages`` by resultOffset and ``count`` for the returnCountOnly query."""

    def callback(request):
        q = parse_qs(urlparse(request.url).query)
        if q.get("returnCountOnly") == ["true"]:
            seen.append("count")
            return 200, {}, json.dumps({"count": count})
        offset = int(q["resultOffset"][0])
        seen.append(offset)
        return 200, {}, json.dumps(pages[offset])

    responses.add_callback(
        responses.GET, dc.LAYER_URL, callback=callback, content_type="application/json"
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(dc.time, "sleep", lambda _s: None)


@responses.activate
def test_download_dc_pages_while_transfer_limit_exceeded(tmp_path):
    """The transfer-limit flag drives paging: a full last page without it still ends the run."""
    pages = {0: _page([1, 2], True), 2: _page([3, 4], True), 4: _page([5, 6], False)}
    seen: list = []
    _register_layer(pages, count=6, seen=seen)

    dc.download_dc(tmp_path, start_date="2019-01-01", page_size=2)

    assert seen == [0, 2, 4, "count"]
    assert sorted(p.name for p in tmp_path.glob("page_*.json")) == [
        "page_0000000.json",
        "page_0000002.json",
        "page_0000004.json",
    ]
    assert list(tmp_path.glob("*.tmp")) == []
    meta = json.loads((tmp_path / dc.META_NAME).read_text())
    assert meta["start_date"] == "2019-01-01"
    assert meta["total_count"] == 6
    assert meta["page_size"] == 2
    assert meta["layer_url"] == dc.LAYER_URL
    assert meta["where"] == build_query_params("2019-01-01", 0)["where"]
    assert meta["fetched_at"]


@responses.activate
def test_download_dc_raises_when_payload_has_no_features(tmp_path):
    responses.add(responses.GET, dc.LAYER_URL, json={"objectIdFieldName": "OBJECTID"}, status=200)
    with pytest.raises(RuntimeError, match="features"):
        dc.download_dc(tmp_path, start_date="2019-01-01", page_size=2)


@responses.activate
def test_download_dc_retries_error_payload_then_raises(tmp_path):
    responses.add(responses.GET, dc.LAYER_URL, json={"error": {"code": 500}}, status=200)
    with pytest.raises(RuntimeError, match="offset 0"):
        dc.download_dc(tmp_path, start_date="2019-01-01", page_size=2)
    assert len(responses.calls) == dc.RETRIES
    assert list(tmp_path.glob("*")) == []


@responses.activate
def test_download_dc_does_not_retry_client_errors(tmp_path):
    responses.add(responses.GET, dc.LAYER_URL, json={"error": "nope"}, status=404)
    with pytest.raises(RuntimeError):
        dc.download_dc(tmp_path, start_date="2019-01-01", page_size=2)
    assert len(responses.calls) == 1


@responses.activate
def test_download_dc_resume_skips_existing_page(tmp_path):
    (tmp_path / "page_0000000.json").write_text(json.dumps(_page([1, 2], False)))
    seen: list = []
    _register_layer({}, count=2, seen=seen)

    dc.download_dc(tmp_path, start_date="2019-01-01", page_size=2)

    assert seen == ["count"]  # no page was refetched


@responses.activate
def test_download_dc_refetches_corrupt_page(tmp_path):
    page_path = tmp_path / "page_0000000.json"
    page_path.write_text("{truncated")
    seen: list = []
    _register_layer({0: _page([1, 2], False)}, count=2, seen=seen)

    dc.download_dc(tmp_path, start_date="2019-01-01", page_size=2)

    assert seen == [0, "count"]
    assert json.loads(page_path.read_text())["features"][0]["attributes"]["OBJECTID"] == 1


@responses.activate
def test_download_dc_raises_on_count_mismatch(tmp_path):
    seen: list = []
    _register_layer({0: _page([1, 2], False)}, count=99, seen=seen)
    with pytest.raises(RuntimeError, match="99"):
        dc.download_dc(tmp_path, start_date="2019-01-01", page_size=2)


@responses.activate
def test_download_dc_refuses_to_mix_windows(tmp_path):
    seen: list = []
    _register_layer({0: _page([1, 2], False)}, count=2, seen=seen)
    dc.download_dc(tmp_path, start_date="2019-01-01", page_size=2)

    with pytest.raises(RuntimeError, match="clear"):
        dc.download_dc(tmp_path, start_date="2020-01-01", page_size=2)


def test_build_processed_raises_on_page_without_features(tmp_path):
    (tmp_path / "page_0000000.json").write_text(json.dumps({"objectIdFieldName": "OBJECTID"}))
    with pytest.raises(ValueError, match="features"):
        dc.build_processed(tmp_path, tmp_path / "out.parquet")


def test_build_processed_raises_on_empty_raw_dir(tmp_path):
    with pytest.raises(ValueError):
        dc.build_processed(tmp_path, tmp_path / "out.parquet")


def test_build_processed_is_stably_sorted_and_microsecond_dated(tmp_path):
    (tmp_path / "page_0000000.json").write_text(json.dumps(_page([2, 1], False)))
    df = dc.build_processed(tmp_path, tmp_path / "out.parquet")
    assert df["date"].dtype == "datetime64[us]"
    assert df["txn_id"].tolist() == ["dc:1", "dc:2"]  # tie on date broken by txn_id
    assert pd.read_parquet(tmp_path / "out.parquet")["date"].dtype == "datetime64[us]"


def test_validate_processed_rejects_empty_frame():
    df = features_to_frame([])
    with pytest.raises(ValueError, match="empty"):
        validate_processed(df)


def test_validate_processed_rejects_unknown_source_and_blank_merchant():
    df = features_to_frame(FAKE_FEATURES)
    bad_source = df.copy()
    bad_source.loc[0, "source"] = "nope"
    with pytest.raises(ValueError, match="source"):
        validate_processed(bad_source)

    blank = df.copy()
    blank.loc[0, "raw_merchant"] = "   "
    with pytest.raises(ValueError, match="raw_merchant"):
        validate_processed(blank)
