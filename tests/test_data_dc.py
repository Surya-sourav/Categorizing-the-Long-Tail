import pytest

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
