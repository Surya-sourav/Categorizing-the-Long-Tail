import io

import pandas as pd
import pytest
import responses
from loguru import logger

from txcat.data import oklahoma
from txcat.data.oklahoma import csv_to_frame
from txcat.data.schema import PROCESSED_COLUMNS, validate_processed

CSV = (
    "CALENDAR_YEAR,CALENDAR_MONTH,AGENCYNBR,AGENCYNAME,LAST_NAME,FIRST_INITIAL,ITEM_DESCR,AMOUNT"
    ",MERCHANT,TRANSACTION_DATE,POST_DATE,MCC_DESCRIPTION,ROWID\n"
    "2024,07, 01000,OKLAHOMA STATE UNIVERSITY,Tivis,J,Bar Audio Wired In-Ear Hea PCE,24.99,"
    "AMZN Mktp US RC9J295A2,29-Jun-24,01-Jul-24,BOOK STORES,AAAJGhAANAANWuXAAO\n"
    "2024,07, 01000,OKLAHOMA STATE UNIVERSITY,Wadley,M,GENERAL PURCHASE,49.98,"
    "WAL-MART #4241,28-Jun-24,01-Jul-24,GROCERY STORES  SUPERMARKETS,AAAJGhAANAANWuXAAU\n"
    "2024,07, 01000,OKLAHOMA STATE UNIVERSITY,Nobody,X,,1.00,"
    ",28-Jun-24,01-Jul-24,X,AAAJGhAANAANWuXAAZ\n"
)


def test_csv_to_frame_keeps_only_processed_columns():
    df = csv_to_frame(io.StringIO(CSV))
    assert list(df.columns) == PROCESSED_COLUMNS
    assert len(df) == 2
    assert df.loc[0, "txn_id"] == "ok:AAAJGhAANAANWuXAAO"
    assert df.loc[0, "raw_merchant"] == "AMZN Mktp US RC9J295A2"
    assert str(df.loc[0, "date"].date()) == "2024-06-29"
    assert df.loc[1, "mcc_description"] == "GROCERY STORES  SUPERMARKETS"
    validate_processed(df)


CSV_NO_ROWID = (
    "CALENDAR_YEAR,CALENDAR_MONTH,AGENCYNBR,AGENCYNAME,LAST_NAME,FIRST_INITIAL,ITEM_DESCR,AMOUNT"
    ",MERCHANT,TRANSACTION_DATE,POST_DATE,MCC_DESCRIPTION\n"
    "2023,8,1000,OKLAHOMA STATE UNIVERSITY,Caselman,K,6900 LARGE FULL FACEPIECE EA,198.45,"
    "DXP ENTERPRISES,3-Aug-23,4-Aug-23,INDUSTRIAL SUPPLIES\n"
    "2023,8,1000,OKLAHOMA STATE UNIVERSITY,Nobody,X,,1.00,"
    ",3-Aug-23,4-Aug-23,X\n"
    "2023,8,1000,OKLAHOMA STATE UNIVERSITY,Smith,A,GENERAL PURCHASE,5.00,"
    "WAL-MART #4241,4-Aug-23,7-Aug-23,GROCERY STORES\n"
)


def test_csv_to_frame_synthesizes_ids_when_rowid_missing():
    """Two 2023 monthly files ship without ROWID; ids fall back to file id + source row."""
    df = csv_to_frame(io.StringIO(CSV_NO_ROWID), file_id="PCard_Public_202308")
    assert list(df.columns) == PROCESSED_COLUMNS
    assert len(df) == 2
    assert df.loc[0, "txn_id"] == "ok:PCard_Public_202308:row0"
    # source row position, not renumbered after filtering:
    assert df.loc[1, "txn_id"] == "ok:PCard_Public_202308:row2"
    assert df.loc[1, "raw_merchant"] == "WAL-MART #4241"
    validate_processed(df)


def test_csv_to_frame_tolerates_utf8_bom():
    """One monthly file is UTF-8 with a BOM on the header line."""
    df = csv_to_frame(io.StringIO("﻿" + CSV))
    assert list(df.columns) == PROCESSED_COLUMNS
    assert df.loc[0, "txn_id"] == "ok:AAAJGhAANAANWuXAAO"


def test_file_id_namespaces_rowids_reused_across_months():
    """ROWID is an Oracle rowid the publisher reuses month to month; ids must not collide."""
    a = csv_to_frame(io.StringIO(CSV), file_id="PCard_Public_202207")
    b = csv_to_frame(io.StringIO(CSV), file_id="PCard_Public_202208")
    assert a.loc[0, "txn_id"] == "ok:PCard_Public_202207:AAAJGhAANAANWuXAAO"
    assert not set(a["txn_id"]) & set(b["txn_id"])


# --- download layer (mocked HTTP) -------------------------------------------------------------

PKG = "purchase-card-pcard-fiscal-year-2024"
RES_URL = "https://data.ok.gov/dataset/x/resource/y/PCard_Public_202407.csv"


def _register_package(resources: list[dict]) -> None:
    responses.add(
        responses.GET,
        oklahoma.CKAN_SHOW.format(pkg=PKG),
        json={"result": {"resources": resources}},
        status=200,
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(oklahoma.time, "sleep", lambda _s: None)


def test_csv_to_frame_raises_on_missing_required_columns():
    headerless = CSV.replace("MERCHANT,TRANSACTION_DATE", "SELLER,TRANSACTION_DATE", 1)
    with pytest.raises(ValueError, match="MERCHANT"):
        csv_to_frame(io.StringIO(headerless))


@responses.activate
def test_download_oklahoma_skips_existing_file(tmp_path):
    _register_package([{"name": "PCard_Public_202407.csv", "format": "CSV", "url": RES_URL}])
    (tmp_path / "PCard_Public_202407.csv").write_text(CSV)

    oklahoma.download_oklahoma(tmp_path, [PKG])

    assert len(responses.calls) == 1  # only the package_show call


@responses.activate
def test_download_oklahoma_raises_on_truncated_body(tmp_path):
    """A body shorter than Content-Length must not leave a file that looks complete."""
    _register_package([{"name": "PCard_Public_202407.csv", "format": "CSV", "url": RES_URL}])
    responses.add(
        responses.GET,
        RES_URL,
        body=CSV,
        status=200,
        headers={"Content-Length": str(len(CSV.encode()) + 500)},
    )
    with pytest.raises(RuntimeError, match="PCard_Public_202407"):
        oklahoma.download_oklahoma(tmp_path, [PKG])
    assert list(tmp_path.iterdir()) == []  # no partial file, no leftover tmp


def test_write_bytes_atomic_rejects_short_write(tmp_path):
    """Guards the disk side: a body that does not match Content-Length is never promoted."""
    dest = tmp_path / "PCard_Public_202407.csv"
    with pytest.raises(RuntimeError, match="Content-Length"):
        oklahoma._write_bytes_atomic(dest, b"hello", expected_size=500)
    assert list(tmp_path.iterdir()) == []


def test_write_bytes_atomic_promotes_complete_write(tmp_path):
    dest = tmp_path / "PCard_Public_202407.csv"
    oklahoma._write_bytes_atomic(dest, b"hello", expected_size=5)
    assert dest.read_bytes() == b"hello"
    assert list(tmp_path.glob("*.tmp")) == []


@responses.activate
def test_download_oklahoma_writes_complete_file(tmp_path):
    _register_package([{"name": "PCard_Public_202407.csv", "format": "CSV", "url": RES_URL}])
    responses.add(responses.GET, RES_URL, body=CSV, status=200)

    oklahoma.download_oklahoma(tmp_path, [PKG])

    assert (tmp_path / "PCard_Public_202407.csv").read_text() == CSV
    assert list(tmp_path.glob("*.tmp")) == []


@responses.activate
def test_download_oklahoma_does_not_retry_client_errors(tmp_path):
    _register_package([{"name": "PCard_Public_202407.csv", "format": "CSV", "url": RES_URL}])
    responses.add(responses.GET, RES_URL, body="gone", status=404)
    with pytest.raises(RuntimeError):
        oklahoma.download_oklahoma(tmp_path, [PKG])
    assert len(responses.calls) == 2  # package_show + one non-retried fetch


@responses.activate
def test_download_oklahoma_retries_rate_limits(tmp_path):
    _register_package([{"name": "PCard_Public_202407.csv", "format": "CSV", "url": RES_URL}])
    responses.add(responses.GET, RES_URL, body="slow down", status=429)
    responses.add(responses.GET, RES_URL, body=CSV, status=200)

    oklahoma.download_oklahoma(tmp_path, [PKG])

    assert (tmp_path / "PCard_Public_202407.csv").read_text() == CSV


def test_build_processed_raises_on_empty_raw_dir(tmp_path):
    with pytest.raises(ValueError):
        oklahoma.build_processed(tmp_path, tmp_path / "out.parquet")


def test_build_processed_is_stably_sorted_and_microsecond_dated(tmp_path):
    (tmp_path / "PCard_Public_202407.csv").write_text(CSV)
    df = oklahoma.build_processed(tmp_path, tmp_path / "out.parquet")
    assert df["date"].dtype == "datetime64[us]"
    assert df["date"].is_monotonic_increasing
    assert pd.read_parquet(tmp_path / "out.parquet")["date"].dtype == "datetime64[us]"


def test_rowid_fallback_warns_once():
    """The synthesized-id path is loud, once per file, so it cannot go unnoticed."""
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        csv_to_frame(io.StringIO(CSV_NO_ROWID), file_id="PCard_Public_202308")
    finally:
        logger.remove(sink_id)
    assert sum("ROWID" in m for m in messages) == 1
    assert sum("PCard_Public_202308" in m for m in messages) == 1
