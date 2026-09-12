import io

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
