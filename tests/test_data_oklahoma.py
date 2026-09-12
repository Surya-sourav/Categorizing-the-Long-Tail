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
