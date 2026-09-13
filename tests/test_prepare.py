import pandas as pd

from txcat.data.prepare import prepare


def test_prepare_adds_merchant_and_labels(tmp_path):
    (tmp_path / "mcc_to_category.csv").write_text(
        "mcc,mcc_description,category,ambiguous,notes\n5943,Stationery,office_supplies,0,\n"
    )
    (tmp_path / "mcc_description_aliases.csv").write_text(
        "source,observed_description,mcc,match\ndc,Stationery Stores,5943,exact\n"
    )
    df = pd.DataFrame(
        {
            "txn_id": ["dc:1"],
            "date": pd.to_datetime(["2024-01-01"]),
            "raw_merchant": ["STAPLES       00102186"],
            "mcc_description": ["Stationery Stores"],
            "source": ["dc"],
        }
    )
    out = prepare(df, tmp_path)
    assert out.loc[0, "merchant"] == "STAPLES"
    assert out.loc[0, "category"] == "office_supplies" and out.loc[0, "ambiguous"] == 0
