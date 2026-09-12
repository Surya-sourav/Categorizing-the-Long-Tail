import pandas as pd

from txcat.data.taxonomy import attach_labels, match_description, norm_desc

SOURCE = pd.DataFrame({
    "mcc": [5411, 5943, 3503, 5999],
    "edited_description": ["Grocery Stores, Supermarkets",
                           "Stationery Stores, Office and School Supply Stores",
                           "Sheraton Hotels", "Miscellaneous and Specialty Retail Stores"],
    "combined_description": ["Grocery Stores, Supermarkets",
                             "Stationery, Office & School Supply Stores",
                             "Sheraton", "Miscellaneous Retail"],
})


def test_norm_desc():
    assert norm_desc("GROCERY STORES  SUPERMARKETS") == "GROCERY STORES SUPERMARKETS"
    assert norm_desc("Stationery, Office & School Supply Stores") == (
        "STATIONERY OFFICE SCHOOL SUPPLY STORES")


def test_match_exact_then_fuzzy():
    assert match_description("GROCERY STORES  SUPERMARKETS", SOURCE) == (5411, "exact")
    assert match_description("Stationery, Office & School Supply Stores", SOURCE) == (5943, "exact")
    assert match_description("SHERATON", SOURCE) == (3503, "exact")
    mcc, how = match_description("Stationery Office School Supply Store", SOURCE)
    assert mcc == 5943 and how == "fuzzy"
    assert match_description("TOTALLY UNKNOWN THING", SOURCE) == (None, "none")


def test_attach_labels(tmp_path):
    (tmp_path / "mcc_to_category.csv").write_text(
        "mcc,mcc_description,category,ambiguous,notes\n"
        "5411,Grocery,groceries,0,\n5999,Misc,AMBIGUOUS,1,x\n")
    (tmp_path / "mcc_description_aliases.csv").write_text(
        "source,observed_description,mcc,match\n"
        "dc,GROCERY STORES,5411,exact\ndc,MISC RETAIL,5999,exact\n")
    df = pd.DataFrame({"mcc_description": ["GROCERY STORES", "MISC RETAIL", "???"],
                       "source": ["dc"] * 3})
    out = attach_labels(df, tmp_path)
    assert list(out["category"][:2]) == ["groceries", "AMBIGUOUS"] and pd.isna(
        out["category"].iloc[2])
    assert list(out["ambiguous"]) == [0, 1, 1]
    assert list(out["mcc"].fillna(-1).astype(int)) == [5411, 5999, -1]
