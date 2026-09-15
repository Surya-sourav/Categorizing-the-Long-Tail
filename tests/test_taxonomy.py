import pandas as pd

from txcat.data.taxonomy import AMBIGUOUS_MCCS, CATEGORIES, categorize_mcc


def test_fifteen_categories():
    assert len(CATEGORIES) == 15
    assert "AMBIGUOUS" not in CATEGORIES


def test_spot_checks():
    assert categorize_mcc(5411) == ("groceries", 0)
    assert categorize_mcc(5812) == ("restaurants", 0)
    assert categorize_mcc(7011) == ("lodging", 0)
    assert categorize_mcc(3501) == ("lodging", 0)  # Holiday Inn brand MCC
    assert categorize_mcc(3000) == ("airlines_travel", 0)  # United brand MCC
    assert categorize_mcc(3351) == ("transport_auto_fuel", 0)  # car rental brand MCC
    assert categorize_mcc(5734) == ("software_electronics", 0)
    assert categorize_mcc(4814) == ("telecom_utilities", 0)
    assert categorize_mcc(5943) == ("office_supplies", 0)
    assert categorize_mcc(5085) == ("industrial_hardware", 0)
    assert categorize_mcc(8062) == ("health", 0)
    assert categorize_mcc(8220) == ("education_gov_membership", 0)
    assert categorize_mcc(9402) == ("financial_postal_shipping", 0)
    assert categorize_mcc(7829) == ("entertainment_media", 0)
    for m in (5999, 5399, 7399, 8999):
        cat, amb = categorize_mcc(m)
        assert amb == 1 and m in AMBIGUOUS_MCCS


def test_every_source_mcc_is_mapped():
    src = pd.read_csv("data/taxonomy/mcc_codes_source.csv", dtype={"mcc": str})
    for m in src["mcc"]:
        cat, amb = categorize_mcc(int(m))
        assert cat in CATEGORIES or (cat == "AMBIGUOUS" and amb == 1)
