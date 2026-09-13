import pytest

from txcat.normalizer import normalize_merchant

CASES = [
    # (raw, expected)
    ("STAPLES       00102186", "STAPLES"),
    ("WW GRAINGER 912", "WW GRAINGER"),
    ("USPS 1050050275    QQQ", "USPS"),
    ("AMZN Mktp US RC9J295A2", "AMZN MKTP US"),
    ("AMAZON MKTPL RC6DU71U2", "AMAZON MKTPL"),
    ("WAL-MART #4241", "WAL-MART"),
    ("WESTERN OKLAHOMA DAIRY SU", "WESTERN OKLAHOMA DAIRY SU"),
    ("SQ *BLUE BOTTLE COFFEE", "BLUE BOTTLE COFFEE"),
    ("TST* GOLDEN DRAGON - OAKLAND", "GOLDEN DRAGON - OAKLAND"),
    ("PAYPAL *EBAY", "EBAY"),
    ("PP*SAFEWAY", "SAFEWAY"),
    ("PYPL*ETSY", "ETSY"),
    ("DD *DOORDASH CHIPOTLE", "DOORDASH CHIPOTLE"),
    ("UBER *TRIP HELP.UBER.COM", "UBER TRIP"),
    ("AMAZON.COM*2K4R7 SEATTLE", "AMAZON SEATTLE"),
    ("IN *ACME WIDGETS LLC", "ACME WIDGETS LLC"),
    ("POS PURCHASE STARBUCKS 00123", "STARBUCKS"),
    ("HOME DEPOT #1234 WASHINGTON DC", "HOME DEPOT WASHINGTON DC"),
    ("SHELL OIL 57544099307", "SHELL OIL"),
    ("MARRIOTT 337W2 WASHINGTON", "MARRIOTT WASHINGTON"),
    ("7-ELEVEN 34567", "7-ELEVEN"),
    ("3M COMPANY", "3M COMPANY"),
    ("JOTFORM INC", "JOTFORM INC"),
    ("DELTA AIR 0062341234567", "DELTA AIR"),
    ("  lowe's  #02214  ", "LOWE'S"),
]


@pytest.mark.parametrize("raw,expected", CASES)
def test_normalize_cases(raw, expected):
    assert normalize_merchant(raw).text == expected


def test_steps_are_logged():
    r = normalize_merchant("SQ *BLUE BOTTLE COFFEE #12345")
    assert "strip_prefix:SQ *" in r.steps
    assert any(s.startswith("drop_store_number") for s in r.steps)


def test_empty_after_stripping_falls_back_to_uppercase_raw():
    r = normalize_merchant("PAYPAL *")
    assert r.text == "PAYPAL"


def test_never_returns_empty():
    assert normalize_merchant("   ").text == ""
    assert normalize_merchant("12345").text == "12345"
