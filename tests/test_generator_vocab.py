
from txcat.generator.vocab import items_to_vocab, load_tag_map

FAKE_NSI = {
    "nsi": {
        "brands/shop/supermarket": {
            "items": [
                {
                    "displayName": "Safeway",
                    "id": "safeway-1",
                    "locationSet": {"include": ["us"]},
                    "tags": {
                        "brand": "Safeway",
                        "name": "Safeway",
                        "shop": "supermarket",
                        "brand:wikidata": "Q1",
                    },
                },
                {
                    "displayName": "Tesco",
                    "id": "tesco-1",
                    "locationSet": {"include": ["gb"]},
                    "tags": {"brand": "Tesco", "name": "Tesco", "shop": "supermarket"},
                },
            ]
        },
        "brands/amenity/cafe": {
            "items": [
                {
                    "displayName": "Starbucks",
                    "id": "sb-1",
                    "locationSet": {"include": ["001"]},
                    "tags": {"brand": "Starbucks", "name": "Starbucks", "amenity": "cafe"},
                }
            ]
        },
        "brands/shop/unknown_thing": {
            "items": [
                {
                    "displayName": "Mystery",
                    "id": "m-1",
                    "locationSet": {"include": ["us"]},
                    "tags": {"brand": "Mystery", "name": "Mystery", "shop": "unknown_thing"},
                }
            ]
        },
        "operators/amenity/post_office": {
            "items": [
                {
                    "displayName": "USPS",
                    "id": "usps-1",
                    "locationSet": {"include": ["us"]},
                    "tags": {"operator": "USPS", "name": "USPS", "amenity": "post_office"},
                }
            ]
        },
    }
}


def test_items_to_vocab_filters_us_and_maps_categories(tmp_path):
    tm = tmp_path / "osm_tag_to_category.csv"
    tm.write_text(
        "osm_key,osm_value,category\nshop,supermarket,groceries\namenity,cafe,restaurants\n"
    )
    v = items_to_vocab(FAKE_NSI, load_tag_map(tm))
    assert list(v.columns) == [
        "merchant_id",
        "name",
        "osm_key",
        "osm_value",
        "category",
        "wikidata",
    ]
    assert set(v["name"]) == {
        "Safeway",
        "Starbucks",
    }  # Tesco not US, Mystery unmapped, USPS not brands/
    assert v.set_index("name").loc["Safeway", "category"] == "groceries"
    assert v.set_index("name").loc["Safeway", "wikidata"] == "Q1"
