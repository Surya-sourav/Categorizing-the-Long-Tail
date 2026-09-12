"""Generate data/taxonomy/mcc_to_category.csv from the rules in txcat.data.taxonomy."""

from pathlib import Path

import pandas as pd

from txcat.data.taxonomy import categorize_mcc

src = pd.read_csv("data/taxonomy/mcc_codes_source.csv", dtype={"mcc": str})
rows = []
for _, r in src.iterrows():
    mcc = int(r["mcc"])
    cat, amb = categorize_mcc(mcc)
    note = "NEC / we-don't-know code" if amb else ""
    rows.append({"mcc": mcc, "mcc_description": r["edited_description"], "category": cat,
                 "ambiguous": amb, "notes": note})
out = pd.DataFrame(rows).sort_values("mcc")
Path("data/taxonomy").mkdir(parents=True, exist_ok=True)
out.to_csv("data/taxonomy/mcc_to_category.csv", index=False)
print(out["category"].value_counts())
