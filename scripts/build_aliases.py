"""Build data/taxonomy/mcc_description_aliases.csv from observed descriptions in processed data."""

import pandas as pd

from txcat.data.taxonomy import match_description, norm_desc

src = pd.read_csv("data/taxonomy/mcc_codes_source.csv", dtype={"mcc": int})
# Hand-written aliases for descriptions the automatic matcher cannot resolve (abbreviated MCC text
# such as "TRANSPRTN-SUBRBN + LOCAL COMTR PSNGR", and hotel/airline brand names mapped to the
# generic MCC of the same category). Committed and reviewed; keyed on norm_desc so spelling
# variants across sources match.
manual = pd.read_csv("data/taxonomy/manual_aliases.csv")
manual_map = {
    norm_desc(d): int(m)
    for d, m in zip(manual["observed_description"], manual["mcc"], strict=True)
}
rows, unmatched = [], []
for name in ["dc", "oklahoma"]:
    df = pd.read_parquet(f"data/processed/{name}.parquet")
    counts = df.groupby("mcc_description").size().sort_values(ascending=False)
    for desc, n in counts.items():
        mcc, how = match_description(desc, src)
        if mcc is None and norm_desc(desc) in manual_map:
            mcc, how = manual_map[norm_desc(desc)], "manual"
        if mcc is None:
            unmatched.append({"source": df["source"].iloc[0], "observed_description": desc,
                              "n_rows": n, "mcc": ""})
        else:
            rows.append({"source": df["source"].iloc[0], "observed_description": desc,
                         "mcc": mcc, "match": how, "n_rows": n})
pd.DataFrame(rows).to_csv("data/taxonomy/mcc_description_aliases.csv", index=False)
pd.DataFrame(unmatched).to_csv("data/taxonomy/unmatched_descriptions.csv", index=False)
matched_rows = sum(r["n_rows"] for r in rows)
un_rows = sum(u["n_rows"] for u in unmatched)
print(f"aliases: {len(rows)} matched descriptions covering {matched_rows} rows; "
      f"{len(unmatched)} unmatched covering {un_rows} rows "
      f"({100*un_rows/(matched_rows+un_rows):.2f}%)")
