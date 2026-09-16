# Paper

`main.tex` is the preprint. Build it with [tectonic](https://tectonic-typesetting.github.io/)
(single binary, fetches what it needs):

```bash
brew install tectonic          # or: cargo install tectonic
./paper/build.sh               # byte-identical output on every run
```

`build.sh` pins the timestamp LaTeX embeds in the PDF so the same sources always produce the same
bytes; the title-page date is pinned separately in `main.tex`. Bump both when you post a revision.
A plain `tectonic -X compile paper/main.tex` works too, it just produces a PDF that differs on
every build.

Any TeX Live distribution works too (`pdflatex main && bibtex main && pdflatex main && pdflatex main`).

## Where the numbers come from

Nothing in the paper is typed in by hand from a result. `tables/*.tex` is generated from the
committed CSVs in `results/tables/` by:

```bash
.venv/bin/python scripts/make_paper_tables.py
```

Regenerate them after any rerun of `reproduce.py`, and the paper follows the results automatically.
Figures are included straight from `results/figures/*.pdf`. Numbers that appear in prose (the
abstract, the introduction, section text) are typed, so if an experiment changes, re-check prose
numbers against the regenerated tables.

## Before posting

- **Verify every citation in `refs.bib`.** The entries were written from memory, not from a
  citation database. Check each author list, title, venue, year, and arXiv id against the actual
  paper. The transaction-categorization entry is the least certain and may need replacing.
- Re-read the limitations section against the final experiment set.
- Confirm the author block: Surya Parida, Independent Researcher. No employer affiliation anywhere.
