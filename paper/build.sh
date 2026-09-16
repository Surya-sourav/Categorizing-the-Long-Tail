#!/usr/bin/env bash
# Build the preprint reproducibly: the same sources always produce a byte-identical PDF.
# SOURCE_DATE_EPOCH pins the timestamp LaTeX embeds in the PDF metadata; the title-page date is
# pinned separately in main.tex. Regenerate the tables first if results/ has changed:
#   .venv/bin/python scripts/make_paper_tables.py
set -euo pipefail
cd "$(dirname "$0")"
SOURCE_DATE_EPOCH=1789560000 tectonic -X compile main.tex "$@"
