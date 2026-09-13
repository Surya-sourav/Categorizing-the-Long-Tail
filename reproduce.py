"""Regenerate every table and figure from committed caches. Zero live API calls. Fails loudly on cache misses.

python reproduce.py --config configs/dc.yaml --seeds 42 43 44 --output results/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from loguru import logger

from txcat.config import load_config
from txcat.manifest import build_manifest

STEPS = [
    ["experiments.exp0_data_audit"],
    ["experiments.exp1_tail_characterization", "--mode", "reproduce"],
    ["experiments.exp1_generator_sweep", "--mode", "reproduce"],
    ["experiments.exp2_fallback_comparison", "--mode", "reproduce"],
    ["experiments.exp3_streaming_convergence"],
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dc.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--output", default="results")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = [str(s) for s in (args.seeds or cfg.seeds)]
    for req in (
        "prompts/PROMPT_HASHES.json",
        "prompts/model_versions.json",
        "cache/web_search",
        "cache/llm",
        "cache/embeddings",
    ):
        if not Path(req).exists():
            sys.exit(
                f"missing required artifact: {req} (caches must be present; no live calls are made)"
            )
    frozen = json.loads(Path("prompts/PROMPT_HASHES.json").read_text())
    for p, h in frozen.items():
        from txcat.utils import sha256_text

        if sha256_text(Path(p).read_text()) != h:
            sys.exit(
                f"prompt {p} differs from frozen hash; refusing to reproduce with an edited prompt"
            )
    manifest = build_manifest(
        list(frozen), cfg.model_dump(), json.loads(Path("prompts/model_versions.json").read_text())
    )
    Path(args.output, "logs").mkdir(parents=True, exist_ok=True)
    Path(args.output, "logs", f"manifest_{manifest['timestamp'].replace(':', '')}.json").write_text(
        json.dumps(manifest, indent=2)
    )
    logger.info(
        f"git {manifest['git_commit'][:10]} | {manifest['host']['platform']} | prompts {list(frozen)}"
    )
    for step in STEPS:
        cmd = [sys.executable, "-m", *step, "--config", args.config]
        if step[0] != "experiments.exp0_data_audit":
            cmd += ["--seeds", *seeds]
        logger.info("RUN " + " ".join(cmd))
        subprocess.run(cmd, check=True)
    logger.info("reproduction complete")


if __name__ == "__main__":
    main()
