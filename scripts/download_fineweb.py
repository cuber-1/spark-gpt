#!/usr/bin/env python3
"""Download a pinned, resumable local FineWeb-Edu snapshot."""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

DEFAULT_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", default="data/source/fineweb-edu-10BT")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    path = snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=["README.md", "sample/10BT/*.parquet"],
        local_dir=args.output,
        max_workers=args.workers,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
