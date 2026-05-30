#!/usr/bin/env python3
"""Compare deterministic MPR smoke generation outputs."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare generated_token_ids from two mpr_baseline_qwen3_8b.py logs."
        )
    )
    parser.add_argument("baseline_log", type=Path)
    parser.add_argument("mpr_log", type=Path)
    return parser.parse_args()


def extract_generated_token_ids(path: Path) -> list[int]:
    """Extract the first generated_token_ids list from a smoke log."""
    if not path.exists():
        raise AssertionError(f"Log path does not exist: {path}")

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("generated_token_ids:"):
            continue
        _, value = line.split(":", maxsplit=1)
        token_ids = ast.literal_eval(value.strip())
        if not isinstance(token_ids, list) or not all(
            isinstance(token_id, int) for token_id in token_ids
        ):
            raise AssertionError(
                f"{path}: generated_token_ids must be a list of ints."
            )
        return token_ids

    raise AssertionError(f"{path}: generated_token_ids line was not found.")


def compare_generated_token_ids(
    baseline_path: Path,
    mpr_path: Path,
) -> tuple[list[int], list[int]]:
    """Return matching token ID lists or raise with the first mismatch."""
    baseline_ids = extract_generated_token_ids(baseline_path)
    mpr_ids = extract_generated_token_ids(mpr_path)
    if baseline_ids != mpr_ids:
        first_mismatch = next(
            (
                idx
                for idx, (baseline_id, mpr_id) in enumerate(zip(baseline_ids, mpr_ids))
                if baseline_id != mpr_id
            ),
            min(len(baseline_ids), len(mpr_ids)),
        )
        raise AssertionError(
            "generated_token_ids differ: "
            f"baseline_len={len(baseline_ids)}, mpr_len={len(mpr_ids)}, "
            f"first_mismatch_index={first_mismatch}."
        )
    return baseline_ids, mpr_ids


def main() -> None:
    """Compare two smoke logs and print a compact result."""
    args = parse_args()
    baseline_ids, _ = compare_generated_token_ids(
        args.baseline_log,
        args.mpr_log,
    )
    print("MPR generation output comparison passed")
    print(f"generated_token_count: {len(baseline_ids)}")


if __name__ == "__main__":
    main()
