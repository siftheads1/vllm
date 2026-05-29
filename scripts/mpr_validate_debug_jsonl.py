#!/usr/bin/env python3
"""Validate MPR debug JSONL records from a Step 1.3 smoke run."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the JSONL validator."""
    parser = argparse.ArgumentParser(
        description="Validate MPR observe_kv_write/digest_created debug JSONL."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "JSONL files or glob patterns. Defaults to "
            "/tmp/vllm_mpr_debug/*.jsonl."
        ),
    )
    parser.add_argument(
        "--min-digest-events",
        type=int,
        default=1,
        help="Minimum number of digest_created events required.",
    )
    parser.add_argument(
        "--min-score-events",
        type=int,
        default=0,
        help="Minimum number of score_estimated events required.",
    )
    parser.add_argument(
        "--allow-unmatched-digest-events",
        action="store_true",
        help=(
            "Do not fail when a digest_created event has no matching "
            "observe_kv_write record. This can happen with restrictive "
            "VLLM_MPR_MAX_STEPS or VLLM_MPR_DUMP_EVERY settings."
        ),
    )
    parser.add_argument(
        "--show",
        type=int,
        default=10,
        help="Number of digest_created rows to print in the summary.",
    )
    return parser.parse_args()


def expand_paths(patterns: list[str]) -> list[Path]:
    """Expand input files and glob patterns into sorted JSONL paths."""
    if not patterns:
        patterns = ["/tmp/vllm_mpr_debug/*.jsonl"]

    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))

    return sorted(set(paths))


def load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    """Load JSONL records and annotate each event with source location."""
    events: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise AssertionError(f"JSONL path does not exist: {path}")
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AssertionError(
                        f"Invalid JSON in {path}:{line_no}: {exc}"
                    ) from exc
                if not isinstance(event, dict):
                    raise AssertionError(
                        f"JSONL record must be an object: {path}:{line_no}"
                    )
                event["_source"] = f"{path}:{line_no}"
                events.append(event)
    return events


def require_int(
    event: dict[str, Any],
    field: str,
    *,
    minimum: int | None = None,
) -> int:
    """Read an integer field and optionally enforce a minimum value."""
    value = event.get(field)
    if not isinstance(value, int):
        raise AssertionError(f"{event['_source']}: {field} must be an int.")
    if minimum is not None and value < minimum:
        raise AssertionError(
            f"{event['_source']}: {field} must be >= {minimum}, got {value}."
        )
    return value


def require_shape(event: dict[str, Any], field: str) -> list[int]:
    """Read a two-dimensional positive tensor shape field."""
    value = event.get(field)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(dim, int) and dim > 0 for dim in value)
    ):
        raise AssertionError(
            f"{event['_source']}: {field} must be [num_kv_heads, head_dim]."
        )
    return value


def validate_observe_event(event: dict[str, Any]) -> None:
    """Validate one observe_kv_write event's block/slot invariants."""
    block_size = require_int(event, "block_size", minimum=1)
    require_int(event, "num_slots", minimum=0)
    require_int(event, "num_valid_slots", minimum=0)
    require_int(event, "num_pad_slots", minimum=0)

    unique_block_ids = event.get("unique_block_ids")
    if not isinstance(unique_block_ids, list):
        raise AssertionError(f"{event['_source']}: unique_block_ids must be a list.")
    if not all(isinstance(block_id, int) and block_id >= 0
               for block_id in unique_block_ids):
        raise AssertionError(
            f"{event['_source']}: unique_block_ids must contain non-negative ints."
        )

    min_block_offset = event.get("min_block_offset")
    max_block_offset = event.get("max_block_offset")
    if min_block_offset is None or max_block_offset is None:
        if min_block_offset is not None or max_block_offset is not None:
            raise AssertionError(
                f"{event['_source']}: min/max block offsets must both be null "
                "or both be ints."
            )
        return

    if not isinstance(min_block_offset, int) or not isinstance(max_block_offset, int):
        raise AssertionError(
            f"{event['_source']}: min/max block offsets must be ints or null."
        )
    if not (0 <= min_block_offset <= max_block_offset < block_size):
        raise AssertionError(
            f"{event['_source']}: invalid block offset range "
            f"[{min_block_offset}, {max_block_offset}] for block_size={block_size}."
        )


def validate_digest_event(event: dict[str, Any]) -> None:
    """Validate one digest_created event's digest metadata."""
    block_size = require_int(event, "block_size", minimum=1)
    valid_token_count = require_int(event, "valid_token_count", minimum=1)
    require_int(event, "physical_block_id", minimum=0)
    require_int(event, "num_digest_blocks_for_layer", minimum=1)
    require_int(event, "total_digest_blocks", minimum=1)

    if valid_token_count != block_size:
        raise AssertionError(
            f"{event['_source']}: valid_token_count={valid_token_count} "
            f"must equal block_size={block_size} for full-block digest v0."
        )

    digest_min_shape = require_shape(event, "digest_min_shape")
    digest_max_shape = require_shape(event, "digest_max_shape")
    if digest_min_shape != digest_max_shape:
        raise AssertionError(
            f"{event['_source']}: digest_min_shape={digest_min_shape} "
            f"does not match digest_max_shape={digest_max_shape}."
        )


def validate_score_event(event: dict[str, Any]) -> None:
    """Validate one score_estimated event's score metadata."""
    window_query_len = require_int(event, "window_query_len", minimum=1)
    num_digest_blocks = require_int(event, "num_digest_blocks", minimum=1)
    score_count = require_int(event, "score_count", minimum=1)
    topk = require_int(event, "topk", minimum=0)

    if score_count != num_digest_blocks:
        raise AssertionError(
            f"{event['_source']}: score_count={score_count} must match "
            f"num_digest_blocks={num_digest_blocks}."
        )
    if topk > score_count:
        raise AssertionError(
            f"{event['_source']}: topk={topk} exceeds score_count={score_count}."
        )

    score_agg = event.get("score_agg")
    if score_agg not in ("max", "mean"):
        raise AssertionError(
            f"{event['_source']}: score_agg must be 'max' or 'mean'."
        )

    window_query_shape = event.get("window_query_shape")
    if (
        not isinstance(window_query_shape, list)
        or len(window_query_shape) != 2
        or not all(isinstance(dim, int) and dim > 0 for dim in window_query_shape)
    ):
        raise AssertionError(
            f"{event['_source']}: window_query_shape must be "
            "[num_q_heads, head_dim]."
        )

    topk_block_ids = event.get("topk_block_ids")
    topk_scores = event.get("topk_scores")
    if not isinstance(topk_block_ids, list) or len(topk_block_ids) != topk:
        raise AssertionError(
            f"{event['_source']}: topk_block_ids must have length {topk}."
        )
    if not all(isinstance(block_id, int) and block_id >= 0
               for block_id in topk_block_ids):
        raise AssertionError(
            f"{event['_source']}: topk_block_ids must contain non-negative ints."
        )
    if not isinstance(topk_scores, list) or len(topk_scores) != topk:
        raise AssertionError(
            f"{event['_source']}: topk_scores must have length {topk}."
        )
    if not all(isinstance(score, (int, float)) for score in topk_scores):
        raise AssertionError(
            f"{event['_source']}: topk_scores must contain numeric values."
        )
    if window_query_len > 0 and topk == 0:
        raise AssertionError(
            f"{event['_source']}: score event with digest blocks should have "
            "topk > 0."
        )


def validate_digest_observe_matches(
    digest_events: list[dict[str, Any]],
    observe_events: list[dict[str, Any]],
    *,
    allow_unmatched: bool,
) -> None:
    """Match digest_created events to their observe_kv_write parent events."""
    observes_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for event in observe_events:
        layer_name = event.get("layer_name")
        layer_event_idx = event.get("layer_event_idx")
        if isinstance(layer_name, str) and isinstance(layer_event_idx, int):
            observes_by_key[(layer_name, layer_event_idx)].append(event)

    unmatched: list[str] = []
    for digest_event in digest_events:
        layer_name = digest_event.get("layer_name")
        layer_event_idx = digest_event.get("layer_event_idx")
        physical_block_id = digest_event.get("physical_block_id")
        if not isinstance(layer_name, str) or not isinstance(layer_event_idx, int):
            raise AssertionError(
                f"{digest_event['_source']}: digest event missing layer match key."
            )

        matches = observes_by_key.get((layer_name, layer_event_idx), [])
        matched = False
        for observe_event in matches:
            created_block_ids = observe_event.get("digest_created_block_ids", [])
            unique_block_ids = observe_event.get("unique_block_ids", [])
            if physical_block_id in created_block_ids:
                if physical_block_id not in unique_block_ids:
                    raise AssertionError(
                        f"{observe_event['_source']}: digest block "
                        f"{physical_block_id} missing from unique_block_ids."
                    )
                if observe_event.get("block_size") != digest_event.get("block_size"):
                    raise AssertionError(
                        f"{digest_event['_source']}: digest block_size does not "
                        f"match observe event at {observe_event['_source']}."
                    )
                matched = True
                break

        if not matched:
            unmatched.append(
                f"{digest_event['_source']} "
                f"layer={layer_name} event={layer_event_idx} "
                f"block={physical_block_id}"
            )

    if unmatched and not allow_unmatched:
        details = "\n  ".join(unmatched[:10])
        raise AssertionError(
            "digest_created events without matching observe_kv_write records:\n"
            f"  {details}\n"
            "Use less restrictive debug settings or pass "
            "--allow-unmatched-digest-events."
        )


def print_summary(
    paths: list[Path],
    events: list[dict[str, Any]],
    digest_events: list[dict[str, Any]],
    observe_events: list[dict[str, Any]],
    score_events: list[dict[str, Any]],
    show: int,
) -> None:
    """Print a compact validation summary."""
    digest_counts_by_layer = Counter(
        event.get("layer_name") for event in digest_events
    )
    observed_valid_slots_by_layer: defaultdict[str, int] = defaultdict(int)
    block_sizes_by_layer: defaultdict[str, set[int]] = defaultdict(set)
    for event in observe_events:
        layer_name = event.get("layer_name")
        if not isinstance(layer_name, str):
            continue
        observed_valid_slots_by_layer[layer_name] += event["num_valid_slots"]
        block_sizes_by_layer[layer_name].add(event["block_size"])

    print("MPR debug JSONL validation passed")
    print(f"files: {len(paths)}")
    print(f"events: {len(events)}")
    print(f"observe_kv_write: {len(observe_events)}")
    print(f"digest_created: {len(digest_events)}")
    print(f"score_estimated: {len(score_events)}")
    print(f"digest_layers: {len(digest_counts_by_layer)}")

    if digest_counts_by_layer:
        print("\ndigest count by layer:")
        for layer_name, count in digest_counts_by_layer.most_common():
            print(f"  {count:4d}  {layer_name}")

    if observed_valid_slots_by_layer:
        print("\nobserved KV write slots by layer:")
        print(
            "  valid_slots  block_size  lower_bound_full_blocks  "
            "digests  layer_name"
        )
        for layer_name, valid_slots in sorted(
            observed_valid_slots_by_layer.items()
        ):
            block_sizes = block_sizes_by_layer[layer_name]
            if len(block_sizes) == 1:
                block_size = next(iter(block_sizes))
                lower_bound_full_blocks = valid_slots // block_size
                block_size_text = str(block_size)
                full_blocks_text = str(lower_bound_full_blocks)
            else:
                block_size_text = str(sorted(block_sizes))
                full_blocks_text = "n/a"
            print(
                f"  {valid_slots:11d}  "
                f"{block_size_text:10s}  "
                f"{full_blocks_text:23s}  "
                f"{digest_counts_by_layer.get(layer_name, 0):7d}  "
                f"{layer_name}"
            )

    if show > 0 and digest_events:
        print("\nfirst digest_created events:")
        print(
            "  layer_event_idx  physical_block_id  "
            "shape              block_size  layer_name"
        )
        for event in digest_events[:show]:
            shape = event["digest_min_shape"]
            print(
                f"  {event['layer_event_idx']:15d}  "
                f"{event['physical_block_id']:17d}  "
                f"{str(shape):17s}  "
                f"{event['block_size']:10d}  "
                f"{event['layer_name']}"
            )

    if score_events:
        score_counts_by_layer = Counter(
            event.get("layer_name") for event in score_events
        )
        print("\nscore count by layer:")
        for layer_name, count in score_counts_by_layer.most_common():
            print(f"  {count:4d}  {layer_name}")


def main() -> None:
    """Run JSONL validation and print a compact summary."""
    args = parse_args()
    paths = expand_paths(args.paths)
    if not paths:
        raise AssertionError("No JSONL paths matched.")

    events = load_jsonl(paths)
    if not events:
        raise AssertionError("No JSONL events found.")

    observe_events = [
        event for event in events if event.get("event") == "observe_kv_write"
    ]
    digest_events = [
        event for event in events if event.get("event") == "digest_created"
    ]
    score_events = [
        event for event in events if event.get("event") == "score_estimated"
    ]

    for event in observe_events:
        validate_observe_event(event)
    for event in digest_events:
        validate_digest_event(event)
    for event in score_events:
        validate_score_event(event)

    if len(digest_events) < args.min_digest_events:
        raise AssertionError(
            f"Expected at least {args.min_digest_events} digest_created events, "
            f"found {len(digest_events)}."
        )
    if len(score_events) < args.min_score_events:
        raise AssertionError(
            f"Expected at least {args.min_score_events} score_estimated events, "
            f"found {len(score_events)}."
        )

    validate_digest_observe_matches(
        digest_events,
        observe_events,
        allow_unmatched=args.allow_unmatched_digest_events,
    )
    print_summary(
        paths,
        events,
        digest_events,
        observe_events,
        score_events,
        args.show,
    )


if __name__ == "__main__":
    main()
