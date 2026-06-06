#!/usr/bin/env python3
"""Validate MPR debug JSONL records from a Step 1.3 smoke run."""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SUPPORTED_DIGEST_KINDS = {"arkvale", "raw_minmax"}
SUPPORTED_SCORING_BACKENDS = {"torch_quest", "quest_cuda"}
SUPPORTED_SCORE_GRANULARITIES = {"block", "kv_head", "query_head"}
SUPPORTED_RECOVERY_POLICIES = {"topk_block", "threshold_block"}
CPU_BACKUP_STAT_FIELDS = (
    "cpu_backup_block_count",
    "cpu_backup_bytes",
    "cpu_backup_fp16_payload_bytes",
    "cpu_backup_int8_payload_bytes",
    "cpu_backup_int8_scale_bytes",
    "cpu_backup_total_actual_bytes",
)


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
        "--min-recovery-events",
        type=int,
        default=0,
        help="Minimum number of recovery_materialized events required.",
    )
    parser.add_argument(
        "--min-test-mutation-events",
        type=int,
        default=0,
        help="Minimum number of recovery_test_mutated events required.",
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
        "--strict-current-request-scores",
        action="store_true",
        help=(
            "Require every score_estimated event to score exactly the current "
            "request's finalized block IDs. This is intended for single-request "
            "Step 1.6 smoke validation, not multi-request serving."
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


def require_int_list(
    event: dict[str, Any],
    field: str,
    *,
    minimum: int | None = None,
) -> list[int]:
    """Read an integer list field and optionally enforce a minimum value."""
    value = event.get(field)
    if not isinstance(value, list):
        raise AssertionError(f"{event['_source']}: {field} must be a list.")
    if not all(isinstance(item, int) for item in value):
        raise AssertionError(
            f"{event['_source']}: {field} must contain integer values."
        )
    if minimum is not None and any(item < minimum for item in value):
        raise AssertionError(
            f"{event['_source']}: {field} must contain values >= {minimum}."
        )
    return value


def require_nested_int_list(
    event: dict[str, Any],
    field: str,
    *,
    outer_len: int,
    inner_len: int,
    minimum: int | None = None,
) -> list[list[int]]:
    """Read a nested integer list and enforce exact outer/inner lengths."""
    value = event.get(field)
    if not isinstance(value, list) or len(value) != outer_len:
        raise AssertionError(
            f"{event['_source']}: {field} must be a list with "
            f"{outer_len} rows."
        )
    rows: list[list[int]] = []
    for row_idx, row in enumerate(value):
        if not isinstance(row, list) or len(row) != inner_len:
            raise AssertionError(
                f"{event['_source']}: {field}[{row_idx}] must have "
                f"length {inner_len}."
            )
        if not all(isinstance(item, int) for item in row):
            raise AssertionError(
                f"{event['_source']}: {field}[{row_idx}] must contain "
                "integer values."
            )
        if minimum is not None and any(item < minimum for item in row):
            raise AssertionError(
                f"{event['_source']}: {field}[{row_idx}] must contain "
                f"values >= {minimum}."
            )
        rows.append(row)
    return rows


def require_nested_finite_number_list(
    event: dict[str, Any],
    field: str,
    *,
    outer_len: int,
    inner_len: int,
) -> list[list[float]]:
    """Read a nested finite-number list and enforce exact row lengths."""
    value = event.get(field)
    if not isinstance(value, list) or len(value) != outer_len:
        raise AssertionError(
            f"{event['_source']}: {field} must be a list with "
            f"{outer_len} rows."
        )
    rows: list[list[float]] = []
    for row_idx, row in enumerate(value):
        if not isinstance(row, list) or len(row) != inner_len:
            raise AssertionError(
                f"{event['_source']}: {field}[{row_idx}] must have "
                f"length {inner_len}."
            )
        if not all(
            isinstance(item, (int, float)) and math.isfinite(float(item))
            for item in row
        ):
            raise AssertionError(
                f"{event['_source']}: {field}[{row_idx}] must contain finite "
                "numeric values."
            )
        rows.append([float(item) for item in row])
    return rows


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

    digest_kind = event.get("digest_kind")
    if digest_kind is not None and digest_kind not in SUPPORTED_DIGEST_KINDS:
        raise AssertionError(
            f"{event['_source']}: digest_kind must be one of "
            f"{sorted(SUPPORTED_DIGEST_KINDS)}, got {digest_kind!r}."
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

    scoring_backend = event.get("scoring_backend")
    if (
        scoring_backend is not None
        and scoring_backend not in SUPPORTED_SCORING_BACKENDS
    ):
        raise AssertionError(
            f"{event['_source']}: scoring_backend must be one of "
            f"{sorted(SUPPORTED_SCORING_BACKENDS)}, got {scoring_backend!r}."
        )

    digest_kind = event.get("digest_kind")
    if digest_kind is not None and digest_kind not in SUPPORTED_DIGEST_KINDS:
        raise AssertionError(
            f"{event['_source']}: digest_kind must be one of "
            f"{sorted(SUPPORTED_DIGEST_KINDS)}, got {digest_kind!r}."
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

    num_q_heads = event.get("num_q_heads")
    num_kv_heads = event.get("num_kv_heads")
    gqa_group_size = event.get("gqa_group_size")
    score_granularity = event.get("score_granularity", "block")
    if score_granularity not in SUPPORTED_SCORE_GRANULARITIES:
        raise AssertionError(
            f"{event['_source']}: score_granularity must be one of "
            f"{sorted(SUPPORTED_SCORE_GRANULARITIES)}, "
            f"got {score_granularity!r}."
        )
    if num_q_heads is not None or num_kv_heads is not None:
        num_q_heads = require_int(event, "num_q_heads", minimum=1)
        num_kv_heads = require_int(event, "num_kv_heads", minimum=1)
        if num_q_heads != window_query_shape[0]:
            raise AssertionError(
                f"{event['_source']}: num_q_heads={num_q_heads} must match "
                f"window_query_shape[0]={window_query_shape[0]}."
            )
        if num_q_heads % num_kv_heads != 0:
            raise AssertionError(
                f"{event['_source']}: num_q_heads must be a multiple of "
                f"num_kv_heads, got {num_q_heads} and {num_kv_heads}."
            )
        expected_group_size = num_q_heads // num_kv_heads
        if gqa_group_size is not None:
            gqa_group_size = require_int(event, "gqa_group_size", minimum=1)
        if gqa_group_size is not None and gqa_group_size != expected_group_size:
            raise AssertionError(
                f"{event['_source']}: gqa_group_size={gqa_group_size} "
                f"must equal {expected_group_size}."
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
    if not all(
        isinstance(score, (int, float)) and math.isfinite(float(score))
        for score in topk_scores
    ):
        raise AssertionError(
            f"{event['_source']}: topk_scores must contain finite numeric values."
        )
    if window_query_len > 0 and topk == 0:
        raise AssertionError(
            f"{event['_source']}: score event with digest blocks should have "
            "topk > 0."
        )

    observed_digest_block_ids = event.get("observed_digest_block_ids")
    if observed_digest_block_ids is not None:
        observed_digest_block_ids = require_int_list(
            event,
            "observed_digest_block_ids",
            minimum=0,
        )
        if len(observed_digest_block_ids) != score_count:
            raise AssertionError(
                f"{event['_source']}: observed_digest_block_ids length must "
                f"match score_count={score_count}."
            )
        if not set(topk_block_ids).issubset(set(observed_digest_block_ids)):
            raise AssertionError(
                f"{event['_source']}: topk_block_ids must be a subset of "
                "observed_digest_block_ids."
            )

    if score_granularity in ("kv_head", "query_head"):
        num_score_heads = require_int(event, "num_score_heads", minimum=1)
        head_score_count = require_int(event, "head_score_count", minimum=1)
        if head_score_count != score_count * num_score_heads:
            raise AssertionError(
                f"{event['_source']}: head_score_count={head_score_count} "
                f"must equal score_count * num_score_heads "
                f"({score_count * num_score_heads})."
            )
        expected_heads = num_kv_heads if score_granularity == "kv_head" else num_q_heads
        if expected_heads is not None and num_score_heads != expected_heads:
            raise AssertionError(
                f"{event['_source']}: num_score_heads={num_score_heads} "
                f"does not match {score_granularity} expected head count "
                f"{expected_heads}."
            )
        topk_block_ids_by_head = require_nested_int_list(
            event,
            "topk_block_ids_by_head",
            outer_len=num_score_heads,
            inner_len=topk,
            minimum=0,
        )
        require_nested_finite_number_list(
            event,
            "topk_scores_by_head",
            outer_len=num_score_heads,
            inner_len=topk,
        )
        if observed_digest_block_ids is not None:
            observed_set = set(observed_digest_block_ids)
            for row_idx, block_ids in enumerate(topk_block_ids_by_head):
                if not set(block_ids).issubset(observed_set):
                    raise AssertionError(
                        f"{event['_source']}: topk_block_ids_by_head[{row_idx}] "
                        "must be a subset of observed_digest_block_ids."
                    )

    if event.get("block_table_row") is not None:
        require_int_list(event, "block_table_row")

    for field in (
        "valid_block_ids",
        "finalized_block_ids",
        "protected_block_ids",
        "score_candidate_block_ids",
        "missing_digest_blocks",
        "extra_digest_blocks",
    ):
        if event.get(field) is not None:
            require_int_list(event, field, minimum=0)

    for field in ("recent_tokens", "protected_tail_entries"):
        if event.get(field) is not None:
            require_int(event, field, minimum=0)

    if event.get("seq_lens") is not None:
        require_int_list(event, "seq_lens", minimum=0)

    block_size = event.get("block_size")
    if block_size is not None and (
        not isinstance(block_size, int) or block_size <= 0
    ):
        raise AssertionError(
            f"{event['_source']}: block_size must be a positive int when set."
        )


def validate_recovery_materialized_event(event: dict[str, Any]) -> None:
    """Validate one recovery_materialized event's recovery metadata."""
    recovery_policy = event.get("recovery_policy")
    if recovery_policy not in SUPPORTED_RECOVERY_POLICIES:
        raise AssertionError(
            f"{event['_source']}: recovery_policy must be one of "
            f"{sorted(SUPPORTED_RECOVERY_POLICIES)}, got {recovery_policy!r}."
        )
    require_int(event, "recovery_topk", minimum=1)
    threshold = event.get("recovery_threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise AssertionError(
            f"{event['_source']}: recovery_threshold must be a finite number."
        )

    selected_block_ids = require_int_list(
        event,
        "recovery_selected_block_ids",
        minimum=0,
    )
    test_mutate = event.get("recovery_test_mutate", "off")
    if test_mutate not in ("off", "zero_selected", "zero_all"):
        raise AssertionError(
            f"{event['_source']}: recovery_test_mutate must be 'off', "
            f"'zero_selected', or 'zero_all', got {test_mutate!r}."
        )
    mutated_scope = event.get("recovery_test_mutated_scope", "selected")
    if mutated_scope not in ("selected", "all_kv_cache"):
        raise AssertionError(
            f"{event['_source']}: recovery_test_mutated_scope must be "
            f"'selected' or 'all_kv_cache', got {mutated_scope!r}."
        )
    test_mode = event.get("recovery_test_mode", "recover")
    if test_mode not in ("recover", "mutate_only"):
        raise AssertionError(
            f"{event['_source']}: recovery_test_mode must be 'recover' or "
            f"'mutate_only', got {test_mode!r}."
        )
    test_mutated_block_ids = event.get("recovery_test_mutated_block_ids")
    if test_mutated_block_ids is not None:
        test_mutated_block_ids = require_int_list(
            event,
            "recovery_test_mutated_block_ids",
            minimum=0,
        )
        if (
            mutated_scope == "selected"
            and not set(test_mutated_block_ids).issubset(set(selected_block_ids))
        ):
            raise AssertionError(
                f"{event['_source']}: recovery_test_mutated_block_ids must be "
                "a subset of recovery_selected_block_ids."
            )
    recovered_block_ids = require_int_list(
        event,
        "recovered_block_ids",
        minimum=0,
    )
    missing_backup_block_ids = require_int_list(
        event,
        "missing_backup_block_ids",
        minimum=0,
    )
    skipped_block_ids = require_int_list(
        event,
        "skipped_block_ids",
        minimum=0,
    )
    selected_set = set(selected_block_ids)
    for field, block_ids in (
        ("recovered_block_ids", recovered_block_ids),
        ("missing_backup_block_ids", missing_backup_block_ids),
        ("skipped_block_ids", skipped_block_ids),
    ):
        if not set(block_ids).issubset(selected_set):
            raise AssertionError(
                f"{event['_source']}: {field} must be a subset of "
                "recovery_selected_block_ids."
            )

    recovered_bytes = require_int(event, "recovered_bytes", minimum=0)
    copy_wall_ms = event.get("recovery_copy_wall_ms")
    if (
        not isinstance(copy_wall_ms, (int, float))
        or not math.isfinite(float(copy_wall_ms))
        or float(copy_wall_ms) < 0
    ):
        raise AssertionError(
            f"{event['_source']}: recovery_copy_wall_ms must be a finite "
            "non-negative number."
        )
    if recovered_block_ids and recovered_bytes <= 0:
        raise AssertionError(
            f"{event['_source']}: recovered_bytes must be > 0 when blocks "
            "were recovered."
        )
    if not recovered_block_ids and recovered_bytes != 0:
        raise AssertionError(
            f"{event['_source']}: recovered_bytes must be 0 when no blocks "
            "were recovered."
        )

    if event.get("kv_cache_shape") is not None:
        kv_cache_shape = event["kv_cache_shape"]
        if (
            not isinstance(kv_cache_shape, list)
            or len(kv_cache_shape) != 5
            or kv_cache_shape[0] != 2
            or not all(isinstance(dim, int) and dim > 0 for dim in kv_cache_shape)
        ):
            raise AssertionError(
                f"{event['_source']}: kv_cache_shape must be "
                "[2, num_blocks, block_size, num_kv_heads, head_dim]."
            )

    for field in CPU_BACKUP_STAT_FIELDS:
        if event.get(field) is not None:
            require_int(event, field, minimum=0)

    for field in (
        "valid_block_ids",
        "finalized_block_ids",
        "protected_block_ids",
        "score_candidate_block_ids",
        "observed_digest_block_ids",
        "missing_digest_blocks",
        "extra_digest_blocks",
    ):
        if event.get(field) is not None:
            require_int_list(event, field, minimum=0)


def validate_recovery_test_mutated_event(event: dict[str, Any]) -> None:
    """Validate one recovery_test_mutated event's validation metadata."""
    recovery_policy = event.get("recovery_policy")
    if recovery_policy not in SUPPORTED_RECOVERY_POLICIES:
        raise AssertionError(
            f"{event['_source']}: recovery_policy must be one of "
            f"{sorted(SUPPORTED_RECOVERY_POLICIES)}, got {recovery_policy!r}."
        )
    require_int(event, "recovery_topk", minimum=1)
    threshold = event.get("recovery_threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise AssertionError(
            f"{event['_source']}: recovery_threshold must be a finite number."
        )

    test_mutate = event.get("recovery_test_mutate")
    if test_mutate not in ("zero_selected", "zero_all"):
        raise AssertionError(
            f"{event['_source']}: recovery_test_mutate must be "
            f"'zero_selected' or 'zero_all', got {test_mutate!r}."
        )
    mutated_scope = event.get("recovery_test_mutated_scope", "selected")
    if mutated_scope not in ("selected", "all_kv_cache"):
        raise AssertionError(
            f"{event['_source']}: recovery_test_mutated_scope must be "
            f"'selected' or 'all_kv_cache', got {mutated_scope!r}."
        )
    test_mode = event.get("recovery_test_mode")
    if test_mode != "mutate_only":
        raise AssertionError(
            f"{event['_source']}: recovery_test_mode must be 'mutate_only', "
            f"got {test_mode!r}."
        )

    selected_block_ids = require_int_list(
        event,
        "recovery_selected_block_ids",
        minimum=0,
    )
    mutated_block_ids = require_int_list(
        event,
        "recovery_test_mutated_block_ids",
        minimum=0,
    )
    if (
        mutated_scope == "selected"
        and not set(mutated_block_ids).issubset(set(selected_block_ids))
    ):
        raise AssertionError(
            f"{event['_source']}: recovery_test_mutated_block_ids must be a "
            "subset of recovery_selected_block_ids."
        )

    if event.get("kv_cache_shape") is not None:
        kv_cache_shape = event["kv_cache_shape"]
        if (
            not isinstance(kv_cache_shape, list)
            or len(kv_cache_shape) != 5
            or kv_cache_shape[0] != 2
            or not all(isinstance(dim, int) and dim > 0 for dim in kv_cache_shape)
        ):
            raise AssertionError(
                f"{event['_source']}: kv_cache_shape must be "
                "[2, num_blocks, block_size, num_kv_heads, head_dim]."
            )

    for field in CPU_BACKUP_STAT_FIELDS:
        if event.get(field) is not None:
            require_int(event, field, minimum=0)

    for field in (
        "valid_block_ids",
        "finalized_block_ids",
        "protected_block_ids",
        "score_candidate_block_ids",
        "observed_digest_block_ids",
        "missing_digest_blocks",
        "extra_digest_blocks",
    ):
        if event.get(field) is not None:
            require_int_list(event, field, minimum=0)


def validate_recovery_skipped_event(event: dict[str, Any]) -> None:
    """Validate one recovery_skipped event's metadata."""
    reason = event.get("skipped_reason")
    if not isinstance(reason, str) or not reason:
        raise AssertionError(
            f"{event['_source']}: skipped_reason must be a non-empty string."
        )
    recovery_policy = event.get("recovery_policy")
    if recovery_policy not in SUPPORTED_RECOVERY_POLICIES:
        raise AssertionError(
            f"{event['_source']}: recovery_policy must be one of "
            f"{sorted(SUPPORTED_RECOVERY_POLICIES)}, got {recovery_policy!r}."
        )
    require_int(event, "recovery_topk", minimum=1)
    threshold = event.get("recovery_threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise AssertionError(
            f"{event['_source']}: recovery_threshold must be a finite number."
        )
    for field in ("cpu_backup_enabled", "scoring_enabled"):
        value = event.get(field)
        if not isinstance(value, bool):
            raise AssertionError(f"{event['_source']}: {field} must be a bool.")


def validate_strict_current_request_score_event(event: dict[str, Any]) -> None:
    """Validate score candidates against the current request's finalized blocks."""
    observed_digest_block_ids = require_int_list(
        event,
        "observed_digest_block_ids",
        minimum=0,
    )
    finalized_block_ids = require_int_list(
        event,
        "finalized_block_ids",
        minimum=0,
    )
    topk_block_ids = require_int_list(event, "topk_block_ids", minimum=0)
    missing_digest_blocks = require_int_list(
        event,
        "missing_digest_blocks",
        minimum=0,
    )
    extra_digest_blocks = require_int_list(
        event,
        "extra_digest_blocks",
        minimum=0,
    )
    score_candidate_block_ids = event.get("score_candidate_block_ids")
    if score_candidate_block_ids is not None:
        expected_block_ids = require_int_list(
            event,
            "score_candidate_block_ids",
            minimum=0,
        )
        expected_label = "score_candidate_block_ids"
    else:
        expected_block_ids = finalized_block_ids
        expected_label = "finalized_block_ids"

    if missing_digest_blocks:
        raise AssertionError(
            f"{event['_source']}: strict score validation found missing "
            f"candidate digest blocks: {missing_digest_blocks}."
        )
    if extra_digest_blocks:
        raise AssertionError(
            f"{event['_source']}: strict score validation found extra cached "
            f"digest blocks: {extra_digest_blocks}."
        )

    observed_set = set(observed_digest_block_ids)
    expected_set = set(expected_block_ids)
    if observed_set != expected_set:
        raise AssertionError(
            f"{event['_source']}: scored digest blocks must match "
            f"{expected_label}; observed={sorted(observed_set)}, "
            f"expected={sorted(expected_set)}."
        )
    if not set(topk_block_ids).issubset(expected_set):
        raise AssertionError(
            f"{event['_source']}: topk_block_ids must be a subset of "
            f"{expected_label}."
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
    recovery_events: list[dict[str, Any]],
    recovery_test_mutated_events: list[dict[str, Any]],
    recovery_skipped_events: list[dict[str, Any]],
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
    print(f"recovery_materialized: {len(recovery_events)}")
    print(f"recovery_test_mutated: {len(recovery_test_mutated_events)}")
    print(f"recovery_skipped: {len(recovery_skipped_events)}")
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

        missing_digest_events = [
            event for event in score_events if event.get("missing_digest_blocks")
        ]
        extra_digest_events = [
            event for event in score_events if event.get("extra_digest_blocks")
        ]
        if missing_digest_events:
            print(
                "\nscore events with missing finalized digest blocks: "
                f"{len(missing_digest_events)}"
            )
            for event in missing_digest_events[:show]:
                print(
                    f"  {event['_source']}  layer={event.get('layer_name')}  "
                    f"missing={event.get('missing_digest_blocks')}"
                )
        if extra_digest_events:
            print(
                "\nscore events with extra cached digest blocks: "
                f"{len(extra_digest_events)}"
            )
            for event in extra_digest_events[:show]:
                print(
                    f"  {event['_source']}  layer={event.get('layer_name')}  "
                    f"extra={event.get('extra_digest_blocks')}"
                )

    if recovery_events:
        recovery_counts_by_layer = Counter(
            event.get("layer_name") for event in recovery_events
        )
        print("\nrecovery count by layer:")
        for layer_name, count in recovery_counts_by_layer.most_common():
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
    recovery_events = [
        event for event in events if event.get("event") == "recovery_materialized"
    ]
    recovery_test_mutated_events = [
        event for event in events if event.get("event") == "recovery_test_mutated"
    ]
    recovery_skipped_events = [
        event for event in events if event.get("event") == "recovery_skipped"
    ]

    for event in observe_events:
        validate_observe_event(event)
    for event in digest_events:
        validate_digest_event(event)
    for event in score_events:
        validate_score_event(event)
        if args.strict_current_request_scores:
            validate_strict_current_request_score_event(event)
    for event in recovery_events:
        validate_recovery_materialized_event(event)
    for event in recovery_test_mutated_events:
        validate_recovery_test_mutated_event(event)
    for event in recovery_skipped_events:
        validate_recovery_skipped_event(event)

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
    if len(recovery_events) < args.min_recovery_events:
        raise AssertionError(
            f"Expected at least {args.min_recovery_events} "
            "recovery_materialized events, "
            f"found {len(recovery_events)}."
        )
    if len(recovery_test_mutated_events) < args.min_test_mutation_events:
        raise AssertionError(
            f"Expected at least {args.min_test_mutation_events} "
            "recovery_test_mutated events, found "
            f"{len(recovery_test_mutated_events)}."
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
        recovery_events,
        recovery_test_mutated_events,
        recovery_skipped_events,
        args.show,
    )
    if args.strict_current_request_scores:
        print("strict_current_request_scores: passed")


if __name__ == "__main__":
    main()
