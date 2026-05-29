# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from scripts.mpr_validate_debug_jsonl import (
    load_jsonl,
    validate_digest_event,
    validate_digest_observe_matches,
    validate_score_event,
    validate_observe_event,
)


def test_debug_jsonl_validator_accepts_matching_digest_event(tmp_path):
    path = tmp_path / "mpr.jsonl"
    observe_event = {
        "event": "observe_kv_write",
        "layer_name": "model.layers.0.self_attn.attn",
        "layer_event_idx": 3,
        "num_slots": 16,
        "num_valid_slots": 16,
        "num_pad_slots": 0,
        "block_size": 16,
        "unique_block_ids": [7],
        "min_block_offset": 0,
        "max_block_offset": 15,
        "digest_created_block_ids": [7],
    }
    digest_event = {
        "event": "digest_created",
        "layer_name": "model.layers.0.self_attn.attn",
        "layer_event_idx": 3,
        "physical_block_id": 7,
        "digest_min_shape": [8, 128],
        "digest_max_shape": [8, 128],
        "valid_token_count": 16,
        "block_size": 16,
        "num_digest_blocks_for_layer": 1,
        "total_digest_blocks": 1,
    }
    path.write_text(
        "\n".join(json.dumps(event) for event in [observe_event, digest_event]),
        encoding="utf-8",
    )

    events = load_jsonl([path])
    observe_events = [event for event in events if event["event"] == "observe_kv_write"]
    digest_events = [event for event in events if event["event"] == "digest_created"]

    validate_observe_event(observe_events[0])
    validate_digest_event(digest_events[0])
    validate_digest_observe_matches(
        digest_events,
        observe_events,
        allow_unmatched=False,
    )


def test_debug_jsonl_validator_rejects_unmatched_digest_event(tmp_path):
    path = tmp_path / "mpr.jsonl"
    digest_event = {
        "event": "digest_created",
        "layer_name": "model.layers.0.self_attn.attn",
        "layer_event_idx": 3,
        "physical_block_id": 7,
        "digest_min_shape": [8, 128],
        "digest_max_shape": [8, 128],
        "valid_token_count": 16,
        "block_size": 16,
        "num_digest_blocks_for_layer": 1,
        "total_digest_blocks": 1,
    }
    path.write_text(json.dumps(digest_event), encoding="utf-8")

    digest_events = load_jsonl([path])
    validate_digest_event(digest_events[0])
    with pytest.raises(AssertionError, match="without matching observe_kv_write"):
        validate_digest_observe_matches(
            digest_events,
            observe_events=[],
            allow_unmatched=False,
        )


def test_debug_jsonl_validator_accepts_score_event(tmp_path):
    path = tmp_path / "mpr.jsonl"
    score_event = {
        "event": "score_estimated",
        "layer_name": "model.layers.0.self_attn.attn",
        "layer_event_idx": 4,
        "query_shape": [1, 32, 128],
        "window_query_shape": [32, 128],
        "window_query_len": 3,
        "num_digest_blocks": 2,
        "score_count": 2,
        "score_agg": "max",
        "topk": 2,
        "topk_block_ids": [7, 8],
        "topk_scores": [4.0, 3.0],
        "num_reqs": 1,
        "max_query_len": 1,
        "num_actual_tokens": 1,
        "seq_lens": [33],
        "block_size": 16,
        "block_table_shape": [1, 4],
        "block_table_row": [7, 8, 9, 0],
        "valid_block_ids": [7, 8, 9],
        "finalized_block_ids": [7, 8],
        "observed_digest_block_ids": [7, 8],
        "missing_digest_blocks": [],
        "extra_digest_blocks": [],
    }
    path.write_text(json.dumps(score_event), encoding="utf-8")

    events = load_jsonl([path])
    validate_score_event(events[0])


def test_debug_jsonl_validator_rejects_topk_outside_scored_digests(tmp_path):
    path = tmp_path / "mpr.jsonl"
    score_event = {
        "event": "score_estimated",
        "layer_name": "model.layers.0.self_attn.attn",
        "layer_event_idx": 4,
        "query_shape": [1, 32, 128],
        "window_query_shape": [32, 128],
        "window_query_len": 3,
        "num_digest_blocks": 2,
        "score_count": 2,
        "score_agg": "max",
        "topk": 1,
        "topk_block_ids": [9],
        "topk_scores": [4.0],
        "observed_digest_block_ids": [7, 8],
    }
    path.write_text(json.dumps(score_event), encoding="utf-8")

    events = load_jsonl([path])
    with pytest.raises(AssertionError, match="subset"):
        validate_score_event(events[0])
