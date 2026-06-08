# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys

import pytest

from scripts.mpr_validate_debug_jsonl import (
    load_jsonl,
    main as validate_debug_jsonl_main,
    validate_digest_event,
    validate_digest_observe_matches,
    validate_observe_event,
    validate_recovery_materialized_event,
    validate_recovery_skipped_event,
    validate_recovery_test_mutated_event,
    validate_score_event,
    validate_strict_current_request_score_event,
    validate_tiered_skip_unrecovered_requirement,
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
        "digest_kind": "arkvale",
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
        "scoring_backend": "torch_quest",
        "digest_kind": "raw_minmax",
        "num_q_heads": 32,
        "num_kv_heads": 8,
        "gqa_group_size": 4,
        "score_granularity": "block",
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
        "recent_tokens": 64,
        "protected_tail_entries": 1,
        "protected_block_ids": [9],
        "score_candidate_block_ids": [7, 8],
        "observed_digest_block_ids": [7, 8],
        "missing_digest_blocks": [],
        "extra_digest_blocks": [],
    }
    path.write_text(json.dumps(score_event), encoding="utf-8")

    events = load_jsonl([path])
    validate_score_event(events[0])
    validate_strict_current_request_score_event(events[0])


def test_debug_jsonl_validator_accepts_recent_protected_score_candidates(tmp_path):
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
        "topk_block_ids": [7],
        "topk_scores": [4.0],
        "valid_block_ids": [7, 8, 9, 10],
        "finalized_block_ids": [7, 8, 9],
        "recent_tokens": 64,
        "protected_tail_entries": 2,
        "protected_block_ids": [9, 10],
        "score_candidate_block_ids": [7, 8],
        "observed_digest_block_ids": [7, 8],
        "missing_digest_blocks": [],
        "extra_digest_blocks": [],
    }
    path.write_text(json.dumps(score_event), encoding="utf-8")

    events = load_jsonl([path])
    validate_score_event(events[0])
    validate_strict_current_request_score_event(events[0])


def test_debug_jsonl_validator_accepts_head_granularity_score_event(tmp_path):
    path = tmp_path / "mpr.jsonl"
    score_event = {
        "event": "score_estimated",
        "layer_name": "model.layers.0.self_attn.attn",
        "layer_event_idx": 4,
        "query_shape": [1, 4, 128],
        "window_query_shape": [4, 128],
        "window_query_len": 3,
        "num_digest_blocks": 3,
        "score_count": 3,
        "score_agg": "max",
        "scoring_backend": "torch_quest",
        "digest_kind": "raw_minmax",
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "gqa_group_size": 2,
        "score_granularity": "kv_head",
        "num_score_heads": 2,
        "head_score_count": 6,
        "topk": 2,
        "topk_block_ids": [7, 8],
        "topk_scores": [4.0, 3.0],
        "topk_block_ids_by_head": [[7, 9], [8, 7]],
        "topk_scores_by_head": [[5.0, 2.0], [6.0, 1.0]],
        "observed_digest_block_ids": [7, 8, 9],
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


def test_debug_jsonl_validator_rejects_non_current_request_scores(tmp_path):
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
        "topk_block_ids": [7],
        "topk_scores": [4.0],
        "finalized_block_ids": [7],
        "observed_digest_block_ids": [7, 8],
        "missing_digest_blocks": [],
        "extra_digest_blocks": [],
    }
    path.write_text(json.dumps(score_event), encoding="utf-8")

    events = load_jsonl([path])
    validate_score_event(events[0])
    with pytest.raises(AssertionError, match="must match finalized"):
        validate_strict_current_request_score_event(events[0])


def test_debug_jsonl_validator_accepts_recovery_materialized_event(tmp_path):
    path = tmp_path / "mpr.jsonl"
    recovery_event = {
        "event": "recovery_materialized",
        "layer_name": "model.layers.0.self_attn.attn",
        "layer_event_idx": 4,
        "query_shape": [1, 4, 128],
        "window_query_shape": [4, 128],
        "window_query_len": 3,
        "kv_cache_shape": [2, 16, 32, 2, 128],
        "recovery_policy": "threshold_block",
        "recovery_topk": 2,
        "recovery_threshold": 1.5,
        "recovery_test_mutate": "zero_selected",
        "recovery_test_mode": "recover",
        "recovery_test_mutated_block_ids": [7, 8],
        "recovery_selected_block_ids": [7, 8, 9],
        "recovered_block_ids": [7, 8],
        "missing_backup_block_ids": [9],
        "skipped_block_ids": [],
        "recovered_bytes": 32768,
        "recovery_copy_wall_ms": 0.25,
        "cpu_backup_block_count": 5,
        "cpu_backup_bytes": 81920,
        "cpu_backup_fp16_payload_bytes": 65536,
        "cpu_backup_int8_payload_bytes": 16384,
        "cpu_backup_int8_scale_bytes": 4096,
        "cpu_backup_total_actual_bytes": 86016,
        "valid_block_ids": [7, 8, 9],
        "finalized_block_ids": [7, 8],
        "score_candidate_block_ids": [7, 8],
        "observed_digest_block_ids": [7, 8],
        "missing_digest_blocks": [],
        "extra_digest_blocks": [],
    }
    path.write_text(json.dumps(recovery_event), encoding="utf-8")

    events = load_jsonl([path])
    validate_recovery_materialized_event(events[0])


def test_debug_jsonl_validator_accepts_tiered_recovery_materialized_event(
    tmp_path,
):
    path = tmp_path / "mpr.jsonl"
    recovery_event = {
        "event": "recovery_materialized",
        "layer_name": "model.layers.0.self_attn.attn",
        "layer_event_idx": 4,
        "query_shape": [1, 4, 128],
        "window_query_shape": [4, 128],
        "window_query_len": 3,
        "kv_cache_shape": [2, 16, 32, 2, 128],
        "recovery_policy": "threshold_block",
        "recovery_topk": 2,
        "recovery_threshold": 1.5,
        "recovery_test_mutate": "off",
        "recovery_test_mode": "recover",
        "recovery_selected_block_ids": [7, 8, 9],
        "recovered_block_ids": [7, 8],
        "missing_backup_block_ids": [],
        "skipped_block_ids": [9],
        "recovered_bytes": 32768,
        "recovery_copy_wall_ms": 0.25,
        "precision_tiering_enabled": True,
        "precision_policy": "top_ratio",
        "tier_fp16_block_ids": [7],
        "tier_int8_block_ids": [8],
        "tier_int4_block_ids": [10],
        "tier_skip_block_ids": [9],
        "recovered_fp16_block_ids": [7],
        "recovered_int8_block_ids": [8],
        "recovered_int4_block_ids": [10],
        "missing_fp16_block_ids": [],
        "missing_int8_block_ids": [],
        "missing_int4_block_ids": [],
        "fp16_payload_bytes": 16384,
        "int8_payload_bytes": 10240,
        "int4_payload_bytes": 4096,
        "int4_scale_bytes": 2048,
        "effective_recovery_transfer_bytes": 32768,
        "cpu_backup_block_count": 5,
        "cpu_backup_bytes": 92160,
        "cpu_backup_fp16_payload_bytes": 65536,
        "cpu_backup_int8_payload_bytes": 16384,
        "cpu_backup_int8_scale_bytes": 4096,
        "cpu_backup_int4_payload_bytes": 4096,
        "cpu_backup_int4_scale_bytes": 2048,
        "cpu_backup_total_actual_bytes": 92160,
        "valid_block_ids": [7, 8, 9, 10],
        "finalized_block_ids": [7, 8, 10],
        "score_candidate_block_ids": [7, 8, 10],
        "observed_digest_block_ids": [7, 8, 10],
        "missing_digest_blocks": [],
        "extra_digest_blocks": [],
    }
    path.write_text(json.dumps(recovery_event), encoding="utf-8")

    events = load_jsonl([path])
    validate_recovery_materialized_event(events[0])


def test_debug_jsonl_validator_requires_tiered_skip_unrecovered(
    tmp_path,
    monkeypatch,
    capsys,
):
    path = tmp_path / "mpr.jsonl"
    recovery_event = {
        "event": "recovery_materialized",
        "layer_name": "model.layers.0.self_attn.attn",
        "layer_event_idx": 4,
        "kv_cache_shape": [2, 16, 32, 2, 128],
        "recovery_policy": "threshold_block",
        "recovery_topk": 2,
        "recovery_threshold": 1.5,
        "recovery_test_mutate": "zero_selected",
        "recovery_test_mode": "recover",
        "recovery_test_mutated_block_ids": [7, 8, 9],
        "recovery_selected_block_ids": [7, 8, 9],
        "recovered_block_ids": [7, 8],
        "missing_backup_block_ids": [],
        "skipped_block_ids": [9],
        "recovered_bytes": 32768,
        "recovery_copy_wall_ms": 0.25,
        "precision_tiering_enabled": True,
        "precision_policy": "top_ratio",
        "tier_fp16_block_ids": [7],
        "tier_int8_block_ids": [8],
        "tier_skip_block_ids": [9],
        "recovered_fp16_block_ids": [7],
        "recovered_int8_block_ids": [8],
        "missing_fp16_block_ids": [],
        "missing_int8_block_ids": [],
        "fp16_payload_bytes": 16384,
        "int8_payload_bytes": 10240,
        "effective_recovery_transfer_bytes": 26624,
    }
    path.write_text(json.dumps(recovery_event), encoding="utf-8")

    events = load_jsonl([path])
    validate_tiered_skip_unrecovered_requirement(events)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mpr_validate_debug_jsonl.py",
            str(path),
            "--min-digest-events",
            "0",
            "--min-recovery-events",
            "1",
            "--require-tiered-skip-unrecovered",
        ],
    )
    validate_debug_jsonl_main()
    captured = capsys.readouterr()
    assert "MPR debug JSONL validation passed" in captured.out


def test_debug_jsonl_validator_rejects_tiered_skip_recovered(tmp_path):
    path = tmp_path / "mpr.jsonl"
    recovery_event = {
        "event": "recovery_materialized",
        "layer_name": "model.layers.0.self_attn.attn",
        "layer_event_idx": 4,
        "kv_cache_shape": [2, 16, 32, 2, 128],
        "recovery_policy": "threshold_block",
        "recovery_topk": 2,
        "recovery_threshold": 1.5,
        "recovery_test_mutate": "zero_selected",
        "recovery_test_mode": "recover",
        "recovery_test_mutated_block_ids": [7, 8, 9],
        "recovery_selected_block_ids": [7, 8, 9],
        "recovered_block_ids": [7, 8, 9],
        "missing_backup_block_ids": [],
        "skipped_block_ids": [9],
        "recovered_bytes": 49152,
        "recovery_copy_wall_ms": 0.25,
        "precision_tiering_enabled": True,
        "precision_policy": "top_ratio",
        "tier_fp16_block_ids": [7],
        "tier_int8_block_ids": [8],
        "tier_skip_block_ids": [9],
        "recovered_fp16_block_ids": [7],
        "recovered_int8_block_ids": [8],
        "missing_fp16_block_ids": [],
        "missing_int8_block_ids": [],
        "fp16_payload_bytes": 16384,
        "int8_payload_bytes": 10240,
        "effective_recovery_transfer_bytes": 26624,
    }
    path.write_text(json.dumps(recovery_event), encoding="utf-8")

    events = load_jsonl([path])
    with pytest.raises(AssertionError, match="skip-tier blocks"):
        validate_tiered_skip_unrecovered_requirement(events)


def test_debug_jsonl_validator_rejects_tiered_skip_not_mutated(tmp_path):
    path = tmp_path / "mpr.jsonl"
    recovery_event = {
        "event": "recovery_materialized",
        "layer_name": "model.layers.0.self_attn.attn",
        "layer_event_idx": 4,
        "kv_cache_shape": [2, 16, 32, 2, 128],
        "recovery_policy": "threshold_block",
        "recovery_topk": 2,
        "recovery_threshold": 1.5,
        "recovery_test_mutate": "zero_selected",
        "recovery_test_mode": "recover",
        "recovery_test_mutated_block_ids": [7, 8],
        "recovery_selected_block_ids": [7, 8, 9],
        "recovered_block_ids": [7, 8],
        "missing_backup_block_ids": [],
        "skipped_block_ids": [9],
        "recovered_bytes": 32768,
        "recovery_copy_wall_ms": 0.25,
        "precision_tiering_enabled": True,
        "precision_policy": "top_ratio",
        "tier_fp16_block_ids": [7],
        "tier_int8_block_ids": [8],
        "tier_skip_block_ids": [9],
        "recovered_fp16_block_ids": [7],
        "recovered_int8_block_ids": [8],
        "missing_fp16_block_ids": [],
        "missing_int8_block_ids": [],
        "fp16_payload_bytes": 16384,
        "int8_payload_bytes": 10240,
        "effective_recovery_transfer_bytes": 26624,
    }
    path.write_text(json.dumps(recovery_event), encoding="utf-8")

    events = load_jsonl([path])
    with pytest.raises(AssertionError, match="skip-tier blocks"):
        validate_tiered_skip_unrecovered_requirement(events)


def test_debug_jsonl_validator_accepts_recovery_test_mutated_event(tmp_path):
    path = tmp_path / "mpr.jsonl"
    mutation_event = {
        "event": "recovery_test_mutated",
        "layer_name": "model.layers.0.self_attn.attn",
        "layer_event_idx": 4,
        "query_shape": [1, 4, 128],
        "window_query_shape": [4, 128],
        "window_query_len": 3,
        "kv_cache_shape": [2, 16, 32, 2, 128],
        "recovery_policy": "threshold_block",
        "recovery_topk": 1,
        "recovery_threshold": -1000000000.0,
        "recovery_test_mutate": "zero_selected",
        "recovery_test_mutated_scope": "selected",
        "recovery_test_mode": "mutate_only",
        "recovery_selected_block_ids": [7, 8, 9],
        "recovery_test_mutated_block_ids": [7, 8],
        "cpu_backup_block_count": 5,
        "cpu_backup_bytes": 81920,
        "cpu_backup_fp16_payload_bytes": 65536,
        "cpu_backup_int8_payload_bytes": 16384,
        "cpu_backup_int8_scale_bytes": 4096,
        "cpu_backup_total_actual_bytes": 86016,
        "valid_block_ids": [7, 8, 9],
        "finalized_block_ids": [7, 8],
        "score_candidate_block_ids": [7, 8],
        "observed_digest_block_ids": [7, 8],
        "missing_digest_blocks": [],
        "extra_digest_blocks": [],
    }
    path.write_text(json.dumps(mutation_event), encoding="utf-8")

    events = load_jsonl([path])
    validate_recovery_test_mutated_event(events[0])


def test_debug_jsonl_validator_accepts_zero_all_recovery_test_mutated_event(
    tmp_path,
):
    path = tmp_path / "mpr.jsonl"
    mutation_event = {
        "event": "recovery_test_mutated",
        "layer_name": "model.layers.0.self_attn.attn",
        "layer_event_idx": 4,
        "kv_cache_shape": [2, 16, 32, 2, 128],
        "recovery_policy": "threshold_block",
        "recovery_topk": 1,
        "recovery_threshold": -1000000000.0,
        "recovery_test_mutate": "zero_all",
        "recovery_test_mutated_scope": "all_kv_cache",
        "recovery_test_mode": "mutate_only",
        "recovery_selected_block_ids": [7, 8],
        "recovery_test_mutated_block_ids": list(range(16)),
    }
    path.write_text(json.dumps(mutation_event), encoding="utf-8")

    events = load_jsonl([path])
    validate_recovery_test_mutated_event(events[0])


def test_debug_jsonl_validator_accepts_recovery_skipped_event(tmp_path):
    path = tmp_path / "mpr.jsonl"
    recovery_event = {
        "event": "recovery_skipped",
        "layer_name": "model.layers.0.self_attn.attn",
        "layer_event_idx": 4,
        "skipped_reason": "cpu_backup_disabled",
        "query_shape": [1, 4, 128],
        "max_query_len": 1,
        "num_actual_tokens": 1,
        "kv_cache_shape": [2, 16, 32, 2, 128],
        "recovery_policy": "topk_block",
        "recovery_topk": 2,
        "recovery_threshold": 0.0,
        "cpu_backup_enabled": False,
        "scoring_enabled": True,
    }
    path.write_text(json.dumps(recovery_event), encoding="utf-8")

    events = load_jsonl([path])
    validate_recovery_skipped_event(events[0])


def test_debug_jsonl_validator_rejects_recovered_outside_selected(tmp_path):
    path = tmp_path / "mpr.jsonl"
    recovery_event = {
        "event": "recovery_materialized",
        "layer_name": "model.layers.0.self_attn.attn",
        "layer_event_idx": 4,
        "kv_cache_shape": [2, 16, 32, 2, 128],
        "recovery_policy": "topk_block",
        "recovery_topk": 2,
        "recovery_threshold": 0.0,
        "recovery_selected_block_ids": [7],
        "recovered_block_ids": [8],
        "missing_backup_block_ids": [],
        "skipped_block_ids": [],
        "recovered_bytes": 1024,
        "recovery_copy_wall_ms": 0.25,
    }
    path.write_text(json.dumps(recovery_event), encoding="utf-8")

    events = load_jsonl([path])
    with pytest.raises(AssertionError, match="subset"):
        validate_recovery_materialized_event(events[0])
