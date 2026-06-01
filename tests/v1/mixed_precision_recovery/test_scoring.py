# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.mixed_precision_recovery.config import MPRConfig
from vllm.v1.mixed_precision_recovery.quest_packing import (
    pack_quest_metadata_cache,
)
from vllm.v1.mixed_precision_recovery.scoring import (
    QuestCudaScorer,
    aggregate_query_head_scores,
    estimate_digest_score_result,
    estimate_digest_scores,
    estimate_query_head_digest_scores,
    get_digest_scoring_backend,
)
from vllm.v1.mixed_precision_recovery.sidecar import BlockDigest, RecoverySidecar


def _manual_scores(
    query_window: torch.Tensor,
    digest_min: torch.Tensor,
    digest_max: torch.Tensor,
    score_agg: str,
) -> torch.Tensor:
    num_blocks, num_kv_heads, head_dim = digest_min.shape
    num_q_heads = query_window.shape[0]
    group_size = num_q_heads // num_kv_heads
    query_by_kv = query_window.reshape(num_kv_heads, group_size, head_dim)
    per_head = []
    for block_idx in range(num_blocks):
        block_scores = []
        for kv_head_idx in range(num_kv_heads):
            for group_idx in range(group_size):
                query = query_by_kv[kv_head_idx, group_idx]
                max_term = query * digest_max[block_idx, kv_head_idx]
                min_term = query * digest_min[block_idx, kv_head_idx]
                block_scores.append(torch.maximum(max_term, min_term).sum())
        stacked = torch.stack(block_scores)
        if score_agg == "max":
            per_head.append(stacked.max())
        else:
            per_head.append(stacked.mean())
    return torch.stack(per_head)


def test_estimate_digest_scores_matches_manual_max_and_mean():
    query_window = torch.tensor(
        [
            [1.0, -2.0],
            [0.5, 3.0],
            [-1.5, 2.0],
            [4.0, -0.25],
        ]
    )
    digest_min = torch.tensor(
        [
            [[-1.0, -0.5], [-2.0, -1.0]],
            [[0.25, -3.0], [-0.5, 1.0]],
        ]
    )
    digest_max = torch.tensor(
        [
            [[2.0, 1.5], [1.0, 3.0]],
            [[1.25, 0.5], [2.5, 4.0]],
        ]
    )

    torch.testing.assert_close(
        estimate_digest_scores(query_window, digest_min, digest_max, "max"),
        _manual_scores(query_window, digest_min, digest_max, "max"),
    )
    torch.testing.assert_close(
        estimate_digest_scores(query_window, digest_min, digest_max, "mean"),
        _manual_scores(query_window, digest_min, digest_max, "mean"),
    )


def test_mpr_config_parses_score_granularity(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_SCORE_GRANULARITY", "kv_head")

    config = MPRConfig.from_env()

    assert config.score_granularity == "kv_head"


def test_mpr_config_defaults_to_quest_style_digest_and_kv_head_scores():
    config = MPRConfig()

    assert config.digest_kind == "raw_minmax"
    assert config.score_granularity == "kv_head"
    assert config.recent_tokens == 64


def test_mpr_config_accepts_quest_cuda_backend(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_SCORING_BACKEND", "quest_cuda")

    config = MPRConfig.from_env()

    assert config.scoring_backend == "quest_cuda"
    assert isinstance(get_digest_scoring_backend(config.scoring_backend), QuestCudaScorer)


def test_estimate_digest_score_result_exposes_gqa_metadata():
    query_window = torch.tensor(
        [
            [1.0, -2.0],
            [0.5, 3.0],
            [-1.5, 2.0],
            [4.0, -0.25],
        ]
    )
    digest_min = torch.tensor([[[-1.0, -0.5], [-2.0, -1.0]]])
    digest_max = torch.tensor([[[2.0, 1.5], [1.0, 3.0]]])

    result = estimate_digest_score_result(
        query_window,
        digest_min,
        digest_max,
        "max",
    )

    assert result.scoring_backend == "torch_quest"
    assert result.num_q_heads == 4
    assert result.num_kv_heads == 2
    assert result.group_size == 2
    assert list(result.block_scores.shape) == [1]
    assert list(result.per_query_head_scores.shape) == [1, 4]
    assert list(result.per_kv_head_scores.shape) == [1, 2]
    torch.testing.assert_close(
        result.block_scores,
        estimate_digest_scores(query_window, digest_min, digest_max, "max"),
    )


def test_query_head_aggregation_uses_conservative_gqa_union_for_max():
    per_query_head_scores = torch.tensor(
        [
            [1.0, 10.0, 2.0, 3.0],
            [4.0, 1.0, 9.0, 7.0],
        ]
    )

    max_block_scores, max_group_scores = aggregate_query_head_scores(
        per_query_head_scores,
        num_kv_heads=2,
        score_agg="max",
    )
    mean_block_scores, mean_group_scores = aggregate_query_head_scores(
        per_query_head_scores,
        num_kv_heads=2,
        score_agg="mean",
    )

    torch.testing.assert_close(
        max_group_scores,
        torch.tensor([[10.0, 3.0], [4.0, 9.0]]),
    )
    torch.testing.assert_close(max_block_scores, torch.tensor([10.0, 9.0]))
    torch.testing.assert_close(
        mean_group_scores,
        torch.tensor([[5.5, 2.5], [2.5, 8.0]]),
    )
    torch.testing.assert_close(mean_block_scores, torch.tensor([4.0, 5.25]))


def test_query_head_scores_are_available_before_block_aggregation():
    query_window = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
        ]
    )
    digest_min = torch.tensor([[[-3.0, -1.0]]])
    digest_max = torch.tensor([[[2.0, 4.0]]])

    per_query_head_scores = estimate_query_head_digest_scores(
        query_window,
        digest_min,
        digest_max,
        "max",
    )

    torch.testing.assert_close(
        per_query_head_scores,
        torch.tensor([[2.0, 3.0]]),
    )


def test_sidecar_head_score_debug_fields_for_kv_head_granularity():
    sidecar = RecoverySidecar(
        config=MPRConfig(
            enabled=True,
            topk=1,
            score_granularity="kv_head",
        )
    )
    query_window = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 2.0],
            [0.0, -3.0],
        ]
    )
    digest_min = torch.tensor(
        [
            [[-2.0, -1.0], [-1.0, -4.0]],
            [[-5.0, -1.0], [-1.0, -1.0]],
        ]
    )
    digest_max = torch.tensor(
        [
            [[3.0, 1.0], [1.0, 2.0]],
            [[1.0, 1.0], [1.0, 7.0]],
        ]
    )
    score_result = sidecar._scoring_backend.estimate(
        query_window=query_window,
        digest_min=digest_min,
        digest_max=digest_max,
        score_agg="max",
    )

    fields = sidecar._head_score_debug_fields(
        score_result=score_result,
        physical_block_ids=[10, 11],
        topk=1,
    )

    assert fields["num_score_heads"] == 2
    assert fields["head_score_count"] == 4
    assert fields["topk_block_ids_by_head"] == [[11], [11]]
    assert len(fields["topk_scores_by_head"]) == 2


def test_sidecar_head_score_debug_fields_for_query_head_granularity():
    sidecar = RecoverySidecar(
        config=MPRConfig(
            enabled=True,
            topk=1,
            score_granularity="query_head",
        )
    )
    score_result = estimate_digest_score_result(
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
        torch.tensor([[[-3.0, -1.0]], [[-1.0, -1.0]]]),
        torch.tensor([[[2.0, 4.0]], [[5.0, 1.0]]]),
        "max",
    )

    fields = sidecar._head_score_debug_fields(
        score_result=score_result,
        physical_block_ids=[20, 21],
        topk=1,
    )

    assert fields["num_score_heads"] == 2
    assert fields["head_score_count"] == 4
    assert fields["topk_block_ids_by_head"] == [[21], [20]]


def test_sidecar_selects_logical_score_candidates_after_recent_tail():
    sidecar = RecoverySidecar(
        config=MPRConfig(
            enabled=True,
            recent_tokens=64,
        )
    )
    context = sidecar._request_block_context(
        attn_metadata=SimpleNamespace(
            max_query_len=1,
            num_actual_tokens=1,
            num_reqs=1,
            seq_lens=torch.tensor([161]),
            block_table=torch.tensor([[10, 4, 7, 8, 9, 99, 0, 0]]),
        ),
        block_size=32,
    )

    assert context.valid_block_ids == [10, 4, 7, 8, 9, 99]
    assert context.finalized_block_ids == [10, 4, 7, 8, 9]
    assert context.protected_tail_entries == 2
    assert context.protected_block_ids == [9, 99]
    assert context.score_candidate_block_ids == [10, 4, 7, 8]


def test_sidecar_packs_candidate_digests_in_logical_order():
    layer_name = "model.layers.0.self_attn.attn"
    sidecar = RecoverySidecar(config=MPRConfig(enabled=True))
    sidecar._digest_cache[layer_name] = {
        4: BlockDigest(
            digest_min=torch.full((1, 2), -4.0),
            digest_max=torch.full((1, 2), 4.0),
            valid_token_count=32,
            block_size=32,
            layer_event_idx=0,
            digest_kind="raw_minmax",
        ),
        7: BlockDigest(
            digest_min=torch.full((1, 2), -7.0),
            digest_max=torch.full((1, 2), 7.0),
            valid_token_count=32,
            block_size=32,
            layer_event_idx=0,
            digest_kind="raw_minmax",
        ),
        10: BlockDigest(
            digest_min=torch.full((1, 2), -10.0),
            digest_max=torch.full((1, 2), 10.0),
            valid_token_count=32,
            block_size=32,
            layer_event_idx=0,
            digest_kind="raw_minmax",
        ),
    }

    block_ids, digest_min, digest_max = sidecar._pack_layer_digests(
        layer_name,
        device=torch.device("cpu"),
        dtype=torch.float32,
        candidate_block_ids=[10, 4, 123, 7],
    )

    assert block_ids == [10, 4, 7]
    torch.testing.assert_close(digest_min[:, 0, 0], torch.tensor([-10.0, -4.0, -7.0]))
    torch.testing.assert_close(digest_max[:, 0, 0], torch.tensor([10.0, 4.0, 7.0]))


def test_quest_metadata_packing_adds_guard_entry():
    digest_min = torch.tensor(
        [
            [[-1.0, -2.0], [-3.0, -4.0]],
            [[-5.0, -6.0], [-7.0, -8.0]],
            [[-9.0, -10.0], [-11.0, -12.0]],
        ]
    )
    digest_max = digest_min.abs()

    packed = pack_quest_metadata_cache(
        digest_min=digest_min,
        digest_max=digest_max,
        metadata_page_size=2,
        entry_block_ids=[10, 4, 7],
    )

    assert list(packed.metadata_data.shape) == [2, 2, 2, 2, 2]
    assert packed.entry_block_ids == [10, 4, 7]
    assert packed.num_score_entries == 3
    assert packed.num_packed_entries == 4
    assert packed.metadata_last_page_len == 2
    assert packed.metadata_last_page_idx == 1
    assert packed.metadata_indices.tolist() == [0, 1]
    assert packed.metadata_indptr.tolist() == [0, 2]
    torch.testing.assert_close(packed.metadata_data[0, 0, 0], digest_max[0])
    torch.testing.assert_close(packed.metadata_data[0, 1, 0], digest_min[0])
    torch.testing.assert_close(packed.metadata_data[1, 0, 0], digest_max[2])
    torch.testing.assert_close(packed.metadata_data[1, 1, 0], digest_min[2])
    torch.testing.assert_close(
        packed.metadata_data[1, :, 1],
        torch.zeros_like(packed.metadata_data[1, :, 1]),
    )


def test_quest_cuda_backend_fails_clearly_without_cuda_tensor():
    scorer = QuestCudaScorer()

    with pytest.raises(RuntimeError, match="requires CUDA"):
        scorer.estimate(
            query_window=torch.ones(4, 2),
            digest_min=torch.zeros(2, 1, 2),
            digest_max=torch.ones(2, 1, 2),
            score_agg="max",
            metadata_page_size=2,
        )


def test_quest_cuda_backend_rejects_unsupported_gqa_group_size():
    scorer = QuestCudaScorer()

    with pytest.raises(RuntimeError, match="group_size=2"):
        scorer.estimate(
            query_window=torch.ones(4, 2),
            digest_min=torch.zeros(2, 2, 2),
            digest_max=torch.ones(2, 2, 2),
            score_agg="max",
            metadata_page_size=2,
        )


def test_estimate_digest_scores_rejects_invalid_head_grouping():
    query_window = torch.ones(3, 2)
    digest_min = torch.zeros(1, 2, 2)
    digest_max = torch.ones(1, 2, 2)

    with pytest.raises(AssertionError, match="num_q_heads"):
        estimate_digest_scores(query_window, digest_min, digest_max, "max")


def test_sidecar_query_window_uses_available_then_recent_queries():
    sidecar = RecoverySidecar(
        config=MPRConfig(enabled=True, window_size=2, topk=1, score_agg="mean")
    )
    layer_name = "model.layers.0.self_attn.attn"
    sidecar._digest_cache[layer_name] = {
        7: BlockDigest(
            digest_min=torch.zeros(1, 2),
            digest_max=torch.ones(1, 2),
            valid_token_count=4,
            block_size=4,
            layer_event_idx=1,
        )
    }
    metadata = SimpleNamespace(max_query_len=1, num_actual_tokens=1)

    sidecar.observe_query(layer_name, torch.tensor([[[1.0, 3.0]]]), metadata)
    assert len(sidecar._query_windows[layer_name]) == 1

    sidecar.observe_query(layer_name, torch.tensor([[[5.0, 7.0]]]), metadata)
    assert len(sidecar._query_windows[layer_name]) == 2

    sidecar.observe_query(layer_name, torch.tensor([[[9.0, 11.0]]]), metadata)
    window_values = list(sidecar._query_windows[layer_name])
    assert len(window_values) == 2
    torch.testing.assert_close(window_values[0], torch.tensor([[5.0, 7.0]]))
    torch.testing.assert_close(window_values[1], torch.tensor([[9.0, 11.0]]))
    assert sidecar.counters["score_estimated"] == 3


def test_sidecar_query_scoring_skips_non_single_request_decode():
    sidecar = RecoverySidecar(config=MPRConfig(enabled=True))
    metadata = SimpleNamespace(max_query_len=1, num_actual_tokens=2)

    sidecar.observe_query(
        "model.layers.0.self_attn.attn",
        torch.ones(2, 1, 2),
        metadata,
    )

    assert sidecar.counters["score_skipped"] == 1
    assert "model.layers.0.self_attn.attn" not in sidecar._query_windows


def test_sidecar_score_debug_fields_describe_block_table_state():
    sidecar = RecoverySidecar(config=MPRConfig(enabled=True, recent_tokens=0))
    metadata = SimpleNamespace(
        max_query_len=1,
        num_actual_tokens=1,
        seq_lens=torch.tensor([33]),
        block_table=torch.tensor([[10, 11, 12, 99]]),
    )
    request_context = sidecar._request_block_context(
        attn_metadata=metadata,
        block_size=16,
    )

    fields = sidecar._score_block_debug_fields(
        request_context=request_context,
        observed_digest_block_ids=[10, 12, 20],
    )

    assert fields["num_reqs"] == 1
    assert fields["seq_lens"] == [33]
    assert fields["block_table_row"] == [10, 11, 12, 99]
    assert fields["valid_block_ids"] == [10, 11, 12]
    assert fields["finalized_block_ids"] == [10, 11]
    assert fields["score_candidate_block_ids"] == [10, 11]
    assert fields["observed_digest_block_ids"] == [10, 12, 20]
    assert fields["missing_digest_blocks"] == [11]
    assert fields["extra_digest_blocks"] == [12, 20]
