# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.mixed_precision_recovery.config import MPRConfig
from vllm.v1.mixed_precision_recovery.scoring import estimate_digest_scores
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
