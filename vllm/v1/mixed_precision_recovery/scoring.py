# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tensor-only scoring helpers for MPR digest/query comparison."""

from __future__ import annotations

import torch


def estimate_digest_scores(
    query_window: torch.Tensor,
    digest_min: torch.Tensor,
    digest_max: torch.Tensor,
    score_agg: str,
) -> torch.Tensor:
    """Estimate one layer-local score per digest block.

    This helper intentionally knows nothing about sidecar dictionaries, layer
    names, request ids, or block ownership. It is the replaceable Step 1.4
    scoring core; callers are responsible for packing query/digest inputs.
    The returned scores are per physical KV block within one layer, not per-head
    scores. Query-head scores are an intermediate value that is aggregated away
    according to ``score_agg``.

    Args:
        query_window: Rolling decode query average, shaped
            ``[num_q_heads, head_dim]``.
        digest_min: Lower digest bounds, shaped
            ``[num_blocks, num_kv_heads, head_dim]``.
        digest_max: Upper digest bounds, shaped
            ``[num_blocks, num_kv_heads, head_dim]``.
        score_agg: Query-head aggregation policy. ``"max"`` preserves the
            strongest query-head signal. ``"mean"`` is closer to ArkVale's
            default all-head averaging when it returns one group.

    Returns:
        Layer-local block scores shaped ``[num_blocks]``.
    """
    if query_window.ndim != 2:
        raise ValueError(
            "MPR scoring expects query_window shaped "
            f"[num_q_heads, head_dim], got {tuple(query_window.shape)}."
        )
    if digest_min.ndim != 3 or digest_max.ndim != 3:
        raise ValueError(
            "MPR scoring expects digest tensors shaped "
            "[num_blocks, num_kv_heads, head_dim]."
        )
    if digest_min.shape != digest_max.shape:
        raise ValueError(
            "MPR scoring requires digest_min and digest_max to have the "
            f"same shape, got {tuple(digest_min.shape)} and "
            f"{tuple(digest_max.shape)}."
        )
    if score_agg not in {"max", "mean"}:
        raise ValueError(
            "MPR scoring score_agg must be 'max' or 'mean', "
            f"got {score_agg!r}."
        )

    num_q_heads = int(query_window.shape[0])
    query_head_dim = int(query_window.shape[1])
    num_blocks = int(digest_min.shape[0])
    num_kv_heads = int(digest_min.shape[1])
    digest_head_dim = int(digest_min.shape[2])
    if num_blocks == 0:
        return torch.empty(0, dtype=query_window.dtype, device=query_window.device)
    if query_head_dim != digest_head_dim:
        raise ValueError(
            "MPR scoring head_dim mismatch: "
            f"query={query_head_dim}, digest={digest_head_dim}."
        )
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise AssertionError(
            "MPR scoring requires num_q_heads to be a positive multiple of "
            f"num_kv_heads, got num_q_heads={num_q_heads}, "
            f"num_kv_heads={num_kv_heads}."
        )

    group_size = num_q_heads // num_kv_heads
    # query_by_kv_head: [num_kv_heads, group_size, head_dim].
    query_by_kv_head = query_window.reshape(num_kv_heads, group_size, query_head_dim)

    # Broadcast to [num_blocks, num_kv_heads, group_size, head_dim], matching
    # each query head to its owning KV head before query-head aggregation.
    query_terms = query_by_kv_head.unsqueeze(0)
    max_terms = query_terms * digest_max.unsqueeze(2)
    min_terms = query_terms * digest_min.unsqueeze(2)

    # per_query_head_scores is an intermediate only:
    # [num_blocks, num_kv_heads, group_size].
    # The returned output is layer-local block scoring, not head-level scoring.
    per_query_head_scores = torch.maximum(max_terms, min_terms).sum(dim=-1)
    flattened_scores = per_query_head_scores.reshape(num_blocks, num_q_heads)
    if score_agg == "max":
        return flattened_scores.max(dim=1).values
    return flattened_scores.mean(dim=1)
