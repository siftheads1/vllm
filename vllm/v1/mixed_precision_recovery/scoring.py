# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tensor-only scoring helpers for MPR digest/query comparison."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import torch

from vllm.v1.mixed_precision_recovery.quest_packing import (
    PackedQuestDigestCache,
    pack_quest_metadata_cache,
)


QUEST_CUDA_SUPPORTED_GROUP_SIZES = frozenset({1, 4, 8})
QUEST_NHD_LAYOUT = 0
PACKED_ESTIMATE_PROFILE_TIMING_NAMES = (
    "quest_packed_estimate_output_alloc",
    "quest_packed_estimate_query_prepare",
    "quest_packed_estimate_custom_op",
    "quest_packed_estimate_transpose",
    "quest_packed_estimate_aggregate",
    "quest_packed_estimate_result_build",
)
_PACKED_ESTIMATE_PROFILE_TIMING: dict[str, dict[str, float | int]] = {
    name: {"count": 0, "total_ms": 0.0, "max_ms": 0.0}
    for name in PACKED_ESTIMATE_PROFILE_TIMING_NAMES
}


def _record_packed_estimate_profile_timing(name: str, start: float) -> None:
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    timing = _PACKED_ESTIMATE_PROFILE_TIMING[name]
    timing["count"] = int(timing["count"]) + 1
    timing["total_ms"] = float(timing["total_ms"]) + elapsed_ms
    timing["max_ms"] = max(float(timing["max_ms"]), elapsed_ms)


def get_packed_estimate_profile_timing() -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for name in PACKED_ESTIMATE_PROFILE_TIMING_NAMES:
        timing = _PACKED_ESTIMATE_PROFILE_TIMING[name]
        count = int(timing["count"])
        total_ms = float(timing["total_ms"])
        result[f"{name}_count"] = count
        result[f"{name}_total_ms"] = total_ms
        result[f"{name}_mean_ms"] = total_ms / count if count else 0.0
        result[f"{name}_max_ms"] = float(timing["max_ms"])
    return result


def reset_packed_estimate_profile_timing() -> None:
    for timing in _PACKED_ESTIMATE_PROFILE_TIMING.values():
        timing["count"] = 0
        timing["total_ms"] = 0.0
        timing["max_ms"] = 0.0


@dataclass(frozen=True)
class DigestScoreResult:
    """Structured result from a digest scoring backend.

    Attributes:
        block_scores: One score per packed digest block. Shape: ``[num_blocks]``.
        per_query_head_scores: Quest-style score before query-head aggregation.
            Shape: ``[num_blocks, num_q_heads]``.
        per_kv_head_scores: Query-head scores aggregated within each GQA/MQA KV
            head group. Shape: ``[num_blocks, num_kv_heads]``.
        score_agg: Query-head aggregation policy used to make ``block_scores``.
        scoring_backend: Backend implementation name.
        num_q_heads: Number of query heads in the scoring input.
        num_kv_heads: Number of KV heads in the digest input.
        group_size: Number of query heads sharing each KV head.
    """

    block_scores: torch.Tensor
    per_query_head_scores: torch.Tensor
    per_kv_head_scores: torch.Tensor
    score_agg: str
    scoring_backend: str
    num_q_heads: int
    num_kv_heads: int
    group_size: int


class DigestScoringBackend(Protocol):
    """Protocol for replaceable digest scoring backends."""

    name: str

    def estimate(
        self,
        *,
        query_window: torch.Tensor,
        digest_min: torch.Tensor,
        digest_max: torch.Tensor,
        score_agg: str,
        metadata_page_size: int | None = None,
    ) -> DigestScoreResult:
        """Estimate scores for packed digest tensors."""
        ...


def _validate_score_inputs(
    query_window: torch.Tensor,
    digest_min: torch.Tensor,
    digest_max: torch.Tensor,
    score_agg: str,
) -> tuple[int, int, int, int, int]:
    """Validate common score inputs and return shape metadata."""
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
    return num_blocks, num_q_heads, num_kv_heads, query_head_dim, group_size


def estimate_query_head_digest_scores(
    query_window: torch.Tensor,
    digest_min: torch.Tensor,
    digest_max: torch.Tensor,
    score_agg: str,
) -> torch.Tensor:
    """Estimate Quest-style digest scores before query-head aggregation.

    This helper intentionally knows nothing about sidecar dictionaries, layer
    names, request ids, or block ownership. It exposes the per-query-head tensor
    needed by GQA/MQA policies before any block-level top-k decision.

    Args:
        query_window: Rolling decode query average, shaped
            ``[num_q_heads, head_dim]``.
        digest_min: Lower digest bounds, shaped
            ``[num_blocks, num_kv_heads, head_dim]``.
        digest_max: Upper digest bounds, shaped
            ``[num_blocks, num_kv_heads, head_dim]``.
        score_agg: Query-head aggregation policy. It is validated here so direct
            callers fail early with the same contract as block-level scoring.

    Returns:
        Per-query-head scores shaped ``[num_blocks, num_q_heads]``.
    """
    (
        num_blocks,
        num_q_heads,
        num_kv_heads,
        query_head_dim,
        group_size,
    ) = _validate_score_inputs(query_window, digest_min, digest_max, score_agg)
    if num_blocks == 0:
        return torch.empty(
            (0, num_q_heads),
            dtype=query_window.dtype,
            device=query_window.device,
        )

    # vLLM GQA head order is treated as contiguous query-head groups per KV head:
    # q_head -> q_head // group_size.
    # query_by_kv_head: [num_kv_heads, group_size, head_dim].
    query_by_kv_head = query_window.reshape(num_kv_heads, group_size, query_head_dim)

    # Broadcast to [num_blocks, num_kv_heads, group_size, head_dim], matching
    # each query head to its owning KV-head digest before aggregation.
    query_terms = query_by_kv_head.unsqueeze(0)
    max_terms = query_terms * digest_max.unsqueeze(2)
    min_terms = query_terms * digest_min.unsqueeze(2)

    # [num_blocks, num_kv_heads, group_size] -> [num_blocks, num_q_heads].
    per_group_scores = torch.maximum(max_terms, min_terms).sum(dim=-1)
    return per_group_scores.reshape(num_blocks, num_q_heads)


def aggregate_query_head_scores(
    per_query_head_scores: torch.Tensor,
    *,
    num_kv_heads: int,
    score_agg: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate per-query-head scores into GQA groups and block scores.

    ``score_agg="max"`` implements the conservative GQA union policy: if any
    query head in a KV group scores a block highly, that signal survives the
    group aggregation, and block-level top-k sees the max over groups.

    Returns:
        A tuple ``(block_scores, per_kv_head_scores)`` with shapes
        ``[num_blocks]`` and ``[num_blocks, num_kv_heads]``.
    """
    if per_query_head_scores.ndim != 2:
        raise ValueError(
            "MPR scoring aggregation expects per_query_head_scores shaped "
            "[num_blocks, num_q_heads], got "
            f"{tuple(per_query_head_scores.shape)}."
        )
    if score_agg not in {"max", "mean"}:
        raise ValueError(
            "MPR scoring score_agg must be 'max' or 'mean', "
            f"got {score_agg!r}."
        )

    num_blocks = int(per_query_head_scores.shape[0])
    num_q_heads = int(per_query_head_scores.shape[1])
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise AssertionError(
            "MPR scoring requires num_q_heads to be a positive multiple of "
            f"num_kv_heads, got num_q_heads={num_q_heads}, "
            f"num_kv_heads={num_kv_heads}."
        )
    group_size = num_q_heads // num_kv_heads
    grouped_scores = per_query_head_scores.reshape(
        num_blocks,
        num_kv_heads,
        group_size,
    )

    if score_agg == "max":
        per_kv_head_scores = grouped_scores.max(dim=2).values
        block_scores = per_kv_head_scores.max(dim=1).values
        return block_scores, per_kv_head_scores

    per_kv_head_scores = grouped_scores.mean(dim=2)
    block_scores = per_kv_head_scores.mean(dim=1)
    return block_scores, per_kv_head_scores


def aggregate_packed_query_head_scores(
    packed_query_head_scores: torch.Tensor,
    *,
    num_kv_heads: int,
    score_agg: str,
) -> torch.Tensor:
    """Aggregate packed Quest output directly to block scores.

    This optimized helper consumes the CUDA Quest output layout
    ``[num_q_heads, num_blocks]`` and returns only ``block_scores``. It avoids
    materializing the reference/debug layout ``[num_blocks, num_q_heads]`` and
    the intermediate per-KV-head scores. It is equivalent to
    ``aggregate_query_head_scores(output.T.contiguous(), ...)`` for the block
    score result because all KV-head groups have the same GQA group size.
    """
    if packed_query_head_scores.ndim != 2:
        raise ValueError(
            "MPR packed scoring aggregation expects scores shaped "
            "[num_q_heads, num_blocks], got "
            f"{tuple(packed_query_head_scores.shape)}."
        )
    if score_agg not in {"max", "mean"}:
        raise ValueError(
            "MPR scoring score_agg must be 'max' or 'mean', "
            f"got {score_agg!r}."
        )
    num_q_heads = int(packed_query_head_scores.shape[0])
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise AssertionError(
            "MPR scoring requires num_q_heads to be a positive multiple of "
            f"num_kv_heads, got num_q_heads={num_q_heads}, "
            f"num_kv_heads={num_kv_heads}."
        )
    if score_agg == "max":
        return packed_query_head_scores.max(dim=0).values
    return packed_query_head_scores.mean(dim=0)


class TorchQuestScorer:
    """PyTorch reference scorer using Quest/ArkVale cuboid score semantics."""

    name = "torch_quest"

    def estimate(
        self,
        *,
        query_window: torch.Tensor,
        digest_min: torch.Tensor,
        digest_max: torch.Tensor,
        score_agg: str,
        metadata_page_size: int | None = None,
    ) -> DigestScoreResult:
        (
            _,
            num_q_heads,
            num_kv_heads,
            _,
            group_size,
        ) = _validate_score_inputs(query_window, digest_min, digest_max, score_agg)
        per_query_head_scores = estimate_query_head_digest_scores(
            query_window=query_window,
            digest_min=digest_min,
            digest_max=digest_max,
            score_agg=score_agg,
        )
        block_scores, per_kv_head_scores = aggregate_query_head_scores(
            per_query_head_scores,
            num_kv_heads=num_kv_heads,
            score_agg=score_agg,
        )
        return DigestScoreResult(
            block_scores=block_scores,
            per_query_head_scores=per_query_head_scores,
            per_kv_head_scores=per_kv_head_scores,
            score_agg=score_agg,
            scoring_backend=self.name,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            group_size=group_size,
        )


class QuestCudaScorer:
    """Quest estimate-kernel scorer reached through the vLLM custom op."""

    name = "quest_cuda"

    def estimate_packed(
        self,
        *,
        query_window: torch.Tensor,
        packed: PackedQuestDigestCache,
        num_kv_heads: int,
        score_agg: str,
        profile_enabled: bool = False,
    ) -> DigestScoreResult:
        """Reference packed Quest scorer with full head-score materialization.

        This original path preserves the debug-friendly result shape: it keeps
        per-query-head scores, transposes them to ``[num_blocks, num_q_heads]``,
        and builds per-KV-head scores before returning block scores. The
        optimized sidecar path should use ``estimate_packed_optimized`` when it
        only needs block-level scores.
        """
        if query_window.ndim != 2:
            raise ValueError(
                "quest_cuda packed scoring expects query_window shaped "
                f"[num_q_heads, head_dim], got {tuple(query_window.shape)}."
            )
        if score_agg not in {"max", "mean"}:
            raise ValueError(
                "MPR scoring score_agg must be 'max' or 'mean', "
                f"got {score_agg!r}."
            )
        num_blocks = packed.num_score_entries
        num_q_heads = int(query_window.shape[0])
        if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
            raise AssertionError(
                "MPR scoring requires num_q_heads to be a positive multiple of "
                f"num_kv_heads, got num_q_heads={num_q_heads}, "
                f"num_kv_heads={num_kv_heads}."
            )
        group_size = num_q_heads // num_kv_heads
        if group_size not in QUEST_CUDA_SUPPORTED_GROUP_SIZES:
            supported = sorted(QUEST_CUDA_SUPPORTED_GROUP_SIZES)
            raise RuntimeError(
                "quest_cuda scoring is unavailable for GQA group_size="
                f"{group_size}; supported group sizes are {supported}."
            )
        if num_blocks <= 0:
            per_query_head_scores = torch.empty(
                (0, num_q_heads),
                dtype=query_window.dtype,
                device=query_window.device,
            )
            block_scores, per_kv_head_scores = aggregate_query_head_scores(
                per_query_head_scores,
                num_kv_heads=num_kv_heads,
                score_agg=score_agg,
            )
            return DigestScoreResult(
                block_scores=block_scores,
                per_query_head_scores=per_query_head_scores,
                per_kv_head_scores=per_kv_head_scores,
                score_agg=score_agg,
                scoring_backend=self.name,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                group_size=group_size,
            )
        if not query_window.is_cuda:
            raise RuntimeError("quest_cuda scoring requires CUDA query tensors.")
        if packed.metadata_data.dtype != query_window.dtype:
            raise RuntimeError(
                "quest_cuda packed metadata dtype must match query dtype, got "
                f"{packed.metadata_data.dtype} and {query_window.dtype}."
            )
        if packed.metadata_data.device != query_window.device:
            raise RuntimeError(
                "quest_cuda packed metadata device must match query device, got "
                f"{packed.metadata_data.device} and {query_window.device}."
            )

        try:
            from vllm import _custom_ops as ops
        except Exception as exc:
            raise RuntimeError(
                "quest_cuda scoring could not import vLLM custom ops."
            ) from exc

        if not hasattr(ops, "mpr_estimate_attn_score"):
            raise RuntimeError(
                "quest_cuda scoring backend is selected, but the "
                "mpr_estimate_attn_score custom op wrapper is unavailable."
            )

        output_start = time.perf_counter() if profile_enabled else 0.0
        output = torch.empty(
            (num_q_heads, packed.num_score_entries),
            dtype=query_window.dtype,
            device=query_window.device,
        )
        if profile_enabled:
            _record_packed_estimate_profile_timing(
                "quest_packed_estimate_output_alloc",
                output_start,
            )

        query_prepare_start = time.perf_counter() if profile_enabled else 0.0
        query_input = query_window.unsqueeze(0).contiguous()
        if profile_enabled:
            _record_packed_estimate_profile_timing(
                "quest_packed_estimate_query_prepare",
                query_prepare_start,
            )

        custom_op_start = time.perf_counter() if profile_enabled else 0.0
        ops.mpr_estimate_attn_score(
            query_input,
            output,
            packed.metadata_data,
            packed.metadata_indices,
            packed.metadata_indptr,
            packed.metadata_last_page_len,
            packed.metadata_last_page_idx,
            QUEST_NHD_LAYOUT,
        )
        if profile_enabled:
            _record_packed_estimate_profile_timing(
                "quest_packed_estimate_custom_op",
                custom_op_start,
            )

        transpose_start = time.perf_counter() if profile_enabled else 0.0
        per_query_head_scores = output.transpose(0, 1).contiguous()
        if profile_enabled:
            _record_packed_estimate_profile_timing(
                "quest_packed_estimate_transpose",
                transpose_start,
            )

        aggregate_start = time.perf_counter() if profile_enabled else 0.0
        block_scores, per_kv_head_scores = aggregate_query_head_scores(
            per_query_head_scores,
            num_kv_heads=num_kv_heads,
            score_agg=score_agg,
        )
        if profile_enabled:
            _record_packed_estimate_profile_timing(
                "quest_packed_estimate_aggregate",
                aggregate_start,
            )

        result_start = time.perf_counter() if profile_enabled else 0.0
        result = DigestScoreResult(
            block_scores=block_scores,
            per_query_head_scores=per_query_head_scores,
            per_kv_head_scores=per_kv_head_scores,
            score_agg=score_agg,
            scoring_backend=self.name,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            group_size=group_size,
        )
        if profile_enabled:
            _record_packed_estimate_profile_timing(
                "quest_packed_estimate_result_build",
                result_start,
            )
        return result

    def estimate_packed_optimized(
        self,
        *,
        query_window: torch.Tensor,
        packed: PackedQuestDigestCache,
        num_kv_heads: int,
        score_agg: str,
        profile_enabled: bool = False,
        materialize_head_scores: bool = False,
    ) -> DigestScoreResult:
        """Optimized packed Quest scorer for the block-score hot path.

        When ``materialize_head_scores`` is false, this path keeps the custom
        op's native ``[num_q_heads, num_blocks]`` output layout and computes
        ``block_scores`` directly from it. That avoids the reference path's
        transpose/contiguous copy and per-KV-head intermediate tensor. The
        returned head-score tensors are empty placeholders in this mode and must
        not be used for debug head top-k output.

        Set ``materialize_head_scores=True`` to fall back to ``estimate_packed``
        for logging/debug paths that need ``per_query_head_scores`` or
        ``per_kv_head_scores``.
        """
        if materialize_head_scores:
            return self.estimate_packed(
                query_window=query_window,
                packed=packed,
                num_kv_heads=num_kv_heads,
                score_agg=score_agg,
                profile_enabled=profile_enabled,
            )
        if query_window.ndim != 2:
            raise ValueError(
                "quest_cuda packed scoring expects query_window shaped "
                f"[num_q_heads, head_dim], got {tuple(query_window.shape)}."
            )
        if score_agg not in {"max", "mean"}:
            raise ValueError(
                "MPR scoring score_agg must be 'max' or 'mean', "
                f"got {score_agg!r}."
            )
        num_blocks = packed.num_score_entries
        num_q_heads = int(query_window.shape[0])
        if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
            raise AssertionError(
                "MPR scoring requires num_q_heads to be a positive multiple of "
                f"num_kv_heads, got num_q_heads={num_q_heads}, "
                f"num_kv_heads={num_kv_heads}."
            )
        group_size = num_q_heads // num_kv_heads
        if group_size not in QUEST_CUDA_SUPPORTED_GROUP_SIZES:
            supported = sorted(QUEST_CUDA_SUPPORTED_GROUP_SIZES)
            raise RuntimeError(
                "quest_cuda scoring is unavailable for GQA group_size="
                f"{group_size}; supported group sizes are {supported}."
            )
        if num_blocks <= 0:
            block_scores = torch.empty(
                (0,),
                dtype=query_window.dtype,
                device=query_window.device,
            )
            per_query_head_scores = torch.empty(
                (0, num_q_heads),
                dtype=query_window.dtype,
                device=query_window.device,
            )
            per_kv_head_scores = torch.empty(
                (0, num_kv_heads),
                dtype=query_window.dtype,
                device=query_window.device,
            )
            return DigestScoreResult(
                block_scores=block_scores,
                per_query_head_scores=per_query_head_scores,
                per_kv_head_scores=per_kv_head_scores,
                score_agg=score_agg,
                scoring_backend=self.name,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                group_size=group_size,
            )
        if not query_window.is_cuda:
            raise RuntimeError("quest_cuda scoring requires CUDA query tensors.")
        if packed.metadata_data.dtype != query_window.dtype:
            raise RuntimeError(
                "quest_cuda packed metadata dtype must match query dtype, got "
                f"{packed.metadata_data.dtype} and {query_window.dtype}."
            )
        if packed.metadata_data.device != query_window.device:
            raise RuntimeError(
                "quest_cuda packed metadata device must match query device, got "
                f"{packed.metadata_data.device} and {query_window.device}."
            )

        try:
            from vllm import _custom_ops as ops
        except Exception as exc:
            raise RuntimeError(
                "quest_cuda scoring could not import vLLM custom ops."
            ) from exc

        if not hasattr(ops, "mpr_estimate_attn_score"):
            raise RuntimeError(
                "quest_cuda scoring backend is selected, but the "
                "mpr_estimate_attn_score custom op wrapper is unavailable."
            )

        output_start = time.perf_counter() if profile_enabled else 0.0
        output = torch.empty(
            (num_q_heads, packed.num_score_entries),
            dtype=query_window.dtype,
            device=query_window.device,
        )
        if profile_enabled:
            _record_packed_estimate_profile_timing(
                "quest_packed_estimate_output_alloc",
                output_start,
            )

        query_prepare_start = time.perf_counter() if profile_enabled else 0.0
        query_input = query_window.unsqueeze(0).contiguous()
        if profile_enabled:
            _record_packed_estimate_profile_timing(
                "quest_packed_estimate_query_prepare",
                query_prepare_start,
            )

        custom_op_start = time.perf_counter() if profile_enabled else 0.0
        ops.mpr_estimate_attn_score(
            query_input,
            output,
            packed.metadata_data,
            packed.metadata_indices,
            packed.metadata_indptr,
            packed.metadata_last_page_len,
            packed.metadata_last_page_idx,
            QUEST_NHD_LAYOUT,
        )
        if profile_enabled:
            _record_packed_estimate_profile_timing(
                "quest_packed_estimate_custom_op",
                custom_op_start,
            )

        aggregate_start = time.perf_counter() if profile_enabled else 0.0
        block_scores = aggregate_packed_query_head_scores(
            output,
            num_kv_heads=num_kv_heads,
            score_agg=score_agg,
        )
        if profile_enabled:
            _record_packed_estimate_profile_timing(
                "quest_packed_estimate_aggregate",
                aggregate_start,
            )

        result_start = time.perf_counter() if profile_enabled else 0.0
        empty_head_scores = torch.empty(
            (num_blocks, 0),
            dtype=query_window.dtype,
            device=query_window.device,
        )
        result = DigestScoreResult(
            block_scores=block_scores,
            per_query_head_scores=empty_head_scores,
            per_kv_head_scores=empty_head_scores,
            score_agg=score_agg,
            scoring_backend=self.name,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            group_size=group_size,
        )
        if profile_enabled:
            _record_packed_estimate_profile_timing(
                "quest_packed_estimate_result_build",
                result_start,
            )
        return result

    def estimate(
        self,
        *,
        query_window: torch.Tensor,
        digest_min: torch.Tensor,
        digest_max: torch.Tensor,
        score_agg: str,
        metadata_page_size: int | None = None,
    ) -> DigestScoreResult:
        (
            num_blocks,
            num_q_heads,
            num_kv_heads,
            _,
            group_size,
        ) = _validate_score_inputs(query_window, digest_min, digest_max, score_agg)
        if num_blocks == 0:
            per_query_head_scores = torch.empty(
                (0, num_q_heads),
                dtype=query_window.dtype,
                device=query_window.device,
            )
            block_scores, per_kv_head_scores = aggregate_query_head_scores(
                per_query_head_scores,
                num_kv_heads=num_kv_heads,
                score_agg=score_agg,
            )
            return DigestScoreResult(
                block_scores=block_scores,
                per_query_head_scores=per_query_head_scores,
                per_kv_head_scores=per_kv_head_scores,
                score_agg=score_agg,
                scoring_backend=self.name,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                group_size=group_size,
            )
        if metadata_page_size is None or metadata_page_size <= 0:
            raise ValueError(
                "quest_cuda scoring requires a positive metadata_page_size "
                "matching the digest/KV block size."
            )
        packed = pack_quest_metadata_cache(
            digest_min=digest_min,
            digest_max=digest_max,
            metadata_page_size=metadata_page_size,
            add_guard_entry=True,
        )
        return self.estimate_packed(
            query_window=query_window,
            packed=packed,
            num_kv_heads=num_kv_heads,
            score_agg=score_agg,
        )


def get_digest_scoring_backend(name: str) -> DigestScoringBackend:
    """Return the configured digest scoring backend."""
    if name == TorchQuestScorer.name:
        return TorchQuestScorer()
    if name == QuestCudaScorer.name:
        return QuestCudaScorer()
    raise ValueError(
        "MPR scoring backend must be 'torch_quest' or 'quest_cuda', "
        f"got {name!r}."
    )


def estimate_digest_score_result(
    query_window: torch.Tensor,
    digest_min: torch.Tensor,
    digest_max: torch.Tensor,
    score_agg: str,
    scoring_backend: str = TorchQuestScorer.name,
    metadata_page_size: int | None = None,
) -> DigestScoreResult:
    """Estimate digest scores and return structured backend metadata."""
    backend = get_digest_scoring_backend(scoring_backend)
    return backend.estimate(
        query_window=query_window,
        digest_min=digest_min,
        digest_max=digest_max,
        score_agg=score_agg,
        metadata_page_size=metadata_page_size,
    )


def estimate_digest_scores(
    query_window: torch.Tensor,
    digest_min: torch.Tensor,
    digest_max: torch.Tensor,
    score_agg: str,
) -> torch.Tensor:
    """Estimate one layer-local score per digest block.

    This legacy wrapper preserves the Step 1.4 API. New code that needs backend
    metadata or GQA group scores should call ``estimate_digest_score_result``.
    """
    return estimate_digest_score_result(
        query_window=query_window,
        digest_min=digest_min,
        digest_max=digest_max,
        score_agg=score_agg,
    ).block_scores
