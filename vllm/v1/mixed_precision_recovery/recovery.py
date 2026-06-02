# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-backup based KV recovery helpers for MPR."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import torch

from vllm.v1.mixed_precision_recovery.cpu_backup import (
    CPUBackupKey,
    CPUBackupStore,
)
from vllm.v1.mixed_precision_recovery.scoring import DigestScoreResult


class RecoveryScoreResult(Protocol):
    """Minimal scoring result shape consumed by recovery selection."""

    block_scores: torch.Tensor


@dataclass(frozen=True)
class RecoveryResult:
    """Result metadata from one recovery materialization attempt."""

    selected_block_ids: list[int]
    recovered_block_ids: list[int]
    missing_backup_block_ids: list[int]
    skipped_block_ids: list[int]
    recovered_bytes: int
    copy_wall_seconds: float


def select_recovery_block_ids(
    *,
    score_result: DigestScoreResult | RecoveryScoreResult,
    physical_block_ids: list[int],
    policy: str,
    topk: int,
    threshold: float,
) -> list[int]:
    """Select physical block ids for M3 whole-block recovery.

    ``topk_block`` returns the highest-scoring block ids in score order.
    ``threshold_block`` returns all candidate block ids whose block score is
    greater than or equal to ``threshold``, preserving candidate order. This
    mirrors the eventual precision-tier policy more closely than top-k-only
    selection while keeping M3 materialization deterministic.
    """
    scores = score_result.block_scores
    if scores.ndim != 1:
        raise ValueError(
            "MPR recovery expects block_scores shaped [num_blocks], got "
            f"{tuple(scores.shape)}."
        )
    if len(physical_block_ids) != int(scores.numel()):
        raise ValueError(
            "MPR recovery requires physical_block_ids length to match "
            f"block_scores, got {len(physical_block_ids)} ids and "
            f"{int(scores.numel())} scores."
        )
    if topk < 1:
        raise ValueError(f"MPR recovery topk must be >= 1, got {topk}.")
    if policy == "topk_block":
        k = min(topk, int(scores.numel()))
        if k == 0:
            return []
        _, indices = torch.topk(scores, k=k)
        return [physical_block_ids[int(index)] for index in indices.detach().cpu()]
    if policy == "threshold_block":
        selected = torch.nonzero(scores >= threshold, as_tuple=False).reshape(-1)
        return [physical_block_ids[int(index)] for index in selected.detach().cpu()]
    raise ValueError(
        "MPR recovery policy must be 'topk_block' or 'threshold_block', "
        f"got {policy!r}."
    )


class BlockRecoveryManager:
    """Materialize selected full KV blocks from CPU backup into GPU KV cache."""

    def recover(
        self,
        *,
        score_result: DigestScoreResult | RecoveryScoreResult,
        physical_block_ids: list[int],
        kv_cache: torch.Tensor,
        cpu_backup_store: CPUBackupStore,
        layer_name: str,
        policy: str,
        topk: int,
        threshold: float,
    ) -> RecoveryResult:
        """Recover selected whole physical blocks into ``kv_cache``."""
        selected_block_ids = select_recovery_block_ids(
            score_result=score_result,
            physical_block_ids=physical_block_ids,
            policy=policy,
            topk=topk,
            threshold=threshold,
        )
        return self.materialize_blocks(
            selected_block_ids=selected_block_ids,
            kv_cache=kv_cache,
            cpu_backup_store=cpu_backup_store,
            layer_name=layer_name,
        )

    def materialize_blocks(
        self,
        *,
        selected_block_ids: list[int],
        kv_cache: torch.Tensor,
        cpu_backup_store: CPUBackupStore,
        layer_name: str,
    ) -> RecoveryResult:
        """Copy selected CPU backup blocks into ``kv_cache`` in-place."""
        _validate_kv_cache(kv_cache)

        recovered_block_ids: list[int] = []
        missing_backup_block_ids: list[int] = []
        skipped_block_ids: list[int] = []
        recovered_bytes = 0
        copy_wall_seconds = 0.0
        num_blocks = int(kv_cache.shape[1])

        for block_id in selected_block_ids:
            block_id = int(block_id)
            if block_id < 0 or block_id >= num_blocks:
                skipped_block_ids.append(block_id)
                continue

            key = CPUBackupKey(
                layer_name=layer_name,
                physical_block_id=block_id,
            )
            backup = cpu_backup_store.get(key)
            if backup is None:
                missing_backup_block_ids.append(block_id)
                continue

            target = kv_cache[:, block_id]
            if tuple(backup.shape) != tuple(target.shape):
                skipped_block_ids.append(block_id)
                continue

            start = time.perf_counter()
            materialized = backup.to(
                device=target.device,
                dtype=target.dtype,
                copy=False,
            )
            target.copy_(materialized)
            copy_wall_seconds += time.perf_counter() - start
            recovered_bytes += target.numel() * target.element_size()
            recovered_block_ids.append(block_id)

        return RecoveryResult(
            selected_block_ids=[int(block_id) for block_id in selected_block_ids],
            recovered_block_ids=recovered_block_ids,
            missing_backup_block_ids=missing_backup_block_ids,
            skipped_block_ids=skipped_block_ids,
            recovered_bytes=recovered_bytes,
            copy_wall_seconds=copy_wall_seconds,
        )


def _validate_kv_cache(kv_cache: torch.Tensor) -> None:
    """Validate the FlashAttention KV cache layout expected by M3."""
    if kv_cache.ndim != 5 or int(kv_cache.shape[0]) != 2:
        raise ValueError(
            "MPR recovery expects FlashAttention KV cache layout "
            "[2, num_blocks, block_size, num_kv_heads, head_dim], got "
            f"{tuple(kv_cache.shape)}."
        )
