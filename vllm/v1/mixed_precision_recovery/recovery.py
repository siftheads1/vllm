# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-backup based KV recovery helpers for MPR."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import torch

from vllm.v1.mixed_precision_recovery.backup_codec import (
    FP16BackupCodec,
    INT8BackupCodec,
)
from vllm.v1.mixed_precision_recovery.cpu_backup import (
    CPUBackupKey,
    CPUBackupStore,
)
from vllm.v1.mixed_precision_recovery.recovery_payload import (
    FP16RecoveryPayloadEntry,
    INT8RecoveryPayloadEntry,
    TieredRecoveryPayloads,
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
    recovered_fp16_block_ids: list[int] = field(default_factory=list)
    recovered_int8_block_ids: list[int] = field(default_factory=list)
    recovered_int4_block_ids: list[int] = field(default_factory=list)
    missing_fp16_block_ids: list[int] = field(default_factory=list)
    missing_int8_block_ids: list[int] = field(default_factory=list)
    missing_int4_block_ids: list[int] = field(default_factory=list)
    tier_skipped_block_ids: list[int] = field(default_factory=list)
    fp16_payload_bytes: int = 0
    int8_payload_bytes: int = 0
    int4_payload_bytes: int = 0
    effective_recovery_transfer_bytes: int = 0


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

    def __init__(self) -> None:
        self._fp16_codec = FP16BackupCodec()
        self._int8_codec = INT8BackupCodec()

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

    def materialize_tiered_payloads(
        self,
        *,
        tiered_payloads: TieredRecoveryPayloads,
        kv_cache: torch.Tensor,
    ) -> RecoveryResult:
        """Copy tiered recovery payloads into ``kv_cache`` in-place.

        This is the Milestone 4 primitive for precision-aware materialization.
        It is intentionally not wired into the sidecar runtime path until the
        next integration step.
        """
        _validate_kv_cache(kv_cache)

        recovered_fp16_block_ids: list[int] = []
        recovered_int8_block_ids: list[int] = []
        if tiered_payloads.int4_payloads:
            raise ValueError(
                "MPR tiered int4 recovery materialization requires "
                "Step 4.5.4 support."
            )
        skipped_block_ids = [
            int(block_id) for block_id in tiered_payloads.skipped_block_ids
        ]
        recovered_bytes = 0
        fp16_payload_bytes = 0
        int8_payload_bytes = 0
        effective_recovery_transfer_bytes = 0
        copy_wall_seconds = 0.0
        num_blocks = int(kv_cache.shape[1])

        for entry in tiered_payloads.fp16_payloads:
            target_bytes, payload_bytes, elapsed = self._materialize_fp16_entry(
                entry=entry,
                kv_cache=kv_cache,
                num_blocks=num_blocks,
            )
            recovered_fp16_block_ids.append(int(entry.physical_block_id))
            recovered_bytes += target_bytes
            fp16_payload_bytes += payload_bytes
            effective_recovery_transfer_bytes += payload_bytes
            copy_wall_seconds += elapsed

        for entry in tiered_payloads.int8_payloads:
            target_bytes, payload_bytes, elapsed = self._materialize_int8_entry(
                entry=entry,
                kv_cache=kv_cache,
                num_blocks=num_blocks,
            )
            recovered_int8_block_ids.append(int(entry.physical_block_id))
            recovered_bytes += target_bytes
            int8_payload_bytes += payload_bytes
            effective_recovery_transfer_bytes += payload_bytes
            copy_wall_seconds += elapsed

        missing_fp16_block_ids = [
            int(block_id) for block_id in tiered_payloads.missing_fp16_block_ids
        ]
        missing_int8_block_ids = [
            int(block_id) for block_id in tiered_payloads.missing_int8_block_ids
        ]
        missing_int4_block_ids = [
            int(block_id) for block_id in tiered_payloads.missing_int4_block_ids
        ]
        # Missing payload ids are provider/store availability results, not
        # target materialization failures. For the current eager provider,
        # missing int8 usually means the store was not populated with int8
        # payloads (for example fp16_only storage); Step 4.7 should decide
        # whether that integration-time combination is a hard config error.
        recovered_block_ids = [
            *recovered_fp16_block_ids,
            *recovered_int8_block_ids,
        ]
        missing_backup_block_ids = [
            *missing_fp16_block_ids,
            *missing_int8_block_ids,
            *missing_int4_block_ids,
        ]
        selected_block_ids = [
            *tiered_payloads.fp16_block_ids,
            *tiered_payloads.int8_block_ids,
            *tiered_payloads.int4_block_ids,
            *missing_fp16_block_ids,
            *missing_int8_block_ids,
            *missing_int4_block_ids,
            *tiered_payloads.skipped_block_ids,
        ]

        return RecoveryResult(
            selected_block_ids=selected_block_ids,
            recovered_block_ids=recovered_block_ids,
            missing_backup_block_ids=missing_backup_block_ids,
            skipped_block_ids=skipped_block_ids,
            recovered_bytes=recovered_bytes,
            copy_wall_seconds=copy_wall_seconds,
            recovered_fp16_block_ids=recovered_fp16_block_ids,
            recovered_int8_block_ids=recovered_int8_block_ids,
            missing_fp16_block_ids=missing_fp16_block_ids,
            missing_int8_block_ids=missing_int8_block_ids,
            missing_int4_block_ids=missing_int4_block_ids,
            tier_skipped_block_ids=[
                int(block_id) for block_id in tiered_payloads.skipped_block_ids
            ],
            fp16_payload_bytes=fp16_payload_bytes,
            int8_payload_bytes=int8_payload_bytes,
            effective_recovery_transfer_bytes=(
                effective_recovery_transfer_bytes
            ),
        )

    def _materialize_fp16_entry(
        self,
        *,
        entry: FP16RecoveryPayloadEntry,
        kv_cache: torch.Tensor,
        num_blocks: int,
    ) -> tuple[int, int, float]:
        block_id = int(entry.physical_block_id)
        if block_id < 0 or block_id >= num_blocks:
            raise ValueError(
                "MPR tiered fp16 recovery block id is outside kv_cache "
                f"block range: block_id={block_id}, num_blocks={num_blocks}."
            )
        target = kv_cache[:, block_id]
        if tuple(entry.payload.original_shape) != tuple(target.shape):
            raise ValueError(
                "MPR tiered fp16 recovery payload shape does not match "
                f"target block shape for block_id={block_id}: "
                f"payload_shape={entry.payload.original_shape}, "
                f"target_shape={tuple(target.shape)}."
            )

        start = time.perf_counter()
        materialized = self._fp16_codec.materialize(
            entry.payload,
            target_dtype=target.dtype,
            target_device=target.device,
        )
        target.copy_(materialized)
        elapsed = time.perf_counter() - start
        return (
            target.numel() * target.element_size(),
            entry.payload.payload_nbytes,
            elapsed,
        )

    def _materialize_int8_entry(
        self,
        *,
        entry: INT8RecoveryPayloadEntry,
        kv_cache: torch.Tensor,
        num_blocks: int,
    ) -> tuple[int, int, float]:
        block_id = int(entry.physical_block_id)
        if block_id < 0 or block_id >= num_blocks:
            raise ValueError(
                "MPR tiered int8 recovery block id is outside kv_cache "
                f"block range: block_id={block_id}, num_blocks={num_blocks}."
            )
        target = kv_cache[:, block_id]
        if tuple(entry.payload.original_shape) != tuple(target.shape):
            raise ValueError(
                "MPR tiered int8 recovery payload shape does not match "
                f"target block shape for block_id={block_id}: "
                f"payload_shape={entry.payload.original_shape}, "
                f"target_shape={tuple(target.shape)}."
            )

        start = time.perf_counter()
        materialized = self._int8_codec.materialize(
            entry.payload,
            target_dtype=target.dtype,
            target_device=target.device,
        )
        target.copy_(materialized)
        elapsed = time.perf_counter() - start
        return (
            target.numel() * target.element_size(),
            entry.payload.payload_nbytes,
            elapsed,
        )


def _validate_kv_cache(kv_cache: torch.Tensor) -> None:
    """Validate the FlashAttention KV cache layout expected by M3."""
    if kv_cache.ndim != 5 or int(kv_cache.shape[0]) != 2:
        raise ValueError(
            "MPR recovery expects FlashAttention KV cache layout "
            "[2, num_blocks, block_size, num_kv_heads, head_dim], got "
            f"{tuple(kv_cache.shape)}."
        )
