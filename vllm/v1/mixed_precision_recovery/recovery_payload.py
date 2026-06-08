# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Recovery payload provider boundaries for MPR precision tiering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vllm.v1.mixed_precision_recovery.backup_codec import (
    FP16_BACKUP_FORMAT,
    FP16BackupPayload,
    INT8_BACKUP_FORMAT,
    INT8BackupPayload,
)
from vllm.v1.mixed_precision_recovery.cpu_backup import (
    CPUBackupKey,
    CPUBackupStore,
)
from vllm.v1.mixed_precision_recovery.precision_policy import TierAssignment


@dataclass(frozen=True)
class FP16RecoveryPayloadEntry:
    """One fp16 recovery payload bound to a physical KV block id."""

    physical_block_id: int
    payload: FP16BackupPayload


@dataclass(frozen=True)
class INT8RecoveryPayloadEntry:
    """One int8 recovery payload bound to a physical KV block id."""

    physical_block_id: int
    payload: INT8BackupPayload


@dataclass(frozen=True)
class TieredRecoveryPayloads:
    """Payload fetch result for one precision-tier assignment."""

    fp16_payloads: list[FP16RecoveryPayloadEntry]
    int8_payloads: list[INT8RecoveryPayloadEntry]
    skipped_block_ids: list[int]
    missing_fp16_block_ids: list[int]
    missing_int8_block_ids: list[int]

    @property
    def fp16_block_ids(self) -> list[int]:
        """Return block ids with available fp16 payloads."""
        return [entry.physical_block_id for entry in self.fp16_payloads]

    @property
    def int8_block_ids(self) -> list[int]:
        """Return block ids with available int8 payloads."""
        return [entry.physical_block_id for entry in self.int8_payloads]

    @property
    def fp16_payload_bytes(self) -> int:
        """Return total logical fp16 payload bytes fetched."""
        return sum(entry.payload.payload_nbytes for entry in self.fp16_payloads)

    @property
    def int8_payload_bytes(self) -> int:
        """Return total logical int8 quantized-plus-scale bytes fetched."""
        return sum(entry.payload.payload_nbytes for entry in self.int8_payloads)


class RecoveryPayloadProvider(Protocol):
    """Fetch precision-specific recovery payloads for assigned tiers."""

    def fetch(
        self,
        *,
        assignment: TierAssignment,
        cpu_backup_store: CPUBackupStore,
        layer_name: str,
    ) -> TieredRecoveryPayloads:
        ...


class EagerRecoveryPayloadProvider:
    """Fetch already-materialized fp16/int8 payloads from CPU backup storage."""

    def fetch(
        self,
        *,
        assignment: TierAssignment,
        cpu_backup_store: CPUBackupStore,
        layer_name: str,
    ) -> TieredRecoveryPayloads:
        """Return available payloads and tier-specific missing ids."""
        if assignment.int4_block_ids:
            raise ValueError(
                "MPR INT4 tier assignment requires INT4 backup payload "
                "support from a later M4.5 step, got int4_block_ids="
                f"{assignment.int4_block_ids}."
            )

        fp16_payloads: list[FP16RecoveryPayloadEntry] = []
        int8_payloads: list[INT8RecoveryPayloadEntry] = []
        missing_fp16_block_ids: list[int] = []
        missing_int8_block_ids: list[int] = []

        for block_id in assignment.fp16_block_ids:
            block_id = int(block_id)
            key = CPUBackupKey(
                layer_name=layer_name,
                physical_block_id=block_id,
            )
            payload = cpu_backup_store.get_payload(key, FP16_BACKUP_FORMAT)
            if payload is None:
                missing_fp16_block_ids.append(block_id)
                continue
            if not isinstance(payload, FP16BackupPayload):
                raise TypeError(
                    "MPR expected an fp16 backup payload for block "
                    f"{block_id}, got {type(payload).__name__}."
                )
            fp16_payloads.append(
                FP16RecoveryPayloadEntry(
                    physical_block_id=block_id,
                    payload=payload,
                )
            )

        for block_id in assignment.int8_block_ids:
            block_id = int(block_id)
            key = CPUBackupKey(
                layer_name=layer_name,
                physical_block_id=block_id,
            )
            payload = cpu_backup_store.get_payload(key, INT8_BACKUP_FORMAT)
            if payload is None:
                missing_int8_block_ids.append(block_id)
                continue
            if not isinstance(payload, INT8BackupPayload):
                raise TypeError(
                    "MPR expected an int8 backup payload for block "
                    f"{block_id}, got {type(payload).__name__}."
                )
            int8_payloads.append(
                INT8RecoveryPayloadEntry(
                    physical_block_id=block_id,
                    payload=payload,
                )
            )

        return TieredRecoveryPayloads(
            fp16_payloads=fp16_payloads,
            int8_payloads=int8_payloads,
            skipped_block_ids=[
                int(block_id) for block_id in assignment.skipped_block_ids
            ],
            missing_fp16_block_ids=missing_fp16_block_ids,
            missing_int8_block_ids=missing_int8_block_ids,
        )
