# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU KV backup stores for the MPR sidecar."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import torch

from vllm.v1.mixed_precision_recovery.backup_codec import (
    FP16_BACKUP_FORMAT,
    FP16BackupCodec,
    FP16BackupPayload,
    INT8_BACKUP_FORMAT,
    INT8BackupCodec,
    INT8BackupPayload,
    INT4_BACKUP_FORMAT,
    INT4BackupCodec,
    INT4BackupPayload,
)


@dataclass(frozen=True)
class CPUBackupKey:
    """Internal key wrapper for one MPR CPU backup entry.

    Milestone 2 only fills ``layer_name`` and ``physical_block_id``. The
    optional fields keep tuple-shaped keys from leaking through the prototype
    while leaving room for request/logical/generation keying later.
    """

    layer_name: str
    physical_block_id: int
    request_id: str | None = None
    logical_block_idx: int | None = None
    generation: int | None = None


@dataclass(frozen=True)
class CPUBackupPutResult:
    """Result metadata for one CPU backup put."""

    key: CPUBackupKey
    shape: tuple[int, ...]
    dtype: torch.dtype
    num_bytes: int
    fp16_payload_bytes: int
    int8_payload_bytes: int
    int8_scale_bytes: int
    int4_payload_bytes: int
    int4_scale_bytes: int
    total_actual_backup_bytes: int
    copy_wall_seconds: float


@dataclass(frozen=True)
class CPUBackupReleaseResult:
    """Result metadata for one CPU backup release call."""

    released_entries: int
    released_bytes: int
    fp16_payload_bytes: int
    int8_payload_bytes: int
    int8_scale_bytes: int
    int4_payload_bytes: int
    int4_scale_bytes: int
    total_actual_backup_bytes: int


@dataclass(frozen=True)
class CPUBackupStats:
    """Point-in-time CPU backup store stats."""

    block_count: int
    total_bytes: int
    put_count: int
    release_count: int
    total_copy_wall_seconds: float
    fp16_payload_bytes: int
    int8_payload_bytes: int
    int8_scale_bytes: int
    int4_payload_bytes: int
    int4_scale_bytes: int
    total_actual_backup_bytes: int


@dataclass(frozen=True)
class _CPUBackupEntry:
    """Payload-aware backup entry for one semantic K/V block."""

    fp16_payload: FP16BackupPayload
    int8_payload: INT8BackupPayload | None = None
    int4_payload: INT4BackupPayload | None = None

    @property
    def fp16_payload_bytes(self) -> int:
        return self.fp16_payload.payload_nbytes

    @property
    def int8_payload_bytes(self) -> int:
        if self.int8_payload is None:
            return 0
        return _tensor_nbytes(self.int8_payload.quantized)

    @property
    def int8_scale_bytes(self) -> int:
        if self.int8_payload is None:
            return 0
        return _tensor_nbytes(self.int8_payload.scale)

    @property
    def int4_payload_bytes(self) -> int:
        if self.int4_payload is None:
            return 0
        return _tensor_nbytes(self.int4_payload.packed)

    @property
    def int4_scale_bytes(self) -> int:
        if self.int4_payload is None:
            return 0
        return _tensor_nbytes(self.int4_payload.scale)

    @property
    def total_actual_backup_bytes(self) -> int:
        return (
            self.fp16_payload_bytes
            + self.int8_payload_bytes
            + self.int8_scale_bytes
            + self.int4_payload_bytes
            + self.int4_scale_bytes
        )


class CPUBackupStore(Protocol):
    """Small internal interface for MPR CPU backup implementations."""

    def put(
        self,
        *,
        layer_name: str,
        physical_block_id: int,
        kv_block: torch.Tensor,
        request_id: str | None = None,
        logical_block_idx: int | None = None,
        generation: int | None = None,
        backup_storage_mode: str = "fp16_only",
    ) -> CPUBackupPutResult:
        ...

    def get(self, key: CPUBackupKey) -> torch.Tensor | None:
        ...

    def get_payload(
        self,
        key: CPUBackupKey,
        backup_format: str,
    ) -> FP16BackupPayload | INT8BackupPayload | INT4BackupPayload | None:
        ...

    def release_blocks(
        self,
        physical_block_ids: set[int],
    ) -> CPUBackupReleaseResult:
        ...

    def stats(self) -> CPUBackupStats:
        ...


class SemanticCPUBackupStore:
    """Semantic CPU tensor backup store for MPR recovery prototypes."""

    def __init__(self) -> None:
        self._entries: dict[CPUBackupKey, _CPUBackupEntry] = {}
        self._fp16_payload_bytes = 0
        self._int8_payload_bytes = 0
        self._int8_scale_bytes = 0
        self._int4_payload_bytes = 0
        self._int4_scale_bytes = 0
        self._put_count = 0
        self._release_count = 0
        self._total_copy_wall_seconds = 0.0
        self._fp16_codec = FP16BackupCodec()
        self._int8_codec = INT8BackupCodec()
        self._int4_codec = INT4BackupCodec()

    def put(
        self,
        *,
        layer_name: str,
        physical_block_id: int,
        kv_block: torch.Tensor,
        request_id: str | None = None,
        logical_block_idx: int | None = None,
        generation: int | None = None,
        backup_storage_mode: str = "fp16_only",
    ) -> CPUBackupPutResult:
        """Synchronously copy one semantic K/V block to CPU backup payloads."""
        _validate_backup_storage_mode(backup_storage_mode)
        key = CPUBackupKey(
            layer_name=layer_name,
            physical_block_id=int(physical_block_id),
            request_id=request_id,
            logical_block_idx=logical_block_idx,
            generation=generation,
        )

        start = time.perf_counter()
        fp16_payload = self._fp16_codec.encode(kv_block)
        int8_payload = (
            self._int8_codec.encode(fp16_payload.tensor)
            if backup_storage_mode in {
                "eager_fp16_int8",
                "eager_fp16_int8_int4",
            }
            else None
        )
        int4_payload = (
            self._int4_codec.encode(fp16_payload.tensor)
            if backup_storage_mode == "eager_fp16_int8_int4"
            else None
        )
        entry = _CPUBackupEntry(
            fp16_payload=fp16_payload,
            int8_payload=int8_payload,
            int4_payload=int4_payload,
        )
        elapsed = time.perf_counter() - start

        old = self._entries.get(key)
        if old is not None:
            self._subtract_entry_bytes(old)

        self._entries[key] = entry
        self._add_entry_bytes(entry)
        self._put_count += 1
        self._total_copy_wall_seconds += elapsed
        return CPUBackupPutResult(
            key=key,
            shape=tuple(fp16_payload.tensor.shape),
            dtype=fp16_payload.tensor.dtype,
            num_bytes=entry.fp16_payload_bytes,
            fp16_payload_bytes=entry.fp16_payload_bytes,
            int8_payload_bytes=entry.int8_payload_bytes,
            int8_scale_bytes=entry.int8_scale_bytes,
            int4_payload_bytes=entry.int4_payload_bytes,
            int4_scale_bytes=entry.int4_scale_bytes,
            total_actual_backup_bytes=entry.total_actual_backup_bytes,
            copy_wall_seconds=elapsed,
        )

    def get(self, key: CPUBackupKey) -> torch.Tensor | None:
        """Return the stored fp16 CPU tensor by key, if present."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        return entry.fp16_payload.tensor

    def get_payload(
        self,
        key: CPUBackupKey,
        backup_format: str,
    ) -> FP16BackupPayload | INT8BackupPayload | INT4BackupPayload | None:
        """Return a precision-specific logical payload by key, if present."""
        if backup_format not in {
            FP16_BACKUP_FORMAT,
            INT8_BACKUP_FORMAT,
            INT4_BACKUP_FORMAT,
        }:
            raise ValueError(
                "backup_format must be 'fp16', 'int8', or 'int4', got "
                f"{backup_format!r}."
            )
        entry = self._entries.get(key)
        if entry is None:
            return None
        if backup_format == FP16_BACKUP_FORMAT:
            return entry.fp16_payload
        if backup_format == INT8_BACKUP_FORMAT:
            return entry.int8_payload
        return entry.int4_payload

    def release_blocks(
        self,
        physical_block_ids: set[int],
    ) -> CPUBackupReleaseResult:
        """Release every backup entry whose external GPU block id matches."""
        if not physical_block_ids:
            return CPUBackupReleaseResult(
                released_entries=0,
                released_bytes=0,
                fp16_payload_bytes=0,
                int8_payload_bytes=0,
                int8_scale_bytes=0,
                int4_payload_bytes=0,
                int4_scale_bytes=0,
                total_actual_backup_bytes=0,
            )
        released_entries = 0
        released_fp16_payload_bytes = 0
        released_int8_payload_bytes = 0
        released_int8_scale_bytes = 0
        released_int4_payload_bytes = 0
        released_int4_scale_bytes = 0
        for key in list(self._entries):
            if key.physical_block_id not in physical_block_ids:
                continue
            entry = self._entries.pop(key)
            self._subtract_entry_bytes(entry)
            released_entries += 1
            released_fp16_payload_bytes += entry.fp16_payload_bytes
            released_int8_payload_bytes += entry.int8_payload_bytes
            released_int8_scale_bytes += entry.int8_scale_bytes
            released_int4_payload_bytes += entry.int4_payload_bytes
            released_int4_scale_bytes += entry.int4_scale_bytes
            # This prototype store releases ownership by dropping Python
            # references and letting the PyTorch CPU allocator reuse/free the
            # storage. If allocation/free overhead or host-memory reuse becomes
            # important, replace this backend with an explicit CPU block pool
            # and free list rather than relying on allocator behavior.
            del entry
        if released_entries:
            self._release_count += released_entries
        released_actual_bytes = (
            released_fp16_payload_bytes
            + released_int8_payload_bytes
            + released_int8_scale_bytes
            + released_int4_payload_bytes
            + released_int4_scale_bytes
        )
        return CPUBackupReleaseResult(
            released_entries=released_entries,
            released_bytes=released_actual_bytes,
            fp16_payload_bytes=released_fp16_payload_bytes,
            int8_payload_bytes=released_int8_payload_bytes,
            int8_scale_bytes=released_int8_scale_bytes,
            int4_payload_bytes=released_int4_payload_bytes,
            int4_scale_bytes=released_int4_scale_bytes,
            total_actual_backup_bytes=released_actual_bytes,
        )

    def stats(self) -> CPUBackupStats:
        """Return current CPU backup store stats."""
        total_actual_bytes = self._total_actual_backup_bytes()
        return CPUBackupStats(
            block_count=len(self._entries),
            total_bytes=total_actual_bytes,
            put_count=self._put_count,
            release_count=self._release_count,
            total_copy_wall_seconds=self._total_copy_wall_seconds,
            fp16_payload_bytes=self._fp16_payload_bytes,
            int8_payload_bytes=self._int8_payload_bytes,
            int8_scale_bytes=self._int8_scale_bytes,
            int4_payload_bytes=self._int4_payload_bytes,
            int4_scale_bytes=self._int4_scale_bytes,
            total_actual_backup_bytes=total_actual_bytes,
        )

    def _add_entry_bytes(self, entry: _CPUBackupEntry) -> None:
        self._fp16_payload_bytes += entry.fp16_payload_bytes
        self._int8_payload_bytes += entry.int8_payload_bytes
        self._int8_scale_bytes += entry.int8_scale_bytes
        self._int4_payload_bytes += entry.int4_payload_bytes
        self._int4_scale_bytes += entry.int4_scale_bytes

    def _subtract_entry_bytes(self, entry: _CPUBackupEntry) -> None:
        self._fp16_payload_bytes -= entry.fp16_payload_bytes
        self._int8_payload_bytes -= entry.int8_payload_bytes
        self._int8_scale_bytes -= entry.int8_scale_bytes
        self._int4_payload_bytes -= entry.int4_payload_bytes
        self._int4_scale_bytes -= entry.int4_scale_bytes

    def _total_actual_backup_bytes(self) -> int:
        return (
            self._fp16_payload_bytes
            + self._int8_payload_bytes
            + self._int8_scale_bytes
            + self._int4_payload_bytes
            + self._int4_scale_bytes
        )


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _validate_backup_storage_mode(backup_storage_mode: str) -> None:
    if backup_storage_mode not in {
        "fp16_only",
        "eager_fp16_int8",
        "eager_fp16_int8_int4",
    }:
        raise ValueError(
            "backup_storage_mode must be 'fp16_only', 'eager_fp16_int8', "
            "or 'eager_fp16_int8_int4', "
            f"got {backup_storage_mode!r}."
        )
