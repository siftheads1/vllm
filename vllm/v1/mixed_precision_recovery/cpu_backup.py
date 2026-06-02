# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU KV backup stores for the MPR sidecar."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import torch


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
    copy_wall_seconds: float


@dataclass(frozen=True)
class CPUBackupReleaseResult:
    """Result metadata for one CPU backup release call."""

    released_entries: int
    released_bytes: int


@dataclass(frozen=True)
class CPUBackupStats:
    """Point-in-time CPU backup store stats."""

    block_count: int
    total_bytes: int
    put_count: int
    release_count: int
    total_copy_wall_seconds: float


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
    ) -> CPUBackupPutResult:
        ...

    def get(self, key: CPUBackupKey) -> torch.Tensor | None:
        ...

    def release_blocks(
        self,
        physical_block_ids: set[int],
    ) -> CPUBackupReleaseResult:
        ...

    def stats(self) -> CPUBackupStats:
        ...


class SemanticCPUBackupStore:
    """Semantic fp16 CPU tensor backup store for the M2 prototype."""

    def __init__(self) -> None:
        self._entries: dict[CPUBackupKey, torch.Tensor] = {}
        self._total_bytes = 0
        self._put_count = 0
        self._release_count = 0
        self._total_copy_wall_seconds = 0.0

    def put(
        self,
        *,
        layer_name: str,
        physical_block_id: int,
        kv_block: torch.Tensor,
        request_id: str | None = None,
        logical_block_idx: int | None = None,
        generation: int | None = None,
    ) -> CPUBackupPutResult:
        """Synchronously copy one semantic K/V block to CPU fp16."""
        if kv_block.ndim < 2 or int(kv_block.shape[0]) != 2:
            raise ValueError(
                "MPR CPU backup expects a semantic K/V block whose first "
                f"dimension is 2, got shape={tuple(kv_block.shape)}."
            )
        key = CPUBackupKey(
            layer_name=layer_name,
            physical_block_id=int(physical_block_id),
            request_id=request_id,
            logical_block_idx=logical_block_idx,
            generation=generation,
        )
        old = self._entries.get(key)
        if old is not None:
            self._total_bytes -= old.numel() * old.element_size()

        start = time.perf_counter()
        copied = kv_block.detach().to(
            device="cpu",
            dtype=torch.float16,
            copy=True,
        )
        elapsed = time.perf_counter() - start
        num_bytes = copied.numel() * copied.element_size()
        self._entries[key] = copied
        self._total_bytes += num_bytes
        self._put_count += 1
        self._total_copy_wall_seconds += elapsed
        return CPUBackupPutResult(
            key=key,
            shape=tuple(copied.shape),
            dtype=copied.dtype,
            num_bytes=num_bytes,
            copy_wall_seconds=elapsed,
        )

    def get(self, key: CPUBackupKey) -> torch.Tensor | None:
        """Return a stored CPU block by key, if present."""
        return self._entries.get(key)

    def release_blocks(
        self,
        physical_block_ids: set[int],
    ) -> CPUBackupReleaseResult:
        """Release every backup entry whose external GPU block id matches."""
        if not physical_block_ids:
            return CPUBackupReleaseResult(released_entries=0, released_bytes=0)
        released_entries = 0
        released_bytes = 0
        for key in list(self._entries):
            if key.physical_block_id not in physical_block_ids:
                continue
            tensor = self._entries.pop(key)
            num_bytes = tensor.numel() * tensor.element_size()
            self._total_bytes -= num_bytes
            released_entries += 1
            released_bytes += num_bytes
            # This prototype store releases ownership by dropping Python
            # references and letting the PyTorch CPU allocator reuse/free the
            # storage. If allocation/free overhead or host-memory reuse becomes
            # important, replace this backend with an explicit CPU block pool
            # and free list rather than relying on allocator behavior.
            del tensor
        if released_entries:
            self._release_count += released_entries
        return CPUBackupReleaseResult(
            released_entries=released_entries,
            released_bytes=released_bytes,
        )

    def stats(self) -> CPUBackupStats:
        """Return current CPU backup store stats."""
        return CPUBackupStats(
            block_count=len(self._entries),
            total_bytes=self._total_bytes,
            put_count=self._put_count,
            release_count=self._release_count,
            total_copy_wall_seconds=self._total_copy_wall_seconds,
        )
