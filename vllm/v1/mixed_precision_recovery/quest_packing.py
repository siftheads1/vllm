# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Quest-compatible digest metadata packing helpers for MPR."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class PackedQuestDigestCache:
    """Packed digest cache with the tensor contract expected by Quest estimate.

    Attributes:
        metadata_data: Quest-style NHD metadata tensor shaped
            ``[num_metadata_pages, 2, page_size, num_kv_heads, head_dim]``.
            Plane 0 stores digest maxima and plane 1 stores digest minima.
        metadata_indices: Physical metadata page ids, shaped
            ``[num_metadata_pages]``.
        metadata_indptr: Batch-size-1 page indptr, shaped ``[2]``.
        metadata_last_page_len: Valid metadata entries in the final metadata
            page. Quest estimate internally subtracts one entry from this.
        metadata_last_page_idx: Physical id of the final metadata page.
        entry_block_ids: Mapping from score entry index to vLLM physical block id.
        num_score_entries: Number of real digest entries scored by the kernel.
        num_packed_entries: Number of metadata entries including the guard entry.
        metadata_page_size: Number of digest entries per metadata page.
    """

    metadata_data: torch.Tensor
    metadata_indices: torch.Tensor
    metadata_indptr: torch.Tensor
    metadata_last_page_len: int
    metadata_last_page_idx: int
    entry_block_ids: list[int]
    num_score_entries: int
    num_packed_entries: int
    metadata_page_size: int


@dataclass
class QuestMetadataStore:
    """Append-only Quest-style metadata cache for one layer.

    This is a first-step persistent cache for the single-request append-only
    smoke path. It stores every finalized digest in Quest's NHD metadata layout
    at digest creation time, so scoring can pass a prefix view to the Quest
    estimate kernel without rebuilding the dense metadata tensor on every
    decode step.
    """

    metadata_page_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    metadata_data: torch.Tensor = field(init=False)
    metadata_indices: torch.Tensor = field(init=False)
    metadata_indptr: torch.Tensor = field(init=False)
    entry_block_ids: list[int] = field(default_factory=list)
    block_id_to_entry: dict[int, int] = field(default_factory=dict)
    _cached_prefix: PackedQuestDigestCache | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cached_prefix_num_score_entries: int | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _metadata_indptr_num_pages: int | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _last_zeroed_guard_entry: int | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.metadata_page_size <= 0:
            raise ValueError(
                "Quest metadata store requires a positive metadata_page_size, "
                f"got {self.metadata_page_size}."
            )
        if self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError(
                "Quest metadata store requires positive num_kv_heads and "
                f"head_dim, got {self.num_kv_heads}, {self.head_dim}."
            )
        self.metadata_data = torch.zeros(
            (
                1,
                2,
                self.metadata_page_size,
                self.num_kv_heads,
                self.head_dim,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        self.metadata_indices = torch.arange(
            1,
            dtype=torch.int32,
            device=self.device,
        )
        self.metadata_indptr = torch.tensor(
            [0, 1],
            dtype=torch.int32,
            device=self.device,
        )
        self._metadata_indptr_num_pages = 1

    @property
    def num_entries(self) -> int:
        """Return the number of real digest entries appended so far."""
        return len(self.entry_block_ids)

    def can_view_prefix(self, block_ids: list[int]) -> bool:
        """Return whether ``block_ids`` matches the store's entry prefix."""
        block_count = len(block_ids)
        if block_count > self.num_entries:
            return False
        if block_count == 0:
            return True
        if self.entry_block_ids[0] != block_ids[0]:
            return False
        if self.entry_block_ids[block_count - 1] != block_ids[-1]:
            return False
        return all(
            self.entry_block_ids[idx] == block_id
            for idx, block_id in enumerate(block_ids[1:-1], start=1)
        )

    def append_digest(
        self,
        *,
        block_id: int,
        digest_min: torch.Tensor,
        digest_max: torch.Tensor,
        profile_callback: Callable[[str, float], None] | None = None,
    ) -> None:
        """Append one block digest and keep one zero guard entry after it."""
        shape_check_start = (
            time.perf_counter() if profile_callback is not None else 0.0
        )
        if block_id in self.block_id_to_entry:
            raise ValueError(
                f"Quest metadata store already has block_id={block_id}."
            )
        expected_shape = (self.num_kv_heads, self.head_dim)
        if tuple(digest_min.shape) != expected_shape:
            raise ValueError(
                "Quest metadata store digest_min shape mismatch: "
                f"expected {expected_shape}, got {tuple(digest_min.shape)}."
            )
        if tuple(digest_max.shape) != expected_shape:
            raise ValueError(
                "Quest metadata store digest_max shape mismatch: "
                f"expected {expected_shape}, got {tuple(digest_max.shape)}."
            )
        if profile_callback is not None:
            profile_callback(
                "counter_quest_append_shape_check",
                shape_check_start,
            )

        entry_idx = self.num_entries
        # Keep capacity for the newly appended entry plus Quest's dummy guard
        # entry. The guard exists because the current Quest estimate kernel
        # always excludes the final logical metadata entry from scoring.
        ensure_start = time.perf_counter() if profile_callback is not None else 0.0
        self._ensure_entry_capacity(entry_idx + 2)
        if profile_callback is not None:
            profile_callback(
                "counter_quest_append_ensure_capacity",
                ensure_start,
            )
        page_idx = entry_idx // self.metadata_page_size
        page_offset = entry_idx % self.metadata_page_size
        copy_max_start = (
            time.perf_counter() if profile_callback is not None else 0.0
        )
        self.metadata_data[page_idx, 0, page_offset].copy_(
            digest_max.to(device=self.device, dtype=self.dtype)
        )
        if profile_callback is not None:
            profile_callback("counter_quest_append_copy_max", copy_max_start)
        copy_min_start = (
            time.perf_counter() if profile_callback is not None else 0.0
        )
        self.metadata_data[page_idx, 1, page_offset].copy_(
            digest_min.to(device=self.device, dtype=self.dtype)
        )
        if profile_callback is not None:
            profile_callback("counter_quest_append_copy_min", copy_min_start)
        python_index_start = (
            time.perf_counter() if profile_callback is not None else 0.0
        )
        self.entry_block_ids.append(block_id)
        self.block_id_to_entry[block_id] = entry_idx
        if profile_callback is not None:
            profile_callback(
                "counter_quest_append_python_index",
                python_index_start,
            )
        zero_guard_start = (
            time.perf_counter() if profile_callback is not None else 0.0
        )
        self._zero_entry(entry_idx + 1)
        if profile_callback is not None:
            profile_callback("counter_quest_append_zero_guard", zero_guard_start)
        invalidate_start = (
            time.perf_counter() if profile_callback is not None else 0.0
        )
        self._invalidate_prefix_cache()
        if profile_callback is not None:
            profile_callback(
                "counter_quest_append_invalidate_prefix",
                invalidate_start,
            )

    def view_prefix(self, num_score_entries: int) -> PackedQuestDigestCache:
        """Return a Quest packed-cache view over the first score entries.

        The returned metadata includes one trailing guard entry because Quest's
        estimate kernel drops the final logical metadata entry.
        """
        if num_score_entries <= 0:
            raise ValueError(
                "Quest metadata store prefix view requires at least one score "
                f"entry, got {num_score_entries}."
            )
        if num_score_entries > self.num_entries:
            raise ValueError(
                "Quest metadata store prefix view exceeds stored entries: "
                f"{num_score_entries} > {self.num_entries}."
            )
        if (
            self._cached_prefix is not None
            and self._cached_prefix_num_score_entries == num_score_entries
        ):
            return self._cached_prefix

        num_packed_entries = num_score_entries + 1
        self._ensure_entry_capacity(num_packed_entries)
        if (
            num_score_entries == self.num_entries
            and self._last_zeroed_guard_entry != num_score_entries
        ):
            self._zero_entry(num_score_entries)
        num_metadata_pages = (
            num_packed_entries + self.metadata_page_size - 1
        ) // self.metadata_page_size
        metadata_last_page_len = (
            (num_packed_entries - 1) % self.metadata_page_size
        ) + 1
        self._set_metadata_indptr_num_pages(num_metadata_pages)
        packed = PackedQuestDigestCache(
            metadata_data=self.metadata_data[:num_metadata_pages],
            metadata_indices=self.metadata_indices[:num_metadata_pages],
            metadata_indptr=self.metadata_indptr,
            metadata_last_page_len=metadata_last_page_len,
            metadata_last_page_idx=num_metadata_pages - 1,
            entry_block_ids=list(self.entry_block_ids[:num_score_entries]),
            num_score_entries=num_score_entries,
            num_packed_entries=num_packed_entries,
            metadata_page_size=self.metadata_page_size,
        )
        self._cached_prefix = packed
        self._cached_prefix_num_score_entries = num_score_entries
        return packed

    def _ensure_entry_capacity(self, required_entries: int) -> None:
        required_pages = (
            required_entries + self.metadata_page_size - 1
        ) // self.metadata_page_size
        current_pages = int(self.metadata_data.shape[0])
        if required_pages <= current_pages:
            return

        new_pages = max(required_pages, current_pages * 2)
        new_metadata_data = torch.zeros(
            (
                new_pages,
                2,
                self.metadata_page_size,
                self.num_kv_heads,
                self.head_dim,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        new_metadata_data[:current_pages].copy_(self.metadata_data)
        self.metadata_data = new_metadata_data
        self.metadata_indices = torch.arange(
            new_pages,
            dtype=torch.int32,
            device=self.device,
        )
        self._invalidate_prefix_cache()

    def _zero_entry(self, entry_idx: int) -> None:
        self._ensure_entry_capacity(entry_idx + 1)
        page_idx = entry_idx // self.metadata_page_size
        page_offset = entry_idx % self.metadata_page_size
        self.metadata_data[page_idx, :, page_offset].zero_()
        self._last_zeroed_guard_entry = entry_idx

    def _set_metadata_indptr_num_pages(self, num_metadata_pages: int) -> None:
        if self._metadata_indptr_num_pages == num_metadata_pages:
            return
        self.metadata_indptr[0] = 0
        self.metadata_indptr[1] = num_metadata_pages
        self._metadata_indptr_num_pages = num_metadata_pages

    def _invalidate_prefix_cache(self) -> None:
        self._cached_prefix = None
        self._cached_prefix_num_score_entries = None


def pack_quest_metadata_cache(
    *,
    digest_min: torch.Tensor,
    digest_max: torch.Tensor,
    metadata_page_size: int,
    entry_block_ids: list[int] | None = None,
    add_guard_entry: bool = True,
) -> PackedQuestDigestCache:
    """Pack MPR digest tensors into Quest's paged metadata-cache layout.

    Quest's estimate kernel excludes the final logical metadata entry. The
    default ``add_guard_entry=True`` appends one dummy entry so every real MPR
    candidate digest is scored while preserving the unmodified Quest kernel
    behavior.

    This guard is an explicit compatibility shim for the current Quest kernel
    contract, not a general MPR policy. Once the estimate binding exposes an
    output length or exclusion count directly, this guard should be removed in
    favor of passing the exact score-candidate entry count to the kernel.
    """
    if digest_min.ndim != 3 or digest_max.ndim != 3:
        raise ValueError(
            "Quest metadata packing expects digest tensors shaped "
            "[num_entries, num_kv_heads, head_dim]."
        )
    if digest_min.shape != digest_max.shape:
        raise ValueError(
            "Quest metadata packing requires digest_min and digest_max to have "
            f"the same shape, got {tuple(digest_min.shape)} and "
            f"{tuple(digest_max.shape)}."
        )
    if metadata_page_size <= 0:
        raise ValueError(
            "Quest metadata packing requires a positive metadata_page_size, "
            f"got {metadata_page_size}."
        )

    num_score_entries = int(digest_min.shape[0])
    num_kv_heads = int(digest_min.shape[1])
    head_dim = int(digest_min.shape[2])
    if entry_block_ids is None:
        entry_block_ids = list(range(num_score_entries))
    elif len(entry_block_ids) != num_score_entries:
        raise ValueError(
            "Quest metadata packing requires entry_block_ids length to match "
            f"num_score_entries={num_score_entries}, got {len(entry_block_ids)}."
        )

    # Quest estimate currently treats the last logical metadata entry as the
    # always-recalled current page and drops it from score output. MPR's recent
    # protection is already handled before packing, so we append one dummy guard
    # entry to satisfy that Quest-specific contract without losing a real score
    # candidate. This is intentionally ad hoc and should disappear if/when the
    # binding lets MPR pass an explicit output length or exclusion count.
    num_packed_entries = num_score_entries + (1 if add_guard_entry else 0)
    if num_packed_entries <= 0:
        raise ValueError("Quest metadata packing requires at least one entry.")

    num_metadata_pages = (
        num_packed_entries + metadata_page_size - 1
    ) // metadata_page_size

    entries = torch.zeros(
        (
            num_metadata_pages,
            metadata_page_size,
            2,
            num_kv_heads,
            head_dim,
        ),
        dtype=digest_max.dtype,
        device=digest_max.device,
    )
    if num_score_entries > 0:
        entries_flat = entries.reshape(
            num_metadata_pages * metadata_page_size,
            2,
            num_kv_heads,
            head_dim,
        )
        entries_flat[:num_score_entries, 0] = digest_max
        entries_flat[:num_score_entries, 1] = digest_min.to(dtype=digest_max.dtype)

    metadata_data = entries.permute(0, 2, 1, 3, 4).contiguous()
    metadata_indices = torch.arange(
        num_metadata_pages,
        dtype=torch.int32,
        device=digest_max.device,
    )
    metadata_indptr = torch.tensor(
        [0, num_metadata_pages],
        dtype=torch.int32,
        device=digest_max.device,
    )
    metadata_last_page_len = (
        (num_packed_entries - 1) % metadata_page_size
    ) + 1

    return PackedQuestDigestCache(
        metadata_data=metadata_data,
        metadata_indices=metadata_indices,
        metadata_indptr=metadata_indptr,
        metadata_last_page_len=metadata_last_page_len,
        metadata_last_page_idx=num_metadata_pages - 1,
        entry_block_ids=list(entry_block_ids),
        num_score_entries=num_score_entries,
        num_packed_entries=num_packed_entries,
        metadata_page_size=metadata_page_size,
    )
