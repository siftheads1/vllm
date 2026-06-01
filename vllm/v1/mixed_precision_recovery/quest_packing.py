# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Quest-compatible digest metadata packing helpers for MPR."""

from __future__ import annotations

from dataclasses import dataclass

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
