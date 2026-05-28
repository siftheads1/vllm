# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ArkVale-style key digest helpers for MPR."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KeyBlockDigest:
    """ArkVale-style summary for one full key-cache block.

    Attributes:
        digest_min: Lower digest bound, shaped ``[num_kv_heads, head_dim]``.
        digest_max: Upper digest bound, shaped ``[num_kv_heads, head_dim]``.
        valid_token_count: Number of token slots summarized. For Step 1.3 this
            is always equal to ``block_size`` because partial-block digests are
            not created.
        block_size: Number of token slots in the source block.
    """

    digest_min: torch.Tensor
    digest_max: torch.Tensor
    valid_token_count: int
    block_size: int


def summarize_key_block(key_block: torch.Tensor) -> KeyBlockDigest:
    """Summarize one full KV-cache key block.

    Args:
        key_block: FlashAttention key-cache block with shape
            ``[block_size, num_kv_heads, head_dim]``. This is expected to be
            ``kv_cache[0, physical_block_id]`` where the leading ``0`` selects
            keys rather than values from FlashAttention's KV cache.

    Returns:
        An ArkVale-style bounding digest with min/max tensors shaped
        ``[num_kv_heads, head_dim]``.
    """
    if key_block.ndim != 3:
        raise ValueError(
            "MPR key block digest expects a 3D tensor shaped "
            f"[block_size, num_kv_heads, head_dim], got {tuple(key_block.shape)}."
        )

    key_block = key_block.detach()

    # block_size: number of token slots in this physical KV block.
    # num_kv_heads/head_dim remain in dimensions 1 and 2.
    block_size = int(key_block.shape[0])
    if block_size <= 0:
        raise ValueError("MPR key block digest requires a non-empty block.")

    # raw_max/raw_min: per-head/per-channel extrema over the block token axis.
    # Shape: [num_kv_heads, head_dim].
    raw_max = key_block.amax(dim=0)
    raw_min = key_block.amin(dim=0)

    # centers: midpoint of the raw bounding box.
    # Shape: [num_kv_heads, head_dim].
    centers = (raw_max + raw_min) / 2

    # dists: mean absolute distance from center over the block token axis.
    # centers.unsqueeze(0) broadcasts to [block_size, num_kv_heads, head_dim].
    # Shape: [num_kv_heads, head_dim].
    dists = (centers.unsqueeze(0) - key_block).abs().mean(dim=0)
    return KeyBlockDigest(
        # digest_min/digest_max: ArkVale-style tightened bounds.
        # Shape: [num_kv_heads, head_dim].
        digest_min=centers - dists,
        digest_max=centers + dists,
        valid_token_count=block_size,
        block_size=block_size,
    )
