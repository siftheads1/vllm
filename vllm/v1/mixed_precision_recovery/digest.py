# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ArkVale-style key digest helpers for MPR."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KeyBlockDigest:
    digest_min: torch.Tensor
    digest_max: torch.Tensor
    valid_token_count: int
    block_size: int


def summarize_key_block(key_block: torch.Tensor) -> KeyBlockDigest:
    """Summarize one full KV-cache key block.

    Args:
        key_block: FlashAttention key-cache block with shape
            ``[block_size, num_kv_heads, head_dim]``.

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
    block_size = int(key_block.shape[0])
    if block_size <= 0:
        raise ValueError("MPR key block digest requires a non-empty block.")

    raw_max = key_block.amax(dim=0)
    raw_min = key_block.amin(dim=0)
    centers = (raw_max + raw_min) / 2
    dists = (centers.unsqueeze(0) - key_block).abs().mean(dim=0)
    return KeyBlockDigest(
        digest_min=centers - dists,
        digest_max=centers + dists,
        valid_token_count=block_size,
        block_size=block_size,
    )
