# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Precision-specific KV backup payload codecs for MPR.

Backup codecs define logical payload formats and reference encode/materialize
behavior. They do not define how payloads are stored in CPU memory; CPU backup
layout and lifecycle are handled by the backup store in later M4 steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


FP16_BACKUP_FORMAT = "fp16"
INT8_BACKUP_FORMAT = "int8"
PER_TOKEN_PER_KV_HEAD_SCALE = "per_token_per_kv_head"
INT8_QMAX = 127.0


@dataclass(frozen=True)
class FP16BackupPayload:
    """Logical fp16 backup payload for one semantic K/V block."""

    tensor: torch.Tensor
    original_shape: tuple[int, ...]
    format: str = FP16_BACKUP_FORMAT

    @property
    def payload_nbytes(self) -> int:
        """Return payload tensor storage bytes."""
        return self.tensor.numel() * self.tensor.element_size()


@dataclass(frozen=True)
class INT8BackupPayload:
    """Logical int8 backup payload for one semantic K/V block.

    ``quantized`` stores signed symmetric int8 values. ``scale`` stores one
    scale for each K/V, token, and KV head vector. The actual CPU store may
    keep these tensors in any layout as long as this logical contract is
    preserved for materialization.
    """

    quantized: torch.Tensor
    scale: torch.Tensor
    original_shape: tuple[int, ...]
    scale_granularity: str = PER_TOKEN_PER_KV_HEAD_SCALE
    format: str = INT8_BACKUP_FORMAT

    @property
    def payload_nbytes(self) -> int:
        """Return quantized data plus scale tensor storage bytes."""
        return (
            self.quantized.numel() * self.quantized.element_size()
            + self.scale.numel() * self.scale.element_size()
        )


class BackupCodec(Protocol):
    """Encode and materialize one precision-specific KV backup payload."""

    def encode(self, kv_block: torch.Tensor):
        """Encode a semantic K/V block into a logical backup payload."""
        ...

    def materialize(
        self,
        payload,
        *,
        target_dtype: torch.dtype,
        target_device: torch.device | str,
    ) -> torch.Tensor:
        """Materialize a payload into the target dtype/device."""
        ...


class FP16BackupCodec:
    """Reference codec for semantic fp16 K/V backup payloads."""

    def encode(self, kv_block: torch.Tensor) -> FP16BackupPayload:
        """Copy one semantic K/V block to a CPU fp16 payload."""
        _validate_kv_block(kv_block)
        copied = kv_block.detach().to(
            device="cpu",
            dtype=torch.float16,
            copy=True,
        )
        return FP16BackupPayload(
            tensor=copied,
            original_shape=tuple(kv_block.shape),
        )

    def materialize(
        self,
        payload: FP16BackupPayload,
        *,
        target_dtype: torch.dtype,
        target_device: torch.device | str,
    ) -> torch.Tensor:
        """Materialize an fp16 payload into the target dtype/device."""
        return payload.tensor.to(
            device=target_device,
            dtype=target_dtype,
            copy=False,
        )


class INT8BackupCodec:
    """Reference codec for per-token-per-kv-head int8 K/V payloads.

    All-zero vectors use scale ``1.0`` as a placeholder to avoid division by
    zero. Their quantized values remain zero, so dequantization still exactly
    reconstructs the original zero vector.
    """

    def encode(self, kv_block: torch.Tensor) -> INT8BackupPayload:
        """Quantize one semantic K/V block into a CPU int8 payload."""
        _validate_kv_block(kv_block)
        source = kv_block.detach().to(
            device="cpu",
            dtype=torch.float32,
            copy=True,
        )
        max_abs = source.abs().amax(dim=-1)
        scale = max_abs / INT8_QMAX
        scale = torch.where(
            max_abs == 0,
            torch.ones_like(scale),
            scale,
        )
        quantized = torch.round(source / scale.unsqueeze(-1)).clamp(
            -INT8_QMAX,
            INT8_QMAX,
        ).to(torch.int8)
        return INT8BackupPayload(
            quantized=quantized,
            scale=scale,
            original_shape=tuple(kv_block.shape),
        )

    def materialize(
        self,
        payload: INT8BackupPayload,
        *,
        target_dtype: torch.dtype,
        target_device: torch.device | str,
    ) -> torch.Tensor:
        """Dequantize an int8 payload into the target dtype/device."""
        quantized = payload.quantized.to(device=target_device, copy=False)
        scale = payload.scale.to(device=target_device, copy=False)
        materialized = quantized.to(torch.float32) * scale.unsqueeze(-1)
        return materialized.to(dtype=target_dtype)


def _validate_kv_block(kv_block: torch.Tensor) -> None:
    if kv_block.ndim != 4 or int(kv_block.shape[0]) != 2:
        raise ValueError(
            "MPR backup codec expects a semantic K/V block shaped "
            "[2, block_size, num_kv_heads, head_dim], got "
            f"{tuple(kv_block.shape)}."
        )
