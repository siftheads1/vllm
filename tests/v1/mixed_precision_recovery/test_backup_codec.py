# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.mixed_precision_recovery.backup_codec import (
    FP16_BACKUP_FORMAT,
    FP16BackupCodec,
    INT8_BACKUP_FORMAT,
    INT8_QMAX,
    INT8BackupCodec,
    PER_TOKEN_PER_KV_HEAD_SCALE,
)


def _kv_block() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[0.0, 1.0, -2.0], [3.0, -4.0, 5.0]],
                [[-0.25, 0.5, 0.75], [1.25, -1.5, 2.0]],
            ],
            [
                [[-1.0, 2.0, 3.0], [0.0, -0.5, 0.5]],
                [[4.0, -3.0, 2.0], [-2.0, 1.0, -0.25]],
            ],
        ],
        dtype=torch.float32,
    )


def test_fp16_backup_codec_creates_cpu_fp16_payload():
    source = _kv_block()
    payload = FP16BackupCodec().encode(source)

    assert payload.format == FP16_BACKUP_FORMAT
    assert payload.original_shape == tuple(source.shape)
    assert payload.tensor.device.type == "cpu"
    assert payload.tensor.dtype == torch.float16
    assert tuple(payload.tensor.shape) == tuple(source.shape)
    assert payload.payload_nbytes == (
        payload.tensor.numel() * payload.tensor.element_size()
    )
    torch.testing.assert_close(payload.tensor, source.to(torch.float16))


def test_fp16_backup_codec_materializes_to_target_dtype_and_device():
    source = _kv_block()
    codec = FP16BackupCodec()
    payload = codec.encode(source)

    materialized = codec.materialize(
        payload,
        target_dtype=torch.float32,
        target_device="cpu",
    )

    assert materialized.device.type == "cpu"
    assert materialized.dtype == torch.float32
    torch.testing.assert_close(materialized, source.to(torch.float16).float())


def test_int8_backup_codec_creates_expected_payload_metadata():
    source = _kv_block()
    payload = INT8BackupCodec().encode(source)

    assert payload.format == INT8_BACKUP_FORMAT
    assert payload.scale_granularity == PER_TOKEN_PER_KV_HEAD_SCALE
    assert payload.original_shape == tuple(source.shape)
    assert payload.quantized.device.type == "cpu"
    assert payload.scale.device.type == "cpu"
    assert payload.quantized.dtype == torch.int8
    assert payload.scale.dtype == torch.float32
    assert tuple(payload.quantized.shape) == tuple(source.shape)
    assert tuple(payload.scale.shape) == tuple(source.shape[:-1])
    assert payload.payload_nbytes == (
        payload.quantized.numel() * payload.quantized.element_size()
        + payload.scale.numel() * payload.scale.element_size()
    )


def test_int8_backup_codec_uses_per_token_per_kv_head_scale():
    source = _kv_block()
    payload = INT8BackupCodec().encode(source)

    expected_scale = source.abs().amax(dim=-1) / INT8_QMAX
    expected_scale = torch.where(
        source.abs().amax(dim=-1) == 0,
        torch.ones_like(expected_scale),
        expected_scale,
    )
    torch.testing.assert_close(payload.scale, expected_scale)


def test_int8_backup_codec_round_trip_error_is_bounded():
    source = _kv_block()
    codec = INT8BackupCodec()
    payload = codec.encode(source)

    materialized = codec.materialize(
        payload,
        target_dtype=torch.float32,
        target_device="cpu",
    )

    error = (materialized - source).abs()
    bound = payload.scale.unsqueeze(-1) / 2
    assert bool(torch.all(error <= bound + 1e-6))


def test_int8_backup_codec_zero_vector_handling_is_stable():
    source = torch.zeros(2, 2, 3, 4, dtype=torch.float32)
    codec = INT8BackupCodec()
    payload = codec.encode(source)
    materialized = codec.materialize(
        payload,
        target_dtype=torch.float32,
        target_device="cpu",
    )

    assert torch.isfinite(payload.scale).all()
    torch.testing.assert_close(payload.scale, torch.ones_like(payload.scale))
    torch.testing.assert_close(
        payload.quantized,
        torch.zeros_like(payload.quantized),
    )
    torch.testing.assert_close(materialized, source)


def test_backup_codecs_reject_invalid_kv_block_shape():
    invalid = torch.zeros(1, 2, 3, 4)

    with pytest.raises(ValueError, match="semantic K/V block"):
        FP16BackupCodec().encode(invalid)
    with pytest.raises(ValueError, match="semantic K/V block"):
        INT8BackupCodec().encode(invalid)
