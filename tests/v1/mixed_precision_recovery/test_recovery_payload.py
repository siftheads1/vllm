# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.mixed_precision_recovery.backup_codec import (
    FP16_BACKUP_FORMAT,
    INT8_BACKUP_FORMAT,
    INT4_BACKUP_FORMAT,
)
from vllm.v1.mixed_precision_recovery.cpu_backup import (
    CPUBackupKey,
    SemanticCPUBackupStore,
)
from vllm.v1.mixed_precision_recovery.precision_policy import TierAssignment
from vllm.v1.mixed_precision_recovery.recovery_payload import (
    EagerRecoveryPayloadProvider,
)


LAYER_NAME = "model.layers.0.self_attn.attn"


def _kv_block(offset: float = 0.0) -> torch.Tensor:
    return (
        torch.arange(16, dtype=torch.float32).reshape(2, 4, 1, 2)
        + offset
    )


def test_eager_provider_fetches_fp16_int8_and_int4_payload_groups():
    store = SemanticCPUBackupStore()
    store.put(
        layer_name=LAYER_NAME,
        physical_block_id=1,
        kv_block=_kv_block(1.0),
        backup_storage_mode="eager_fp16_int8",
    )
    store.put(
        layer_name=LAYER_NAME,
        physical_block_id=2,
        kv_block=_kv_block(2.0),
        backup_storage_mode="eager_fp16_int8",
    )
    store.put(
        layer_name=LAYER_NAME,
        physical_block_id=3,
        kv_block=_kv_block(3.0),
        backup_storage_mode="eager_fp16_int8_int4",
    )
    assignment = TierAssignment(
        fp16_block_ids=[1],
        int8_block_ids=[2],
        int4_block_ids=[3],
        skipped_block_ids=[4],
    )

    result = EagerRecoveryPayloadProvider().fetch(
        assignment=assignment,
        cpu_backup_store=store,
        layer_name=LAYER_NAME,
    )

    assert result.fp16_block_ids == [1]
    assert result.int8_block_ids == [2]
    assert result.int4_block_ids == [3]
    assert result.skipped_block_ids == [4]
    assert result.missing_fp16_block_ids == []
    assert result.missing_int8_block_ids == []
    assert result.missing_int4_block_ids == []
    assert result.fp16_payloads[0].payload.format == FP16_BACKUP_FORMAT
    assert result.int8_payloads[0].payload.format == INT8_BACKUP_FORMAT
    assert result.int4_payloads[0].payload.format == INT4_BACKUP_FORMAT
    assert result.fp16_payload_bytes > 0
    assert result.int8_payload_bytes > 0
    assert result.int4_payload_bytes > 0


def test_eager_provider_reports_missing_payloads_by_tier():
    store = SemanticCPUBackupStore()
    store.put(
        layer_name=LAYER_NAME,
        physical_block_id=1,
        kv_block=_kv_block(),
        backup_storage_mode="fp16_only",
    )
    assignment = TierAssignment(
        fp16_block_ids=[4],
        int8_block_ids=[1],
        int4_block_ids=[1],
        skipped_block_ids=[],
    )

    result = EagerRecoveryPayloadProvider().fetch(
        assignment=assignment,
        cpu_backup_store=store,
        layer_name=LAYER_NAME,
    )

    assert result.fp16_payloads == []
    assert result.int8_payloads == []
    assert result.int4_payloads == []
    assert result.missing_fp16_block_ids == [4]
    assert result.missing_int8_block_ids == [1]
    assert result.missing_int4_block_ids == [1]
    assert result.skipped_block_ids == []


def test_eager_provider_reports_missing_int4_payload():
    store = SemanticCPUBackupStore()
    store.put(
        layer_name=LAYER_NAME,
        physical_block_id=7,
        kv_block=_kv_block(),
        backup_storage_mode="eager_fp16_int8",
    )
    assignment = TierAssignment(
        fp16_block_ids=[],
        int8_block_ids=[],
        int4_block_ids=[7],
        skipped_block_ids=[],
    )

    result = EagerRecoveryPayloadProvider().fetch(
        assignment=assignment,
        cpu_backup_store=store,
        layer_name=LAYER_NAME,
    )

    assert result.int4_payloads == []
    assert result.missing_int4_block_ids == [7]


def test_eager_provider_does_not_fetch_skip_tier_payloads():
    class SpyStore(SemanticCPUBackupStore):

        def __init__(self) -> None:
            super().__init__()
            self.requested_payload_ids: list[int] = []

        def get_payload(self, key: CPUBackupKey, backup_format: str):
            self.requested_payload_ids.append(key.physical_block_id)
            return super().get_payload(key, backup_format)

    store = SpyStore()
    store.put(
        layer_name=LAYER_NAME,
        physical_block_id=1,
        kv_block=_kv_block(),
        backup_storage_mode="eager_fp16_int8",
    )
    assignment = TierAssignment(
        fp16_block_ids=[1],
        int8_block_ids=[],
        int4_block_ids=[],
        skipped_block_ids=[99],
    )

    result = EagerRecoveryPayloadProvider().fetch(
        assignment=assignment,
        cpu_backup_store=store,
        layer_name=LAYER_NAME,
    )

    assert result.skipped_block_ids == [99]
    assert store.requested_payload_ids == [1]
