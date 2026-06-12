# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.mixed_precision_recovery.backup_codec import (
    FP16_BACKUP_FORMAT,
    INT8_BACKUP_FORMAT,
    INT8BackupCodec,
    INT4_BACKUP_FORMAT,
    INT4BackupCodec,
)
from vllm.v1.mixed_precision_recovery.config import MPRConfig
from vllm.v1.mixed_precision_recovery.cpu_backup import (
    CPUBackupKey,
    SemanticCPUBackupStore,
)
from vllm.v1.mixed_precision_recovery.quest_packing import (
    QuestMetadataStore,
    pack_quest_metadata_cache,
)
from vllm.v1.mixed_precision_recovery.scoring import (
    QUEST_NHD_LAYOUT,
    QuestCudaScorer,
    TorchQuestScorer,
    aggregate_packed_query_head_scores,
    aggregate_query_head_scores,
    estimate_digest_score_result,
    estimate_digest_scores,
    estimate_query_head_digest_scores,
    get_digest_scoring_backend,
)
from vllm.v1.mixed_precision_recovery.sidecar import BlockDigest, RecoverySidecar
from vllm.v1.core.kv_cache_manager import KVCacheManager, _mpr_env_enabled


def _manual_scores(
    query_window: torch.Tensor,
    digest_min: torch.Tensor,
    digest_max: torch.Tensor,
    score_agg: str,
) -> torch.Tensor:
    num_blocks, num_kv_heads, head_dim = digest_min.shape
    num_q_heads = query_window.shape[0]
    group_size = num_q_heads // num_kv_heads
    query_by_kv = query_window.reshape(num_kv_heads, group_size, head_dim)
    per_head = []
    for block_idx in range(num_blocks):
        block_scores = []
        for kv_head_idx in range(num_kv_heads):
            for group_idx in range(group_size):
                query = query_by_kv[kv_head_idx, group_idx]
                max_term = query * digest_max[block_idx, kv_head_idx]
                min_term = query * digest_min[block_idx, kv_head_idx]
                block_scores.append(torch.maximum(max_term, min_term).sum())
        stacked = torch.stack(block_scores)
        if score_agg == "max":
            per_head.append(stacked.max())
        else:
            per_head.append(stacked.mean())
    return torch.stack(per_head)


def test_estimate_digest_scores_matches_manual_max_and_mean():
    query_window = torch.tensor(
        [
            [1.0, -2.0],
            [0.5, 3.0],
            [-1.5, 2.0],
            [4.0, -0.25],
        ]
    )
    digest_min = torch.tensor(
        [
            [[-1.0, -0.5], [-2.0, -1.0]],
            [[0.25, -3.0], [-0.5, 1.0]],
        ]
    )
    digest_max = torch.tensor(
        [
            [[2.0, 1.5], [1.0, 3.0]],
            [[1.25, 0.5], [2.5, 4.0]],
        ]
    )

    torch.testing.assert_close(
        estimate_digest_scores(query_window, digest_min, digest_max, "max"),
        _manual_scores(query_window, digest_min, digest_max, "max"),
    )
    torch.testing.assert_close(
        estimate_digest_scores(query_window, digest_min, digest_max, "mean"),
        _manual_scores(query_window, digest_min, digest_max, "mean"),
    )


def test_mpr_config_parses_score_granularity(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_SCORE_GRANULARITY", "kv_head")

    config = MPRConfig.from_env()

    assert config.score_granularity == "kv_head"


def test_mpr_config_defaults_to_quest_style_digest_and_kv_head_scores():
    config = MPRConfig()

    assert config.digest_kind == "raw_minmax"
    assert config.score_granularity == "kv_head"
    assert config.recent_tokens == 64
    assert not config.cpu_backup_enabled
    assert config.scoring_enabled
    assert not config.recovery_enabled
    assert config.recovery_topk == 8
    assert config.recovery_policy == "topk_block"
    assert config.recovery_threshold == 0.0
    assert config.recovery_test_mutate == "off"
    assert not config.precision_tiering_enabled
    assert config.precision_policy == "top_ratio"
    assert config.tier_fp16_ratio == 0.25
    assert config.tier_int8_ratio == 0.50
    assert config.tier_int4_ratio == 0.0
    assert config.tier_high_threshold == 0.0
    assert config.tier_mid_threshold == 0.0
    assert config.tier_low_threshold == 0.0
    assert config.backup_storage_mode == "eager_fp16_int8"


def test_mpr_config_parses_cpu_backup_flag(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_CPU_BACKUP", "1")
    monkeypatch.setenv("VLLM_MPR_BACKUP_STORAGE_MODE", "fp16_only")

    config = MPRConfig.from_env()

    assert config.cpu_backup_enabled
    assert config.backup_storage_mode == "fp16_only"


def test_mpr_config_parses_scoring_enable_flag(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_SCORING_ENABLE", "0")

    config = MPRConfig.from_env()

    assert not config.scoring_enabled


def test_mpr_config_parses_recovery_flags(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_RECOVERY_ENABLE", "1")
    monkeypatch.setenv("VLLM_MPR_RECOVERY_TOPK", "4")
    monkeypatch.setenv("VLLM_MPR_RECOVERY_POLICY", "topk_block")
    monkeypatch.setenv("VLLM_MPR_RECOVERY_THRESHOLD", "12.5")
    monkeypatch.setenv("VLLM_MPR_RECOVERY_TEST_MUTATE", "zero_all")
    monkeypatch.setenv("VLLM_MPR_RECOVERY_TEST_MODE", "mutate_only")

    config = MPRConfig.from_env()

    assert config.recovery_enabled
    assert config.recovery_topk == 4
    assert config.recovery_policy == "topk_block"
    assert config.recovery_threshold == 12.5
    assert config.recovery_test_mutate == "zero_all"
    assert config.recovery_test_mode == "mutate_only"


def test_mpr_config_accepts_threshold_recovery_policy(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_RECOVERY_POLICY", "threshold_block")
    monkeypatch.setenv("VLLM_MPR_RECOVERY_THRESHOLD", "-3.25")

    config = MPRConfig.from_env()

    assert config.recovery_policy == "threshold_block"
    assert config.recovery_threshold == -3.25


def test_mpr_config_parses_top_ratio_precision_tiering(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_PRECISION_TIERING_ENABLE", "1")
    monkeypatch.setenv("VLLM_MPR_PRECISION_POLICY", "top_ratio")
    monkeypatch.setenv("VLLM_MPR_TIER_FP16_RATIO", "0.2")
    monkeypatch.setenv("VLLM_MPR_TIER_INT8_RATIO", "0.6")
    monkeypatch.setenv("VLLM_MPR_TIER_INT4_RATIO", "0.1")
    monkeypatch.setenv("VLLM_MPR_BACKUP_STORAGE_MODE", "fp16_only")

    config = MPRConfig.from_env()

    assert config.precision_tiering_enabled
    assert config.precision_policy == "top_ratio"
    assert config.tier_fp16_ratio == 0.2
    assert config.tier_int8_ratio == 0.6
    assert config.tier_int4_ratio == 0.1
    assert config.backup_storage_mode == "fp16_only"


def test_mpr_config_accepts_eager_int4_backup_storage_mode(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_BACKUP_STORAGE_MODE", "eager_fp16_int8_int4")

    config = MPRConfig.from_env()

    assert config.backup_storage_mode == "eager_fp16_int8_int4"


def test_mpr_config_parses_threshold_precision_tiering(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_PRECISION_POLICY", "threshold")
    monkeypatch.setenv("VLLM_MPR_TIER_HIGH_THRESHOLD", "7.5")
    monkeypatch.setenv("VLLM_MPR_TIER_MID_THRESHOLD", "4.5")
    monkeypatch.setenv("VLLM_MPR_TIER_LOW_THRESHOLD", "2.25")

    config = MPRConfig.from_env()

    assert config.precision_policy == "threshold"
    assert config.tier_high_threshold == 7.5
    assert config.tier_mid_threshold == 4.5
    assert config.tier_low_threshold == 2.25


def test_mpr_config_rejects_invalid_precision_policy(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_PRECISION_POLICY", "ranked")

    with pytest.raises(ValueError, match="VLLM_MPR_PRECISION_POLICY"):
        MPRConfig.from_env()


def test_mpr_config_rejects_invalid_backup_storage_mode(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_BACKUP_STORAGE_MODE", "int8_only")

    with pytest.raises(ValueError, match="VLLM_MPR_BACKUP_STORAGE_MODE"):
        MPRConfig.from_env()


@pytest.mark.parametrize(
    "backup_storage_mode",
    ["eager_fp16_int8", "eager_fp16_int8_int4"],
)
def test_mpr_config_rejects_eager_low_precision_backup_without_tiering(
    backup_storage_mode: str,
):
    with pytest.raises(ValueError, match="precision_tiering_enabled=True"):
        MPRConfig(
            cpu_backup_enabled=True,
            precision_tiering_enabled=False,
            backup_storage_mode=backup_storage_mode,
        )


def test_mpr_config_rejects_negative_tier_ratio(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_TIER_FP16_RATIO", "-0.1")

    with pytest.raises(ValueError, match="tier_fp16_ratio"):
        MPRConfig.from_env()


def test_mpr_config_rejects_invalid_int4_tier_ratio(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_TIER_INT4_RATIO", "1.1")

    with pytest.raises(ValueError, match="tier_int4_ratio"):
        MPRConfig.from_env()


def test_mpr_config_rejects_tier_ratio_sum_above_one(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_TIER_FP16_RATIO", "0.7")
    monkeypatch.setenv("VLLM_MPR_TIER_INT8_RATIO", "0.2")
    monkeypatch.setenv("VLLM_MPR_TIER_INT4_RATIO", "0.2")

    with pytest.raises(ValueError, match="tier_fp16_ratio \\+ tier_int8_ratio"):
        MPRConfig.from_env()


def test_mpr_config_rejects_inverted_tier_thresholds(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_TIER_HIGH_THRESHOLD", "1.0")
    monkeypatch.setenv("VLLM_MPR_TIER_MID_THRESHOLD", "2.0")

    with pytest.raises(ValueError, match="tier_high_threshold"):
        MPRConfig.from_env()


def test_mpr_config_rejects_inverted_mid_low_tier_thresholds(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_TIER_HIGH_THRESHOLD", "3.0")
    monkeypatch.setenv("VLLM_MPR_TIER_MID_THRESHOLD", "1.0")
    monkeypatch.setenv("VLLM_MPR_TIER_LOW_THRESHOLD", "2.0")

    with pytest.raises(ValueError, match="tier_mid_threshold"):
        MPRConfig.from_env()


def test_mpr_config_accepts_quest_cuda_backend(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_SCORING_BACKEND", "quest_cuda")

    config = MPRConfig.from_env()

    assert config.scoring_backend == "quest_cuda"
    assert isinstance(get_digest_scoring_backend(config.scoring_backend), QuestCudaScorer)


def test_quest_cuda_uses_flashinfer_nhd_layout_value():
    assert QUEST_NHD_LAYOUT == 0


def test_semantic_cpu_backup_store_copies_fp16_and_releases_by_gpu_block_id():
    store = SemanticCPUBackupStore()
    kv_block = torch.arange(16, dtype=torch.float32).reshape(2, 4, 1, 2)

    result = store.put(
        layer_name="model.layers.0.self_attn.attn",
        physical_block_id=7,
        kv_block=kv_block,
    )

    key = CPUBackupKey(
        layer_name="model.layers.0.self_attn.attn",
        physical_block_id=7,
    )
    backup = store.get(key)
    assert backup is not None
    assert backup.device.type == "cpu"
    assert backup.dtype == torch.float16
    assert result.num_bytes == backup.numel() * backup.element_size()
    torch.testing.assert_close(backup, kv_block.to(torch.float16))

    stats = store.stats()
    assert stats.block_count == 1
    assert stats.total_bytes == result.num_bytes
    assert stats.put_count == 1

    release = store.release_blocks({7})
    assert release.released_entries == 1
    assert release.released_bytes == result.num_bytes
    assert store.get(key) is None
    assert store.stats().block_count == 0


def test_semantic_cpu_backup_store_eagerly_stores_fp16_and_int8_payloads():
    store = SemanticCPUBackupStore()
    kv_block = torch.arange(16, dtype=torch.float32).reshape(2, 4, 1, 2)

    result = store.put(
        layer_name="model.layers.0.self_attn.attn",
        physical_block_id=7,
        kv_block=kv_block,
        backup_storage_mode="eager_fp16_int8",
    )

    key = CPUBackupKey(
        layer_name="model.layers.0.self_attn.attn",
        physical_block_id=7,
    )
    backup = store.get(key)
    fp16_payload = store.get_payload(key, FP16_BACKUP_FORMAT)
    int8_payload = store.get_payload(key, INT8_BACKUP_FORMAT)
    assert backup is not None
    assert fp16_payload is not None
    assert int8_payload is not None
    assert backup is fp16_payload.tensor
    assert backup.dtype == torch.float16
    assert int8_payload.quantized.dtype == torch.int8
    assert int8_payload.scale.dtype == torch.float32
    materialized_int8 = INT8BackupCodec().materialize(
        int8_payload,
        target_dtype=torch.float32,
        target_device="cpu",
    )
    int8_error = (materialized_int8 - fp16_payload.tensor.float()).abs()
    int8_bound = int8_payload.scale.unsqueeze(-1) / 2
    assert bool(torch.all(int8_error <= int8_bound + 1e-6))

    fp16_bytes = fp16_payload.payload_nbytes
    int8_bytes = (
        int8_payload.quantized.numel()
        * int8_payload.quantized.element_size()
    )
    scale_bytes = int8_payload.scale.numel() * int8_payload.scale.element_size()
    total_bytes = fp16_bytes + int8_bytes + scale_bytes
    assert result.num_bytes == fp16_bytes
    assert result.fp16_payload_bytes == fp16_bytes
    assert result.int8_payload_bytes == int8_bytes
    assert result.int8_scale_bytes == scale_bytes
    assert result.total_actual_backup_bytes == total_bytes

    stats = store.stats()
    assert stats.block_count == 1
    assert stats.fp16_payload_bytes == fp16_bytes
    assert stats.int8_payload_bytes == int8_bytes
    assert stats.int8_scale_bytes == scale_bytes
    assert stats.total_actual_backup_bytes == total_bytes
    assert stats.total_bytes == total_bytes

    release = store.release_blocks({7})
    assert release.released_entries == 1
    assert release.fp16_payload_bytes == fp16_bytes
    assert release.int8_payload_bytes == int8_bytes
    assert release.int8_scale_bytes == scale_bytes
    assert release.total_actual_backup_bytes == total_bytes
    assert release.released_bytes == total_bytes
    assert store.get(key) is None
    assert store.get_payload(key, INT8_BACKUP_FORMAT) is None
    assert store.stats().block_count == 0
    assert store.stats().total_actual_backup_bytes == 0


def test_semantic_cpu_backup_store_eagerly_stores_int4_payloads():
    store = SemanticCPUBackupStore()
    kv_block = torch.arange(16, dtype=torch.float32).reshape(2, 4, 1, 2)

    result = store.put(
        layer_name="model.layers.0.self_attn.attn",
        physical_block_id=7,
        kv_block=kv_block,
        backup_storage_mode="eager_fp16_int8_int4",
    )

    key = CPUBackupKey(
        layer_name="model.layers.0.self_attn.attn",
        physical_block_id=7,
    )
    fp16_payload = store.get_payload(key, FP16_BACKUP_FORMAT)
    int8_payload = store.get_payload(key, INT8_BACKUP_FORMAT)
    int4_payload = store.get_payload(key, INT4_BACKUP_FORMAT)
    assert fp16_payload is not None
    assert int8_payload is not None
    assert int4_payload is not None
    assert int4_payload.packed.dtype == torch.uint8
    assert int4_payload.scale.dtype == torch.float32

    materialized_int4 = INT4BackupCodec().materialize(
        int4_payload,
        target_dtype=torch.float32,
        target_device="cpu",
    )
    int4_error = (materialized_int4 - fp16_payload.tensor.float()).abs()
    int4_bound = int4_payload.scale.unsqueeze(-1) / 2
    assert bool(torch.all(int4_error <= int4_bound + 1e-6))

    fp16_bytes = fp16_payload.payload_nbytes
    int8_bytes = (
        int8_payload.quantized.numel()
        * int8_payload.quantized.element_size()
    )
    int8_scale_bytes = (
        int8_payload.scale.numel() * int8_payload.scale.element_size()
    )
    int4_bytes = (
        int4_payload.packed.numel() * int4_payload.packed.element_size()
    )
    int4_scale_bytes = (
        int4_payload.scale.numel() * int4_payload.scale.element_size()
    )
    total_bytes = (
        fp16_bytes
        + int8_bytes
        + int8_scale_bytes
        + int4_bytes
        + int4_scale_bytes
    )
    assert result.num_bytes == fp16_bytes
    assert result.int4_payload_bytes == int4_bytes
    assert result.int4_scale_bytes == int4_scale_bytes
    assert result.total_actual_backup_bytes == total_bytes

    stats = store.stats()
    assert stats.int4_payload_bytes == int4_bytes
    assert stats.int4_scale_bytes == int4_scale_bytes
    assert stats.total_actual_backup_bytes == total_bytes

    release = store.release_blocks({7})
    assert release.int4_payload_bytes == int4_bytes
    assert release.int4_scale_bytes == int4_scale_bytes
    assert release.total_actual_backup_bytes == total_bytes
    assert store.get_payload(key, INT4_BACKUP_FORMAT) is None
    assert store.stats().total_actual_backup_bytes == 0


def test_semantic_cpu_backup_store_only_creates_int4_for_int4_storage_mode():
    store = SemanticCPUBackupStore()
    kv_block = torch.arange(16, dtype=torch.float32).reshape(2, 4, 1, 2)

    store.put(
        layer_name="model.layers.0.self_attn.attn",
        physical_block_id=1,
        kv_block=kv_block,
        backup_storage_mode="fp16_only",
    )
    store.put(
        layer_name="model.layers.0.self_attn.attn",
        physical_block_id=2,
        kv_block=kv_block,
        backup_storage_mode="eager_fp16_int8",
    )

    fp16_key = CPUBackupKey(
        layer_name="model.layers.0.self_attn.attn",
        physical_block_id=1,
    )
    int8_key = CPUBackupKey(
        layer_name="model.layers.0.self_attn.attn",
        physical_block_id=2,
    )
    assert store.get_payload(fp16_key, INT8_BACKUP_FORMAT) is None
    assert store.get_payload(fp16_key, INT4_BACKUP_FORMAT) is None
    assert store.get_payload(int8_key, INT8_BACKUP_FORMAT) is not None
    assert store.get_payload(int8_key, INT4_BACKUP_FORMAT) is None


def test_sidecar_creates_cpu_backup_when_full_block_digest_is_created():
    layer_name = "model.layers.0.self_attn.attn"
    sidecar = RecoverySidecar(
        config=MPRConfig(
            enabled=True,
            cpu_backup_enabled=True,
            backup_storage_mode="fp16_only",
        )
    )
    kv_cache = torch.arange(2 * 2 * 4 * 1 * 2, dtype=torch.float32).reshape(
        2,
        2,
        4,
        1,
        2,
    )

    sidecar.observe_kv_write(
        layer_name=layer_name,
        kv_cache=kv_cache,
        slot_mapping=torch.tensor([0, 1, 2, 3]),
        block_size=4,
    )

    key = CPUBackupKey(layer_name=layer_name, physical_block_id=0)
    backup = sidecar._cpu_backup_store.get(key)
    assert backup is not None
    assert backup.dtype == torch.float16
    torch.testing.assert_close(backup, kv_cache[:, 0].to(torch.float16))
    assert sidecar.cpu_backup_stats().block_count == 1
    assert sidecar.counters["cpu_backup_created"] == 1


def test_sidecar_cpu_backup_created_debug_reports_payload_bytes(tmp_path):
    layer_name = "model.layers.0.self_attn.attn"
    sidecar = RecoverySidecar(
        config=MPRConfig(
            enabled=True,
            enable_logging=True,
            debug_dir=str(tmp_path),
            cpu_backup_enabled=True,
            precision_tiering_enabled=True,
            backup_storage_mode="eager_fp16_int8",
        )
    )
    sidecar._layer_indices[layer_name] = 0

    sidecar._maybe_backup_kv_block(
        layer_name=layer_name,
        block_id=3,
        kv_block=torch.arange(16, dtype=torch.float32).reshape(2, 4, 1, 2),
        layer_event_idx=1,
    )
    sidecar._debug_writer.close()

    events = []
    for path in tmp_path.glob("*.jsonl"):
        with path.open(encoding="utf-8") as f:
            events.extend(json.loads(line) for line in f if line.strip())
    backup_events = [
        event for event in events if event["event"] == "cpu_backup_created"
    ]
    assert len(backup_events) == 1
    event = backup_events[0]
    assert event["cpu_backup_fp16_payload_bytes"] > 0
    assert event["cpu_backup_int8_payload_bytes"] > 0
    assert event["cpu_backup_int8_scale_bytes"] > 0
    assert event["cpu_backup_total_actual_bytes"] == (
        event["cpu_backup_fp16_payload_bytes"]
        + event["cpu_backup_int8_payload_bytes"]
        + event["cpu_backup_int8_scale_bytes"]
    )
    assert event["cpu_backup_total_bytes"] == (
        event["cpu_backup_store_total_actual_bytes"]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_sidecar_cpu_backup_matches_cuda_kv_cache_block():
    layer_name = "model.layers.0.self_attn.attn"
    sidecar = RecoverySidecar(
        config=MPRConfig(
            enabled=True,
            cpu_backup_enabled=True,
            backup_storage_mode="fp16_only",
        )
    )
    kv_cache = torch.arange(
        2 * 2 * 4 * 1 * 2,
        device="cuda",
        dtype=torch.float16,
    ).reshape(2, 2, 4, 1, 2)

    sidecar.observe_kv_write(
        layer_name=layer_name,
        kv_cache=kv_cache,
        slot_mapping=torch.tensor([4, 5, 6, 7], device="cuda"),
        block_size=4,
    )

    key = CPUBackupKey(layer_name=layer_name, physical_block_id=1)
    backup = sidecar._cpu_backup_store.get(key)
    assert backup is not None
    assert backup.device.type == "cpu"
    assert backup.dtype == torch.float16
    torch.testing.assert_close(
        backup,
        kv_cache[:, 1].detach().cpu().to(torch.float16),
    )


def test_estimate_digest_score_result_exposes_gqa_metadata():
    query_window = torch.tensor(
        [
            [1.0, -2.0],
            [0.5, 3.0],
            [-1.5, 2.0],
            [4.0, -0.25],
        ]
    )
    digest_min = torch.tensor([[[-1.0, -0.5], [-2.0, -1.0]]])
    digest_max = torch.tensor([[[2.0, 1.5], [1.0, 3.0]]])

    result = estimate_digest_score_result(
        query_window,
        digest_min,
        digest_max,
        "max",
    )

    assert result.scoring_backend == "torch_quest"
    assert result.num_q_heads == 4
    assert result.num_kv_heads == 2
    assert result.group_size == 2
    assert list(result.block_scores.shape) == [1]
    assert list(result.per_query_head_scores.shape) == [1, 4]
    assert list(result.per_kv_head_scores.shape) == [1, 2]
    torch.testing.assert_close(
        result.block_scores,
        estimate_digest_scores(query_window, digest_min, digest_max, "max"),
    )


def test_query_head_aggregation_uses_conservative_gqa_union_for_max():
    per_query_head_scores = torch.tensor(
        [
            [1.0, 10.0, 2.0, 3.0],
            [4.0, 1.0, 9.0, 7.0],
        ]
    )

    max_block_scores, max_group_scores = aggregate_query_head_scores(
        per_query_head_scores,
        num_kv_heads=2,
        score_agg="max",
    )
    mean_block_scores, mean_group_scores = aggregate_query_head_scores(
        per_query_head_scores,
        num_kv_heads=2,
        score_agg="mean",
    )

    torch.testing.assert_close(
        max_group_scores,
        torch.tensor([[10.0, 3.0], [4.0, 9.0]]),
    )
    torch.testing.assert_close(max_block_scores, torch.tensor([10.0, 9.0]))
    torch.testing.assert_close(
        mean_group_scores,
        torch.tensor([[5.5, 2.5], [2.5, 8.0]]),
    )
    torch.testing.assert_close(mean_block_scores, torch.tensor([4.0, 5.25]))


def test_packed_query_head_aggregation_matches_reference_block_scores():
    per_query_head_scores = torch.tensor(
        [
            [1.0, 10.0, 2.0, 3.0],
            [4.0, 1.0, 9.0, 7.0],
            [3.0, 8.0, 5.0, 6.0],
        ]
    )
    packed_scores = per_query_head_scores.transpose(0, 1).contiguous()

    for score_agg in ("max", "mean"):
        reference_block_scores, _ = aggregate_query_head_scores(
            per_query_head_scores,
            num_kv_heads=2,
            score_agg=score_agg,
        )
        optimized_block_scores = aggregate_packed_query_head_scores(
            packed_scores,
            num_kv_heads=2,
            score_agg=score_agg,
        )

        torch.testing.assert_close(optimized_block_scores, reference_block_scores)


def test_query_head_scores_are_available_before_block_aggregation():
    query_window = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
        ]
    )
    digest_min = torch.tensor([[[-3.0, -1.0]]])
    digest_max = torch.tensor([[[2.0, 4.0]]])

    per_query_head_scores = estimate_query_head_digest_scores(
        query_window,
        digest_min,
        digest_max,
        "max",
    )

    torch.testing.assert_close(
        per_query_head_scores,
        torch.tensor([[2.0, 3.0]]),
    )


def test_sidecar_head_score_debug_fields_for_kv_head_granularity():
    sidecar = RecoverySidecar(
        config=MPRConfig(
            enabled=True,
            topk=1,
            score_granularity="kv_head",
            scoring_backend="torch_quest",
        )
    )
    query_window = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 2.0],
            [0.0, -3.0],
        ]
    )
    digest_min = torch.tensor(
        [
            [[-2.0, -1.0], [-1.0, -4.0]],
            [[-5.0, -1.0], [-1.0, -1.0]],
        ]
    )
    digest_max = torch.tensor(
        [
            [[3.0, 1.0], [1.0, 2.0]],
            [[1.0, 1.0], [1.0, 7.0]],
        ]
    )
    score_result = sidecar._scoring_backend.estimate(
        query_window=query_window,
        digest_min=digest_min,
        digest_max=digest_max,
        score_agg="max",
    )

    fields = sidecar._head_score_debug_fields(
        score_result=score_result,
        physical_block_ids=[10, 11],
        topk=1,
    )

    assert fields["num_score_heads"] == 2
    assert fields["head_score_count"] == 4
    assert fields["topk_block_ids_by_head"] == [[11], [11]]
    assert len(fields["topk_scores_by_head"]) == 2


def test_sidecar_head_score_debug_fields_for_query_head_granularity():
    sidecar = RecoverySidecar(
        config=MPRConfig(
            enabled=True,
            topk=1,
            score_granularity="query_head",
        )
    )
    score_result = estimate_digest_score_result(
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
        torch.tensor([[[-3.0, -1.0]], [[-1.0, -1.0]]]),
        torch.tensor([[[2.0, 4.0]], [[5.0, 1.0]]]),
        "max",
    )

    fields = sidecar._head_score_debug_fields(
        score_result=score_result,
        physical_block_ids=[20, 21],
        topk=1,
    )

    assert fields["num_score_heads"] == 2
    assert fields["head_score_count"] == 4
    assert fields["topk_block_ids_by_head"] == [[21], [20]]


def test_sidecar_selects_logical_score_candidates_after_recent_tail():
    sidecar = RecoverySidecar(
        config=MPRConfig(
            enabled=True,
            recent_tokens=64,
        )
    )
    context = sidecar._request_block_context(
        attn_metadata=SimpleNamespace(
            max_query_len=1,
            num_actual_tokens=1,
            num_reqs=1,
            seq_lens=torch.tensor([161]),
            block_table=torch.tensor([[10, 4, 7, 8, 9, 99, 0, 0]]),
        ),
        block_size=32,
    )

    assert context.valid_block_ids == [10, 4, 7, 8, 9, 99]
    assert context.finalized_block_ids == [10, 4, 7, 8, 9]
    assert context.protected_tail_entries == 2
    assert context.protected_block_ids == [9, 99]
    assert context.score_candidate_block_ids == [10, 4, 7, 8]


def test_sidecar_packs_candidate_digests_in_logical_order():
    layer_name = "model.layers.0.self_attn.attn"
    sidecar = RecoverySidecar(config=MPRConfig(enabled=True))
    sidecar._digest_cache[layer_name] = {
        4: BlockDigest(
            digest_min=torch.full((1, 2), -4.0),
            digest_max=torch.full((1, 2), 4.0),
            valid_token_count=32,
            block_size=32,
            layer_event_idx=0,
            digest_kind="raw_minmax",
        ),
        7: BlockDigest(
            digest_min=torch.full((1, 2), -7.0),
            digest_max=torch.full((1, 2), 7.0),
            valid_token_count=32,
            block_size=32,
            layer_event_idx=0,
            digest_kind="raw_minmax",
        ),
        10: BlockDigest(
            digest_min=torch.full((1, 2), -10.0),
            digest_max=torch.full((1, 2), 10.0),
            valid_token_count=32,
            block_size=32,
            layer_event_idx=0,
            digest_kind="raw_minmax",
        ),
    }

    block_ids, digest_min, digest_max = sidecar._pack_layer_digests(
        layer_name,
        device=torch.device("cpu"),
        dtype=torch.float32,
        candidate_block_ids=[10, 4, 123, 7],
    )

    assert block_ids == [10, 4, 7]
    torch.testing.assert_close(digest_min[:, 0, 0], torch.tensor([-10.0, -4.0, -7.0]))
    torch.testing.assert_close(digest_max[:, 0, 0], torch.tensor([10.0, 4.0, 7.0]))


def test_sidecar_release_blocks_cleans_block_keyed_state():
    layer_name = "model.layers.0.self_attn.attn"
    sidecar = RecoverySidecar(
        config=MPRConfig(
            enabled=True,
            cpu_backup_enabled=True,
            backup_storage_mode="fp16_only",
            scoring_backend="quest_cuda",
        )
    )
    sidecar._block_offsets[layer_name] = {4: {0, 1}, 7: {0, 1}}
    sidecar._digest_cache[layer_name] = {
        block_id: BlockDigest(
            digest_min=torch.full((1, 2), -float(block_id)),
            digest_max=torch.full((1, 2), float(block_id)),
            valid_token_count=2,
            block_size=2,
            layer_event_idx=0,
            digest_kind="raw_minmax",
        )
        for block_id in [4, 7]
    }
    store = QuestMetadataStore(
        metadata_page_size=2,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    store.append_digest(
        block_id=4,
        digest_min=torch.full((1, 2), -4.0),
        digest_max=torch.full((1, 2), 4.0),
    )
    sidecar._quest_metadata_stores[layer_name] = store
    sidecar._cpu_backup_store.put(
        layer_name=layer_name,
        physical_block_id=4,
        kv_block=torch.ones(2, 2, 1, 2),
        backup_storage_mode="fp16_only",
    )

    sidecar.release_blocks([4], reason="test")

    assert sidecar._block_offsets[layer_name] == {7: {0, 1}}
    assert sorted(sidecar._digest_cache[layer_name]) == [7]
    assert layer_name not in sidecar._quest_metadata_stores
    assert sidecar.cpu_backup_stats().block_count == 0
    assert sidecar.counters["blocks_released"] == 1


def test_quest_metadata_packing_adds_guard_entry():
    digest_min = torch.tensor(
        [
            [[-1.0, -2.0], [-3.0, -4.0]],
            [[-5.0, -6.0], [-7.0, -8.0]],
            [[-9.0, -10.0], [-11.0, -12.0]],
        ]
    )
    digest_max = digest_min.abs()

    packed = pack_quest_metadata_cache(
        digest_min=digest_min,
        digest_max=digest_max,
        metadata_page_size=2,
        entry_block_ids=[10, 4, 7],
    )

    assert list(packed.metadata_data.shape) == [2, 2, 2, 2, 2]
    assert packed.entry_block_ids == [10, 4, 7]
    assert packed.num_score_entries == 3
    assert packed.num_packed_entries == 4
    assert packed.metadata_last_page_len == 2
    assert packed.metadata_last_page_idx == 1
    assert packed.metadata_indices.tolist() == [0, 1]
    assert packed.metadata_indptr.tolist() == [0, 2]
    torch.testing.assert_close(packed.metadata_data[0, 0, 0], digest_max[0])
    torch.testing.assert_close(packed.metadata_data[0, 1, 0], digest_min[0])
    torch.testing.assert_close(packed.metadata_data[1, 0, 0], digest_max[2])
    torch.testing.assert_close(packed.metadata_data[1, 1, 0], digest_min[2])
    torch.testing.assert_close(
        packed.metadata_data[1, :, 1],
        torch.zeros_like(packed.metadata_data[1, :, 1]),
    )


def test_quest_metadata_store_prefix_view_matches_compact_packer():
    digest_min = torch.tensor(
        [
            [[-1.0, -2.0], [-3.0, -4.0]],
            [[-5.0, -6.0], [-7.0, -8.0]],
            [[-9.0, -10.0], [-11.0, -12.0]],
        ]
    )
    digest_max = digest_min.abs()
    store = QuestMetadataStore(
        metadata_page_size=2,
        num_kv_heads=2,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    for block_id, min_digest, max_digest in zip(
        [10, 4, 7],
        digest_min,
        digest_max,
    ):
        store.append_digest(
            block_id=block_id,
            digest_min=min_digest,
            digest_max=max_digest,
        )

    packed = store.view_prefix(3)
    compact = pack_quest_metadata_cache(
        digest_min=digest_min,
        digest_max=digest_max,
        metadata_page_size=2,
        entry_block_ids=[10, 4, 7],
    )

    assert store.can_view_prefix([10, 4])
    assert not store.can_view_prefix([10, 7])
    assert packed.entry_block_ids == compact.entry_block_ids
    assert packed.num_score_entries == compact.num_score_entries
    assert packed.num_packed_entries == compact.num_packed_entries
    assert packed.metadata_last_page_len == compact.metadata_last_page_len
    assert packed.metadata_last_page_idx == compact.metadata_last_page_idx
    torch.testing.assert_close(packed.metadata_data, compact.metadata_data)
    torch.testing.assert_close(packed.metadata_indices, compact.metadata_indices)
    torch.testing.assert_close(packed.metadata_indptr, compact.metadata_indptr)


def test_quest_metadata_store_reuses_cached_prefix_view():
    store = QuestMetadataStore(
        metadata_page_size=2,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    for block_id in [10, 4]:
        store.append_digest(
            block_id=block_id,
            digest_min=torch.full((1, 2), -float(block_id)),
            digest_max=torch.full((1, 2), float(block_id)),
        )

    first = store.view_prefix(2)
    second = store.view_prefix(2)

    assert second is first
    assert second.metadata_indptr.data_ptr() == first.metadata_indptr.data_ptr()
    assert second.metadata_indptr.tolist() == [0, 2]

    store.append_digest(
        block_id=7,
        digest_min=torch.full((1, 2), -7.0),
        digest_max=torch.full((1, 2), 7.0),
    )
    after_append = store.view_prefix(3)

    assert after_append is not first
    assert after_append.metadata_indptr.data_ptr() == first.metadata_indptr.data_ptr()
    assert after_append.metadata_indptr.tolist() == [0, 2]
    torch.testing.assert_close(
        after_append.metadata_data[1, :, 1],
        torch.zeros_like(after_append.metadata_data[1, :, 1]),
    )


def test_quest_metadata_store_keeps_real_entry_when_viewing_short_prefix():
    store = QuestMetadataStore(
        metadata_page_size=2,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    for block_id in [10, 4]:
        store.append_digest(
            block_id=block_id,
            digest_min=torch.full((1, 2), -float(block_id)),
            digest_max=torch.full((1, 2), float(block_id)),
        )

    packed = store.view_prefix(1)

    assert packed.entry_block_ids == [10]
    torch.testing.assert_close(
        packed.metadata_data[0, :, 1],
        torch.tensor([[[4.0, 4.0]], [[-4.0, -4.0]]]),
    )

    full_prefix = store.view_prefix(2)
    torch.testing.assert_close(
        full_prefix.metadata_data[0, 0, 1],
        torch.full((1, 2), 4.0),
    )
    torch.testing.assert_close(
        full_prefix.metadata_data[0, 1, 1],
        torch.full((1, 2), -4.0),
    )
    torch.testing.assert_close(
        full_prefix.metadata_data[1, :, 0],
        torch.zeros_like(full_prefix.metadata_data[1, :, 0]),
    )


def test_sidecar_quest_packed_prefix_fast_path_requires_prefix():
    layer_name = "model.layers.0.self_attn.attn"
    sidecar = RecoverySidecar(
        config=MPRConfig(enabled=True, scoring_backend="quest_cuda")
    )
    sidecar._digest_cache[layer_name] = {
        block_id: BlockDigest(
            digest_min=torch.full((1, 2), -float(block_id)),
            digest_max=torch.full((1, 2), float(block_id)),
            valid_token_count=2,
            block_size=2,
            layer_event_idx=0,
            digest_kind="raw_minmax",
        )
        for block_id in [10, 4, 7]
    }
    store = QuestMetadataStore(
        metadata_page_size=2,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    for block_id in [10, 4, 7]:
        digest = sidecar._digest_cache[layer_name][block_id]
        store.append_digest(
            block_id=block_id,
            digest_min=digest.digest_min,
            digest_max=digest.digest_max,
        )
    sidecar._quest_metadata_stores[layer_name] = store

    packed, reason = sidecar._try_get_quest_packed_prefix(
        layer_name=layer_name,
        physical_block_ids=[10, 4],
        block_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert reason is None
    assert packed is not None
    assert packed.entry_block_ids == [10, 4]

    packed, reason = sidecar._try_get_quest_packed_prefix(
        layer_name=layer_name,
        physical_block_ids=[10, 7],
        block_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert packed is None
    assert reason == "non_prefix_score_candidates"


def test_quest_cuda_backend_fails_clearly_without_cuda_tensor():
    scorer = QuestCudaScorer()

    with pytest.raises(RuntimeError, match="requires CUDA"):
        scorer.estimate(
            query_window=torch.ones(4, 2),
            digest_min=torch.zeros(2, 1, 2),
            digest_max=torch.ones(2, 1, 2),
            score_agg="max",
            metadata_page_size=2,
        )


def test_quest_cuda_backend_rejects_unsupported_gqa_group_size():
    scorer = QuestCudaScorer()

    with pytest.raises(RuntimeError, match="group_size=2"):
        scorer.estimate(
            query_window=torch.ones(4, 2),
            digest_min=torch.zeros(2, 2, 2),
            digest_max=torch.ones(2, 2, 2),
            score_agg="max",
            metadata_page_size=2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_quest_cuda_backend_matches_torch_reference_when_op_is_built():
    torch.manual_seed(0)
    query_window = torch.randn(4, 64, device="cuda", dtype=torch.float16)
    digest_center = torch.randn(3, 1, 64, device="cuda", dtype=torch.float16)
    digest_radius = torch.rand(3, 1, 64, device="cuda", dtype=torch.float16)
    digest_min = digest_center - digest_radius
    digest_max = digest_center + digest_radius

    torch_result = TorchQuestScorer().estimate(
        query_window=query_window,
        digest_min=digest_min,
        digest_max=digest_max,
        score_agg="max",
        metadata_page_size=2,
    )
    try:
        scorer = QuestCudaScorer()
        cuda_result = scorer.estimate(
            query_window=query_window,
            digest_min=digest_min,
            digest_max=digest_max,
            score_agg="max",
            metadata_page_size=2,
        )
    except RuntimeError as exc:
        message = str(exc)
        if (
            "custom op is not registered" in message
            or "placeholder" in message
            or "could not import vLLM custom ops" in message
        ):
            pytest.skip(f"quest_cuda custom op is not built: {message}")
        raise

    packed = pack_quest_metadata_cache(
        digest_min=digest_min,
        digest_max=digest_max,
        metadata_page_size=2,
        add_guard_entry=True,
    )
    packed_cuda_result = scorer.estimate_packed(
        query_window=query_window,
        packed=packed,
        num_kv_heads=1,
        score_agg="max",
    )

    torch.testing.assert_close(
        cuda_result.per_query_head_scores,
        torch_result.per_query_head_scores,
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        packed_cuda_result.per_query_head_scores,
        torch_result.per_query_head_scores,
        atol=2e-2,
        rtol=2e-2,
    )


def test_estimate_digest_scores_rejects_invalid_head_grouping():
    query_window = torch.ones(3, 2)
    digest_min = torch.zeros(1, 2, 2)
    digest_max = torch.ones(1, 2, 2)

    with pytest.raises(AssertionError, match="num_q_heads"):
        estimate_digest_scores(query_window, digest_min, digest_max, "max")


def test_sidecar_query_scoring_uses_current_query_without_window_state():
    sidecar = RecoverySidecar(
        config=MPRConfig(
            enabled=True,
            window_size=2,
            topk=1,
            score_agg="mean",
            scoring_backend="torch_quest",
        )
    )
    layer_name = "model.layers.0.self_attn.attn"
    sidecar._digest_cache[layer_name] = {
        7: BlockDigest(
            digest_min=torch.zeros(1, 2),
            digest_max=torch.ones(1, 2),
            valid_token_count=4,
            block_size=4,
            layer_event_idx=1,
        )
    }
    metadata = SimpleNamespace(max_query_len=1, num_actual_tokens=1)

    sidecar.observe_query(layer_name, torch.tensor([[[1.0, 3.0]]]), metadata)
    sidecar.observe_query(layer_name, torch.tensor([[[5.0, 7.0]]]), metadata)
    sidecar.observe_query(layer_name, torch.tensor([[[9.0, 11.0]]]), metadata)

    assert layer_name not in sidecar._query_windows
    assert sidecar.counters["score_estimated"] == 3


def test_sidecar_optimized_scoring_uses_block_score_only_packed_path():
    class FakeOptimizedBackend:
        def __init__(self):
            self.materialize_head_scores: list[bool] = []

        def estimate_packed_optimized(self, **kwargs):
            self.materialize_head_scores.append(kwargs["materialize_head_scores"])
            return SimpleNamespace(
                block_scores=torch.tensor([1.0, 2.0]),
                per_query_head_scores=torch.empty(2, 0),
                per_kv_head_scores=torch.empty(2, 0),
                score_agg=kwargs["score_agg"],
                scoring_backend="fake_quest_cuda",
                num_q_heads=1,
                num_kv_heads=1,
                group_size=1,
            )

        def estimate(self, **kwargs):
            raise AssertionError("optimized packed path should be used")

    layer_name = "model.layers.0.self_attn.attn"
    sidecar = RecoverySidecar(
        config=MPRConfig(enabled=True, scoring_backend="quest_cuda")
    )
    sidecar._scoring_backend = FakeOptimizedBackend()
    sidecar._digest_cache[layer_name] = {
        block_id: BlockDigest(
            digest_min=torch.full((1, 2), -float(block_id)),
            digest_max=torch.full((1, 2), float(block_id)),
            valid_token_count=2,
            block_size=2,
            layer_event_idx=0,
        )
        for block_id in [3, 4]
    }
    store = QuestMetadataStore(
        metadata_page_size=2,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    for block_id in [3, 4]:
        digest = sidecar._digest_cache[layer_name][block_id]
        store.append_digest(
            block_id=block_id,
            digest_min=digest.digest_min,
            digest_max=digest.digest_max,
        )
    sidecar._quest_metadata_stores[layer_name] = store

    score_context = sidecar._estimate_query_scores_optimized(
        layer_name=layer_name,
        query=torch.ones(1, 1, 2),
        attn_metadata=SimpleNamespace(max_query_len=1, num_actual_tokens=1),
        should_record=False,
        layer_event_idx=0,
        block_size_override=2,
    )

    assert score_context is not None
    assert sidecar._scoring_backend.materialize_head_scores == [False]
    torch.testing.assert_close(
        score_context.score_result.block_scores,
        torch.tensor([1.0, 2.0]),
    )


def test_sidecar_optimized_scoring_materializes_heads_for_head_debug():
    class FakeOptimizedBackend:
        def __init__(self):
            self.materialize_head_scores: list[bool] = []

        def estimate_packed_optimized(self, **kwargs):
            self.materialize_head_scores.append(kwargs["materialize_head_scores"])
            return SimpleNamespace(
                block_scores=torch.tensor([1.0, 2.0]),
                per_query_head_scores=torch.tensor([[1.0], [2.0]]),
                per_kv_head_scores=torch.tensor([[1.0], [2.0]]),
                score_agg=kwargs["score_agg"],
                scoring_backend="fake_quest_cuda",
                num_q_heads=1,
                num_kv_heads=1,
                group_size=1,
            )

        def estimate(self, **kwargs):
            raise AssertionError("optimized packed path should be used")

    layer_name = "model.layers.0.self_attn.attn"
    sidecar = RecoverySidecar(
        config=MPRConfig(
            enabled=True,
            enable_logging=True,
            scoring_backend="quest_cuda",
            score_granularity="kv_head",
            topk=1,
        )
    )
    sidecar._scoring_backend = FakeOptimizedBackend()
    sidecar._digest_cache[layer_name] = {
        block_id: BlockDigest(
            digest_min=torch.full((1, 2), -float(block_id)),
            digest_max=torch.full((1, 2), float(block_id)),
            valid_token_count=2,
            block_size=2,
            layer_event_idx=0,
        )
        for block_id in [3, 4]
    }
    store = QuestMetadataStore(
        metadata_page_size=2,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    for block_id in [3, 4]:
        digest = sidecar._digest_cache[layer_name][block_id]
        store.append_digest(
            block_id=block_id,
            digest_min=digest.digest_min,
            digest_max=digest.digest_max,
        )
    sidecar._quest_metadata_stores[layer_name] = store

    score_context = sidecar._estimate_query_scores_optimized(
        layer_name=layer_name,
        query=torch.ones(1, 1, 2),
        attn_metadata=SimpleNamespace(max_query_len=1, num_actual_tokens=1),
        should_record=True,
        layer_event_idx=0,
        block_size_override=2,
    )

    assert score_context is not None
    assert sidecar._scoring_backend.materialize_head_scores == [True]
    assert score_context.head_debug["num_score_heads"] == 1
    assert score_context.head_debug["topk_block_ids_by_head"] == [[4]]


def test_sidecar_query_scoring_skips_non_single_request_decode():
    sidecar = RecoverySidecar(config=MPRConfig(enabled=True))
    metadata = SimpleNamespace(max_query_len=1, num_actual_tokens=2)

    sidecar.observe_query(
        "model.layers.0.self_attn.attn",
        torch.ones(2, 1, 2),
        metadata,
    )

    assert sidecar.counters["score_skipped"] == 1
    assert "model.layers.0.self_attn.attn" not in sidecar._query_windows


def test_sidecar_query_scoring_can_be_disabled_for_cpu_backup_only():
    sidecar = RecoverySidecar(
        config=MPRConfig(enabled=True, scoring_enabled=False)
    )
    layer_name = "model.layers.0.self_attn.attn"
    sidecar._digest_cache[layer_name] = {
        7: BlockDigest(
            digest_min=torch.zeros(1, 2),
            digest_max=torch.ones(1, 2),
            valid_token_count=4,
            block_size=4,
            layer_event_idx=1,
        )
    }

    sidecar.observe_query(
        layer_name,
        torch.tensor([[[1.0, 3.0]]]),
        SimpleNamespace(max_query_len=1, num_actual_tokens=1),
    )

    assert sidecar.counters["score_skipped_disabled"] == 1
    assert layer_name not in sidecar._query_windows
    assert sidecar.counters["score_estimated"] == 0


def test_sidecar_score_debug_fields_describe_block_table_state():
    sidecar = RecoverySidecar(config=MPRConfig(enabled=True, recent_tokens=0))
    metadata = SimpleNamespace(
        max_query_len=1,
        num_actual_tokens=1,
        seq_lens=torch.tensor([33]),
        block_table=torch.tensor([[10, 11, 12, 99]]),
    )
    request_context = sidecar._request_block_context(
        attn_metadata=metadata,
        block_size=16,
    )

    fields = sidecar._score_block_debug_fields(
        request_context=request_context,
        observed_digest_block_ids=[10, 12, 20],
    )

    assert fields["num_reqs"] == 1
    assert fields["seq_lens"] == [33]
    assert fields["block_table_row"] == [10, 11, 12, 99]
    assert fields["valid_block_ids"] == [10, 11, 12]
    assert fields["finalized_block_ids"] == [10, 11]
    assert fields["score_candidate_block_ids"] == [10, 11]
    assert fields["observed_digest_block_ids"] == [10, 12, 20]
    assert fields["missing_digest_blocks"] == [11]
    assert fields["extra_digest_blocks"] == [12, 20]


def test_kv_cache_manager_mpr_env_helper_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_MPR_ENABLE", raising=False)

    assert not _mpr_env_enabled("VLLM_MPR_ENABLE")


def test_kv_cache_manager_mpr_hook_is_noop_when_env_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_MPR_ENABLE", raising=False)

    class FakeManager:
        def get_block_ids(self, request_id):
            raise AssertionError("block ids should not be queried when MPR is off")

    KVCacheManager._maybe_release_mpr_blocks(
        FakeManager(),
        SimpleNamespace(request_id="req-0"),
    )


def test_kv_cache_manager_mpr_hook_releases_flattened_block_ids(monkeypatch):
    monkeypatch.setenv("VLLM_MPR_ENABLE", "1")
    calls = []

    class FakeManager:
        def get_block_ids(self, request_id):
            assert request_id == "req-1"
            return ([3, 1], [2, 3])

    class FakeSidecar:
        def release_blocks(self, block_ids, *, request_id, reason):
            calls.append((block_ids, request_id, reason))

    import vllm.v1.mixed_precision_recovery as mpr

    monkeypatch.setattr(mpr, "get_mpr_sidecar", lambda: FakeSidecar())

    KVCacheManager._maybe_release_mpr_blocks(
        FakeManager(),
        SimpleNamespace(request_id="req-1"),
    )

    assert calls == [([1, 2, 3], "req-1", "kv_cache_manager_free")]
