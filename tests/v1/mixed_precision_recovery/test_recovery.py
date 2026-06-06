# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.mixed_precision_recovery.config import MPRConfig
from vllm.v1.mixed_precision_recovery.cpu_backup import SemanticCPUBackupStore
from vllm.v1.mixed_precision_recovery.recovery import (
    BlockRecoveryManager,
    select_recovery_block_ids,
)
from vllm.v1.mixed_precision_recovery.sidecar import BlockDigest, RecoverySidecar
from vllm.model_executor.layers.attention import attention as attention_module


def _score_result(values: list[float]):
    return SimpleNamespace(block_scores=torch.tensor(values, dtype=torch.float32))


def _make_recovery_sidecar(
    *,
    recovery_enabled: bool = True,
    recovery_policy: str = "topk_block",
    recovery_threshold: float = 0.0,
    recovery_test_mutate: str = "off",
    recovery_test_mode: str = "recover",
) -> RecoverySidecar:
    layer_name = "model.layers.0.self_attn.attn"
    sidecar = RecoverySidecar(
        config=MPRConfig(
            enabled=True,
            recent_tokens=0,
            cpu_backup_enabled=True,
            backup_storage_mode="fp16_only",
            recovery_enabled=recovery_enabled,
            recovery_topk=1,
            recovery_policy=recovery_policy,
            recovery_threshold=recovery_threshold,
            recovery_test_mutate=recovery_test_mutate,
            recovery_test_mode=recovery_test_mode,
        )
    )
    sidecar._digest_cache[layer_name] = {
        0: BlockDigest(
            digest_min=torch.zeros(1, 2),
            digest_max=torch.ones(1, 2),
            valid_token_count=4,
            block_size=4,
            layer_event_idx=0,
            digest_kind="raw_minmax",
        ),
        1: BlockDigest(
            digest_min=torch.zeros(1, 2),
            digest_max=torch.full((1, 2), 5.0),
            valid_token_count=4,
            block_size=4,
            layer_event_idx=0,
            digest_kind="raw_minmax",
        ),
    }
    return sidecar


def _decode_metadata():
    return SimpleNamespace(
        max_query_len=1,
        num_actual_tokens=1,
        num_reqs=1,
        seq_lens=torch.tensor([8]),
        block_table=torch.tensor([[0, 1]]),
    )


def test_select_recovery_block_ids_topk_block():
    selected = select_recovery_block_ids(
        score_result=_score_result([0.5, 7.0, 1.25, 3.0]),
        physical_block_ids=[10, 4, 7, 8],
        policy="topk_block",
        topk=2,
        threshold=0.0,
    )

    assert selected == [4, 8]


def test_select_recovery_block_ids_threshold_block_preserves_candidate_order():
    selected = select_recovery_block_ids(
        score_result=_score_result([0.5, 7.0, 1.25, 3.0]),
        physical_block_ids=[10, 4, 7, 8],
        policy="threshold_block",
        topk=1,
        threshold=2.0,
    )

    assert selected == [4, 8]


def test_select_recovery_block_ids_rejects_mismatched_score_count():
    with pytest.raises(ValueError, match="physical_block_ids length"):
        select_recovery_block_ids(
            score_result=_score_result([1.0, 2.0]),
            physical_block_ids=[10],
            policy="topk_block",
            topk=1,
            threshold=0.0,
        )


def test_block_recovery_materializes_cpu_backup_into_target_block():
    layer_name = "model.layers.0.self_attn.attn"
    store = SemanticCPUBackupStore()
    kv_cache = torch.zeros(2, 3, 4, 1, 2, dtype=torch.float32)
    backup_source = torch.arange(16, dtype=torch.float32).reshape(2, 4, 1, 2)
    store.put(
        layer_name=layer_name,
        physical_block_id=1,
        kv_block=backup_source,
    )

    result = BlockRecoveryManager().materialize_blocks(
        selected_block_ids=[1],
        kv_cache=kv_cache,
        cpu_backup_store=store,
        layer_name=layer_name,
    )

    assert result.selected_block_ids == [1]
    assert result.recovered_block_ids == [1]
    assert result.missing_backup_block_ids == []
    assert result.skipped_block_ids == []
    assert result.recovered_bytes == kv_cache[:, 1].numel() * kv_cache.element_size()
    torch.testing.assert_close(kv_cache[:, 1], backup_source.to(kv_cache.dtype))
    torch.testing.assert_close(kv_cache[:, 0], torch.zeros_like(kv_cache[:, 0]))
    torch.testing.assert_close(kv_cache[:, 2], torch.zeros_like(kv_cache[:, 2]))


def test_block_recovery_reports_missing_and_out_of_range_blocks():
    layer_name = "model.layers.0.self_attn.attn"
    store = SemanticCPUBackupStore()
    kv_cache = torch.zeros(2, 2, 4, 1, 2, dtype=torch.float32)

    result = BlockRecoveryManager().materialize_blocks(
        selected_block_ids=[1, 4],
        kv_cache=kv_cache,
        cpu_backup_store=store,
        layer_name=layer_name,
    )

    assert result.selected_block_ids == [1, 4]
    assert result.recovered_block_ids == []
    assert result.missing_backup_block_ids == [1]
    assert result.skipped_block_ids == [4]
    assert result.recovered_bytes == 0
    torch.testing.assert_close(kv_cache, torch.zeros_like(kv_cache))


def test_block_recovery_skips_shape_mismatch_without_mutating_target():
    layer_name = "model.layers.0.self_attn.attn"
    store = SemanticCPUBackupStore()
    kv_cache = torch.ones(2, 2, 4, 1, 2, dtype=torch.float32)
    store.put(
        layer_name=layer_name,
        physical_block_id=1,
        kv_block=torch.zeros(2, 3, 1, 2),
    )

    result = BlockRecoveryManager().materialize_blocks(
        selected_block_ids=[1],
        kv_cache=kv_cache,
        cpu_backup_store=store,
        layer_name=layer_name,
    )

    assert result.recovered_block_ids == []
    assert result.missing_backup_block_ids == []
    assert result.skipped_block_ids == [1]
    torch.testing.assert_close(kv_cache, torch.ones_like(kv_cache))


def test_block_recovery_recover_uses_threshold_policy():
    layer_name = "model.layers.0.self_attn.attn"
    store = SemanticCPUBackupStore()
    kv_cache = torch.zeros(2, 3, 4, 1, 2, dtype=torch.float32)
    block_one = torch.ones(2, 4, 1, 2)
    block_two = torch.full((2, 4, 1, 2), 2.0)
    store.put(layer_name=layer_name, physical_block_id=1, kv_block=block_one)
    store.put(layer_name=layer_name, physical_block_id=2, kv_block=block_two)

    result = BlockRecoveryManager().recover(
        score_result=_score_result([0.5, 3.0, 4.0]),
        physical_block_ids=[0, 1, 2],
        kv_cache=kv_cache,
        cpu_backup_store=store,
        layer_name=layer_name,
        policy="threshold_block",
        topk=1,
        threshold=2.0,
    )

    assert result.selected_block_ids == [1, 2]
    assert result.recovered_block_ids == [1, 2]
    torch.testing.assert_close(kv_cache[:, 1], block_one)
    torch.testing.assert_close(kv_cache[:, 2], block_two)


def test_sidecar_recovery_disabled_does_not_mutate_kv_cache():
    layer_name = "model.layers.0.self_attn.attn"
    sidecar = _make_recovery_sidecar(recovery_enabled=False)
    kv_cache = torch.zeros(2, 2, 4, 1, 2, dtype=torch.float32)
    before = kv_cache.clone()

    sidecar.recover_before_attention_with_test_mutation(
        layer_name=layer_name,
        query=torch.ones(1, 1, 2),
        attn_metadata=_decode_metadata(),
        kv_cache=kv_cache,
        block_size=4,
    )

    torch.testing.assert_close(kv_cache, before)
    assert layer_name not in sidecar._query_windows
    assert sidecar.counters["recovery_materialized"] == 0


def test_sidecar_recovery_materializes_topk_block():
    layer_name = "model.layers.0.self_attn.attn"
    sidecar = _make_recovery_sidecar(recovery_policy="topk_block")
    kv_cache = torch.zeros(2, 2, 4, 1, 2, dtype=torch.float32)
    backup = torch.full((2, 4, 1, 2), 9.0)
    sidecar._cpu_backup_store.put(
        layer_name=layer_name,
        physical_block_id=1,
        kv_block=backup,
    )

    sidecar.recover_before_attention_with_test_mutation(
        layer_name=layer_name,
        query=torch.ones(1, 1, 2),
        attn_metadata=_decode_metadata(),
        kv_cache=kv_cache,
        block_size=4,
    )

    torch.testing.assert_close(kv_cache[:, 0], torch.zeros_like(kv_cache[:, 0]))
    torch.testing.assert_close(kv_cache[:, 1], backup)
    assert len(sidecar._query_windows[layer_name]) == 1
    assert sidecar.counters["recovery_materialized"] == 1


def test_sidecar_recovery_threshold_reports_missing_backup():
    layer_name = "model.layers.0.self_attn.attn"
    sidecar = _make_recovery_sidecar(
        recovery_policy="threshold_block",
        recovery_threshold=1.0,
    )
    kv_cache = torch.zeros(2, 2, 4, 1, 2, dtype=torch.float32)
    backup = torch.full((2, 4, 1, 2), 3.0)
    sidecar._cpu_backup_store.put(
        layer_name=layer_name,
        physical_block_id=1,
        kv_block=backup,
    )

    sidecar.recover_before_attention_with_test_mutation(
        layer_name=layer_name,
        query=torch.ones(1, 1, 2),
        attn_metadata=_decode_metadata(),
        kv_cache=kv_cache,
        block_size=4,
    )

    torch.testing.assert_close(kv_cache[:, 0], torch.zeros_like(kv_cache[:, 0]))
    torch.testing.assert_close(kv_cache[:, 1], backup)
    assert sidecar.counters["recovery_materialized"] == 1


def test_sidecar_recovery_zero_selected_mutation_restores_from_backup():
    layer_name = "model.layers.0.self_attn.attn"
    sidecar = _make_recovery_sidecar(
        recovery_policy="topk_block",
        recovery_test_mutate="zero_selected",
    )
    kv_cache = torch.full((2, 2, 4, 1, 2), -5.0, dtype=torch.float32)
    backup = torch.full((2, 4, 1, 2), 11.0)
    sidecar._cpu_backup_store.put(
        layer_name=layer_name,
        physical_block_id=1,
        kv_block=backup,
    )

    sidecar.recover_before_attention_with_test_mutation(
        layer_name=layer_name,
        query=torch.ones(1, 1, 2),
        attn_metadata=_decode_metadata(),
        kv_cache=kv_cache,
        block_size=4,
    )

    torch.testing.assert_close(kv_cache[:, 0], torch.full_like(kv_cache[:, 0], -5.0))
    torch.testing.assert_close(kv_cache[:, 1], backup)
    assert sidecar.counters["recovery_materialized"] == 1


def test_sidecar_recovery_test_mutate_only_skips_materialization():
    layer_name = "model.layers.0.self_attn.attn"
    sidecar = _make_recovery_sidecar(
        recovery_policy="topk_block",
        recovery_test_mutate="zero_selected",
        recovery_test_mode="mutate_only",
    )
    kv_cache = torch.full((2, 2, 4, 1, 2), -5.0, dtype=torch.float32)
    backup = torch.full((2, 4, 1, 2), 11.0)
    sidecar._cpu_backup_store.put(
        layer_name=layer_name,
        physical_block_id=1,
        kv_block=backup,
    )

    sidecar.recover_before_attention_with_test_mutation(
        layer_name=layer_name,
        query=torch.ones(1, 1, 2),
        attn_metadata=_decode_metadata(),
        kv_cache=kv_cache,
        block_size=4,
    )

    torch.testing.assert_close(kv_cache[:, 0], torch.full_like(kv_cache[:, 0], -5.0))
    torch.testing.assert_close(kv_cache[:, 1], torch.zeros_like(kv_cache[:, 1]))
    assert sidecar.counters["recovery_test_mutated"] == 1
    assert sidecar.counters["recovery_materialized"] == 0


def test_sidecar_recovery_test_zero_all_mutates_whole_kv_cache():
    layer_name = "model.layers.0.self_attn.attn"
    sidecar = _make_recovery_sidecar(
        recovery_policy="topk_block",
        recovery_test_mutate="zero_all",
        recovery_test_mode="mutate_only",
    )
    kv_cache = torch.full((2, 2, 4, 1, 2), -5.0, dtype=torch.float32)
    backup = torch.full((2, 4, 1, 2), 11.0)
    sidecar._cpu_backup_store.put(
        layer_name=layer_name,
        physical_block_id=1,
        kv_block=backup,
    )

    sidecar.recover_before_attention_with_test_mutation(
        layer_name=layer_name,
        query=torch.ones(1, 1, 2),
        attn_metadata=_decode_metadata(),
        kv_cache=kv_cache,
        block_size=4,
    )

    torch.testing.assert_close(kv_cache, torch.zeros_like(kv_cache))
    assert sidecar.counters["recovery_test_mutated"] == 1
    assert sidecar.counters["recovery_materialized"] == 0


def test_sidecar_recovery_entrypoint_does_not_apply_test_mutation():
    layer_name = "model.layers.0.self_attn.attn"
    sidecar = _make_recovery_sidecar(
        recovery_policy="topk_block",
        recovery_test_mutate="zero_selected",
    )
    kv_cache = torch.full((2, 2, 4, 1, 2), -5.0, dtype=torch.float32)

    sidecar.recover_before_attention(
        layer_name=layer_name,
        query=torch.ones(1, 1, 2),
        attn_metadata=_decode_metadata(),
        kv_cache=kv_cache,
        block_size=4,
    )

    torch.testing.assert_close(kv_cache[:, 0], torch.full_like(kv_cache[:, 0], -5.0))
    torch.testing.assert_close(kv_cache[:, 1], torch.full_like(kv_cache[:, 1], -5.0))
    assert sidecar.counters["recovery_materialized"] == 1


def test_attention_mpr_hook_observes_query_when_recovery_disabled(monkeypatch):
    calls = []

    class FakeSidecar:
        config = SimpleNamespace(recovery_enabled=False, recovery_test_mutate="off")

        def observe_query(self, **kwargs):
            calls.append(("observe", kwargs))

        def recover_before_attention(self, **kwargs):
            calls.append(("recover", kwargs))

        def recover_before_attention_with_test_mutation(self, **kwargs):
            calls.append(("recover_test", kwargs))

    monkeypatch.setattr(attention_module.envs, "VLLM_MPR_ENABLE", True)
    monkeypatch.setattr(
        attention_module,
        "get_forward_context",
        lambda: SimpleNamespace(is_dummy_run=False),
    )
    import vllm.v1.mixed_precision_recovery as mpr

    monkeypatch.setattr(mpr, "get_mpr_sidecar", lambda: FakeSidecar())

    query = torch.ones(1, 1, 2)
    metadata = _decode_metadata()
    attention_module._maybe_observe_or_recover_mpr_query(
        "model.layers.0.self_attn.attn",
        SimpleNamespace(impl=SimpleNamespace(block_size=4)),
        torch.zeros(2, 2, 4, 1, 2),
        query,
        metadata,
    )

    assert len(calls) == 1
    name, kwargs = calls[0]
    assert name == "observe"
    assert kwargs["layer_name"] == "model.layers.0.self_attn.attn"
    assert kwargs["query"] is query
    assert kwargs["attn_metadata"] is metadata


def test_attention_mpr_hook_recovers_when_recovery_enabled(monkeypatch):
    calls = []

    class FakeSidecar:
        config = SimpleNamespace(recovery_enabled=True, recovery_test_mutate="off")

        def observe_query(self, **kwargs):
            calls.append(("observe", kwargs))

        def recover_before_attention(self, **kwargs):
            calls.append(("recover", kwargs))

        def recover_before_attention_with_test_mutation(self, **kwargs):
            calls.append(("recover_test", kwargs))

    monkeypatch.setattr(attention_module.envs, "VLLM_MPR_ENABLE", True)
    monkeypatch.setattr(
        attention_module,
        "get_forward_context",
        lambda: SimpleNamespace(is_dummy_run=False),
    )
    import vllm.v1.mixed_precision_recovery as mpr

    monkeypatch.setattr(mpr, "get_mpr_sidecar", lambda: FakeSidecar())

    kv_cache = torch.zeros(2, 2, 4, 1, 2)
    query = torch.ones(1, 1, 2)
    metadata = _decode_metadata()
    attention_module._maybe_observe_or_recover_mpr_query(
        "model.layers.0.self_attn.attn",
        SimpleNamespace(impl=SimpleNamespace(block_size=4)),
        kv_cache,
        query,
        metadata,
    )

    assert len(calls) == 1
    name, kwargs = calls[0]
    assert name == "recover"
    assert kwargs["layer_name"] == "model.layers.0.self_attn.attn"
    assert kwargs["query"] is query
    assert kwargs["attn_metadata"] is metadata
    assert kwargs["kv_cache"] is kv_cache
    assert kwargs["block_size"] == 4


def test_attention_mpr_hook_uses_test_mutation_wrapper(monkeypatch):
    calls = []

    class FakeSidecar:
        config = SimpleNamespace(
            recovery_enabled=True,
            recovery_test_mutate="zero_selected",
        )

        def observe_query(self, **kwargs):
            calls.append(("observe", kwargs))

        def recover_before_attention(self, **kwargs):
            calls.append(("recover", kwargs))

        def recover_before_attention_with_test_mutation(self, **kwargs):
            calls.append(("recover_test", kwargs))

    monkeypatch.setattr(attention_module.envs, "VLLM_MPR_ENABLE", True)
    monkeypatch.setattr(
        attention_module,
        "get_forward_context",
        lambda: SimpleNamespace(is_dummy_run=False),
    )
    import vllm.v1.mixed_precision_recovery as mpr

    monkeypatch.setattr(mpr, "get_mpr_sidecar", lambda: FakeSidecar())

    kv_cache = torch.zeros(2, 2, 4, 1, 2)
    query = torch.ones(1, 1, 2)
    metadata = _decode_metadata()
    attention_module._maybe_observe_or_recover_mpr_query(
        "model.layers.0.self_attn.attn",
        SimpleNamespace(impl=SimpleNamespace(block_size=4)),
        kv_cache,
        query,
        metadata,
    )

    assert len(calls) == 1
    name, kwargs = calls[0]
    assert name == "recover_test"
    assert kwargs["layer_name"] == "model.layers.0.self_attn.attn"
    assert kwargs["query"] is query
    assert kwargs["attn_metadata"] is metadata
    assert kwargs["kv_cache"] is kv_cache
    assert kwargs["block_size"] == 4
