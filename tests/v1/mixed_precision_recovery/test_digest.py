# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.mixed_precision_recovery.config import MPRConfig
from vllm.v1.mixed_precision_recovery.digest import (
    ARKVALE_DIGEST_KIND,
    RAW_MINMAX_DIGEST_KIND,
    summarize_key_block,
)
from vllm.v1.mixed_precision_recovery.sidecar import RecoverySidecar


def test_summarize_key_block_matches_arkvale_formula():
    """Check that the standalone digest helper matches the ArkVale formula."""
    key_block = torch.tensor(
        [
            [[1.0, 3.0], [2.0, 4.0]],
            [[5.0, 1.0], [6.0, 2.0]],
            [[3.0, 7.0], [4.0, 8.0]],
        ]
    )

    raw_max = key_block.amax(dim=0)
    raw_min = key_block.amin(dim=0)
    centers = (raw_max + raw_min) / 2
    dists = (centers.unsqueeze(0) - key_block).abs().mean(dim=0)

    digest = summarize_key_block(key_block)

    assert digest.block_size == 3
    assert digest.valid_token_count == 3
    assert digest.digest_kind == ARKVALE_DIGEST_KIND
    torch.testing.assert_close(digest.digest_min, centers - dists)
    torch.testing.assert_close(digest.digest_max, centers + dists)


def test_summarize_key_block_supports_raw_minmax_digest():
    """Check that raw_minmax preserves Quest-style page extrema."""
    key_block = torch.tensor(
        [
            [[1.0, 3.0], [2.0, 4.0]],
            [[5.0, 1.0], [6.0, 2.0]],
            [[3.0, 7.0], [4.0, 8.0]],
        ]
    )

    digest = summarize_key_block(
        key_block,
        digest_kind=RAW_MINMAX_DIGEST_KIND,
    )

    assert digest.digest_kind == RAW_MINMAX_DIGEST_KIND
    torch.testing.assert_close(digest.digest_min, key_block.amin(dim=0))
    torch.testing.assert_close(digest.digest_max, key_block.amax(dim=0))


def test_sidecar_creates_digest_once_block_is_full():
    """Check that sidecar stores the exact digest for a completed KV block."""
    sidecar = RecoverySidecar(
        config=MPRConfig(enabled=True, digest_kind=ARKVALE_DIGEST_KIND)
    )
    kv_cache = torch.zeros(2, 2, 4, 1, 2)
    kv_cache[0, 1] = torch.tensor(
        [
            [[1.0, 3.0]],
            [[5.0, 1.0]],
            [[3.0, 7.0]],
            [[9.0, 5.0]],
        ]
    )

    sidecar.observe_kv_write(
        layer_name="model.layers.0.self_attn.attn",
        kv_cache=kv_cache,
        slot_mapping=torch.tensor([4, 5]),
        block_size=4,
    )

    assert sidecar._digest_cache["model.layers.0.self_attn.attn"] == {}

    sidecar.observe_kv_write(
        layer_name="model.layers.0.self_attn.attn",
        kv_cache=kv_cache,
        slot_mapping=torch.tensor([6, 7]),
        block_size=4,
    )

    block_digest = sidecar._digest_cache["model.layers.0.self_attn.attn"][1]
    expected_digest = summarize_key_block(kv_cache[0, 1])

    assert block_digest.block_size == 4
    assert block_digest.valid_token_count == 4
    assert block_digest.digest_kind == ARKVALE_DIGEST_KIND
    assert list(block_digest.digest_min.shape) == [1, 2]
    assert list(block_digest.digest_max.shape) == [1, 2]
    torch.testing.assert_close(block_digest.digest_min, expected_digest.digest_min)
    torch.testing.assert_close(block_digest.digest_max, expected_digest.digest_max)


def test_sidecar_counter_backend_creates_digest_at_decode_boundary():
    leader_layer_name = "model.layers.0.self_attn.attn"
    follower_layer_name = "model.layers.1.self_attn.attn"
    sidecar = RecoverySidecar(
        config=MPRConfig(
            enabled=True,
            observe_backend="counter",
            digest_kind=RAW_MINMAX_DIGEST_KIND,
        )
    )
    kv_cache = torch.zeros(2, 4, 4, 1, 2)
    kv_cache[0, 2] = torch.tensor(
        [
            [[1.0, 3.0]],
            [[5.0, 1.0]],
            [[3.0, 7.0]],
            [[9.0, 5.0]],
        ]
    )
    prefill_metadata = SimpleNamespace(
        num_actual_tokens=3,
        max_query_len=3,
        block_table=torch.tensor([[2, 3]]),
    )
    decode_metadata = SimpleNamespace(
        num_actual_tokens=1,
        max_query_len=1,
        block_table=torch.tensor([[2, 3]]),
    )

    use_counter, seq_lens = sidecar.prepare_counter_kv_write(
        layer_name=leader_layer_name,
        attn_metadata=prefill_metadata,
    )
    assert not use_counter
    assert seq_lens == [3]
    use_counter, seq_lens = sidecar.prepare_counter_kv_write(
        layer_name=leader_layer_name,
        attn_metadata=decode_metadata,
    )
    assert use_counter
    assert seq_lens == [4]
    created = sidecar.observe_kv_write_by_counter(
        layer_name=follower_layer_name,
        kv_cache=kv_cache,
        attn_metadata=decode_metadata,
        block_size=4,
        seq_lens=seq_lens,
    )

    assert created == [2]
    block_digest = sidecar._digest_cache[follower_layer_name][2]
    torch.testing.assert_close(block_digest.digest_min, kv_cache[0, 2].amin(dim=0))
    torch.testing.assert_close(block_digest.digest_max, kv_cache[0, 2].amax(dim=0))


def test_sidecar_counter_backend_skips_non_boundary_decode_step():
    sidecar = RecoverySidecar(config=MPRConfig(enabled=True, observe_backend="counter"))
    kv_cache = torch.zeros(2, 4, 4, 1, 2)
    prefill_metadata = SimpleNamespace(
        num_actual_tokens=4,
        max_query_len=4,
        block_table=torch.tensor([[2, 3]]),
    )
    decode_metadata = SimpleNamespace(
        num_actual_tokens=1,
        max_query_len=1,
        block_table=torch.tensor([[2, 3]]),
    )

    sidecar.prepare_counter_kv_write(
        layer_name="model.layers.0.self_attn.attn",
        attn_metadata=prefill_metadata,
    )
    use_counter, seq_lens = sidecar.prepare_counter_kv_write(
        layer_name="model.layers.0.self_attn.attn",
        attn_metadata=decode_metadata,
    )
    assert use_counter
    assert seq_lens == [5]

    created = sidecar.observe_kv_write_by_counter(
        layer_name="model.layers.0.self_attn.attn",
        kv_cache=kv_cache,
        attn_metadata=decode_metadata,
        block_size=4,
        seq_lens=seq_lens,
    )

    assert created == []
    assert sidecar._digest_cache == {}


def test_sidecar_counter_backend_rejects_mixed_batch():
    sidecar = RecoverySidecar(config=MPRConfig(enabled=True, observe_backend="counter"))
    attn_metadata = SimpleNamespace(
        num_actual_tokens=4,
        max_query_len=3,
        block_table=torch.tensor([[2, 3], [4, 5]]),
    )

    use_counter, seq_lens = sidecar.prepare_counter_kv_write(
        layer_name="model.layers.0.self_attn.attn",
        attn_metadata=attn_metadata,
    )
    assert not use_counter
    assert seq_lens is None


def test_sidecar_can_create_raw_minmax_digest():
    sidecar = RecoverySidecar(
        config=MPRConfig(enabled=True, digest_kind=RAW_MINMAX_DIGEST_KIND)
    )
    kv_cache = torch.zeros(2, 1, 3, 1, 2)
    kv_cache[0, 0] = torch.tensor(
        [
            [[1.0, 3.0]],
            [[5.0, 1.0]],
            [[3.0, 7.0]],
        ]
    )

    sidecar.observe_kv_write(
        layer_name="model.layers.0.self_attn.attn",
        kv_cache=kv_cache,
        slot_mapping=torch.tensor([0, 1, 2]),
        block_size=3,
    )

    block_digest = sidecar._digest_cache["model.layers.0.self_attn.attn"][0]
    assert block_digest.digest_kind == RAW_MINMAX_DIGEST_KIND
    torch.testing.assert_close(block_digest.digest_min, kv_cache[0, 0].amin(dim=0))
    torch.testing.assert_close(block_digest.digest_max, kv_cache[0, 0].amax(dim=0))
