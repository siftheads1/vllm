# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.mixed_precision_recovery.precision_policy import (
    ThresholdPrecisionPolicy,
    TierAssignment,
    TopRatioPrecisionPolicy,
)


def _scores(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def _assert_all_candidates_assigned_once(
    assignment: TierAssignment,
    physical_block_ids: list[int],
) -> None:
    assigned = assignment.all_block_ids
    assert len(assigned) == len(physical_block_ids)
    assert set(assigned) == set(physical_block_ids)
    assert len(set(assigned)) == len(assigned)


def test_top_ratio_policy_orders_by_score_and_preserves_physical_ids():
    assignment = TopRatioPrecisionPolicy(
        fp16_ratio=0.25,
        int8_ratio=0.50,
        int4_ratio=0.0,
    ).assign_tiers(
        block_scores=_scores([0.5, 7.0, 1.25, 3.0]),
        physical_block_ids=[10, 4, 7, 8],
    )

    assert assignment.fp16_block_ids == [4]
    assert assignment.int8_block_ids == [8, 7]
    assert assignment.int4_block_ids == []
    assert assignment.skipped_block_ids == [10]
    _assert_all_candidates_assigned_once(assignment, [10, 4, 7, 8])


def test_top_ratio_policy_assigns_int4_after_int8():
    assignment = TopRatioPrecisionPolicy(
        fp16_ratio=0.25,
        int8_ratio=0.25,
        int4_ratio=0.25,
    ).assign_tiers(
        block_scores=_scores([0.5, 7.0, 1.25, 3.0]),
        physical_block_ids=[10, 4, 7, 8],
    )

    assert assignment.fp16_block_ids == [4]
    assert assignment.int8_block_ids == [8]
    assert assignment.int4_block_ids == [7]
    assert assignment.skipped_block_ids == [10]
    _assert_all_candidates_assigned_once(assignment, [10, 4, 7, 8])


def test_top_ratio_policy_preserves_candidate_order_for_score_ties():
    assignment = TopRatioPrecisionPolicy(
        fp16_ratio=0.50,
        int8_ratio=0.25,
        int4_ratio=0.0,
    ).assign_tiers(
        block_scores=_scores([3.0, 3.0, 1.0, 1.0]),
        physical_block_ids=[10, 4, 7, 8],
    )

    assert assignment.fp16_block_ids == [10, 4]
    assert assignment.int8_block_ids == [7]
    assert assignment.int4_block_ids == []
    assert assignment.skipped_block_ids == [8]
    _assert_all_candidates_assigned_once(assignment, [10, 4, 7, 8])


def test_top_ratio_policy_uses_ceil_rounding_and_clamps_to_candidates():
    assignment = TopRatioPrecisionPolicy(
        fp16_ratio=0.25,
        int8_ratio=0.50,
        int4_ratio=0.25,
    ).assign_tiers(
        block_scores=_scores([7.0, 6.0, 5.0, 4.0]),
        physical_block_ids=[1, 2, 3, 4],
    )

    assert assignment.fp16_block_ids == [1]
    assert assignment.int8_block_ids == [2, 3]
    assert assignment.int4_block_ids == [4]
    assert assignment.skipped_block_ids == []
    _assert_all_candidates_assigned_once(assignment, [1, 2, 3, 4])


def test_top_ratio_policy_clamps_later_tiers_for_small_candidate_count():
    assignment = TopRatioPrecisionPolicy(
        fp16_ratio=0.34,
        int8_ratio=0.33,
        int4_ratio=0.33,
    ).assign_tiers(
        block_scores=_scores([7.0, 6.0]),
        physical_block_ids=[1, 2],
    )

    assert assignment.fp16_block_ids == [1]
    assert assignment.int8_block_ids == [2]
    assert assignment.int4_block_ids == []
    assert assignment.skipped_block_ids == []
    _assert_all_candidates_assigned_once(assignment, [1, 2])


def test_top_ratio_policy_clamp_preserves_skip_remainder():
    assignment = TopRatioPrecisionPolicy(
        fp16_ratio=0.25,
        int8_ratio=0.50,
        int4_ratio=0.0,
    ).assign_tiers(
        block_scores=_scores([7.0, 6.0, 5.0, 4.0]),
        physical_block_ids=[1, 2, 3, 4],
    )

    assert assignment.fp16_block_ids == [1]
    assert assignment.int8_block_ids == [2, 3]
    assert assignment.int4_block_ids == []
    assert assignment.skipped_block_ids == [4]
    _assert_all_candidates_assigned_once(assignment, [1, 2, 3, 4])


def test_top_ratio_policy_accepts_empty_candidates():
    assignment = TopRatioPrecisionPolicy(
        fp16_ratio=0.25,
        int8_ratio=0.50,
    ).assign_tiers(
        block_scores=torch.empty(0),
        physical_block_ids=[],
    )

    assert assignment.fp16_block_ids == []
    assert assignment.int8_block_ids == []
    assert assignment.int4_block_ids == []
    assert assignment.skipped_block_ids == []


def test_threshold_policy_uses_inclusive_boundaries():
    assignment = ThresholdPrecisionPolicy(
        high_threshold=5.0,
        mid_threshold=3.0,
        low_threshold=2.0,
    ).assign_tiers(
        block_scores=_scores([5.0, 3.0, 2.0, 1.0]),
        physical_block_ids=[10, 4, 7, 8],
    )

    assert assignment.fp16_block_ids == [10]
    assert assignment.int8_block_ids == [4]
    assert assignment.int4_block_ids == [7]
    assert assignment.skipped_block_ids == [8]
    _assert_all_candidates_assigned_once(assignment, [10, 4, 7, 8])


def test_threshold_policy_preserves_candidate_order():
    assignment = ThresholdPrecisionPolicy(
        high_threshold=5.0,
        mid_threshold=3.0,
        low_threshold=2.0,
    ).assign_tiers(
        block_scores=_scores([3.0, 7.0, 2.5, 6.0, 1.0]),
        physical_block_ids=[10, 4, 7, 8, 11],
    )

    assert assignment.fp16_block_ids == [4, 8]
    assert assignment.int8_block_ids == [10]
    assert assignment.int4_block_ids == [7]
    assert assignment.skipped_block_ids == [11]
    _assert_all_candidates_assigned_once(assignment, [10, 4, 7, 8, 11])


def test_precision_policy_rejects_mismatched_score_count():
    with pytest.raises(ValueError, match="physical_block_ids length"):
        TopRatioPrecisionPolicy(
            fp16_ratio=0.25,
            int8_ratio=0.50,
        ).assign_tiers(
            block_scores=_scores([1.0, 2.0]),
            physical_block_ids=[10],
        )


def test_precision_policy_rejects_non_1d_score_tensor():
    with pytest.raises(ValueError, match="block_scores shaped"):
        ThresholdPrecisionPolicy(
            high_threshold=5.0,
            mid_threshold=3.0,
            low_threshold=2.0,
        ).assign_tiers(
            block_scores=torch.ones(1, 2),
            physical_block_ids=[10, 4],
        )


def test_precision_policy_rejects_duplicate_physical_block_ids():
    with pytest.raises(ValueError, match="unique physical block ids"):
        TopRatioPrecisionPolicy(
            fp16_ratio=0.25,
            int8_ratio=0.50,
        ).assign_tiers(
            block_scores=_scores([1.0, 2.0]),
            physical_block_ids=[10, 10],
        )


def test_top_ratio_policy_rejects_invalid_ratio():
    with pytest.raises(ValueError, match="fp16_ratio"):
        TopRatioPrecisionPolicy(fp16_ratio=-0.1, int8_ratio=0.50)

    with pytest.raises(ValueError, match="int4_ratio"):
        TopRatioPrecisionPolicy(
            fp16_ratio=0.25,
            int8_ratio=0.50,
            int4_ratio=1.25,
        )


def test_top_ratio_policy_rejects_ratio_sum_above_one():
    with pytest.raises(ValueError, match="fp16_ratio \\+ int8_ratio"):
        TopRatioPrecisionPolicy(
            fp16_ratio=0.50,
            int8_ratio=0.40,
            int4_ratio=0.20,
        )


def test_threshold_policy_rejects_inverted_thresholds():
    with pytest.raises(ValueError, match="high_threshold"):
        ThresholdPrecisionPolicy(
            high_threshold=1.0,
            mid_threshold=2.0,
            low_threshold=0.0,
        )

    with pytest.raises(ValueError, match="mid_threshold"):
        ThresholdPrecisionPolicy(
            high_threshold=3.0,
            mid_threshold=1.0,
            low_threshold=2.0,
        )
