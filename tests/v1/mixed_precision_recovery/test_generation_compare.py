# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from scripts.mpr_compare_generation_outputs import (
    compare_generated_token_ids,
    extract_generated_token_ids,
)


def test_extract_generated_token_ids(tmp_path):
    path = tmp_path / "smoke.log"
    path.write_text(
        "\n".join(
            [
                "baseline_model: Qwen/Qwen3-8B",
                "generated_token_ids: [101, 202, 303]",
                "generated_token_count: 3",
            ]
        ),
        encoding="utf-8",
    )

    assert extract_generated_token_ids(path) == [101, 202, 303]


def test_compare_generated_token_ids_rejects_mismatch(tmp_path):
    baseline_path = tmp_path / "baseline.log"
    mpr_path = tmp_path / "mpr.log"
    baseline_path.write_text("generated_token_ids: [1, 2, 3]\n", encoding="utf-8")
    mpr_path.write_text("generated_token_ids: [1, 9, 3]\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="first_mismatch_index=1"):
        compare_generated_token_ids(baseline_path, mpr_path)
