# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path


def test_cudagraph_capture_marks_forward_context_as_dummy():
    """Ensure MPR skips CUDA graph capture dummy KV writes."""
    repo_root = Path(__file__).parents[3]
    capture_source = (
        repo_root / "vllm" / "v1" / "worker" / "gpu" / "cudagraph_utils.py"
    ).read_text(encoding="utf-8")

    assert "prepare_dummy_inputs" in capture_source
    assert "prepare_inputs_to_capture" in capture_source
    assert "is_dummy_run=True" in capture_source
