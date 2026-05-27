# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Debug artifact writer for the MPR sidecar."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from vllm.v1.mixed_precision_recovery.config import MPRConfig


def _json_default(value: Any) -> str:
    return repr(value)


class MPRDebugWriter:
    """Rank-local JSONL writer.

    The writer is inert unless MPR is enabled and ``VLLM_MPR_DEBUG_DIR`` is set.
    """

    def __init__(self, config: MPRConfig) -> None:
        self.path: Path | None = None
        self._file = None

        if not config.enabled or config.debug_dir is None:
            return

        debug_dir = Path(config.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        rank = os.getenv("LOCAL_RANK") or os.getenv("RANK") or "0"
        self.path = debug_dir / f"mpr_rank{rank}_pid{os.getpid()}.jsonl"
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        if self._file is None:
            return
        json.dump(record, self._file, default=_json_default, sort_keys=True)
        self._file.write("\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
