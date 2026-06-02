# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mixed-Precision Recovery sidecar scaffolding for vLLM v1."""

from vllm.v1.mixed_precision_recovery.config import MPRConfig
from vllm.v1.mixed_precision_recovery.cpu_backup import (
    CPUBackupKey,
    CPUBackupPutResult,
    CPUBackupReleaseResult,
    CPUBackupStats,
    SemanticCPUBackupStore,
)
from vllm.v1.mixed_precision_recovery.digest import (
    ARKVALE_DIGEST_KIND,
    KeyBlockDigest,
    RAW_MINMAX_DIGEST_KIND,
    summarize_key_block,
)
from vllm.v1.mixed_precision_recovery.quest_packing import (
    PackedQuestDigestCache,
    QuestMetadataStore,
    pack_quest_metadata_cache,
)
from vllm.v1.mixed_precision_recovery.recovery import (
    BlockRecoveryManager,
    RecoveryResult,
    select_recovery_block_ids,
)
from vllm.v1.mixed_precision_recovery.scoring import (
    DigestScoreResult,
    QuestCudaScorer,
    TorchQuestScorer,
    aggregate_query_head_scores,
    estimate_digest_score_result,
    estimate_digest_scores,
    estimate_query_head_digest_scores,
    get_digest_scoring_backend,
)
from vllm.v1.mixed_precision_recovery.sidecar import (
    RecoverySidecar,
    get_mpr_sidecar,
    reset_mpr_sidecar,
)

__all__ = [
    "MPRConfig",
    "CPUBackupKey",
    "CPUBackupPutResult",
    "CPUBackupReleaseResult",
    "CPUBackupStats",
    "SemanticCPUBackupStore",
    "ARKVALE_DIGEST_KIND",
    "KeyBlockDigest",
    "RAW_MINMAX_DIGEST_KIND",
    "PackedQuestDigestCache",
    "QuestMetadataStore",
    "BlockRecoveryManager",
    "RecoveryResult",
    "RecoverySidecar",
    "DigestScoreResult",
    "QuestCudaScorer",
    "TorchQuestScorer",
    "aggregate_query_head_scores",
    "estimate_digest_score_result",
    "estimate_digest_scores",
    "estimate_query_head_digest_scores",
    "get_digest_scoring_backend",
    "get_mpr_sidecar",
    "pack_quest_metadata_cache",
    "reset_mpr_sidecar",
    "select_recovery_block_ids",
    "summarize_key_block",
]
