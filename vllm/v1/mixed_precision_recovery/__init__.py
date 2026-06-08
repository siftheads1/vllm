# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mixed-Precision Recovery sidecar scaffolding for vLLM v1."""

from vllm.v1.mixed_precision_recovery.config import MPRConfig
from vllm.v1.mixed_precision_recovery.backup_codec import (
    BackupCodec,
    FP16_BACKUP_FORMAT,
    FP16BackupCodec,
    FP16BackupPayload,
    INT8_BACKUP_FORMAT,
    INT8BackupCodec,
    INT8BackupPayload,
    INT4_BACKUP_FORMAT,
    INT4BackupCodec,
    INT4BackupPayload,
    PER_TOKEN_PER_KV_HEAD_SCALE,
)
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
from vllm.v1.mixed_precision_recovery.precision_policy import (
    PrecisionPolicy,
    PrecisionTier,
    ThresholdPrecisionPolicy,
    TierAssignment,
    TopRatioPrecisionPolicy,
)
from vllm.v1.mixed_precision_recovery.recovery_payload import (
    EagerRecoveryPayloadProvider,
    FP16RecoveryPayloadEntry,
    INT8RecoveryPayloadEntry,
    RecoveryPayloadProvider,
    TieredRecoveryPayloads,
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
    "BackupCodec",
    "FP16_BACKUP_FORMAT",
    "FP16BackupCodec",
    "FP16BackupPayload",
    "INT8_BACKUP_FORMAT",
    "INT8BackupCodec",
    "INT8BackupPayload",
    "INT4_BACKUP_FORMAT",
    "INT4BackupCodec",
    "INT4BackupPayload",
    "PER_TOKEN_PER_KV_HEAD_SCALE",
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
    "PrecisionPolicy",
    "PrecisionTier",
    "ThresholdPrecisionPolicy",
    "TierAssignment",
    "TopRatioPrecisionPolicy",
    "EagerRecoveryPayloadProvider",
    "FP16RecoveryPayloadEntry",
    "INT8RecoveryPayloadEntry",
    "RecoveryPayloadProvider",
    "TieredRecoveryPayloads",
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
