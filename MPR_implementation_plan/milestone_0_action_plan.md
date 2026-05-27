# Milestone 0 Action Plan

Milestone 0의 목표는 코드를 수정하기 전에 vLLM 기반 mixed-precision recovery를 어디에, 어떤 metadata contract로 붙일지 확정하는 것이다.

이 단계에서는 구현하지 않는다. 대신 다음 네 가지를 명확히 한다.

1. Decode attention에서 필요한 tensor와 metadata가 어디서 만나는가
2. KV block lifecycle은 누가 관리하는가
3. CPU offload/reload는 어떤 metadata와 copy 단위로 동작하는가
4. ArkVale digest scoring을 vLLM block table에 맞추려면 어떤 변환이 필요한가

## Scope

초기 scope는 의도적으로 좁게 잡는다.

```text
vLLM v1
single GPU
single request first
decode path first
one attention backend only
no preemption first
score-only first
```

Preemption, multi-request batching, mixed backend support, direct mixed-precision attention kernel은 후속 단계로 미룬다.

## Step 0.1: Target vLLM Path 고정

먼저 vLLM의 어떤 execution path를 대상으로 할지 정한다.

확인할 파일:

| File | 확인할 것 |
|---|---|
| `vllm/vllm/v1/worker/gpu/model_runner.py` | model forward entry, input batch, attention metadata 생성 위치 |
| `vllm/vllm/v1/worker/gpu/input_batch.py` | request/block table/seq len metadata 보관 방식 |
| `vllm/vllm/v1/attention/selector.py` | 어떤 attention backend가 선택되는지 |
| `vllm/vllm/v1/attention/backend.py` | common attention metadata interface |

산출물:

```text
target_backend = ...
decode_forward_entry = ...
attention_metadata_type = ...
block_table_source = ...
seq_lens_source = ...
```

## Step 0.2: Decode Attention Tensor Flow 추적

Decode step에서 다음 값들이 어디서 생성되고 attention backend로 들어가는지 추적한다.

필수 추적 대상:

```text
query
key
value
kv_cache tensor
block_table
seq_lens
slot_mapping or token/block mapping
request id / sequence id
layer id
```

확인할 질문:

1. Query는 어느 shape으로 attention backend에 들어가는가?
2. Key/value는 KV cache에 write되기 전 어느 위치에서 관찰 가능한가?
3. Block table은 logical block id를 physical block id로 어떻게 매핑하는가?
4. Decode에서 현재 token의 KV가 어떤 block/offset에 들어가는가?
5. Layer id는 attention module에서 안정적으로 접근 가능한가?

산출물:

```text
query_hook_candidate = ...
kv_write_hook_candidate = ...
block_table_hook_candidate = ...
slot_mapping_source = ...
layer_id_source = ...
```

## Step 0.3: KV Block Lifecycle 추적

Sidecar digest/backup은 vLLM block lifecycle과 반드시 맞아야 한다.

확인할 파일:

| File | 확인할 것 |
|---|---|
| `vllm/vllm/v1/core/kv_cache_manager.py` | block allocation/free path |
| `vllm/vllm/v1/core/block_pool.py` | physical block pool 구조 |
| `vllm/vllm/v1/core/sched/scheduler.py` | scheduling, preemption, request finish 처리 |
| `vllm/vllm/v1/worker/gpu/block_table.py` | worker-side block table representation |

확인할 질문:

1. Physical block id는 언제 할당되는가?
2. Block이 free되는 정확한 hook은 어디인가?
3. Request 종료 시 block cleanup은 어디서 일어나는가?
4. Preemption이 발생하면 block id와 request mapping은 어떻게 변하는가?
5. Scheduler-side block id와 worker-side block id가 동일한가?
6. Layer별 KV block이 같은 physical block id namespace를 공유하는가?

초기 scope에서는 preemption을 제외하되, 나중에 sidecar cleanup을 위해 free path는 반드시 기록한다.

산출물:

```text
block_alloc_hook = ...
block_free_hook = ...
request_finish_hook = ...
physical_block_id_semantics = ...
preemption_ignored_for_v0 = yes/no
```

## Step 0.4: Simple KV Offload 경로 분석

CPU backup/recovery layout은 Milestone 2부터 필요하지만, Milestone 0에서 후보를 조사한다.

확인할 파일:

| File | 확인할 것 |
|---|---|
| `vllm/vllm/v1/simple_kv_offload/worker.py` | GPU KV tensor를 CPU block tensor로 보는 방식 |
| `vllm/vllm/v1/simple_kv_offload/metadata.py` | load/store metadata 구조 |
| `vllm/vllm/v1/simple_kv_offload/copy_backend.py` | block copy primitive |
| `vllm/vllm/v1/simple_kv_offload/manager.py` | scheduler-side load/store decision |

확인할 질문:

1. CPU offload는 physical block 단위인가?
2. CPU block id와 GPU block id mapping은 누가 관리하는가?
3. Copy는 raw byte view인가, typed tensor view인가?
4. Layer별 KV cache가 같은 CPU block id를 공유하는가?
5. 이 경로를 CPU fp16 backup으로 재사용할 수 있는가?
6. Reuse한다면 backup layout은 vLLM raw block layout을 그대로 따르는가?

산출물:

```text
offload_copy_unit = ...
cpu_block_layout_candidate = ...
can_reuse_simple_offload = yes/no/partial
required_changes_for_backup = ...
```

## Step 0.5: ArkVale Digest Scoring Input Contract 정리

ArkVale kernel을 바로 쓰려면 vLLM metadata를 ArkVale의 paged KV input contract로 바꿀 수 있어야 한다.

확인할 파일:

| File | 확인할 것 |
|---|---|
| `source/arkvale/kernels.py` | `estimate_scores` Python signature |
| `source/arkvale_cpp/src/estimate.cu` | kernel이 `indices`, `indptr`, `last_page_len`, `seq_len`을 어떻게 해석하는지 |
| `source/arkvale/infer_state.py` | digest cache allocation, `dg_indptrs`, `dg_last_page_lens` 생성 방식 |
| `source/arkvale/kv_cache.py` | `c2p`, `cc2gp`, `gc2cc` semantics |

확인할 질문:

1. `dg_indices`는 physical digest page id list인가?
2. `dg_indptr`는 packed 1D indices 기준인가, padded 2D flatten 기준으로도 가능한가?
3. `dg_last_page_len`은 마지막 digest page의 token 수인가, 유효 page 수인가?
4. Output score의 index는 original KV page id와 어떻게 대응되는가?
5. vLLM block size와 ArkVale page size를 같게 둬야 하는가?
6. ArkVale digest page 하나가 vLLM KV block 하나에 대응되는가?

산출물:

```text
arkvale_estimate_contract = ...
vllm_to_arkvale_metadata_mapping = ...
score_index_to_block_id_mapping = ...
kernel_reuse_feasibility = yes/no/after_adapter
```

## Step 0.6: Score-Only Sidecar API 확정

Milestone 1에서 구현할 최소 sidecar API를 정한다.

초기 후보:

```python
class RecoverySidecar:
    def observe_kv_write(
        self,
        layer_id: int,
        physical_block_ids,
        block_offsets,
        keys,
        values,
    ):
        ...

    def finalize_block_digest(
        self,
        layer_id: int,
        physical_block_id: int,
    ):
        ...

    def estimate_scores(
        self,
        layer_id: int,
        query,
        block_table,
        seq_lens,
    ):
        ...

    def release_blocks(
        self,
        physical_block_ids,
    ):
        ...
```

Milestone 0에서는 구현하지 않는다. 필요한 argument가 vLLM 어디서 나오는지만 확인한다.

산출물:

```text
sidecar_api_v0 = ...
required_hooks = ...
unsupported_cases = ...
```

## Step 0.7: Go / Pivot Decision

Milestone 0 끝에서 다음 중 하나를 결정한다.

| Decision | 조건 |
|---|---|
| Go with vLLM | query, block table, KV write hook, block lifecycle hook이 모두 명확함 |
| vLLM score-only only | score path는 가능하지만 recovery lifecycle이 아직 불명확함 |
| Pivot to InfiniGen prototype | vLLM hook이 너무 깊거나 attention backend 수정이 과도함 |

최종 산출물:

```text
vllm_integration_notes.md
target backend decision
hook list
metadata mapping diagram
Milestone 1 implementation plan
```

## Completion Criteria

Milestone 0은 다음이 문서화되면 완료로 본다.

1. Target attention backend가 정해져 있다.
2. Query, key/value, block table, seq_lens hook 후보가 정리돼 있다.
3. Physical block id lifecycle이 설명돼 있다.
4. Simple KV offload를 CPU backup에 재사용할 수 있는지 판단돼 있다.
5. ArkVale `estimate_scores` contract와 vLLM metadata mapping이 정리돼 있다.
6. Milestone 1 score-only prototype의 최소 구현 위치가 정해져 있다.

