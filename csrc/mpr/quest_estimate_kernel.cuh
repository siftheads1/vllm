/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// MPR estimate-only subset derived from Quest/FlashInfer decode_attn.cuh.
// This file keeps only the max-possible digest scoring path used by the
// Quest-style MPR backend.

#pragma once

#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

#include "flashinfer/cp_async.cuh"
#include "flashinfer/layout.cuh"
#include "flashinfer/math.cuh"
#include "flashinfer/utils.cuh"
#include "flashinfer/vec_dtypes.cuh"
#include "mpr/quest_paged_kv.cuh"

namespace vllm_mpr {

namespace cg = cooperative_groups;
using flashinfer::ceil_div;
using flashinfer::cp_async::PrefetchMode;
using flashinfer::cp_async::SharedMemFillMode;
using flashinfer::vec_t;

template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size,
          typename DTypeIn, typename DTypeOut>
__device__ __forceinline__ void compute_max_possible(
    const DTypeIn* max_smem, const DTypeIn* min_smem,
    const vec_t<float, vec_size>& q_vec, uint32_t kv_idx_base,
    uint32_t iter_base, uint32_t iter_bound, DTypeOut* out) {
  const uint32_t tx = threadIdx.x;
  const uint32_t tz = threadIdx.z;

#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> max_vec;
    vec_t<float, vec_size> min_vec;
    max_vec.cast_load(max_smem + (j * bdx + tx) * vec_size);
    min_vec.cast_load(min_smem + (j * bdx + tx) * vec_size);

    float max_possible = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      const float max_term = q_vec[i] * max_vec[i];
      const float min_term = q_vec[i] * min_vec[i];
      max_possible += max_term > min_term ? max_term : min_term;
    }
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      max_possible += flashinfer::math::shfl_xor_sync(max_possible, offset);
    }

    if (iter_base + tz * tile_size + j < iter_bound && tx == 0) {
      out[kv_idx_base + tz * tile_size + j] =
          static_cast<DTypeOut>(max_possible);
    }
  }
}

template <uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          PageStorage page_storage, flashinfer::QKVLayout kv_layout,
          typename DTypeIn, typename DTypeOut, typename IdType>
__global__ void MaxPossibleSampleWithPagedKVCacheKernel(
    DTypeIn* __restrict__ q,
    paged_digest_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
    DTypeOut* __restrict__ out) {
  auto block = cg::this_thread_block();

  constexpr uint32_t head_dim = bdx * vec_size;
  const uint32_t batch_idx = blockIdx.x;
  const uint32_t kv_head_idx = blockIdx.y;
  const uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;
  const uint32_t num_qo_heads = gridDim.y * bdy;
  const uint32_t cur_chunk_start = 0U;

  const uint32_t cur_page_indptr_begin = paged_kv.indptr[batch_idx];
  const uint32_t cur_page_indptr_end = paged_kv.indptr[batch_idx + 1];

  // Quest estimate always drops the final logical metadata entry. MPR's
  // Python packer currently appends a guard entry so real score candidates are
  // not lost by this Quest-specific convention.
  const uint32_t cur_last_page_len =
      (batch_idx == gridDim.x - 1) ? (paged_kv.last_page_len - 1)
                                   : paged_kv.page_size;
  const uint32_t kv_chunk_len =
      cur_page_indptr_begin != cur_page_indptr_end
          ? (cur_page_indptr_end - cur_page_indptr_begin - 1) *
                    paged_kv.page_size +
                cur_last_page_len
          : 0;

  extern __shared__ uint8_t smem[];
  DTypeIn* k_smem = reinterpret_cast<DTypeIn*>(smem);
  DTypeIn* v_smem = reinterpret_cast<DTypeIn*>(
      smem + num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                 sizeof(DTypeIn));
  DTypeIn** k_ptrs_smem = reinterpret_cast<DTypeIn**>(
      smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
                 head_dim * sizeof(DTypeIn));

  const uint32_t tx = threadIdx.x;
  const uint32_t ty = threadIdx.y;
  const uint32_t tz = threadIdx.z;

  vec_t<float, vec_size> q_vec;
  q_vec.cast_load(q + (batch_idx * num_qo_heads + qo_head_idx) * head_dim +
                  tx * vec_size);
  block.sync();

  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  static_assert(num_stages_smem <= bdx);
#pragma unroll
  for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    k_ptrs_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
        paged_kv.protective_get_k_ptr(
            cur_page_indptr_begin +
                (((j * bdz + tz) * bdy + ty) * bdx + tx) /
                    paged_kv.page_size,
            kv_head_idx,
            (((j * bdz + tz) * bdy + ty) * bdx + tx) % paged_kv.page_size, 0,
            last_indptr);
  }
  block.sync();

  DTypeIn* k_ptrs[tile_size_per_bdx];
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      k_ptrs[j] =
          k_ptrs_smem[((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j] +
          tx * vec_size;
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      flashinfer::cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch,
                                      SharedMemFillMode::kNoFill>(
          k_smem +
              (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) *
                  head_dim +
              tx * vec_size,
          k_ptrs[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j <
              kv_chunk_len);
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      DTypeIn* v_ptr = k_ptrs[j] + paged_kv.kv_offset_delta();
      flashinfer::cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch,
                                      SharedMemFillMode::kFillZero>(
          v_smem +
              (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) *
                  head_dim +
              tx * vec_size,
          v_ptr,
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j <
              kv_chunk_len);
    }
    flashinfer::cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

#pragma unroll 2
  for (uint32_t iter = 0;
       iter < ceil_div(kv_chunk_len, tile_size_per_bdx * bdy * bdz); ++iter) {
    if ((iter + num_stages_smem) % bdx == 0) {
#pragma unroll
      for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        k_ptrs_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
            paged_kv.protective_get_k_ptr(
                cur_page_indptr_begin +
                    ((iter + num_stages_smem) * tile_size_per_bdx * bdy *
                         bdz +
                     ((j * bdz + tz) * bdy + ty) * bdx + tx) /
                        paged_kv.page_size,
                kv_head_idx,
                ((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
                 ((j * bdz + tz) * bdy + ty) * bdx + tx) %
                    paged_kv.page_size,
                0, last_indptr);
      }
    }

    flashinfer::cp_async::wait_group<num_stages_smem - 1>();
    block.sync();
    compute_max_possible<vec_size, bdx, bdy * tile_size_per_bdx>(
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim,
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim,
        q_vec, cur_chunk_start + iter * tile_size_per_bdx * bdy * bdz,
        iter * tile_size_per_bdx * bdy * bdz, kv_chunk_len,
        out + (num_qo_heads * batch_idx + qo_head_idx) * kv_chunk_len);
    block.sync();

#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      k_ptrs[j] =
          k_ptrs_smem[(((iter + num_stages_smem) % bdx * bdz + tz) * bdy +
                       ty) *
                          tile_size_per_bdx +
                      j] +
          tx * vec_size;
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      flashinfer::cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch,
                                      SharedMemFillMode::kNoFill>(
          k_smem +
              (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) *
                  head_dim +
              tx * vec_size,
          k_ptrs[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) *
                      tile_size_per_bdx +
                  j <
              kv_chunk_len);
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      DTypeIn* v_ptr = k_ptrs[j] + paged_kv.kv_offset_delta();
      flashinfer::cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch,
                                      SharedMemFillMode::kFillZero>(
          v_smem +
              (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) *
                  head_dim +
              tx * vec_size,
          v_ptr,
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) *
                      tile_size_per_bdx +
                  j <
              kv_chunk_len);
    }
    flashinfer::cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  flashinfer::cp_async::wait_group<0>();
}

template <PageStorage page_storage, flashinfer::QKVLayout kv_layout,
          typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t MaxPossibleSampleWithPagedKVCache(
    DTypeIn* q, paged_digest_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
    DTypeOut* out, uint32_t num_qo_heads, cudaStream_t stream = nullptr) {
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  const uint32_t batch_size = paged_kv.batch_size;
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads
            << " is not a multiple of num_kv_heads " << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }

  SWITCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    SWITCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr uint32_t vec_size =
          std::max(16UL / sizeof(DTypeIn), HEAD_DIM / 32UL);
      constexpr uint32_t num_stages_smem = 2U;
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      static_assert(bdx <= 32);
      constexpr uint32_t bdy = GROUP_SIZE;
      constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
      constexpr uint32_t bdz = num_threads / (bdx * bdy);
      constexpr uint32_t tile_size_per_bdx =
          GROUP_SIZE == 1 ? (sizeof(DTypeIn) == 1 ? 2U : 4U) : 1U;
      const uint32_t smem_size =
          2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
              sizeof(DTypeIn) +
          tile_size_per_bdx * num_threads * sizeof(DTypeIn*);
      dim3 nblks(batch_size, num_kv_heads);
      dim3 nthrs(bdx, bdy, bdz);
      auto kernel =
          MaxPossibleSampleWithPagedKVCacheKernel<num_stages_smem,
                                                  tile_size_per_bdx, vec_size,
                                                  bdx, bdy, bdz, page_storage,
                                                  kv_layout, DTypeIn, DTypeOut,
                                                  IdType>;
      FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      void* args[] = {reinterpret_cast<void*>(&q),
                      reinterpret_cast<void*>(&paged_kv),
                      reinterpret_cast<void*>(&out)};
      FLASHINFER_CUDA_CALL(
          cudaLaunchKernel(reinterpret_cast<void*>(kernel), nblks, nthrs, args,
                           smem_size, stream));
    })
  });
  return cudaSuccess;
}

}  // namespace vllm_mpr
