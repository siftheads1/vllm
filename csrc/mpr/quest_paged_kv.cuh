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

// MPR estimate-only subset derived from Quest/FlashInfer paged_kv_t. This
// accessor treats digest max/min metadata as the K/V planes expected by the
// Quest estimate kernel; it does not own or mutate the real vLLM KV cache.

#pragma once

#include <cuda_runtime.h>

#include "flashinfer/layout.cuh"

namespace vllm_mpr {

enum class PageStorage {
  kIndices = 0U,
};

template <PageStorage page_storage, flashinfer::QKVLayout layout, typename DType,
          typename IdType>
struct paged_digest_t {
  uint32_t num_heads;
  uint32_t page_size;
  uint32_t head_dim;
  uint32_t batch_size;
  uint32_t page_budget;
  uint32_t last_page_len;
  IdType last_page_idx;

  // Internal layout:
  // [num_pages, 2, num_heads, page_size, head_dim] if layout == HND
  // [num_pages, 2, page_size, num_heads, head_dim] if layout == NHD
  //
  // Plane 0 stores digest maxima and plane 1 stores digest minima.
  DType* data;
  IdType* indices;
  IdType* indptr;

  __host__ __device__ __forceinline__ paged_digest_t()
      : num_heads(0),
        page_size(0),
        head_dim(0),
        batch_size(0),
        page_budget(0),
        last_page_len(0),
        last_page_idx(0),
        data(nullptr),
        indices(nullptr),
        indptr(nullptr) {}

  __host__ __device__ __forceinline__ paged_digest_t(
      uint32_t num_heads, uint32_t page_size, uint32_t head_dim,
      uint32_t batch_size, uint32_t page_budget, uint32_t last_page_len,
      IdType last_page_idx, DType* data, IdType* indices, IdType* indptr)
      : num_heads(num_heads),
        page_size(page_size),
        head_dim(head_dim),
        batch_size(batch_size),
        page_budget(page_budget),
        last_page_len(last_page_len),
        last_page_idx(last_page_idx),
        data(data),
        indices(indices),
        indptr(indptr) {}

  __host__ __device__ __forceinline__ size_t get_k_elem_offset(
      size_t page_idx, size_t head_idx, size_t entry_idx,
      size_t feat_idx) const {
    return layout == flashinfer::QKVLayout::kHND
               ? ((page_idx * 2 * num_heads + head_idx) * page_size +
                  entry_idx) *
                         head_dim +
                     feat_idx
               : ((page_idx * 2 * page_size + entry_idx) * num_heads +
                  head_idx) *
                         head_dim +
                     feat_idx;
  }

  __host__ __device__ __forceinline__ size_t get_k_elem_offset_in_page(
      size_t head_idx, size_t entry_idx, size_t feat_idx) const {
    return layout == flashinfer::QKVLayout::kHND
               ? (head_idx * page_size + entry_idx) * head_dim + feat_idx
               : (entry_idx * num_heads + head_idx) * head_dim + feat_idx;
  }

  __host__ __device__ __forceinline__ size_t get_v_elem_offset(
      size_t page_idx, size_t head_idx, size_t entry_idx,
      size_t feat_idx) const {
    return layout == flashinfer::QKVLayout::kHND
               ? (((page_idx * 2 + 1) * num_heads + head_idx) * page_size +
                  entry_idx) *
                         head_dim +
                     feat_idx
               : (((page_idx * 2 + 1) * page_size + entry_idx) * num_heads +
                  head_idx) *
                         head_dim +
                     feat_idx;
  }

  __host__ __device__ __forceinline__ size_t get_v_elem_offset_in_page(
      size_t head_idx, size_t entry_idx, size_t feat_idx) const {
    return layout == flashinfer::QKVLayout::kHND
               ? ((num_heads + head_idx) * page_size + entry_idx) * head_dim +
                     feat_idx
               : ((page_size + entry_idx) * num_heads + head_idx) * head_dim +
                     feat_idx;
  }

  __host__ __device__ __forceinline__ uint32_t kv_offset_delta() const {
    return num_heads * page_size * head_dim;
  }

  __device__ __forceinline__ DType* protective_get_k_ptr(
      IdType page_iter, uint32_t head_idx, uint32_t entry_idx,
      uint32_t feat_idx, IdType last_indptr) const {
    static_assert(page_storage == PageStorage::kIndices);
    if (page_iter < last_indptr) {
      return data + get_k_elem_offset(__ldg(indices + page_iter), head_idx,
                                      entry_idx, feat_idx);
    }
    return data;
  }
};

}  // namespace vllm_mpr
