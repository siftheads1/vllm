// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cstdint>
#include <string>

#include "flashinfer/layout.cuh"
#include "mpr/quest_estimate_kernel.cuh"

namespace {

constexpr int64_t kQuestNHDLayout = 0;
constexpr int64_t kQuestHNDLayout = 1;

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

bool is_supported_group_size(int64_t group_size) {
  return group_size == 1 || group_size == 4 || group_size == 8;
}

bool is_supported_head_dim(int64_t head_dim) {
  return head_dim == 64 || head_dim == 128 || head_dim == 256;
}

}  // namespace

void mpr_estimate_attn_score(
    torch::Tensor& q, torch::Tensor& out, torch::Tensor& metadata_data,
    torch::Tensor& metadata_indices, torch::Tensor& metadata_indptr,
    int64_t metadata_last_page_len, int64_t metadata_last_page_idx,
    int64_t layout) {
  check_cuda_contiguous(q, "q");
  check_cuda_contiguous(out, "out");
  check_cuda_contiguous(metadata_data, "metadata_data");
  check_cuda_contiguous(metadata_indices, "metadata_indices");
  check_cuda_contiguous(metadata_indptr, "metadata_indptr");

  TORCH_CHECK(layout == kQuestNHDLayout,
              "MPR Quest estimate currently supports only NHD layout ",
              kQuestNHDLayout, "; got layout=", layout, ". HND is ",
              kQuestHNDLayout, ".");

  TORCH_CHECK(q.dim() == 3, "q must be shaped [1, num_q_heads, head_dim].");
  TORCH_CHECK(q.size(0) == 1,
              "MPR Quest estimate currently supports batch size 1, got ",
              q.size(0), ".");
  TORCH_CHECK(metadata_data.dim() == 5,
              "metadata_data must be shaped "
              "[num_pages, 2, page_size, num_kv_heads, head_dim].");
  TORCH_CHECK(metadata_indices.dim() == 1,
              "metadata_indices must be a 1D int32 tensor.");
  TORCH_CHECK(metadata_indptr.dim() == 1,
              "metadata_indptr must be a 1D int32 tensor.");
  TORCH_CHECK(metadata_indptr.numel() == 2,
              "MPR Quest estimate PoC currently supports a single request "
              "metadata_indptr shaped [2], got numel=",
              metadata_indptr.numel(), ".");
  TORCH_CHECK(out.dim() == 2,
              "out must be shaped [num_q_heads, num_score_entries].");

  TORCH_CHECK(q.scalar_type() == torch::kFloat16,
              "MPR Quest estimate currently supports fp16 q tensors only, got ",
              q.scalar_type(), ".");
  TORCH_CHECK(out.scalar_type() == q.scalar_type(),
              "out dtype must match q dtype.");
  TORCH_CHECK(metadata_data.scalar_type() == q.scalar_type(),
              "metadata_data dtype must match q dtype.");
  TORCH_CHECK(metadata_indices.scalar_type() == torch::kInt32,
              "metadata_indices must be int32.");
  TORCH_CHECK(metadata_indptr.scalar_type() == torch::kInt32,
              "metadata_indptr must be int32.");

  const int64_t num_metadata_pages = metadata_data.size(0);
  const int64_t page_size = metadata_data.size(2);
  const int64_t num_kv_heads = metadata_data.size(3);
  const int64_t head_dim = metadata_data.size(4);
  const int64_t num_q_heads = q.size(1);
  TORCH_CHECK(metadata_data.size(1) == 2,
              "metadata_data.size(1) must be 2 for digest max/min planes.");
  TORCH_CHECK(q.size(2) == head_dim, "q head_dim ", q.size(2),
              " must match metadata_data head_dim ", head_dim, ".");
  TORCH_CHECK(out.size(0) == num_q_heads, "out.size(0) ", out.size(0),
              " must match num_q_heads ", num_q_heads, ".");
  TORCH_CHECK(num_metadata_pages > 0,
              "metadata_data must contain at least one metadata page.");
  TORCH_CHECK(metadata_indices.numel() > 0,
              "metadata_indices must contain at least one page id.");
  TORCH_CHECK(page_size > 0, "metadata page_size must be positive.");
  TORCH_CHECK(num_kv_heads > 0, "num_kv_heads must be positive.");
  TORCH_CHECK(num_q_heads % num_kv_heads == 0,
              "num_q_heads must be a multiple of num_kv_heads, got ",
              num_q_heads, " and ", num_kv_heads, ".");

  const int64_t group_size = num_q_heads / num_kv_heads;
  TORCH_CHECK(is_supported_group_size(group_size),
              "MPR Quest estimate supports GQA group sizes {1, 4, 8}, got ",
              group_size, ".");
  TORCH_CHECK(is_supported_head_dim(head_dim),
              "MPR Quest estimate supports head_dim {64, 128, 256}, got ",
              head_dim, ".");
  TORCH_CHECK(metadata_last_page_len >= 1 &&
                  metadata_last_page_len <= page_size,
              "metadata_last_page_len must be in [1, page_size], got ",
              metadata_last_page_len, " with page_size=", page_size, ".");
  TORCH_CHECK(metadata_last_page_idx >= 0 &&
                  metadata_last_page_idx < num_metadata_pages,
              "metadata_last_page_idx must be a valid metadata_data page id, "
              "got ",
              metadata_last_page_idx, " with num_metadata_pages=",
              num_metadata_pages, ".");

  const int64_t expected_score_entries =
      (metadata_indices.numel() - 1) * page_size + metadata_last_page_len - 1;
  TORCH_CHECK(out.size(1) == expected_score_entries,
              "out.size(1) must match Quest estimate score-entry count, got ",
              out.size(1), " but expected ", expected_score_entries, ".");

  using c_type = half;
  vllm_mpr::paged_digest_t<vllm_mpr::PageStorage::kIndices,
                           flashinfer::QKVLayout::kNHD, c_type, int32_t>
      paged_kv(static_cast<uint32_t>(num_kv_heads),
               static_cast<uint32_t>(page_size),
               static_cast<uint32_t>(head_dim),
               /*batch_size=*/1,
               /*page_budget=*/0,
               static_cast<uint32_t>(metadata_last_page_len),
               static_cast<int32_t>(metadata_last_page_idx),
               reinterpret_cast<c_type*>(metadata_data.data_ptr()),
               reinterpret_cast<int32_t*>(metadata_indices.data_ptr()),
               reinterpret_cast<int32_t*>(metadata_indptr.data_ptr()));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.device().index());
  cudaError_t status =
      vllm_mpr::MaxPossibleSampleWithPagedKVCache<
          vllm_mpr::PageStorage::kIndices, flashinfer::QKVLayout::kNHD, c_type,
          c_type, int32_t>(reinterpret_cast<c_type*>(q.data_ptr()), paged_kv,
                           reinterpret_cast<c_type*>(out.data_ptr()),
                           static_cast<uint32_t>(num_q_heads), stream);
  TORCH_CHECK(status == cudaSuccess,
              "MPR Quest estimate kernel failed with error: ",
              cudaGetErrorString(status));
}
