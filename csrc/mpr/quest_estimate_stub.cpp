#include <cstdint>
#include <stdexcept>

#include <torch/library.h>

void mpr_estimate_attn_score(
    torch::Tensor& q, torch::Tensor& out, torch::Tensor& metadata_data,
    torch::Tensor& metadata_indices, torch::Tensor& metadata_indptr,
    int64_t metadata_last_page_len, int64_t metadata_last_page_idx,
    int64_t layout) {
  (void)q;
  (void)out;
  (void)metadata_data;
  (void)metadata_indices;
  (void)metadata_indptr;
  (void)metadata_last_page_len;
  (void)metadata_last_page_idx;
  (void)layout;

  throw std::runtime_error(
      "mpr_estimate_attn_score is registered as an MPR Quest CUDA binding "
      "placeholder, but the Quest estimate kernel implementation has not been "
      "linked yet.");
}
