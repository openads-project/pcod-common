// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> pillar_stats_cuda(
    torch::Tensor points_mask,
    torch::Tensor points_xyz,
    double x_min,
    double y_min,
    double z_min,
    double x_max,
    double y_max,
    double z_max,
    double voxel_x,
    double voxel_y,
    int64_t grid_x,
    int64_t grid_y);

std::vector<torch::Tensor> pillar_preprocess_cuda(
    torch::Tensor points_mask,
    torch::Tensor points_xyz,
    torch::Tensor points_feature,
    double x_min,
    double y_min,
    double z_min,
    double x_max,
    double y_max,
    double z_max,
    double voxel_x,
    double voxel_y,
    int64_t grid_x,
    int64_t grid_y);

std::vector<torch::Tensor> pillar_stats(
    torch::Tensor points_mask,
    torch::Tensor points_xyz,
    double x_min,
    double y_min,
    double z_min,
    double x_max,
    double y_max,
    double z_max,
    double voxel_x,
    double voxel_y,
    int64_t grid_x,
    int64_t grid_y) {
  if (points_mask.is_cuda()) {
    return pillar_stats_cuda(
        points_mask,
        points_xyz,
        x_min,
        y_min,
        z_min,
        x_max,
        y_max,
        z_max,
        voxel_x,
        voxel_y,
        grid_x,
        grid_y);
  }
  TORCH_CHECK(false, "pillar_stats is implemented for CUDA inputs only");
}

std::vector<torch::Tensor> pillar_preprocess(
    torch::Tensor points_mask,
    torch::Tensor points_xyz,
    torch::Tensor points_feature,
    double x_min,
    double y_min,
    double z_min,
    double x_max,
    double y_max,
    double z_max,
    double voxel_x,
    double voxel_y,
    int64_t grid_x,
    int64_t grid_y) {
  if (points_mask.is_cuda()) {
    return pillar_preprocess_cuda(
        points_mask,
        points_xyz,
        points_feature,
        x_min,
        y_min,
        z_min,
        x_max,
        y_max,
        z_max,
        voxel_x,
        voxel_y,
        grid_x,
        grid_y);
  }
  TORCH_CHECK(false, "pillar_preprocess is implemented for CUDA inputs only");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "pillar_stats",
      &pillar_stats,
      "Compute per-pillar counts/sums and pillar_ids (CUDA)");
  m.def(
      "pillar_preprocess",
      &pillar_preprocess,
      "Compute pillar ids, counts, sums, and per-point features (CUDA)");
}
