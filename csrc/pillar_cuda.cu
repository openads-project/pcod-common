#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

__global__ void pillar_stats_kernel(
    const bool* points_mask,
    const float* points_xyz,
    int64_t* pillar_ids_out,
    int32_t* point_count,
    float* xyz_sum,
    const int64_t batch_size,
    const int64_t num_points,
    const int64_t num_pillars,
    const float x_min,
    const float y_min,
    const float z_min,
    const float x_max,
    const float y_max,
    const float z_max,
    const float voxel_x,
    const float voxel_y,
    const int64_t grid_x,
    const int64_t grid_y) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * num_points;
  if (idx >= total) {
    return;
  }
  const int b = idx / num_points;
  const int p = idx - b * num_points;
  const int64_t mask_offset = b * num_points + p;
  if (!points_mask[mask_offset]) {
    pillar_ids_out[mask_offset] = -1;
    return;
  }

  const int64_t xyz_offset = (b * num_points + p) * 3;
  const float x = points_xyz[xyz_offset + 0];
  const float y = points_xyz[xyz_offset + 1];
  const float z = points_xyz[xyz_offset + 2];

  if (x < x_min || x >= x_max || y < y_min || y >= y_max || z < z_min || z >= z_max) {
    pillar_ids_out[mask_offset] = -1;
    return;
  }

  const int gx = static_cast<int>(grid_x);
  const int gy = static_cast<int>(grid_y);
  const int ix = max(0, min(static_cast<int>((x - x_min) / voxel_x), gx - 1));
  const int iy = max(0, min(static_cast<int>((y - y_min) / voxel_y), gy - 1));
  const int64_t pillar_id = ix * grid_y + iy;

  pillar_ids_out[mask_offset] = pillar_id;
  const int64_t count_offset = b * num_pillars + pillar_id;
  atomicAdd(point_count + count_offset, 1);
  const int64_t sum_offset = (b * num_pillars + pillar_id) * 3;
  atomicAdd(xyz_sum + sum_offset + 0, x);
  atomicAdd(xyz_sum + sum_offset + 1, y);
  atomicAdd(xyz_sum + sum_offset + 2, z);
}

__global__ void pillar_preprocess_pass1_kernel(
    const bool* points_mask,
    const float* points_xyz,
    int64_t* pillar_ids_out,
    int32_t* point_count,
    float* xyz_sum,
    float* xyz_sq_sum,
    const int64_t batch_size,
    const int64_t num_points,
    const int64_t num_pillars,
    const float x_min,
    const float y_min,
    const float z_min,
    const float x_max,
    const float y_max,
    const float z_max,
    const float voxel_x,
    const float voxel_y,
    const int64_t grid_x,
    const int64_t grid_y) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * num_points;
  if (idx >= total) {
    return;
  }
  const int b = idx / num_points;
  const int p = idx - b * num_points;
  const int64_t mask_offset = b * num_points + p;
  if (!points_mask[mask_offset]) {
    pillar_ids_out[mask_offset] = -1;
    return;
  }

  const int64_t xyz_offset = (b * num_points + p) * 3;
  const float x = points_xyz[xyz_offset + 0];
  const float y = points_xyz[xyz_offset + 1];
  const float z = points_xyz[xyz_offset + 2];

  if (x < x_min || x >= x_max || y < y_min || y >= y_max || z < z_min || z >= z_max) {
    pillar_ids_out[mask_offset] = -1;
    return;
  }

  const int gx = static_cast<int>(grid_x);
  const int gy = static_cast<int>(grid_y);
  const int ix = max(0, min(static_cast<int>((x - x_min) / voxel_x), gx - 1));
  const int iy = max(0, min(static_cast<int>((y - y_min) / voxel_y), gy - 1));
  const int64_t pillar_id = ix * grid_y + iy;

  pillar_ids_out[mask_offset] = pillar_id;
  const int64_t count_offset = b * num_pillars + pillar_id;
  atomicAdd(point_count + count_offset, 1);
  const int64_t sum_offset = (b * num_pillars + pillar_id) * 3;
  atomicAdd(xyz_sum + sum_offset + 0, x);
  atomicAdd(xyz_sum + sum_offset + 1, y);
  atomicAdd(xyz_sum + sum_offset + 2, z);
  atomicAdd(xyz_sq_sum + sum_offset + 0, x * x);
  atomicAdd(xyz_sq_sum + sum_offset + 1, y * y);
  atomicAdd(xyz_sq_sum + sum_offset + 2, z * z);
}

__global__ void pillar_preprocess_pass2_kernel(
    const float* points_xyz,
    const float* points_feature,
    const int64_t* pillar_ids,
    const int32_t* point_count,
    const float* xyz_sum,
    const float* xyz_sq_sum,
    float* point_features_out,
    const int64_t batch_size,
    const int64_t num_points,
    const int64_t num_pillars,
    const float x_min,
    const float y_min,
    const float z_min,
    const float z_max,
    const float voxel_x,
    const float voxel_y,
    const int64_t grid_x,
    const int64_t grid_y,
    const int64_t extra_feature_dim,
    const int64_t feature_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * num_points;
  if (idx >= total) {
    return;
  }

  const int b = idx / num_points;
  const int p = idx - b * num_points;
  const int64_t mask_offset = b * num_points + p;
  const int64_t pillar_id = pillar_ids[mask_offset];
  if (pillar_id < 0) {
    return;
  }

  const int64_t xyz_offset = (b * num_points + p) * 3;
  const float x = points_xyz[xyz_offset + 0];
  const float y = points_xyz[xyz_offset + 1];
  const float z = points_xyz[xyz_offset + 2];

  const int gx = static_cast<int>(grid_x);
  const int gy = static_cast<int>(grid_y);
  const int ix = max(0, min(static_cast<int>((x - x_min) / voxel_x), gx - 1));
  const int iy = max(0, min(static_cast<int>((y - y_min) / voxel_y), gy - 1));

  const float center_x = x_min + (static_cast<float>(ix) + 0.5f) * voxel_x;
  const float center_y = y_min + (static_cast<float>(iy) + 0.5f) * voxel_y;
  const float center_z = z_min + (z_max - z_min) * 0.5f;

  const int64_t count_offset = b * num_pillars + pillar_id;
  const int32_t count = point_count[count_offset];
  float mean_x = 0.0f;
  float mean_y = 0.0f;
  float mean_z = 0.0f;
  float mean_x2 = 0.0f;
  float mean_y2 = 0.0f;
  float mean_z2 = 0.0f;
  if (count > 0) {
    const int64_t sum_offset = count_offset * 3;
    const float inv_count = 1.0f / static_cast<float>(count);
    mean_x = xyz_sum[sum_offset + 0] * inv_count;
    mean_y = xyz_sum[sum_offset + 1] * inv_count;
    mean_z = xyz_sum[sum_offset + 2] * inv_count;
    mean_x2 = xyz_sq_sum[sum_offset + 0] * inv_count;
    mean_y2 = xyz_sq_sum[sum_offset + 1] * inv_count;
    mean_z2 = xyz_sq_sum[sum_offset + 2] * inv_count;
  }

  const float f_cluster_x = x - mean_x;
  const float f_cluster_y = y - mean_y;
  const float f_cluster_z = z - mean_z;
  const float f_center_x = x - center_x;
  const float f_center_y = y - center_y;
  const float f_center_z = z - center_z;
  const float r = sqrtf(x * x + y * y);
  const float z_rel = z - z_min;
  const float eps = 1e-6f;
  const float inv_r = 1.0f / (r > eps ? r : eps);
  const float theta = atan2f(y, x);
  const float sin_theta = sinf(theta);
  const float cos_theta = cosf(theta);
  const float var_x = fmaxf(mean_x2 - mean_x * mean_x, 0.0f);
  const float var_y = fmaxf(mean_y2 - mean_y * mean_y, 0.0f);
  const float var_z = fmaxf(mean_z2 - mean_z * mean_z, 0.0f);

  float* out = point_features_out + (static_cast<int64_t>(b) * num_points + p) * feature_dim;
  int64_t offset = 0;
  // Feature order: [x, y, z, r, z_rel, inv_r, sin_theta, cos_theta, var_x, var_y, var_z, extra..., f_cluster(3), f_center(3)]
  out[offset++] = x;
  out[offset++] = y;
  out[offset++] = z;
  out[offset++] = r;
  out[offset++] = z_rel;
  out[offset++] = inv_r;
  out[offset++] = sin_theta;
  out[offset++] = cos_theta;
  out[offset++] = var_x;
  out[offset++] = var_y;
  out[offset++] = var_z;
  for (int64_t c = 0; c < extra_feature_dim; ++c) {
    const int64_t feat_offset = (static_cast<int64_t>(b) * num_points + p) * extra_feature_dim + c;
    out[offset++] = points_feature[feat_offset];
  }
  out[offset++] = f_cluster_x;
  out[offset++] = f_cluster_y;
  out[offset++] = f_cluster_z;
  out[offset++] = f_center_x;
  out[offset++] = f_center_y;
  out[offset++] = f_center_z;
}

}  // namespace

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
    int64_t grid_y) {
  TORCH_CHECK(points_mask.is_cuda(), "points_mask must be CUDA");
  TORCH_CHECK(points_xyz.is_cuda(), "points_xyz must be CUDA");
  TORCH_CHECK(points_mask.dim() == 2, "points_mask must be [B, P]");
  TORCH_CHECK(points_xyz.dim() == 3 && points_xyz.size(2) == 3, "points_xyz must be [B, P, 3]");

  auto opts_i64 = points_mask.options().dtype(torch::kInt64);
  auto opts_f = points_xyz.options().dtype(torch::kFloat);
  auto opts_i32 = points_xyz.options().dtype(torch::kInt32);
  const int64_t batch_size = points_mask.size(0);
  const int64_t num_points = points_mask.size(1);
  const int64_t num_pillars = grid_x * grid_y;

  auto pillar_ids = torch::full({batch_size, num_points}, -1, opts_i64);
  auto point_count = torch::zeros({batch_size, num_pillars}, opts_i32);
  auto xyz_sum = torch::zeros({batch_size, num_pillars, 3}, opts_f);

  const int threads = 256;
  const int blocks = (batch_size * num_points + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  pillar_stats_kernel<<<blocks, threads, 0, stream>>>(
      points_mask.data_ptr<bool>(),
      points_xyz.data_ptr<float>(),
      pillar_ids.data_ptr<int64_t>(),
      point_count.data_ptr<int32_t>(),
      xyz_sum.data_ptr<float>(),
      batch_size,
      num_points,
      num_pillars,
      static_cast<float>(x_min),
      static_cast<float>(y_min),
      static_cast<float>(z_min),
      static_cast<float>(x_max),
      static_cast<float>(y_max),
      static_cast<float>(z_max),
      static_cast<float>(voxel_x),
      static_cast<float>(voxel_y),
      grid_x,
      grid_y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {pillar_ids, point_count, xyz_sum};
}

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
    int64_t grid_y) {
  TORCH_CHECK(points_mask.is_cuda(), "points_mask must be CUDA");
  TORCH_CHECK(points_xyz.is_cuda(), "points_xyz must be CUDA");
  TORCH_CHECK(points_feature.is_cuda(), "points_feature must be CUDA");
  TORCH_CHECK(points_mask.dim() == 2, "points_mask must be [B, P]");
  TORCH_CHECK(points_xyz.dim() == 3 && points_xyz.size(2) == 3, "points_xyz must be [B, P, 3]");
  TORCH_CHECK(points_feature.dim() == 3, "points_feature must be [B, P, C]");
  TORCH_CHECK(points_feature.size(0) == points_xyz.size(0), "points_feature batch must match points_xyz");
  TORCH_CHECK(points_feature.size(1) == points_xyz.size(1), "points_feature points must match points_xyz");
  TORCH_CHECK(points_mask.is_contiguous(), "points_mask must be contiguous");
  TORCH_CHECK(points_xyz.is_contiguous(), "points_xyz must be contiguous");
  TORCH_CHECK(points_feature.is_contiguous(), "points_feature must be contiguous");
  TORCH_CHECK(points_xyz.scalar_type() == torch::kFloat, "points_xyz must be float32");
  TORCH_CHECK(points_feature.scalar_type() == torch::kFloat, "points_feature must be float32");

  auto opts_i64 = points_mask.options().dtype(torch::kInt64);
  auto opts_i32 = points_mask.options().dtype(torch::kInt32);
  auto opts_f = points_xyz.options().dtype(torch::kFloat);
  const int64_t batch_size = points_mask.size(0);
  const int64_t num_points = points_mask.size(1);
  const int64_t num_pillars = grid_x * grid_y;
  const int64_t extra_feature_dim = points_feature.size(2);
  const int64_t feature_dim = 11 + extra_feature_dim + 6;

  auto pillar_ids = torch::full({batch_size, num_points}, -1, opts_i64);
  auto point_count = torch::zeros({batch_size, num_pillars}, opts_i32);
  auto xyz_sum = torch::zeros({batch_size, num_pillars, 3}, opts_f);
  auto xyz_sq_sum = torch::zeros({batch_size, num_pillars, 3}, opts_f);
  auto point_features = torch::zeros({batch_size, num_points, feature_dim}, opts_f);

  const int threads = 256;
  const int blocks = (batch_size * num_points + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  pillar_preprocess_pass1_kernel<<<blocks, threads, 0, stream>>>(
      points_mask.data_ptr<bool>(),
      points_xyz.data_ptr<float>(),
      pillar_ids.data_ptr<int64_t>(),
      point_count.data_ptr<int32_t>(),
      xyz_sum.data_ptr<float>(),
      xyz_sq_sum.data_ptr<float>(),
      batch_size,
      num_points,
      num_pillars,
      static_cast<float>(x_min),
      static_cast<float>(y_min),
      static_cast<float>(z_min),
      static_cast<float>(x_max),
      static_cast<float>(y_max),
      static_cast<float>(z_max),
      static_cast<float>(voxel_x),
      static_cast<float>(voxel_y),
      grid_x,
      grid_y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  pillar_preprocess_pass2_kernel<<<blocks, threads, 0, stream>>>(
      points_xyz.data_ptr<float>(),
      points_feature.data_ptr<float>(),
      pillar_ids.data_ptr<int64_t>(),
      point_count.data_ptr<int32_t>(),
      xyz_sum.data_ptr<float>(),
      xyz_sq_sum.data_ptr<float>(),
      point_features.data_ptr<float>(),
      batch_size,
      num_points,
      num_pillars,
      static_cast<float>(x_min),
      static_cast<float>(y_min),
      static_cast<float>(z_min),
      static_cast<float>(z_max),
      static_cast<float>(voxel_x),
      static_cast<float>(voxel_y),
      grid_x,
      grid_y,
      extra_feature_dim,
      feature_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {pillar_ids, point_count, xyz_sum, point_features};
}
