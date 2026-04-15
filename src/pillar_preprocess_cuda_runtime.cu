#include "pcod_common/pillar_preprocess_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace pcod_common {

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr float kRadiusEpsilon = 1e-6f;

__device__ float NormalizePointFeatureDevice(float intensity, const PillarPreprocessCudaConfig& config) {
  switch (config.normalization_type) {
    case PointFeatureNormalizationType::kNone:
      return intensity;
    case PointFeatureNormalizationType::kValueThreshold: {
      if (config.value_threshold <= 0.0f) {
        return intensity;
      }
      const float clipped = fminf(fmaxf(intensity, 0.0f), config.value_threshold);
      return clipped / fmaxf(config.value_threshold, config.epsilon);
    }
    case PointFeatureNormalizationType::kMinMax: {
      const float denom = fmaxf(config.max_value - config.min_value, config.epsilon);
      const float scaled = (intensity - config.min_value) / denom;
      return fminf(fmaxf(scaled, 0.0f), 1.0f);
    }
    case PointFeatureNormalizationType::kZScore:
      return (intensity - config.z_score_mean) / fmaxf(config.z_score_std, config.epsilon);
  }

  return intensity;
}

__global__ void InitializePointOutputsKernel(float* point_features, std::int64_t* pillar_ids, bool* valid_mask,
                                             std::int32_t max_num_points, std::int32_t feature_dim,
                                             std::int64_t sentinel) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_num_points) {
    return;
  }

  valid_mask[idx] = false;
  pillar_ids[idx] = sentinel;
  float* feature_row = point_features + static_cast<std::size_t>(idx) * static_cast<std::size_t>(feature_dim);
  for (int feature_idx = 0; feature_idx < feature_dim; ++feature_idx) {
    feature_row[feature_idx] = 0.0f;
  }
}

__global__ void PreprocessPass1Kernel(const PillarPreprocessPoint* points, std::int32_t num_points,
                                      const PillarPreprocessCudaConfig config, std::int64_t* pillar_ids,
                                      bool* valid_mask, std::int32_t* pillar_counts, float* pillar_sum,
                                      float* pillar_sq_sum) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) {
    return;
  }

  const PillarPreprocessPoint point = points[idx];
  if (point.x < config.x_min || point.x >= config.x_max || point.y < config.y_min || point.y >= config.y_max ||
      point.z < config.z_min || point.z >= config.z_max) {
    pillar_ids[idx] = static_cast<std::int64_t>(config.num_pillars);
    valid_mask[idx] = false;
    return;
  }

  const int ix = max(0, min(static_cast<int>((point.x - config.x_min) / config.voxel_x), config.grid_x - 1));
  const int iy = max(0, min(static_cast<int>((point.y - config.y_min) / config.voxel_y), config.grid_y - 1));
  const int pillar_id = ix * config.grid_y + iy;

  pillar_ids[idx] = static_cast<std::int64_t>(pillar_id);
  valid_mask[idx] = true;
  atomicAdd(pillar_counts + pillar_id, 1);

  const std::size_t sum_offset = static_cast<std::size_t>(pillar_id) * 3u;
  atomicAdd(pillar_sum + sum_offset + 0u, point.x);
  atomicAdd(pillar_sum + sum_offset + 1u, point.y);
  atomicAdd(pillar_sum + sum_offset + 2u, point.z);
  atomicAdd(pillar_sq_sum + sum_offset + 0u, point.x * point.x);
  atomicAdd(pillar_sq_sum + sum_offset + 1u, point.y * point.y);
  atomicAdd(pillar_sq_sum + sum_offset + 2u, point.z * point.z);
}

__global__ void FinalizePillarMasksKernel(const std::int32_t* pillar_counts, std::int32_t num_pillars,
                                          bool* pillar_masks) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pillars) {
    return;
  }
  pillar_masks[idx] = pillar_counts[idx] > 0;
}

__global__ void PreprocessPass2Kernel(const PillarPreprocessPoint* points, std::int32_t num_points,
                                      const PillarPreprocessCudaConfig config, const std::int64_t* pillar_ids,
                                      const std::int32_t* pillar_counts, const float* pillar_sum,
                                      const float* pillar_sq_sum, float* point_features) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) {
    return;
  }

  const PillarPreprocessPoint point = points[idx];
  const int pillar_id = static_cast<int>(pillar_ids[idx]);
  if (pillar_id < 0 || pillar_id >= config.num_pillars) {
    return;
  }
  const std::size_t sum_offset = static_cast<std::size_t>(pillar_id) * 3u;
  const int raw_point_count = pillar_counts[pillar_id];
  const int point_count = raw_point_count > 0 ? raw_point_count : 1;
  const float inv_count = 1.0f / static_cast<float>(point_count);

  const float mean_x = pillar_sum[sum_offset + 0u] * inv_count;
  const float mean_y = pillar_sum[sum_offset + 1u] * inv_count;
  const float mean_z = pillar_sum[sum_offset + 2u] * inv_count;
  const float mean_x2 = pillar_sq_sum[sum_offset + 0u] * inv_count;
  const float mean_y2 = pillar_sq_sum[sum_offset + 1u] * inv_count;
  const float mean_z2 = pillar_sq_sum[sum_offset + 2u] * inv_count;

  const int ix = max(0, min(static_cast<int>((point.x - config.x_min) / config.voxel_x), config.grid_x - 1));
  const int iy = max(0, min(static_cast<int>((point.y - config.y_min) / config.voxel_y), config.grid_y - 1));
  const float center_x = config.x_min + (static_cast<float>(ix) + 0.5f) * config.voxel_x;
  const float center_y = config.y_min + (static_cast<float>(iy) + 0.5f) * config.voxel_y;
  const float r2 = point.x * point.x + point.y * point.y;
  const float r = sqrtf(r2);
  const float inv_r = 1.0f / (r > kRadiusEpsilon ? r : kRadiusEpsilon);
  float sin_theta = 0.0f;
  float cos_theta = 1.0f;
  if (r > 0.0f) {
    const float inv_norm = 1.0f / r;
    sin_theta = point.y * inv_norm;
    cos_theta = point.x * inv_norm;
  }

  float* feature_row = point_features + static_cast<std::size_t>(idx) * static_cast<std::size_t>(config.feature_dim);
  feature_row[0] = point.x;
  feature_row[1] = point.y;
  feature_row[2] = point.z;
  feature_row[3] = r;
  feature_row[4] = point.z - config.z_min;
  feature_row[5] = inv_r;
  feature_row[6] = sin_theta;
  feature_row[7] = cos_theta;
  feature_row[8] = fmaxf(mean_x2 - mean_x * mean_x, 0.0f);
  feature_row[9] = fmaxf(mean_y2 - mean_y * mean_y, 0.0f);
  feature_row[10] = fmaxf(mean_z2 - mean_z * mean_z, 0.0f);
  feature_row[11] = NormalizePointFeatureDevice(point.intensity, config);
  feature_row[12] = point.x - mean_x;
  feature_row[13] = point.y - mean_y;
  feature_row[14] = point.z - mean_z;
  feature_row[15] = point.x - center_x;
  feature_row[16] = point.y - center_y;
  feature_row[17] = point.z - config.center_z;
}

bool CheckCuda(cudaError_t status, const char* operation, std::string* error_message) {
  if (status == cudaSuccess) {
    return true;
  }
  if (error_message != nullptr) {
    *error_message = std::string(operation) + " failed: " + cudaGetErrorString(status);
  }
  cudaGetLastError();
  return false;
}

}  // namespace

struct PillarPreprocessCudaContext::Impl {
  PillarPreprocessPoint* points_device = nullptr;
  std::int64_t* pillar_ids_device = nullptr;
  bool* valid_mask_device = nullptr;
  bool* pillar_masks_device = nullptr;
  std::int32_t* pillar_counts_device = nullptr;
  float* pillar_sum_device = nullptr;
  float* pillar_sq_sum_device = nullptr;
  float* point_features_device = nullptr;
  std::int32_t point_capacity = 0;
  std::int32_t num_pillars_capacity = 0;
  std::int32_t feature_dim_capacity = 0;

  ~Impl() {
    cudaFree(points_device);
    cudaFree(pillar_ids_device);
    cudaFree(valid_mask_device);
    cudaFree(pillar_masks_device);
    cudaFree(pillar_counts_device);
    cudaFree(pillar_sum_device);
    cudaFree(pillar_sq_sum_device);
    cudaFree(point_features_device);
  }
};

bool HasCudaPillarPreprocessSupport() {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return device_count > 0;
}

PillarPreprocessCudaContext::PillarPreprocessCudaContext() : impl_(std::make_unique<Impl>()) {}

PillarPreprocessCudaContext::~PillarPreprocessCudaContext() = default;

PillarPreprocessCudaContext::PillarPreprocessCudaContext(PillarPreprocessCudaContext&&) noexcept = default;

PillarPreprocessCudaContext& PillarPreprocessCudaContext::operator=(PillarPreprocessCudaContext&&) noexcept = default;

bool PillarPreprocessCudaContext::isAvailable() const { return HasCudaPillarPreprocessSupport(); }

namespace {

bool ValidateRunArguments(const PillarPreprocessPoint* points, std::int32_t num_points,
                          const PillarPreprocessCudaConfig& config, std::string* error_message) {
  if (points == nullptr && num_points > 0) {
    if (error_message != nullptr) {
      *error_message = "CUDA preprocessing received null input points";
    }
    return false;
  }

  if (num_points < 0 || num_points > config.max_num_points || config.num_pillars <= 0 || config.feature_dim <= 0) {
    if (error_message != nullptr) {
      *error_message = "CUDA preprocessing received invalid dimensions";
    }
    return false;
  }

  if (config.grid_x <= 0 || config.grid_y <= 0 || config.num_pillars != config.grid_x * config.grid_y ||
      !(config.x_min < config.x_max) || !(config.y_min < config.y_max) || !(config.z_min < config.z_max) ||
      !(config.voxel_x > 0.0f) || !(config.voxel_y > 0.0f)) {
    if (error_message != nullptr) {
      *error_message = "CUDA preprocessing received inconsistent grid/range configuration";
    }
    return false;
  }

  return true;
}

}  // namespace

bool PillarPreprocessCudaContext::runToDevice(const PillarPreprocessPoint* points, std::int32_t num_points,
                                              const PillarPreprocessCudaConfig& config,
                                              const PillarPreprocessCudaDeviceOutputs& outputs,
                                              std::string* error_message) {
  if (!isAvailable()) {
    if (error_message != nullptr) {
      *error_message = "CUDA preprocessing is not available at runtime";
    }
    return false;
  }

  if (outputs.point_features == nullptr || outputs.pillar_ids == nullptr || outputs.valid_mask == nullptr ||
      outputs.pillar_masks == nullptr) {
    if (error_message != nullptr) {
      *error_message = "CUDA preprocessing received null input/output buffers";
    }
    return false;
  }

  if (!ValidateRunArguments(points, num_points, config, error_message)) {
    return false;
  }

  auto ensure_buffer = [&](auto*& ptr, std::size_t bytes, auto& capacity_value, auto needed_capacity,
                           const char* name) -> bool {
    if (capacity_value >= needed_capacity && ptr != nullptr) {
      return true;
    }
    if (ptr != nullptr) {
      if (!CheckCuda(cudaFree(ptr), name, error_message)) {
        return false;
      }
      ptr = nullptr;
    }
    if (!CheckCuda(cudaMalloc(&ptr, bytes), name, error_message)) {
      return false;
    }
    capacity_value = needed_capacity;
    return true;
  };

  if (!ensure_buffer(impl_->points_device, static_cast<std::size_t>(config.max_num_points) * sizeof(PillarPreprocessPoint),
                     impl_->point_capacity, config.max_num_points, "cudaMalloc(points_device)")) {
    return false;
  }
  if (!ensure_buffer(impl_->pillar_counts_device, static_cast<std::size_t>(config.num_pillars) * sizeof(std::int32_t),
                     impl_->num_pillars_capacity, config.num_pillars, "cudaMalloc(pillar_counts_device)")) {
    return false;
  }
  if (!ensure_buffer(impl_->pillar_sum_device,
                     static_cast<std::size_t>(config.num_pillars) * 3u * sizeof(float), impl_->num_pillars_capacity,
                     config.num_pillars, "cudaMalloc(pillar_sum_device)")) {
    return false;
  }
  if (!ensure_buffer(impl_->pillar_sq_sum_device,
                     static_cast<std::size_t>(config.num_pillars) * 3u * sizeof(float), impl_->num_pillars_capacity,
                     config.num_pillars, "cudaMalloc(pillar_sq_sum_device)")) {
    return false;
  }

  const int point_blocks = (config.max_num_points + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const int selected_point_blocks = (num_points + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const int pillar_blocks = (config.num_pillars + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const std::int64_t sentinel = static_cast<std::int64_t>(config.num_pillars);
  auto* point_features_device = outputs.point_features;
  auto* pillar_ids_device = outputs.pillar_ids;
  auto* valid_mask_device = outputs.valid_mask;
  auto* pillar_masks_device = outputs.pillar_masks;

  if (!CheckCuda(cudaMemcpy(impl_->points_device, points,
                            static_cast<std::size_t>(num_points) * sizeof(PillarPreprocessPoint),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(points_device)", error_message)) {
    return false;
  }

  InitializePointOutputsKernel<<<point_blocks, kThreadsPerBlock>>>(point_features_device, pillar_ids_device,
                                                                   valid_mask_device, config.max_num_points,
                                                                   config.feature_dim, sentinel);
  if (!CheckCuda(cudaGetLastError(), "InitializePointOutputsKernel launch", error_message)) {
    return false;
  }
  if (!CheckCuda(cudaMemset(pillar_masks_device, 0, static_cast<std::size_t>(config.num_pillars) * sizeof(bool)),
                 "cudaMemset(pillar_masks_device)", error_message)) {
    return false;
  }
  if (!CheckCuda(cudaMemset(impl_->pillar_counts_device, 0,
                            static_cast<std::size_t>(config.num_pillars) * sizeof(std::int32_t)),
                 "cudaMemset(pillar_counts_device)", error_message)) {
    return false;
  }
  if (!CheckCuda(cudaMemset(impl_->pillar_sum_device, 0, static_cast<std::size_t>(config.num_pillars) * 3u * sizeof(float)),
                 "cudaMemset(pillar_sum_device)", error_message)) {
    return false;
  }
  if (!CheckCuda(
          cudaMemset(impl_->pillar_sq_sum_device, 0, static_cast<std::size_t>(config.num_pillars) * 3u * sizeof(float)),
          "cudaMemset(pillar_sq_sum_device)", error_message)) {
    return false;
  }

  if (num_points > 0) {
    PreprocessPass1Kernel<<<selected_point_blocks, kThreadsPerBlock>>>(
        impl_->points_device, num_points, config, pillar_ids_device, valid_mask_device,
        impl_->pillar_counts_device, impl_->pillar_sum_device, impl_->pillar_sq_sum_device);
    if (!CheckCuda(cudaGetLastError(), "PreprocessPass1Kernel launch", error_message)) {
      return false;
    }

    FinalizePillarMasksKernel<<<pillar_blocks, kThreadsPerBlock>>>(impl_->pillar_counts_device, config.num_pillars,
                                                                   pillar_masks_device);
    if (!CheckCuda(cudaGetLastError(), "FinalizePillarMasksKernel launch", error_message)) {
      return false;
    }

    PreprocessPass2Kernel<<<selected_point_blocks, kThreadsPerBlock>>>(
        impl_->points_device, num_points, config, pillar_ids_device, impl_->pillar_counts_device, impl_->pillar_sum_device,
        impl_->pillar_sq_sum_device, point_features_device);
    if (!CheckCuda(cudaGetLastError(), "PreprocessPass2Kernel launch", error_message)) {
      return false;
    }
  }

  return CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", error_message);
}

bool PillarPreprocessCudaContext::run(const PillarPreprocessPoint* points, std::int32_t num_points,
                                      const PillarPreprocessCudaConfig& config,
                                      const PillarPreprocessCudaOutputs& outputs, std::string* error_message) {
  if (outputs.point_features == nullptr || outputs.pillar_ids == nullptr || outputs.valid_mask == nullptr ||
      outputs.pillar_masks == nullptr) {
    if (error_message != nullptr) {
      *error_message = "CUDA preprocessing received null input/output buffers";
    }
    return false;
  }

  if (!ValidateRunArguments(points, num_points, config, error_message)) {
    return false;
  }

  auto ensure_buffer = [&](auto*& ptr, std::size_t bytes, auto& capacity_value, auto needed_capacity,
                           const char* name) -> bool {
    if (capacity_value >= needed_capacity && ptr != nullptr) {
      return true;
    }
    if (ptr != nullptr) {
      if (!CheckCuda(cudaFree(ptr), name, error_message)) {
        return false;
      }
      ptr = nullptr;
    }
    if (!CheckCuda(cudaMalloc(&ptr, bytes), name, error_message)) {
      return false;
    }
    capacity_value = needed_capacity;
    return true;
  };

  if (!ensure_buffer(impl_->pillar_ids_device, static_cast<std::size_t>(config.max_num_points) * sizeof(std::int64_t),
                     impl_->point_capacity, config.max_num_points, "cudaMalloc(pillar_ids_device)")) {
    return false;
  }
  if (!ensure_buffer(impl_->valid_mask_device, static_cast<std::size_t>(config.max_num_points) * sizeof(bool),
                     impl_->point_capacity, config.max_num_points, "cudaMalloc(valid_mask_device)")) {
    return false;
  }
  if (!ensure_buffer(impl_->point_features_device,
                     static_cast<std::size_t>(config.max_num_points) * static_cast<std::size_t>(config.feature_dim) *
                         sizeof(float),
                     impl_->feature_dim_capacity, config.feature_dim, "cudaMalloc(point_features_device)")) {
    return false;
  }
  if (!ensure_buffer(impl_->pillar_masks_device, static_cast<std::size_t>(config.num_pillars) * sizeof(bool),
                     impl_->num_pillars_capacity, config.num_pillars, "cudaMalloc(pillar_masks_device)")) {
    return false;
  }

  const PillarPreprocessCudaDeviceOutputs device_outputs{impl_->point_features_device, impl_->pillar_ids_device,
                                                         impl_->valid_mask_device, impl_->pillar_masks_device};
  if (!runToDevice(points, num_points, config, device_outputs, error_message)) {
    return false;
  }

  if (!CheckCuda(cudaMemcpy(outputs.point_features, impl_->point_features_device,
                            static_cast<std::size_t>(config.max_num_points) * static_cast<std::size_t>(config.feature_dim) *
                                sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(point_features)", error_message)) {
    return false;
  }
  if (!CheckCuda(cudaMemcpy(outputs.pillar_ids, impl_->pillar_ids_device,
                            static_cast<std::size_t>(config.max_num_points) * sizeof(std::int64_t),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(pillar_ids)", error_message)) {
    return false;
  }
  if (!CheckCuda(cudaMemcpy(outputs.valid_mask, impl_->valid_mask_device,
                            static_cast<std::size_t>(config.max_num_points) * sizeof(bool), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(valid_mask)", error_message)) {
    return false;
  }
  if (!CheckCuda(cudaMemcpy(outputs.pillar_masks, impl_->pillar_masks_device,
                            static_cast<std::size_t>(config.num_pillars) * sizeof(bool), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(pillar_masks)", error_message)) {
    return false;
  }

  return true;
}

}  // namespace pcod_common
