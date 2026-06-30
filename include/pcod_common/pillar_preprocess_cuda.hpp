// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "pcod_common/point_preprocess.hpp"

namespace pcod_common {

struct PillarPreprocessPoint {
  float x;
  float y;
  float z;
  float intensity;
};

struct PillarPreprocessCudaConfig {
  float x_min = 0.0f;
  float x_max = 0.0f;
  float y_min = 0.0f;
  float y_max = 0.0f;
  float z_min = 0.0f;
  float z_max = 0.0f;
  float voxel_x = 0.0f;
  float voxel_y = 0.0f;
  float value_threshold = 1.0f;
  float min_value = 0.0f;
  float max_value = 1.0f;
  float epsilon = 1e-6f;
  float z_score_mean = 0.0f;
  float z_score_std = 1.0f;
  float center_z = 0.0f;
  std::int32_t grid_x = 0;
  std::int32_t grid_y = 0;
  std::int32_t num_pillars = 0;
  std::int32_t max_num_points = 0;
  std::int32_t feature_dim = 0;
  PointFeatureNormalizationType normalization_type = PointFeatureNormalizationType::kNone;
};

struct PillarPreprocessCudaOutputs {
  float* point_features = nullptr;
  std::int64_t* pillar_ids = nullptr;
  bool* valid_mask = nullptr;
  bool* pillar_masks = nullptr;
};

struct PillarPreprocessCudaDeviceOutputs {
  float* point_features = nullptr;
  std::int64_t* pillar_ids = nullptr;
  bool* valid_mask = nullptr;
  bool* pillar_masks = nullptr;
};

bool HasCudaPillarPreprocessSupport();

class PillarPreprocessCudaContext {
 public:
  PillarPreprocessCudaContext();
  ~PillarPreprocessCudaContext();

  PillarPreprocessCudaContext(const PillarPreprocessCudaContext&) = delete;
  PillarPreprocessCudaContext& operator=(const PillarPreprocessCudaContext&) = delete;
  PillarPreprocessCudaContext(PillarPreprocessCudaContext&&) noexcept;
  PillarPreprocessCudaContext& operator=(PillarPreprocessCudaContext&&) noexcept;

  bool isAvailable() const;
  bool run(const PillarPreprocessPoint* points, std::int32_t num_points, const PillarPreprocessCudaConfig& config,
           const PillarPreprocessCudaOutputs& outputs, std::string* error_message);
  bool runToDevice(const PillarPreprocessPoint* points, std::int32_t num_points, const PillarPreprocessCudaConfig& config,
                   const PillarPreprocessCudaDeviceOutputs& outputs, std::string* error_message);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pcod_common
