// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "pcod_common/point_preprocess.hpp"

namespace pcod_common {

/** Host representation of one XYZI point consumed by CUDA preprocessing. */
struct PillarPreprocessPoint {
  float x;          ///< X coordinate.
  float y;          ///< Y coordinate.
  float z;          ///< Z coordinate.
  float intensity;  ///< Additional scalar point feature.
};

/** Complete geometry, normalization, and output-shape contract for CUDA preprocessing. */
struct PillarPreprocessCudaConfig {
  float x_min = 0.0f;               ///< Minimum accepted X coordinate.
  float x_max = 0.0f;               ///< Maximum accepted X coordinate.
  float y_min = 0.0f;               ///< Minimum accepted Y coordinate.
  float y_max = 0.0f;               ///< Maximum accepted Y coordinate.
  float z_min = 0.0f;               ///< Minimum accepted Z coordinate.
  float z_max = 0.0f;               ///< Maximum accepted Z coordinate.
  float voxel_x = 0.0f;             ///< Voxel width along X.
  float voxel_y = 0.0f;             ///< Voxel width along Y.
  float value_threshold = 1.0f;     ///< Divisor for value-threshold normalization.
  float min_value = 0.0f;           ///< Minimum for min-max normalization.
  float max_value = 1.0f;           ///< Maximum for min-max normalization.
  float epsilon = 1e-6f;            ///< Lower bound for unsafe normalization divisors.
  float z_score_mean = 0.0f;        ///< Mean for Z-score normalization.
  float z_score_std = 1.0f;         ///< Standard deviation for Z-score normalization.
  float center_z = 0.0f;            ///< Z center written to pillar coordinates.
  std::int32_t grid_x = 0;          ///< Number of grid cells along X.
  std::int32_t grid_y = 0;          ///< Number of grid cells along Y.
  std::int32_t num_pillars = 0;     ///< Total number of output pillars.
  std::int32_t max_num_points = 0;  ///< Maximum number of input points.
  std::int32_t feature_dim = 0;     ///< Number of output features per point.
  PointFeatureNormalizationType normalization_type = PointFeatureNormalizationType::kNone;  ///< Feature transform.
};

/** Host-accessible output buffers populated by CUDA preprocessing. */
struct PillarPreprocessCudaOutputs {
  float* point_features = nullptr;     ///< Flat point-feature output.
  std::int64_t* pillar_ids = nullptr;  ///< Pillar index per accepted point.
  bool* valid_mask = nullptr;          ///< Validity flag per input point.
  bool* pillar_masks = nullptr;        ///< Occupancy flag per pillar.
};

/** Device-resident output buffers populated without a device-to-host copy. */
struct PillarPreprocessCudaDeviceOutputs {
  float* point_features = nullptr;     ///< Device point-feature output.
  std::int64_t* pillar_ids = nullptr;  ///< Device pillar index output.
  bool* valid_mask = nullptr;          ///< Device point-validity output.
  bool* pillar_masks = nullptr;        ///< Device pillar-occupancy output.
};

/** Return whether this build and runtime provide CUDA pillar preprocessing. */
bool HasCudaPillarPreprocessSupport();

/** Own the reusable CUDA resources required for pillar preprocessing. */
class PillarPreprocessCudaContext {
 public:
  /** Construct a context, retaining an unavailable state when CUDA initialization fails. */
  PillarPreprocessCudaContext();
  /** Release all owned CUDA resources. */
  ~PillarPreprocessCudaContext();

  PillarPreprocessCudaContext(const PillarPreprocessCudaContext&) = delete;
  PillarPreprocessCudaContext& operator=(const PillarPreprocessCudaContext&) = delete;
  /** @param other Context whose resources are transferred. */
  PillarPreprocessCudaContext(PillarPreprocessCudaContext&& other) noexcept;
  /** @param other Context whose resources are transferred. @return This context. */
  PillarPreprocessCudaContext& operator=(PillarPreprocessCudaContext&& other) noexcept;

  /** Return whether this context can execute CUDA preprocessing. */
  bool isAvailable() const;
  /** Run preprocessing and copy results to host-accessible buffers.
   * @param points Host input point array.
   * @param num_points Number of input points.
   * @param config Preprocessing contract.
   * @param outputs Host output buffers.
   * @param error_message Optional failure detail destination.
   * @return Whether preprocessing completed successfully.
   */
  bool run(const PillarPreprocessPoint* points,
           std::int32_t num_points,
           const PillarPreprocessCudaConfig& config,
           const PillarPreprocessCudaOutputs& outputs,
           std::string* error_message);
  /** Run preprocessing directly into device-resident buffers.
   * @param points Host input point array.
   * @param num_points Number of input points.
   * @param config Preprocessing contract.
   * @param outputs Device output buffers.
   * @param error_message Optional failure detail destination.
   * @return Whether preprocessing completed successfully.
   */
  bool runToDevice(const PillarPreprocessPoint* points,
                   std::int32_t num_points,
                   const PillarPreprocessCudaConfig& config,
                   const PillarPreprocessCudaDeviceOutputs& outputs,
                   std::string* error_message);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pcod_common
