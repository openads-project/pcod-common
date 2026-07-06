// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>

namespace pcod_common {

/** Spatial dimensions and center coordinates of the decoded pillar grid. */
struct PillarGrid {
  int grid_x = 0;              ///< Number of grid cells along X.
  int grid_y = 0;              ///< Number of grid cells along Y.
  float x_min = 0.0f;          ///< Minimum X coordinate.
  float x_max = 0.0f;          ///< Maximum X coordinate.
  float y_min = 0.0f;          ///< Minimum Y coordinate.
  float y_max = 0.0f;          ///< Maximum Y coordinate.
  float z_min = 0.0f;          ///< Minimum Z coordinate.
  float z_max = 0.0f;          ///< Maximum Z coordinate.
  std::vector<float> centers;  ///< Flat center array with shape `[num_pillars, 3]`.

  /** Return the three-coordinate center of pillar @p idx. */
  const float* center_at(int idx) const { return centers.data() + static_cast<std::size_t>(idx) * 3; }
};

/**
 * Build the output pillar grid represented by an exported model.
 * @param pillar_map_size Base map dimensions in X and Y.
 * @param pillar_map_range XYZ coordinate ranges.
 * @param first_up_stride Upsampling factor applied before model strides.
 * @param stride Model output stride.
 * @return Grid dimensions, ranges, and XYZ center coordinates.
 */
PillarGrid BuildPillarGrid(const std::array<int, 2>& pillar_map_size,
                           const std::array<std::array<float, 2>, 3>& pillar_map_range,
                           int first_up_stride,
                           int stride);

}  // namespace pcod_common
