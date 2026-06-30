// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>

namespace pcod_common {

struct PillarGrid {
  int grid_x = 0;
  int grid_y = 0;
  float x_min = 0.0f;
  float x_max = 0.0f;
  float y_min = 0.0f;
  float y_max = 0.0f;
  float z_min = 0.0f;
  float z_max = 0.0f;
  std::vector<float> centers;  // [num_pillars, 3]

  const float* center_at(int idx) const { return centers.data() + static_cast<std::size_t>(idx) * 3; }
};

PillarGrid BuildPillarGrid(const std::array<int, 2>& pillar_map_size,
                           const std::array<std::array<float, 2>, 3>& pillar_map_range,
                           int first_up_stride,
                           int stride);

}  // namespace pcod_common
