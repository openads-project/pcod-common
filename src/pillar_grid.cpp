// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/pillar_grid.hpp"

#include <cmath>

namespace pcod_common {

PillarGrid BuildPillarGrid(const std::array<int, 2>& pillar_map_size,
                           const std::array<std::array<float, 2>, 3>& pillar_map_range,
                           int first_up_stride,
                           int stride) {
  PillarGrid grid;
  const int size_x = pillar_map_size[0];
  const int size_y = pillar_map_size[1];
  const int stride_val = stride > 0 ? stride : 1;
  const int up_stride = first_up_stride > 0 ? first_up_stride : 1;

  grid.grid_x = size_x * up_stride / stride_val;
  grid.grid_y = size_y * up_stride / stride_val;
  grid.x_min = pillar_map_range[0][0];
  grid.x_max = pillar_map_range[0][1];
  grid.y_min = pillar_map_range[1][0];
  grid.y_max = pillar_map_range[1][1];
  grid.z_min = pillar_map_range[2][0];
  grid.z_max = pillar_map_range[2][1];

  const float dx = (grid.x_max - grid.x_min) / static_cast<float>(grid.grid_x);
  const float dy = (grid.y_max - grid.y_min) / static_cast<float>(grid.grid_y);
  const float half_dx = dx * 0.5F;
  const float half_dy = dy * 0.5F;

  const int num_pillars = grid.grid_x * grid.grid_y;
  grid.centers.resize(static_cast<std::size_t>(num_pillars) * 3);

  std::size_t offset = 0;
  for (int ix = 0; ix < grid.grid_x; ++ix) {
    const float x_center = grid.x_min + half_dx + static_cast<float>(ix) * dx;
    for (int iy = 0; iy < grid.grid_y; ++iy) {
      const float y_center = grid.y_min + half_dy + static_cast<float>(iy) * dy;
      grid.centers[offset++] = x_center;
      grid.centers[offset++] = y_center;
      grid.centers[offset++] = 0.0F;
    }
  }

  return grid;
}

}  // namespace pcod_common
