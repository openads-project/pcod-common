// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/pillar_grid.hpp"

#include <cassert>
#include <cmath>

int main() {
  {
    auto grid = pcod_common::BuildPillarGrid({2, 3}, {{{0.0f, 2.0f}, {0.0f, 3.0f}, {-1.0f, 1.0f}}}, 1, 1);
    assert(grid.grid_x == 2);
    assert(grid.grid_y == 3);

    const float* center_00 = grid.center_at(0);  // ix=0, iy=0
    assert(std::abs(center_00[0] - 0.5f) < 1e-6f);
    assert(std::abs(center_00[1] - 0.5f) < 1e-6f);

    const float* center_02 = grid.center_at(2);  // ix=0, iy=2
    assert(std::abs(center_02[0] - 0.5f) < 1e-6f);
    assert(std::abs(center_02[1] - 2.5f) < 1e-6f);

    const float* center_10 = grid.center_at(3);  // ix=1, iy=0
    assert(std::abs(center_10[0] - 1.5f) < 1e-6f);
    assert(std::abs(center_10[1] - 0.5f) < 1e-6f);

    const float* center_12 = grid.center_at(5);  // ix=1, iy=2
    assert(std::abs(center_12[0] - 1.5f) < 1e-6f);
    assert(std::abs(center_12[1] - 2.5f) < 1e-6f);
  }

  {
    auto scaled = pcod_common::BuildPillarGrid({4, 4}, {{{0.0f, 4.0f}, {0.0f, 4.0f}, {-1.0f, 1.0f}}}, 1, 2);
    assert(scaled.grid_x == 2);
    assert(scaled.grid_y == 2);

    const float* first = scaled.center_at(0);
    const float* last = scaled.center_at(3);
    assert(std::abs(first[0] - 1.0f) < 1e-6f);
    assert(std::abs(first[1] - 1.0f) < 1e-6f);
    assert(std::abs(last[0] - 3.0f) < 1e-6f);
    assert(std::abs(last[1] - 3.0f) < 1e-6f);
  }

  return 0;
}
