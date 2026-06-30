// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/pillar_grid.hpp"

int main() {
  const auto grid = pcod_common::BuildPillarGrid({2, 2}, {{{0.0f, 2.0f}, {0.0f, 2.0f}, {0.0f, 1.0f}}}, 1, 1);
  return grid.centers.size() == 12 ? 0 : 1;
}
