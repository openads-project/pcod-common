// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/pbod_postprocess.hpp"

#include <cassert>
#include <functional>
#include <stdexcept>
#include <vector>

#include "pcod_common/pillar_grid.hpp"

namespace {

/** Return whether invoking @p fn throws the expected invalid-argument error.
 * @param fn Operation expected to throw.
 * @return Whether an invalid-argument error was observed.
 */
bool ExpectInvalidArgument(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

}  // namespace

/** Run PBOD decoder input-validation regression checks. */
int main() {
  using pcod_common::BuildPillarGrid;
  using pcod_common::DecodePbod;
  using pcod_common::PbodOutputsView;
  using pcod_common::PillarGrid;

  PillarGrid grid = BuildPillarGrid({1, 1}, {{{0.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 1.0f}}}, 1, 1);
  pcod_common::PbodPostprocessConfig config;
  config.class_names = {"car"};

  {
    PbodOutputsView bad_view;
    bad_view.num_pillars = 1;
    bad_view.num_classes = 1;
    bad_view.reg_dim = 7;
    assert(ExpectInvalidArgument([&]() { (void)DecodePbod(bad_view, grid, config); }));
  }

  float focal_logits[1] = {0.0f};
  float class_logits[1] = {1.0f};
  float size_posterior[3] = {1.0f, 1.0f, 1.0f};
  float reg_logits[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  {
    PbodOutputsView bad_view;
    bad_view.focal_logits = focal_logits;
    bad_view.class_logits = class_logits;
    bad_view.size_posterior = size_posterior;
    bad_view.reg_logits = reg_logits;
    bad_view.num_pillars = 1;
    bad_view.num_classes = 1;
    bad_view.reg_dim = 6;
    assert(ExpectInvalidArgument([&]() { (void)DecodePbod(bad_view, grid, config); }));
  }

  {
    PbodOutputsView good_view;
    good_view.focal_logits = focal_logits;
    good_view.class_logits = class_logits;
    good_view.size_posterior = size_posterior;
    good_view.reg_logits = reg_logits;
    good_view.num_pillars = 1;
    good_view.num_classes = 1;
    good_view.reg_dim = 7;

    PillarGrid bad_grid = grid;
    bad_grid.centers.clear();
    assert(ExpectInvalidArgument([&]() { (void)DecodePbod(good_view, bad_grid, config); }));
  }

  return 0;
}
