// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/bounding_box.hpp"

#include <cassert>
#include <cmath>

namespace {

/** Build a minimal oriented bounding box for geometry tests.
 * @param x Box center X.
 * @param y Box center Y.
 * @param length Box length.
 * @param width Box width.
 * @param yaw Box heading.
 * @return Configured bounding box.
 */
pcod_common::BoundingBox MakeBox(float x, float y, float length, float width, float yaw) {
  pcod_common::BoundingBox box;
  box.center = {x, y};
  box.length = length;
  box.width = width;
  box.yaw = yaw;
  return box;
}

}  // namespace

/** Run bounding-box geometry regression checks. */
int main() {
  const pcod_common::BoundingBox base = MakeBox(0.0f, 0.0f, 4.0f, 2.0f, 0.0f);

  {
    const pcod_common::BoundingBox same = MakeBox(0.0f, 0.0f, 4.0f, 2.0f, 0.0f);
    const float area_a = base.intersection_area(same);
    const float area_b = same.intersection_area(base);
    assert(std::abs(area_a - area_b) < 1e-6f);
    assert(std::abs(area_a - (base.length * base.width)) < 1e-6f);
    assert(base.overlaps(same, 0.99f));
  }

  {
    const pcod_common::BoundingBox far = MakeBox(100.0f, 100.0f, 4.0f, 2.0f, 0.0f);
    assert(base.intersection_area(far) == 0.0f);
    assert(!base.overlaps(far, 0.0f));
  }

  {
    const pcod_common::BoundingBox shifted = MakeBox(1.0f, 0.0f, 4.0f, 2.0f, 0.0f);
    const float area_forward = base.intersection_area(shifted);
    const float area_reverse = shifted.intersection_area(base);
    assert(area_forward > 0.0f);
    assert(std::abs(area_forward - area_reverse) < 1e-6f);
  }

  {
    const pcod_common::BoundingBox rotated = MakeBox(0.0f, 0.0f, 4.0f, 2.0f, 1.57079632679f);
    const float area = base.intersection_area(rotated);
    assert(area > 0.0f);
    assert(area < base.length * base.width);
  }

  return 0;
}
