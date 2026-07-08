// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/bounding_box.hpp"

#include <algorithm>
#include <cmath>

namespace pcod_common {

float BoundingBox::intersection_area(const BoundingBox& other) const {
  auto rect1 = rectangle_vertices();
  auto rect2 = other.rectangle_vertices();

  std::vector<BoundingBoxVertex> intersection = rect1;

  for (std::size_t i = 0; i < rect2.size(); ++i) {
    if (intersection.size() <= 2) {
      break;
    }

    Line line(rect2[i], rect2[(i + 1) % rect2.size()]);
    std::vector<BoundingBoxVertex> new_intersection;
    std::vector<float> line_values;

    for (const auto& t : intersection) {
      line_values.push_back(line(t));
    }

    for (std::size_t j = 0; j < intersection.size(); ++j) {
      const BoundingBoxVertex& s = intersection[j];
      const BoundingBoxVertex& t = intersection[(j + 1) % intersection.size()];
      float s_value = line_values[j];
      float t_value = line_values[(j + 1) % line_values.size()];

      if (s_value <= 0) {
        new_intersection.push_back(s);
      }

      if (s_value * t_value < 0) {
        BoundingBoxVertex intersection_point = line.intersection(Line(s, t));
        new_intersection.push_back(intersection_point);
      }
    }

    intersection = new_intersection;
  }

  if (intersection.size() <= 2) {
    return 0.0F;
  }

  float area = 0.0F;
  for (std::size_t i = 0; i < intersection.size(); ++i) {
    const BoundingBoxVertex& p = intersection[i];
    const BoundingBoxVertex& q = intersection[(i + 1) % intersection.size()];
    area += p.cross(q);
  }

  return 0.5F * std::abs(area);
}

bool BoundingBox::overlaps(const BoundingBox& other, float iou_threshold) const {
  float intersection = intersection_area(other);
  float union_area = length * width + other.length * other.width - intersection;
  float iou = intersection / union_area;
  return iou > iou_threshold;
}

std::vector<BoundingBoxVertex> BoundingBox::rectangle_vertices() const {
  float angle = yaw;
  float dx = length / 2.0F;
  float dy = width / 2.0F;
  float dxcos = dx * std::cos(angle);
  float dxsin = dx * std::sin(angle);
  float dycos = dy * std::cos(angle);
  float dysin = dy * std::sin(angle);

  std::vector<BoundingBoxVertex> vertices;
  vertices.emplace_back(center[0] + (-dxcos - -dysin), center[1] + (-dxsin + -dycos));
  vertices.emplace_back(center[0] + (dxcos - -dysin), center[1] + (dxsin + -dycos));
  vertices.emplace_back(center[0] + (dxcos - dysin), center[1] + (dxsin + dycos));
  vertices.emplace_back(center[0] + (-dxcos - dysin), center[1] + (-dxsin + dycos));
  return vertices;
}

}  // namespace pcod_common
