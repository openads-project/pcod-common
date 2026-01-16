#pragma once

#include <array>
#include <vector>

namespace pcod_common {

struct ClassificationEntry {
  std::size_t class_idx;
  float score;
};

struct BoundingBoxVertex {
  float x;
  float y;

  BoundingBoxVertex(float x_in = 0.0f, float y_in = 0.0f) : x(x_in), y(y_in) {}

  BoundingBoxVertex operator+(const BoundingBoxVertex& v) const { return {x + v.x, y + v.y}; }
  BoundingBoxVertex operator-(const BoundingBoxVertex& v) const { return {x - v.x, y - v.y}; }
  float cross(const BoundingBoxVertex& v) const { return x * v.y - y * v.x; }
};

struct Line {
  float a;
  float b;
  float c;

  Line(const BoundingBoxVertex& v1, const BoundingBoxVertex& v2) {
    a = v2.y - v1.y;
    b = v1.x - v2.x;
    c = v2.cross(v1);
  }

  float operator()(const BoundingBoxVertex& p) const { return a * p.x + b * p.y + c; }

  BoundingBoxVertex intersection(const Line& other) const {
    float w = a * other.b - b * other.a;
    return BoundingBoxVertex((b * other.c - c * other.b) / w, (c * other.a - a * other.c) / w);
  }
};

struct BoundingBox {
  std::array<float, 2> center;
  float z = 0.0f;
  float length = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  float yaw = 0.0f;
  float existence_probability = 0.0f;
  std::vector<ClassificationEntry> classification;
  bool has_velocity = false;
  float v_x = 0.0f;
  float v_y = 0.0f;

  bool overlaps(const BoundingBox& other, float iou_threshold) const;
  float intersection_area(const BoundingBox& other) const;
  std::vector<BoundingBoxVertex> rectangle_vertices() const;
};

}  // namespace pcod_common
