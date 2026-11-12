// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <vector>

namespace pcod_common {

/** Class prediction associated with a detected bounding box. */
struct ClassificationEntry {
  std::size_t class_idx;  ///< Zero-based model class index.
  float score;            ///< Confidence assigned to the class.
};

/** Two-dimensional vertex used by rotated-box geometry operations. */
struct BoundingBoxVertex {
  float x;  ///< X coordinate.
  float y;  ///< Y coordinate.

  /** Construct a vertex at (@p x_in, @p y_in).
   * @param x_in X coordinate.
   * @param y_in Y coordinate.
   */
  BoundingBoxVertex(float x_in = 0.0f, float y_in = 0.0f) : x(x_in), y(y_in) {}

  /** @param v Vertex to add. @return Component-wise sum. */
  BoundingBoxVertex operator+(const BoundingBoxVertex& v) const { return {x + v.x, y + v.y}; }
  /** @param v Vertex to subtract. @return Component-wise difference. */
  BoundingBoxVertex operator-(const BoundingBoxVertex& v) const { return {x - v.x, y - v.y}; }
  /** @param v Other vector. @return Two-dimensional cross product. */
  float cross(const BoundingBoxVertex& v) const { return x * v.y - y * v.x; }
};

/** Implicit two-dimensional line equation `a*x + b*y + c = 0`. */
struct Line {
  float a;  ///< X coefficient.
  float b;  ///< Y coefficient.
  float c;  ///< Constant term.

  /** Construct the line passing through two vertices.
   * @param v1 First vertex.
   * @param v2 Second vertex.
   */
  Line(const BoundingBoxVertex& v1, const BoundingBoxVertex& v2) {
    a = v2.y - v1.y;
    b = v1.x - v2.x;
    c = v2.cross(v1);
  }

  /** @param p Evaluation point. @return Value of the line equation. */
  float operator()(const BoundingBoxVertex& p) const { return a * p.x + b * p.y + c; }

  /** @param other Non-parallel line. @return Intersection vertex. */
  BoundingBoxVertex intersection(const Line& other) const {
    float w = a * other.b - b * other.a;
    return BoundingBoxVertex((b * other.c - c * other.b) / w, (c * other.a - a * other.c) / w);
  }
};

/** Oriented three-dimensional detection box and its semantic attributes. */
struct BoundingBox {
  std::array<float, 2> center;                      ///< XY center in metres.
  float z = 0.0f;                                   ///< Z center in metres.
  float length = 0.0f;                              ///< Length along the local X axis.
  float width = 0.0f;                               ///< Width along the local Y axis.
  float height = 0.0f;                              ///< Height along the Z axis.
  float yaw = 0.0f;                                 ///< Heading in radians.
  float existence_probability = 0.0f;               ///< Probability that an object exists, independent of class.
  std::optional<float> detection_score;             ///< Class-weighted score for filtering and NMS, if available.
  std::vector<ClassificationEntry> classification;  ///< Ranked semantic predictions.
  bool has_velocity = false;                        ///< Whether velocity components are valid.
  float v_x = 0.0f;                                 ///< X velocity when @ref has_velocity is true.
  float v_y = 0.0f;                                 ///< Y velocity when @ref has_velocity is true.

  /** @param other Box to compare. @param iou_threshold Required IoU. @return Whether the boxes overlap sufficiently. */
  bool overlaps(const BoundingBox& other, float iou_threshold) const;
  /** @param other Box to compare. @return XY intersection area. */
  float intersection_area(const BoundingBox& other) const;
  /** Return the four XY vertices in rectangle order. */
  std::vector<BoundingBoxVertex> rectangle_vertices() const;
};

}  // namespace pcod_common
