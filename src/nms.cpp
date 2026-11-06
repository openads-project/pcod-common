// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/nms.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace pcod_common {
namespace {

/** XY point with double precision for intersection calculations. */
using Point = std::array<double, 2>;

/** Geometry cached once per candidate, shared by NMS and merging. */
struct PreparedBox {
  std::array<Point, 4> vertices;  ///< Counterclockwise footprint corners.
  Point minimum;                  ///< Axis-aligned XY minimum.
  Point maximum;                  ///< Axis-aligned XY maximum.
  double bottom;                  ///< Lower Z bound.
  double top;                     ///< Upper Z bound.
  double area;                    ///< Footprint area.
  bool has_height;                ///< Whether a positive height is available.

  /** Cache geometry without allocating per-pair polygon buffers.
   * @param box Candidate geometry.
   */
  explicit PreparedBox(const BoundingBox& box)
      : minimum{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()},
        maximum{-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()},
        bottom(box.z - box.height * 0.5),
        top(box.z + box.height * 0.5),
        area(static_cast<double>(box.length) * box.width),
        has_height(box.height > 0.0f) {
    const double c = std::cos(box.yaw), s = std::sin(box.yaw);
    const double dx = box.length * 0.5, dy = box.width * 0.5;
    vertices = {{{box.center[0] - dx * c + dy * s, box.center[1] - dx * s - dy * c},
                 {box.center[0] + dx * c + dy * s, box.center[1] + dx * s - dy * c},
                 {box.center[0] + dx * c - dy * s, box.center[1] + dx * s + dy * c},
                 {box.center[0] - dx * c - dy * s, box.center[1] - dx * s + dy * c}}};
    for (const auto& p : vertices) {
      for (std::size_t k = 0; k < 2; ++k) {
        minimum[k] = std::min(minimum[k], p[k]);
        maximum[k] = std::max(maximum[k], p[k]);
      }
    }
  }
};

/** Evaluate the left half-plane of an oriented edge.
 * @param a Edge start.
 * @param b Edge end.
 * @param p Point to evaluate.
 * @return Signed cross product.
 */
double HalfPlane(const Point& a, const Point& b, const Point& p) {
  return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
}

/** Clip two convex footprints using fixed-size scratch buffers.
 * @param a First cached box.
 * @param b Second cached box.
 * @return Intersection area.
 */
double IntersectionArea(const PreparedBox& a, const PreparedBox& b) {
  std::array<Point, 16> polygon{}, output{};
  std::copy(a.vertices.begin(), a.vertices.end(), polygon.begin());
  std::size_t count = 4;
  for (std::size_t edge = 0; edge < 4 && count > 2; ++edge) {
    const auto& start = b.vertices[edge];
    const auto& end = b.vertices[(edge + 1) % 4];
    std::size_t next_count = 0;
    Point previous = polygon[count - 1];
    double previous_distance = HalfPlane(start, end, previous);
    for (std::size_t i = 0; i < count; ++i) {
      const Point current = polygon[i];
      const double distance = HalfPlane(start, end, current);
      if ((distance >= 0.0) != (previous_distance >= 0.0)) {
        const double t = previous_distance / (previous_distance - distance);
        output[next_count++] = {previous[0] + t * (current[0] - previous[0]), previous[1] + t * (current[1] - previous[1])};
      }
      if (distance >= 0.0) {
        output[next_count++] = current;
      }
      previous = current;
      previous_distance = distance;
    }
    polygon.swap(output);
    count = next_count;
  }
  if (count < 3) {
    return 0.0;
  }
  double area = 0.0;
  for (std::size_t i = 1; i + 1 < count; ++i) {
    area += HalfPlane(polygon[0], polygon[i], polygon[i + 1]);
  }
  return std::abs(area) * 0.5;
}

/** Results shared by ordinary IoU suppression and class-independent merging. */
struct PairOverlap {
  double iou = 0.0;    ///< Ordinary intersection over union.
  bool merge = false;  ///< Whether the geometric-mean-volume score exceeds its threshold.
};

/** Evaluate overlap only when inexpensive bounds can affect NMS or merging.
 * @param a First cached box.
 * @param b Second cached box.
 * @param nms_threshold Ordinary IoU threshold, or one to disable suppression.
 * @param merge_threshold Geometric-mean-volume score threshold, or one to disable merging.
 * @param stats Operation counts.
 * @return Ordinary IoU and the independent merge decision.
 */
PairOverlap EvaluateOverlap(
    const PreparedBox& a, const PreparedBox& b, float nms_threshold, float merge_threshold, NmsStats& stats) {
  double area_bound = std::min(a.area, b.area);
  double rectangle_area = 1.0;
  for (std::size_t k = 0; k < 2; ++k) {
    const double extent = std::min(a.maximum[k], b.maximum[k]) - std::max(a.minimum[k], b.minimum[k]);
    if (extent <= 0.0) {
      return {};
    }
    rectangle_area *= extent;
  }
  area_bound = std::min(area_bound, rectangle_area);
  double overlap_height = 1.0, volume_a = a.area, volume_b = b.area;
  const bool has_height = a.has_height && b.has_height;
  if (has_height) {
    overlap_height = std::min(a.top, b.top) - std::max(a.bottom, b.bottom);
    if (overlap_height <= 0.0) {
      return {};
    }
    volume_a *= a.top - a.bottom;
    volume_b *= b.top - b.bottom;
  }
  if (!(volume_a > 0.0 && volume_b > 0.0) || !std::isfinite(volume_a) || !std::isfinite(volume_b)) {
    return {};
  }
  const double intersection_bound = area_bound * overlap_height;
  const double merge_bound = static_cast<double>(merge_threshold) * merge_threshold * volume_a * volume_b;
  const bool possible_suppression = intersection_bound > nms_threshold * (volume_a + volume_b - intersection_bound);
  const bool possible_merge = has_height && intersection_bound * intersection_bound > merge_bound;
  if (intersection_bound <= 0.0 || (!possible_suppression && !possible_merge)) {
    return {};
  }
  ++stats.iou_evaluations;
  const double intersection = std::min(IntersectionArea(a, b), area_bound) * overlap_height;
  return {intersection / std::max(volume_a + volume_b - intersection, 1e-7),
          has_height && intersection * intersection > merge_bound};
}

/** Find the winning class once, before any semantic fusion.
 * @param box Candidate box with class logits.
 * @return Winning model class index.
 */
std::size_t BestClass(const BoundingBox& box) {
  float best = -std::numeric_limits<float>::infinity();
  std::size_t index = 0;
  for (const auto& entry : box.classification) {
    if (entry.score > best) {
      best = entry.score;
      index = entry.class_idx;
    }
  }
  return index;
}

/** Weighted class distribution accumulated only for representatives that merge. */
struct ClassMixture {
  std::vector<ClassificationEntry> probabilities;  ///< Class-indexed weighted probability sums.
  double weight = 0.0;                             ///< Total detection-score weight.

  /** Add a full, independently normalized distribution without inflating confidence.
   * @param box Original box with class logits.
   * @param score Nonnegative mixture weight.
   */
  void add(const BoundingBox& box, double score) {
    if (box.classification.empty()) {
      throw std::invalid_argument("Box merging requires class logits");
    }
    float maximum = -std::numeric_limits<float>::infinity();
    for (const auto& entry : box.classification) {
      if (std::isnan(entry.score) || entry.score == std::numeric_limits<float>::infinity()) {
        throw std::invalid_argument("Box merging requires finite logits or negative infinity");
      }
      maximum = std::max(maximum, entry.score);
    }
    if (!std::isfinite(maximum)) {
      throw std::invalid_argument("Box merging requires at least one finite class logit");
    }
    double denominator = 0.0;
    for (const auto& entry : box.classification) {
      denominator += std::exp(static_cast<double>(entry.score) - maximum);
    }
    for (const auto& entry : box.classification) {
      auto existing = std::find_if(probabilities.begin(), probabilities.end(),
                                   [&](const ClassificationEntry& value) { return value.class_idx == entry.class_idx; });
      const float contribution = static_cast<float>(score * std::exp(static_cast<double>(entry.score) - maximum) / denominator);
      if (existing == probabilities.end()) {
        probabilities.push_back({entry.class_idx, contribution});
      } else {
        existing->score += contribution;
      }
    }
    weight += score;
  }

  /** Store log probabilities for the existing ROS softmax conversion.
   * @param box Representative whose other attributes are unchanged.
   */
  void finish(BoundingBox& box) {
    box.classification = std::move(probabilities);
    std::sort(box.classification.begin(), box.classification.end(),
              [](const ClassificationEntry& a, const ClassificationEntry& b) { return a.class_idx < b.class_idx; });
    for (auto& entry : box.classification) {
      entry.score = static_cast<float>(std::log(entry.score / weight));
    }
  }
};

/** Score-ranked candidate with cached original semantics and geometry. */
struct Candidate {
  float score;            ///< Ranking and mixture score.
  std::size_t index;      ///< Original input index, also the deterministic tie breaker.
  std::size_t class_idx;  ///< Original winning class.
  PreparedBox geometry;   ///< Cached geometry.
};

}  // namespace

void ApplyRotatedNms(std::vector<BoundingBox>& bboxes, const NmsConfig& config) { ApplyRotatedNms(bboxes, config, nullptr); }

void ApplyRotatedNms(std::vector<BoundingBox>& bboxes, const NmsConfig& config, NmsStats* stats) {
  NmsStats counts;
  if (stats != nullptr) {
    *stats = counts;
  }
  if (config.merge_enabled && (!std::isfinite(config.merge_score_threshold) || config.merge_score_threshold < 0.0f ||
                               config.merge_score_threshold > 1.0f)) {
    throw std::invalid_argument("Box merging score threshold must be finite and in [0, 1]");
  }
  if (config.max_detections <= 0) {
    bboxes.clear();
    return;
  }
  std::vector<Candidate> scored;
  scored.reserve(bboxes.size());
  for (std::size_t i = 0; i < bboxes.size(); ++i) {
    const auto& box = bboxes[i];
    const std::size_t class_idx = BestClass(box);
    const float threshold = config.score_thresholds.empty()
                                ? config.internal_score_threshold
                                : config.score_thresholds[std::min(class_idx, config.score_thresholds.size() - 1)];
    const float score = box.detection_score.value_or(box.existence_probability);
    if (std::isfinite(score) && score >= threshold) {
      scored.push_back({score, i, class_idx, PreparedBox(box)});
    }
  }
  counts.candidates = scored.size();
  std::sort(scored.begin(), scored.end(), [](const Candidate& a, const Candidate& b) {
    return a.score > b.score || (a.score == b.score && a.index < b.index);
  });
  std::vector<BoundingBox> kept;
  std::vector<std::size_t> representatives;
  std::vector<ClassMixture> mixtures;
  const std::size_t capacity = std::min(scored.size(), static_cast<std::size_t>(config.max_detections));
  kept.reserve(capacity);
  representatives.reserve(capacity);
  if (config.merge_enabled) {
    mixtures.resize(capacity);
  }
  for (std::size_t i = 0; i < scored.size(); ++i) {
    const auto& candidate = scored[i];
    bool suppressed = false, merged = false;
    for (std::size_t k = 0; k < kept.size(); ++k) {
      const auto& representative = scored[representatives[k]];
      const bool same_class = candidate.class_idx == representative.class_idx;
      const bool can_merge = config.merge_enabled && candidate.geometry.has_height && representative.geometry.has_height;
      if (!same_class && !can_merge) {
        continue;
      }
      const auto overlap = EvaluateOverlap(candidate.geometry, representative.geometry, same_class ? config.iou_threshold : 1.0f,
                                           can_merge ? config.merge_score_threshold : 1.0f, counts);
      if (can_merge && overlap.merge) {
        auto& mixture = mixtures[k];
        // All-zero clusters use uniform weights. Once any positive score is
        // present, zero-score members contribute no probability mass.
        const double weight = representative.score > 0.0f ? std::max(0.0f, candidate.score) : 1.0;
        if (mixture.probabilities.empty()) {
          mixture.add(bboxes[representative.index], representative.score > 0.0f ? representative.score : 1.0);
        }
        mixture.add(bboxes[candidate.index], weight);
        ++counts.merged_boxes;
        merged = true;
        break;
      }
      if (same_class && overlap.iou > config.iou_threshold) {
        suppressed = true;
        if (!config.merge_enabled) {
          break;
        }
      }
    }
    if (!merged && !suppressed && kept.size() < capacity) {
      kept.push_back(bboxes[candidate.index]);
      representatives.push_back(i);
      if (!config.merge_enabled && kept.size() == capacity) {
        break;
      }
    }
  }
  for (std::size_t k = 0; k < kept.size() && config.merge_enabled; ++k) {
    if (!mixtures[k].probabilities.empty()) {
      mixtures[k].finish(kept[k]);
    }
  }
  bboxes = std::move(kept);
  if (stats != nullptr) {
    *stats = counts;
  }
}

}  // namespace pcod_common
