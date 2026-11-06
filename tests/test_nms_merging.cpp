// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/nms.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

/** Construct a box with known, normalized two-class probabilities.
 * @param x Center X.
 * @param score Independent ranking score.
 * @param probability Probability of class zero.
 * @return Candidate with logits and independent existence probability.
 */
pcod_common::BoundingBox MakeBox(float x, float score, float probability) {
  pcod_common::BoundingBox box;
  box.center = {x, 0.0f};
  box.length = 4.0f;
  box.width = 2.0f;
  box.height = 2.0f;
  box.existence_probability = 0.7f;
  box.detection_score = score;
  box.classification = {{0, std::log(probability)}, {1, std::log(1.0f - probability)}};
  return box;
}

/** Recover a normalized probability from the public logit representation.
 * @param box Box whose classification is inspected.
 * @param index Model class index.
 * @return Softmax probability, or zero for an absent class.
 */
double Probability(const pcod_common::BoundingBox& box, std::size_t index) {
  double denominator = 0.0, numerator = 0.0;
  float maximum = -std::numeric_limits<float>::infinity();
  for (const auto& entry : box.classification) {
    maximum = std::max(maximum, entry.score);
  }
  for (const auto& entry : box.classification) {
    const double value = std::exp(static_cast<double>(entry.score) - maximum);
    denominator += value;
    if (entry.class_idx == index) {
      numerator += value;
    }
  }
  return numerator / denominator;
}

/** Independently reproduce ordinary greedy NMS for comparison with the optimized geometry.
 * @param boxes Original scored boxes.
 * @param config NMS configuration with merging disabled.
 * @return Retained box X coordinates in rank order.
 */
std::vector<float> ReferenceNms(std::vector<pcod_common::BoundingBox> boxes, const pcod_common::NmsConfig& config) {
  std::sort(boxes.begin(), boxes.end(), [](const auto& a, const auto& b) { return *a.detection_score > *b.detection_score; });
  std::vector<pcod_common::BoundingBox> kept;
  for (const auto& box : boxes) {
    bool suppressed = false;
    const bool class_zero = Probability(box, 0) > 0.5;
    for (const auto& other : kept) {
      if (class_zero == (Probability(other, 0) > 0.5) && box.overlaps(other, config.iou_threshold)) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      kept.push_back(box);
      if (kept.size() == static_cast<std::size_t>(config.max_detections)) {
        break;
      }
    }
  }
  std::vector<float> coordinates;
  for (const auto& box : kept) {
    coordinates.push_back(box.center[0]);
  }
  return coordinates;
}

}  // namespace

/** Exercise class-independent fusion, representative membership, limits, numeric stability and legacy parity. */
int main() {
  // Existing aggregate initialization and function-pointer signatures remain valid.
  pcod_common::NmsConfig legacy_config{{0.1f}, 0.2f, 64, 0.5f};
  assert(!legacy_config.merge_enabled && legacy_config.internal_score_threshold == 0.5f);
  void (*legacy_api)(std::vector<pcod_common::BoundingBox>&, const pcod_common::NmsConfig&) = pcod_common::ApplyRotatedNms;
  std::vector<pcod_common::BoundingBox> empty;
  legacy_api(empty, legacy_config);
  pcod_common::NmsConfig config;
  config.score_thresholds = {0.0f};
  config.merge_enabled = true;
  config.merge_score_threshold = 0.8f;
  config.iou_threshold = 0.2f;
  pcod_common::NmsStats stats;

  {
    // Containment and size similarity share a single class-independent score:
    // a half-volume contained prediction has IoU 0.5 but score sqrt(0.5).
    auto contained = config;
    contained.merge_score_threshold = 0.7f;
    contained.iou_threshold = 1.0f;
    auto large = MakeBox(0.0f, 0.9f, 0.9f), small = MakeBox(0.0f, 0.8f, 0.1f);
    small.length *= 0.5f;
    std::vector<pcod_common::BoundingBox> boxes{large, small};
    pcod_common::ApplyRotatedNms(boxes, contained, &stats);
    assert(boxes.size() == 1 && stats.merged_boxes == 1);
    // Reverse the ranking and winning classes: geometry alone decides membership.
    large.detection_score = 0.7f;
    boxes = {small, large};
    pcod_common::ApplyRotatedNms(boxes, contained, &stats);
    assert(boxes.size() == 1 && boxes[0].length == small.length);
    // A tiny contained object must survive, including when it has the same class.
    small.length = large.length * 0.01f;
    for (float probability : {0.1f, 0.9f}) {
      small.classification = MakeBox(0.0f, 0.8f, probability).classification;
      boxes = {large, small};
      pcod_common::ApplyRotatedNms(boxes, contained, &stats);
      assert(boxes.size() == 2 && stats.merged_boxes == 0 && stats.iou_evaluations == 0);
    }
    // Strict boundary: I/sqrt(Va*Vb) is exactly 0.5 for quarter volume.
    contained.merge_score_threshold = 0.5f;
    small.length = large.length * 0.25f;
    boxes = {large, small};
    pcod_common::ApplyRotatedNms(boxes, contained, &stats);
    assert(boxes.size() == 2 && stats.merged_boxes == 0);
    contained.merge_score_threshold = 0.0f;
    small.center[0] = 20.0f;
    boxes = {large, small};
    pcod_common::ApplyRotatedNms(boxes, contained);
    assert(boxes.size() == 2);
  }
  {
    const auto representative = MakeBox(0.0f, 0.9f, 0.8f);
    std::vector<pcod_common::BoundingBox> boxes{representative, MakeBox(0.1f, 0.6f, 0.2f)};
    pcod_common::ApplyRotatedNms(boxes, config, &stats);
    assert(boxes.size() == 1 && stats.merged_boxes == 1 && stats.iou_evaluations == 1);
    assert(std::abs(Probability(boxes[0], 0) - 0.56) < 1e-6);
    assert(std::abs(Probability(boxes[0], 1) - 0.44) < 1e-6);
    assert(boxes[0].center == representative.center && boxes[0].yaw == representative.yaw);
    assert(boxes[0].existence_probability == representative.existence_probability);
    assert(boxes[0].detection_score == representative.detection_score);
  }
  {
    // A member's full distribution can change the winner, but not the seed's
    // original class used for ordinary suppression of later candidates.
    std::vector<pcod_common::BoundingBox> boxes{MakeBox(0.0f, 0.9f, 0.51f), MakeBox(0.0f, 0.8f, 0.01f),
                                                MakeBox(1.0f, 0.7f, 0.9f)};
    pcod_common::ApplyRotatedNms(boxes, config);
    assert(boxes.size() == 1 && Probability(boxes[0], 1) > 0.5);
  }
  {
    // Membership is always compared with the original seed, not a moving box
    // or a transitive connected component. A-B and B-C overlap, A-C do not.
    auto independent = config;
    independent.iou_threshold = 1.0f;
    independent.merge_score_threshold = 0.8f;
    std::vector<pcod_common::BoundingBox> boxes{MakeBox(0.0f, 0.9f, 0.9f), MakeBox(0.5f, 0.8f, 0.1f), MakeBox(1.0f, 0.7f, 0.1f)};
    pcod_common::ApplyRotatedNms(boxes, independent);
    assert(boxes.size() == 2 && boxes[1].center[0] == 1.0f);
  }
  {
    // The output cap must not prevent later members from contributing.
    auto capped = config;
    capped.max_detections = 1;
    std::vector<pcod_common::BoundingBox> boxes{MakeBox(0.0f, 0.9f, 0.8f), MakeBox(20.0f, 0.8f, 0.9f), MakeBox(0.0f, 0.6f, 0.2f)};
    pcod_common::ApplyRotatedNms(boxes, capped);
    assert(boxes.size() == 1 && std::abs(Probability(boxes[0], 0) - 0.56) < 1e-6);
    capped.max_detections = 0;
    pcod_common::ApplyRotatedNms(boxes, capped, &stats);
    assert(boxes.empty() && stats.candidates == 0 && stats.merged_boxes == 0);
  }
  {
    // Filter by each candidate's original winning class before merging.
    auto filtered = config;
    filtered.score_thresholds = {0.0f, 0.7f};
    std::vector<pcod_common::BoundingBox> boxes{MakeBox(0.0f, 0.9f, 0.8f), MakeBox(0.0f, 0.6f, 0.2f)};
    pcod_common::ApplyRotatedNms(boxes, filtered, &stats);
    assert(boxes.size() == 1 && stats.candidates == 1 && stats.merged_boxes == 0);
    assert(std::abs(Probability(boxes[0], 0) - 0.8) < 1e-6);
  }
  {
    std::vector<pcod_common::BoundingBox> boxes{MakeBox(0.0f, 0.0f, 0.8f), MakeBox(0.0f, 0.0f, 0.2f)};
    pcod_common::ApplyRotatedNms(boxes, config);
    assert(boxes.size() == 1 && std::abs(Probability(boxes[0], 0) - 0.5) < 1e-6);
  }
  {
    // Stable softmax must handle large logits, shuffled/sparse class indices,
    // multiple members and zero-probability classes without double softmax.
    auto a = MakeBox(0.0f, 0.9f, 0.8f), b = a;
    a.classification = {{3, 1000.0f}, {7, 999.0f}, {9, -std::numeric_limits<float>::infinity()}};
    b.classification = {{7, -999.0f}, {3, -1000.0f}};
    b.detection_score = 0.3f;
    std::vector<pcod_common::BoundingBox> boxes{a, b, b};
    pcod_common::ApplyRotatedNms(boxes, config);
    const double expected = (0.9 / (1.0 + std::exp(-1.0)) + 0.6 / (1.0 + std::exp(1.0))) / 1.5;
    assert(boxes.size() == 1 && std::abs(Probability(boxes[0], 3) - expected) < 1e-6);
    assert(Probability(boxes[0], 9) == 0.0);
  }
  {
    auto a = MakeBox(0.0f, 0.9f, 0.8f), b = MakeBox(0.0f, 0.8f, 0.2f);
    b.z = 3.0f;
    std::vector<pcod_common::BoundingBox> boxes{a, b};
    pcod_common::ApplyRotatedNms(boxes, config, &stats);
    assert(boxes.size() == 2 && stats.iou_evaluations == 0);
    b.z = 0.0f;
    b.height = 0.0f;
    boxes = {a, b};
    pcod_common::ApplyRotatedNms(boxes, config);
    assert(boxes.size() == 2);
    auto strict = config;
    strict.merge_score_threshold = 1.0f;
    b.height = a.height;
    boxes = {a, b};
    pcod_common::ApplyRotatedNms(boxes, strict);
    assert(boxes.size() == 2);
  }
  {
    // Representatives retain heading and valid velocity even across a pi yaw
    // ambiguity; no averaging of headings or class-specific geometry occurs.
    auto a = MakeBox(0.0f, 0.9f, 0.8f), b = MakeBox(0.0f, 0.8f, 0.2f);
    a.yaw = 0.4f;
    a.has_velocity = true;
    a.v_x = 3.0f;
    b.yaw = a.yaw + static_cast<float>(M_PI);
    std::vector<pcod_common::BoundingBox> boxes{a, b};
    pcod_common::ApplyRotatedNms(boxes, config);
    assert(boxes.size() == 1 && boxes[0].yaw == a.yaw && boxes[0].has_velocity && boxes[0].v_x == 3.0f);
  }
  {
    auto invalid = config;
    invalid.merge_score_threshold = std::numeric_limits<float>::quiet_NaN();
    std::vector<pcod_common::BoundingBox> boxes;
    bool rejected = false;
    try {
      pcod_common::ApplyRotatedNms(boxes, invalid);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
    boxes = {MakeBox(0.0f, 0.9f, 0.8f), MakeBox(0.0f, 0.8f, 0.2f)};
    boxes[1].classification[0].score = std::numeric_limits<float>::quiet_NaN();
    rejected = false;
    try {
      pcod_common::ApplyRotatedNms(boxes, config);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  {
    // Verify the broad-phase bounds and fixed-buffer clipping against the
    // existing public geometry, across random rotations and positive heights.
    std::mt19937 random(42);
    std::uniform_real_distribution<float> coordinate(-10.0f, 10.0f), yaw(-3.14f, 3.14f), size(0.5f, 6.0f);
    auto legacy = config;
    legacy.merge_enabled = false;
    legacy.max_detections = 64;
    for (int trial = 0; trial < 100; ++trial) {
      std::vector<pcod_common::BoundingBox> boxes;
      for (int i = 0; i < 80; ++i) {
        auto box = MakeBox(coordinate(random), 1.0f - i * 0.01f, i % 2 == 0 ? 0.8f : 0.2f);
        box.center[1] = coordinate(random);
        box.z = coordinate(random) * 0.2f;
        box.length = size(random);
        box.width = size(random);
        box.height = size(random);
        box.yaw = yaw(random);
        boxes.push_back(box);
      }
      const auto expected = ReferenceNms(boxes, legacy);
      pcod_common::ApplyRotatedNms(boxes, legacy);
      assert(boxes.size() == expected.size());
      for (std::size_t i = 0; i < boxes.size(); ++i) {
        assert(boxes[i].center[0] == expected[i]);
      }
    }
  }
  {
    // Independently check the squared decision and broad-phase rejection using
    // public polygon geometry, random rotations, dimensions and height offsets.
    std::mt19937 random(57);
    std::uniform_real_distribution<float> coordinate(-3.0f, 3.0f), yaw(-3.14f, 3.14f), size(0.5f, 6.0f);
    for (int trial = 0; trial < 300; ++trial) {
      auto a = MakeBox(coordinate(random), 0.9f, 0.9f), b = MakeBox(coordinate(random), 0.8f, 0.1f);
      for (auto* box : {&a, &b}) {
        box->center[1] = coordinate(random);
        box->z = coordinate(random) * 0.2f;
        box->length = size(random);
        box->width = size(random);
        box->height = size(random);
        box->yaw = yaw(random);
      }
      const double height = std::max(
          0.0, std::min(a.z + a.height * 0.5, b.z + b.height * 0.5) - std::max(a.z - a.height * 0.5, b.z - b.height * 0.5));
      const double volume_a = static_cast<double>(a.length) * a.width * a.height;
      const double volume_b = static_cast<double>(b.length) * b.width * b.height;
      const double reference = a.intersection_area(b) * height / std::sqrt(volume_a * volume_b);
      for (float threshold : {0.1f, 0.3f, 0.7f}) {
        if (std::abs(reference - threshold) < 1e-4) {
          continue;
        }
        auto checked = config;
        checked.merge_score_threshold = threshold;
        std::vector<pcod_common::BoundingBox> boxes{a, b};
        pcod_common::ApplyRotatedNms(boxes, checked);
        assert(boxes.size() == (reference > threshold ? 1u : 2u));
      }
    }
  }
  return 0;
}
