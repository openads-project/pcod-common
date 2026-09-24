// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/nms.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

/** Build a scored bounding box whose unused velocity field stores the test index.
 * @param idx Test index stored in the unused velocity field.
 * @param x Box center X.
 * @param y Box center Y.
 * @param length Box length.
 * @param width Box width.
 * @param yaw Box heading.
 * @param score Detection score.
 * @return Configured bounding box.
 */
pcod_common::BoundingBox MakeBox(std::size_t idx, float x, float y, float length, float width, float yaw, float score) {
  pcod_common::BoundingBox box;
  box.center = {x, y};
  box.z = 0.0f;
  box.v_x = static_cast<float>(idx);
  box.length = length;
  box.width = width;
  box.height = 1.0f;
  box.yaw = yaw;
  box.existence_probability = score;
  box.classification.push_back({0, 1.0f});
  return box;
}

/** Run one NMS scenario and compare retained box indices.
 * @param boxes Candidate boxes.
 * @param iou_threshold Suppression IoU threshold.
 * @param max_detections Maximum retained detections.
 * @param expected_indices Expected retained test indices.
 */
void RunNmsCase(std::vector<pcod_common::BoundingBox> boxes,
                float iou_threshold,
                int max_detections,
                const std::vector<std::size_t>& expected_indices) {
  pcod_common::NmsConfig config;
  config.score_thresholds = {0.1f};
  config.iou_threshold = iou_threshold;
  config.max_detections = max_detections;

  pcod_common::ApplyRotatedNms(boxes, config);

  assert(boxes.size() == expected_indices.size());
  for (std::size_t i = 0; i < expected_indices.size(); ++i) {
    assert(static_cast<std::size_t>(std::lround(boxes[i].v_x)) == expected_indices[i]);
  }
}

}  // namespace

/** Run rotated NMS regression checks. */
int main() {
  {
    pcod_common::BoundingBox box_a;
    box_a.center = {0.0f, 0.0f};
    box_a.length = 4.0f;
    box_a.width = 2.0f;
    box_a.existence_probability = 0.9f;
    box_a.classification.push_back({0, 1.0f});

    pcod_common::BoundingBox box_b = box_a;
    box_b.center = {0.5f, 0.0f};
    box_b.existence_probability = 0.8f;

    std::vector<pcod_common::BoundingBox> boxes{box_a, box_b};

    pcod_common::NmsConfig config;
    config.score_thresholds = {0.1f};
    config.iou_threshold = 0.1f;
    config.max_detections = 10;

    pcod_common::ApplyRotatedNms(boxes, config);
    assert(boxes.size() == 1);
    assert(std::abs(boxes[0].center[0]) < 1e-6f);
    assert(std::abs(boxes[0].center[1]) < 1e-6f);
  }

  {
    RunNmsCase(
        {
            MakeBox(0, 0.0f, 0.0f, 4.0f, 2.0f, 0.0f, 0.9f),
            MakeBox(1, 0.0f, 0.0f, 4.0f, 2.0f, 0.0f, 0.8f),
        },
        0.1f, 10, {0});

    RunNmsCase(
        {
            MakeBox(0, 0.0f, 0.0f, 4.0f, 2.0f, 0.0f, 0.9f),
            MakeBox(1, 0.5f, 0.0f, 4.0f, 2.0f, 0.0f, 0.8f),
        },
        0.1f, 10, {0});

    RunNmsCase(
        {
            MakeBox(0, 0.0f, 0.0f, 4.0f, 2.0f, 0.0f, 0.9f),
            MakeBox(1, 3.5f, 0.0f, 4.0f, 2.0f, 0.0f, 0.8f),
        },
        0.1f, 10, {0, 1});

    RunNmsCase(
        {
            MakeBox(0, 0.0f, 0.0f, 4.0f, 2.0f, 0.0f, 0.9f),
            MakeBox(1, 4.0f, 0.0f, 4.0f, 2.0f, 0.0f, 0.8f),
        },
        0.0f, 10, {0, 1});

    RunNmsCase(
        {
            MakeBox(0, 0.0f, 0.0f, 6.0f, 4.0f, 0.0f, 0.9f),
            MakeBox(1, 0.0f, 0.0f, 2.0f, 1.0f, 0.0f, 0.8f),
        },
        0.05f, 10, {0});

    RunNmsCase(
        {
            MakeBox(0, 0.0f, 0.0f, 4.0f, 2.0f, static_cast<float>(M_PI / 4.0), 0.9f),
            MakeBox(1, 0.2f, 0.1f, 4.0f, 2.0f, static_cast<float>(M_PI / 4.0), 0.8f),
        },
        0.1f, 10, {0});

    RunNmsCase(
        {
            MakeBox(0, 0.0f, 0.0f, 4.0f, 2.0f, static_cast<float>(M_PI / 4.0), 0.9f),
            MakeBox(1, 0.0f, 0.0f, 4.0f, 2.0f, static_cast<float>(-M_PI / 4.0), 0.8f),
        },
        0.1f, 10, {0});

    RunNmsCase(
        {
            MakeBox(0, 0.0f, 0.0f, 4.0f, 2.0f, static_cast<float>(M_PI / 4.0), 0.9f),
            MakeBox(1, 4.0f, 4.0f, 4.0f, 2.0f, static_cast<float>(M_PI / 4.0), 0.8f),
        },
        0.1f, 10, {0, 1});

    RunNmsCase(
        {
            MakeBox(0, 0.0f, 0.0f, 4.0f, 2.0f, 0.0f, 0.8f),
            MakeBox(1, 0.5f, 0.0f, 4.0f, 2.0f, 0.0f, 0.95f),
            MakeBox(2, 8.0f, 0.0f, 4.0f, 2.0f, 0.0f, 0.7f),
        },
        0.1f, 10, {1, 2});

    RunNmsCase(
        {
            MakeBox(0, 0.0f, 0.0f, 4.0f, 2.0f, 0.0f, 0.9f),
            MakeBox(1, 8.0f, 0.0f, 4.0f, 2.0f, 0.0f, 0.8f),
            MakeBox(2, 16.0f, 0.0f, 4.0f, 2.0f, 0.0f, 0.7f),
        },
        0.1f, 2, {0, 1});
  }

  {
    pcod_common::BoundingBox keep_box;
    keep_box.center = {0.0f, 0.0f};
    keep_box.length = 1.0f;
    keep_box.width = 1.0f;
    keep_box.existence_probability = 0.6f;
    keep_box.classification.push_back({0, 0.9f});
    keep_box.classification.push_back({1, 0.1f});

    pcod_common::BoundingBox drop_box;
    drop_box.center = {10.0f, 10.0f};  // ensure no IoU suppression interaction
    drop_box.length = 1.0f;
    drop_box.width = 1.0f;
    drop_box.existence_probability = 0.4f;
    drop_box.classification.push_back({0, 0.1f});
    drop_box.classification.push_back({1, 0.9f});

    std::vector<pcod_common::BoundingBox> boxes{keep_box, drop_box};
    pcod_common::NmsConfig config;
    config.score_thresholds = {0.1f, 0.9f};
    config.iou_threshold = 0.1f;
    config.max_detections = 10;
    config.internal_score_threshold = 0.5f;

    pcod_common::ApplyRotatedNms(boxes, config);
    assert(boxes.size() == 1);
    assert(std::abs(boxes[0].center[0]) < 1e-6f);
    assert(std::abs(boxes[0].center[1]) < 1e-6f);
  }

  return 0;
}
