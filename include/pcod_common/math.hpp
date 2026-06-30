// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>

namespace pcod_common {

template <typename T>
constexpr const inline T& clamp(const T& value, const T& low, const T& high) {
  return (value < low) ? low : (high < value) ? high : value;
}

template <typename T>
T scale_score(T score, T old_thresh, T new_thresh) {
  const T one = static_cast<T>(1);
  if (score <= old_thresh) {
    return score * new_thresh / old_thresh;
  }
  return one - (one - score) * (one - new_thresh) / (one - old_thresh);
}

inline float wrap_to_range(float val, float min_val, float max_val) {
  float range = max_val - min_val;
  return val - range * std::floor((val - min_val) / range);
}

}  // namespace pcod_common
