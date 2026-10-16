// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

torch::Tensor rotated_nms_cuda(torch::Tensor boxes, torch::Tensor scores, double iou_threshold, int64_t max_out);
torch::Tensor rotated_nms_cuda_batched(torch::Tensor sorted_boxes,
                                       torch::Tensor group_offsets,
                                       double iou_threshold,
                                       int64_t max_out);
torch::Tensor oriented_iou_aligned_cuda(torch::Tensor boxes_a, torch::Tensor boxes_b);

namespace {

using Point = std::array<float, 2>;
using Polygon = std::vector<Point>;

inline float cross(const Point& a, const Point& b) { return a[0] * b[1] - a[1] * b[0]; }

inline Point operator+(const Point& a, const Point& b) { return {a[0] + b[0], a[1] + b[1]}; }

inline Point operator-(const Point& a, const Point& b) { return {a[0] - b[0], a[1] - b[1]}; }

inline Point operator*(const Point& a, float t) { return {a[0] * t, a[1] * t}; }

Polygon rectangle_vertices(const at::Tensor& box) {
  // box: [x, y, z, l, w, h, yaw]
  const float cx = box[0].item<float>();
  const float cy = box[1].item<float>();
  const float length = box[3].item<float>();
  const float width = box[4].item<float>();
  const float yaw = box[6].item<float>();

  const float dx = length * 0.5f;
  const float dy = width * 0.5f;
  const float c = std::cos(yaw);
  const float s = std::sin(yaw);

  Polygon verts;
  verts.reserve(4);
  verts.push_back({cx - dx * c + dy * s, cy - dx * s - dy * c});
  verts.push_back({cx + dx * c + dy * s, cy + dx * s - dy * c});
  verts.push_back({cx + dx * c - dy * s, cy + dx * s + dy * c});
  verts.push_back({cx - dx * c - dy * s, cy - dx * s + dy * c});

  // --- enforce CCW winding ---
  float area2 = 0.0f;
  for (int i = 0; i < 4; i++) {
    const auto& p = verts[i];
    const auto& q = verts[(i + 1) % 4];
    area2 += p[0] * q[1] - p[1] * q[0];
  }
  if (area2 < 0.0f) {
    std::reverse(verts.begin(), verts.end());
  }

  return verts;
}

float polygon_area(const Polygon& poly) {
  if (poly.size() < 3) {
    return 0.0f;
  }
  float area = 0.0f;
  for (size_t i = 0; i < poly.size(); ++i) {
    const Point& p = poly[i];
    const Point& q = poly[(i + 1) % poly.size()];
    area += cross(p, q);
  }
  return 0.5f * std::abs(area);
}

Polygon clip_polygon(const Polygon& subject, const Point& edge_start, const Point& edge_end) {
  Polygon output;
  if (subject.empty()) {
    return output;
  }
  const Point edge = edge_end - edge_start;
  const auto eval = [&](const Point& p) {
    Point v = p - edge_start;
    return edge[0] * v[1] - edge[1] * v[0];  // cross(edge, v)
  };

  for (size_t i = 0; i < subject.size(); ++i) {
    const Point& curr = subject[i];
    const Point& prev = subject[(i + subject.size() - 1) % subject.size()];
    float curr_val = eval(curr);
    float prev_val = eval(prev);

    // For CCW polygons, interior is on the LEFT side => cross >= 0
    bool curr_inside = curr_val >= 0.0f;
    bool prev_inside = prev_val >= 0.0f;

    if (curr_inside && prev_inside) {
      output.push_back(curr);
    } else if (prev_inside && !curr_inside) {
      float t = prev_val / (prev_val - curr_val + 1e-12f);
      Point intersect = prev + (curr - prev) * t;
      output.push_back(intersect);
    } else if (!prev_inside && curr_inside) {
      float t = prev_val / (prev_val - curr_val + 1e-12f);
      Point intersect = prev + (curr - prev) * t;
      output.push_back(intersect);
      output.push_back(curr);
    }
  }
  return output;
}

Polygon polygon_intersection(const Polygon& a, const Polygon& b) {
  Polygon output = a;
  for (size_t i = 0; i < b.size(); ++i) {
    if (output.size() < 3) {
      break;
    }
    const Point& p1 = b[i];
    const Point& p2 = b[(i + 1) % b.size()];
    output = clip_polygon(output, p1, p2);
  }
  return output;
}

float oriented_iou(const at::Tensor& box_a, const at::Tensor& box_b) {
  Polygon poly_a = rectangle_vertices(box_a);
  Polygon poly_b = rectangle_vertices(box_b);
  Polygon inter = polygon_intersection(poly_a, poly_b);
  float inter_area = polygon_area(inter);
  if (inter_area <= 0.0f) {
    return 0.0f;
  }
  float area_a = polygon_area(poly_a);
  float area_b = polygon_area(poly_b);
  float uni = area_a + area_b - inter_area;
  return inter_area / std::max(uni, 1e-6f);
}

}  // namespace

// boxes: [N,7], scores: [N], return indices to keep
torch::Tensor rotated_nms(torch::Tensor boxes, torch::Tensor scores, double iou_threshold, int64_t max_out) {
  TORCH_CHECK(boxes.dim() == 2 && boxes.size(1) == 7, "boxes must be [N,7]");
  TORCH_CHECK(scores.dim() == 1 && scores.size(0) == boxes.size(0), "scores must be [N]");

  auto boxes_cpu = boxes.contiguous().to(torch::kCPU);
  auto scores_cpu = scores.contiguous().to(torch::kCPU);
  auto order = std::get<1>(scores_cpu.sort(0, /*descending=*/true));

  std::vector<int64_t> keep;
  keep.reserve(boxes_cpu.size(0));

  for (int64_t oi = 0; oi < order.size(0); ++oi) {
    int64_t idx = order[oi].item<int64_t>();
    bool suppress = false;
    for (int64_t kept_idx : keep) {
      float iou = oriented_iou(boxes_cpu[idx], boxes_cpu[kept_idx]);
      const auto a = boxes_cpu[idx];
      const auto b = boxes_cpu[kept_idx];
      const float ha = a[5].item<float>();
      const float hb = b[5].item<float>();
      if (ha > 0 && hb > 0) {
        const float aa = a[3].item<float>() * a[4].item<float>();
        const float ab = b[3].item<float>() * b[4].item<float>();
        const float za = a[2].item<float>();
        const float zb = b[2].item<float>();
        const float inter_h = std::max(0.0f, std::min(za + ha / 2, zb + hb / 2) - std::max(za - ha / 2, zb - hb / 2));
        const float inter = iou * (aa + ab) / (1 + iou) * inter_h;
        iou = inter / std::max(aa * ha + ab * hb - inter, 1e-7f);
      }
      if (iou > static_cast<float>(iou_threshold)) {
        suppress = true;
        break;
      }
    }
    if (!suppress) {
      keep.push_back(idx);
      if (static_cast<int64_t>(keep.size()) >= max_out) {
        break;
      }
    }
  }

  auto options = torch::TensorOptions().dtype(torch::kLong).device(boxes.device());
  return torch::from_blob(keep.data(), {static_cast<long>(keep.size())}, options).clone();
}

torch::Tensor oriented_iou_aligned(torch::Tensor boxes_a, torch::Tensor boxes_b) {
  TORCH_CHECK(boxes_a.dim() == 2 && boxes_a.size(1) == 7, "boxes_a must be [N,7]");
  TORCH_CHECK(boxes_b.dim() == 2 && boxes_b.size(1) == 7, "boxes_b must be [N,7]");
  TORCH_CHECK(boxes_a.sizes() == boxes_b.sizes(), "boxes_a and boxes_b must have the same shape");
  auto boxes_a_c = boxes_a.contiguous().to(torch::kCPU);
  auto boxes_b_c = boxes_b.contiguous().to(torch::kCPU);
  const auto N = boxes_a_c.size(0);
  auto out = torch::zeros({N}, boxes_a.options().device(torch::kCPU).dtype(torch::kFloat));
  for (int64_t i = 0; i < N; ++i) {
    out[i] = oriented_iou(boxes_a_c[i], boxes_b_c[i]);
  }
  return out.to(boxes_a.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rotated_nms", &rotated_nms, "Rotated NMS (CPU)");
  m.def("rotated_nms_cuda", &rotated_nms_cuda, "Rotated NMS (CUDA)");
  m.def("rotated_nms_cuda_batched", &rotated_nms_cuda_batched, "Batched rotated NMS (CUDA)");
  m.def("oriented_iou_aligned", &oriented_iou_aligned, "Aligned oriented IoU (CPU)");
  m.def("oriented_iou_aligned_cuda", &oriented_iou_aligned_cuda, "Aligned oriented IoU (CUDA)");
}
