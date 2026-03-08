#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAMacros.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

namespace {

struct Vec2 {
  float x;
  float y;
};

__device__ __forceinline__ Vec2 make_vec2(float x, float y) {
  Vec2 v;
  v.x = x;
  v.y = y;
  return v;
}

__device__ __forceinline__ Vec2 operator+(const Vec2& a, const Vec2& b) {
  return make_vec2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ Vec2 operator-(const Vec2& a, const Vec2& b) {
  return make_vec2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ Vec2 operator*(const Vec2& a, float t) {
  return make_vec2(a.x * t, a.y * t);
}

__device__ __forceinline__ float cross(const Vec2& a, const Vec2& b) {
  return a.x * b.y - a.y * b.x;
}

__device__ __forceinline__ void box_to_corners(const float* box, Vec2 corners[4]) {
  // box: [x, y, z, l, w, h, yaw] (only BEV terms used)
  const float cx = box[0];
  const float cy = box[1];
  const float length = box[3];
  const float width = box[4];
  const float yaw = box[6];

  const float dx = length * 0.5f;
  const float dy = width * 0.5f;
  const float c = cosf(yaw);
  const float s = sinf(yaw);

  corners[0] = make_vec2(cx - dx * c + dy * s, cy - dx * s - dy * c);
  corners[1] = make_vec2(cx + dx * c + dy * s, cy + dx * s - dy * c);
  corners[2] = make_vec2(cx + dx * c - dy * s, cy + dx * s + dy * c);
  corners[3] = make_vec2(cx - dx * c - dy * s, cy - dx * s + dy * c);

  // --- enforce CCW winding ---
  float area2 = 0.0f;
  for (int i = 0; i < 4; i++) {
      const Vec2& p = corners[i];
      const Vec2& q = corners[(i + 1) % 4];
      area2 += p.x * q.y - p.y * q.x;
  }
  if (area2 < 0.0f) {
      // reverse order: swap 1<->3
      Vec2 tmp = corners[1];
      corners[1] = corners[3];
      corners[3] = tmp;
  }
}

__device__ __forceinline__ float polygon_area(const Vec2* poly, int n) {
  if (n < 3) {
    return 0.0f;
  }
  float area = 0.0f;
  for (int i = 0; i < n; ++i) {
    const Vec2& p = poly[i];
    const Vec2& q = poly[(i + 1) % n];
    area += cross(p, q);
  }
  return 0.5f * fabsf(area);
}

__device__ __forceinline__ int clip_polygon(
    const Vec2* subject,
    int subject_size,
    const Vec2& edge_start,
    const Vec2& edge_end,
    Vec2* output) {
  if (subject_size == 0) {
    return 0;
  }
  Vec2 edge = edge_end - edge_start;
  auto eval = [&](const Vec2& p) {
    Vec2 v = p - edge_start;
    return edge.x * v.y - edge.y * v.x;  // cross(edge, v)
  };

  int out_size = 0;
  for (int i = 0; i < subject_size; ++i) {
    const Vec2& curr = subject[i];
    const Vec2& prev = subject[(i + subject_size - 1) % subject_size];
    const float curr_val = eval(curr);
    const float prev_val = eval(prev);

    // For CCW polygons, interior is on the LEFT side => cross >= 0
    const bool curr_inside = curr_val >= 0.0f;
    const bool prev_inside = prev_val >= 0.0f;

    if (curr_inside && prev_inside) {
      if (out_size < 8) {
        output[out_size++] = curr;
      }
    } else if (prev_inside && !curr_inside) {
      const float t = prev_val / (prev_val - curr_val + 1e-7f);
      Vec2 intersect = prev + (curr - prev) * t;
      if (out_size < 8) {
        output[out_size++] = intersect;
      }
    } else if (!prev_inside && curr_inside) {
      const float t = prev_val / (prev_val - curr_val + 1e-7f);
      Vec2 intersect = prev + (curr - prev) * t;
      if (out_size < 8) {
        output[out_size++] = intersect;
      }
      if (out_size < 8) {
        output[out_size++] = curr;
      }
    }
  }
  return out_size;
}

__device__ __forceinline__ float oriented_iou_single(const float* box_a, const float* box_b) {
  Vec2 poly_a[8];
  Vec2 poly_b[4];
  Vec2 subject[8];
  Vec2 temp[8];

  box_to_corners(box_a, poly_a);
  box_to_corners(box_b, poly_b);

  int subject_size = 4;
  for (int i = 0; i < subject_size; ++i) {
    subject[i] = poly_a[i];
  }

  for (int i = 0; i < 4; ++i) {
    if (subject_size < 3) {
      break;
    }
    const Vec2& p1 = poly_b[i];
    const Vec2& p2 = poly_b[(i + 1) % 4];
    subject_size = clip_polygon(subject, subject_size, p1, p2, temp);
    for (int j = 0; j < subject_size; ++j) {
      subject[j] = temp[j];
    }
  }

  const float inter_area = polygon_area(subject, subject_size);
  if (inter_area <= 0.0f || !isfinite(inter_area)) {
    return 0.0f;
  }

  const float area_a = polygon_area(poly_a, 4);
  const float area_b = polygon_area(poly_b, 4);
  const float uni = area_a + area_b - inter_area;
  if (uni <= 0.0f || !isfinite(uni)) {
    return 0.0f;
  }
  return inter_area / uni;
}

__global__ void rotated_nms_cuda_kernel(
    const float* boxes,
    const int64_t* order,
    bool* suppressed,
    bool* selected,
    int64_t num_boxes,
    float iou_threshold,
    int64_t max_output) {
  __shared__ int64_t kept;
  __shared__ int keep_current;
  __shared__ int stop;

  if (threadIdx.x == 0) {
    kept = 0;
  }
  __syncthreads();

  for (int64_t sorted_idx = 0; sorted_idx < num_boxes; ++sorted_idx) {
    __syncthreads();
    if (threadIdx.x == 0) {
      stop = kept >= max_output;
    }
    __syncthreads();
    if (stop) {
      break;
    }

    const bool is_suppressed = suppressed[sorted_idx];
    if (threadIdx.x == 0) {
      keep_current = (!is_suppressed && kept < max_output) ? 1 : 0;
      if (keep_current) {
        selected[sorted_idx] = true;
        ++kept;
      }
    }
    __syncthreads();

    if (!keep_current) {
      continue;
    }

    const int64_t idx_a = order[sorted_idx];
    const float* box_a = boxes + idx_a * 7;

    for (int64_t j = sorted_idx + 1 + threadIdx.x; j < num_boxes; j += blockDim.x) {
      if (suppressed[j]) {
        continue;
      }
      const int64_t idx_b = order[j];
      const float* box_b = boxes + idx_b * 7;
      const float iou = oriented_iou_single(box_a, box_b);
      if (iou > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
}

}  // namespace

torch::Tensor rotated_nms_cuda(torch::Tensor boxes, torch::Tensor scores, double iou_threshold, int64_t max_out) {
  TORCH_CHECK(boxes.is_cuda(), "rotated_nms_cuda expects CUDA input for boxes");
  TORCH_CHECK(scores.is_cuda(), "rotated_nms_cuda expects CUDA input for scores");
  TORCH_CHECK(boxes.dim() == 2 && boxes.size(1) >= 7, "boxes must be [N,7]");
  TORCH_CHECK(scores.dim() == 1 && scores.size(0) == boxes.size(0), "scores must be [N]");

  auto boxes_contig = boxes.contiguous();
  if (boxes_contig.scalar_type() != torch::kFloat) {
    boxes_contig = boxes_contig.to(torch::kFloat);
  }
  auto scores_contig = scores.contiguous();

  const auto num_boxes = boxes_contig.size(0);
  if (num_boxes == 0 || max_out <= 0) {
    return torch::empty({0}, boxes.options().dtype(torch::kLong));
  }

  max_out = std::min<int64_t>(max_out, num_boxes);

  auto sort_result = scores_contig.sort(0, /*descending=*/true);
  auto order = std::get<1>(sort_result);
  auto suppressed = torch::zeros({num_boxes}, boxes.options().dtype(torch::kBool));
  auto selected = torch::zeros({num_boxes}, boxes.options().dtype(torch::kBool));

  const int threads = 512;
  const dim3 blocks(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  rotated_nms_cuda_kernel<<<blocks, threads, 0, stream>>>(
      boxes_contig.data_ptr<float>(),
      order.data_ptr<int64_t>(),
      suppressed.data_ptr<bool>(),
      selected.data_ptr<bool>(),
      num_boxes,
      static_cast<float>(iou_threshold),
      max_out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto keep_positions = torch::nonzero(selected).flatten();
  if (keep_positions.numel() == 0) {
    return torch::empty({0}, boxes.options().dtype(torch::kLong));
  }

  auto keep = order.index({keep_positions});
  if (keep.size(0) > max_out) {
    keep = keep.slice(0, 0, max_out);
  }
  return keep;
}

__global__ void oriented_iou_aligned_kernel(
    const float* boxes_a,
    const float* boxes_b,
    float* ious,
    int64_t num_boxes) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_boxes) {
    return;
  }
  const float* box_a = boxes_a + idx * 7;
  const float* box_b = boxes_b + idx * 7;
  ious[idx] = oriented_iou_single(box_a, box_b);
}

torch::Tensor oriented_iou_aligned_cuda(torch::Tensor boxes_a, torch::Tensor boxes_b) {
  TORCH_CHECK(boxes_a.is_cuda() && boxes_b.is_cuda(), "oriented_iou_aligned_cuda expects CUDA inputs");
  TORCH_CHECK(boxes_a.sizes() == boxes_b.sizes(), "boxes_a and boxes_b must have the same shape");
  TORCH_CHECK(boxes_a.dim() == 2 && boxes_a.size(1) >= 7, "boxes must be [N,7]");
  auto boxes_a_c = boxes_a.contiguous();
  auto boxes_b_c = boxes_b.contiguous();
  if (boxes_a_c.scalar_type() != torch::kFloat) {
    boxes_a_c = boxes_a_c.to(torch::kFloat);
  }
  if (boxes_b_c.scalar_type() != torch::kFloat) {
    boxes_b_c = boxes_b_c.to(torch::kFloat);
  }
  const auto N = boxes_a_c.size(0);
  auto out = torch::zeros({N}, boxes_a.options().dtype(torch::kFloat));
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  oriented_iou_aligned_kernel<<<blocks, threads, 0, stream>>>(
      boxes_a_c.data_ptr<float>(), boxes_b_c.data_ptr<float>(), out.data_ptr<float>(), N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
