"""Postprocessing utilities (NMS, IoU helpers)."""

from __future__ import annotations

from typing import List, Tuple

from pcod_common.torch_extensions.rotated_nms import load_rotated_nms_extension
import torch
from torchvision.ops import nms

_ROTATED_NMS_EXT = None


def _get_rotated_ext():
    global _ROTATED_NMS_EXT
    if _ROTATED_NMS_EXT is not None:
        return _ROTATED_NMS_EXT
    _ROTATED_NMS_EXT = load_rotated_nms_extension(force_build=True)
    return _ROTATED_NMS_EXT


def _boxes_to_corners_torch(boxes_xywlh: torch.Tensor) -> torch.Tensor:
    center_x = boxes_xywlh[:, 0:1]
    center_y = boxes_xywlh[:, 1:2]
    length = boxes_xywlh[:, 3:4]
    width = boxes_xywlh[:, 4:5]
    yaw = boxes_xywlh[:, 6:7]

    half_length = length * 0.5
    half_width = width * 0.5
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)

    corners = torch.stack(
        [
            torch.cat(
                [
                    -half_length * cos_y + half_width * sin_y + center_x,
                    -half_length * sin_y - half_width * cos_y + center_y,
                ],
                dim=1,
            ),
            torch.cat(
                [
                    half_length * cos_y + half_width * sin_y + center_x,
                    half_length * sin_y - half_width * cos_y + center_y,
                ],
                dim=1,
            ),
            torch.cat(
                [
                    half_length * cos_y - half_width * sin_y + center_x,
                    half_length * sin_y + half_width * cos_y + center_y,
                ],
                dim=1,
            ),
            torch.cat(
                [
                    -half_length * cos_y - half_width * sin_y + center_x,
                    -half_length * sin_y + half_width * cos_y + center_y,
                ],
                dim=1,
            ),
        ],
        dim=1,
    )
    return corners


def _clip_polygon_torch(subject: torch.Tensor, edge_start: torch.Tensor, edge_end: torch.Tensor) -> torch.Tensor:
    if subject.numel() == 0:
        return subject
    edge = edge_end - edge_start
    output = []
    for i in range(subject.shape[0]):
        curr = subject[i]
        prev = subject[i - 1]
        cross_curr = edge[0] * (curr - edge_start)[1] - edge[1] * (curr - edge_start)[0]
        cross_prev = edge[0] * (prev - edge_start)[1] - edge[1] * (prev - edge_start)[0]
        inside_curr = cross_curr <= 0
        inside_prev = cross_prev <= 0
        if inside_curr and inside_prev:
            output.append(curr)
        elif inside_prev and not inside_curr:
            t = cross_prev / (cross_prev - cross_curr + 1e-12)
            output.append(prev + t * (curr - prev))
        elif not inside_prev and inside_curr:
            t = cross_prev / (cross_prev - cross_curr + 1e-12)
            output.append(prev + t * (curr - prev))
            output.append(curr)
    if not output:
        return subject.new_zeros((0, 2))
    return torch.stack(output, dim=0)


def _polygon_area_torch(poly: torch.Tensor) -> torch.Tensor:
    if poly.numel() == 0 or poly.shape[0] < 3:
        return poly.new_tensor(0.0)
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * torch.abs(torch.dot(x, torch.roll(y, -1)) - torch.dot(y, torch.roll(x, -1)))


def _oriented_iou_torch(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    poly_a = _boxes_to_corners_torch(box_a.unsqueeze(0))[0]
    poly_b = _boxes_to_corners_torch(box_b.unsqueeze(0))[0]
    inter = poly_a
    for i in range(poly_b.shape[0]):
        if inter.shape[0] < 3:
            break
        p1 = poly_b[i]
        p2 = poly_b[(i + 1) % poly_b.shape[0]]
        inter = _clip_polygon_torch(inter, p1, p2)
    inter_area = _polygon_area_torch(inter)
    if inter_area <= 0:
        return box_a.new_tensor(0.0)
    area_a = _polygon_area_torch(poly_a)
    area_b = _polygon_area_torch(poly_b)
    union = area_a + area_b - inter_area
    return inter_area / torch.clamp(union, min=1e-6)


def _nms_rotated_torch(
    boxes_xywlh: torch.Tensor,
    scores_vec: torch.Tensor,
    iou_threshold: float,
    max_num_objects: int,
) -> torch.Tensor:
    order = torch.argsort(scores_vec, descending=True)
    keep = []
    for idx in order.tolist():
        suppress = False
        for kept_idx in keep:
            iou = _oriented_iou_torch(boxes_xywlh[idx], boxes_xywlh[kept_idx])
            if iou > iou_threshold:
                suppress = True
                break
        if not suppress:
            keep.append(idx)
        if len(keep) >= max_num_objects:
            break
    return torch.as_tensor(keep, dtype=torch.long, device=boxes_xywlh.device)


def _nms_axis_aligned(
    boxes_xywlh: torch.Tensor,
    scores_vec: torch.Tensor,
    iou_threshold: float,
    max_num_objects: int,
) -> torch.Tensor:
    half_w = boxes_xywlh[:, 3] * 0.5
    half_h = boxes_xywlh[:, 4] * 0.5
    boxes_xyxy = torch.stack(
        [
            boxes_xywlh[:, 0] - half_w,
            boxes_xywlh[:, 1] - half_h,
            boxes_xywlh[:, 0] + half_w,
            boxes_xywlh[:, 1] + half_h,
        ],
        dim=1,
    )
    keep_indices = nms(boxes_xyxy, scores_vec, iou_threshold)
    return keep_indices[:max_num_objects]


def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    score_thresholds: List[float],
    iou_threshold: float,
    max_num_objects: int,
    *,
    per_class_topk: bool = False,
    use_rotated: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes, scores, labels

    if boxes.ndim != 2 or boxes.size(1) < 7:
        raise ValueError('boxes must have shape (N, 7+)')

    if labels.numel() != scores.numel():
        raise ValueError('labels and scores must have the same length')

    keep_indices = []
    unique_labels = labels.unique()

    def _nms_rotated_safe(boxes_xywlh: torch.Tensor, scores_vec: torch.Tensor) -> torch.Tensor:
        if boxes_xywlh.numel() == 0:
            return boxes_xywlh.new_zeros((0,), dtype=torch.long)
        if use_rotated:
            try:
                ext = _get_rotated_ext()
                if boxes_xywlh.is_cuda and scores_vec.is_cuda:
                    return ext.rotated_nms_cuda(boxes_xywlh, scores_vec, iou_threshold, max_num_objects)
                return ext.rotated_nms(boxes_xywlh, scores_vec, iou_threshold, max_num_objects)
            except Exception:
                return _nms_rotated_torch(boxes_xywlh, scores_vec, iou_threshold, max_num_objects)
        return _nms_axis_aligned(boxes_xywlh, scores_vec, iou_threshold, max_num_objects)

    for class_label in unique_labels.tolist():
        class_mask = labels == class_label
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        if class_scores.numel() == 0:
            continue

        threshold = score_thresholds[min(class_label, len(score_thresholds) - 1)]
        score_mask = class_scores >= threshold
        if not score_mask.any():
            continue

        class_boxes = class_boxes[score_mask]
        class_scores = class_scores[score_mask]
        class_keep = _nms_rotated_safe(class_boxes, class_scores)
        if class_keep.numel() == 0:
            continue

        original_indices = class_mask.nonzero(as_tuple=False).squeeze(1)
        original_indices = original_indices[score_mask]
        keep_indices.append(original_indices[class_keep])

    if not keep_indices:
        return boxes[:0], scores[:0], labels[:0]

    keep_indices = torch.cat(keep_indices, dim=0)
    if not per_class_topk:
        order = scores[keep_indices].argsort(descending=True)
        keep_indices = keep_indices[order][:max_num_objects]

    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]
