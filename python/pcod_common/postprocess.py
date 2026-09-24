# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

"""Postprocessing utilities (NMS, IoU helpers)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from pcod_common.box_ops import aligned_box_iou
from pcod_common.torch_extensions.rotated_nms import load_rotated_nms_extension
from torchvision.ops import nms

_ROTATED_NMS_EXT = None
NmsResult = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
NmsResultWithIndices = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


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
        inside_curr = cross_curr >= 0
        inside_prev = cross_prev >= 0
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
            pair = boxes_xywlh[[idx, kept_idx]]
            iou = aligned_box_iou(pair[:1], pair[1:], three_d=bool((pair[:, 5] > 0).all()))[0]
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
    pre_nms_topk: int | None = None,
    return_indices: bool = False,
) -> NmsResult | NmsResultWithIndices:
    """Apply NMS, optionally returning indices into the input candidate tensors."""
    if boxes.numel() == 0:
        result = (boxes, scores, labels)
        return (*result, labels.new_empty((0,), dtype=torch.long)) if return_indices else result

    if boxes.ndim != 2 or boxes.size(1) != 7:
        raise ValueError("boxes must have shape (N, 7)")

    if labels.numel() != scores.numel():
        raise ValueError("labels and scores must have the same length")

    keep_indices = []
    unique_labels = labels.unique()

    def _nms_rotated_safe(boxes_xywlh: torch.Tensor, scores_vec: torch.Tensor) -> torch.Tensor:
        if boxes_xywlh.numel() == 0:
            return boxes_xywlh.new_zeros((0,), dtype=torch.long)
        if use_rotated:
            ext = _get_rotated_ext()
            if boxes_xywlh.is_cuda and scores_vec.is_cuda:
                if not hasattr(ext, "rotated_nms_cuda"):
                    raise RuntimeError("Rotated NMS extension missing CUDA NMS entrypoint.")
                return ext.rotated_nms_cuda(boxes_xywlh, scores_vec, iou_threshold, max_num_objects)
            if not hasattr(ext, "rotated_nms"):
                raise RuntimeError("Rotated NMS extension missing CPU NMS entrypoint.")
            return ext.rotated_nms(boxes_xywlh, scores_vec, iou_threshold, max_num_objects)
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
        original_indices = class_mask.nonzero(as_tuple=False).squeeze(1)
        original_indices = original_indices[score_mask]

        if pre_nms_topk is not None and class_scores.numel() > pre_nms_topk:
            top_scores, top_indices = torch.topk(
                class_scores,
                k=int(pre_nms_topk),
                largest=True,
                sorted=False,
            )
            class_boxes = class_boxes[top_indices]
            class_scores = top_scores
            original_indices = original_indices[top_indices]

        class_keep = _nms_rotated_safe(class_boxes, class_scores)
        if class_keep.numel() == 0:
            continue

        keep_indices.append(original_indices[class_keep])

    if not keep_indices:
        result = (boxes[:0], scores[:0], labels[:0])
        return (*result, labels.new_empty((0,), dtype=torch.long)) if return_indices else result

    keep_indices = torch.cat(keep_indices, dim=0)
    if not per_class_topk:
        order = scores[keep_indices].argsort(descending=True)
        keep_indices = keep_indices[order][:max_num_objects]

    result = (boxes[keep_indices], scores[keep_indices], labels[keep_indices])
    return (*result, keep_indices) if return_indices else result


def apply_nms_batch(
    boxes_batch: Sequence[torch.Tensor],
    scores_batch: Sequence[torch.Tensor],
    labels_batch: Sequence[torch.Tensor],
    score_thresholds: List[float],
    iou_threshold: float,
    max_num_objects: int,
    *,
    per_class_topk: bool = False,
    use_rotated: bool = True,
    pre_nms_topk: int | None = None,
    return_indices: bool = False,
) -> List[NmsResult | NmsResultWithIndices]:
    """Apply independent class-aware NMS groups in one CUDA launch.

    The CUDA path is semantically identical to calling :func:`apply_nms` for every
    item, but executes all sample/class groups concurrently. Unsupported devices or
    modes deliberately use the reference API. Optional indices refer to each
    sample's input candidate tensors.
    """
    if not (len(boxes_batch) == len(scores_batch) == len(labels_batch)):
        raise ValueError("boxes_batch, scores_batch, and labels_batch must have equal lengths")
    if not boxes_batch:
        return []
    if not use_rotated or not all(boxes.is_cuda and scores.is_cuda for boxes, scores in zip(boxes_batch, scores_batch)):
        return [
            apply_nms(
                boxes,
                scores,
                labels,
                score_thresholds,
                iou_threshold,
                max_num_objects,
                per_class_topk=per_class_topk,
                use_rotated=use_rotated,
                pre_nms_topk=pre_nms_topk,
                return_indices=return_indices,
            )
            for boxes, scores, labels in zip(boxes_batch, scores_batch, labels_batch)
        ]

    ext = _get_rotated_ext()
    if not hasattr(ext, "rotated_nms_cuda_batched"):
        return [
            apply_nms(
                boxes,
                scores,
                labels,
                score_thresholds,
                iou_threshold,
                max_num_objects,
                per_class_topk=per_class_topk,
                use_rotated=True,
                pre_nms_topk=pre_nms_topk,
                return_indices=return_indices,
            )
            for boxes, scores, labels in zip(boxes_batch, scores_batch, labels_batch)
        ]

    for boxes, scores, labels in zip(boxes_batch, scores_batch, labels_batch):
        if boxes.ndim != 2 or boxes.size(1) != 7:
            raise ValueError("boxes must have shape (N, 7)")
        if labels.numel() != scores.numel():
            raise ValueError("labels and scores must have the same length")

    device = boxes_batch[0].device
    sample_chunks = [
        torch.full((scores.numel(),), sample_idx, device=device, dtype=torch.long)
        for sample_idx, scores in enumerate(scores_batch)
    ]
    original_chunks = [torch.arange(scores.numel(), device=device, dtype=torch.long) for scores in scores_batch]
    flat_boxes_input = torch.cat(list(boxes_batch), dim=0)
    flat_scores_input = torch.cat(list(scores_batch), dim=0)
    flat_labels_input = torch.cat(list(labels_batch), dim=0)
    flat_sample_input = torch.cat(sample_chunks, dim=0)
    flat_original_input = torch.cat(original_chunks, dim=0)
    if flat_scores_input.numel() == 0:
        if return_indices:
            return [
                (boxes[:0], scores[:0], labels[:0], labels.new_empty((0,), dtype=torch.long))
                for boxes, scores, labels in zip(boxes_batch, scores_batch, labels_batch)
            ]
        return [(boxes[:0], scores[:0], labels[:0]) for boxes, scores, labels in zip(boxes_batch, scores_batch, labels_batch)]

    num_classes = len(score_thresholds)
    thresholds = flat_scores_input.new_tensor(score_thresholds)
    valid_labels = (flat_labels_input >= 0) & (flat_labels_input < num_classes)
    keep = valid_labels & (flat_scores_input >= thresholds[flat_labels_input.clamp(min=0, max=max(num_classes - 1, 0))])
    flat_boxes_input = flat_boxes_input[keep]
    flat_scores_input = flat_scores_input[keep]
    flat_labels_input = flat_labels_input[keep]
    flat_sample_input = flat_sample_input[keep]
    flat_original_input = flat_original_input[keep]

    group_ids = flat_sample_input * num_classes + flat_labels_input
    group_counts = torch.bincount(group_ids, minlength=len(boxes_batch) * num_classes).cpu().tolist()
    group_order = torch.argsort(group_ids, stable=True)
    flat_boxes_input = flat_boxes_input[group_order]
    flat_scores_input = flat_scores_input[group_order]
    flat_sample_input = flat_sample_input[group_order]
    flat_original_input = flat_original_input[group_order]

    sorted_boxes_chunks: List[torch.Tensor] = []
    sorted_original_chunks: List[torch.Tensor] = []
    sample_id_chunks: List[torch.Tensor] = []
    group_offsets = [0]
    start = 0
    for count in group_counts:
        end = start + count
        group_scores = flat_scores_input[start:end]
        group_original = flat_original_input[start:end]
        group_boxes = flat_boxes_input[start:end]
        group_samples = flat_sample_input[start:end]
        if pre_nms_topk is not None and count > pre_nms_topk:
            group_scores, top_indices = torch.topk(
                group_scores,
                k=int(pre_nms_topk),
                largest=True,
                sorted=False,
            )
            group_original = group_original[top_indices]
            group_boxes = group_boxes[top_indices]
            group_samples = group_samples[top_indices]
        _, score_order = group_scores.sort(descending=True)
        group_boxes = group_boxes[score_order]
        group_original = group_original[score_order]
        group_samples = group_samples[score_order]
        sorted_boxes_chunks.append(group_boxes)
        sorted_original_chunks.append(group_original)
        sample_id_chunks.append(group_samples)
        group_offsets.append(group_offsets[-1] + group_original.numel())
        start = end

    flat_boxes = torch.cat(sorted_boxes_chunks, dim=0)
    flat_original = torch.cat(sorted_original_chunks, dim=0)
    flat_sample_ids = torch.cat(sample_id_chunks, dim=0)
    offsets = torch.tensor(group_offsets, device=flat_boxes.device, dtype=torch.long)
    selected = ext.rotated_nms_cuda_batched(flat_boxes, offsets, iou_threshold, max_num_objects)
    kept_original = flat_original[selected]
    kept_sample_ids = flat_sample_ids[selected]
    counts = torch.bincount(kept_sample_ids, minlength=len(boxes_batch)).cpu().tolist()
    per_sample_kept = kept_original.split(counts)

    results: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for boxes, scores, labels, keep in zip(boxes_batch, scores_batch, labels_batch, per_sample_kept):
        if not per_class_topk and keep.numel() > max_num_objects:
            order = scores[keep].argsort(descending=True)[:max_num_objects]
            keep = keep[order]
        result = (boxes[keep], scores[keep], labels[keep])
        results.append((*result, keep) if return_indices else result)
    return results
