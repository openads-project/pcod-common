# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

"""Differentiable rotated box geometry, computed in float32 under mixed precision."""

from __future__ import annotations

import torch


def corners_bev(boxes: torch.Tensor) -> torch.Tensor:
    """Return the four counterclockwise ground-plane corners of each box."""
    signs = boxes.new_tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    local = signs * boxes[..., None, 3:5] * 0.5
    c, s = boxes[..., 6].cos(), boxes[..., 6].sin()
    x, y = local.unbind(-1)
    return torch.stack((x * c[..., None] - y * s[..., None], x * s[..., None] + y * c[..., None]), -1) + boxes[..., None, :2]


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def aligned_box_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    three_d: bool = True,
    distance_penalty: bool = False,
    return_iou: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Aligned exact rotated IoU (or DIoU), with piecewise analytic autograd.

    The intersection polygon consists of contained corners and edge crossings.
    Topology/sorting is discrete; vertex positions remain differentiable. DIoU
    additionally supplies center gradients for disjoint boxes. With return_iou,
    return (score, plain_iou) to reuse intersection geometry for quality targets.
    """
    dtype = torch.float64 if pred.dtype == torch.float64 else torch.float32
    a, b = pred.to(dtype), target.to(dtype)
    ca, cb = corners_bev(a), corners_bev(b)
    ea, eb = ca.roll(-1, -2) - ca, cb.roll(-1, -2) - cb
    inside_a = (_cross(eb[..., None, :, :], ca[..., :, None, :] - cb[..., None, :, :]) >= -1e-7).all(-1)
    inside_b = (_cross(ea[..., None, :, :], cb[..., :, None, :] - ca[..., None, :, :]) >= -1e-7).all(-1)
    delta = cb[..., None, :, :] - ca[..., :, None, :]
    den = _cross(ea[..., :, None, :], eb[..., None, :, :])
    safe_den = torch.where(den.abs() > 1e-8, den, torch.ones_like(den))
    t = _cross(delta, eb[..., None, :, :]) / safe_den
    u = _cross(delta, ea[..., :, None, :]) / safe_den
    crossings = (den.abs() > 1e-8) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    points = ca[..., :, None, :] + t[..., None] * ea[..., :, None, :]
    vertices = torch.cat((ca, cb, points.flatten(-3, -2)), -2)
    valid = torch.cat((inside_a, inside_b, crossings.flatten(-2)), -1)
    count = valid.sum(-1)
    center = (vertices * valid[..., None]).sum(-2) / count.clamp_min(1)[..., None]
    relative = vertices - center[..., None, :]
    # Invalid coordinates must not participate in the centroid, ordering or area.
    angles = torch.atan2(relative[..., 1], relative[..., 0]).masked_fill(~valid, 10.0)
    order = angles.argsort(-1)
    polygon = vertices.gather(-2, order[..., None].expand_as(vertices))
    index = torch.arange(vertices.shape[-2], device=a.device)
    next_index = (index + 1) % count.clamp_min(1)[..., None]
    following = polygon.gather(-2, next_index[..., None].expand_as(polygon))
    terms = _cross(polygon - center[..., None, :], following - center[..., None, :])
    intersection = (terms * (index < count[..., None])).sum(-1).abs() * 0.5
    intersection = torch.where(count >= 3, intersection, torch.zeros_like(intersection))
    area_a, area_b = a[..., 3] * a[..., 4], b[..., 3] * b[..., 4]
    if three_d:
        low = torch.maximum(a[..., 2] - a[..., 5] / 2, b[..., 2] - b[..., 5] / 2)
        high = torch.minimum(a[..., 2] + a[..., 5] / 2, b[..., 2] + b[..., 5] / 2)
        intersection = intersection * (high - low).clamp_min(0)
        area_a, area_b = area_a * a[..., 5], area_b * b[..., 5]
    iou = (intersection / (area_a + area_b - intersection).clamp_min(1e-7)).clamp(0, 1)
    overlap_iou = iou
    if distance_penalty:
        all_corners = torch.cat((ca, cb), -2)
        diagonal = (all_corners.amax(-2) - all_corners.amin(-2)).square().sum(-1)
        distance = (a[..., :2] - b[..., :2]).square().sum(-1)
        if three_d:
            z_span = torch.maximum(a[..., 2] + a[..., 5] / 2, b[..., 2] + b[..., 5] / 2) - torch.minimum(
                a[..., 2] - a[..., 5] / 2, b[..., 2] - b[..., 5] / 2
            )
            diagonal = diagonal + z_span.square()
            distance = distance + (a[..., 2] - b[..., 2]).square()
        iou = iou - distance / diagonal.clamp_min(1e-7)
    return (iou, overlap_iou) if return_iou else iou


def decode_boxes(reg: torch.Tensor, priors: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Decode (...,7) encoded boxes without half-precision exponentiation."""
    reg, priors, centers = reg.float(), priors.float(), centers.float()
    return torch.cat(
        (
            reg[..., :3] * priors + centers,
            reg[..., 3:6].clamp(-10, 10).exp() * priors,
            reg[..., 6:7],
        ),
        -1,
    )


def decode_pbod(
    reg_logits: torch.Tensor,
    focal_logits: torch.Tensor,
    class_logits: torch.Tensor,
    size_priors: torch.Tensor,
    centers: torch.Tensor,
    score_threshold: float = 0.0,
    yaw_flip_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode flattened PBOD outputs (B,N,C*7), matching C++ DecodePbod.

    One physical box per cell, labeled by its best class. Scores combine the
    exported presence/quality probability and the conditional class probability.
    Returns batch indices, absolute (N,8) boxes including local class, and scores.
    """
    b, n, _ = reg_logits.shape
    c = class_logits.shape[-1]
    probabilities = class_logits.float().softmax(-1)
    class_probability, labels = probabilities.max(-1)
    scores = focal_logits.float().squeeze(-1).sigmoid() * class_probability
    batch, cell = torch.nonzero(scores >= score_threshold, as_tuple=True)
    cls = labels[batch, cell]
    reg = reg_logits.reshape(b, n, c, 7)[batch, cell, cls]
    priors = size_priors.reshape(b, n, c, 3)[batch, cell, cls]
    boxes = decode_boxes(reg, priors, centers[batch, cell])
    yaw = boxes[:, 6]
    if yaw_flip_logits is not None:
        desired = yaw_flip_logits[batch, cell].reshape(-1) >= 0
        yaw = yaw + ((yaw.cos() >= 0) != desired).float() * torch.pi
    boxes[:, 6] = torch.remainder(yaw + torch.pi, 2 * torch.pi) - torch.pi
    finite = torch.isfinite(boxes).all(-1) & torch.isfinite(scores[batch, cell])
    return (
        batch[finite],
        torch.cat((boxes, cls[:, None].float()), -1)[finite],
        scores[batch, cell][finite],
    )
