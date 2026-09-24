# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0
"""Shared decoder and 3D NMS contracts."""

import pytest
import torch
from pcod_common import postprocess
from pcod_common.box_ops import aligned_box_iou, decode_pbod
from pcod_common.postprocess import apply_nms, apply_nms_batch


def test_decode_scores_geometry_and_class_selection():
    """Decode the best class and apply the same score and size rules as C++."""
    reg = torch.zeros(1, 2, 14)
    reg[0, 0, 3:6] = torch.tensor([100.0, -100.0, 0.0])
    focal = torch.tensor([[[0.0], [2.0]]])
    cls = torch.tensor([[[0.7, 0.2], [0.1, 1.2]]])
    sizes = torch.ones(1, 2, 6)
    centers = torch.tensor([[[0.5, 0.5, 0.0], [1.5, 0.5, 0.0]]])
    batch, boxes, scores = decode_pbod(reg, focal, cls, sizes, centers)
    assert batch.tolist() == [0, 0]
    assert boxes[:, 7].tolist() == [0, 1]
    torch.testing.assert_close(scores, focal.sigmoid().reshape(-1) * cls.softmax(-1).amax(-1).reshape(-1))
    assert torch.isfinite(boxes).all()
    torch.testing.assert_close(boxes[0, 3:6], torch.tensor([10.0, -10.0, 0.0]).exp())


def test_aligned_iou_accounts_for_height_and_supports_gradients():
    """Disjoint heights have no 3D overlap, while partial overlap differentiates."""
    pred = torch.tensor([[0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0]], requires_grad=True)
    target = torch.tensor([[0.5, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0]])
    iou = aligned_box_iou(pred, target)
    torch.testing.assert_close(iou, torch.tensor([0.6]))
    iou.sum().backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[0, 0] > 0
    target[0, 2] = 3.0
    torch.testing.assert_close(aligned_box_iou(pred.detach(), target), torch.zeros(1))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nms_preserves_stacked_and_different_class_objects(device):
    """Keep spatially separated and differently classified boxes."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available to torch")
    try:
        postprocess._get_rotated_ext()
    except Exception as exc:  # pragma: no cover - environment-dependent extension toolchain
        pytest.skip(f"rotated NMS extension unavailable: {exc}")
    boxes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 3.0, 2.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0],
        ],
        device=device,
    )
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6], device=device)
    labels = torch.tensor([0, 0, 1, 0], device=device)
    result = apply_nms(boxes, scores, labels, [0.1, 0.1], 0.5, 10)
    assert result[1].tolist() == pytest.approx([0.9, 0.8, 0.7])
    batched = apply_nms_batch([boxes], [scores], [labels], [0.1, 0.1], 0.5, 10)[0]
    for expected, actual in zip(result, batched):
        torch.testing.assert_close(expected, actual)
