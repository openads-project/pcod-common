from __future__ import annotations

import importlib.util
import pytest

HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_TORCHVISION = importlib.util.find_spec('torchvision') is not None
pytestmark = pytest.mark.skipif(
    not (HAS_TORCH and HAS_TORCHVISION),
    reason='postprocess tests require torch and torchvision',
)

if HAS_TORCH and HAS_TORCHVISION:
    import torch

    from pcod_common.postprocess import apply_nms
    import pcod_common.postprocess as postprocess


def _make_box(x: float, y: float, length: float = 1.0, width: float = 1.0, yaw: float = 0.0) -> list[float]:
    return [x, y, 0.0, length, width, 1.0, yaw]


def test_apply_nms_empty_inputs():
    boxes = torch.zeros((0, 7), dtype=torch.float32)
    scores = torch.zeros((0,), dtype=torch.float32)
    labels = torch.zeros((0,), dtype=torch.long)

    out_boxes, out_scores, out_labels = apply_nms(
        boxes,
        scores,
        labels,
        score_thresholds=[0.5],
        iou_threshold=0.1,
        max_num_objects=10,
        use_rotated=False,
    )

    assert out_boxes.shape == (0, 7)
    assert out_scores.shape == (0,)
    assert out_labels.shape == (0,)


def test_apply_nms_rejects_invalid_shapes():
    with pytest.raises(ValueError, match='boxes must have shape'):
        apply_nms(
            torch.zeros((3, 6), dtype=torch.float32),
            torch.ones((3,), dtype=torch.float32),
            torch.zeros((3,), dtype=torch.long),
            score_thresholds=[0.1],
            iou_threshold=0.1,
            max_num_objects=10,
            use_rotated=False,
        )

    with pytest.raises(ValueError, match='labels and scores must have the same length'):
        apply_nms(
            torch.zeros((3, 7), dtype=torch.float32),
            torch.ones((3,), dtype=torch.float32),
            torch.zeros((2,), dtype=torch.long),
            score_thresholds=[0.1],
            iou_threshold=0.1,
            max_num_objects=10,
            use_rotated=False,
        )


def test_apply_nms_uses_per_class_thresholds():
    boxes = torch.tensor(
        [
            _make_box(0.0, 0.0),  # class 0 keep (0.6 >= 0.5)
            _make_box(10.0, 0.0),  # class 0 drop (0.4 < 0.5)
            _make_box(0.0, 10.0),  # class 1 keep (0.8 >= 0.7)
            _make_box(10.0, 10.0),  # class 1 drop (0.6 < 0.7)
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.6, 0.4, 0.8, 0.6], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    out_boxes, out_scores, out_labels = apply_nms(
        boxes,
        scores,
        labels,
        score_thresholds=[0.5, 0.7],
        iou_threshold=0.1,
        max_num_objects=10,
        use_rotated=False,
    )

    assert out_boxes.shape[0] == 2
    assert sorted(out_labels.tolist()) == [0, 1]
    assert sorted([round(float(v), 3) for v in out_scores.tolist()]) == [0.6, 0.8]


def test_apply_nms_global_topk_when_not_per_class_topk():
    boxes = torch.tensor(
        [
            _make_box(0.0, 0.0),   # class 0
            _make_box(10.0, 0.0),  # class 1
            _make_box(20.0, 0.0),  # class 1
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.95, 0.90, 0.70], dtype=torch.float32)
    labels = torch.tensor([0, 1, 1], dtype=torch.long)

    out_boxes, out_scores, out_labels = apply_nms(
        boxes,
        scores,
        labels,
        score_thresholds=[0.0, 0.0],
        iou_threshold=0.1,
        max_num_objects=2,
        per_class_topk=False,
        use_rotated=False,
    )

    assert out_boxes.shape[0] == 2
    assert [round(float(v), 3) for v in out_scores.tolist()] == [0.95, 0.9]
    assert out_labels.tolist() == [0, 1]


def test_apply_nms_per_class_topk_keeps_per_class_results():
    boxes = torch.tensor(
        [
            _make_box(0.0, 0.0),   # class 0
            _make_box(10.0, 0.0),  # class 1
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.8, 0.7], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)

    out_boxes, out_scores, out_labels = apply_nms(
        boxes,
        scores,
        labels,
        score_thresholds=[0.0, 0.0],
        iou_threshold=0.1,
        max_num_objects=1,
        per_class_topk=True,
        use_rotated=False,
    )

    assert out_boxes.shape[0] == 2
    assert sorted(out_labels.tolist()) == [0, 1]
    assert sorted([round(float(v), 3) for v in out_scores.tolist()]) == [0.7, 0.8]


def test_apply_nms_rotated_falls_back_when_extension_fails(monkeypatch: pytest.MonkeyPatch):
    def _raise_ext():
        raise RuntimeError('extension unavailable')

    def _fallback(boxes_xywlh, scores_vec, iou_threshold: float, max_num_objects: int):
        return torch.tensor([0], dtype=torch.long, device=boxes_xywlh.device)

    monkeypatch.setattr(postprocess, '_get_rotated_ext', _raise_ext)
    monkeypatch.setattr(postprocess, '_nms_rotated_torch', _fallback)

    boxes = torch.tensor([_make_box(0.0, 0.0), _make_box(0.1, 0.0)], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
    labels = torch.tensor([0, 0], dtype=torch.long)

    out_boxes, out_scores, out_labels = apply_nms(
        boxes,
        scores,
        labels,
        score_thresholds=[0.0],
        iou_threshold=0.1,
        max_num_objects=10,
        use_rotated=True,
    )

    assert out_boxes.shape[0] == 1
    assert out_labels.tolist() == [0]
    assert [round(float(v), 3) for v in out_scores.tolist()] == [0.9]
