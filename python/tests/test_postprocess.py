# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import math

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None
pytestmark = pytest.mark.skipif(
    not (HAS_TORCH and HAS_TORCHVISION),
    reason="postprocess tests require torch and torchvision",
)

if HAS_TORCH and HAS_TORCHVISION:
    import pcod_common.postprocess as postprocess
    import torch
    from pcod_common.postprocess import apply_nms, apply_nms_batch


def _make_box(x: float, y: float, length: float = 1.0, width: float = 1.0, yaw: float = 0.0) -> list[float]:
    return [x, y, 0.0, length, width, 1.0, yaw]


ROTATED_NMS_CASES = [
    pytest.param(
        [_make_box(0.0, 0.0, 4.0, 2.0), _make_box(0.0, 0.0, 4.0, 2.0)],
        [0.9, 0.8],
        0.1,
        10,
        [0],
        id="identical",
    ),
    pytest.param(
        [_make_box(0.0, 0.0, 4.0, 2.0), _make_box(0.5, 0.0, 4.0, 2.0)],
        [0.9, 0.8],
        0.1,
        10,
        [0],
        id="axis_aligned_high_overlap",
    ),
    pytest.param(
        [_make_box(0.0, 0.0, 4.0, 2.0), _make_box(3.5, 0.0, 4.0, 2.0)],
        [0.9, 0.8],
        0.1,
        10,
        [0, 1],
        id="axis_aligned_low_overlap",
    ),
    pytest.param(
        [_make_box(0.0, 0.0, 4.0, 2.0), _make_box(4.0, 0.0, 4.0, 2.0)],
        [0.9, 0.8],
        0.0,
        10,
        [0, 1],
        id="touching_edges",
    ),
    pytest.param(
        [_make_box(0.0, 0.0, 6.0, 4.0), _make_box(0.0, 0.0, 2.0, 1.0)],
        [0.9, 0.8],
        0.05,
        10,
        [0],
        id="contained_box",
    ),
    pytest.param(
        [_make_box(0.0, 0.0, 4.0, 2.0, math.pi / 4.0), _make_box(0.2, 0.1, 4.0, 2.0, math.pi / 4.0)],
        [0.9, 0.8],
        0.1,
        10,
        [0],
        id="rotated_same_yaw",
    ),
    pytest.param(
        [_make_box(0.0, 0.0, 4.0, 2.0, math.pi / 4.0), _make_box(0.0, 0.0, 4.0, 2.0, -math.pi / 4.0)],
        [0.9, 0.8],
        0.1,
        10,
        [0],
        id="rotated_crossing",
    ),
    pytest.param(
        [_make_box(0.0, 0.0, 4.0, 2.0, math.pi / 4.0), _make_box(4.0, 4.0, 4.0, 2.0, math.pi / 4.0)],
        [0.9, 0.8],
        0.1,
        10,
        [0, 1],
        id="rotated_separated",
    ),
    pytest.param(
        [_make_box(0.0, 0.0, 4.0, 2.0), _make_box(0.5, 0.0, 4.0, 2.0), _make_box(8.0, 0.0, 4.0, 2.0)],
        [0.8, 0.95, 0.7],
        0.1,
        10,
        [1, 2],
        id="higher_score_second_suppresses_first",
    ),
    pytest.param(
        [_make_box(0.0, 0.0, 4.0, 2.0), _make_box(8.0, 0.0, 4.0, 2.0), _make_box(16.0, 0.0, 4.0, 2.0)],
        [0.9, 0.8, 0.7],
        0.1,
        2,
        [0, 1],
        id="max_output_limit",
    ),
]


def test_apply_nms_empty_inputs():
    """Return empty inputs unchanged."""
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
    """Reject malformed postprocessing tensors."""
    with pytest.raises(ValueError, match="boxes must have shape"):
        apply_nms(
            torch.zeros((3, 6), dtype=torch.float32),
            torch.ones((3,), dtype=torch.float32),
            torch.zeros((3,), dtype=torch.long),
            score_thresholds=[0.1],
            iou_threshold=0.1,
            max_num_objects=10,
            use_rotated=False,
        )

    with pytest.raises(ValueError, match="boxes must have shape"):
        apply_nms(
            torch.zeros((3, 8), dtype=torch.float32),
            torch.ones((3,), dtype=torch.float32),
            torch.zeros((3,), dtype=torch.long),
            score_thresholds=[0.1],
            iou_threshold=0.1,
            max_num_objects=10,
            use_rotated=False,
        )

    with pytest.raises(ValueError, match="labels and scores must have the same length"):
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
    """Apply score thresholds independently per class."""
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
    """Apply a global result limit when per-class top-k is disabled."""
    boxes = torch.tensor(
        [
            _make_box(0.0, 0.0),  # class 0
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
    """Keep limited results independently for each class."""
    boxes = torch.tensor(
        [
            _make_box(0.0, 0.0),  # class 0
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


def test_rotated_torch_iou_reports_identical_boxes_as_full_overlap():
    """Compute non-zero IoU for CCW rotated boxes in the pure torch fallback."""
    box = torch.tensor(_make_box(0.0, 0.0, length=4.0, width=2.0), dtype=torch.float32)

    iou = postprocess._oriented_iou_torch(box, box)

    assert float(iou) == pytest.approx(1.0)


def test_rotated_torch_nms_suppresses_overlapping_boxes():
    """Suppress a lower-scored rotated box when IoU exceeds the threshold."""
    boxes = torch.tensor(
        [
            _make_box(0.0, 0.0, length=4.0, width=2.0),
            _make_box(0.5, 0.0, length=4.0, width=2.0),
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32)

    keep = postprocess._nms_rotated_torch(boxes, scores, iou_threshold=0.1, max_num_objects=10)

    assert keep.tolist() == [0]


@pytest.mark.parametrize(("boxes_values", "scores_values", "iou_threshold", "max_num_objects", "expected"), ROTATED_NMS_CASES)
def test_rotated_torch_nms_cases_match_expected(
    boxes_values: list[list[float]],
    scores_values: list[float],
    iou_threshold: float,
    max_num_objects: int,
    expected: list[int],
):
    """Cover representative rotated NMS geometry and ordering cases."""
    boxes = torch.tensor(boxes_values, dtype=torch.float32)
    scores = torch.tensor(scores_values, dtype=torch.float32)

    keep = postprocess._nms_rotated_torch(boxes, scores, iou_threshold, max_num_objects)

    assert keep.tolist() == expected


@pytest.mark.parametrize(("boxes_values", "scores_values", "iou_threshold", "max_num_objects", "expected"), ROTATED_NMS_CASES)
def test_rotated_extension_cpu_nms_cases_match_torch_and_expected(
    boxes_values: list[list[float]],
    scores_values: list[float],
    iou_threshold: float,
    max_num_objects: int,
    expected: list[int],
):
    """Keep CPU extension NMS in parity with the pure torch fallback."""
    boxes = torch.tensor(boxes_values, dtype=torch.float32)
    scores = torch.tensor(scores_values, dtype=torch.float32)
    expected_tensor = postprocess._nms_rotated_torch(boxes, scores, iou_threshold, max_num_objects)

    try:
        ext = postprocess._get_rotated_ext()
    except Exception as exc:  # pragma: no cover - environment-dependent extension toolchain
        pytest.skip(f"rotated NMS extension unavailable: {exc}")

    keep = ext.rotated_nms(boxes, scores, iou_threshold, max_num_objects)

    assert expected_tensor.tolist() == expected
    assert keep.cpu().tolist() == expected


@pytest.mark.parametrize(("boxes_values", "scores_values", "iou_threshold", "max_num_objects", "expected"), ROTATED_NMS_CASES)
def test_rotated_extension_cuda_nms_cases_match_cpu_and_expected(
    boxes_values: list[list[float]],
    scores_values: list[float],
    iou_threshold: float,
    max_num_objects: int,
    expected: list[int],
):
    """Keep CUDA extension NMS in parity with the CPU extension when CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available to torch")

    try:
        ext = postprocess._get_rotated_ext()
    except Exception as exc:  # pragma: no cover - environment-dependent extension toolchain
        pytest.skip(f"rotated NMS extension unavailable: {exc}")
    if not hasattr(ext, "rotated_nms_cuda"):
        pytest.skip("rotated NMS extension has no CUDA entrypoint")

    boxes_cpu = torch.tensor(boxes_values, dtype=torch.float32)
    scores_cpu = torch.tensor(scores_values, dtype=torch.float32)
    cpu_keep = ext.rotated_nms(boxes_cpu, scores_cpu, iou_threshold, max_num_objects)
    cuda_keep = ext.rotated_nms_cuda(
        boxes_cpu.cuda(),
        scores_cpu.cuda(),
        iou_threshold,
        max_num_objects,
    )

    assert cpu_keep.cpu().tolist() == expected
    assert cuda_keep.cpu().tolist() == expected


@pytest.mark.parametrize(
    ("pre_nms_topk", "per_class_topk"),
    [(None, True), (12, True), (12, False)],
)
def test_apply_nms_batch_cuda_matches_per_sample_reference(
    pre_nms_topk: int | None,
    per_class_topk: bool,
):
    """Keep batched CUDA NMS exactly equal to the public per-sample API."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available to torch")

    try:
        ext = postprocess._get_rotated_ext()
    except Exception as exc:  # pragma: no cover - environment-dependent extension toolchain
        pytest.skip(f"rotated NMS extension unavailable: {exc}")
    if not hasattr(ext, "rotated_nms_cuda_batched"):
        pytest.skip("rotated NMS extension has no batched CUDA entrypoint")

    generator = torch.Generator(device="cuda").manual_seed(20260923)
    boxes_batch = []
    scores_batch = []
    labels_batch = []
    for sample_idx in range(9):
        count = 0 if sample_idx == 0 else 32 + sample_idx
        boxes = torch.randn(count, 7, generator=generator, device="cuda")
        boxes[:, :2] *= 4.0
        boxes[:, 3:6] = boxes[:, 3:6].abs() + 0.5
        boxes_batch.append(boxes)
        scores_batch.append(torch.rand(count, generator=generator, device="cuda"))
        labels_batch.append(torch.randint(0, 3, (count,), generator=generator, device="cuda"))

    kwargs = {
        "score_thresholds": [0.1, 0.2, 0.3],
        "iou_threshold": 0.5,
        "max_num_objects": 16,
        "per_class_topk": per_class_topk,
        "use_rotated": True,
        "pre_nms_topk": pre_nms_topk,
    }
    expected = [
        apply_nms(boxes, scores, labels, **kwargs)
        for boxes, scores, labels in zip(boxes_batch, scores_batch, labels_batch)
    ]

    actual = apply_nms_batch(boxes_batch, scores_batch, labels_batch, **kwargs)

    for actual_sample, expected_sample in zip(actual, expected):
        for actual_tensor, expected_tensor in zip(actual_sample, expected_sample):
            assert torch.equal(actual_tensor, expected_tensor)


def test_rotated_extension_cpu_rejects_extra_box_columns():
    """Reject wider tensors instead of silently ignoring extra columns."""
    try:
        ext = postprocess._get_rotated_ext()
    except Exception as exc:  # pragma: no cover - environment-dependent extension toolchain
        pytest.skip(f"rotated NMS extension unavailable: {exc}")

    boxes = torch.tensor(
        [
            _make_box(0.0, 0.0, length=4.0, width=2.0) + [100.0],
            _make_box(0.5, 0.0, length=4.0, width=2.0) + [200.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32)

    with pytest.raises(RuntimeError, match=r"boxes must be \[N,7\]"):
        ext.rotated_nms(boxes, scores, 0.1, 10)


def test_rotated_extension_cuda_rejects_extra_box_columns():
    """Reject wider tensors in the CUDA entrypoint too."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available to torch")

    try:
        ext = postprocess._get_rotated_ext()
    except Exception as exc:  # pragma: no cover - environment-dependent extension toolchain
        pytest.skip(f"rotated NMS extension unavailable: {exc}")
    if not hasattr(ext, "rotated_nms_cuda"):
        pytest.skip("rotated NMS extension has no CUDA entrypoint")

    boxes = torch.tensor(
        [
            _make_box(0.0, 0.0, length=4.0, width=2.0) + [100.0],
            _make_box(0.5, 0.0, length=4.0, width=2.0) + [200.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32)

    with pytest.raises(RuntimeError, match=r"boxes must be \[N,7\]"):
        ext.rotated_nms_cuda(boxes.cuda(), scores.cuda(), 0.1, 10)


def test_apply_nms_rotated_fails_fast_when_extension_fails(monkeypatch: pytest.MonkeyPatch):
    """Propagate rotated-NMS extension loading failures."""

    def _raise_ext():
        raise RuntimeError("extension unavailable")

    monkeypatch.setattr(postprocess, "_get_rotated_ext", _raise_ext)

    boxes = torch.tensor([_make_box(0.0, 0.0), _make_box(0.1, 0.0)], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
    labels = torch.tensor([0, 0], dtype=torch.long)

    with pytest.raises(RuntimeError, match="extension unavailable"):
        apply_nms(
            boxes,
            scores,
            labels,
            score_thresholds=[0.0],
            iou_threshold=0.1,
            max_num_objects=10,
            use_rotated=True,
        )
