# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

"""Utilities to convert raw point clouds into PBOD pillar tensors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from pcod_common.torch_extensions.pillar_cuda import pillar_preprocess


@dataclass(frozen=True)
class PillarPreprocessorConfig:
    """Configuration for point-cloud pillar preprocessing."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    voxel_x: float
    voxel_y: float
    point_feature_dim: int
    descriptor_dim: int = 10


class PillarPreprocessor(torch.nn.Module):
    """Converts batched raw point clouds into PBOD pillar tensors.

    Uses dense grid representation where all grid cells are allocated.
    Emits point-wise features and pillar mappings for downstream aggregation.
    """

    def __init__(self, cfg: PillarPreprocessorConfig) -> None:
        """Initialize the preprocessor from a pillar configuration."""
        super().__init__()
        self.x_min = float(cfg.x_min)
        self.y_min = float(cfg.y_min)
        self.z_min = float(cfg.z_min)
        self.x_max = float(cfg.x_max)
        self.y_max = float(cfg.y_max)
        self.z_max = float(cfg.z_max)
        self.voxel_x = float(cfg.voxel_x)
        self.voxel_y = float(cfg.voxel_y)

        self.grid_x = int(math.ceil((self.x_max - self.x_min) / self.voxel_x))
        self.grid_y = int(math.ceil((self.y_max - self.y_min) / self.voxel_y))
        self.num_pillars = self.grid_x * self.grid_y
        self.point_feature_dim = int(cfg.point_feature_dim)
        self.descriptor_dim = int(cfg.descriptor_dim)

        self.register_buffer(
            "voxel_size_tensor",
            torch.tensor([self.voxel_x, self.voxel_y, 0.0], dtype=torch.float32),
            persistent=False,
        )

        grid_x_idx = torch.arange(self.grid_x, dtype=torch.long)
        grid_y_idx = torch.arange(self.grid_y, dtype=torch.long)
        mesh = torch.stack(torch.meshgrid(grid_x_idx, grid_y_idx, indexing="ij"), dim=-1)
        coords = mesh.view(-1, 2)
        self.register_buffer("pillar_coords", coords, persistent=False)

        center_x = self.x_min + (coords[:, 0].to(torch.float32) + 0.5) * self.voxel_x
        center_y = self.y_min + (coords[:, 1].to(torch.float32) + 0.5) * self.voxel_y
        center_z = torch.full_like(center_x, self.z_min + (self.z_max - self.z_min) * 0.5)
        centers = torch.stack([center_x, center_y, center_z], dim=1)
        self.register_buffer("pillar_centers", centers, persistent=False)

    def forward(
        self,
        points_mask: torch.Tensor,
        points_feature: torch.Tensor,
        points_xyz: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Convert raw point clouds to pillar tensors using a fully tensorised pipeline.

        Designed to stay ONNX-traceable and avoids Python loops over the batch.

        Args:
            points_mask: (B, P) int/bool mask, nonzero marks valid points.
            points_feature: (B, P, C_in) float32 per-point extra features.
            points_xyz: (B, P, 3) float32 XYZ coordinates.

        Returns:
            point_features: (B, P, C_out) float32 engineered per-point features.
            pillar_ids: (B, P) int64 pillar index per point (sentinel = num_pillars).
            valid_mask: (B, P) bool mask of in-range points.
            pillar_masks: (B, num_pillars) bool mask of pillars with any points.
            pillar_indices: (B, num_pillars, 2) int64 grid indices per pillar.

        """
        batch_size, max_points, _ = points_xyz.shape
        device = points_xyz.device
        dtype = points_xyz.dtype
        num_pillars = self.num_pillars
        sentinel_idx = num_pillars  # Extra bin for invalid points

        points_mask_bool = points_mask > 0

        if not points_xyz.is_cuda or not points_mask.is_cuda or not points_feature.is_cuda:
            raise RuntimeError("PillarPreprocessor requires CUDA inputs.")

        pillar_ids_raw, point_count_raw, _xyz_sum_raw, point_features = pillar_preprocess(
            points_mask_bool.contiguous(),
            points_xyz.contiguous().float(),
            points_feature.contiguous().float(),
            float(self.x_min),
            float(self.y_min),
            float(self.z_min),
            float(self.x_max),
            float(self.y_max),
            float(self.z_max),
            float(self.voxel_x),
            float(self.voxel_y),
            self.grid_x,
            self.grid_y,
        )

        pillar_ids = pillar_ids_raw
        valid_mask = pillar_ids >= 0
        pillar_ids_safe = torch.where(valid_mask, pillar_ids, torch.full_like(pillar_ids, sentinel_idx))

        max_points_t = torch.tensor(max_points, device=device, dtype=dtype)
        point_count = point_count_raw.to(dtype).clamp(max=max_points_t)

        pillar_masks = point_count > 0
        valid_mask_float = valid_mask.unsqueeze(-1).to(point_features.dtype)
        point_features = point_features * valid_mask_float

        pillar_indices = self.pillar_coords.to(device).unsqueeze(0).expand(batch_size, -1, -1)

        return {
            "pillar_masks": pillar_masks,
            "pillar_indices": pillar_indices,
            "point_features": point_features,
            "pillar_ids": pillar_ids_safe,
            "valid_mask": valid_mask,
        }
