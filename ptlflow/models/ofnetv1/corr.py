# =============================================================================
# Copyright 2024 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code adapted from RAFT: https://github.com/princeton-vl/RAFT/blob/master/core/corr.py
# =============================================================================

import math

import torch
import torch.nn.functional as F
from .utils import bilinear_sampler

try:
    import alt_cuda_corr
except:
    alt_cuda_corr = None
from ptlflow.utils.correlation import IterativeCorrBlock


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyCorrBlock:
    # Customized correlation block that doesn't compute all-pairs correlation but only samples a local neighborhood from fmap2 on the fly.
    def __init__(self, fmap1, fmap2, num_levels=1, radius=4):
        self.radius = radius
        self.num_levels = num_levels
        self.fmap1 = fmap1                              # [B, C, H, W]
        self.fmap2_pyramid = [fmap2]
        for _ in range(num_levels - 1):
            self.fmap2_pyramid.append(
                F.avg_pool2d(self.fmap2_pyramid[-1], 2, stride=2)
            )

    def __call__(self, coords):
        # coords: [B, 2, H, W] in pixel units of the current pyramid level
        B, _, H, W = coords.shape
        r = self.radius
        K = 2 * r + 1
        dtype, device = coords.dtype, coords.device
        C = self.fmap1.shape[1]

        dy = torch.linspace(-r, r, K, dtype=dtype, device=device)
        dx = torch.linspace(-r, r, K, dtype=dtype, device=device)
        dy, dx = torch.meshgrid(dy, dx, indexing="ij")
        # delta = torch.stack([dx, dy], dim=-1)           # [K, K, 2] Original implementation.
        delta = torch.stack([dy, dx], dim=-1)           # [K, K, 2] Mathamatically flawless but to make comply with original CorrBlock.

        # [B, H, W, 2]
        coords_perm = coords.permute(0, 2, 3, 1)

        out_levels = []
        for i in range(self.num_levels):
            fmap2_i = self.fmap2_pyramid[i]
            _, _, H2, W2 = fmap2_i.shape

            coords_i = coords_perm / (2 ** i)           # [B, H, W, 2]

            # Tile each (h, w) into a KxK block of sample locations.
            # broadcast: [B, H, 1, W, 1, 2] + [1, 1, K, 1, K, 2] -> [B, H, K, W, K, 2]
            grid = (
                coords_i[:, :, None, :, None, :]
                + delta[None, None, :, None, :, :]
            )
            grid = grid.reshape(B, H * K, W * K, 2)     # [B, H*K, W*K, 2]

            # Normalize to [-1, 1] for grid_sample.
            xn = 2.0 * grid[..., 0] / max(W2 - 1, 1) - 1.0
            yn = 2.0 * grid[..., 1] / max(H2 - 1, 1) - 1.0
            grid = torch.stack([xn, yn], dim=-1)

            warped = F.grid_sample(
                fmap2_i, grid, mode="bilinear",
                padding_mode="zeros", align_corners=True,
            )                                            # [B, C, H*K, W*K]

            warped = warped.view(B, C, H, K, W, K)
            warped = warped.permute(0, 1, 3, 5, 2, 4).contiguous()
            warped = warped.reshape(B, C, K * K, H, W)   # [B, C, K*K, H, W]

            # Per-pixel dot product with fmap1 -> correlation.
            # [B, C, 1, H, W] * [B, C, K*K, H, W] -> sum_C -> [B, K*K, H, W]
            corr = (self.fmap1.unsqueeze(2) * warped).sum(dim=1) / math.sqrt(C)
            out_levels.append(corr)

        return torch.cat(out_levels, dim=1)


class TIDLCorrBlock:
    # Correlation block that samples a local neighborhood from fmap2 on the fly, but computes correlation in the normalized coordinate space of fmap2.
    def __init__(self, fmap1, fmap2, num_levels=1, radius=4):
        self.radius = radius
        self.num_levels = num_levels
        self.fmap1 = fmap1
        self.fmap2_pyramid = [fmap2]
        for _ in range(num_levels - 1):
            self.fmap2_pyramid.append(F.avg_pool2d(self.fmap2_pyramid[-1], 2, stride=2))

    def __call__(self, coords0, flow):
        # coords0: [B,2,H,W]  Pixel coordinate grid (constant)
        # flow:    [B,2,H,W]  flow (in pixel units)
        B, _, H, W = coords0.shape
        r = self.radius
        K = 2 * r + 1
        dtype, device = coords0.dtype, coords0.device
        C = self.fmap1.shape[1]

        dy = torch.linspace(-r, r, K, dtype=dtype, device=device)
        dx = torch.linspace(-r, r, K, dtype=dtype, device=device)
        dy, dx = torch.meshgrid(dy, dx, indexing="ij")
        delta = torch.stack([dy, dx], dim=-1)

        coords0_perm = coords0.permute(0, 2, 3, 1)
        flow_perm = flow.permute(0, 2, 3, 1)

        out_levels = []
        for i in range(self.num_levels):
            fmap2_i = self.fmap2_pyramid[i]
            _, _, H2, W2 = fmap2_i.shape
            si = 1.0 / (2 ** i)
            sx = 2.0 / max(W2 - 1, 1)
            sy = 2.0 / max(H2 - 1, 1)

            # Normalized coords0 (coords0 is constant → can be folded as constant in ONNX)
            c0x = coords0_perm[..., 0] * (si * sx) - 1.0
            c0y = coords0_perm[..., 1] * (si * sy) - 1.0
            # flow contribution (only small scale is multiplied, no -1)
            fx = flow_perm[..., 0] * (si * sx)
            fy = flow_perm[..., 1] * (si * sy)
            # Normalized coordinates with flow contribution → ~[-1,1]
            coords_n = torch.stack([c0x + fx, c0y + fy], dim=-1) # [B,H,W,2]

            # delta also in normalized unit
            delta_n = torch.stack([delta[..., 0] * sx, delta[..., 1] * sy], dim=-1) # [K,K,2]

            grid = (coords_n[:, :, None, :, None, :] + delta_n[None, None, :, None, :, :]) # [B,H,K,W,K,2]
            grid = grid.reshape(B, H * K, W * K, 2) # ~[-1.1,1.1]

            warped = F.grid_sample(fmap2_i, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            warped = warped.view(B, C, H, K, W, K)
            warped = warped.permute(0, 1, 3, 5, 2, 4).contiguous()
            warped = warped.reshape(B, C, K * K, H, W)
            corr = (self.fmap1.unsqueeze(2) * warped).sum(dim=1) / math.sqrt(C)
            out_levels.append(corr)

        return torch.cat(out_levels, dim=1)


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(
                -r, r, 2 * r + 1, device=coords.device, dtype=coords.dtype
            )
            dy = torch.linspace(
                -r, r, 2 * r + 1, device=coords.device, dtype=coords.dtype
            )
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        # Lets set N = H * W
        # fmap1, fmap2: [B, C, N]
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        # [B, C, N]^T @ [B, C, N]
        # [B, N, C] @ [B, C, N] -> [B, N, N]
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        # [B, N, N] -> [B, H, W, 1, H, W]
        # Each entry in corr corresponds to the correlation between a pixel in fmap1 and a pixel in fmap2.
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / math.sqrt(dim)


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            if coords.dtype == torch.float16:
                fmap1_i = fmap1_i.float()
                fmap2_i = fmap2_i.float()
                coords_i = coords_i.float()
            (corr,) = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            if coords.dtype == torch.float16:
                corr = corr.half()
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / math.sqrt(dim)


def get_corr_block(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
    num_levels: int = 4,
    radius: int = 4,
    alternate_corr: bool = False,
):
    if alternate_corr:
        if alt_cuda_corr is None or fmap1.device == torch.device("cpu"):
            corr_fn = IterativeCorrBlock
        else:
            corr_fn = AlternateCorrBlock
    else:
        # corr_fn = CorrBlock
        # corr_fn = MyCorrBlock
        corr_fn = TIDLCorrBlock
    return corr_fn(fmap1=fmap1, fmap2=fmap2, radius=radius, num_levels=num_levels)
