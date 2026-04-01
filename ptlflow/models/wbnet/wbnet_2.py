"""
WBNet-2: Efficient Warping-Based optical flow Network with lightweight MobileNet-style
encoder, iterative warp-based refinement (inspired by WAFT), and convex upsampling.

Key improvements over WBNet-1:
- Lightweight encoder using inverted residual blocks (~5-10x fewer encoder FLOPs)
- Iterative warping refinement with ConvGRU hidden state (6 iters vs 2 fixed stages)
- Warping-only approach (no correlation volume, inspired by WAFT)
- Single-scale operation at 1/8 with 8x convex upsampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.registry import register_model, trainable
from ..base_model.base_model import BaseModel


# ---------------------------------------------------------------------------
# Lightweight encoder building blocks
# ---------------------------------------------------------------------------


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block:
    1x1 expand -> 3x3 depthwise -> 1x1 project, with residual when possible.
    """

    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
        super().__init__()
        self.use_residual = stride == 1 and in_ch == out_ch
        hidden_ch = in_ch * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                    nn.BatchNorm2d(hidden_ch),
                    nn.ReLU6(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    hidden_ch,
                    hidden_ch,
                    3,
                    stride=stride,
                    padding=1,
                    groups=hidden_ch,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class LightweightEncoder(nn.Module):
    """Lightweight MobileNetV2-style encoder producing features at 1/8 resolution.

    Architecture:
        Stem:    3 -> 32,  stride 2  -> 1/2
        Stage 1: 32 -> 24, stride 1  -> 1/2  (x2 blocks)
        Stage 2: 24 -> 48, stride 2  -> 1/4  (x3 blocks)
        Stage 3: 48 -> 96, stride 2  -> 1/8  (x4 blocks)
        Project: 96 -> out_dim                -> 1/8
    """

    def __init__(self, out_dim=96):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.stage1 = nn.Sequential(
            InvertedResidual(32, 24, stride=1, expand_ratio=1),
            InvertedResidual(24, 24, stride=1, expand_ratio=6),
        )
        self.stage2 = nn.Sequential(
            InvertedResidual(24, 48, stride=2, expand_ratio=6),
            InvertedResidual(48, 48, stride=1, expand_ratio=6),
            InvertedResidual(48, 48, stride=1, expand_ratio=6),
        )
        self.stage3 = nn.Sequential(
            InvertedResidual(48, 96, stride=2, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(96, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU6(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# Iterative refinement blocks (WAFT-inspired)
# ---------------------------------------------------------------------------


class ConvGRU(nn.Module):
    """Convolutional GRU for iterative hidden state update."""

    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


def warp(feat, flow):
    """Warp feature map *feat* using optical flow (in pixels at current resolution)."""
    B, _C, H, W = feat.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=feat.device, dtype=feat.dtype),
        torch.arange(W, device=feat.device, dtype=feat.dtype),
        indexing="ij",
    )
    grid_x = grid_x[None].expand(B, -1, -1)
    grid_y = grid_y[None].expand(B, -1, -1)

    x = grid_x + flow[:, 0]
    y = grid_y + flow[:, 1]

    x = 2.0 * x / (W - 1) - 1.0
    y = 2.0 * y / (H - 1) - 1.0

    grid = torch.stack([x, y], dim=-1)
    return F.grid_sample(
        feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class SequenceLoss(nn.Module):
    """Exponentially-weighted L1 loss over the sequence of flow predictions."""

    def __init__(self, gamma=0.8, max_flow=400.0):
        super().__init__()
        self.gamma = gamma
        self.max_flow = max_flow

    def forward(self, outputs, inputs):
        flow_preds = outputs["flow_preds"]
        flow_gt = inputs["flows"][:, 0]
        valid = inputs["valids"][:, 0]

        n_predictions = len(flow_preds)
        flow_loss = 0.0

        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid * i_loss).mean()

        return flow_loss


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class WBNet2(BaseModel):
    """WBNet-2: lightweight encoder + iterative warp refinement.

    Architecture overview
    ---------------------
    1. Shared lightweight encoder (MobileNetV2-style, wider) extracts features at 1/8.
    2. Separate context encoder on image-1 initialises the ConvGRU hidden state.
    3. At each iteration:
       a. Current flow is detached (stop gradient).
       b. Image-2 features are warped to image-1 using the current flow.
       c. A motion encoder fuses [f1, f2_warped, |f1-f2_warped|, flow].
       d. A ConvGRU updates the hidden state with the motion features.
       e. A flow head predicts a residual flow update from the hidden state.
       f. Convex upsampling yields a full-resolution prediction for the loss.
    4. Output: final full-resolution flow prediction.
    """

    pretrained_checkpoints = {}

    def __init__(
        self,
        feat_dim: int = 96,
        hidden_dim: int = 96,
        iters: int = 6,
        gamma: float = 0.8,
        max_flow: float = 400.0,
        **kwargs,
    ):
        super().__init__(
            output_stride=8,
            loss_fn=SequenceLoss(gamma=gamma, max_flow=max_flow),
            **kwargs,
        )

        self.iters = iters
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # --- Shared siamese encoder ---
        self.encoder = LightweightEncoder(out_dim=feat_dim)

        # --- Context encoder: separate network on image-1 for GRU init ---
        self.context_encoder = LightweightEncoder(out_dim=hidden_dim)
        self.context_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Tanh(),
        )

        # --- Motion encoder: [f1, f2_warped, |f1 - f2_warped|, flow] ---
        motion_in_ch = feat_dim * 3 + 2  # f1 + f2_warped + diff + flow
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(motion_in_ch, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
        )

        # --- ConvGRU ---
        self.gru = ConvGRU(hidden_dim, hidden_dim)

        # --- Flow head: hidden -> residual flow ---
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 3, padding=1),
        )

        # --- Convex upsampling mask head: 1/8 -> full (8x) ---
        # For each 1/8-pixel, predict soft weights over a 3x3 neighbourhood
        # to produce an 8x8 output patch  =>  9 * 64 = 576 channels.
        self.upsample_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, 8 * 8 * 9, 1),
        )

    @staticmethod
    def convex_upsample(flow, mask, scale=8):
        """Upsample flow [H/s, W/s, 2] -> [H, W, 2] via learned convex combination."""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(scale * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, scale * H, scale * W)

    def forward(self, inputs):
        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=True,
            resize_mode="pad",
            pad_mode="replicate",
            pad_two_side=True,
        )

        image1 = images[:, 0]
        image2 = images[:, 1]
        N, _, H, W = image1.shape

        # --- Feature extraction (shared encoder, 1/8 resolution) ---
        f1 = self.encoder(image1)
        f2 = self.encoder(image2)

        # --- Initialise hidden state from separate context encoder ---
        ctx = self.context_encoder(image1)
        h = self.context_head(ctx)
        flow = torch.zeros(N, 2, H // 8, W // 8, device=image1.device)

        flow_preds = []

        for _itr in range(self.iters):
            flow = flow.detach()

            # Warp image-2 features to image-1 using current flow
            f2_warped = warp(f2, flow)

            # Encode motion cues (including |f1 - f2_warped| difference)
            diff = torch.abs(f1 - f2_warped)
            motion = self.motion_encoder(
                torch.cat([f1, f2_warped, diff, flow], dim=1)
            )

            # Update hidden state
            h = self.gru(h, motion)

            # Predict residual flow
            delta_flow = self.flow_head(h)
            flow = flow + delta_flow

            # Convex upsample to full resolution
            up_mask = 0.125 * self.upsample_head(h)
            flow_up = self.convex_upsample(flow, up_mask, scale=8)
            flow_up = self.postprocess_predictions(
                flow_up, image_resizer, is_flow=True
            )
            flow_preds.append(flow_up)

        outputs = {}
        if self.training:
            outputs["flow_preds"] = flow_preds
            outputs["flows"] = flow_preds[-1][:, None]
        else:
            outputs["flows"] = flow_preds[-1][:, None]

        return outputs


@register_model
@trainable
class wbnet_2(WBNet2):
    pass
