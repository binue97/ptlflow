"""
WBNet-1: Warping-Based optical flow Network with pretrained ResNet backbone,
multi-scale warping refinement, depthwise separable decoder, and convex upsampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.registry import register_model, trainable
from ..base_model.base_model import BaseModel


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock."""

    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNetFPNEncoder(nn.Module):
    """
    Shared Siamese encoder based on ResNet18 with FPN-style multi-scale outputs.
    Produces features at 1/4, 1/8, 1/16 resolutions.
    """

    def __init__(
        self,
        pretrain="resnet18",
        norm_layer=nn.BatchNorm2d,
        init_weight=True,
    ):
        super().__init__()
        self.init_weight = init_weight

        if pretrain == "resnet18":
            dims = [64, 128, 256, 512]
            n_blocks = [2, 2, 2, 2]
        elif pretrain == "resnet34":
            dims = [64, 128, 256, 512]
            n_blocks = [3, 4, 6, 3]
        else:
            raise NotImplementedError(f"Unsupported pretrain: {pretrain}")

        self.in_planes = 64

        # Stem: 1/2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        # layer1: 1/2 (no stride change from stem)
        self.layer1 = self._make_layer(
            BasicBlock, dims[0], n_blocks[0], stride=1, norm_layer=norm_layer
        )
        # layer2: 1/4
        self.layer2 = self._make_layer(
            BasicBlock, dims[1], n_blocks[1], stride=2, norm_layer=norm_layer
        )
        # layer3: 1/8
        self.layer3 = self._make_layer(
            BasicBlock, dims[2], n_blocks[2], stride=2, norm_layer=norm_layer
        )
        # layer4: 1/16
        self.layer4 = self._make_layer(
            BasicBlock, dims[3], n_blocks[3], stride=2, norm_layer=norm_layer
        )

        # FPN lateral connections (project all levels to 128-ch)
        self.fpn_dim = 128
        self.lateral4 = nn.Conv2d(dims[3], self.fpn_dim, 1)
        self.lateral3 = nn.Conv2d(dims[2], self.fpn_dim, 1)
        self.lateral2 = nn.Conv2d(dims[1], self.fpn_dim, 1)

        # FPN smooth convolutions
        self.smooth4 = nn.Conv2d(self.fpn_dim, self.fpn_dim, 3, padding=1)
        self.smooth3 = nn.Conv2d(self.fpn_dim, self.fpn_dim, 3, padding=1)
        self.smooth2 = nn.Conv2d(self.fpn_dim, self.fpn_dim, 3, padding=1)

        self._init_weights(pretrain)

    def _make_layer(self, block, dim, num_blocks, stride, norm_layer):
        layers = [block(self.in_planes, dim, stride=stride, norm_layer=norm_layer)]
        self.in_planes = dim
        for _ in range(1, num_blocks):
            layers.append(block(dim, dim, stride=1, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _init_weights(self, pretrain):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.init_weight:
            from torchvision.models import (
                resnet18,
                ResNet18_Weights,
                resnet34,
                ResNet34_Weights,
            )

            if pretrain == "resnet18":
                pretrained_dict = resnet18(
                    weights=ResNet18_Weights.IMAGENET1K_V1
                ).state_dict()
            else:
                pretrained_dict = resnet34(
                    weights=ResNet34_Weights.IMAGENET1K_V1
                ).state_dict()

            model_dict = self.state_dict()
            # Only load matching keys (backbone weights, skip FPN)
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)

    def forward(self, x):
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))  # 1/2

        # Backbone stages
        c2 = self.layer1(x)   # 1/2, 64ch
        c3 = self.layer2(c2)  # 1/4, 128ch
        c4 = self.layer3(c3)  # 1/8, 256ch
        c5 = self.layer4(c4)  # 1/16, 512ch

        # FPN top-down
        p5 = self.lateral4(c5)  # 1/16, 128ch
        p4 = self.lateral3(c4) + F.interpolate(
            p5, size=c4.shape[2:], mode="bilinear", align_corners=False
        )  # 1/8, 128ch
        p3 = self.lateral2(c3) + F.interpolate(
            p4, size=c3.shape[2:], mode="bilinear", align_corners=False
        )  # 1/4, 128ch

        # Smooth
        p5 = self.smooth4(p5)
        p4 = self.smooth3(p4)
        p3 = self.smooth2(p3)

        return {"1/4": p3, "1/8": p4, "1/16": p5}


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise 3x3 + pointwise 1x1."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class RefineBlock(nn.Module):
    """
    Lightweight coarse-to-fine refinement block.
    Takes: f1 features, warped f2 features, current flow, and their difference.
    Predicts: residual delta flow.
    """

    def __init__(self, feat_ch, mid_ch=128):
        super().__init__()
        # Input: f1(feat_ch) + f2_warped(feat_ch) + diff(feat_ch) + flow(2)
        in_ch = feat_ch * 3 + 2
        self.net = nn.Sequential(
            DepthwiseSeparableConv(in_ch, mid_ch),
            DepthwiseSeparableConv(mid_ch, mid_ch),
            DepthwiseSeparableConv(mid_ch, mid_ch // 2),
            nn.Conv2d(mid_ch // 2, 2, kernel_size=3, padding=1),
        )

    def forward(self, f1, f2_warped, flow):
        diff = torch.abs(f1 - f2_warped)
        x = torch.cat([f1, f2_warped, diff, flow], dim=1)
        return self.net(x)


def warp(feat, flow):
    """
    Warp feature map feat using optical flow.
    feat: [B, C, H, W]
    flow: [B, 2, H, W]  (dx, dy in pixels at this resolution)
    """
    B, C, H, W = feat.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=feat.device, dtype=feat.dtype),
        torch.arange(W, device=feat.device, dtype=feat.dtype),
        indexing="ij",
    )
    grid_x = grid_x[None].expand(B, -1, -1)  # [B, H, W]
    grid_y = grid_y[None].expand(B, -1, -1)

    # Apply flow offset
    x = grid_x + flow[:, 0]
    y = grid_y + flow[:, 1]

    # Normalize to [-1, 1]
    x = 2.0 * x / (W - 1) - 1.0
    y = 2.0 * y / (H - 1) - 1.0

    grid = torch.stack([x, y], dim=-1)  # [B, H, W, 2]
    return F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class MultiScaleRefineLoss(nn.Module):
    """
    Multi-scale loss on the full-resolution flow predictions from each refinement stage,
    with exponentially decaying weights for earlier (coarser) stages.
    """

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


class WBNet1(BaseModel):
    pretrained_checkpoints = {}

    def __init__(
        self,
        pretrain: str = "resnet18",
        init_weight: bool = True,
        feat_dim: int = 128,
        refine_mid_ch: int = 128,
        gamma: float = 0.8,
        max_flow: float = 400.0,
        **kwargs,
    ):
        super().__init__(
            output_stride=16,
            loss_fn=MultiScaleRefineLoss(gamma=gamma, max_flow=max_flow),
            **kwargs,
        )

        self.pretrain = pretrain
        self.feat_dim = feat_dim

        # Shared siamese encoder with FPN
        self.encoder = ResNetFPNEncoder(
            pretrain=pretrain,
            norm_layer=nn.BatchNorm2d,
            init_weight=init_weight,
        )

        # Initial flow predictor at 1/16
        self.init_flow_head = nn.Sequential(
            DepthwiseSeparableConv(feat_dim * 2, refine_mid_ch),
            DepthwiseSeparableConv(refine_mid_ch, refine_mid_ch),
            nn.Conv2d(refine_mid_ch, 2, kernel_size=3, padding=1),
        )

        # Refinement blocks for each scale
        self.refine_8 = RefineBlock(feat_dim, refine_mid_ch)
        self.refine_4 = RefineBlock(feat_dim, refine_mid_ch)

        # Convex upsampling head: 1/4 → full resolution (4x upsample)
        # For each pixel at 1/4, predict weights over 3x3 neighborhood
        # to produce 4x4 output patch = 9 * 16 = 144 channels
        self.upsample_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim * 2, 16 * 9, 1, padding=0),
        )

    def convex_upsample(self, flow, mask, scale=4):
        """Upsample flow field [H/s, W/s, 2] -> [H, W, 2] using convex combination."""
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

        image1 = images[:, 0]  # [B, 3, H, W]
        image2 = images[:, 1]

        # Shared feature extraction
        feats1 = self.encoder(image1)  # {"1/4": ..., "1/8": ..., "1/16": ...}
        feats2 = self.encoder(image2)

        flow_preds = []

        # --- Stage 1: Initial flow at 1/16 ---
        f1_16 = feats1["1/16"]
        f2_16 = feats2["1/16"]
        init_input = torch.cat([f1_16, f2_16], dim=1)
        flow_16 = self.init_flow_head(init_input)  # [B, 2, H/16, W/16]

        # Upsample to full res for loss
        flow_up = F.interpolate(
            flow_16 * 16, size=image1.shape[2:], mode="bilinear", align_corners=False
        )
        flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)
        flow_preds.append(flow_up)

        # --- Stage 2: Refine at 1/8 ---
        f1_8 = feats1["1/8"]
        f2_8 = feats2["1/8"]
        # Upsample flow from 1/16 to 1/8
        flow_8 = F.interpolate(
            flow_16 * 2.0, size=f1_8.shape[2:], mode="bilinear", align_corners=False
        )
        # Warp f2 features using current flow estimate
        f2_8_warped = warp(f2_8, flow_8)
        delta_flow_8 = self.refine_8(f1_8, f2_8_warped, flow_8)
        flow_8 = flow_8 + delta_flow_8

        flow_up = F.interpolate(
            flow_8 * 8, size=image1.shape[2:], mode="bilinear", align_corners=False
        )
        flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)
        flow_preds.append(flow_up)

        # --- Stage 3: Refine at 1/4 ---
        f1_4 = feats1["1/4"]
        f2_4 = feats2["1/4"]
        # Upsample flow from 1/8 to 1/4
        flow_4 = F.interpolate(
            flow_8 * 2.0, size=f1_4.shape[2:], mode="bilinear", align_corners=False
        )
        f2_4_warped = warp(f2_4, flow_4)
        delta_flow_4 = self.refine_4(f1_4, f2_4_warped, flow_4)
        flow_4 = flow_4 + delta_flow_4

        # --- Convex upsampling 1/4 → full ---
        upsample_mask = 0.25 * self.upsample_head(f1_4)
        out_flow = self.convex_upsample(flow_4, upsample_mask, scale=4)
        out_flow = self.postprocess_predictions(out_flow, image_resizer, is_flow=True)
        flow_preds.append(out_flow)

        outputs = {}
        if self.training:
            outputs["flow_preds"] = flow_preds
            outputs["flows"] = out_flow[:, None]
        else:
            outputs["flows"] = out_flow[:, None]

        return outputs


@register_model
@trainable
class wbnet_1(WBNet1):
    pass
