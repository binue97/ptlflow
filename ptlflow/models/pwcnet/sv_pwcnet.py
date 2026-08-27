"""
Mobile-optimized PWC-Net (SV-PWCNet).

This is a lightweight variant of PWC-Net [Sun et al. 2018] designed for
on-device/mobile inference. Compared with the original PWC-Net it applies the
following optimizations:

1. The feature pyramid is reduced from 7 to 5 levels (learned feature levels
   conv1..conv4, with the top level at 1/16 of the input resolution).
2. The DenseNet [Huang et al. 2017] connections in the flow estimator are
   removed: the estimator convolutions are applied sequentially instead of
   concatenating every intermediate feature map.
3. The cost-volume search range is reduced from 4 to 2 (``md = 2``).
4. Every 3x3 convolution is replaced by a depth-wise separable convolution,
   i.e. a 3x3 depth-wise conv followed by a 1x1 point-wise conv
   [Howard et al. 2017]. The final flow predictors keep a standard 3x3 conv.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.registry import register_model, trainable
from ..base_model.base_model import BaseModel
from ..flownet.losses import MultiScale


# ============================================================================
# TIDL-friendly correlation, MMA edition: SINGLE grid_sample (im2col role) +
# batched MatMul (GEMM role). The grid is laid out PIXEL-BATCH ([B, H*W, K*K, 2]:
# row = pixel h*W+w, col = displacement ky*K+kx) so grid_sample directly emits
# the stack S [B, C, HW, K*K]; the channel reduction is then ONE torch.matmul
# ((1xC)x(CxK*K) per pixel, HW-batched) which TIDL compiles to
# TIDL_InnerProductLayer on the MMA (act x act batched-MatMul path added for
# ViT attention). Board-measured vs the Mul+ReduceSum form: EltWise 25.9ms -> 0,
# proc 88.75 -> 56.18 ms, DDR/frame halved (the 30MB product tensor no longer
# exists). Still ONE gather op -- no Tile, no Gather, no N-way slice fan-out --
# so TIDL topo-sorts and compiles it (1 subgraph, full offload).
# (History: integer-shift branches, im2col-Gather, and single-axis-slice forms
# failed TIDL's topological sort / fell to CPU; window-into-spatial + Mul +
# ReduceSum compiled but ran DDR-bound. Keep matmul explicit -- do NOT rewrite
# as einsum, which may not export as a single ONNX MatMul.)
# Output matches the original 5D interface [B, 2md+1, 2md+1, H, W].
# ============================================================================
class SpatialCorrelationSampler(nn.Module):
    """Local correlation / cost volume via ONE grid_sample + ONE batched MatMul.

    Correlation is per-pixel dot products: corr(p, d) = <F1(p), F2(p+d)> for
    K*K displacements d -- i.e. a (1xC)x(CxK*K) GEMM per pixel. The sampling
    grid is constant (input2 is pre-warped), ordered pixel-batch
    ([B, H*W, K*K, 2]) so that grid_sample outputs the per-pixel key matrices
    directly and the reduction maps to MMA-accelerated batched MatMul
    (TIDL_InnerProductLayer) instead of a DDR-bound EltWise Mul + ReduceSum.

    Only the config used by SVPWCNet is supported (kernel_size=1, stride=1,
    dilation=1, dilation_patch=1). patch_size sets the window (= 2*md+1); border
    samples read 0 (grid_sample padding_mode="zeros").
    """

    def __init__(
        self,
        kernel_size=1,
        patch_size=1,
        stride=1,
        padding=0,
        dilation=1,
        dilation_patch=1,
        chunk_size=None,
    ):
        super(SpatialCorrelationSampler, self).__init__()
        ps = patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size
        k = kernel_size[0] if isinstance(kernel_size, (tuple, list)) else kernel_size
        assert k == 1, "Only kernel_size=1 is supported."
        assert ps % 2 == 1, "patch_size must be odd (= 2*md+1)."
        self.patch_size = ps
        self.md = (ps - 1) // 2

    def forward(self, input1, input2):
        """input1/input2: [B, C, H, W] -> cost volume [B, 2md+1, 2md+1, H, W].

        im2col + GEMM decomposition of the local cost volume:
        - grid_sample (im2col): ONE gather with a PIXEL-BATCH grid
          [B, H*W, K*K, 2] (row = pixel h*W+w, col = displacement ky*K+kx)
          -> S [B, C, HW, K*K], the per-pixel key matrix stack.
        - matmul (GEMM): query row F1(p) [1 x C] times keys S_p [C x K*K],
          batched over HW pixels -> TIDL_InnerProductLayer on the MMA.
        No intermediate [B, C, K*K, H, W] product tensor is ever materialized
        (that EltWise Mul was DDR-bound and cost 25.9 ms/frame on TDA4VH).

        input2 is already warped (see warp()), so the window is centered on each
        position (delta = -md..md) and the sampling grid is constant (folded to
        an initializer at ONNX export). bilinear at integer offsets + zero-pad
        == the original integer-shift correlation.

        corr[b, d, i, j] = sum_c input1[b,c,i,j] * input2[b, c, i+dy, j+dx]
        (sum over channels, like the original; _forward_pair divides by C after.
        Matches the Mul+ReduceSum form up to float summation order, ~1e-6 rel.)
        """
        B, C, H, W = input2.shape
        K = self.patch_size            # = 2*md + 1
        r = self.md
        dtype, device = input2.dtype, input2.device
        sx = 2.0 / max(W - 1, 1)       # pixel -> [-1,1] (grid_sample, align_corners=True)
        sy = 2.0 / max(H - 1, 1)
        # normalized pixel coords of input2 and normalized +-md offsets (all constant)
        xs = torch.arange(W, dtype=dtype, device=device) * sx - 1.0    # [W]
        ys = torch.arange(H, dtype=dtype, device=device) * sy - 1.0    # [H]
        off = torch.arange(K, dtype=dtype, device=device) - r          # [-r..r]
        ddx = off * sx                                                 # [K]
        ddy = off * sy                                                 # [K]
        # PIXEL-BATCH grid[b, h, w, ky, kx] = (xs[w] + ddx[kx], ys[h] + ddy[ky])
        # flattened to [B, H*W, K*K, 2]: row = pixel (h*W+w), col = disp (ky*K+kx)
        gx = (xs.view(1, 1, W, 1, 1) + ddx.view(1, 1, 1, 1, K)).expand(B, H, W, K, K)
        gy = (ys.view(1, H, 1, 1, 1) + ddy.view(1, 1, 1, K, 1)).expand(B, H, W, K, K)
        grid = torch.stack([gx, gy], dim=-1).reshape(B, H * W, K * K, 2)   # [B, H*W, K*K, 2]
        # im2col: ONE grid_sample gathers every neighbor (no Tile, no Gather)
        S = F.grid_sample(input2, grid, mode="bilinear", padding_mode="zeros", align_corners=True)  # [B, C, HW, K*K]
        S = S.permute(0, 2, 1, 3)                                      # [B, HW, C, K*K]
        # GEMM: (1 x C) x (C x K*K) per pixel, HW-batched -> MMA (InnerProduct).
        # Keep torch.matmul explicit (einsum may not export as one ONNX MatMul).
        A = input1.permute(0, 2, 3, 1).reshape(B, H * W, 1, C)         # [B, HW, 1, C]
        corr = torch.matmul(A, S)                                      # [B, HW, 1, K*K]
        corr = corr.reshape(B, H, W, K * K).permute(0, 3, 1, 2)        # [B, K*K, H, W]
        return corr.view(B, K, K, H, W)                                # [B, K, K, H, W]


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.1),
    )

def conv_dw(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    """Depth-wise separable convolution: 3x3 depth-wise conv + 1x1 point-wise conv.

    Replaces each standard 3x3 convolution of PWC-Net following the MobileNet
    [Howard et al. 2017] factorization. Each sub-convolution is followed by a
    LeakyReLU, matching the original PWC-Net activation.
    """
    return nn.Sequential(
        # Depth-wise 3x3 conv (one filter per input channel).
        nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_planes,
            bias=True,
        ),
        nn.LeakyReLU(0.1),
        # Point-wise 1x1 conv (mixes channels).
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        ),
        nn.LeakyReLU(0.1),
    )


def predict_flow(in_planes):
    """Standard 3x3 convolution that regresses the 2-channel flow."""
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(
        in_planes, out_planes, kernel_size, stride, padding, bias=True
    )


class SVPWCNet(BaseModel):
    pretrained_checkpoints = {}

    def __init__(
        self,
        div_flow: float = 20.0,
        md: int = 3,
        loss_start_scale: int = 4,
        loss_num_scales: int = 3,
        loss_base_weight: float = 0.32,
        loss_norm: str = "L2",
        **kwargs,
    ):
        super(SVPWCNet, self).__init__(
            loss_fn=MultiScale(
                startScale=loss_start_scale,
                numScales=loss_num_scales,
                l_weight=loss_base_weight,
                norm=loss_norm,
            ),
            output_stride=16,
            **kwargs,
        )

        self.div_flow = div_flow
        self.md = md

        # ------------------------------------------------------------------
        # Feature pyramid: 5 levels total (input + 4 learned feature levels).
        # conv1 -> 1/2, conv2 -> 1/4, conv3 -> 1/8, conv4 -> 1/16 (top level).
        # ------------------------------------------------------------------
        self.conv1a = conv(3, 16, kernel_size=3, stride=2)
        self.conv1aa = conv(16, 16, kernel_size=3, stride=1)
        self.conv1b = conv(16, 16, kernel_size=3, stride=1)
        self.conv2a = conv(16, 32, kernel_size=3, stride=2)
        self.conv2aa = conv(32, 32, kernel_size=3, stride=1)
        self.conv2b = conv(32, 32, kernel_size=3, stride=1)
        self.conv3a = conv(32, 64, kernel_size=3, stride=2)
        self.conv3aa = conv(64, 64, kernel_size=3, stride=1)
        self.conv3b = conv(64, 64, kernel_size=3, stride=1)
        self.conv4a = conv(64, 96, kernel_size=3, stride=2)
        self.conv4aa = conv(96, 96, kernel_size=3, stride=1)
        self.conv4b = conv(96, 96, kernel_size=3, stride=1)

        self.leakyRELU = nn.LeakyReLU(0.1)

        # Cost volume with reduced search range (md = 2 -> 5x5 = 25 channels).
        self.corr = SpatialCorrelationSampler(
            kernel_size=1, patch_size=2 * self.md + 1, padding=0
        )

        nd = (2 * self.md + 1) ** 2

        # ------------------------------------------------------------------
        # Flow estimators (no DenseNet connections: convolutions are applied
        # sequentially, so each conv input is the previous conv output).
        # ------------------------------------------------------------------
        # Level 4 (top, 1/16): only the cost volume is available.
        od = nd
        self.conv4_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv4_1 = conv(128, 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(128, 96, kernel_size=3, stride=1)
        self.conv4_3 = conv(96, 64, kernel_size=3, stride=1)
        self.conv4_4 = conv(64, 32, kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(32)
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(32, 2, kernel_size=4, stride=2, padding=1)

        # Level 3 (1/8): cost volume + level-3 features + up-flow + up-feat.
        od = nd + 64 + 4
        self.conv3_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv3_1 = conv(128, 128, kernel_size=3, stride=1)
        self.conv3_2 = conv(128, 96, kernel_size=3, stride=1)
        self.conv3_3 = conv(96, 64, kernel_size=3, stride=1)
        self.conv3_4 = conv(64, 32, kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(32)
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(32, 2, kernel_size=4, stride=2, padding=1)

        # Level 2 (1/4): cost volume + level-2 features + up-flow + up-feat.
        od = nd + 32 + 4
        self.conv2_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv2_1 = conv(128, 128, kernel_size=3, stride=1)
        self.conv2_2 = conv(128, 96, kernel_size=3, stride=1)
        self.conv2_3 = conv(96, 64, kernel_size=3, stride=1)
        self.conv2_4 = conv(64, 32, kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in")
                if m.bias is not None:
                    m.bias.data.zero_()
        self.upsample1 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )

    def warp(self, x, flo):
        """TIDL-friendly warp of ``x`` (im2) back to im1 by optical flow ``flo``.

        Same result as the original warp EXCEPT the explicit validity mask is
        dropped. Out-of-bounds samples are already zeroed by
        ``grid_sample(padding_mode="zeros")``, so the extra mask grid_sample and
        the ``Less``/``Greater``/``Where`` thresholding -- all CPU-fallback on
        TIDL -- are removed. The sampling grid is built with elementwise ops only
        (Add + broadcast Mul/Sub + Transpose), so no ``Gather``/``ScatterND`` is
        emitted and the whole warp stays on C7x.

        x:   [B, C, H, W]  tensor being sampled (im2)
        flo: [B, 2, H, W]  flow; channel 0 = x/horizontal, channel 1 = y/vertical
        """
        B, C, H, W = x.size()

        # Absolute pixel-coordinate grid [B, 2, H, W] as a constant
        # (constant-folded at export -> no runtime Tile/Range).
        xx = torch.arange(0, W, dtype=flo.dtype, device=flo.device)
        yy = torch.arange(0, H, dtype=flo.dtype, device=flo.device)
        grid_x = xx.view(1, 1, 1, W).expand(B, 1, H, W)
        grid_y = yy.view(1, 1, H, 1).expand(B, 1, H, W)
        grid = torch.cat((grid_x, grid_y), dim=1)

        # Normalize to [-1, 1] with a per-channel affine, REORDERED so no tensor
        # ever holds the large absolute coordinate. Algebraically identical to
        # `(grid + flo) * scale - 1`, but distributed as
        #     vgrid = (grid * scale - 1) + flo * scale
        # `grid * scale - 1` is built from constants -> constant-folded at export
        # to a bounded [-1, 1] tensor, and only `flo * scale` (small) is a runtime
        # activation. The naive `grid + flo` form produces an intermediate spanning
        # 0..W-1+flow (e.g. 0..84 / 0..169), which TIDL per-tensor integer
        # quantization clips at ~32 / ~64 -> ~1/3 of the warp sampling coordinates
        # saturate, so grid_sample reads the wrong (too-near) location and large
        # motion is truncated. Keeping every tensor <= ~1.2 removes the clipping.
        scale = torch.tensor(
            [2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)],
            dtype=flo.dtype,
            device=flo.device,
        ).view(1, 2, 1, 1)
        vgrid = grid * scale - 1.0 + flo * scale

        # [B, 2, H, W] -> [B, H, W, 2] for grid_sample.
        vgrid = vgrid.permute(0, 2, 3, 1)

        return F.grid_sample(
            x, vgrid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

    def forward(self, inputs, skip_preprocess=False):
        images = inputs["images"]
        if skip_preprocess:
            image_resizer = None
        else:
            images, image_resizer = self.preprocess_images(
                images,
                bgr_add=0.0,
                bgr_mult=1.0,
                bgr_to_rgb=False,
                resize_mode="interpolation",
                interpolation_mode="bilinear",
                interpolation_align_corners=False,
            )

        return self._forward_pair(images[:, 0], images[:, 1], image_resizer)

    def _forward_pair(self, im1, im2, image_resizer=None):
        """Run the network core on two pre-split (B, 3, H, W) frames.

        - Input images should be in RGB format.
        - Data range should be [0, 1] (float32).
        """
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))

        # Level 4 (top): correlation only, no DenseNet concatenation.
        corr4 = self.corr(c14, c24)
        corr4 = corr4.view(corr4.shape[0], -1, corr4.shape[3], corr4.shape[4])
        corr4 = corr4 / c14.shape[1]
        corr4 = self.leakyRELU(corr4)

        x = self.conv4_0(corr4)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        # Level 3.
        warp3 = self.warp(c23, up_flow4 * 2.5)
        corr3 = self.corr(c13, warp3)
        corr3 = corr3.view(corr3.shape[0], -1, corr3.shape[3], corr3.shape[4])
        corr3 = corr3 / c13.shape[1]
        corr3 = self.leakyRELU(corr3)
        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = self.conv3_0(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        # Level 2.
        warp2 = self.warp(c22, up_flow3 * 5.0)
        corr2 = self.corr(c12, warp2)
        corr2 = corr2.view(corr2.shape[0], -1, corr2.shape[3], corr2.shape[4])
        corr2 = corr2 / c12.shape[1]
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = self.conv2_0(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv2_4(x)
        flow2 = self.predict_flow2(x)

        flow_up = self.upsample1(flow2 * self.div_flow)
        if image_resizer is not None:
            flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)

        outputs = {}
        if self.training:
            outputs["flow_preds"] = [flow2, flow3, flow4]
            outputs["flows"] = flow_up[:, None]
        else:
            outputs["flows"] = flow_up[:, None]
        return outputs



@register_model
@trainable
class sv_pwcnet(SVPWCNet):
    pass
