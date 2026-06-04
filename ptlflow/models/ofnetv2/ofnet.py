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
# Adapted from RAFT: https://github.com/princeton-vl/RAFT
# =============================================================================

import math

from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.registry import register_model, trainable, ptlflow_trained
from ptlflow.utils.utils import forward_interpolate_batch
from .pwc_modules import rescale_flow
from .update import UpdateBlock
from .corr import get_corr_block
from .local_timm.norm import LayerNorm2d
from ..base_model.base_model import BaseModel

from .next1d_encoder import NeXt1DEncoder
from .next1d import NeXt1DStage

try:
    import alt_cuda_corr
except:
    alt_cuda_corr = None


class SequenceLoss(nn.Module):
    def __init__(self, gamma: float, max_flow: float):
        super().__init__()
        self.gamma = gamma
        self.max_flow = max_flow

    def forward(self, outputs, inputs):
        """Loss function defined over sequence of flow predictions"""

        flow_preds = outputs["flow_preds"]
        flow_gt = inputs["flows"][:, 0]
        valid = inputs["valids"][:, 0]

        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid * i_loss).mean()

        return flow_loss


class OFNetV2(BaseModel):
    pretrained_checkpoints = {
    }

    def __init__(
        self,
        pyramid_ranges: tuple[int, int] = (32, 8),
        iters: int = 6,
        corr_mode: str = "allpairs",
        corr_levels: int = 1,
        corr_range: int = 4,
        enc_hidden_chs: int = 64,
        enc_out_chs: int = 128,
        enc_stem_stride: int = 4,
        enc_mlp_ratio: float = 4.0,
        enc_depth: int = 4,
        dec_net_chs: int = 64,
        dec_inp_chs: int = 64,
        dec_motion_chs: int = 128,
        dec_depth: int = 2,
        dec_mlp_ratio: float = 4.0,
        fuse_next1d_weights: bool = False,
        simple_io: bool = False,
        gamma: float = 0.8,
        max_flow: float = 400,
        **kwargs,
    ) -> None:
        num_recurrent_layers = int(math.log2(max(pyramid_ranges))) - 1
        super().__init__(
            output_stride=int(2 ** (num_recurrent_layers + 1)),
            loss_fn=SequenceLoss(gamma, max_flow),
            **kwargs,
        )

        self.pyramid_ranges = pyramid_ranges
        self.iters = iters
        self.corr_mode = corr_mode
        self.corr_levels = corr_levels
        self.corr_range = corr_range
        self.enc_hidden_chs = enc_hidden_chs
        self.enc_out_chs = enc_out_chs
        self.enc_stem_stride = enc_stem_stride
        self.enc_mlp_ratio = enc_mlp_ratio
        self.enc_depth = enc_depth
        self.dec_net_chs = dec_net_chs
        self.dec_inp_chs = dec_inp_chs
        self.dec_motion_chs = dec_motion_chs
        self.dec_depth = dec_depth
        self.dec_mlp_ratio = dec_mlp_ratio
        self.fuse_next1d_weights = fuse_next1d_weights
        self.simple_io = simple_io

        for v in self.pyramid_ranges:
            assert (
                v > 1
            ), f"--pyramid_ranges values must be larger than 1, but found {v}"
            log_res = math.log2(v)
            assert (log_res) - int(
                log_res
            ) < 1e-3, f"--pyramid_ranges values must be powers of 2, but found {v}"
        num_recurrent_layers = int(math.log2(max(self.pyramid_ranges))) - 1

        self.pyramid_levels = [
            num_recurrent_layers + 1 - int(math.log2(v)) for v in self.pyramid_ranges
        ]

        max_pyr_range = (min(self.pyramid_ranges), max(self.pyramid_ranges))
        self.fnet = NeXt1DEncoder(
            max_pyr_range=max_pyr_range,
            stem_stride=self.enc_stem_stride,
            num_recurrent_layers=num_recurrent_layers,
            hidden_chs=self.enc_hidden_chs,
            out_chs=self.enc_out_chs,
            mlp_ratio=self.enc_mlp_ratio,
            norm_layer=LayerNorm2d,
            depth=self.enc_depth,
            fuse_next1d_weights=self.fuse_next1d_weights,
        )

        self.cnet = NeXt1DEncoder(
            max_pyr_range=max_pyr_range,
            stem_stride=self.enc_stem_stride,
            num_recurrent_layers=num_recurrent_layers,
            hidden_chs=self.enc_hidden_chs,
            out_chs=self.enc_out_chs,
            mlp_ratio=self.enc_mlp_ratio,
            norm_layer=LayerNorm2d,
            depth=self.enc_depth,
            fuse_next1d_weights=self.fuse_next1d_weights,
        )

        self.dim_corr = (self.corr_range * 2 + 1) ** 2 * self.corr_levels

        self.update_block = UpdateBlock(
            pyramid_ranges=pyramid_ranges,
            corr_levels=corr_levels,
            corr_range=corr_range,
            dec_net_chs=dec_net_chs,
            dec_inp_chs=dec_inp_chs,
            dec_motion_chs=dec_motion_chs,
            dec_depth=dec_depth,
            dec_mlp_ratio=dec_mlp_ratio,
            fuse_next1d_weights=fuse_next1d_weights,
        )

        self.upnet_layer = nn.Sequential(
            nn.Conv2d(2 * self.dec_net_chs, self.dec_net_chs, 1),
            nn.ReLU(inplace=True),
            NeXt1DStage(
                self.dec_net_chs,
                self.dec_net_chs,
                stride=1,
                depth=2,
                mlp_ratio=self.dec_mlp_ratio,
                norm_layer=LayerNorm2d,
                fuse_next1d_weights=self.fuse_next1d_weights,
            ),
        )

        self.has_trained_on_ptlflow = True

        if self.corr_mode == "local" and alt_cuda_corr is None:
            logger.warning(
                "!!! alt_cuda_corr is not compiled! The slower IterativeCorrBlock will be used instead !!!"
            )

    def coords_grid(self, batch, ht, wd, dtype, device):
        coords = torch.meshgrid(
            # zero to ht-1
            torch.arange(ht, dtype=dtype, device=device),
            # zero to wd-1
            torch.arange(wd, dtype=dtype, device=device),
            indexing="ij",
        )
        coords = torch.stack(coords[::-1], dim=0).to(dtype=dtype)
        return coords[None].repeat(batch, 1, 1, 1)


    def forward(self, inputs, image2=None):
        if self.simple_io:
            # Preprocess each image independently — no stacking needed.
            x1_raw, image_resizer = self.preprocess_images(
                inputs[:, None],
                bgr_add=-0.5,
                bgr_mult=2.0,
                bgr_to_rgb=False,
                resize_mode="pad",
                pad_mode="replicate",
                pad_two_side=True,
            )
            x2_raw, _ = self.preprocess_images(
                image2[:, None],
                bgr_add=-0.5,
                bgr_mult=2.0,
                bgr_to_rgb=False,
                resize_mode="pad",
                pad_mode="replicate",
                pad_two_side=True,
            )
            x1_raw = x1_raw[:, 0]
            x2_raw = x2_raw[:, 0]
        else:
            images = inputs["images"]
            images, image_resizer = self.preprocess_images(
                images,
                bgr_add=-0.5,
                bgr_mult=2.0,
                bgr_to_rgb=False,
                resize_mode="pad",
                pad_mode="replicate",
                pad_two_side=True,
            )
            x1_raw = images[:, 0]
            x2_raw = images[:, 1]

        b, _, height_im, width_im = x1_raw.size()

        x1_pyramid = self.fnet(x1_raw)
        x2_pyramid = self.fnet(x2_raw)

        # outputs
        flows = []

        cnet_pyramid = self.cnet(x1_raw)

        pred_stride = min(self.pyramid_ranges)

        start_level, output_level = self.pyramid_levels
        pass_pyramid1 = x1_pyramid[start_level : output_level + 1]
        pass_pyramid2 = x2_pyramid[start_level : output_level + 1]
        pass_pyramid_cnet = cnet_pyramid[start_level : output_level + 1]

        # Calculate iterations per one pyramid level.
        # It distributes iterations evenly across levels.
        iters_per_level = [
            int(math.ceil(float(self.iters) / (output_level - start_level + 1)))
        ] * (output_level - start_level + 1)

        # init
        (
            b_size,
            _,
            h_x1,
            w_x1,
        ) = pass_pyramid1[0].size()
        init_device = pass_pyramid1[0].device

        if (
            not self.simple_io
            and "prev_flows" in inputs
            and inputs["prev_flows"] is not None
        ):
            flow = F.interpolate(
                inputs["prev_flows"][:, 0],
                [pass_pyramid1[0].shape[-2], pass_pyramid1[0].shape[-1]],
                mode="bilinear",
                align_corners=True,
            )
            flow = rescale_flow(flow, width_im, height_im, to_local=True)
            flow = forward_interpolate_batch(flow)
        else:
            flow = torch.zeros(
                b_size, 2, h_x1, w_x1, dtype=x1_raw.dtype, device=init_device
            )

        net = None
        for l, (x1, x2, cnet) in enumerate(
            zip(pass_pyramid1, pass_pyramid2, pass_pyramid_cnet)
        ):
            coords0 = self.coords_grid(
                x1.shape[0], x1.shape[2], x1.shape[3], x1.dtype, x1.device
            )

            corr_fn = get_corr_block(
                x1,
                x2,
                self.corr_levels,
                self.corr_range,
                alternate_corr=self.corr_mode == "local",
            )

            net_tmp, inp = torch.split(cnet, [self.dec_net_chs, self.dec_inp_chs], dim=1)

            inp = torch.relu(inp)

            if net is None:
                net = torch.tanh(net_tmp)
            else:
                net = F.interpolate(
                    net,
                    [x1.shape[-2], x1.shape[-1]],
                    mode="bilinear",
                    align_corners=True,
                )

                net_skip = torch.tanh(net_tmp)
                gate = torch.sigmoid(
                    self.upnet_layer(torch.cat([net, net_skip], dim=1))
                )
                net = gate * net + (1.0 - gate) * net_skip

            if l > 0:
                flow = rescale_flow(flow, x1.shape[-1], x1.shape[-2], to_local=False)
                flow = F.interpolate(
                    flow,
                    [x1.shape[-2], x1.shape[-1]],
                    mode="bilinear",
                    align_corners=True,
                )

            for k in range(iters_per_level[l]):
                flow = flow.detach()

                # correlation
                # out_corr = corr_fn(coords0 + flow)
                out_corr = corr_fn(coords0, flow)

                flow_res, net, _ = self.update_block(
                    net, inp, out_corr, flow
                )
                flow = flow + flow_res

                out_flow = rescale_flow(flow, width_im, height_im, to_local=False)
                out_flow = F.interpolate(
                    out_flow,
                    [x1_raw.shape[-2], x1_raw.shape[-1]],
                    mode="bilinear",
                    align_corners=True,
                )
                out_flow = self.postprocess_predictions(
                    out_flow, image_resizer, is_flow=True
                )
                flows.append(out_flow)

        if self.simple_io:
            return flows[-1]
        else:
            outputs = {}
            outputs["flows"] = flows[-1][:, None]
            if self.training:
                outputs["flow_preds"] = flows
            return outputs


class OFNetV2T(OFNetV2):
    def __init__(
        self,
        pyramid_ranges: tuple[int, int] = (32, 8),
        iters: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            pyramid_ranges,
            iters,
            **kwargs,
        )

class OFNetV2S(OFNetV2):
    def __init__(
        self,
        pyramid_ranges: tuple[int, int] = (32, 8),
        iters: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(
            pyramid_ranges,
            iters,
            **kwargs,
        )


class OFNetV2M(OFNetV2):
    def __init__(
        self,
        pyramid_ranges: tuple[int, int] = (32, 8),
        iters: int = 6,
        **kwargs,
    ) -> None:
        super().__init__(
            pyramid_ranges,
            iters,
            **kwargs,
        )


class OFNetV2L(OFNetV2):
    def __init__(
        self,
        pyramid_ranges: tuple[int, int] = (32, 8),
        iters: int = 12,
        **kwargs,
    ) -> None:
        super().__init__(
            pyramid_ranges,
            iters,
            **kwargs,
        )

@register_model
@trainable
@ptlflow_trained
class ofnet_v2(OFNetV2):
    pass


@register_model
@trainable
@ptlflow_trained
class ofnet_v2t(OFNetV2T):
    pass


@register_model
@trainable
@ptlflow_trained
class ofnet_v2s(OFNetV2S):
    pass


@register_model
@trainable
@ptlflow_trained
class ofnet_v2m(OFNetV2M):
    pass


@register_model
@trainable
@ptlflow_trained
class ofnet_v2l(OFNetV2L):
    pass

