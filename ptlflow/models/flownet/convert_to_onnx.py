"""Export a FlowNetS-WB model to ONNX.

Usage example:
    python convert_to_onnx.py --ckpt_path things \
        --input_size 240 640 --opset_version 17 --output_path .
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import onnx
import torch
import torch.nn as nn

from ptlflow import restore_model
from ptlflow.models.flownet.flownets_wb import flownets_wb

# 6 stride-2 layers → inputs must be divisible by 64
_ALIGN = 64


class OnnxWrapper(nn.Module):
    """Wraps FlowNetSWB to accept two separate (B, 3, H, W) RGB float tensors.

    Inputs are expected in [0, 1] range (RGB).  Normalization that the model
    normally applies internally (bgr_add + bgr_to_rgb) is reproduced here so
    the ONNX graph is self-contained and skip_preprocess=True can be used to
    avoid tracing any Pad/Slice ops from the InputPadder.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Equivalent to: convert RGB→BGR, add (-0.406,-0.456,-0.485), convert BGR→RGB
        # = subtract (0.485, 0.456, 0.406) from RGB channels
        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        _, _, H, W = image1.shape
        assert H % _ALIGN == 0 and W % _ALIGN == 0, (
            f"Input spatial dims must be divisible by {_ALIGN}, got ({H}, {W})"
        )
        image1 = image1 - self.rgb_mean
        image2 = image2 - self.rgb_mean
        images = torch.stack([image1, image2], dim=1)  # (B, 2, 3, H, W)
        outputs = self.model({"images": images}, skip_preprocess=True)
        return outputs["flows"][:, 0]  # (B, 2, H, W)


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Export FlowNetS-WB model to ONNX.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help=(
            "Pretrained weight key (e.g. 'things') "
            "or an absolute/relative path to a local .ckpt file. "
            "If omitted, random weights are used."
        ),
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=[240, 640],
        help="Spatial size (height width) of the two input frames fed to the model.",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=17,
        help="ONNX opset version to use during export.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=".",
        help="Directory where the exported .onnx file will be saved.",
    )
    return parser


@torch.no_grad()
def export(args) -> None:
    model = flownets_wb()
    model = restore_model(model, args.ckpt_path)
    model.eval()

    wrapper = OnnxWrapper(model)
    wrapper.eval()

    H, W = args.input_size
    if H % _ALIGN != 0 or W % _ALIGN != 0:
        raise ValueError(
            f"--input_size must be divisible by {_ALIGN}, got ({H}, {W})"
        )

    img1 = torch.rand(1, 3, H, W)
    img2 = torch.rand(1, 3, H, W)

    sample_output = wrapper(img1, img2)
    print(f"image1 shape : {list(img1.shape)}")
    print(f"image2 shape : {list(img2.shape)}")
    print(f"Output shape : {list(sample_output.shape)}")

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = "flownets_wb"
    if args.ckpt_path is not None:
        stem += f"_{Path(args.ckpt_path).stem}"
    output_file = str(output_dir / f"{stem}.onnx")

    torch.onnx.export(
        wrapper,
        (img1, img2),
        output_file,
        opset_version=args.opset_version,
        input_names=["image1", "image2"],
        output_names=["flows"],
        # dynamic_axes={
        #     "image1": {0: "batch", 2: "height", 3: "width"},
        #     "image2": {0: "batch", 2: "height", 3: "width"},
        #     "flows":  {0: "batch", 2: "height_out", 3: "width_out"},
        # },
        do_constant_folding=True,
    )
    print(f"Saved : {output_file}")

    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)

    print("\n--- ONNX model inputs ---")
    for inp in onnx_model.graph.input:
        dims = [
            d.dim_param if d.dim_param else d.dim_value
            for d in inp.type.tensor_type.shape.dim
        ]
        print(f"  {inp.name}: {dims}")

    print("--- ONNX model outputs ---")
    for out in onnx_model.graph.output:
        dims = [
            d.dim_param if d.dim_param else d.dim_value
            for d in out.type.tensor_type.shape.dim
        ]
        print(f"  {out.name}: {dims}")


if __name__ == "__main__":
    parser = _init_parser()
    args = parser.parse_args()
    export(args)
