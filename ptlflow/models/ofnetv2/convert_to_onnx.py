"""Export an OFNet v1 model to ONNX.

Usage example:
    python convert_to_onnx.py --model ofnet_v2m --ckpt_path kitti \
        --input_size 256 640 --opset_version 17 --output_path .
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

# Ensure the workspace root is on sys.path so that 'ptlflow' always resolves
# to this editable install, not to any older system-wide install.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import onnx
import torch
import torch.nn as nn

from ptlflow import restore_model
from ptlflow.models.ofnetv2.ofnet import (
    ofnet_v2,
    ofnet_v2t,
    ofnet_v2s,
    ofnet_v2m,
    ofnet_v2l,
)

_MODELS = {
    "ofnet_v2": ofnet_v2,
    "ofnet_v2t": ofnet_v2t,
    "ofnet_v2s": ofnet_v2s,
    "ofnet_v2m": ofnet_v2m,
    "ofnet_v2l": ofnet_v2l,
}


_ALIGN = 32


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


class _IdentityResizer:
    """A no-op resizer that avoids tracing any Slice/Pad ops."""

    def fill(self, x):
        return x

    def unfill(self, x):
        return x


class OnnxWrapper(nn.Module):
    """Wraps the model to accept two separate (B, 3, H, W) image tensors.

    The model is constructed with ``simple_io=True`` so it accepts a raw
    ``(B, 2, 3, H, W)`` tensor and returns ``(B, 2, H, W)`` flow directly.

    The model's internal InputPadder is replaced with a no-op so that
    onnxsim cannot introduce multi-axis Slice nodes.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Monkey-patch preprocess_images so the internal InputPadder is never
        # created — we handle padding externally in this wrapper.
        _orig_preprocess = model.preprocess_images

        def _preprocess_no_pad(images, **kwargs):
            # Apply only the color normalization (bgr_add / bgr_mult), skip
            # InputPadder entirely by forcing image_resizer=_IdentityResizer().
            return _orig_preprocess(images, image_resizer=_IdentityResizer(), **kwargs)

        model.preprocess_images = _preprocess_no_pad

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        _, _, H, W = image1.shape
        assert H % _ALIGN == 0 and W % _ALIGN == 0, (
            f"Input spatial dims must be divisible by {_ALIGN}, got ({H}, {W})"
        )
        flow = self.model(image1, image2)  # (B, 2, H, W)
        return flow


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Export an OFNet v2 model to ONNX.")
    parser.add_argument(
        "--model",
        type=str,
        default="ofnet_v2m",
        choices=list(_MODELS.keys()),
        help="OFNet v2 model variant.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help=(
            "Pretrained weight key (e.g. 'chairs', 'things', 'sintel', 'kitti') "
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
    model_cls = _MODELS[args.model]
    model = model_cls(simple_io=True)
    model = restore_model(model, args.ckpt_path)
    model.eval()

    wrapper = OnnxWrapper(model)
    wrapper.eval()

    H, W = args.input_size
    img1 = torch.rand(1, 3, H, W)
    img2 = torch.rand(1, 3, H, W)

    sample_output = wrapper(img1, img2)
    print(f"image1 shape : {list(img1.shape)}")
    print(f"image2 shape : {list(img2.shape)}")
    print(f"Output shape : {list(sample_output.shape)}")

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.model
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

