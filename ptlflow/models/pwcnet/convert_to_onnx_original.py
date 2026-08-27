"""Export a PWC-Net model to ONNX.

Usage example:
    python convert_to_onnx_original.py --model pwcnet \
        --input_size 384 512 --opset_version 16 --output_path .
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import onnx
import torch
import torch.nn as nn

from ptlflow import restore_model
from ptlflow.models.pwcnet.pwcnet import pwcnet, pwcnet_nodc

# 6 stride-2 feature levels (7-level pyramid) → inputs must be divisible by 64.
_ALIGN = 64

_MODELS = {
    "pwcnet": pwcnet,
    "pwcnet_nodc": pwcnet_nodc,
}


class OnnxWrapper(nn.Module):
    """Wraps PWC-Net to accept two separate (B, 3, H, W) RGB float tensors.

    The two frames are passed straight into the network core
    (``model._forward_pair``), so the exported ONNX graph takes two
    (B, 3, H, W) inputs and never stacks them into a 5D (B, 2, 3, H, W) tensor.

    Calling the core directly also bypasses preprocessing (the BGR->RGB flip and
    the interpolation resize to a multiple of the output stride), just like
    skip_preprocess=True. The resize is a no-op when H and W are already
    multiples of the stride, so this matches the model's normal output for
    aligned inputs. Intensity is used as-is (bgr_add=0, bgr_mult=1).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        _, _, H, W = image1.shape
        assert H % _ALIGN == 0 and W % _ALIGN == 0, (
            f"Input spatial dims must be divisible by {_ALIGN}, got ({H}, {W})"
        )
        # Feed the two frames straight into the network core as separate
        # (B, 3, H, W) tensors. No torch.stack, so the ONNX graph never builds a
        # 5D (B, 2, 3, H, W) tensor (it would only be sliced back apart anyway).
        outputs = self.model._forward_pair(image1, image2)
        return outputs["flows"][:, 0]  # (B, 2, H, W)


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Export PWC-Net model to ONNX.")
    parser.add_argument(
        "--model",
        type=str,
        default="pwcnet",
        choices=list(_MODELS.keys()),
        help="Which PWC-Net variant to export.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help=(
            "Pretrained weight key or an absolute/relative path to a local .ckpt "
            "file. If omitted, random weights are used."
        ),
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=[256, 640],
        help="Spatial size (height width) of the two input frames fed to the model.",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=17,
        help="ONNX opset version to use during export (>=16 is required for grid_sample).",
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
    model = _MODELS[args.model]()
    model = restore_model(model, args.ckpt_path)
    model.eval()

    wrapper = OnnxWrapper(model)
    wrapper.eval()

    H, W = args.input_size
    if H % _ALIGN != 0 or W % _ALIGN != 0:
        raise ValueError(f"--input_size must be divisible by {_ALIGN}, got ({H}, {W})")

    if args.opset_version < 16:
        raise ValueError(
            f"opset_version must be >= 16 (grid_sample support), got {args.opset_version}"
        )

    img1 = torch.rand(1, 3, H, W)
    img2 = torch.rand(1, 3, H, W)
    if torch.cuda.is_available():
        wrapper = wrapper.cuda()
        img1 = img1.cuda()
        img2 = img2.cuda()

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
