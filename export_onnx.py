"""Export an optical flow model to ONNX format.

This script exports any registered PTLFlow model to ONNX, optionally loading
pretrained weights. The exported model takes a single tensor of shape
(B, 2, 3, H, W) and returns the predicted optical flow of shape (B, 2, H, W).

"""

import sys
from pathlib import Path

from jsonargparse import ArgumentParser
from loguru import logger
import torch
import torch.nn as nn

import ptlflow
from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils.lightning.ptlflow_cli import PTLFlowCLI
from ptlflow.utils.registry import RegisteredModel


class OnnxWrapper(nn.Module):
    """Wrap a PTLFlow model for ONNX export.

    PTLFlow models expect a dict input (``{"images": tensor}``) and return a
    dict output (``{"flows": tensor, ...}``).  ONNX export requires plain
    tensor inputs/outputs, so this wrapper handles the conversion.
    """

    def __init__(self, model: BaseModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run the model and return only the flow prediction.

        Parameters
        ----------
        images : torch.Tensor
            A tensor of shape (B, 2, 3, H, W) with the two input frames.

        Returns
        -------
        torch.Tensor
            The predicted optical flow of shape (B, 2, H, W).
        """
        preds = self.model({"images": images})
        # flows shape: (B, N, 2, H, W) – take the first (and usually only) prediction
        return preds["flows"][:, 0]


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to a checkpoint file for the chosen model.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path("outputs/onnx")),
        help="Path to the directory where the ONNX model will be saved.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Height and width of the input images used for tracing.",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=16,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="If set, export the model in half precision.",
    )
    return parser


@torch.no_grad()
def export(args, model: BaseModel) -> str:
    """Export a PTLFlow model to ONNX.

    Parameters
    ----------
    args : Namespace
        Export configuration.
    model : BaseModel
        The model to export.

    Returns
    -------
    str
        Path to the exported ONNX file.
    """
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        if args.fp16:
            model = model.half()

    wrapper = OnnxWrapper(model)
    wrapper.eval()

    sample_input = torch.rand(1, 2, 3, args.input_size[0], args.input_size[1])
    if torch.cuda.is_available():
        sample_input = sample_input.cuda()
        if args.fp16:
            sample_input = sample_input.half()

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name
    if args.ckpt_path is not None:
        model_name += f"_{Path(args.ckpt_path).stem}"
    output_file = str(output_dir / f"{model_name}.onnx")

    # Print input/output sizes and FLOPs/MACs
    with torch.no_grad():
        sample_output = wrapper(sample_input)
    logger.info("Input size: {}", list(sample_input.shape))
    logger.info("Output size: {}", list(sample_output.shape))

    from torch.profiler import profile, ProfilerActivity

    with profile(
        activities=[ProfilerActivity.CPU]
        + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
        record_shapes=True,
        with_flops=True,
    ) as prof:
        wrapper(sample_input)
    flops = sum(k.flops for k in prof.key_averages())
    logger.info("GFLOPs: {:.4f}", flops / 1e9)
    logger.info("GMACs: {:.4f}", flops / 2 / 1e9)

    torch.onnx.export(
        wrapper,
        sample_input,
        output_file,
        opset_version=args.opset_version,
        input_names=["images"],
        output_names=["flows"],
        dynamic_axes={
            "images": {0: "batch", 3: "height", 4: "width"},
            "flows": {0: "batch", 2: "height", 3: "width"},
        },
    )

    logger.info("ONNX model saved to: {}", output_file)
    return output_file


def _show_v04_warning():
    ignore_args = ["-h", "--help", "--model", "--config"]
    for arg in ignore_args:
        if arg in sys.argv:
            return

    logger.warning(
        "Since v0.4, it is now necessary to inform the model using the --model argument. "
        "For example, use: python export_onnx.py --model raft --ckpt_path things"
    )


if __name__ == "__main__":
    _show_v04_warning()

    parser = _init_parser()

    cli = PTLFlowCLI(
        model_class=RegisteredModel,
        subclass_mode_model=True,
        parser_kwargs={"parents": [parser]},
        run=False,
        parse_only=False,
        auto_configure_optimizers=False,
    )

    cfg = cli.config
    cfg.model_name = cfg.model.class_path.split(".")[-1]

    model = cli.model
    model = ptlflow.restore_model(model, cfg.ckpt_path)

    export(cfg, model)
