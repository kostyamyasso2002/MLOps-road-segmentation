from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from road_segmentation.constants import constants
from road_segmentation.models.large_unet import LargeUNet
from road_segmentation.models.mini_unet import MiniUNet


class FullPipeline(nn.Module):

    def __init__(self, model_name: str, ckpt_path: Path):
        super().__init__()

        model_name = model_name.lower()
        if model_name not in constants.MODEL_NAMES:
            raise ValueError(
                f"model_name must be one of {constants.MODEL_NAMES}, got '{model_name}'"
            )

        if model_name == "mini_unet":
            lightning_model = MiniUNet.load_from_checkpoint(ckpt_path, map_location="cpu")
        elif model_name == "large_unet":
            lightning_model = LargeUNet.load_from_checkpoint(ckpt_path, map_location="cpu")
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        lightning_model.eval()

        self.threshold = float(lightning_model.hparams.threshold)

        self.unet = lightning_model

        for param in self.unet.parameters():
            param.requires_grad_(False)

    def forward(self, input_uint8: torch.Tensor) -> torch.Tensor:
        input = input_uint8.to(dtype=torch.float32) / constants.PIXEL_MAX

        x_resized = F.interpolate(
            input, size=constants.PIPELINE_PICTURE_SIZE, mode="bilinear", align_corners=False
        )

        logits = self.unet(x_resized)

        probs = torch.sigmoid(logits)
        binary_mask = (probs > self.threshold).to(torch.float32)
        return binary_mask
