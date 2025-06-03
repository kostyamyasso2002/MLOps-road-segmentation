import sys
from pathlib import Path

import fire

from road_segmentation.cli_impl.infer import run_inference
from road_segmentation.cli_impl.train import train_main


class Commands:
    def train(self):
        """Train the model."""
        # remove train from sys.argv to avoid confusion with hydra
        sys.argv.remove("train")
        train_main()

    def infer(self, onnx: str, image: str, out: str):
        """Run inference using the ONNX model."""
        run_inference(Path(onnx), Path(image), Path(out))


if __name__ == "__main__":
    fire.Fire(Commands)
