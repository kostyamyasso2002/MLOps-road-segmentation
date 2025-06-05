import sys
from pathlib import Path

import fire

from road_segmentation.cli_impl.infer import run_inference
from road_segmentation.cli_impl.tensorrt_convert import tensorrt_convert
from road_segmentation.cli_impl.train import train_main
from road_segmentation.cli_impl.triton_client import run_triton_triton_request
from road_segmentation.cli_impl.triton_server import run_triton_server


class Commands:
    def train(self):
        """Train the model."""
        # remove train from sys.argv to avoid confusion with hydra
        sys.argv.remove("train")
        train_main()

    def infer(self, onnx: str, image: str, out: str):
        """Run inference using the ONNX model."""
        run_inference(Path(onnx), Path(image), Path(out))

    def triton_server(
        self,
        model_type: str,
        model_path: str,
        container_name: str = "triton_server",
        http_port: int = 8000,
        use_gpus: bool = True,
    ):
        run_triton_server(
            model_type=model_type,
            model_path=Path(model_path),
            container_name=container_name,
            http_port=http_port,
            use_gpus=use_gpus,
        )

    def triton_client(
        self,
        image: str,
        out: str,
        triton_url: str = "localhost:8000",
        model_name: str = "roads-segmentation",
    ):
        """Run Triton client to send an image for inference and receive a mask."""
        run_triton_triton_request(
            triton_url=triton_url,
            model_name=model_name,
            image_path=Path(image),
            output_path=Path(out),
            input_name="raw_image",
            output_name="binary_mask",
        )

    def tensorrt_convert(self, onnx: str, trt: str):
        """Convert ONNX model to TensorRT format."""
        tensorrt_convert(Path(onnx), Path(trt))


if __name__ == "__main__":
    fire.Fire(Commands)
