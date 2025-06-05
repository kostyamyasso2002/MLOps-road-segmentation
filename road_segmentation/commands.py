import sys
from pathlib import Path

import fire

from road_segmentation.cli_impl.infer import run_inference
from road_segmentation.cli_impl.tensorrt_convert import tensorrt_convert
from road_segmentation.cli_impl.train import train_main
from road_segmentation.cli_impl.triton_client import run_triton_triton_request
from road_segmentation.cli_impl.triton_server import ModelType, run_triton_server
from road_segmentation.constants import constants


class Commands:
    def train(self):
        """Train the model."""
        sys.argv.remove("train")
        train_main()

    def infer(self, onnx: Path, image: Path, out: str):
        """Run inference using the ONNX model."""
        run_inference(onnx, image, Path(out))

    def triton_server(
        self,
        model_type: ModelType,
        model_path: Path,
        container_name: str = "triton_server",
        http_port: int = 8000,
        use_gpus: bool = True,
        triton_docker_version: str = "24.04-py3",
    ):
        """Run Triton Inference Server with the specified model."""
        run_triton_server(
            model_type=model_type,
            model_path=model_path,
            container_name=container_name,
            http_port=http_port,
            use_gpus=use_gpus,
            triton_docker_version=triton_docker_version,
        )

    def triton_client(
        self,
        image: Path,
        out: Path,
        triton_url: str = "localhost:8000",
        model_name: str = "roads-segmentation",
    ):
        """Run Triton client to send an image for inference and receive a mask."""
        run_triton_triton_request(
            triton_url=triton_url,
            model_name=model_name,
            image_path=image,
            output_path=out,
            input_name=constants.ONNX_INPUT_NAME,
            output_name=constants.ONNX_OUTPUT_NAME,
        )

    def tensorrt_convert(self, onnx: Path, trt: Path, trt_docker_version: str = "24.04-py3"):
        """Convert ONNX model to TensorRT format."""
        tensorrt_convert(onnx, trt, trt_docker_version=trt_docker_version)


if __name__ == "__main__":
    fire.Fire(Commands)
