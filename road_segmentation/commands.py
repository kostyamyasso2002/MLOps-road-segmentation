import sys
from pathlib import Path

import fire

from road_segmentation.cli_impl.infer import run_inference
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
        model_path: str,
        container_name: str = "triton_server",
        http_port: int = 8000,
        grpc_port: int = 8001,
        metrics_port: int = 8002,
        use_gpus: bool = True,
    ):
        run_triton_server(
            model_path=Path(model_path),
            container_name=container_name,
            http_port=http_port,
            grpc_port=grpc_port,
            metrics_port=metrics_port,
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


if __name__ == "__main__":
    fire.Fire(Commands)
