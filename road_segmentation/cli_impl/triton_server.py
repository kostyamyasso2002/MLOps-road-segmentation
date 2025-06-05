import enum
import shutil
import subprocess
import sys
from pathlib import Path

from road_segmentation.constants import constants


class ModelType(enum.Enum):
    ONNX = "onnx"
    TRT = "trt"


def run_triton_server(
    model_type: ModelType,
    model_path: Path,
    container_name: str,
    http_port: int,
    use_gpus: bool,
    triton_docker_version: str = "24.04-py3",
) -> None:
    """
    Запускает Triton Inference Server с моделью из указанной директории.
    """
    if model_type != ModelType.ONNX and model_type != ModelType.TRT:
        raise ValueError("Unsupported model type. Use 'onnx' or 'trt'.")

    target_location = (
        Path(__file__).resolve().parents[2]
        / constants.TRITON_REPO_DIRNAME
        / constants.TRITON_MODEL_NAME
        / "1"
    )
    if model_type == ModelType.ONNX:
        target_location /= constants.TRITON_ONNX_MODEL_FILENAME
    elif model_type == ModelType.TRT:
        target_location /= constants.TRITON_TRT_MODEL_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    if not model_path.is_file():
        raise ValueError(f"Model path {model_path} is not a file.")
    shutil.copy(model_path, target_location)
    print(f"Copied model from {model_path} to {target_location}")

    pbtxt_target_location = target_location.resolve().parents[1] / constants.PBTXT_CONFIG_NAME
    pbtxt_src_location = (
        target_location.resolve().parents[1] / constants.ONNX_PBTXT_CONFIG_NAME
        if model_type == ModelType.ONNX
        else target_location.resolve().parents[1] / constants.TRT_PBTXT_CONFIG_NAME
    )

    if not pbtxt_src_location.exists():
        raise FileNotFoundError(f"Config file {pbtxt_src_location} does not exist.")
    if not pbtxt_src_location.is_file():
        raise ValueError(f"Config file {pbtxt_src_location} is not a file.")
    shutil.copy(pbtxt_src_location, pbtxt_target_location)
    print(f"Copied config file from {pbtxt_src_location} to {pbtxt_target_location}")

    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
    ]
    if use_gpus:
        cmd += ["--gpus", "all"]
    cmd += [
        "-p",
        f"{http_port}:8000",
        "-v",
        f"{str(Path(__file__).resolve().parents[2] / constants.TRITON_REPO_DIRNAME)}:/models",
        f"nvcr.io/nvidia/tritonserver:{triton_docker_version}",
        "tritonserver",
        "--model-repository=/models",
    ]

    print("Running Triton Inference Server with command:")
    print(" ".join(cmd))
    # Запускаем команду
    try:
        # Запускаем процесс и передаём вывод прямо в stdout/stderr текущего процесса
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении docker run: {e}", file=sys.stderr)
        sys.exit(e.returncode)
