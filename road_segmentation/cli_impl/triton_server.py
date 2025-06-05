import shutil
import subprocess
import sys
from pathlib import Path


def run_triton_server(
    model_type: str,
    model_path: Path,
    container_name: str,
    http_port: int,
    use_gpus: bool,
) -> None:
    """
    Запускает Triton Inference Server с моделью из указанной директории.
    """
    if model_type != "onnx" and model_type != "trt":
        raise ValueError("Unsupported model type. Use 'onnx' or 'trt'.")

    target_location = (
        Path(__file__).resolve().parents[2] / "triton_repo" / "roads-segmentation" / "1"
    )
    if model_type == "onnx":
        target_location /= "model.onnx"
    elif model_type == "trt":
        target_location /= "model.plan"

    # copy the model from model_path to target_location
    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    if not model_path.is_file():
        raise ValueError(f"Model path {model_path} is not a file.")
    shutil.copy(model_path, target_location)
    print(f"Copied model from {model_path} to {target_location}")

    pbtxt_target_location = target_location.resolve().parents[1] / "config.pbtxt"
    pbtxt_src_location = (
        target_location.resolve().parents[1] / "config_onnx.pbtxt"
        if model_type == "onnx"
        else target_location.resolve().parents[1] / "config_trt.pbtxt"
    )

    # copy the config.pbtxt file
    if not pbtxt_src_location.exists():
        raise FileNotFoundError(f"Config file {pbtxt_src_location} does not exist.")
    if not pbtxt_src_location.is_file():
        raise ValueError(f"Config file {pbtxt_src_location} is not a file.")
    shutil.copy(pbtxt_src_location, pbtxt_target_location)
    print(f"Copied config file from {pbtxt_src_location} to {pbtxt_target_location}")

    # Запускаем Triton Inference Server
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
        f"{str(Path(__file__).resolve().parents[2] / 'triton_repo')}:/models",
        # Образ и команда внутри контейнера
        "nvcr.io/nvidia/tritonserver:24.04-py3",
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
