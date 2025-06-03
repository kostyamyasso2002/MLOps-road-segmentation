import shutil
import subprocess
import sys
from pathlib import Path


def run_triton_server(
    model_path: Path,
    container_name: str,
    http_port: int,
    grpc_port: int,
    metrics_port: int,
    use_gpus: bool,
) -> None:
    """
    Запускает Triton Inference Server с моделью из указанной директории.
    """
    target_location = (
        Path(__file__).resolve().parents[2]
        / "triton_repo"
        / "roads-segmentation"
        / "1"
        / "model.onnx"
    )
    # copy the model from model_path to target_location
    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    if not model_path.is_file():
        raise ValueError(f"Model path {model_path} is not a file.")
    shutil.copy(model_path, target_location)
    print(f"Copied model from {model_path} to {target_location}")

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
        "-p",
        f"{grpc_port}:8001",
        "-p",
        f"{metrics_port}:8002",
        # Монтируем локальную папку с моделями внутрь контейнера
        "-v",
        f"{str(target_location.resolve().parents[3])}:/models",
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
