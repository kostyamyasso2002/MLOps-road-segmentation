import subprocess
import sys
from pathlib import Path

from road_segmentation.constants import constants


def tensorrt_convert(onnx_path: Path, trt_path: Path, trt_docker_version) -> None:
    real_onnx_path = onnx_path.resolve()
    real_trt_path = trt_path.resolve()

    if not real_onnx_path.is_file():
        print(f"Ошибка: ONNX-файл не найден по пути: {real_onnx_path}", file=sys.stderr)
        sys.exit(1)

    trt_dir: Path = real_trt_path.parent
    if trt_dir and not trt_dir.exists():
        try:
            trt_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Не удалось создать директорию для вывода: {trt_dir}\n{e}", file=sys.stderr)
            sys.exit(1)

    onnx_dir: Path = real_onnx_path.parent
    onnx_fname: str = real_onnx_path.name
    trt_fname: str = real_trt_path.name

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{onnx_dir}:/workspace/input:ro",
        "-v",
        f"{trt_dir}:/workspace/output",
        f"nvcr.io/nvidia/tensorrt:{trt_docker_version}",
        "trtexec",
        f"--onnx=/workspace/input/{onnx_fname}",
        f"--saveEngine=/workspace/output/{trt_fname}",
        f"--minShapes=raw_image:{constants.TRT_MIN_SHAPE}",
        f"--optShapes=raw_image:{constants.TRT_OPT_SHAPE}",
        f"--maxShapes=raw_image:{constants.TRT_MAX_SHAPE}",
    ]

    print("Запускаем Docker-контейнер для конвертации ONNX -> TRT:")
    print(" ".join(docker_cmd))

    try:
        subprocess.run(docker_cmd, check=True)
        print(f"\nУспешно создан TensorRT-движок: {real_trt_path}")
    except subprocess.CalledProcessError as e:
        print("\nОшибка при выполнении trtexec внутри Docker:", file=sys.stderr)
        sys.exit(e.returncode)
