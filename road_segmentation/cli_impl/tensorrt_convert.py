import subprocess
import sys
from pathlib import Path


def tensorrt_convert(onnx_path: Path, trt_path: Path) -> None:
    # Получаем «истинный» путь (без символических ссылок)
    real_onnx_path = onnx_path.resolve()
    real_trt_path = trt_path.resolve()

    # Проверяем, что исходный ONNX-файл существует
    if not real_onnx_path.is_file():
        print(f"Ошибка: ONNX-файл не найден по пути: {real_onnx_path}", file=sys.stderr)
        sys.exit(1)

    # Убедимся, что директория для вывода .plan существует (если нет — создадим)
    trt_dir: Path = real_trt_path.parent
    if trt_dir and not trt_dir.exists():
        try:
            trt_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Не удалось создать директорию для вывода: {trt_dir}\n{e}", file=sys.stderr)
            sys.exit(1)

    # Извлекаем имя файла и родительскую директорию (после resolve())
    onnx_dir: Path = real_onnx_path.parent
    onnx_fname: str = real_onnx_path.name
    trt_fname: str = real_trt_path.name

    # Монтируем onnx_dir как /workspace/input, а trt_dir как /workspace/output
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{onnx_dir}:/workspace/input:ro",  # монтируем ONNX-папку (только для чтения)
        "-v",
        f"{trt_dir}:/workspace/output",  # монтируем папку для вывода TRI
        "nvcr.io/nvidia/tensorrt:24.04-py3",
        "trtexec",
        f"--onnx=/workspace/input/{onnx_fname}",
        f"--saveEngine=/workspace/output/{trt_fname}",
        "--minShapes=raw_image:1x3x1x1",
        "--optShapes=raw_image:1x3x400x400",
        "--maxShapes=raw_image:1x3x1024x1024",
    ]

    print("Запускаем Docker-контейнер для конвертации ONNX → TRT:")
    print(" ".join(docker_cmd))

    try:
        subprocess.run(docker_cmd, check=True)
        print(f"\nУспешно создан TensorRT-движок: {real_trt_path}")
    except subprocess.CalledProcessError as e:
        print("\nОшибка при выполнении trtexec внутри Docker:", file=sys.stderr)
        sys.exit(e.returncode)
