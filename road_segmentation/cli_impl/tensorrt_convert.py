import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Конвертация ONNX-модели в TensorRT-движок через Docker"
    )
    parser.add_argument(
        "--onnx-path",
        required=True,
        type=Path,
        help="Путь к исходному ONNX-файлу (например: /home/user/model.onnx)",
    )
    parser.add_argument(
        "--trt-path",
        required=True,
        type=Path,
        help="Путь, куда сохранить сгенерированный .plan (например: /home/user/model.plan)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    onnx_path: Path = args.onnx_path.resolve()
    trt_path: Path = args.trt_path.resolve()

    # Проверяем, что исходный ONNX-файл существует
    if not onnx_path.is_file():
        print(f"Ошибка: ONNX-файл не найден по пути: {onnx_path}", file=sys.stderr)
        sys.exit(1)

    # Убедимся, что директория для вывода .plan существует (если нет — создадим)
    trt_dir: Path = trt_path.parent
    if trt_dir and not trt_dir.exists():
        try:
            trt_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Не удалось создать директорию для вывода: {trt_dir}\n{e}", file=sys.stderr)
            sys.exit(1)

    # Извлекаем имя файла и родительскую директорию для монтирования в Docker
    onnx_dir: Path = onnx_path.parent
    onnx_fname: str = onnx_path.name
    trt_fname: str = trt_path.name

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
    ]

    print("Запускаем Docker-контейнер для конвертации ONNX → TRT:")
    print(" ".join(docker_cmd))

    try:
        subprocess.run(docker_cmd, check=True)
        print(f"\nУспешно создан TensorRT-движок: {trt_path}")
    except subprocess.CalledProcessError as e:
        print("\nОшибка при выполнении trtexec внутри Docker:", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
