#!/usr/bin/env python3
"""
run_triton.py

Скрипт для запуска Triton Inference Server через Docker с опциональным флагом GPU.
Поддерживает два режима модели: TensorRT (.plan) и ONNX (.onnx).

Извлекает из ONNX-модели имена размеры и тип входов/выходов, чтобы правильно заполнить config.pbtxt.

Пример использования:
  # TensorRT-движок (GPU):
  poetry run python -m road_segmentation.cli_impl.triton_server \
    --model-type trt \
    --model-file /path/to/model.plan \
    --model-name my_trt_model \
    --use-gpus

  # ONNX-модель (CPU или GPU, если указан --use-gpus):
  poetry run python -m road_segmentation.cli_impl.triton_server \
    --model-type onnx \
    --model-file /path/to/model.onnx \
    --model-name my_onnx_model
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import onnx


def parse_args():
    parser = argparse.ArgumentParser(
        description="Запуск Triton Inference Server (Docker) с GPU/CPU флагом"
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["trt", "onnx"],
        help="Тип модели: 'trt' (TensorRT .plan) или 'onnx' (.onnx)",
    )
    parser.add_argument("--model-file", required=True, help="Путь к файлу модели (.plan или .onnx)")
    parser.add_argument(
        "--model-name", required=True, help="Имя модели в Triton (имя папки в model_repository)"
    )
    parser.add_argument(
        "--port-http", type=int, default=8000, help="Порт для HTTP/REST (по умолчанию 8000)"
    )
    parser.add_argument(
        "--port-grpc", type=int, default=8001, help="Порт для gRPC (по умолчанию 8001)"
    )
    parser.add_argument(
        "--port-metrics", type=int, default=8002, help="Порт для метрик (по умолчанию 8002)"
    )
    parser.add_argument(
        "--use-gpus",
        action="store_true",
        help="Если указан, добавляет --gpus=all к Docker (по умолчанию False)",
    )
    return parser.parse_args()


def _onnx_dtype_to_triton(elem_type: int) -> str:
    """
    Конвертирует ONNX elem_type (TensorProto.*) в строку Triton data_type.
    Поддерживаются UINT8, FP32, FP16, INT8, INT32.
    """
    from onnx import TensorProto

    mapping = {
        TensorProto.FLOAT: "FP32",
        TensorProto.FLOAT16: "FP16",
        TensorProto.UINT8: "UINT8",
        TensorProto.INT8: "INT8",
        TensorProto.INT32: "INT32",
    }
    return mapping.get(elem_type, "TYPE_INVALID")


def extract_onnx_io(onnx_path: Path):
    """
    Загружает ONNX-модель и возвращает кортеж:
      (
        input_name: str,
        input_dims_without_batch: List[int],
        input_type: str (Triton, например "UINT8" или "FP32"),
        output_name: str,
        output_dims_without_batch: List[int],
        output_type: str
      )
    Димы без batch. Если dim не задан или 0, оставляем -1.
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph

    init_names = {init.name for init in graph.initializer}
    # первый настоящий input
    input_vi = next(x for x in graph.input if x.name not in init_names)
    input_name = input_vi.name
    input_dims = []
    for dim in input_vi.type.tensor_type.shape.dim:
        if dim.dim_value and dim.dim_value > 0:
            input_dims.append(dim.dim_value)
        else:
            input_dims.append(-1)
    # убираем batch
    input_dims = input_dims[1:]
    input_type = _onnx_dtype_to_triton(input_vi.type.tensor_type.elem_type)

    # первый output
    output_vi = graph.output[0]
    output_name = output_vi.name
    output_dims = []
    for dim in output_vi.type.tensor_type.shape.dim:
        if dim.dim_value and dim.dim_value > 0:
            output_dims.append(dim.dim_value)
        else:
            output_dims.append(-1)
    output_dims = output_dims[1:]
    output_type = _onnx_dtype_to_triton(output_vi.type.tensor_type.elem_type)

    return input_name, input_dims, input_type, output_name, output_dims, output_type


def prepare_model_repository(model_type: str, model_file: Path, model_name: str) -> Path:
    """
    Создаёт структуру model_repository/<model_name>/1/,
    копирует файл модели и генерирует config.pbtxt с учетом типов/размеров из ONNX.
    """
    work_dir = Path.cwd()
    model_repo = work_dir / "model_repository"
    model_dir = model_repo / model_name
    version_dir = model_dir / "1"

    version_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "trt":
        target_filename = "model.plan"
        platform = "tensorrt_plan"
        # Стандартные имена и типы (можно поменять вручную, если нужно)
        input_name = "input__0"
        input_dims = [3, -1, -1]
        input_type = "FP32"
        output_name = "output__0"
        output_dims = [-1]
        output_type = "FP32"
    else:  # onnx
        target_filename = "model.onnx"
        platform = "onnxruntime_onnx"
        input_name, input_dims, input_type, output_name, output_dims, output_type = extract_onnx_io(
            model_file
        )

    # Копируем файл модели
    shutil.copy(model_file, version_dir / target_filename)

    # Генерируем config.pbtxt
    config_file = model_dir / "config.pbtxt"
    with open(config_file, "w") as f:
        f.write(f'name: "{model_name}"\n')
        f.write(f'platform: "{platform}"\n')
        f.write("max_batch_size: 1\n\n")

        f.write("input [\n")
        f.write("  {\n")
        f.write(f'    name: "{input_name}"\n')
        f.write(f"    data_type: TYPE_{input_type}\n")
        f.write(f"    dims: [ {', '.join(str(d) for d in input_dims)} ]\n")
        f.write("  }\n")
        f.write("]\n\n")

        f.write("output [\n")
        f.write("  {\n")
        f.write(f'    name: "{output_name}"\n')
        f.write(f"    data_type: TYPE_{output_type}\n")
        f.write(f"    dims: [ {', '.join(str(d) for d in output_dims)} ]\n")
        f.write("  }\n")
        f.write("]\n")

    return model_dir


def run_triton(model_dir: Path, port_http: int, port_grpc: int, port_metrics: int, use_gpus: bool):
    """
    Запускает Triton Server в Docker. Если use_gpus=True, добавляет --gpus=all.
    """
    image = "nvcr.io/nvidia/tritonserver:24.04-py3"
    docker_cmd = ["docker", "run"]

    if use_gpus:
        docker_cmd += ["--gpus=all"]

    docker_cmd += [
        "--rm",
        f"-p{port_http}:8000",
        f"-p{port_grpc}:8001",
        f"-p{port_metrics}:8002",
        "-v",
        f"{model_dir.parent.absolute()}:/models",
        image,
        "tritonserver",
        "--model-repository=/models",
    ]

    print(f"Запуск Triton Inference Server с моделью: {model_dir.name}")
    print(f"HTTP порт:    {port_http}")
    print(f"gRPC порт:    {port_grpc}")
    print(f"Metrics порт: {port_metrics}")
    print(f"Использовать GPU: {'да' if use_gpus else 'нет'}")
    print("Команда Docker:")
    print(" ".join(docker_cmd))
    print()

    try:
        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при запуске Tritон: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    args = parse_args()

    model_file = Path(args.model_file)
    if not model_file.exists():
        print(f"ERROR: Файл модели '{model_file}' не найден.", file=sys.stderr)
        sys.exit(1)

    model_dir = prepare_model_repository(
        model_type=args.model_type, model_file=model_file, model_name=args.model_name
    )

    run_triton(
        model_dir=model_dir,
        port_http=args.port_http,
        port_grpc=args.port_grpc,
        port_metrics=args.port_metrics,
        use_gpus=args.use_gpus,
    )


if __name__ == "__main__":
    main()
