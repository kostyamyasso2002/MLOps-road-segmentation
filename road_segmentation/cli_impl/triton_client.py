#!/usr/bin/env python3
"""
triton_client.py

Пример Python-клиента для Triton Inference Server, который:
  • Считывает PNG-изображение (3×H×W, uint8)
  • Отправляет его на сервер в модель (ONNX или TensorRT), ожидающую тот же формат
  • Получает маску (1×H×W, uint8) в ответ
  • Сохраняет маску в PNG

Использование:
    python triton_client.py \
        --url localhost:8000 \
        --model-name my_model \
        --input-name raw_image \
        --output-name binary_mask \
        --image-path /path/to/input.png \
        --output-path /path/to/output_mask.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests
import torch
from torchvision.io import read_image, write_png


def parse_args():
    parser = argparse.ArgumentParser(
        description="Triton client: отправка изображения на инференс и получение маски"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="localhost:8000",
        help="Адрес Triton Server (host:port), по умолчанию localhost:8000",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Имя модели в Triton (имя папки в model_repository)",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="raw_image",
        help="Имя входного тензора в модели (по config.pbtxt)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="binary_mask",
        help="Имя выходного тензора в модели (по config.pbtxt)",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        required=True,
        help="Путь к входному изображению (PNG, 3×H×W, uint8)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Куда сохранить выходную маску (PNG, 1×H×W, uint8)",
    )
    return parser.parse_args()


def load_image(image_path: Path) -> np.ndarray:
    """
    Считывает PNG-изображение в uint8 numpy array shape=(1,3,H,W)
    """
    img_t = read_image(str(image_path))  # torch.uint8 tensor, shape=(3, H, W)
    img_np = img_t.numpy()  # numpy.uint8, shape=(3, H, W)
    # Добавляем batch-ось: (1, 3, H, W)
    return np.expand_dims(img_np, axis=0)


def build_infer_request(model_name: str, input_name: str, input_tensor: np.ndarray) -> dict:
    """
    Формирует JSON-пayload для HTTP/REST-инференса Triton:
      {
        "inputs": [
          {
            "name": input_name,
            "shape": [1, 3, H, W],
            "datatype": "UINT8",
            "data": [ ...flat array of ints... ]
          }
        ]
      }
    """
    # Triton REST ждёт data как обычный список Python-чисел
    flat = input_tensor.flatten().tolist()
    shape = list(input_tensor.shape)  # [1, 3, H, W]

    return {"inputs": [{"name": input_name, "shape": shape, "datatype": "UINT8", "data": flat}]}


def parse_infer_response(response: dict, output_name: str) -> np.ndarray:
    """
    Из JSON-ответа Triton извлекает выходной тензор с именем output_name
    и возвращает его как numpy.uint8 array shape=(1, 1, H, W).
    """
    for output in response.get("outputs", []):
        if output.get("name") == output_name:
            data = output.get("data")
            shape = output.get("shape")
            return np.array(data, dtype=np.uint8).reshape(shape)
    raise RuntimeError(f"Output tensor '{output_name}' not found in response")


def run_inference(
    triton_url: str,
    model_name: str,
    input_name: str,
    output_name: str,
    image_path: Path,
    output_path: Path,
):
    # 1) Загружаем изображение
    img_batch = load_image(image_path)  # numpy.uint8, shape=(1,3,H,W)

    # 2) Формируем REST-запрос
    payload = build_infer_request(model_name, input_name, img_batch)
    headers = {"Content-Type": "application/json"}
    infer_url = f"http://{triton_url}/v2/models/{model_name}/infer"

    # 3) Отправляем запрос
    resp = requests.post(infer_url, headers=headers, data=json.dumps(payload))
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} — {resp.text}")

    # 4) Разбираем ответ
    resp_json = resp.json()
    mask_batch = parse_infer_response(resp_json, output_name)
    # mask_batch: numpy.uint8, shape=(1,1,H_mask,W_mask)

    # 5) Сохраняем PNG: убираем batch=1 и канал=1 → (H, W)
    mask_2d = mask_batch.squeeze(0).squeeze(0)  # numpy.uint8, shape=(H, W)
    mask_tensor = torch.from_numpy(mask_2d).unsqueeze(0)  # torch.uint8, shape=(1, H, W)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_png(mask_tensor, str(output_path))
    print(f"Saved binary mask to: {output_path}")


def main():
    args = parse_args()

    # Проверяем, что файл изображения существует
    if not args.image_path.is_file():
        print(f"ERROR: Input image '{args.image_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # Выполняем инференс
    run_inference(
        triton_url=args.url,
        model_name=args.model_name,
        input_name=args.input_name,
        output_name=args.output_name,
        image_path=args.image_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
