import json
from pathlib import Path

import numpy as np
import requests
import torch
from torchvision.io import read_image, write_png


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


def run_triton_triton_request(
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
