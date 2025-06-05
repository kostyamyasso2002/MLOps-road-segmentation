import json
from pathlib import Path

import numpy as np
import requests
import torch
from torchvision.io import read_image, write_png


def load_image(image_path: Path) -> np.ndarray:
    img_t = read_image(str(image_path))
    img_np = img_t.numpy()
    return np.expand_dims(img_np, axis=0)


def build_infer_request(input_name: str, input_tensor: np.ndarray) -> dict:
    flat = input_tensor.flatten().tolist()
    shape = list(input_tensor.shape)

    return {"inputs": [{"name": input_name, "shape": shape, "datatype": "UINT8", "data": flat}]}


def parse_infer_response(response: dict, output_name: str) -> np.ndarray:
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
    img_batch = load_image(image_path)

    payload = build_infer_request(input_name, img_batch)
    headers = {"Content-Type": "application/json"}
    infer_url = f"http://{triton_url}/v2/models/{model_name}/infer"

    # 3) Отправляем запрос
    resp = requests.post(infer_url, headers=headers, data=json.dumps(payload))
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} — {resp.text}")

    resp_json = resp.json()
    mask_batch = parse_infer_response(resp_json, output_name)

    mask_2d = mask_batch.squeeze(0).squeeze(0).astype(np.uint8) * 255
    mask_tensor = torch.from_numpy(mask_2d).unsqueeze(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_png(mask_tensor, str(output_path))
    print(f"Saved binary mask to: {output_path}")
