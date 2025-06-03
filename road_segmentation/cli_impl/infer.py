from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torchvision.io import read_image, write_png


def run_inference(onnx_path: Path, image_path: Path, out_path: Path) -> None:
    # 1) Создаём ONNXRuntime сессию
    sess = ort.InferenceSession(str(onnx_path))

    # 2) Находим имена входа и выхода (обычно "raw_image" и "binary_mask")
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # 3) Считываем исходное изображение в формате uint8, shape=(3, H, W)
    img_t = read_image(str(image_path))  # torch.uint8, (3, H, W)
    img_np = img_t.numpy()  # numpy.uint8, (3, H, W)

    # 4) Добавляем batch-ось → (1, 3, H, W)
    img_batch = np.expand_dims(img_np, axis=0)

    # 5) Передаём в ONNXRuntime → получаем numpy-маску shape=(1, 1, H_mask, W_mask)
    ort_outs = sess.run([output_name], {input_name: img_batch})
    mask_np = ort_outs[0]  # numpy.uint8, (1, 1, H_mask, W_mask)

    # 6) Убираем batch и канал → shape=(H_mask, W_mask)
    mask_2d = mask_np.squeeze(0).squeeze(0)  # numpy.uint8, (H_mask, W_mask)

    # 7) Приводим к torch.Tensor и добавляем канал: (1, H_mask, W_mask)
    mask_tensor = torch.from_numpy(mask_2d).unsqueeze(0)  # torch.uint8, (1, H_mask, W_mask)

    # 8) Сохраняем PNG
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_png(mask_tensor, str(out_path))
    print(f"Saved binary mask to: {out_path}")
