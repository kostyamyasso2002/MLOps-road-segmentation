from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torchvision.io import read_image, write_png

from road_segmentation.constants import constants


def run_inference(onnx_path: Path, image_path: Path, out_path: Path) -> None:
    sess = ort.InferenceSession(str(onnx_path))

    input_name = constants.ONNX_INPUT_NAME
    output_name = constants.ONNX_OUTPUT_NAME

    img_t = read_image(str(image_path))
    img_np = img_t.numpy()

    img_batch = np.expand_dims(img_np, axis=0)

    ort_outs = sess.run([output_name], {input_name: img_batch})
    mask_np = ort_outs[0].astype(np.uint8) * constants.PIXEL_MAX

    mask_2d = mask_np.squeeze(0).squeeze(0)

    mask_tensor = torch.from_numpy(mask_2d).unsqueeze(0)

    # 8) Сохраняем PNG
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_png(mask_tensor, str(out_path))
    print(f"Saved binary mask to: {out_path}")
