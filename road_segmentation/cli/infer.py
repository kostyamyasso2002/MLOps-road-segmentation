import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from torchvision.io import read_image, write_png
from torchvision.transforms import functional as TF

from road_segmentation.models.large_unet import LargeUNet

# Импортируем оба класса заранее
from road_segmentation.models.mini_unet import MiniUNet

MODEL_REGISTRY = {
    "mini_unet": MiniUNet,
    "large_unet": LargeUNet,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for a trained Lightning segmentation model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Выбор модели: 'mini_unet' или 'large_unet'",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Путь до чекпойнта Lightning (например, final.ckpt или last.ckpt)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Путь до входного изображения (PNG, JPG и т.д.)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Куда сохранить выходную маску (PNG).",
    )
    return parser.parse_args()


def load_model(model_name: str, ckpt_path: Path, device: torch.device) -> pl.LightningModule:
    """
    Загружает LightningModule (MiniUNet или LargeUNet) из чекпойнта и переводит его на device.
    Ожидается, что threshold сохранён в hparams.
    """
    cls = MODEL_REGISTRY[model_name]
    model = cls.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.to(device)
    model.eval()
    return model


def load_and_preprocess_image(image_path: Path, target_size=(400, 400)) -> torch.Tensor:
    """
    Считывает изображение, переводит в float-тензор [0,1], ресайзит до target_size.
    Возвращает тензор формы (1, C, H, W).
    """
    img = read_image(str(image_path))  # uint8, (C, H, W), знач. 0–255
    img = img.to(torch.float32) / 255.0  # float32, 0–1
    img = TF.resize(img, target_size)  # (C, H, W)
    img = img.unsqueeze(0)  # (1, C, H, W)
    return img


def postprocess_and_save_mask(probs: torch.Tensor, threshold: float, out_path: Path) -> None:
    """
    Бинаризует тензор вероятностей по threshold, умножает на 255 → uint8,
    сохраняет PNG в out_path. Ожидается probs.shape == (1, 1, H, W).
    """
    mask_bin = (probs > threshold).to(torch.uint8) * 255  # (1,1,H,W)
    mask = mask_bin.squeeze(0)  # (H, W)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_png(mask, str(out_path))


def run_inference(model_name: str, ckpt: Path, image: Path, out: Path) -> None:
    # 1) выбираем устройство
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 2) загружаем модель (MiniUNet или LargeUNet) из чекпойнта
    model = load_model(model_name, ckpt, device)

    # 3) доставем threshold из hparams
    threshold = float(model.hparams.threshold)

    # 4) загружаем и предобрабатываем изображение
    img_tensor = load_and_preprocess_image(image).to(device)

    # 5) прямой проход → логиты → вероятности
    with torch.no_grad():
        logits = model(img_tensor)  # (1,1,H,W)
        probs = torch.sigmoid(logits)  # (1,1,H,W)

    # 6) бинаризация + сохранение маски
    postprocess_and_save_mask(probs, threshold, out)
    print(f"Saved binary mask to: {out}  (threshold = {threshold})")


def main():
    args = parse_args()
    run_inference(args.model, args.ckpt, args.image, args.out)


if __name__ == "__main__":
    main()
