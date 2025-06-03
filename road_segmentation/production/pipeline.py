from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from road_segmentation.models.large_unet import LargeUNet
from road_segmentation.models.mini_unet import MiniUNet


class FullPipeline(nn.Module):
    """
    Обёртка, которая объединяет:
      1) Препроцессинг: uint8 → float32 → normalize (÷255) → Resize→400×400.
      2) Вызов Lightning‐модели (MiniUNet или LargeUNet) для получения логитов.
      3) Постпроцессинг: sigmoid → порог → uint8‐маска (0 или 255).

    В конструкторе мы принимаем:
      - model_name: одна из двух строк: "mini_unet" или "large_unet".
      - ckpt_path: путь к файлу .ckpt, который был сохранён Lightning (last.ckpt).

    После создания экземпляра FullPipeline, можно напрямую делать:
       out_mask = model_pp(raw_uint8_tensor)
    где raw_uint8_tensor.shape == (B, 3, H_in, W_in), dtype=torch.uint8.
    На выходе получим маску shape==(B, 1, 400, 400), dtype=torch.uint8, {0,255}.
    """

    def __init__(self, model_name: str, ckpt_path: Path):
        super().__init__()

        # Проверяем корректность имени модели
        model_name = model_name.lower()
        if model_name not in ("mini_unet", "large_unet"):
            raise ValueError(f"model_name must be 'mini_unet' or 'large_unet', got '{model_name}'")

        # 1) Загружаем Lightning‐модель из чекпоинта (на CPU).
        #    При этом модель перейдёт в eval() и будет на CPU.
        if model_name == "mini_unet":
            lightning_model = MiniUNet.load_from_checkpoint(ckpt_path, map_location="cpu")
        else:  # model_name == "large_unet"
            lightning_model = LargeUNet.load_from_checkpoint(ckpt_path, map_location="cpu")

        lightning_model.eval()

        # 2) Сохраняем порог (threshold) из hparams
        #    Lightning автоматически прописал `self.hparams.threshold` при save_hyperparams().
        self.threshold = float(lightning_model.hparams.threshold)

        # 3) Берем саму PyTorch‐часть модели (ниже мы будем вызывать .unet(input_tensor))
        #    LightningModule наследует nn.Module, так что можно напрямую использовать
        #    весь LightningModule, но нам нужно только forward‐логику, без train/val step.
        self.unet = lightning_model

        # Отключаем градиенты, т.к. в продакшене они не нужны:
        for param in self.unet.parameters():
            param.requires_grad_(False)

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        """
        x_uint8: torch.Tensor с dtype=torch.uint8, shape=(B, 3, H_in, W_in),
                 значения ∈ {0,1,...,255}.
        Возвращает: torch.Tensor с dtype=torch.uint8, shape=(B, 1, 400, 400),
                    где каждая ячейка ∈ {0,255}.
        """

        # --- Шаг 1: конвертация в float и нормализация ---
        # (B,3,H_in,W_in), uint8 → (B,3,H_in,W_in), float32 ∈ [0..1]
        x = x_uint8.to(dtype=torch.float32) / 255.0

        # --- Шаг 2: приводим к фиксированному размеру 400×400 ---
        # Используем F.interpolate, потому что он экспортируется в ONNX как оператор Resize.
        # linear (bilinear) интерполяция для цветного изображения.
        x_resized = F.interpolate(
            x, size=(400, 400), mode="bilinear", align_corners=False
        )  # shape=(B, 3, 400, 400)

        # --- Шаг 3: прогоняем через U‐Net, получаем логиты ---
        #    lightning.unet(x_resized) вернёт (B,1,400,400) float32 (logits)
        logits = self.unet(x_resized)

        # --- Шаг 4: сигмоида + бинаризация по threshold ---
        probs = torch.sigmoid(logits)  # float32 ∈ [0..1], shape=(B,1,400,400)
        binary_mask = (probs > self.threshold).to(torch.uint8) * 255
        # теперь binary_mask.shape == (B,1,400,400), dtype=torch.uint8, values {0,255}

        return binary_mask
