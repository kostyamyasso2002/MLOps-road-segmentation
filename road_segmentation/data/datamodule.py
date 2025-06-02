import random
from pathlib import Path
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torchvision
from dvc.repo import Repo
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import functional as F
from torchvision.transforms import v2 as T

_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "dataset"


def _ensure_data() -> None:
    """
    Make sure the training images are present locally.
    If they’re missing, fetch them from the DVC remote.
    """
    images_dir = _DATA_ROOT / "training" / "images"

    if not _DATA_ROOT.exists():
        with Repo(str(_DATA_ROOT)) as repo:
            # Equivalent to: `dvc pull -q training/images`
            repo.pull(
                targets=[str(images_dir.relative_to(_DATA_ROOT))],
                quiet=True,  # mirrors the CLI’s `-q`
            )


class _SegDataset(Dataset):
    """
    Повторяет rand_data() из run.py:
    • берёт 4 случайных кадра и маски
    • случайные флипы (p=0.5)
    • собирает мозаику 2×2 (800×800)
    • случайный crop 400×400
    • y → (H,W)   без канала, как в run.py
    """

    def __init__(
        self,
        img_paths: List[Path],
        mask_paths: List[Path],
        size: int = 400,
        augment: bool = False,
    ):
        self.imgs, self.masks = img_paths, mask_paths
        self.size = size
        self.augment = augment

        # базовые преобразования: только в float32 [0,1]
        self.tf_img = T.ToDtype(torch.float32, scale=True)
        self.tf_mask = T.ToDtype(torch.float32, scale=True)

        # флипы «как у автора»
        self.hflip = T.RandomHorizontalFlip(p=1.0)
        self.vflip = T.RandomVerticalFlip(p=1.0)

    def __len__(self) -> int:
        return len(self.imgs)

    # ---------- helpers ----------
    def _maybe_flip(
        self, img: torch.Tensor, msk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.randint(0, 1):
            img, msk = self.hflip(img), self.hflip(msk)
        if random.randint(0, 1):
            img, msk = self.vflip(img), self.vflip(msk)
        return img, msk

    def _mosaic4(self) -> Tuple[torch.Tensor, torch.Tensor]:
        idxs = random.sample(range(len(self.imgs)), 4)
        xs, ys = [], []
        for i in idxs:
            x = self.tf_img(read_image(str(self.imgs[i])))
            y = self.tf_mask(read_image(str(self.masks[i]))[:1])  # 1-канал
            x, y = self._maybe_flip(x, y)
            xs.append(x)
            ys.append(y)

        # 2×2 мозаика, padding=0
        x = torchvision.utils.make_grid(xs, nrow=2, padding=0)  # (3, 800, 800)
        y = torchvision.utils.make_grid(ys, nrow=2, padding=0)[:1]  # (1, 800, 800)

        # случайный crop 400×400
        i = random.randint(0, x.shape[1] - self.size)
        j = random.randint(0, x.shape[2] - self.size)
        x = F.crop(x, i, j, self.size, self.size)  # (3, 400, 400)
        y = F.crop(y, i, j, self.size, self.size)
        return x, y

    # ---------- main ----------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.augment:
            return self._mosaic4()

        # *** валидация без аугментаций ***
        x = self.tf_img(read_image(str(self.imgs[idx])))
        y = self.tf_mask(read_image(str(self.masks[idx]))[:1])  # (H,W)
        return x, y


class SegDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4, num_workers: int = 2, val_split: float = 0.1):
        super().__init__()
        self.save_hyperparameters()
        _ensure_data()

    # Lightning hooks
    def setup(self, stage: Optional[str] = None):
        train_dir = _DATA_ROOT / "training"
        imgs = sorted((train_dir / "images").glob("*.png"))
        masks = sorted((train_dir / "groundtruth").glob("*.png"))

        split = int((1 - self.hparams.val_split) * len(imgs))

        # train — с мозаикой; val — без
        self.train_ds = _SegDataset(imgs[:split], masks[:split], augment=True)
        self.val_ds = _SegDataset(imgs[split:], masks[split:], augment=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
