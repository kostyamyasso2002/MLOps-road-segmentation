import random
from pathlib import Path
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torchvision
from dvc.repo import Repo
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision.io import read_image
from torchvision.transforms import functional as F
from torchvision.transforms import v2 as T

from road_segmentation.constants import constants

_DATA_ROOT = Path(__file__).resolve().parents[2] / constants.DATA_DIR / constants.DATASET_DIR


def _ensure_data() -> None:
    images_dir = _DATA_ROOT / constants.TRAIN_DIR / constants.IMAGES_DIR
    if not images_dir.exists():
        with Repo(str(Path(__file__).resolve().parents[2])) as repo:
            repo.pull()


class _SegDataset(Dataset):
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

        self.tf_img = T.ToDtype(torch.float32, scale=True)
        self.tf_mask = T.ToDtype(torch.float32, scale=True)

        self.hflip = T.RandomHorizontalFlip(p=1.0)
        self.vflip = T.RandomVerticalFlip(p=1.0)

    def __len__(self) -> int:
        return len(self.imgs)

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

        x = torchvision.utils.make_grid(xs, nrow=2, padding=0)
        y = torchvision.utils.make_grid(ys, nrow=2, padding=0)[:1]

        # случайный crop 400×400
        i = random.randint(0, x.shape[1] - self.size)
        j = random.randint(0, x.shape[2] - self.size)
        x = F.crop(x, i, j, self.size, self.size)
        y = F.crop(y, i, j, self.size, self.size)
        y = (y > 0.5).float()
        assert torch.all((y < 1e-6) | (y > 0.999)), "Mask contains non-binary values"
        return x, y

    # ---------- main ----------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.augment:
            return self._mosaic4()

        x = self.tf_img(read_image(str(self.imgs[idx])))
        y = self.tf_mask(read_image(str(self.masks[idx]))[:1])  # (H,W)
        y = (y > 0.5).float()
        assert torch.all((y < 1e-6) | (y > 0.999)), "Mask contains non-binary values"
        return x, y


class SegDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 4,
        num_workers: int = 2,
        val_split: float = 0.1,
        steps_per_epoch: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        _ensure_data()

    # Lightning hooks
    def setup(self, stage: Optional[str] = None):
        train_dir = _DATA_ROOT / constants.TRAIN_DIR
        imgs = sorted((train_dir / constants.IMAGES_DIR).glob("*.png"))
        masks = sorted((train_dir / constants.GROUND_TRUTH_DIR).glob("*.png"))

        split = int((1 - self.hparams.val_split) * len(imgs))

        self.train_ds = _SegDataset(imgs[:split], masks[:split], augment=True)
        self.val_ds = _SegDataset(imgs[split:], masks[split:], augment=False)

    def train_dataloader(self):
        batch_size = self.hparams.batch_size
        num_workers = self.hparams.num_workers
        steps = self.hparams.steps_per_epoch

        if steps is not None:
            sampler = RandomSampler(
                self.train_ds,
                replacement=True,
                num_samples=steps * batch_size,
            )
            return DataLoader(
                self.train_ds,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                drop_last=True,  # на всякий случай
            )
        else:
            return DataLoader(
                self.train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
