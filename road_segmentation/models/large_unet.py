import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchvision.transforms import functional as TF

# --------------------------- building blocks --------------------------- #


class _ConvBlock(nn.Module):
    """(Conv ‑ BN ‑ ReLU) ×2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _Down(nn.Module):
    """Downscale with MaxPool2d then a ConvBlock"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class _Up(nn.Module):
    """Upscale, concat skip, then ConvBlock. Handles crop mis‑match."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # concat -> out_ch*2 channels
        self.conv = _ConvBlock(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = self.up(x)
        # --- spatial alignment (center crop skip) ---
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            skip = TF.center_crop(skip, [x.shape[2], x.shape[3]])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# --------------------------- main network --------------------------- #


class LargeUNet(pl.LightningModule):
    """Глубокий U‑Net (~16М параметров) в стиле MiniUNet."""

    def __init__(self, lr: float, threshold: float, pos_weight: float):
        super().__init__()
        self.save_hyperparameters()

        # encoder
        self.stem = _ConvBlock(3, 32)
        self.down1 = _Down(32, 64)  # 200×200
        self.down2 = _Down(64, 128)  # 100×100
        self.down3 = _Down(128, 256)  # 50×50

        # bottleneck
        self.bottleneck = nn.Sequential(
            _ConvBlock(256, 512),
            _ConvBlock(512, 512),
        )  # 25×25

        # decoder
        self.up3 = _Up(512, 256)  # 50×50, skip from down3
        self.up2 = _Up(256, 128)  # 100×100, skip from down2
        self.up1 = _Up(128, 64)  # 200×200, skip from down1

        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

        pos_w = torch.tensor([pos_weight], dtype=torch.float32)
        self.register_buffer("pos_weight", pos_w)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        self.f1 = torchmetrics.F1Score(task="binary")

        # ----------------------- forward ----------------------- #

    def forward(self, x):
        s0 = self.stem(x)  # 32, 400×400
        s1 = self.down1(s0)  # 64, 200×200
        s2 = self.down2(s1)  # 128,100×100
        s3 = self.down3(s2)  # 256,50×50
        b = self.bottleneck(s3)  # 512,25×25

        d2 = self.up3(b, s3)  # 256,50×50
        d1 = self.up2(d2, s2)  # 128,100×100
        d0 = self.up1(d1, s1)  # 64, 200×200
        out = self.head(d0)  # 1, 200×200
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out

    # ----------------------- steps ----------------------- #
    def _shared_step(self, batch, stage: str):
        imgs, masks = batch  # masks: (B,1,H,W)
        logits = self(imgs)
        loss = self.loss_fn(logits, masks)
        preds = (torch.sigmoid(logits) > self.hparams.threshold).int()

        f1 = self.f1(preds, masks.int())
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_f1", f1, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    # ----------------------- optim ----------------------- #
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
