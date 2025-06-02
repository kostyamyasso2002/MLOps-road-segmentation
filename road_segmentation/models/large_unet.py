import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class _ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            # Align spatial and channel dims
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class _Down(nn.Module):

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            _ResidualBlock(in_c, out_c, stride=2),
            _ResidualBlock(out_c, out_c),
        )

    def forward(self, x):
        return self.block(x)


class _Up(nn.Module):

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        # halve channels during upsample
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            _ResidualBlock(in_c, out_c),
            _ResidualBlock(out_c, out_c),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = self.up(x)
        # tensor sizes may differ by 1 due to pooling; crop if necessary
        if x.shape[-2:] != skip.shape[-2:]:
            # center crop skip to match x
            diff_y = skip.shape[-2] - x.shape[-2]
            diff_x = skip.shape[-1] - x.shape[-1]
            skip = skip[
                :,
                :,
                diff_y // 2 : skip.shape[-2] - diff_y // 2,
                diff_x // 2 : skip.shape[-1] - diff_x // 2,
            ]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LargeUNet(pl.LightningModule):

    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Encoder
        self.stem = nn.Sequential(
            _ResidualBlock(3, 32),
            _ResidualBlock(32, 32),
        )
        self.down1 = _Down(32, 64)  # 1/2
        self.down2 = _Down(64, 128)  # 1/4
        self.down3 = _Down(128, 256)  # 1/8

        # Bottleneck
        self.bottleneck = nn.Sequential(
            _ResidualBlock(256, 512),
            _ResidualBlock(512, 512),
        )

        # Decoder
        self.up3 = _Up(512, 256)  # 1/4
        self.up2 = _Up(256, 128)  # 1/2
        self.up1 = _Up(128, 64)  # 1

        # Head
        self.head = nn.Sequential(
            _ResidualBlock(64, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.f1 = torchmetrics.classification.BinaryF1Score()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)  # (B,32,H,W)
        s1 = self.down1(s0)  # (B,64,H/2,W/2)
        s2 = self.down2(s1)  # (B,128,H/4,W/4)
        s3 = self.down3(s2)  # (B,256,H/8,W/8)

        b = self.bottleneck(s3)  # (B,512,H/8,W/8)

        d2 = self.up3(b, s3)  # (B,256,H/4,W/4)
        d1 = self.up2(d2, s2)  # (B,128,H/2,W/2)
        d0 = self.up1(d1, s1)  # (B,64, H,W)

        out = self.head(d0)  # (B,1,H,W)
        return out

    def _shared_step(self, batch, stage: str):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, masks)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        preds = torch.sigmoid(logits)
        self.f1(preds, masks.int())
        self.log(f"{stage}_f1", self.f1, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
