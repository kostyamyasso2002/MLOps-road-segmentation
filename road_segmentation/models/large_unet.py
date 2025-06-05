import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class _ConvBlock(nn.Module):
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

    def forward(self, inp):
        return self.net(inp)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _ConvBlock(in_ch, out_ch)

    def forward(self, inp):
        inp = self.pool(inp)
        return self.conv(inp)


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = _ConvBlock(out_ch * 2, out_ch)

    def forward(self, inp: torch.Tensor, skip: torch.Tensor):
        inp = self.up(inp)
        _, _, h_skip, w_skip = skip.shape
        _, _, h_x, w_x = inp.shape

        dh = (h_skip - h_x) // 2
        dw = (w_skip - w_x) // 2

        skip = skip[:, :, dh : dh + h_x, dw : dw + w_x]

        inp = torch.cat([inp, skip], dim=1)
        return self.conv(inp)


class LargeUNet(pl.LightningModule):
    def __init__(self, lr: float, threshold: float, pos_weight: float):
        super().__init__()
        self.save_hyperparameters()

        # encoder
        self.stem = _ConvBlock(3, 32)
        self.down1 = _Down(32, 64)
        self.down2 = _Down(64, 128)
        self.down3 = _Down(128, 256)

        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = nn.Sequential(
            _ConvBlock(256, 512),
            _ConvBlock(512, 512),
        )

        # decoder
        self.up3 = _Up(512, 256)
        self.up2 = _Up(256, 128)
        self.up1 = _Up(128, 64)

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
        self.accuracy = torchmetrics.Accuracy(task="binary")

    def forward(self, inp):
        s0 = self.stem(inp)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        b = self.bottleneck(self.pool(s3))

        d2 = self.up3(b, s3)
        d1 = self.up2(d2, s2)
        d0 = self.up1(d1, s1)
        out = self.head(d0)
        out = F.interpolate(out, size=inp.shape[2:], mode="bilinear", align_corners=False)
        return out

    def _shared_step(self, batch, stage: str):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, masks)
        preds = (torch.sigmoid(logits) > self.hparams.threshold).int()

        f1 = self.f1(preds, masks.int())
        acc = self.accuracy(preds, masks.int())

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_f1", f1, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
