import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class MiniUNet(pl.LightningModule):
    def __init__(self, lr: float, threshold: float, pos_weight: float):
        super().__init__()
        self.save_hyperparameters()

        self.down1 = _ConvBlock(3, 32)
        self.down2 = _ConvBlock(32, 64)
        self.down3 = _ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = _ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = _ConvBlock(64, 32)

        self.final = nn.Conv2d(32, 1, 1)

        pw = torch.tensor([self.hparams.pos_weight])
        self.register_buffer("pos_weight_buf", pw)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_buf)

        self.f1 = torchmetrics.F1Score(task="binary")
        self.accuracy = torchmetrics.Accuracy(task="binary")

    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(self.pool(c1))
        c3 = self.down3(self.pool(c2))

        u2 = self.up2(c3)
        x = self.conv2(torch.cat([u2, c2], dim=1))
        u1 = self.up1(x)
        x = self.conv1(torch.cat([u1, c1], dim=1))
        return self.final(x)

    # Lightning steps
    def _shared_step(self, batch, stage: str):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, masks)

        preds = (torch.sigmoid(logits) > self.hparams.threshold).int()
        f1 = self.f1(preds, masks.int())
        acc = self.accuracy(preds, masks.int())

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_f1", f1, prog_bar=True)
        self.log(f"{stage}_accuracy", acc, prog_bar=True)

        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
