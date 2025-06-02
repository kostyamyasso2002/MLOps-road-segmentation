from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl


class PlotMetricsCallback(pl.Callback):
    def __init__(self, root: str = "plots"):
        self.root = Path(root)

    def on_train_end(self, trainer: pl.Trainer, *_):
        csv_files = list(self.root.rglob("metrics.csv"))
        if not csv_files:
            return
        csv_path = csv_files[-1]
        df = pd.read_csv(csv_path)
        df = df[df["step"].isna()]

        plots = {
            "train_loss": "train_loss.png",
            "val_loss": "val_loss.png",
            "val_f1": "val_f1.png",
        }
        for col, fname in plots.items():
            if col not in df.columns:
                continue
            plt.figure()
            plt.plot(df["epoch"], df[col], marker="o")
            plt.xlabel("epoch")
            plt.ylabel(col)
            plt.tight_layout()
            plt.savefig(csv_path.parent / fname)
            plt.close()
