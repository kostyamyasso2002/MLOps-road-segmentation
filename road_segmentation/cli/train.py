from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from road_segmentation.plots_drawer.plots_drawer import PlotMetricsCallback


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    dm = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)

    logger = hydra.utils.instantiate(cfg.logger)

    ckpt_cb = ModelCheckpoint(
        dirpath=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        save_top_k=0,
        save_last=True,
        verbose=True,
    )

    draw_plots_cb = PlotMetricsCallback()

    print(logger.run_id)
    print(logger.experiment.tracking_uri)

    callbacks = [ckpt_cb, draw_plots_cb]
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir="outputs",
        **cfg.trainer,
    )
    trainer.fit(model, dm)

    # save model parameters
    root_dir = Path().cwd()
    ckpt_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "last.ckpt"
    # create a symlink to the final checkpoint in the root directory

    print(f"{root_dir=}")
    if (root_dir / "outputs" / "last.ckpt").exists():
        (root_dir / "outputs" / "last.ckpt").unlink()
    (root_dir / "outputs" / "last.ckpt").symlink_to(ckpt_path)


if __name__ == "__main__":
    main()
