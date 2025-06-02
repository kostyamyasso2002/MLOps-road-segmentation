import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    dm = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)

    mlflow_logger = hydra.utils.instantiate(cfg.mlflow_logger)
    run_name = mlflow_logger.run_id
    plots_logger = hydra.utils.instantiate(cfg.plots_logger, version=run_name)

    ckpt_cb = ModelCheckpoint(
        dirpath=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        filename="final",
        save_top_k=0,
        save_last=True,
        verbose=True,
    )

    callbacks = [hydra.utils.instantiate(cfg.callback), ckpt_cb]
    trainer = pl.Trainer(
        logger=[mlflow_logger, plots_logger],
        callbacks=callbacks,
        default_root_dir="outputs",
        **cfg.trainer,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
