from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from road_segmentation.plots_drawer.plots_drawer import PlotMetricsCallback
from road_segmentation.production.pipeline import FullPipeline


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

    draw_plots_cb = PlotMetricsCallback(out_dir=cfg.plots.out_dir)

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
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    ckpt_path = hydra_output_dir / "last.ckpt"
    # create a symlink to the final checkpoint in the root directory

    (root_dir / "outputs" / "last.ckpt").unlink(missing_ok=True)
    (root_dir / "outputs" / "last.ckpt").symlink_to(ckpt_path)

    if cfg.production.export_onnx:
        target_cls_name = cfg.model._target_.split(".")[-1]
        if target_cls_name.lower().startswith("miniunet"):
            model_name = "mini_unet"
        elif target_cls_name.lower().startswith("largeunet"):
            model_name = "large_unet"
        else:
            raise ValueError(f"Unsupported model class: {target_cls_name}")

        # 8.1) Создаём FullPipeline, который внутри forward выполняет весь препро и postпро
        pipeline = FullPipeline(model_name=model_name, ckpt_path=str(ckpt_path))
        pipeline.eval().cpu()

        # 8.2) Готовим "dummy input" типа uint8, размер 1×3×400×400
        dummy_input = torch.randint(0, 256, (1, 3, 400, 400), dtype=torch.uint8)

        # 8.3) Сохраняем в ONNX: <Hydra-run-dir>/model_full.onnx
        onnx_path = hydra_output_dir / "model_full.onnx"
        torch.onnx.export(
            pipeline,  # FullPipeline с препро/постпро
            dummy_input,  # пример uint8 входа
            str(onnx_path),  # куда сохранить ONNX
            opset_version=13,
            input_names=["raw_image"],
            output_names=["binary_mask"],
            dynamic_axes={
                "raw_image": {0: "batch", 2: "height", 3: "width"},
                "binary_mask": {0: "batch", 2: "height", 3: "width"},
            },
        )
        print(f"ONNX модель сохранена в: {onnx_path}")
        # create a symlink to the ONNX model in the root directory
        (root_dir / "outputs" / "model_full.onnx").unlink(missing_ok=True)
        (root_dir / "outputs" / "model_full.onnx").symlink_to(onnx_path)


if __name__ == "__main__":
    main()
