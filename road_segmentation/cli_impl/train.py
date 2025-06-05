from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from road_segmentation.constants import constants
from road_segmentation.plots_drawer.plots_drawer import PlotMetricsCallback
from road_segmentation.production.pipeline import FullPipeline


def export_to_onnx(cfg: DictConfig, ckpt_path: Path):
    """
    Экспортирует модель вместе с пайплайном препроцессинга и постпроцессинга в ONNX.
    """
    target_cls_name = cfg.model._target_.split(".")[-2]
    if target_cls_name in constants.MODEL_NAMES:
        model_name = target_cls_name
    else:
        raise ValueError(
            f"Модель {target_cls_name} не поддерживается для экспорта в ONNX,"
            f" используйте одну из: {', '.join(constants.MODEL_NAMES)}"
        )

    pipeline = FullPipeline(model_name=model_name, ckpt_path=ckpt_path)
    pipeline.eval().cpu()

    dummy_input = torch.randint(
        constants.PIXEL_MIN,
        constants.PIXEL_MAX + 1,
        constants.DUMMY_INPUT_ONNX_SHAPE,
        dtype=torch.uint8,
    )

    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    onnx_path = hydra_output_dir / constants.ONNX_MODEL_NAME

    torch.onnx.export(
        pipeline,
        dummy_input,
        str(onnx_path),
        opset_version=constants.ONNX_OPSET_VERSION,
        input_names=[constants.ONNX_INPUT_NAME],
        output_names=[constants.ONNX_OUTPUT_NAME],
        dynamic_axes={
            constants.ONNX_INPUT_NAME: {0: "batch", 2: "height", 3: "width"},
            constants.ONNX_OUTPUT_NAME: {0: "batch", 2: "height", 3: "width"},
        },
    )
    print(f"ONNX модель сохранена в: {onnx_path}")

    root_dir = Path(__file__).resolve().parents[2]
    (root_dir / constants.HYDRA_OUTPUT_DIR / constants.ONNX_MODEL_NAME).unlink(missing_ok=True)
    (root_dir / constants.HYDRA_OUTPUT_DIR / constants.ONNX_MODEL_NAME).symlink_to(onnx_path)


@hydra.main(
    version_base="1.3",
    config_path=str(Path(__file__).resolve().parents[2] / "configs"),
    config_name="train",
)
def train_main(cfg: DictConfig):
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

    callbacks = [ckpt_cb, draw_plots_cb]
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=constants.HYDRA_OUTPUT_DIR,
        **cfg.trainer,
    )
    trainer.fit(model, dm)

    root_dir = Path().cwd()
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    ckpt_path = hydra_output_dir / constants.CKPT_FILE_NAME

    (root_dir / constants.HYDRA_OUTPUT_DIR / constants.CKPT_FILE_NAME).unlink(missing_ok=True)
    (root_dir / constants.HYDRA_OUTPUT_DIR / constants.CKPT_FILE_NAME).symlink_to(ckpt_path)

    if cfg.production.export_onnx:
        export_to_onnx(cfg, ckpt_path)
