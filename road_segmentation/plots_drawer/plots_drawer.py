import os
from pathlib import Path

import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback


class PlotMetricsCallback(Callback):
    """
    В конце тренировки достаёт из MLflow историю метрик:
      - train_loss
      - val_loss
      - val_f1
    и сохраняет два графика:
      1) train_loss и val_loss на одном графике
      2) val_f1 на отдельном графике
    в локальную папку (по умолчанию "./plots").
    """

    def __init__(self, out_dir: str = "plots_uuu"):
        super().__init__()
        self.out_dir = Path(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

    def on_train_end(self, trainer, pl_module):
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        # 1. Убедимся, что logger — это MLFlowLogger
        #    (если вы используете другой type logger, этот код нужно адаптировать)
        logger = trainer.logger
        try:
            run_id = logger.run_id
            client = logger.experiment
        except AttributeError:
            print(
                "[PlotMetricsCallback] Внимание: тренер не использует "
                "MLFlowLogger, пропускаем построение графиков."
            )
            return

        run_name = client.get_run(run_id).data.tags.get("mlflow.runName")
        self.out_dir = self.out_dir / run_name
        self.out_dir.mkdir(exist_ok=True)

        train_loss_history = client.get_metric_history(run_id, key="train_loss")
        val_loss_history = client.get_metric_history(run_id, key="val_loss")
        val_f1_history = client.get_metric_history(run_id, key="val_f1")

        if not train_loss_history or not val_loss_history or not val_f1_history:
            print(f"[PlotMetricsCallback] Не удалось найти все три метрики в MLflow-рaне {run_id}.")
            return

        fig1, ax1 = plt.subplots()
        ax1.plot([m.value for m in train_loss_history], label="train_loss")
        # ax1.plot([m.value for m in val_loss_history], label="val_loss")
        ax1.set_xlabel("Step / 5")
        ax1.set_ylabel("Loss")
        ax1.set_title("Train Loss")
        ax1.legend()
        fig1.savefig(self.out_dir / "loss_curve.png")
        plt.close(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot([m.value for m in val_f1_history], label="val_f1")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("F1")
        ax2.set_title("Validation F1 per Epoch")
        fig2.savefig(self.out_dir / "val_f1_curve.png")
        plt.close(fig2)

        fig3, ax3 = plt.subplots()
        ax3.plot([m.value for m in val_loss_history], label="val_loss")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.set_title("Validation Loss per Epoch")
        ax3.legend()
        fig3.savefig(self.out_dir / "val_loss_curve.png")
        plt.close(fig3)

        print(f"[PlotMetricsCallback] Графики сохранены в папке: {self.out_dir}")
