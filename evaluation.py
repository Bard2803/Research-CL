from avalanche.evaluation.metrics import forgetting_metrics, class_accuracy_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics, gpu_usage_metrics, ram_usage_metrics, \
forward_transfer_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger, TextLogger, WandBLogger
from datetime import datetime



class Evaluation():
    def __init__(self, config):
        self.config = config
        loggers = []
        project_name = "CL-Research"
        group_name = str(datetime.now())
        loggers.append(WandBLogger(project_name=project_name, run_name="Naive", params={"reinit": True, "group": group_name}))
        loggers.append(InteractiveLogger())

        self.eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, stream=True),
        accuracy_metrics(epoch=True, stream=True),
        class_accuracy_metrics(epoch=True, stream=True),
        cpu_usage_metrics(epoch=True),
        gpu_usage_metrics(gpu_id=0, epoch=True),
        ram_usage_metrics(epoch=True),
        disk_usage_metrics(epoch=True),
        # forward_transfer_metrics(stream=True),
        forgetting_metrics(stream=True),
        loggers=loggers,
        strict_checks=False)

    def get_eval_plugin(self):
        return self.eval_plugin