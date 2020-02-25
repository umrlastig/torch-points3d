from typing import Dict
import torchnet as tnt

from src.models.base_model import BaseModel
from .base_tracker import BaseTracker
from .registration_accuracy import compute_accuracy


class PatchRegistrationTracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False):
        """
        generic tracker for registration task.
        to track results, it measures the loss, and the accuracy.
        only useful for the training.
        """

        super(PatchRegistrationTracker, self).__init__(stage, wandb_log, use_tensorboard)

        self.reset(stage)

    def reset(self, stage="train"):
        super().reset(stage=stage)

    def track(self, model: BaseModel):
        """ Add model predictions (accuracy)
        """
        super().track(model)

        outputs = self._convert(model.get_output())

        self._acc = compute_accuracy(outputs[::2], outputs[1::2])

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        metrics["{}_acc".format(self._stage)] = self._acc
        return metrics
