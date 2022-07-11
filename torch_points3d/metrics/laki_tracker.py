import torch

from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.models import model_interface
from plyfile import PlyData
from torch_points3d.core.data_transform import SaveOriginalPosId


class LAKITracker(SegmentationTracker):
    def __init__(
        self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False, ignore_label: int = IGNORE_LABEL
    ):
        """ This is a tracker for LAKI dataset.
        It uses a confusion matrix in the back-end to track results.
        Use the tracker to track an epoch.
        You can use the reset function before you start a new epoch

        Arguments:
            dataset  -- dataset to track (used for the number of classes)

        Keyword Arguments:
            stage {str} -- current stage. (train, validation, test, etc...) (default: {"train"})
            wandb_log {str} --  Log using weight and biases
        """
        super(LAKITracker, self).__init__(dataset, stage, wandb_log, use_tensorboard, ignore_label)

        self._stage == stage

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._stage == stage

        if stage in ["test", "inference"]:

            for i in range(len(self._dataset.test_dataset)):
                if self._dataset.test_dataset[i].name == stage:
                    break

            with open(self._dataset.test_dataset[i].raw_paths[0], "rb") as ply_file:
                plydata = PlyData.read(ply_file)

            self._votes = torch.zeros((plydata["vertex"].data['x'].shape[0], self._dataset.test_dataset[i].num_classes), dtype=torch.float)
            self._prediction_count = torch.zeros(plydata["vertex"].data['x'].shape[0], dtype=torch.int)

            if 'scalar_clasa' in plydata['vertex']:
                self._label = plydata['vertex']['scalar_clasa']
            else:
                self._label = None

    def track(self, model: model_interface.TrackerInterface, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        if not self._dataset.has_labels(self._stage):
            return
        
        if self._stage != "test":
            super().track(model)
        else:
            super(SegmentationTracker, self).track(model)

            originids = model.get_input()[SaveOriginalPosId.KEY]
            self._votes[originids] += model.get_output().cpu()
            self._prediction_count[originids] += 1

    def finalise(self, **kwargs):
        if self._stage in ["test", "inference"]:

            mask = self._prediction_count >= 1
            outputs = self._votes[mask].numpy()
            mask = mask.numpy()

            label = self._label[mask]
            self._compute_metrics(outputs, label)