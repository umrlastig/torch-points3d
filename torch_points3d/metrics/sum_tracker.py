from typing import Dict, Any
import logging
import torch
from plyfile import PlyData, PlyElement
import numpy as np

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.models import model_interface

log = logging.getLogger(__name__)


class SUMTracker(SegmentationTracker):
    def reset(self, stage, *args, **kwargs):
        super().reset(stage, *args, **kwargs)
        self._stage = stage
        self._votes = None
        self._prediction_count = None
        self._vote_miou = None
        self._vote_iou_per_class = {}

    def track(self, model: model_interface.TrackerInterface, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """

        if not self._dataset.has_labels(self._stage):
            return

        BaseTracker.track(self, model)
        
        fids = model.get_input().fid
        outputs = model.get_output()
        targets = model.get_labels()

        unique_fids, fids_map = fids.unique(dim=0, return_inverse=True)
        res = torch.zeros((unique_fids.shape[0], outputs.shape[1]), dtype=outputs.dtype, device=outputs.device).scatter_add_(0, fids_map.unsqueeze(1).expand_as(outputs), outputs)
        outputs = torch.index_select(res, 0, fids_map)

        self._compute_metrics(outputs, targets)

        # Train mode, nothing special to do
        if not self._stage.startswith("test"):
            return

        if self._votes is None:
            self._data = next((x for x in self._dataset.test_dataset if x.name == self._stage), None).data
            self._votes = torch.zeros((self._data.y.shape[0], self._num_classes), dtype=torch.float)
            self._votes = self._votes.to(model.device)
            self._prediction_count = torch.zeros(self._data.y.shape[0], dtype=torch.int)
            self._prediction_count.to(model.device)

        # Gather origin ids and check that it fits with the test set
        inputs = model.get_input()
        if inputs[SaveOriginalPosId.KEY] is None:
            raise ValueError("The inputs given to the model do not have a %s attribute." % SaveOriginalPosId.KEY)

        originids = inputs[SaveOriginalPosId.KEY]
        if originids.dim() == 2:
            originids = originids.flatten()

        # Set predictions
        outputs = model.get_output()
        self._votes[originids] += outputs
        self._prediction_count[originids] += 1
        
    def finalise(self, vote_miou=True, ply_output="", **kwargs):
        per_class_iou = self._confusion_matrix.get_intersection_union_per_class()[0]
        self._iou_per_class = {self._dataset.INV_OBJECT_LABEL[k]: "{:.2f}".format(100 * v) for k, v in enumerate(per_class_iou)}

        if vote_miou and self._votes is not None:
            # Complete for points that have a prediction
            self._votes = self._votes.to("cpu")
            self._prediction_count = self._prediction_count.to("cpu")
            c = ConfusionMatrix(self._num_classes)
            has_prediction = self._prediction_count > 0
            gt = self._data.y[has_prediction].numpy()
            pred = torch.argmax(self._votes[has_prediction], 1).numpy()
            mask = gt != self._ignore_label
            c.count_predicted_batch(gt[mask], pred[mask])
            self._vote_miou = c.get_average_intersection_union() * 100
            per_class_iou = c.get_intersection_union_per_class()[0]
            self._vote_iou_per_class = {self._dataset.INV_OBJECT_LABEL[k]: "{:.2f}".format(100 * v) for k, v in enumerate(per_class_iou)}

            mesh_id = self._data.mesh[has_prediction]
            face_id = self._data.fid[has_prediction]
            loggit = self._votes[has_prediction]
            
            index = torch.cat((mesh_id.reshape(-1,1),face_id.reshape(-1,1)), dim=1)
            unique_index, index_map = index.unique(dim=0, return_inverse=True)
            result = torch.zeros((unique_index.shape[0], loggit.shape[1]), dtype=loggit.dtype, device=loggit.device).scatter_add_(0, index_map.unsqueeze(1).expand_as(loggit), loggit)
            face_pred = torch.argmax(result, 1)

            raw_file_names = next((x for x in self._dataset.test_dataset if x.name == self._stage), None).raw_file_names

            with open("face_labels.csv", "w") as face_label_file:
                face_label_file.write("File,face_id,predicted_label\n")
                for i in range(unique_index.shape[0]):
                    face_label_file.write("{},{},{}\n".format(raw_file_names[unique_index[i,0]],unique_index[i,1],face_pred[i].item()))

            if ply_output:
                pos = self._data.pos[has_prediction].detach().cpu().numpy()
                rgb = (255*self._data.rgb[has_prediction].detach().cpu().numpy()).astype(np.uint8)
                ply_array = np.ones(
                    pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("l", "i4"), ("p", "i4"), ("error", "i4")]
                )
                ply_array["x"] = pos[:, 0]
                ply_array["y"] = pos[:, 1]
                ply_array["z"] = pos[:, 2]
                ply_array["red"] = rgb[:, 0]
                ply_array["green"] = rgb[:, 1]
                ply_array["blue"] = rgb[:, 2]
                ply_array["l"] = gt
                ply_array["p"] = pred
                ply_array["error"] = ((gt != pred) & (gt != self._ignore_label))
                el = PlyElement.describe(ply_array, "vertex")
                PlyData([el]).write(ply_output)


    @property
    def full_confusion_matrix(self):
        return self._full_confusion

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        if verbose:
            metrics["{}_iou_per_class".format(self._stage)] = self._iou_per_class
            if self._vote_miou:
                metrics["{}_vote_miou".format(self._stage)] = self._vote_miou
                metrics["{}_vote_iou_per_class".format(self._stage)] = self._vote_iou_per_class
        return metrics
