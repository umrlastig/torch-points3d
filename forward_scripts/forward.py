import torch
import hydra
import logging
from omegaconf import OmegaConf
import os
import sys
import numpy as np
from typing import Dict
from plyfile import PlyData, PlyElement

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

# Import building function for model and dataset
from torch_points3d.datasets.dataset_factory import instantiate_dataset, get_dataset_class
from torch_points3d.models.model_factory import instantiate_model

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

# Import from metrics
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint
from torch_points3d.core.data_transform import SaveOriginalPosId

# Utils import
from torch_points3d.utils.colors import COLORS

log = logging.getLogger(__name__)


def save(prefix, predicted):
    for key, value in predicted.items():
        filename = os.path.splitext(key)[0]
        out_file = filename + "_pred"
        np.save(os.path.join(prefix, out_file), value)


def run(model: BaseModel, dataset, device, output_path):
    for i in range(len(dataset.test_dataset)):
        
        with open(dataset.test_dataset[i].raw_paths[0], "rb") as ply_file:
            plydata = PlyData.read(ply_file)

        votes = torch.zeros((plydata["vertex"].data['x'].shape[0], dataset.test_dataset[i].num_classes), dtype=torch.float)
        prediction_count = torch.zeros(plydata["vertex"].data['x'].shape[0], dtype=torch.int)

        with Ctq(dataset.test_dataloaders[i]) as tq_test_loader:
            for data in tq_test_loader:
                with torch.no_grad():
                    model.set_input(data, device)
                    model.forward()

                originids = data[SaveOriginalPosId.KEY]
                votes[originids] += model.get_output().cpu()
                prediction_count[originids] += 1
        
        mask = prediction_count >= 1
        pred = torch.argmax(votes[mask], dim=1).numpy()
        mask = mask.numpy()

        if 'scalar_clasa' in plydata["vertex"].data.dtype.names:

            label = plydata['vertex']['scalar_clasa'][mask]

            ply_array = np.ones(
                mask.sum(), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("l", "i4"), ("p", "i4"), ("error", "i4")]
            )
            ply_array["x"] = plydata['vertex']['x'][mask]
            ply_array["y"] = plydata['vertex']['y'][mask]
            ply_array["z"] = plydata['vertex']['z'][mask]
            ply_array["red"] = plydata['vertex']['red'][mask]
            ply_array["green"] = plydata['vertex']['green'][mask]
            ply_array["blue"] = plydata['vertex']['blue'][mask]
            ply_array["l"] = label
            ply_array["p"] = pred
            ply_array["error"] = (label != pred)
            el = PlyElement.describe(ply_array, "vertex")
            PlyData([el]).write("{}.ply".format(dataset.test_dataset[i].name))

        else:
            ply_array = np.ones(
                mask.sum(), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("p", "i4")]
            )
            ply_array["x"] = plydata['vertex']['x'][mask]
            ply_array["y"] = plydata['vertex']['y'][mask]
            ply_array["z"] = plydata['vertex']['z'][mask]
            ply_array["red"] = plydata['vertex']['red'][mask]
            ply_array["green"] = plydata['vertex']['green'][mask]
            ply_array["blue"] = plydata['vertex']['blue'][mask]
            ply_array["p"] = pred
            el = PlyElement.describe(ply_array, "vertex")
            PlyData([el]).write("{}.ply".format(dataset.test_dataset[i].name))


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)

    # Get device
    if cfg.cuda > -1 and torch.cuda.is_available():
        device = "cuda"
        torch.cuda.set_device(cfg.cuda)
    else:
        device = "cpu"
    device = torch.device(device)
    log.info("DEVICE : {}".format(device))

    # Enable CUDNN BACKEND
    torch.backends.cudnn.enabled = cfg.enable_cudnn

    # Checkpoint
    checkpoint = ModelCheckpoint(cfg.checkpoint_dir, cfg.model_name, cfg.weight_name, strict=True)

    # Setup the dataset config
    # Generic config
    train_dataset_cls = get_dataset_class(checkpoint.data_config)
    setattr(checkpoint.data_config, "dataroot", cfg.input_path)

    # Datset specific configs
    if cfg.data:
        for key, value in cfg.data.items():
            checkpoint.data_config.update(key, value)
    if cfg.dataset_config:
        for key, value in cfg.dataset_config.items():
            checkpoint.dataset_properties.update(key, value)

    # Create dataset and mdoel
    model = checkpoint.create_model(checkpoint.dataset_properties, weight_name=cfg.weight_name)
    log.info(model)
    log.info("Model size = %i", sum(param.numel() for param in model.parameters() if param.requires_grad))

    # Set dataloaders
    dataset = instantiate_dataset(checkpoint.data_config)
    dataset.create_dataloaders(
        model, cfg.batch_size, cfg.shuffle, cfg.num_workers, False,
    )
    log.info(dataset)

    model.eval()
    if cfg.enable_dropout:
        model.enable_dropout_in_eval()
    model = model.to(device)

    # Run training / evaluation
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path)

    run(model, dataset, device, cfg.output_path)


if __name__ == "__main__":
    main()
