import os
from omegaconf import DictConfig, OmegaConf

from . import ModelFactory
from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.datasets.multiscale_data import MultiScaleBatch

CUR_FILE = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONFIG = os.path.join(DIR_PATH, "conf/kpconv")


def KPConv(
    architecture: str = None,
    input_nc: int = None,
    output_nc: int = None,
    num_layers: int = None,
    config: DictConfig = None,
    **kwargs
):
    factory = KPConvFactory(
        architecture=architecture,
        num_layers=num_layers,
        input_nc=input_nc,
        output_nc=output_nc,
        config=config,
        **kwargs
    )
    return factory.build()


class KPConvFactory(ModelFactory):
    def _build_unet(self):
        if self._config:
            model_config = self._config
        else:
            path_to_model = os.path.join(PATH_TO_CONFIG, "unet_{}.yaml".format(self.num_layers))
            model_config = OmegaConf.load(path_to_model)
        self.resolve_model(model_config)
        modules_lib = sys.modules[__name__]
        return KPConvUnet(model_config, None, None, modules_lib)


class KPConvUnet(UnwrappedUnetBasedModel):
    CONV_TYPE = "partial_dense"

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters
        -----------
        data:
            a dictionary that contains the data itself and its metadata information.
        device
            Device on which to run the code. cpu or cuda
        """
        data = data.to(device)
        if isinstance(data, MultiScaleBatch):
            self.pre_computed = data.multiscale
            self.upsample = data.upsample
            del data.upsample
            del data.multiscale
        else:
            self.upsample = None
            self.pre_computed = None

        self.input = data

    def forward(self):
        """Run forward pass.
        Input --- D1 -- D2 -- D3 -- U1 -- U2 -- output
                   |      |_________|     |
                   |______________________|

        """
        return super().forward(self.input, precomputed_down=self.pre_computed, precomputed_up=self.upsample)
