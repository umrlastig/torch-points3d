import unittest
from models.utils import find_model_using_name
from omegaconf import OmegaConf
import os

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


class TestModelUtils(unittest.TestCase):
    def setUp(self):
        models_conf = os.path.join(ROOT, 'conf/models/segmentation.yaml')
        config_file = OmegaConf.load(os.path.join(ROOT, 'conf/config.yaml'))

        self.config = OmegaConf.load(models_conf)
        self.config = OmegaConf.merge(self.config, config_file.training)

    def test_findmodel(self):
        for model_name in self.config['models'].keys():
            print(model_name)
            if model_name not in ["MyTemplateModel"]:
                params = self.config['models'][model_name]
                model = find_model_using_name(params.type, 'segmentation', params, 10)
