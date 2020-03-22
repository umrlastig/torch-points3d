import torch
rom typing import List, Optional
from torch_geometric.data import Data
from torch_geometric.data import Batch
from src.datasets.multiscale_data import MultiScaleBatch
import re

class Pair(Data):

    def __init__(
            self,
            x=None,
            y=None,
            pos=None,
            x_target=None,
            pos_target=None,
            **kwargs,
    ):
        super().__init__(x=x, pos=pos,
                         x_target=x_target, pos_target=pos_target, **kwargs)


    @staticmethod
    def make_pair(data_source: Data, data_target: Data):
        """
        add in a Data object the source elem, the target elem.
        """
        # add concatenation of the point cloud

        batch = data_source
        for key_target in data_target.keys:
            batch[key_target+"_target"] = data_target[key_target]
        return batch.contiguous()

    def to_data(self):
        data_source = Data()
        data_target = Data()
        for key in self.keys:
            match = re.search(r"(.+)_target$", key)
            if match is None:
                data_source[key] = self[key]
            else:
                new_key = match.groups()[0]
                data_target[new_key] = self[key]
        return data_source, data_target

    @property
    def num_nodes_target(self):
        for key, item in self('x_target', 'pos_target', 'norm_target', 'batch_target'):
            return item.size(self.__cat_dim__(key, item))
        return None


class MultiScalePair(Pair):
    def __init__(
            self,
            x=None,
            y=None,
            pos=None,
            multiscale: Optional[List[Data]] = None,
            upsample: Optional[List[Data]] = None,
            x_target=None,
            pos_target=None,
            multiscale_target: Optional[List[Data]] = None,
            upsample_target: Optional[List[Data]] = None,
            **kwargs,
    ):
        MultiScaleData.__init__(x=x, pos=pos,
                                multiscale=multiscale,
                                upsample=upsample,
                                x_target=x_target, pos_target=pos_target,
                                multiscale_target=multiscale_target,
                                upsample_target=upsample_target,
                                **kwargs)

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all tensor and Data attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.
        """
        for key, item in self(*keys):
            if torch.is_tensor(item):
                self[key] = func(item)
        for scale in range(self.num_scales):
            self.multiscale[scale] = self.multiscale[scale].apply(func)
            self.multiscale_target[scale] = self.multiscale_target[scale].apply(func)

        for up in range(self.num_upsample):
            self.upsample[up] = self.upsample[up].apply(func)
            self.upsample_target[up] = self.upsample_target[up].apply(func)
        return self

    @property
    def num_scales(self):
        """ Number of scales in the multiscale array
        """
        return len(self.multiscale) if self.multiscale else 0

    @property
    def num_upsample(self):
        """ Number of upsample operations
        """
        return len(self.upsample) if self.upsample else 0

    @classmethod
    def from_data(cls, data):
        ms_data = cls()
        for k, item in data:
            ms_data[k] = item
        return ms_data


class PairBatch(Pair):

    def __init__(self, batch=None, batch_target=None, **kwargs):
        r"""
        Pair batch for message passing
        """
        self.batch_target = batch_target
        Batch.__init__(batch=batch, **kwargs)
        self.__data_class__ = Pair

    @staticmethod
    def from_data_list(data_list):
        r"""
        from a list of src.datasets.registation.pair.Pair objects, create
        a batch
        Warning : follow_batch is not here yet...
        """
        assert isinstance(data_list[0], Pair)
        data_list_s, data_list_t = list(map(list, zip(*[data.to_data() for data in data_list])))
        batch_s = Batch.from_data_list(data_list_s, follow_batch)
        batch_t = Batch.from_data_list(data_list_t, follow_batch)
        return Pair.make_pair(batch_s, batch_t).contiguous()


class PairMultiScaleBatch(MultiScalePair):

    @staticmethod
    def from_data_list(data_list):
        r"""
        from a list of src.datasets.registation.pair.Pair objects, create
        a batch
        Warning : follow_batch is not here yet...
        """
        data_list_s, data_list_t = list(map(list, zip(*[data.to_pair() for data in data_list])))
        batch_s = MultiScaleBatch.from_data_list(data_list_s)
        batch_t = MultiScaleBatch.from_data_list(data_list_t)
        return MultiScalePair.make_pair(batch_s, batch_t).contiguous()
