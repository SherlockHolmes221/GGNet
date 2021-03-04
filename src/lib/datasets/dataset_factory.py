from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.hico import HICO
from .dataset.vcoco import VCOCO

dataset_factory = {
    'hico': HICO,
    'vcoco': VCOCO,
}


def get_dataset(dataset):
    class Dataset(dataset_factory[dataset]):
        pass

    return Dataset
