import logging

from .batch import Batch
from .batch_provider_tree import *
from .batch_request import BatchRequest
from .build import build
from .coordinate import Coordinate
from .nodes import *
from .producer_pool import ProducerPool
from .provider_spec import ProviderSpec
from .roi import Roi
from .volume import register_volume_type, VolumeType, VolumeTypes, Volume
import gunpowder.caffe

# logging.basicConfig(level=logging.INFO)

def set_verbose(verbose=True):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
