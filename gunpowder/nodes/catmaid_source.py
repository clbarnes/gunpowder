from __future__ import division
import logging
import numpy as np

from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.array import Array, ArrayKey, ArrayKeys
from gunpowder.array_spec import ArraySpec
from .batch_provider import BatchProvider

logger = logging.getLogger(__name__)

ORIENTATION = 'zyx'
ZOOM_LEVEL = 0
NDIMS = 3


class CatmaidSource(BatchProvider):
    '''A data source from CATMAID-compatible tile servers.

    Currently only uint8 greyscale data at zoom level 0, in target orientation 'zyx', is accepted.

    Provides arrays from CATMAID datasets for each array key given.

    It is assumed that the offset is given in world units from the stack origin.

    Args:

        image_fetcher (catpy.image.ImageFetcher) : image fetcher wrapping ProjectStack

        array_key (ArrayKey) : Key of image data (default ArrayKeys.RAW)

        interpolatable (bool) : Whether or not image should be interpolated
    '''

    def __init__(self, image_fetcher, array_key=ArrayKey('RAW'), interpolatable=True):
        assert image_fetcher.target_orientation == ORIENTATION
        self.image_fetcher = image_fetcher
        self.array_key = array_key

        self.zoom_level = ZOOM_LEVEL
        self.ndims = NDIMS  # catpy ImageFetcher currently only supports 3 dimensions

        self.dtype = np.uint8
        self.interpolatable = interpolatable

    def setup(self):
        offset = self.__dict_to_tuple('translation')
        shape = self.__dict_to_tuple('dimension')
        voxel_size = self.__dict_to_tuple('resolution')
        spec = ArraySpec(Roi(offset, Coordinate(shape) * voxel_size), voxel_size, self.interpolatable, self.dtype)

        self.provides(self.array_key, spec)

    def __dict_to_tuple(self, name):
        """offset (world coords), shape (pixels), voxel_size from stack"""
        return Coordinate(getattr(self.image_fetcher.stack, name)[dim] for dim in ORIENTATION)

    def provide(self, request):
        timing = Timing(self)
        timing.start()

        batch = Batch()

        for (array_key, request_spec) in request.array_specs.items():

            logger.debug("Reading %s in %s...", array_key, request_spec.roi)

            voxel_size = self.spec[array_key].voxel_size

            translated_roi = request_spec.roi - self.spec[array_key].roi.get_offset()
            scaled_roi = translated_roi / voxel_size

            # create array spec
            array_spec = self.spec[array_key].copy()
            array_spec.roi = request_spec.roi

            offset = scaled_roi.get_offset()
            catpy_roi = np.array([list(offset), list(offset + scaled_roi.get_shape())])

            # add array to batch
            batch.arrays[array_key] = Array(
                self.image_fetcher.get_scaled_space(catpy_roi, self.zoom_level), array_spec
            )

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __repr__(self):
        return super(CatmaidSource, self).__repr__() + 'wrapping {}'.format(str(self.image_fetcher))
