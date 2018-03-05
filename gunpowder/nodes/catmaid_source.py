import logging

import numpy as np

from .batch_provider import BatchProvider
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.profiling import Timing
from gunpowder.provider_spec import ProviderSpec
from gunpowder.roi import Roi
from gunpowder.volume import Volume, VolumeTypes

logger = logging.getLogger(__name__)


class CatmaidSource(BatchProvider):
    '''A data source pulling tiles over HTTP as CATMAID_ does.

    Uses an ImageFetcher from catpy_.image.ImageFetcher.

    Provides volumes from greyscale image datasets. Resolution (world), offset (world) and dimension (voxel)
    are defined in the passed-in ImageFetcher object.

    Args:

        image_fetcher (catpy.image.ImageFetcher):

        zoom_level (int):

        volume_type (VolumeTypes):

    .. _CATMAID: https://catmaid.readthedocs.io
    .. _catpyL https://github.com/catmaid/catpy
    '''

    def __init__(self, image_fetcher, zoom_level=0, volume_type=VolumeTypes.RAW):
        self.image_fetcher = image_fetcher
        self.zoom_level = zoom_level
        self.volume_type = volume_type
        self.spec = None
        self.resolution_scaled = None

    def setup(self):
        assert self.image_fetcher.target_orientation == 'zyx'  # todo: check

        if self.image_fetcher.mirror is None:
            logger.warning('Finding fastest stack mirror (may take a few seconds)')
            self.image_fetcher.set_fastest_mirror()

        self.spec = ProviderSpec()

        try:
            res_scaled = self.image_fetcher.coord_trans.stack_to_scaled(
                self.image_fetcher.stack.resolution, tgt_zoom=self.zoom_level
            )
            self.resolution_scaled = Coordinate([res_scaled[dim] for dim in self.image_fetcher.target_orientation])
        except AttributeError:
            logger.warning('Image tile stack does not have a defined resolution: using (1, 1, 1)')
            self.resolution_scaled = Coordinate([1, 1, 1])  # or should this be 1s before zooming?

        try:
            offset_world = self.image_fetcher.stack.translation
            offset_scaled = self.image_fetcher.coord_trans.stack_to_scaled(offset_world, tgt_zoom=self.zoom_level)
            offset_coord = [offset_scaled[dim] for dim in self.image_fetcher.target_orientation]
        except AttributeError:
            logger.warning('Image tile stack does not have a defined offset (aka translation): using (0, 0, 0)')
            offset_coord = [0, 0, 0]

        dimension_stack = self.image_fetcher.stack.dimension
        dimension_scaled = self.image_fetcher.coord_trans.stack_to_scaled(dimension_stack, self.zoom_level)
        dimension_arr = [dimension_scaled[dim] for dim in self.image_fetcher.target_orientation]

        self.spec.volumes[self.volume_type] = Roi(offset_coord, dimension_arr)

    def get_spec(self):
        return self.spec

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        spec = self.get_spec()

        batch = Batch()

        for volume_type, roi in request.volumes.items():
            if volume_type not in spec.volumes:
                raise RuntimeError("Asked for %s which this source does not provide" % volume_type)

            if not spec.volumes[volume_type].contains(roi):
                raise RuntimeError("%s's ROI %s outside of my ROI %s" % (volume_type, roi, spec.volumes[volume_type]))

            logger.debug("Reading %s in %s..." % (volume_type, roi))

            dataset_roi = roi.shift(-spec.volumes[volume_type].get_offset())
            roi_as_arr = np.array([
                dataset_roi.get_offset(),
                dataset_roi.get_offset() + dataset_roi.get_shape()
            ])

            batch.volumes[self.volume_type] = Volume(
                self.image_fetcher.get_scaled_space(roi_as_arr, zoom_level=self.zoom_level),
                roi=roi,
                resolution=self.resolution_scaled
            )

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __repr__(self):
        return '<{} wrapping {}>'.format(type(self).__name__, self.image_fetcher)

    def teardown(self):
        self.image_fetcher.clear_cache()  # should we do this?
        self.image_fetcher._session.close()
