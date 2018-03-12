import os
import copy
import logging
import numpy as np

from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.ext import h5py, z5py
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from .batch_provider import BatchProvider

logger = logging.getLogger(__name__)


class Hdf5LikeSource(BatchProvider):
    '''An HDF5-like data source.

        Provides arrays from datasets accessed with an h5py-like API for each array
        key given. If the attribute `resolution` is set in an HDF5 dataset, it will
        be used as the array's `voxel_size` and a warning issued if they differ. If
        the attribute `offset` is set in a dataset, it will be used as the
        offset of the :class:`Roi` for this array. It is assumed that the offset
        is given in world units.

        Args:

            filename (string): The file path.

            datasets (dict): Dictionary of ArrayKey -> dataset names that this
                source offers.

            array_specs (dict, optional): An optional dictionary of
                :class:`ArrayKey` to :class:`ArraySpec` to overwrite the array
                specs automatically determined from the data file. This is useful
                to set a missing ``voxel_size``, for example. Only fields that are
                not ``None`` in the given :class:`ArraySpec` will be used.
        '''
    def __init__(
            self,
            filename,
            datasets,
            array_specs=None):

        self.filename = filename
        self.datasets = datasets

        if array_specs is None:
            self.array_specs = {}
        else:
            self.array_specs = array_specs

        self.ndims = None
    
    def _open_file(self, filename):
        raise NotImplementedError('Only implemented in subclasses')

    def setup(self):
        with self._open_file(self.filename) as data_file:
            for (array_key, ds_name) in self.datasets.items():
        
                if ds_name not in data_file:
                    raise RuntimeError("%s not in %s" % (ds_name, self.filename))
        
                spec = self.__read_spec(array_key, data_file, ds_name)
        
                self.provides(array_key, spec)

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        with self._open_file(self.filename) as data_file:
            for (array_key, request_spec) in request.array_specs.items():
                logger.debug("Reading %s in %s...", array_key, request_spec.roi)

                voxel_size = self.spec[array_key].voxel_size

                # scale request roi to voxel units
                dataset_roi = request_spec.roi / voxel_size

                # shift request roi into dataset
                dataset_roi = dataset_roi - self.spec[array_key].roi.get_offset() / voxel_size

                # create array spec
                array_spec = self.spec[array_key].copy()
                array_spec.roi = request_spec.roi

                # add array to batch
                batch.arrays[array_key] = Array(
                    self.__read(data_file, self.datasets[array_key], dataset_roi),
                    array_spec)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __read_spec(self, array_key, data_file, ds_name):

        dataset = data_file[ds_name]

        dims = Coordinate(dataset.shape)

        if self.ndims is None:
            self.ndims = len(dims)
        else:
            assert self.ndims == len(dims)

        if array_key in self.array_specs:
            spec = self.array_specs[array_key].copy()
        else:
            spec = ArraySpec()

        if spec.voxel_size is None:

            if 'resolution' in dataset.attrs:
                spec.voxel_size = Coordinate(dataset.attrs['resolution'])
            else:
                spec.voxel_size = Coordinate((1,) * self.ndims)
                logger.warning("WARNING: File %s does not contain resolution information "
                               "for %s (dataset %s), voxel size has been set to %s. This "
                               "might not be what you want.",
                               self.filename, array_key, ds_name, spec.voxel_size)

        if spec.roi is None:

            if 'offset' in dataset.attrs:
                offset = Coordinate(dataset.attrs['offset'])
            else:
                offset = Coordinate((0,) * self.ndims)

            spec.roi = Roi(offset, dims * spec.voxel_size)

        if spec.dtype is not None:
            assert spec.dtype == dataset.dtype, ("dtype %s provided in array_specs for %s, "
                                                 "but differs from dataset %s dtype %s" %
                                                 (self.array_specs[array_key].dtype,
                                                  array_key, ds_name, dataset.dtype))
        else:
            spec.dtype = dataset.dtype

        if spec.interpolatable is None:
            spec.interpolatable = spec.dtype in [
                np.float,
                np.float32,
                np.float64,
                np.float128,
                np.uint8  # assuming this is not used for labels
            ]
            logger.warning("WARNING: You didn't set 'interpolatable' for %s "
                           "(dataset %s). Based on the dtype %s, it has been "
                           "set to %s. This might not be what you want.",
                           array_key, ds_name, spec.dtype,
                           spec.interpolatable)

        return spec

    def __read(self, data_file, ds_name, roi):
        return np.asarray(data_file[ds_name][roi.get_bounding_box()])

    def __repr__(self):

        return self.filename

    def infer_source_type(self, filename, *args, **kwargs):
        ext = os.path.splitext(filename)[1].lower()
        return {
            '.hdf': Hdf5Source,
            '.hdf5': Hdf5Source,
            '.h5': Hdf5Source,
            '.zarr': ZarrSource,
            '.zr': ZarrSource,
            '.n5': N5Source,
        }[ext](filename, *args, **kwargs)


class Hdf5Source(Hdf5LikeSource):
    def _open_file(self, filename):
        return h5py.File(filename, 'r')


class N5Source(Hdf5LikeSource):
    def _open_file(self, filename):
        return z5py.File(filename, use_zarr_format=False)


class ZarrSource(Hdf5LikeSource):
    def _open_file(self, filename):
        return z5py.File(filename, use_zarr_format=True)
