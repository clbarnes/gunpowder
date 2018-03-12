from .provider_test import ProviderTest
from gunpowder import *
import numpy as np

try:
    from unittest import mock
except ImportError:
    import mock


class TestCatmaidSource(ProviderTest):
    def setUp(self):
        super(TestCatmaidSource, self).setUp()
        self.target_orientation = 'zyx'

        # based on L1 stack (rounded)
        self.resolution = {'z': 50, 'y': 4, 'x': 4}
        self.dimension = {'z': 5000, 'y': 32000, 'x': 30000}
        self.translation = {'z': 121 * self.resolution['z'], 'y': 0, 'x': 0}

        self.fill = 10

    def get_image_fetcher_mock(self):
        m = mock.Mock()
        m.target_orientation = self.target_orientation
        m.stack.resolution = self.resolution
        m.stack.dimension = self.dimension
        m.stack.translation = self.translation

        def side_effect(roi, zoom_level):
            assert zoom_level == 0
            shape = np.diff(roi, axis=0).squeeze()
            arr = np.empty(shape)
            arr.fill(self.fill)
            return arr

        m.get_scaled_space.side_effect = side_effect

        return m

    def test_output_3d(self):
        # create array keys
        raw = ArrayKey('RAW')

        image_fetcher = self.get_image_fetcher_mock()

        pipeline = (
            CatmaidSource(image_fetcher, raw, True) +
            Snapshot(
                {
                    raw: '/volumes/raw',
                },
                output_filename = self.path_to('catmaid_source_test.hdf')
            )
        )

        offset = Coordinate([self.translation['z'], 2000, 3000])  # world
        expected_shape = 20, 500, 300  # of output array

        world_shape = Coordinate(expected_shape) * tuple([self.resolution[dim] for dim in self.target_orientation])

        with build(pipeline):

            batch = pipeline.request_batch(
                BatchRequest({
                    raw: ArraySpec(roi=Roi(offset, world_shape)),
                })
            )

            self.assertTrue(batch.arrays[raw].spec.interpolatable)
            self.assertEqual(batch.arrays[raw].spec.voxel_size, (50, 4, 4))

            self.assertSequenceEqual(batch.arrays[raw].data.shape, expected_shape)
            self.assertEqual((batch.arrays[raw].data != self.fill).sum(), 0)

            args, kwargs = image_fetcher.get_scaled_space.call_args_list[0]
            roi, zoom = args
            self.assertEqual(zoom, 0)

            expected_roi = [
                [0, 500, 750],
                [20, 1000, 1050]
            ]
            self.assertTrue(np.allclose(roi, expected_roi))
