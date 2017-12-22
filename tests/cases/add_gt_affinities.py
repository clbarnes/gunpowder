from .provider_test import ProviderTest
from gunpowder import *
from itertools import product
import itertools
import numpy as np
import logging

class TestSource(BatchProvider):

    def setup(self):

        self.provides(
            ArrayTypes.GT_LABELS, ArraySpec(
                roi=Roi((-40, -40, -40), (160, 160, 160)),
                voxel_size=(20, 4, 8),
                interpolatable=False))
        self.provides(
            ArrayTypes.GT_MASK, ArraySpec(
                roi=Roi((-40, -40, -40), (160, 160, 160)),
                voxel_size=(20, 4, 8),
                interpolatable=False))

    def provide(self, request):

        batch = Batch()

        roi = request[ArrayTypes.GT_LABELS].roi
        shape = (roi/self.spec[ArrayTypes.GT_LABELS].voxel_size).get_shape()
        spec = self.spec[ArrayTypes.GT_LABELS].copy()
        spec.roi = roi

        batch.arrays[ArrayTypes.GT_LABELS] = Array(
            np.random.randint(
                0, 2,
                shape
            ),
            spec
        )

        roi = request[ArrayTypes.GT_MASK].roi
        shape = (roi/self.spec[ArrayTypes.GT_MASK].voxel_size).get_shape()
        spec = self.spec[ArrayTypes.GT_MASK].copy()
        spec.roi = roi

        batch.arrays[ArrayTypes.GT_MASK] = Array(
            np.random.randint(
                0, 2,
                shape
            ),
            spec
        )

        return batch

class TestAddGtAffinities(ProviderTest):

    def test_output(self):

        # skip the test if malis is not installed
        if isinstance(gunpowder.ext.malis, gunpowder.ext.NoSuchModule):
            return

        neighborhood = [
                Coordinate((-2,0,0)),
                Coordinate((0,-1,0)),
                Coordinate((0,0,1)),
                Coordinate((1,1,1))
        ]

        register_array_type('GT_AFFINITIES_MASK')

        pipeline = (
                TestSource() +
                AddGtAffinities(
                    neighborhood,
                    gt_labels=ArrayTypes.GT_LABELS,
                    gt_labels_mask=ArrayTypes.GT_MASK,
                    gt_affinities=ArrayTypes.GT_AFFINITIES,
                    gt_affinities_mask=ArrayTypes.GT_AFFINITIES_MASK)
        )

        with build(pipeline):

            for i in range(10):

                request = BatchRequest()
                request.add(ArrayTypes.GT_LABELS, (100,16,64))
                request.add(ArrayTypes.GT_MASK, (100,16,64))
                request.add(ArrayTypes.GT_AFFINITIES, (100,16,64))
                request.add(ArrayTypes.GT_AFFINITIES_MASK, (100,16,64))

                batch = pipeline.request_batch(request)

                self.assertTrue(ArrayTypes.GT_LABELS in batch.arrays)
                self.assertTrue(ArrayTypes.GT_MASK in batch.arrays)
                self.assertTrue(ArrayTypes.GT_AFFINITIES in batch.arrays)
                self.assertTrue(ArrayTypes.GT_AFFINITIES_MASK in batch.arrays)

                labels = batch.arrays[ArrayTypes.GT_LABELS]
                labels_mask = batch.arrays[ArrayTypes.GT_MASK]
                affs = batch.arrays[ArrayTypes.GT_AFFINITIES]
                affs_mask = batch.arrays[ArrayTypes.GT_AFFINITIES_MASK]

                self.assertTrue((len(neighborhood),) + labels.data.shape == affs.data.shape)

                voxel_roi = Roi((0,0,0), labels.data.shape)
                for (z,y,x) in product(*[range(d) for d in labels.data.shape]):

                    p = Coordinate((z,y,x))

                    for n in range(len(neighborhood)):

                        pn = p + neighborhood[n]
                        if not voxel_roi.contains(pn):
                            continue

                        a = labels.data[p]
                        b = labels.data[pn]
                        masked = (
                            labels_mask.data[p] == 0 or
                            labels_mask.data[pn] == 0)

                        if a == b and a != 0 and b != 0:
                            self.assertEqual(affs.data[(n,)+p], 1.0, "%s -> %s, %s -> %s, but is not 1"%(p, pn, a, b))
                        else:
                            self.assertEqual(affs.data[(n,)+p], 0.0, "%s -> %s, %s -> %s, but is not 0"%(p, pn, a, b))
                        if masked:
                            self.assertEqual(affs_mask.data[(n,)+p], 0.0, (
                                "%s or %s are masked, but mask is not 0"%
                                (p, pn)))
