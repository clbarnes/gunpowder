from .provider_test import ProviderTest
from gunpowder import *
import numpy as np

class ChunkTestSource(BatchProvider):

    def get_spec(self):

        spec = ProviderSpec()
        spec.volumes[VolumeTypes.RAW] = Roi((1000,1000,1000), (100,100,100))
        spec.volumes[VolumeTypes.GT_LABELS] = Roi((1010,1010,1010), (90,90,90))
        return spec

    def provide(self, request):

        # print("ChunkTestSource: Got request " + str(request))

        batch = Batch()

        # have the pixels encode their position
        for (volume_type, roi) in request.volumes.items():

            # print("ChunkTestSource: Adding " + str(volume_type))

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                    range(int(roi.get_begin()[0]), int(roi.get_end()[0])),
                    range(int(roi.get_begin()[1]), int(roi.get_end()[1])),
                    range(int(roi.get_begin()[2]), int(roi.get_end()[2])), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            # print("Roi is: " + str(roi))

            batch.volumes[volume_type] = Volume(
                    data,
                    roi,
                    (1,1,1))
        return batch

class TestChunk(ProviderTest):

    def test_output(self):

        source = ChunkTestSource()

        raw_roi = source.get_spec().volumes[VolumeTypes.RAW]
        labels_roi = source.get_spec().volumes[VolumeTypes.RAW]

        chunk_request = BatchRequest()
        chunk_request.add_volume_request(VolumeTypes.RAW, (20,15,17))
        chunk_request.add_volume_request(VolumeTypes.GT_LABELS, (10,5,7))

        full_request = BatchRequest({
                VolumeTypes.RAW: raw_roi,
                VolumeTypes.GT_LABELS: labels_roi
            }
        )

        pipeline = ChunkTestSource() + Chunk(chunk_request)

        with build(pipeline):
            batch = pipeline.request_batch(full_request)

        # assert that pixels encode their position
        for (volume_type, volume) in batch.volumes.items():

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                    range(int(volume.roi.get_begin()[0]), int(volume.roi.get_end()[0])),
                    range(int(volume.roi.get_begin()[1]), int(volume.roi.get_end()[1])),
                    range(int(volume.roi.get_begin()[2]), int(volume.roi.get_end()[2])), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            self.assertTrue((volume.data == data).all())

        assert(batch.volumes[VolumeTypes.RAW].roi.get_offset() == (1000, 1000, 1000))
