class BatchSpec:
    '''A possibly partial specification of a batch.
    '''

    def __init__(self, shape, offset=None, source=None, with_gt=False, with_gt_mask=False):
        self.shape = shape
        self.offset = offset
        self.source = source
        self.with_gt = with_gt
        self.with_gt_mask = with_gt_mask

    def get_offset(self):

        if self.offset is None:
            return (0,)*len(self.shape)

        return self.offset

    def get_bounding_box(self):
        offset = self.get_offset()
        return tuple(
                slice(offset[d], self.shape[d] + offset[d])
                for d in range(len(self.shape))
        )