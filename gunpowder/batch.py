class Batch:
    '''Contains the requested batch.
    '''

    def __init__(self, batch_spec):
        self.spec = batch_spec
        self.raw = None
        self.gt = None
        self.gt_mask = None