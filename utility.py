import numpy as np


def sparse_repr(x, ntype):
    coo = x.tocoo()
    row = coo.row[:, np.newaxis]
    col = coo.col[:, np.newaxis]
    indices = np.hstack((row, col)).astype(np.int64)
    values = coo.data.astype(ntype)
    shape = np.array(coo.shape).astype(np.int64)
    return indices, values, shape
