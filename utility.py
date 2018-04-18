import numpy as np
import pandas as pd
import tensorflow as tf


def sparse_repr(x, ntype):
    coo = x.tocoo()
    row = coo.row[:, np.newaxis]
    col = coo.col[:, np.newaxis]
    indices = np.hstack((row, col)).astype(np.int64)
    values = coo.data.astype(ntype)
    shape = np.array(coo.shape).astype(np.int64)
    return indices, values, shape


def cartesian_product(a, b):
    tile_a = tf.tile(a, [b.get_shape()[0]])
    repeat_a = tf.contrib.framework.sort(tile_a)
    tile_b = tf.tile(b, [tf.shape(a)[0]])
    trans = tf.transpose([repeat_a, tile_b])
    cp = tf.contrib.framework.sort(trans, axis=0)
    return cp


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def unique_sparse_matrix(pos):
    def hash_csr(x):
        n_rows = x.shape[0]
        indices = x.indices
        indptr = x.indptr
        data = x.data
        size = 0
        for i in range(n_rows):
            size = max(indices[indptr[i]:indptr[i + 1]].shape[0], size)
        hashed_rows = np.zeros((n_rows, 2), dtype=('S' + str(size * 2 + 4)))
        for i in range(n_rows):
            cols = indices[indptr[i]:indptr[i + 1]]
            dat = data[indptr[i]:indptr[i + 1]]
            hashed_rows[i, 0] = cols.tobytes()
            hashed_rows[i, 1] = dat.tobytes()
        return hashed_rows

    hashed_rows = hash_csr(pos)
    _, row_indices = np.unique(hashed_rows, axis=0,
                               return_index=True)
    x = pos[row_indices]
    return x

# Predefined loss functions
# Should take 2 tf.Ops: outputs, targets and should return tf.Op of element-wise losses
# Be careful about dimensionality -- maybe tf.transpose(outputs) is needed


def loss_logistic(outputs, y):
    margins = -y * tf.transpose(outputs)
    raw_loss = tf.log(tf.add(1.0, tf.exp(margins)))
    return tf.minimum(raw_loss, 100, name='truncated_log_loss')


def num_of_users_from_indices(indices):
    idx = pd.DataFrame(indices, columns=['row', 'col'])
    users = idx.groupby('row', as_index=False).min().col.unique()
    n_users = np.max(users) + 1
    return n_users
