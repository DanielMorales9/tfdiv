import numpy as np
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

# Predefined loss functions
# Should take 2 tf.Ops: outputs, targets and should return tf.Op of element-wise losses
# Be careful about dimensionality -- maybe tf.transpose(outputs) is needed

def loss_logistic(outputs, y):
    margins = -y * tf.transpose(outputs)
    raw_loss = tf.log(tf.add(1.0, tf.exp(margins)))
    return tf.minimum(raw_loss, 100, name='truncated_log_loss')