from scipy.sparse import isspmatrix_csr, csr_matrix, bmat
from collections import defaultdict
from itertools import count
import tensorflow as tf
import pandas as pd
import numpy as np


def sparse_repr(x, ntype):
    coo = x.tocoo()
    row = coo.row[:, np.newaxis]
    col = coo.col[:, np.newaxis]
    indices = np.hstack((row, col)).astype(np.int64)
    values = coo.data.astype(ntype)
    shape = np.array(coo.shape).astype(np.int64)
    return indices, values, shape


def cartesian_product(x, y):
    return np.transpose([np.repeat(x, len(y)), np.tile(y, len(x))])


def matrix_swap_at_k(index, k, matrix):
    for r, c in enumerate(index):
        temp = matrix[r, c + k]
        matrix[r, c + k] = matrix[r, k]
        matrix[r, k] = temp


def csr_unique_min_cols(X):
    cols = []
    for i in range(X.shape[0]):
        cols.append(min(X.indices[X.indptr[i]:X.indptr[i + 1]]))
    cols = np.unique(cols)
    return cols


def ranked_relevance_feedback(rank, rel_feed):
    rs = np.empty(rank.shape)
    for i, (re, ra) in enumerate(zip(rel_feed, rank)):
        rs[i, :] = re[ra]
    return rs


def csr_cartesian_product(users, items):
    n_items, n_features = items.shape
    n_users = users.shape[0]
    n_cartesian = n_items * n_users
    data = np.ones(n_cartesian)
    rows = np.arange(n_cartesian)
    cols = users.repeat(n_items)
    users = csr_matrix((data, (rows, cols)), shape=(n_cartesian, n_features))
    sp = []
    for _ in range(n_users):
        sp.append(items)
    items = bmat(np.array(sp).reshape(-1, 1))
    new_x = items + users
    return new_x


def relevance_feedback(n_users, tot_n_items, tot_n_users, X):
    ui = []
    for i in np.arange(X.shape[0]):
        ui.append(X.indices[X.indptr[i]:X.indptr[i + 1]][:2])
    ui = np.unique(ui, axis=1)

    ui[:, 1] -= tot_n_users
    c = count(0)
    dic = defaultdict(c.__next__)
    for i, u in enumerate(ui[:, 0]):
        ui[i, 0] = dic[u]

    rel = np.zeros((n_users, tot_n_items), dtype=np.int32)
    rel[ui[:, 0], ui[:, 1]] = 1
    return rel


def chunks(L, n):
    """ Yield successive n-sized chunks from L.
    """
    for i in np.arange(0, L.shape[0], n):
        yield L[i:i+n]


def category_mapper(category):
    c = count(0)
    cat_map = defaultdict(c.__next__)
    for i in category:
        cat_map[i]
    return cat_map


# ---- Tensorflow utility ----
def tf_cartesian_product(a, b):
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

def binary_cross_entropy(outputs, y):
    p = tf.sigmoid(outputs)
    return - (y * tf.log(p) + (1 - y) * tf.log(1 - p))


def num_of_users_from_indices(indices):
    idx = pd.DataFrame(indices, columns=['row', 'col'])
    users = idx.groupby('row', as_index=False).min().col.unique()
    n_users = np.max(users) + 1
    return n_users


cond = lambda x, y: y if x is None else x


def relevance_judgements(n_users, n_items, data):
    user_map = category_mapper(np.sort(data.user.unique()))
    item_map = category_mapper(np.sort(data.item.unique()))
    ui = data.values[:, :-2]
    for i in range(ui.shape[0]):
        ui[i, 0] = user_map[ui[i, 0]]
        ui[i, 1] = item_map[ui[i, 1]]
    rel_jud = np.zeros((n_users, n_items), dtype=np.int32)
    rel_jud[ui[:, 0], ui[:, 1]] = 1
    return rel_jud
