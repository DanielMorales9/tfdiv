from scipy.sparse import coo_matrix, isspmatrix_csr
from sklearn.metrics.scorer import _BaseScorer
from tfdiv.metrics import mean_average_precision
import pandas as pd
import numpy as np


# TODO - This need to be re-thought
# TODO 1) Test set creation
# TODO 2) Item Catalog with Item Features
class MAPScorer(_BaseScorer):

    def __init__(self, ground_truth, y=None, k=10):
        super(MAPScorer, self).__init__(mean_average_precision, 1, None)
        assert isspmatrix_csr(ground_truth), 'matrix must be of type csr'
        coo = ground_truth.tocoo()
        n_samples, self.n_features = coo.shape
        if y is None:
            self.y = np.ones(n_samples, dtype=np.int32)
        else:
            assert set(y) == {0, 1}, "Input labels must be in set {0,1}."
            self.y = y
        self.k = k
        self.inter_mat = self.get_interactions(coo)
        self.item_cat = self.get_item_catalog()
        self.user_base = self.get_user_base()
        self.inter_mat = self.inter_mat.values
        self.n_items = self.item_cat.shape[0]
        self.n_users = self.user_base.shape[0]
        self.rel_mat = self.create_relevance_matrix()

    def get_user_base(self):
        return self.inter_mat.user.unique()

    def get_item_catalog(self):
        return self.inter_mat.item.unique()

    def create_relevance_matrix(self):
        u_idx = self.inter_mat[:, 0]
        i_idx = self.inter_mat[:, 1] - self.item_cat[0]
        rel_mat = np.zeros((self.n_users, self.n_items),
                           dtype=np.int32)

        rel_mat[u_idx, i_idx] = self.y
        return rel_mat

    @staticmethod
    def get_interactions(coo):
        row = coo.row
        col = coo.col
        mat = pd.DataFrame()
        mat['row'] = row
        mat['col'] = col
        cols = mat.groupby(row)['col']
        items = cols.max()
        users = cols.min()
        inter_mat = pd.DataFrame()
        inter_mat['user'] = users
        inter_mat['item'] = items
        return inter_mat

    def create_test_set(self, X):
        # TODO - This is probably wrong
        coo = X.tocoo()
        row = coo.row
        col = coo.col
        mat = pd.DataFrame()
        mat['row'] = row
        mat['col'] = col
        user = mat.groupby(row)['col'].min()
        users = user.unique()
        users = np.sort(users)
        n_users = users.shape[0]
        user_idx = np.repeat(np.arange(n_users), self.n_items)
        item_idx = np.tile(self.item_cat, n_users)
        row = np.concatenate((np.arange(n_users * self.n_items),
                              np.arange(n_users * self.n_items)))
        col = np.concatenate((user_idx, item_idx))
        data = np.ones(row.shape[0])
        coo = coo_matrix((data, (row, col)), shape=(n_users * self.n_items,
                                                    self.n_features))
        X = coo.tocsr()
        return X, users, n_users

    def __call__(self, estimator, X, y, sample_weight=None):
        X, users, n_users = self.create_test_set(X)
        pred = estimator.predict(X).reshape(n_users, self.n_items)
        ranking = np.argsort(-pred)[:, :self.k]

        ranking_rel = np.zeros((n_users, self.k), dtype=np.int32)

        for i, (u, user_rank) in enumerate(zip(users, ranking)):
            ranking_rel[i, :] = self.rel_mat[u, user_rank]

        return self._score_func(ranking_rel)
