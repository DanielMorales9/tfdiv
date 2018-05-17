from tfdiv.utility import csr_unique_min_cols, \
    relevance_feedback, csr_cartesian_product, ranked_relevance_feedback
from scipy.sparse import isspmatrix_csr
from sklearn.metrics.scorer import _BaseScorer
from collections import defaultdict
from itertools import count
from tfdiv.fm import Ranking
import warnings


class RankingScorer(_BaseScorer):

    def __init__(self,
                 score_func,
                 sign,
                 kwargs,
                 item_catalogue,
                 k=10):
        super(RankingScorer, self).__init__(score_func, sign, kwargs)
        assert isspmatrix_csr(item_catalogue), "item_catalogue must be of " \
                                               "class scipy.sparse.csr_matrix"
        self.items = item_catalogue
        self.n_items = self.items.shape[0]
        self.k = k

        items = csr_unique_min_cols(item_catalogue)
        c = count(0)
        self.item_map = defaultdict(c.__next__)
        for i in items:
            var = self.item_map[i]

    def __call__(self, estimator, X, y, sample_weight=None):
        assert issubclass(estimator.__class__, Ranking), "Estimator must be of class Ranking"
        assert isspmatrix_csr(X), "X must be a sparse csr matrix"
        if not X.has_sorted_indices:
            X.sort_indices()
            warnings.warn("scipy.sparse.csr_matrix X not in canonical format")
        users = csr_unique_min_cols(X)
        n_users = users.shape[0]
        new_x = csr_cartesian_product(users, self.items)
        _, rank = estimator.predict(new_x, n_users, self.n_items, self.k)
        rel_feed = relevance_feedback(n_users, self.n_items, self.item_map, X)
        rs = ranked_relevance_feedback(rank, rel_feed)
        return self._sign * self._score_func(rs, **self._kwargs)

