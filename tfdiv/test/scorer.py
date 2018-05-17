from unittest import TestCase

import numpy as np
from scipy.sparse import csr_matrix

from tfdiv.fm import Ranking
from tfdiv.metrics import mean_average_precision
from tfdiv.scorer import RankingScorer


class MockEstimator(Ranking):

    def init_computational_graph(self, *args, **kwargs):
        pass

    def init_dataset(self, *args, **kwargs):
        pass

    def init_input(self, *args, **kwargs):
        pass

    def fit(self, *args):
        pass

    def predict(self, X, n_users, n_items, k=10):
        rank = np.tile(np.arange(min(n_items, k)), n_users) \
            .reshape((n_users, n_items))
        return None, rank


class TestRankingScore(TestCase):

    def test_call_map(self):

        items = np.zeros((3, 5), dtype=np.int32)
        for i, e in enumerate(range(2, 5)):
            items[i, e] = 1
        scorer = RankingScorer(mean_average_precision, 1, {}, csr_matrix(items))

        estimator = MockEstimator()

        x = [[1, 0, 0, 0, 1],
             [1, 0, 0, 1, 0],
             [0, 1, 0, 1, 0],
             [0, 1, 1, 0, 0],
             [0, 1, 0, 0, 1]]
        x = csr_matrix(x)
        a = scorer(estimator, x, None)
        self.assertEqual(a, ((0.5+2/3)/2+1)/2)