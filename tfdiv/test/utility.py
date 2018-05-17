from unittest import TestCase

import numpy as np
from scipy.sparse import csr_matrix

from tfdiv.utility import csr_cartesian_product, relevance_feedback, csr_unique_min_cols


class TestUtility(TestCase):

    def test_csr_cartesian_product(self):
        items = np.zeros((3, 5), dtype=np.int32)
        for i, e in enumerate(range(2, 5)):
            items[i, e] = 1
        cat = csr_cartesian_product(np.array([0, 1]), csr_matrix(items)).A

        user0 = np.zeros((3, 5), dtype=np.int32)
        user0[:, 0] = 1
        user1 = np.zeros((3, 5), dtype=np.int32)
        user1[:, 1] = 1
        expected = np.concatenate([items+user0, items+user1])
        self.assertTrue(np.all(cat == expected))

    def test_relevance_feedback(self):

        x = [[1, 0, 1, 0, 0],
             [1, 0, 0, 1, 0],
             [0, 1, 0, 1, 0],
             [0, 1, 1, 0, 0],
             [0, 1, 0, 0, 1]]
        x = csr_matrix(x)
        actual = relevance_feedback(2, 3,
                                    {2: 0,
                                     3: 1,
                                     4: 2}, x)
        expected = [[1, 1, 0],
                    [1, 1, 1]]
        expected = np.array(expected)
        self.assertTrue(np.all(expected == actual))

    def test_csr_unique_min_cols(self):
        x = [[1, 0, 1, 0, 0],
             [1, 0, 0, 1, 0],
             [0, 1, 0, 1, 0],
             [0, 1, 1, 0, 0],
             [0, 1, 0, 0, 1]]
        x = csr_matrix(x)

        col = csr_unique_min_cols(x)

        self.assertTrue(np.all(col == [0, 1]))