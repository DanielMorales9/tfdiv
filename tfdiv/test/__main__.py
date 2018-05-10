from ..graph import PointwiseGraph, \
    PointwiseLFPGraph as PointLFP, \
    BPRLFPGraph
from ..fm import FMRegression
from sklearn.preprocessing import OneHotEncoder
from tfdiv.utility import sparse_repr
from unittest.case import TestCase
import tensorflow as tf
import numpy as np
import unittest


# ---- Classifiers Tests ----
class TestFMRegression(TestCase):
    pass


# ---- Computational Graph Tests ----
class TestPointwiseGraph(TestCase):

    def setUp(self):
        self.graph = PointwiseGraph()

        indices = tf.convert_to_tensor(np.array([[0, 0], [1, 1]]), dtype=tf.int64)
        values = tf.convert_to_tensor(np.array([1, 1]), dtype=tf.float32)
        dense_shape = tf.convert_to_tensor([2, 2], dtype=tf.int64)
        self.n_features = 2

        self.x = tf.SparseTensor(indices=indices,
                                 values=values,
                                 dense_shape=dense_shape)
        self.y = tf.convert_to_tensor([1, 1])

        self.bias = tf.verify_tensor_all_finite(
            tf.Variable(0.0,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        weights = tf.convert_to_tensor(np.arange(self.n_features),
                                       dtype=tf.float32)
        self.weights = tf.verify_tensor_all_finite(
            tf.Variable(weights,
                        trainable=True,
                        name='weights',
                        dtype=tf.float32),
            msg='NaN or Inf in weights')

        params = np.repeat(np.arange(2, dtype=np.float32).reshape(-1, 1), 10, axis=1)
        self.params = tf.verify_tensor_all_finite(
            tf.Variable(params,
                        trainable=True,
                        name='params', dtype=tf.float32),
            msg='NaN or Inf in parameters')

        self.graph.set_params(**{'x': self.x,
                                 'y': self.y,
                                 'bias': self.bias,
                                 'weights': self.weights,
                                 'params': self.params
                                 })
        self.graph.define_graph()

    def test_set_params(self):
        self.assertEqual(self.graph.x, self.x)
        self.assertEqual(self.graph.y, self.y)

    def test_predictions(self):
        with tf.Session() as sess:
            sess.run(self.graph.init_all_vars,
                     feed_dict={self.graph.n_features: self.n_features})
            predictions = sess.run(self.graph.y_hat,
                                   feed_dict={self.graph.n_features:
                                                  self.n_features})
        expected = np.array([[0], [1]])
        self.assertTrue(np.all(expected == predictions))

    def tearDown(self):
        tf.reset_default_graph()


class TestPointwiseLFPGraph(TestCase):

    def setUp(self):
        self.graph = PointLFP()

    def test_predictions(self):
        indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
        values = np.array([1, 1], dtype=np.float32)
        dense_shape = np.array([2, 2], dtype=np.int64)
        n_features = 2

        y = np.array([1, 1], dtype=np.float32)

        self.bias = tf.verify_tensor_all_finite(
            tf.Variable(0.0,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        weights = tf.convert_to_tensor(np.arange(n_features), dtype=tf.float32)
        self.weights = tf.verify_tensor_all_finite(
            tf.Variable(weights,
                        trainable=True,
                        name='weights',
                        dtype=tf.float32),
            msg='NaN or Inf in weights')

        params = np.repeat(np.arange(2, dtype=np.float32).reshape(-1, 1), 10, axis=1)
        self.params = tf.verify_tensor_all_finite(
            tf.Variable(params,
                        trainable=True,
                        name='params', dtype=tf.float32),
            msg='NaN or Inf in parameters')

        self.graph.set_params(**{'bias': self.bias,
                                 'weights': self.weights,
                                 'params': self.params
                                 })
        self.graph.define_graph()
        with tf.Session() as sess:
            sess.run(self.graph.init_all_vars,
                     feed_dict={self.graph.n_features: n_features})
            predictions = sess.run(self.graph.y_hat,
                                   feed_dict={self.graph.x: (indices,
                                                             values,
                                                             dense_shape),
                                              self.graph.y: y})
        expected = np.array([[0], [1]])
        self.assertTrue(np.all(expected == predictions))

    def test_variance_estimate_first(self):
        n_users = 3
        x = np.array([[100, 200],
                      [101, 201],
                      [100, 201],
                      [101, 200],
                      [102, 200],
                      [102, 201]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x)
        indices, values, shape = sparse_repr(new_x, np.float32)
        n_features = 5
        params = np.repeat(np.arange(n_features), 10).reshape((5, 10))
        params = tf.convert_to_tensor(params, dtype=tf.float32)
        params = tf.verify_tensor_all_finite(tf.Variable(params,
                                                         trainable=True,
                                                         name='params', dtype=tf.float32),
                                             msg='NaN or Inf in parameters')
        self.graph.set_params(**{'params': params})
        self.graph.define_graph()
        self.graph.variance_estimate()

        sess = tf.Session()
        sess.run((self.graph.init_variance_vars,
                  self.graph.init_all_vars),
                 feed_dict={self.graph.n_users: n_users,
                            self.graph.n_features: n_features})
        actual_var = sess.run(self.graph.variance,
                              feed_dict={self.graph.x: (indices, values, shape),
                                         self.graph.y: np.array([0, 1, 0, 1, 0, 1],
                                                                dtype=np.float32),
                                         self.graph.n_users: n_users})

        self.assertEqual(actual_var.shape, (3, 10))
        expected_var = np.repeat(np.array([12.5, 6.5, 2.5],
                                          dtype=np.float32), 10) \
            .reshape((3, 10))
        self.assertTrue(np.all(expected_var == actual_var))

    def test_variance_estimate_second(self):
        n_users = 3
        n_features = 5
        n_factors = 10
        x = np.array([[100, 200],
                      [100, 201],
                      [101, 200],
                      [102, 200],
                      [102, 201]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x)
        indices, values, shape = sparse_repr(new_x, np.float32)
        params = np.repeat(np.arange(n_features), 10).reshape((n_features, n_factors))
        params = tf.convert_to_tensor(params, dtype=tf.float32)
        params = tf.verify_tensor_all_finite(tf.Variable(params,
                                                         trainable=True,
                                                         name='params', dtype=tf.float32),
                                             msg='NaN or Inf in parameters')
        self.graph.set_params(**{'params': params})
        self.graph.define_graph()
        self.graph.variance_estimate()

        sess = tf.Session()
        sess.run((self.graph.init_variance_vars,
                  self.graph.init_all_vars),
                 feed_dict={self.graph.n_users: n_users,
                            self.graph.n_features: n_features})
        actual_var = sess.run(self.graph.variance,
                              feed_dict={self.graph.x: (indices, values, shape),
                                         self.graph.y: np.array([0, 1, 0, 1, 0, 1],
                                                                dtype=np.float32),
                                         self.graph.n_users: n_users})

        self.assertEqual(actual_var.shape, (3, 10))
        expected_var = np.repeat(np.array([12.5, 4, 2.5],
                                          dtype=np.float32), 10) \
            .reshape((3, 10))
        self.assertTrue(np.all(expected_var == actual_var))

    def test_variance_estimate_third(self):
        n_users = 3
        n_features = 5
        n_factors = 10
        x = np.array([[100, 200],
                      [101, 201],
                      [100, 201],
                      [101, 200],
                      [102, 200],
                      [102, 201]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x)
        indices, values, shape = sparse_repr(new_x, np.float32)
        params = np.repeat(np.arange(n_features), 10).reshape((n_features,
                                                               n_factors))
        params = tf.convert_to_tensor(params, dtype=tf.float32)
        params = tf.verify_tensor_all_finite(tf.Variable(params,
                                                         trainable=True,
                                                         name='params',
                                                         dtype=tf.float32),
                                             msg='NaN or Inf in parameters')
        self.graph.set_params(**{'params': params})
        self.graph.define_graph()
        self.graph.variance_estimate()

        sess = tf.Session()
        sess.run((self.graph.init_variance_vars,
                  self.graph.init_all_vars),
                 feed_dict={self.graph.n_users: n_users,
                            self.graph.n_features: n_features})
        actual_var = sess.run(self.graph.variance,
                              feed_dict={self.graph.x: (indices, values, shape),
                                         self.graph.y: np.array([0, 1, 0, 1, 0, 1],
                                                                dtype=np.float32),
                                         self.graph.n_users: n_users})

        self.assertEqual(actual_var.shape, (3, 10))
        expected_var = np.repeat(np.array([12.5, 6.5, 2.5],
                                          dtype=np.float32), 10) \
            .reshape((3, 10))
        self.assertTrue(np.all(expected_var == actual_var))

        x = np.array([[100, 200],
                      [100, 201],
                      [101, 200],
                      [102, 200],
                      [102, 201]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x)
        indices, values, shape = sparse_repr(new_x, np.float32)
        sess.run(self.graph.init_variance_vars, feed_dict={
            self.graph.n_users: n_users
        })
        actual_var = sess.run(self.graph.variance,
                              feed_dict={self.graph.x: (indices, values, shape),
                                         self.graph.y: np.array([0, 1, 0, 1, 0, 1],
                                                                dtype=np.float32),

                                         self.graph.n_users: n_users})

        self.assertEqual(actual_var.shape, (3, 10))
        expected_var = np.repeat(np.array([12.5, 4, 2.5],
                                          dtype=np.float32), 10) \
            .reshape((3, 10))
        self.assertTrue(np.all(expected_var == actual_var))

    def test_ranking_computation(self):
        n_features = 5
        # cartesian product of users and items
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)

        self.bias = tf.verify_tensor_all_finite(
            tf.Variable(0.0,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        weights = tf.convert_to_tensor(np.arange(n_features),
                                       dtype=tf.float32)
        self.weights = tf.verify_tensor_all_finite(
            tf.Variable(weights,
                        trainable=True,
                        name='weights',
                        dtype=tf.float32),
            msg='NaN or Inf in weights')

        params = np.repeat(np.arange(n_features,
                                     dtype=np.float32)
                           .reshape(-1, 1), 10, axis=1)
        self.params = tf.verify_tensor_all_finite(
            tf.Variable(params,
                        trainable=True,
                        name='params', dtype=tf.float32),
            msg='NaN or Inf in parameters')

        self.graph.set_params(**{'bias': self.bias,
                                 'weights': self.weights,
                                 'params': self.params
                                 })
        self.graph.define_graph()
        y = np.array([0, 1, 0, 1, 0, 1],
                     dtype=np.float32)

        sess = tf.Session()
        sess.run(self.graph.init_all_vars,
                 feed_dict={self.graph.n_features: n_features})
        res = sess.run(self.graph.y_hat,
                       feed_dict={self.graph.x: sparse_x,
                                  self.graph.y: y})
        expected_res = np.array([2., 3., 4., 23., 34., 45.],
                                dtype=np.float32).reshape(-1, 1)
        self.assertTrue(np.all(res == expected_res))

        n_users = 2
        n_items = 3
        self.graph.ranking_computation()
        sorted_pred, ranking = sess.run(self.graph.ranking_results,
                                        feed_dict={self.graph.x: sparse_x,
                                                   self.graph.y: y,
                                                   self.graph.k: n_items,
                                                   self.graph.n_items: n_items,
                                                   self.graph.n_users: n_users})
        expected_pred = np.array([[4, 3, 2], [45, 34, 23]], dtype=np.float32)
        expected_ranking = np.array([[2, 1, 0], [2, 1, 0]], dtype=np.int32)
        self.assertTrue(np.all(sorted_pred == expected_pred))
        self.assertTrue(np.all(ranking == expected_ranking))

    def test_reshape_dataset(self):
        n_features = 5
        # cartesian product of users and items
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)

        self.bias = tf.verify_tensor_all_finite(
            tf.Variable(0.0,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        weights = tf.convert_to_tensor(np.arange(n_features),
                                       dtype=tf.float32)
        self.weights = tf.verify_tensor_all_finite(
            tf.Variable(weights,
                        trainable=True,
                        name='weights',
                        dtype=tf.float32),
            msg='NaN or Inf in weights')

        params = np.repeat(np.arange(n_features,
                                     dtype=np.float32)
                           .reshape(-1, 1), 10, axis=1)
        self.params = tf.verify_tensor_all_finite(
            tf.Variable(params,
                        trainable=True,
                        name='params', dtype=tf.float32),
            msg='NaN or Inf in parameters')

        self.graph.set_params(**{'bias': self.bias,
                                 'weights': self.weights,
                                 'params': self.params
                                 })
        self.graph.define_graph()

        sess = tf.Session()
        sess.run(self.graph.init_all_vars,
                 feed_dict={self.graph.n_features: n_features})

        n_users = 2
        ranking = np.array([[2, 1, 0], [2, 1, 0]], dtype=np.int32)

        zeroed_x, users_x = self.graph.reshape_dataset(self.graph.n_users,
                                                       self.graph.rankings,
                                                       self.graph.x)
        np_zeroed_x = sess.run(tf.sparse_tensor_to_dense(zeroed_x),
                               feed_dict={self.graph.x: sparse_x,
                                          self.graph.n_users: n_users,
                                          self.graph.rankings: ranking})
        np_users_x = sess.run(tf.sparse_tensor_to_dense(users_x),
                              feed_dict={self.graph.x: sparse_x,
                                         self.graph.n_users: n_users,
                                         self.graph.rankings: ranking})
        expected_dataset = np.array([[0, 0, 0, 0, 1],
                                     [0, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 1],
                                     [0, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0]],
                                    dtype=np.float32).reshape(2, 3, 5)
        self.assertTrue(np.all(np_zeroed_x == expected_dataset))
        expected_users = np.array([[1, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0],
                                   [0, 1, 0, 0, 0],
                                   [0, 1, 0, 0, 0]],
                                  dtype=np.float32).reshape(2, 3, 5)
        self.assertTrue(np.all(np_users_x == expected_users))

    def test_variance_lookup(self):
        n_features = 5
        n_users = 2
        # cartesian product of users and items
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)

        self.bias = tf.verify_tensor_all_finite(
            tf.Variable(0.0,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        weights = tf.convert_to_tensor(np.arange(n_features),
                                       dtype=tf.float32)
        self.weights = tf.verify_tensor_all_finite(
            tf.Variable(weights,
                        trainable=True,
                        name='weights',
                        dtype=tf.float32),
            msg='NaN or Inf in weights')

        params = np.repeat(np.arange(n_features,
                                     dtype=np.float32)
                           .reshape(-1, 1), 10, axis=1)
        self.params = tf.verify_tensor_all_finite(
            tf.Variable(params,
                        trainable=True,
                        name='params', dtype=tf.float32),
            msg='NaN or Inf in parameters')

        self.graph.set_params(**{'bias': self.bias,
                                 'weights': self.weights,
                                 'params': self.params
                                 })
        self.graph.define_graph()
        self.graph.variance_estimate()
        _, users_x = self.graph.reshape_dataset(self.graph.n_users,
                                                self.graph.rankings,
                                                self.graph.x)
        user_idx = users_x.indices[:1, 2]
        lookup_var = PointLFP.variance_lookup(user_idx, self.graph.variance)

        sess = tf.Session()
        sess.run((self.graph.init_all_vars,
                  self.graph.init_variance_vars), feed_dict={
            self.graph.n_users: n_users,
            self.graph.n_features: n_features
        })

        ranking = np.array([[2, 1, 0], [2, 1, 0]], dtype=np.int32)

        variance = sess.run(self.graph.variance,
                            feed_dict={self.graph.x: sparse_x,
                                       self.graph.n_users: n_users,
                                       # self.graph.rankings: ranking}
                                       })
        expected_variance = np.repeat(np.array([29 / 3, 14 / 3],
                                               dtype=np.float32), 10) \
            .reshape(n_users, 10)
        self.assertTrue(np.all(expected_variance == variance))

        user_var = sess.run(lookup_var,
                            feed_dict={self.graph.x: sparse_x,
                                       self.graph.n_users: n_users,
                                       self.graph.rankings: ranking
                                       })
        expected_user_var = np.repeat(np.array([29 / 3],
                                               dtype=np.float32), 10) \
            .reshape(1, 10)
        self.assertTrue(np.all(user_var == expected_user_var))

    def test_ranking_coefficient(self):
        n_features = 5
        # cartesian product of users and items
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)

        self.graph.define_graph()
        pk = PointLFP.ranking_coefficient(self.graph.k)

        sess = tf.Session()
        actual_pk = sess.run(pk, feed_dict={self.graph.x: sparse_x,
                                            self.graph.k: 1})
        expected_pk = np.array([0.5], dtype=np.float32)
        self.assertTrue(np.all(expected_pk == actual_pk))
        actual_pk = sess.run(pk, feed_dict={self.graph.x: sparse_x,
                                            self.graph.k: 2})
        expected_pk = np.array([0.25], dtype=np.float32)
        self.assertTrue(np.all(expected_pk == actual_pk))

    def test_dot_product(self):
        n_features = 5
        n_users = 2
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)
        bias = tf.verify_tensor_all_finite(
            tf.Variable(0.0,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        weights = tf.convert_to_tensor(np.arange(n_features),
                                       dtype=tf.float32)
        weights = tf.verify_tensor_all_finite(
            tf.Variable(weights,
                        trainable=True,
                        name='weights',
                        dtype=tf.float32),
            msg='NaN or Inf in weights')

        params = np.repeat(np.arange(n_features,
                                     dtype=np.float32)
                           .reshape(-1, 1), 10, axis=1)
        params = tf.verify_tensor_all_finite(
            tf.Variable(params,
                        trainable=True,
                        name='params', dtype=tf.float32),
            msg='NaN or Inf in parameters')

        self.graph.set_params(**{'bias': bias,
                                 'weights': weights,
                                 'params': params})
        self.graph.define_graph()
        zeroed_x, _ = self.graph.reshape_dataset(self.graph.n_users,
                                                 self.graph.rankings,
                                                 self.graph.x)
        dot = self.graph.dot_product(self.graph.params,
                                     self.graph.rankings,
                                     zeroed_x)

        rankings = np.array([[2, 1, 0],
                             [2, 0, 1]], dtype=np.float32)
        sess = tf.Session()
        sess.run(self.graph.init_all_vars,
                 feed_dict={self.graph.n_features: n_features})
        actual_dot = sess.run(dot, feed_dict={self.graph.x: sparse_x,
                                              self.graph.rankings: rankings,
                                              self.graph.n_users: n_users})
        expected_dot = np.array([[4, 3, 2], [4, 2, 3]], dtype=np.float32)
        expected_dot = np.repeat(expected_dot, 10).reshape(2, 3, 10)
        self.assertTrue(np.all(expected_dot == actual_dot))

    def test_second_term(self):
        n_features = 5
        n_users = 2
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)
        bias = tf.verify_tensor_all_finite(
            tf.Variable(0.0,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        weights = tf.convert_to_tensor(np.arange(n_features),
                                       dtype=tf.float32)
        weights = tf.verify_tensor_all_finite(
            tf.Variable(weights,
                        trainable=True,
                        name='weights',
                        dtype=tf.float32),
            msg='NaN or Inf in weights')

        params = np.repeat(np.arange(n_features,
                                     dtype=np.float32)
                           .reshape(-1, 1), 10, axis=1)
        params = tf.verify_tensor_all_finite(
            tf.Variable(params,
                        trainable=True,
                        name='params', dtype=tf.float32),
            msg='NaN or Inf in parameters')

        self.graph.set_params(**{'bias': bias,
                                 'weights': weights,
                                 'params': params})
        self.graph.define_graph()
        self.graph.variance_estimate()
        zeroed_x, _ = self.graph.reshape_dataset(self.graph.n_users,
                                                 self.graph.rankings,
                                                 self.graph.x)
        dot = self.graph.dot_product(self.graph.params,
                                     self.graph.rankings,
                                     zeroed_x)
        self.graph.pk = self.graph.ranking_coefficient(self.graph.k)
        snd = self.graph.second_term(self.graph.k,
                                     self.graph.pk,
                                     self.graph.variance,
                                     dot)

        rankings = np.array([[2, 1, 0],
                             [2, 0, 1]], dtype=np.float32)
        sess = tf.Session()
        sess.run((self.graph.init_all_vars,
                  self.graph.init_variance_vars),
                 feed_dict={self.graph.n_users: n_users,
                            self.graph.n_features: n_features})
        actual_snd = sess.run(snd, feed_dict={self.graph.x: sparse_x,
                                              self.graph.k: 1,
                                              self.graph.rankings: rankings,
                                              self.graph.n_users: n_users})

        expected_snd = np.array([[29.0 * 15, 29.0 / 3 * 20],
                                 [14.0 / 3 * 20, 14.0 * 15]],
                                dtype=np.float32)
        self.assertTrue(np.allclose(actual_snd, expected_snd))
        actual_snd = sess.run(snd, feed_dict={self.graph.x: sparse_x,
                                              self.graph.k: 0,
                                              self.graph.rankings: rankings,
                                              self.graph.n_users: n_users})

        expected_snd = np.array([[29.0 / 3 * 10 * 16, 29.0 * 30, 29.0 / 3 * 40],
                                 [14.0 / 3 * 10 * 16, 14.0 / 3 * 40, 14.0 * 30, ]],
                                dtype=np.float32)
        self.assertTrue(np.allclose(actual_snd, expected_snd))

    def test_ranking_weight(self):
        k = tf.placeholder(shape=[], dtype=tf.int32)

        pm = self.graph.ranking_weights(k)
        sess = tf.Session()
        actual_pm = sess.run(pm, feed_dict={k: 12})
        expected_pm = np.array([1 / (2.0 ** i) for i in range(12)])
        self.assertTrue(np.all(actual_pm == expected_pm))

    def test_third_term(self):
        n_features = 5
        n_users = 2
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)
        bias = tf.verify_tensor_all_finite(
            tf.Variable(0.0,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        weights = tf.convert_to_tensor(np.arange(n_features),
                                       dtype=tf.float32)
        weights = tf.verify_tensor_all_finite(
            tf.Variable(weights,
                        trainable=True,
                        name='weights',
                        dtype=tf.float32),
            msg='NaN or Inf in weights')

        params = np.repeat(np.arange(n_features,
                                     dtype=np.float32)
                           .reshape(-1, 1), 10, axis=1)
        params = tf.verify_tensor_all_finite(
            tf.Variable(params,
                        trainable=True,
                        name='params', dtype=tf.float32),
            msg='NaN or Inf in parameters')

        self.graph.set_params(**{'bias': bias,
                                 'weights': weights,
                                 'params': params})
        self.graph.define_graph()
        self.graph.variance_estimate()
        zeroed_x, _ = self.graph.reshape_dataset(self.graph.n_users,
                                                 self.graph.rankings,
                                                 self.graph.x)
        dot = self.graph.dot_product(self.graph.params,
                                     self.graph.rankings,
                                     zeroed_x)
        trd = self.graph.third_term(self.graph.k,
                                    self.graph.variance,
                                    dot, dtype=self.graph.dtype)

        rankings = np.array([[2, 1, 0],
                             [2, 0, 1]], dtype=np.float32)
        sess = tf.Session()
        sess.run((self.graph.init_all_vars,
                  self.graph.init_variance_vars),
                 feed_dict={self.graph.n_users: n_users,
                            self.graph.n_features: n_features})
        actual_trd = sess.run(trd, feed_dict={self.graph.x: sparse_x,
                                              self.graph.k: 1,
                                              self.graph.rankings: rankings,
                                              self.graph.n_users: n_users})
        expected_trd = np.array([[1160., 8. * 29 / 3 * 10],
                                 [8 * 14 / 3 * 10, 560.]])

        self.assertTrue(np.allclose(actual_trd, expected_trd))

    def test_delta_f(self):
        n_features = 5
        n_users = 2
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)
        bias = tf.verify_tensor_all_finite(
            tf.Variable(0.0,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        weights = tf.convert_to_tensor(np.arange(n_features),
                                       dtype=tf.float32)
        weights = tf.verify_tensor_all_finite(
            tf.Variable(weights,
                        trainable=True,
                        name='weights',
                        dtype=tf.float32),
            msg='NaN or Inf in weights')

        params = np.repeat(np.arange(n_features,
                                     dtype=np.float32)
                           .reshape(-1, 1), 10, axis=1)
        params = tf.verify_tensor_all_finite(
            tf.Variable(params,
                        trainable=True,
                        name='params', dtype=tf.float32),
            msg='NaN or Inf in parameters')

        self.graph.set_params(**{'bias': bias,
                                 'weights': weights,
                                 'params': params})
        self.graph.define_graph()
        self.graph.ranking_computation()
        self.graph.variance_estimate()
        delta_f = self.graph.delta_f_computation()

        sess = tf.Session()
        sess.run((self.graph.init_all_vars,
                  self.graph.init_variance_vars), feed_dict={
            self.graph.n_users: n_users,
            self.graph.n_features: n_features
        })
        pred = np.array([[4, 3, 2], [45, 34, 23]], dtype=np.float32)
        rank = np.array([[2, 1, 0], [2, 1, 0]], dtype=np.int32)

        d = sess.run(delta_f, feed_dict={
            self.graph.x: sparse_x,
            self.graph.predictions: pred,
            self.graph.rankings: rank,
            self.graph.k: 1,
            self.graph.b: 1.0,
            self.graph.n_users: n_users
        })

        snd = np.array([[29.0 * 15, 29.0 / 3 * 20],
                        [14.0 * 15, 14.0 / 3 * 20]])
        trd = np.array([[1160., 8. * 29 / 3 * 10],
                        [560., 8 * 14 / 3 * 10]])

        expected = 0.5 * (pred[:, 1:] - (snd + 2 * trd))

        self.assertTrue(np.allclose(d, expected))

    def test_unique_sparse_tensor(self):
        n_users = 2
        n_items = 3
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [100, 200],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)
        self.graph.define_graph()
        self.graph.unique_rows_sparse_tensor()

        sess = tf.Session()
        actual = sess.run((self.graph.init_unique_vars,
                           tf.sparse_tensor_to_dense(self.graph.unique_x)),
                          feed_dict={self.graph.x: sparse_x,
                                     self.graph.n_features: n_users + n_items,
                                     self.graph.n_users: n_users,
                                     self.graph.n_items: n_items})[1]
        expected = np.array([[1, 0, 1, 0, 0],
                             [1, 0, 0, 1, 0],
                             [1, 0, 0, 0, 1],
                             [0, 1, 1, 0, 0],
                             [0, 1, 0, 1, 0],
                             [0, 1, 0, 0, 1]])
        self.assertTrue(np.all(actual == expected))

    def test_unique_sparse_tensor_multi(self):
        n_users = 2
        n_items = 3
        n_features = 7
        x = np.array([[100, 200, 300],
                      [100, 201, 300],
                      [100, 202, 301],
                      [100, 200, 300],
                      [101, 200, 300],
                      [101, 201, 300],
                      [101, 202, 301]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)
        self.graph.define_graph()
        self.graph.unique_rows_sparse_tensor()

        sess = tf.Session()
        actual = sess.run((self.graph.init_unique_vars,
                           tf.sparse_tensor_to_dense(self.graph.unique_x)),
                          feed_dict={self.graph.x: sparse_x,
                                     self.graph.n_features: n_features,
                                     self.graph.n_users: n_users,
                                     self.graph.n_items: n_items})[1]
        expected = np.array([[1, 0, 1, 0, 0, 1, 0],
                             [1, 0, 0, 1, 0, 1, 0],
                             [1, 0, 0, 0, 1, 0, 1],
                             [0, 1, 1, 0, 0, 1, 0],
                             [0, 1, 0, 1, 0, 1, 0],
                             [0, 1, 0, 0, 1, 0, 1]])
        self.assertTrue(np.all(actual == expected))

    def tearDown(self):
        tf.reset_default_graph()


class TestBayesianLFPGraph(TestCase):

    def setUp(self):
        self.graph = BPRLFPGraph()

    def test_predictions(self):
        indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
        values = np.array([1, 1], dtype=np.float32)
        dense_shape = np.array([2, 2], dtype=np.int64)
        n_features = 2

        y = np.array([1, 1], dtype=np.float32)

        self.bias = tf.verify_tensor_all_finite(
            tf.Variable(0.0,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        weights = tf.convert_to_tensor(np.arange(n_features), dtype=tf.float32)
        self.weights = tf.verify_tensor_all_finite(
            tf.Variable(weights,
                        trainable=True,
                        name='weights',
                        dtype=tf.float32),
            msg='NaN or Inf in weights')

        params = np.repeat(np.arange(2, dtype=np.float32).reshape(-1, 1), 10, axis=1)
        self.params = tf.verify_tensor_all_finite(
            tf.Variable(params,
                        trainable=True,
                        name='params', dtype=tf.float32),
            msg='NaN or Inf in parameters')

        self.graph.set_params(**{'bias': self.bias,
                                 'weights': self.weights,
                                 'params': self.params
                                 })
        self.graph.define_graph()
        with tf.Session() as sess:
            sess.run(self.graph.init_all_vars,
                     feed_dict={self.graph.n_features: n_features})
            predictions = sess.run(self.graph.y_hat,
                                   feed_dict={self.graph.x: (indices,
                                                             values,
                                                             dense_shape)})
        expected = np.array([[0], [1]])
        self.assertTrue(np.all(expected == predictions))

    def test_variance_estimate_first(self):
        n_users = 3
        x = np.array([[100, 200],
                      [101, 201],
                      [100, 201],
                      [101, 200],
                      [102, 200],
                      [102, 201]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x)
        indices, values, shape = sparse_repr(new_x, np.float32)
        n_features = 5
        params = np.repeat(np.arange(n_features), 10).reshape((5, 10))
        params = tf.convert_to_tensor(params, dtype=tf.float32)
        params = tf.verify_tensor_all_finite(tf.Variable(params,
                                                         trainable=True,
                                                         name='params', dtype=tf.float32),
                                             msg='NaN or Inf in parameters')
        self.graph.set_params(**{'params': params})
        self.graph.define_graph()
        self.graph.variance_estimate()

        sess = tf.Session()
        sess.run((self.graph.init_variance_vars,
                  self.graph.init_all_vars),
                 feed_dict={self.graph.n_users: n_users,
                            self.graph.n_features: n_features})
        actual_var = sess.run(self.graph.variance,
                              feed_dict={self.graph.x: (indices, values, shape),
                                         self.graph.n_users: n_users})

        self.assertEqual(actual_var.shape, (3, 10))
        expected_var = np.repeat(np.array([12.5, 6.5, 2.5],
                                          dtype=np.float32), 10) \
            .reshape((3, 10))
        self.assertTrue(np.all(expected_var == actual_var))

    def test_variance_estimate_second(self):
        n_users = 3
        n_features = 5
        n_factors = 10
        x = np.array([[100, 200],
                      [100, 201],
                      [101, 200],
                      [102, 200],
                      [102, 201]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x)
        indices, values, shape = sparse_repr(new_x, np.float32)
        params = np.repeat(np.arange(n_features), 10).reshape((n_features, n_factors))
        params = tf.convert_to_tensor(params, dtype=tf.float32)
        params = tf.verify_tensor_all_finite(tf.Variable(params,
                                                         trainable=True,
                                                         name='params', dtype=tf.float32),
                                             msg='NaN or Inf in parameters')
        self.graph.set_params(**{'params': params})
        self.graph.define_graph()
        self.graph.variance_estimate()

        sess = tf.Session()
        sess.run((self.graph.init_variance_vars,
                  self.graph.init_all_vars),
                 feed_dict={self.graph.n_users: n_users,
                            self.graph.n_features: n_features})
        actual_var = sess.run(self.graph.variance,
                              feed_dict={self.graph.x: (indices, values, shape),
                                         self.graph.n_users: n_users})

        self.assertEqual(actual_var.shape, (3, 10))
        expected_var = np.repeat(np.array([12.5, 4, 2.5],
                                          dtype=np.float32), 10) \
            .reshape((3, 10))
        self.assertTrue(np.all(expected_var == actual_var))

    def test_variance_estimate_third(self):
        n_users = 3
        n_features = 5
        n_factors = 10
        x = np.array([[100, 200],
                      [101, 201],
                      [100, 201],
                      [101, 200],
                      [102, 200],
                      [102, 201]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x)
        indices, values, shape = sparse_repr(new_x, np.float32)
        params = np.repeat(np.arange(n_features), 10).reshape((n_features,
                                                               n_factors))
        params = tf.convert_to_tensor(params, dtype=tf.float32)
        params = tf.verify_tensor_all_finite(tf.Variable(params,
                                                         trainable=True,
                                                         name='params',
                                                         dtype=tf.float32),
                                             msg='NaN or Inf in parameters')
        self.graph.set_params(**{'params': params})
        self.graph.define_graph()
        self.graph.variance_estimate()

        sess = tf.Session()
        sess.run((self.graph.init_variance_vars,
                  self.graph.init_all_vars),
                 feed_dict={self.graph.n_users: n_users,
                            self.graph.n_features: n_features})
        actual_var = sess.run(self.graph.variance,
                              feed_dict={self.graph.x: (indices, values, shape),
                                         self.graph.n_users: n_users})

        self.assertEqual(actual_var.shape, (3, 10))
        expected_var = np.repeat(np.array([12.5, 6.5, 2.5],
                                          dtype=np.float32), 10) \
            .reshape((3, 10))
        self.assertTrue(np.all(expected_var == actual_var))

        x = np.array([[100, 200],
                      [100, 201],
                      [101, 200],
                      [102, 200],
                      [102, 201]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x)
        indices, values, shape = sparse_repr(new_x, np.float32)
        sess.run(self.graph.init_variance_vars, feed_dict={
            self.graph.n_users: n_users
        })
        actual_var = sess.run(self.graph.variance,
                              feed_dict={self.graph.x: (indices, values, shape),
                                         self.graph.n_users: n_users})

        self.assertEqual(actual_var.shape, (3, 10))
        expected_var = np.repeat(np.array([12.5, 4, 2.5],
                                          dtype=np.float32), 10) \
            .reshape((3, 10))
        self.assertTrue(np.all(expected_var == actual_var))

    def test_ranking_computation(self):
        n_features = 5
        # cartesian product of users and items
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)

        self.bias = tf.verify_tensor_all_finite(
            tf.Variable(0.0,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        weights = tf.convert_to_tensor(np.arange(n_features),
                                       dtype=tf.float32)
        self.weights = tf.verify_tensor_all_finite(
            tf.Variable(weights,
                        trainable=True,
                        name='weights',
                        dtype=tf.float32),
            msg='NaN or Inf in weights')

        params = np.repeat(np.arange(n_features,
                                     dtype=np.float32)
                           .reshape(-1, 1), 10, axis=1)
        self.params = tf.verify_tensor_all_finite(
            tf.Variable(params,
                        trainable=True,
                        name='params', dtype=tf.float32),
            msg='NaN or Inf in parameters')

        self.graph.set_params(**{'bias': self.bias,
                                 'weights': self.weights,
                                 'params': self.params
                                 })
        self.graph.define_graph()

        sess = tf.Session()
        sess.run(self.graph.init_all_vars,
                 feed_dict={self.graph.n_features: n_features})
        res = sess.run(self.graph.y_hat,
                       feed_dict={self.graph.x: sparse_x})
        expected_res = np.array([2., 3., 4., 23., 34., 45.],
                                dtype=np.float32).reshape(-1, 1)
        self.assertTrue(np.all(res == expected_res))

        n_users = 2
        n_items = 3
        self.graph.ranking_computation()
        sorted_pred, ranking = sess.run(self.graph.ranking_results,
                                        feed_dict={self.graph.x: sparse_x,
                                                   self.graph.k: n_items,
                                                   self.graph.n_items: n_items,
                                                   self.graph.n_users: n_users})
        expected_pred = np.array([[4, 3, 2], [45, 34, 23]], dtype=np.float32)
        expected_ranking = np.array([[2, 1, 0], [2, 1, 0]], dtype=np.int32)
        self.assertTrue(np.all(sorted_pred == expected_pred))
        self.assertTrue(np.all(ranking == expected_ranking))

    def test_reshape_dataset(self):
        n_features = 5
        # cartesian product of users and items
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)

        self.bias = tf.verify_tensor_all_finite(
            tf.Variable(0.0,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        weights = tf.convert_to_tensor(np.arange(n_features),
                                       dtype=tf.float32)
        self.weights = tf.verify_tensor_all_finite(
            tf.Variable(weights,
                        trainable=True,
                        name='weights',
                        dtype=tf.float32),
            msg='NaN or Inf in weights')

        params = np.repeat(np.arange(n_features,
                                     dtype=np.float32)
                           .reshape(-1, 1), 10, axis=1)
        self.params = tf.verify_tensor_all_finite(
            tf.Variable(params,
                        trainable=True,
                        name='params', dtype=tf.float32),
            msg='NaN or Inf in parameters')

        self.graph.set_params(**{'bias': self.bias,
                                 'weights': self.weights,
                                 'params': self.params
                                 })
        self.graph.define_graph()

        sess = tf.Session()
        sess.run(self.graph.init_all_vars,
                 feed_dict={self.graph.n_features: n_features})

        n_users = 2
        ranking = np.array([[2, 1, 0], [2, 1, 0]], dtype=np.int32)

        zeroed_x, users_x = self.graph.reshape_dataset(self.graph.n_users,
                                                       self.graph.rankings,
                                                       self.graph.x)
        np_zeroed_x = sess.run(tf.sparse_tensor_to_dense(zeroed_x),
                               feed_dict={self.graph.x: sparse_x,
                                          self.graph.n_users: n_users,
                                          self.graph.rankings: ranking})
        np_users_x = sess.run(tf.sparse_tensor_to_dense(users_x),
                              feed_dict={self.graph.x: sparse_x,
                                         self.graph.n_users: n_users,
                                         self.graph.rankings: ranking})
        expected_dataset = np.array([[0, 0, 0, 0, 1],
                                     [0, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 1],
                                     [0, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0]],
                                    dtype=np.float32).reshape(2, 3, 5)
        self.assertTrue(np.all(np_zeroed_x == expected_dataset))
        expected_users = np.array([[1, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0],
                                   [0, 1, 0, 0, 0],
                                   [0, 1, 0, 0, 0]],
                                  dtype=np.float32).reshape(2, 3, 5)
        self.assertTrue(np.all(np_users_x == expected_users))

    def test_variance_lookup(self):
        n_features = 5
        n_users = 2
        # cartesian product of users and items
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)

        self.bias = tf.verify_tensor_all_finite(
            tf.Variable(0.0,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        weights = tf.convert_to_tensor(np.arange(n_features),
                                       dtype=tf.float32)
        self.weights = tf.verify_tensor_all_finite(
            tf.Variable(weights,
                        trainable=True,
                        name='weights',
                        dtype=tf.float32),
            msg='NaN or Inf in weights')

        params = np.repeat(np.arange(n_features,
                                     dtype=np.float32)
                           .reshape(-1, 1), 10, axis=1)
        self.params = tf.verify_tensor_all_finite(
            tf.Variable(params,
                        trainable=True,
                        name='params', dtype=tf.float32),
            msg='NaN or Inf in parameters')

        self.graph.set_params(**{'bias': self.bias,
                                 'weights': self.weights,
                                 'params': self.params
                                 })
        self.graph.define_graph()
        self.graph.variance_estimate()
        _, users_x = self.graph.reshape_dataset(self.graph.n_users,
                                                self.graph.rankings,
                                                self.graph.x)
        user_idx = users_x.indices[:1, 2]
        lookup_var = PointLFP.variance_lookup(user_idx, self.graph.variance)

        sess = tf.Session()
        sess.run((self.graph.init_all_vars,
                  self.graph.init_variance_vars), feed_dict={
            self.graph.n_users: n_users,
            self.graph.n_features: n_features
        })

        ranking = np.array([[2, 1, 0], [2, 1, 0]], dtype=np.int32)

        variance = sess.run(self.graph.variance,
                            feed_dict={self.graph.x: sparse_x,
                                       self.graph.n_users: n_users,
                                       # self.graph.rankings: ranking}
                                       })
        expected_variance = np.repeat(np.array([29 / 3, 14 / 3],
                                               dtype=np.float32), 10) \
            .reshape(n_users, 10)
        self.assertTrue(np.all(expected_variance == variance))

        user_var = sess.run(lookup_var,
                            feed_dict={self.graph.x: sparse_x,
                                       self.graph.n_users: n_users,
                                       self.graph.rankings: ranking
                                       })
        expected_user_var = np.repeat(np.array([29 / 3],
                                               dtype=np.float32), 10) \
            .reshape(1, 10)
        self.assertTrue(np.all(user_var == expected_user_var))

    def test_ranking_coefficient(self):
        n_features = 5
        # cartesian product of users and items
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)

        self.graph.define_graph()
        pk = PointLFP.ranking_coefficient(self.graph.k)

        sess = tf.Session()
        actual_pk = sess.run(pk, feed_dict={self.graph.x: sparse_x,
                                            self.graph.k: 1})
        expected_pk = np.array([0.5], dtype=np.float32)
        self.assertTrue(np.all(expected_pk == actual_pk))
        actual_pk = sess.run(pk, feed_dict={self.graph.x: sparse_x,
                                            self.graph.k: 2})
        expected_pk = np.array([0.25], dtype=np.float32)
        self.assertTrue(np.all(expected_pk == actual_pk))

    def test_dot_product(self):
        n_features = 5
        n_users = 2
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)
        bias = tf.verify_tensor_all_finite(
            tf.Variable(0.0,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        weights = tf.convert_to_tensor(np.arange(n_features),
                                       dtype=tf.float32)
        weights = tf.verify_tensor_all_finite(
            tf.Variable(weights,
                        trainable=True,
                        name='weights',
                        dtype=tf.float32),
            msg='NaN or Inf in weights')

        params = np.repeat(np.arange(n_features,
                                     dtype=np.float32)
                           .reshape(-1, 1), 10, axis=1)
        params = tf.verify_tensor_all_finite(
            tf.Variable(params,
                        trainable=True,
                        name='params', dtype=tf.float32),
            msg='NaN or Inf in parameters')

        self.graph.set_params(**{'bias': bias,
                                 'weights': weights,
                                 'params': params})
        self.graph.define_graph()
        zeroed_x, _ = self.graph.reshape_dataset(self.graph.n_users,
                                                 self.graph.rankings,
                                                 self.graph.x)
        dot = self.graph.dot_product(self.graph.params,
                                     self.graph.rankings,
                                     zeroed_x)

        rankings = np.array([[2, 1, 0],
                             [2, 0, 1]], dtype=np.float32)
        sess = tf.Session()
        sess.run(self.graph.init_all_vars,
                 feed_dict={self.graph.n_features: n_features})
        actual_dot = sess.run(dot, feed_dict={self.graph.x: sparse_x,
                                              self.graph.rankings: rankings,
                                              self.graph.n_users: n_users})
        expected_dot = np.array([[4, 3, 2], [4, 2, 3]], dtype=np.float32)
        expected_dot = np.repeat(expected_dot, 10).reshape(2, 3, 10)
        self.assertTrue(np.all(expected_dot == actual_dot))

    def test_second_term(self):
        n_features = 5
        n_users = 2
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)
        bias = tf.verify_tensor_all_finite(
            tf.Variable(0.0,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        weights = tf.convert_to_tensor(np.arange(n_features),
                                       dtype=tf.float32)
        weights = tf.verify_tensor_all_finite(
            tf.Variable(weights,
                        trainable=True,
                        name='weights',
                        dtype=tf.float32),
            msg='NaN or Inf in weights')

        params = np.repeat(np.arange(n_features,
                                     dtype=np.float32)
                           .reshape(-1, 1), 10, axis=1)
        params = tf.verify_tensor_all_finite(
            tf.Variable(params,
                        trainable=True,
                        name='params', dtype=tf.float32),
            msg='NaN or Inf in parameters')

        self.graph.set_params(**{'bias': bias,
                                 'weights': weights,
                                 'params': params})
        self.graph.define_graph()
        self.graph.variance_estimate()
        zeroed_x, _ = self.graph.reshape_dataset(self.graph.n_users,
                                                 self.graph.rankings,
                                                 self.graph.x)
        dot = self.graph.dot_product(self.graph.params,
                                     self.graph.rankings,
                                     zeroed_x)
        self.graph.pk = self.graph.ranking_coefficient(self.graph.k)
        snd = self.graph.second_term(self.graph.k,
                                     self.graph.pk,
                                     self.graph.variance,
                                     dot)

        rankings = np.array([[2, 1, 0],
                             [2, 0, 1]], dtype=np.float32)
        sess = tf.Session()
        sess.run((self.graph.init_all_vars,
                  self.graph.init_variance_vars),
                 feed_dict={self.graph.n_users: n_users,
                            self.graph.n_features: n_features})
        actual_snd = sess.run(snd, feed_dict={self.graph.x: sparse_x,
                                              self.graph.k: 1,
                                              self.graph.rankings: rankings,
                                              self.graph.n_users: n_users})

        expected_snd = np.array([[29.0 * 15, 29.0 / 3 * 20],
                                 [14.0 / 3 * 20, 14.0 * 15]],
                                dtype=np.float32)
        self.assertTrue(np.allclose(actual_snd, expected_snd))
        actual_snd = sess.run(snd, feed_dict={self.graph.x: sparse_x,
                                              self.graph.k: 0,
                                              self.graph.rankings: rankings,
                                              self.graph.n_users: n_users})

        expected_snd = np.array([[29.0 / 3 * 10 * 16, 29.0 * 30, 29.0 / 3 * 40],
                                 [14.0 / 3 * 10 * 16, 14.0 / 3 * 40, 14.0 * 30, ]],
                                dtype=np.float32)
        self.assertTrue(np.allclose(actual_snd, expected_snd))

    def test_ranking_weight(self):
        k = tf.placeholder(shape=[], dtype=tf.int32)

        pm = self.graph.ranking_weights(k)
        sess = tf.Session()
        actual_pm = sess.run(pm, feed_dict={k: 12})
        expected_pm = np.array([1 / (2.0 ** i) for i in range(12)])
        self.assertTrue(np.all(actual_pm == expected_pm))

    def test_third_term(self):
        n_features = 5
        n_users = 2
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)
        bias = tf.verify_tensor_all_finite(
            tf.Variable(0.0,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        weights = tf.convert_to_tensor(np.arange(n_features),
                                       dtype=tf.float32)
        weights = tf.verify_tensor_all_finite(
            tf.Variable(weights,
                        trainable=True,
                        name='weights',
                        dtype=tf.float32),
            msg='NaN or Inf in weights')

        params = np.repeat(np.arange(n_features,
                                     dtype=np.float32)
                           .reshape(-1, 1), 10, axis=1)
        params = tf.verify_tensor_all_finite(
            tf.Variable(params,
                        trainable=True,
                        name='params', dtype=tf.float32),
            msg='NaN or Inf in parameters')

        self.graph.set_params(**{'bias': bias,
                                 'weights': weights,
                                 'params': params})
        self.graph.define_graph()
        self.graph.variance_estimate()
        zeroed_x, _ = self.graph.reshape_dataset(self.graph.n_users,
                                                 self.graph.rankings,
                                                 self.graph.x)
        dot = self.graph.dot_product(self.graph.params,
                                     self.graph.rankings,
                                     zeroed_x)
        trd = self.graph.third_term(self.graph.k,
                                    self.graph.variance,
                                    dot, dtype=self.graph.dtype)

        rankings = np.array([[2, 1, 0],
                             [2, 0, 1]], dtype=np.float32)
        sess = tf.Session()
        sess.run((self.graph.init_all_vars,
                  self.graph.init_variance_vars),
                 feed_dict={self.graph.n_users: n_users,
                            self.graph.n_features: n_features})
        actual_trd = sess.run(trd, feed_dict={self.graph.x: sparse_x,
                                              self.graph.k: 1,
                                              self.graph.rankings: rankings,
                                              self.graph.n_users: n_users})
        expected_trd = np.array([[1160., 8. * 29 / 3 * 10],
                                 [8 * 14 / 3 * 10, 560.]])

        self.assertTrue(np.allclose(actual_trd, expected_trd))

    def test_delta_f(self):
        n_features = 5
        n_users = 2
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)
        bias = tf.verify_tensor_all_finite(
            tf.Variable(0.0,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        weights = tf.convert_to_tensor(np.arange(n_features),
                                       dtype=tf.float32)
        weights = tf.verify_tensor_all_finite(
            tf.Variable(weights,
                        trainable=True,
                        name='weights',
                        dtype=tf.float32),
            msg='NaN or Inf in weights')

        params = np.repeat(np.arange(n_features,
                                     dtype=np.float32)
                           .reshape(-1, 1), 10, axis=1)
        params = tf.verify_tensor_all_finite(
            tf.Variable(params,
                        trainable=True,
                        name='params', dtype=tf.float32),
            msg='NaN or Inf in parameters')

        self.graph.set_params(**{'bias': bias,
                                 'weights': weights,
                                 'params': params})
        self.graph.define_graph()
        self.graph.ranking_computation()
        self.graph.variance_estimate()
        delta_f = self.graph.delta_f_computation()

        sess = tf.Session()
        sess.run((self.graph.init_all_vars,
                  self.graph.init_variance_vars), feed_dict={
            self.graph.n_users: n_users,
            self.graph.n_features: n_features
        })
        pred = np.array([[4, 3, 2], [45, 34, 23]], dtype=np.float32)
        rank = np.array([[2, 1, 0], [2, 1, 0]], dtype=np.int32)

        d = sess.run(delta_f, feed_dict={
            self.graph.x: sparse_x,
            self.graph.predictions: pred,
            self.graph.rankings: rank,
            self.graph.k: 1,
            self.graph.b: 1.0,
            self.graph.n_users: n_users
        })

        snd = np.array([[29.0 * 15, 29.0 / 3 * 20],
                        [14.0 * 15, 14.0 / 3 * 20]])
        trd = np.array([[1160., 8. * 29 / 3 * 10],
                        [560., 8 * 14 / 3 * 10]])

        expected = 0.5 * (pred[:, 1:] - (snd + 2 * trd))

        self.assertTrue(np.allclose(d, expected))

    def test_unique_sparse_tensor(self):
        n_users = 2
        n_items = 3
        x = np.array([[100, 200],
                      [100, 201],
                      [100, 202],
                      [100, 200],
                      [101, 200],
                      [101, 201],
                      [101, 202]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)
        self.graph.define_graph()
        self.graph.unique_rows_sparse_tensor()

        sess = tf.Session()
        actual = sess.run((self.graph.init_unique_vars,
                           tf.sparse_tensor_to_dense(self.graph.unique_x)),
                          feed_dict={self.graph.x: sparse_x,
                                     self.graph.n_features: n_users + n_items,
                                     self.graph.n_users: n_users,
                                     self.graph.n_items: n_items})[1]
        expected = np.array([[1, 0, 1, 0, 0],
                             [1, 0, 0, 1, 0],
                             [1, 0, 0, 0, 1],
                             [0, 1, 1, 0, 0],
                             [0, 1, 0, 1, 0],
                             [0, 1, 0, 0, 1]])
        self.assertTrue(np.all(actual == expected))

    def test_unique_sparse_tensor_multi(self):
        n_users = 2
        n_items = 3
        n_features = 7
        x = np.array([[100, 200, 300],
                      [100, 201, 300],
                      [100, 202, 301],
                      [100, 200, 300],
                      [101, 200, 300],
                      [101, 201, 300],
                      [101, 202, 301]])
        enc = OneHotEncoder(dtype=np.float32)
        new_x = enc.fit(x).transform(x).tocsr()
        new_x.sort_indices()
        sparse_x = sparse_repr(new_x, np.float32)
        self.graph.define_graph()
        self.graph.unique_rows_sparse_tensor()

        sess = tf.Session()
        actual = sess.run((self.graph.init_unique_vars,
                           tf.sparse_tensor_to_dense(self.graph.unique_x)),
                          feed_dict={self.graph.x: sparse_x,
                                     self.graph.n_features: n_features,
                                     self.graph.n_users: n_users,
                                     self.graph.n_items: n_items})[1]
        expected = np.array([[1, 0, 1, 0, 0, 1, 0],
                             [1, 0, 0, 1, 0, 1, 0],
                             [1, 0, 0, 0, 1, 0, 1],
                             [0, 1, 1, 0, 0, 1, 0],
                             [0, 1, 0, 1, 0, 1, 0],
                             [0, 1, 0, 0, 1, 0, 1]])
        self.assertTrue(np.all(actual == expected))

    def tearDown(self):
        tf.reset_default_graph()


if __name__ == '__main__':
    unittest.main()
