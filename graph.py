from abc import ABC, abstractmethod
import tensorflow as tf


class ComputationalGraph(ABC):

    def __init__(self,
                 dtype=tf.float32,
                 l2_w=0.001,
                 l2_v=0.001,
                 n_factors=10,
                 init_std=0.01):
        self.n_factors = n_factors
        self.dtype = dtype
        self.l2_v = l2_v
        self.l2_w = l2_w
        self.init_std = init_std
        self.optimizer = None
        self.init_all_vars = None
        self.n_features = None
        self.summary_op = None
        self.lambda_v = None
        self.lambda_w = None
        self.half = None
        self.bias = None
        self.params = None
        self.weights = None
        self.global_step = None
        self.l2_norm = None
        self.y_hat = None
        self.loss = None
        self.reduced_loss = None
        self.target = None
        self.checked_target = None
        self.trainer = None
        self.init_all_vars = None
        self.batch_loss = None
        self.size = None
        self.saver = None
        self.ops = None
        self.x = None
        self.y = None

    def define_graph(self):
        self.global_step = tf.train.create_global_step()
        with tf.name_scope('placeholders'):
            self.init_placeholder()
        with tf.name_scope('parameters'):
            self.init_params()
        with tf.name_scope('main_graph'):
            self.init_main_graph()
        with tf.name_scope('loss'):
            self.init_loss()
            self.init_regularization()
        with tf.name_scope('target'):
            self.init_target()
        with tf.name_scope('training'):
            self.init_trainer()
        self.init_all_vars = tf.global_variables_initializer()
        self.summary_op = tf.summary.merge_all()
        self.set_ops()

    def save(self):
        self.saver = tf.train.Saver()

    def init_params(self):
        self.lambda_w = tf.constant(self.l2_w,
                                    dtype=self.dtype,
                                    name='lambda_w')
        self.lambda_v = tf.constant(self.l2_v,
                                    dtype=self.dtype,
                                    name='lambda_w')
        self.half = tf.constant(0.5,
                                dtype=self.dtype,
                                name='half')
        self.bias = tf.verify_tensor_all_finite(
            tf.Variable(self.init_std,
                        trainable=True,
                        name='bias'),
            msg='NaN or Inf in bias')
        rnd_weights = tf.random_uniform([self.n_features],
                                        minval=-self.init_std,
                                        maxval=self.init_std,
                                        dtype=self.dtype)
        self.weights = tf.verify_tensor_all_finite(
            tf.Variable(rnd_weights,
                        trainable=True,
                        name='weights'),
            msg='NaN or Inf in weights')
        rnd_params = tf.random_uniform([self.n_features, self.n_factors],
                                       minval=-self.init_std,
                                       maxval=self.init_std,
                                       dtype=self.dtype)
        self.params = tf.verify_tensor_all_finite(
            tf.Variable(rnd_params,
                        trainable=True,
                        name='params'),
            msg='NaN or Inf in parameters')
        tf.summary.scalar('bias', self.bias)

    def init_trainer(self):
        self.trainer = self.optimizer.minimize(self.checked_target,
                                               global_step=self.global_step)

    def init_regularization(self):
        self.l2_norm = tf.reduce_sum(
            tf.add(
                tf.multiply(self.lambda_w, tf.pow(self.weights, 2)),
                tf.multiply(self.lambda_v,
                            tf.transpose(tf.pow(self.params, 2)))))

    def init_target(self):
        self.target = self.l2_norm + self.reduced_loss
        self.checked_target = tf.verify_tensor_all_finite(
            self.target,
            msg='NaN or Inf in target value',
            name='target')
        tf.summary.scalar('target', self.checked_target)

    @abstractmethod
    def set_ops(self):
        pass

    @abstractmethod
    def set_params(self, *args):
        pass

    @abstractmethod
    def init_placeholder(self):
        pass

    @abstractmethod
    def init_main_graph(self):
        pass

    @abstractmethod
    def init_loss(self):
        pass


class PointwiseGraph(ComputationalGraph):

    def __init__(self,
                 n_factors=10,
                 dtype=tf.float32,
                 init_std=0.01,
                 loss_function=tf.losses.mean_squared_error,
                 l2_v=0.001,
                 l2_w=0.001,
                 learning_rate=0.001,
                 optimizer=tf.train.AdamOptimizer):
        super(PointwiseGraph, self).__init__(
            dtype=dtype,
            init_std=init_std,
            n_factors=n_factors,
            l2_w=l2_w,
            l2_v=l2_v)
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.optimizer = optimizer(learning_rate=self.learning_rate)

    def set_ops(self):
        self.ops = [self.trainer,
                    self.summary_op,
                    self.global_step,
                    self.batch_loss]

    def init_placeholder(self):
        pass

    def set_params(self, x, y):
        self.x = x
        self.y = y

    def init_main_graph(self):
        x = self.x
        assert x is not None, "x must be set before graph is defined"
        el_wise_mul = x * self.weights
        weighted_sum = tf.sparse_reduce_sum(el_wise_mul, 1, keep_dims=True)
        linear_terms = tf.add(self.bias, weighted_sum, name='linear_terms')
        dot = tf.sparse_tensor_dense_matmul(x, self.params)
        pow_sum = tf.pow(dot, 2)
        pow_x = x * tf.sparse_tensor_to_dense(x)
        pow_v = tf.pow(self.params, 2)
        sum_pow = tf.sparse_tensor_dense_matmul(pow_x, pow_v)
        sub = tf.subtract(pow_sum, sum_pow)
        sum_sub = tf.reduce_sum(sub, 1, keepdims=True)
        pair_interactions = tf.multiply(self.half, sum_sub)
        self.y_hat = tf.add(linear_terms, pair_interactions)

    def init_loss(self):
        y_true = self.y
        assert y_true is not None, "y must be set before graph is defined"
        self.loss = self.loss_function(self.y_hat, y_true)
        self.reduced_loss = tf.reduce_mean(self.loss)
        self.size = tf.cast(tf.size(self.y_hat), dtype=self.dtype)
        self.batch_loss = self.reduced_loss * self.size
        tf.summary.scalar('loss', self.reduced_loss)


class BayesianPersonalizedRankingGraph(ComputationalGraph):

    def __init__(self,
                 n_factors=10,
                 dtype=tf.float32,
                 init_std=0.01,
                 l2_v=0.001,
                 l2_w=0.001,
                 learning_rate=0.001,
                 optimizer=tf.train.AdamOptimizer):
        super(BayesianPersonalizedRankingGraph, self).__init__(
            dtype=dtype,
            init_std=init_std,
            n_factors=n_factors,
            l2_w=l2_w,
            l2_v=l2_v
        )
        self.learning_rate = learning_rate
        self.optimizer = optimizer(learning_rate=self.learning_rate)
        self.y_hat = None
        self.neg_hat = None

    def set_params(self, x, y):
        pass

    def set_ops(self):
        self.ops = [self.trainer,
                    self.summary_op,
                    self.global_step,
                    self.batch_loss]

    def init_placeholder(self):
        self.x = tf.sparse_placeholder(self.dtype,
                                       shape=[None, self.n_features],
                                       name='pos')
        self.y = tf.sparse_placeholder(self.dtype,
                                       shape=[None, self.n_features],
                                       name='neg')

    def init_main_graph(self):
        self.y_hat = self.equation(self.x)
        self.neg_hat = self.equation(self.y)

    def equation(self, x):
        el_wise_mul = x * self.weights
        weighted_sum = tf.sparse_reduce_sum(el_wise_mul, 1, keep_dims=True)
        linear_terms = tf.add(self.bias, weighted_sum, name='linear_terms')
        dot = tf.sparse_tensor_dense_matmul(x, self.params)
        pow_sum = tf.pow(dot, 2)
        pow_x = x * tf.sparse_tensor_to_dense(x)
        pow_v = tf.pow(self.params, 2)
        sum_pow = tf.sparse_tensor_dense_matmul(pow_x, pow_v)
        sub = tf.subtract(pow_sum, sum_pow)
        sum_sub = tf.reduce_sum(sub, 1, keepdims=True)
        pair_interactions = tf.multiply(self.half, sum_sub)
        y_hat = tf.add(linear_terms, pair_interactions)
        return y_hat

    def init_loss(self):
        self.loss = tf.log(tf.sigmoid(tf.subtract(self.y_hat, self.neg_hat)))
        self.reduced_loss = -tf.reduce_mean(self.loss)
        self.size = tf.cast(tf.size(self.y_hat), dtype=self.dtype)
        self.batch_loss = self.reduced_loss * self.size
        tf.summary.scalar('loss', self.reduced_loss)


class LatentFactorPortfolioGraph(ComputationalGraph):

    def __init__(self, **kwargs):
        super(LatentFactorPortfolioGraph, self).__init__(**kwargs)
        self.variance = None
        self.indices = None
        self.values = None
        self.shape = None
        self.init_variance_vars = None
        self.init_unique_vars = None
        self.hash_coo = None
        self.csr = None

    def set_ops(self):
        pass

    def set_params(self, *args):
        pass

    def init_placeholder(self):
        self.indices = tf.placeholder(shape=[None, 2],
                                      name='indices',
                                      dtype=tf.int64)
        self.shape = tf.placeholder(shape=[2],
                                    name='shape',
                                    dtype=tf.int64)
        self.values = tf.placeholder(shape=[None],
                                     name='values',
                                     dtype=self.dtype)

    def init_main_graph(self):
        pass

    def init_loss(self):
        pass

    def variance_estimate(self, n_samples, n_users):
        # Variables and tensors initialization

        # BEWARE - x must be only the unique entries
        x = tf.SparseTensor(self.indices, self.values, self.shape)
        init_sum_of_square = tf.Variable(tf.zeros(shape=[n_users, self.n_factors],
                                                  dtype=self.dtype),
                                         name='sum_of_square')
        init_nu = tf.Variable(tf.zeros([n_users], dtype=tf.int64),
                              name='n_items_per_user')
        ones = tf.ones(dtype=tf.int64, shape=[n_samples])
        u_idx = x.indices[:, 1]
        lim_users = tf.constant(n_users, dtype=tf.int64, shape=[1])
        where = tf.less(u_idx, lim_users)
        indexes = tf.reshape(tf.where(where), shape=[-1])
        indexes = tf.nn.embedding_lookup(x.indices, indexes)[:, 1]

        # computes the square for the batch (batch_size, n_factors)
        # each row represent the square root for a user
        user_v = tf.nn.embedding_lookup(self.params, indexes)
        dot = tf.sparse_tensor_dense_matmul(x, self.params)
        dot = user_v - (dot - user_v)
        sq = tf.square(dot)

        # Nice it should be working
        sum_of_square = tf.scatter_add(init_sum_of_square, indexes, sq)
        nu = tf.scatter_add(init_nu, indexes, ones)
        nu = tf.tile(tf.expand_dims(tf.to_float(nu), 1), [1, self.n_factors])

        self.variance = sum_of_square / nu
        self.init_variance_vars = tf.variables_initializer([init_nu,
                                                            init_sum_of_square])

    def unique_rows_sp_matrix(self, n_rows):
        with tf.name_scope('unique_rows_sparse_matrix'):
            # Placeholders init
            indptr = tf.placeholder(dtype=tf.int64, shape=[None])
            indices = tf.placeholder(dtype=tf.int64, shape=[None])
            data = tf.placeholder(dtype=self.dtype, shape=[None])

            self.csr = (indptr, indices, data)

            # Variables init
            init_i = tf.Variable(tf.constant(0))
            init_col = tf.Variable(tf.constant('', shape=[n_rows, 1]),
                                   name='init_col')
            init_dat = tf.Variable(tf.constant('', shape=[n_rows, 1]),
                                   name='tf_string_data')

            def cond(_):
                return True

            def body(i):
                min_i = tf.gather(indptr, i)
                max_i = tf.gather(indptr, i + 1)
                tf_slice = tf.range(min_i, max_i)
                string_col = tf.as_string(tf.gather(indices, tf_slice))
                string_dat = tf.as_string(tf.gather(data, tf_slice))
                cols = tf.reduce_join(string_col, separator=':',
                                      keep_dims=True)
                dat = tf.reduce_join(string_dat, separator=':',
                                     keep_dims=True)
                update_col = tf.scatter_update(init_col, i, cols)
                update_dat = tf.scatter_update(init_dat, i, dat)
                with tf.control_dependencies([update_col, update_dat]):
                    return i + 1

            loop = tf.while_loop(cond, body, [init_i],
                                 maximum_iterations=n_rows,
                                 back_prop=False)
            with tf.control_dependencies([loop]):
                reshape_col = tf.reshape(init_col, shape=[-1])
                reshape_dat = tf.reshape(init_dat, shape=[-1])
            tf_stack_coo = tf.stack([reshape_col, reshape_dat])
            tf_hash_coo = tf.reduce_join(tf_stack_coo, axis=0, separator=',')

            self.init_unique_vars = tf.variables_initializer([init_i,
                                                              init_col,
                                                              init_dat])
            self.hash_coo = tf_hash_coo


class BPRLFPGraph(BayesianPersonalizedRankingGraph, LatentFactorPortfolioGraph):

    def init_placeholder(self):
        BayesianPersonalizedRankingGraph.init_placeholder(self)
        LatentFactorPortfolioGraph.init_placeholder(self)


class PointwiseLFPGraph(PointwiseGraph, LatentFactorPortfolioGraph):

    def init_placeholder(self):
        PointwiseGraph.init_placeholder(self)
        LatentFactorPortfolioGraph.init_placeholder(self)
